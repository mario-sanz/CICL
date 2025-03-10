import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from datetime import datetime
import gc


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    model_name: str
    dataset_name: str
    shots_range: List[int]
    corrected_proportions_range: List[float]
    seed: int = 42

class CICLExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.class_labels = None
        self.results = {
            "experiment_info": {
                "model_name": config.model_name,
                "dataset_name": config.dataset_name,
                "seed": config.seed,
                "num_classes": None,
                "shots_range": config.shots_range,
                "corrected_proportions_range": config.corrected_proportions_range,
            },
            "results": []
        }

        # Set random seed
        set_seed(config.seed)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        """Initialize models, tokenizers, and dataset."""
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load dataset
        self.dataset = load_from_disk(f"data/{self.config.dataset_name}")
        # Set the class labels as the unique labels in the dataset
        self.class_labels = self.dataset["train"].unique("label")
        print(f"Class labels: {self.class_labels}")
        self.results["experiment_info"]["num_classes"] = len(self.class_labels)

    def select_examples(self, num_shots: int, incorrect_proportion: float) -> Tuple[List[Dict], List[int]]:
        """
        Randomly select examples for ICL, ensuring class balance and a proportion of incorrect predictions.
        
        Args:
            num_shots (int): Total number of examples to select.
            incorrect_proportion (float): Proportion of examples that should be incorrectly classified.

        Returns:
            Tuple[List[Dict], List[int]]: A list of selected examples and their corresponding zero-/few-shot predictions.
        """
        num_classes = len(self.class_labels)
        num_incorrect = int(num_shots * incorrect_proportion)

        # Generate random indices for incorrect predictions
        incorrect_indices = set(np.random.choice(num_shots, num_incorrect, replace=False))

        # Pre-filter the dataset by class
        class_examples = {label: self.dataset["train"].filter(lambda x: x["label"] == label) for label in self.class_labels}

        # Ensure we have enough examples for each class
        class_order = (self.class_labels * (num_shots // num_classes + 1))[:num_shots]
        np.random.shuffle(class_order)  # Randomize class order

        # Select one example per class for the few-shot context
        few_shot_context_examples = []
        for label in self.class_labels:
            few_shot_context_examples.append(np.random.choice(class_examples[label], 1, replace=False)[0])

        selected_examples = []
        predictions = []

        for shot_idx, target_class in enumerate(class_order):
            examples = class_examples[target_class]
            is_incorrect_needed = shot_idx in incorrect_indices

            for example in np.random.permutation(examples):
                temp_prompt = self.generate_icl_prompt(few_shot_context_examples, example["text"]) # Few-shot
                #temp_prompt = self.generate_icl_prompt([], example["text"]) # Zero-shot
                probs = self.compute_label_likelihood(temp_prompt)
                pred = max(probs, key=probs.get)

                is_incorrect = pred != example["label"]
                if (is_incorrect_needed and is_incorrect) or (not is_incorrect_needed and not is_incorrect):
                    selected_examples.append(example)
                    predictions.append(pred)
                    break

        # Shuffle the examples and predictions together
        combined = list(zip(selected_examples, predictions))
        np.random.shuffle(combined)
        selected_examples, predictions = zip(*combined)

        return list(selected_examples), list(predictions)
    
    def generate_icl_prompt(self, examples: List[Dict], test_input: str) -> str:
        """
        Create ICL prompt with examples and test input.
        
        Args:
            examples (List[Dict]): A list of examples with text and label fields.
            test_input (str): The test input text.

        Returns:
            str: The ICL prompt.
        """
        prompt = ""
        for ex in examples:
            prompt += f"Text: {ex['text']}\nLabel: {ex['label']}\n\n"
        prompt += f"Text: {test_input}\nLabel:"
        return prompt
    
    def generate_cicl_prompt(self, examples: List[Dict], predictions: List[str], test_input: str, test_pred: str) -> str:
        """
        Create CICL prompt with examples, predictions, and test input.

        Args:
            examples (List[Dict]): A list of examples with text and label fields.
            predictions (List[str]): A list of predictions for the examples.
            test_input (str): The test input text.
            test_pred (str): The prediction for the test input.

        Returns:
            str: The CICL prompt.
        """
        prompt = ""
        for ex, pred in zip(examples, predictions):
            prompt += f"Text: {ex['text']}\nPredicted label: {pred}\nCorrect label: {ex['label']}\n\n"
        prompt += f"Text: {test_input}\nPredicted label: {test_pred}\nCorrect label:"
        return prompt
    
    def compute_label_likelihood(self, prompt: str):
        """
        Compute the likelihood of each possible label in the dataset given a prompt.

        Args:
            prompt (str): The prompt with the few-shot examples and the test input.

        Returns:
            Dict: A dictionary with the likelihood of each label.
        """
        # Tokenize the prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_length = prompt_ids.size(1)
        
        normalized_likelihoods = {}
        for label in self.class_labels:
            label = " " + label # Last prompt sentence doesn't contain " " after the "Label:"
            
            # Tokenize the label
            label_ids = self.tokenizer(label, return_tensors="pt", add_special_tokens=False)["input_ids"]
            
            # Concatenate prompt and label
            input_ids = torch.cat([prompt_ids, label_ids], dim=-1).to("cuda")

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(input_ids)

            # Extract logits for the label tokens
            logits = outputs.logits[0, prompt_length - 1:-1]  # Skip prompt logits
            label_tokens = input_ids[0, prompt_length:]  # Tokens of the label

            # Compute token probabilities
            probs = torch.softmax(logits, dim=-1)
            token_probs = probs[range(len(label_tokens)), label_tokens]

            # Compute raw sequence likelihood
            raw_likelihood = torch.prod(token_probs).item()

            # Normalize for label length
            normalized_likelihood = raw_likelihood ** (1 / len(label_tokens))
            normalized_likelihoods[label.strip()] = normalized_likelihood

        # Normalize final probabilites (so that they sum to 1)
        total_raw = sum(normalized_likelihoods.values())
        normalized_likelihoods = {label: prob / total_raw for label, prob in normalized_likelihoods.items()}

        return normalized_likelihoods
    
    def evaluate_shots(self, num_shots: int, corrected_proportion: float) -> Dict:
        """
        Evaluate the model with a given number of shots and proportion of incorrect predictions.

        Args:
            num_shots (int): Number of shots to use.
            corrected_proportion (float): Proportion of incorrect predictions.

        Returns:
            Dict: A dictionary with the evaluation metrics.
        """
        test_samples = self.dataset["dev"]
        results = {
            "icl_predictions": [],
            "cicl_predictions": [],
            "true_labels": [],
        }

        # Select examples once for all test samples
        examples, predictions = self.select_examples(num_shots, corrected_proportion)

        for test_sample in tqdm(test_samples):
            # Generate ICL prompt
            icl_prompt = self.generate_icl_prompt(examples, test_sample["text"])
            # Make the ICL pred
            icl_probs = self.compute_label_likelihood(icl_prompt)
            icl_pred = max(icl_probs, key=icl_probs.get)

            # Generate CICL prompt
            cicl_prompt = self.generate_cicl_prompt(examples, predictions, test_sample["text"], icl_pred)
            # Make the CICL pred
            cicl_probs = self.compute_label_likelihood(cicl_prompt)
            cicl_pred = max(cicl_probs, key=cicl_probs.get)

            true_label = test_sample["label"]

            results["icl_predictions"].append(icl_pred)
            results["cicl_predictions"].append(cicl_pred)
            results["true_labels"].append(true_label)

            if len(results["icl_predictions"]) < 5:
                print(cicl_prompt)
                print(f"ICL pred: {icl_pred}, CICL pred: {cicl_pred}, True: {true_label}")
                print("-" * 50)

        # Compute metrics
        metrics = {
            "num_shots": num_shots,
            "corrected_proportion": corrected_proportion,
            "icl_accuracy": accuracy_score(results["true_labels"], results["icl_predictions"]),
            "cicl_accuracy": accuracy_score(results["true_labels"], results["cicl_predictions"]),
            "icl_macro_f1": f1_score(results["true_labels"], results["icl_predictions"], average="macro"),
            "cicl_macro_f1": f1_score(results["true_labels"], results["cicl_predictions"], average="macro"),
        }

        return metrics
    
    def run_experiment(self):
        """Run full experiment."""
        self.setup()
        #for num_shots in self.config.shots_range:
        for corrected_proportion in self.config.corrected_proportions_range:
            try:
                print(f"Running experiment with: {corrected_proportion}")
                result = self.evaluate_shots(self.config.shots_range[0], corrected_proportion)
                self.results["results"].append(result)
            except Exception as e:
                print(f"Skipping {corrected_proportion} shots due to error: {e}")
                break

        self.save_results()
    
    def save_results(self):
        """Save experiment results to a JSON file."""
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config.model_name.split("/")[-2]
        output_file = f"results/{date}_{model_name}_{self.config.dataset_name}_{self.config.seed}.json"
        self.results["experiment_info"]["timestamp"] = date

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {output_file}")

# Parameters for the experiment
# Models to evaluate
model_names = ["meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.3", "Qwen/Qwen2.5-7B", "EleutherAI/gpt-j-6b"]
# Datasets to evaluate
dataset_names = ["ag_news", "cr", "dbpedia", "ethos_binary", "financial_phrasebank", "hate_speech18", "mr", "poem_sentiment", "sst2", "sst5", "subj", "trec_6", "tweet_eval_atheism", "tweet_eval_feminist", "tweet_eval_hate", "tweet_eval_irony", "tweet_eval_offensive"]
# Random seeds
seeds = [0, 1, 22, 42, 99]
# Number of few-shot examples
shots_range = [8]
# Proportions of incorrect predictions (% of the few-shot examples)
corrected_proportions_range = [0.0, 0.25, 0.5, 0.75, 1.0]

# Run the experiment
for model_name in model_names:
    for dataset_name in dataset_names:
        for seed in seeds:
            config = ExperimentConfig(
                model_name=model_name,
                dataset_name=dataset_name,
                shots_range=shots_range,
                corrected_proportions_range=corrected_proportions_range,
                seed=seed
            )
            experiment = CICLExperiment(config)
            experiment.run_experiment()

            # Clear memory
            del experiment
            gc.collect()
            torch.cuda.empty_cache()
