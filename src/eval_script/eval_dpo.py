'''
This program evaluates the format bias of a DPO-trained language model by comparing its preference for original vs. augmented responses across different patterns.
'''
import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM
from accelerate import Accelerator
import sys
import torch.nn.functional as F

# Enable pandas-like progress bar for tqdm
tqdm.pandas()

# --- 1. Script Configuration ---
@dataclass
class ScriptArguments:
    """
    Configuration for the DPO evaluation script.
    """
    # Dataset path of the augmented response pairs
    dataset_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK, e.g. src/data/augment_pairs.json
        metadata={"help": "The path to the dataset containing prompt-response pairs."}
    )
    dpo_model_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK, e.g. https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
        metadata={"help": "The path to the DPO-trained model checkpoint."}
    )
    pi0_model_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK, e.g. https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta
        metadata={"help": "The path to the reference model (e.g., SFT model) checkpoint."}
    )
    tokenizer_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "The path to the tokenizer compatible with the models."}
    )
    gpu_id: int = field(
        default=0,
        metadata={"help": "ID of the GPU to use for this process."}
    )
    log_file_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "Path to the log file for script output."}
    )

# --- 2. Main Execution Block ---
def main():
    """
    Main function to evaluate the format bias of a DPO-trained model.
    """
    # --- 2.1. Setup and Argument Parsing ---
    
    # Define patterns to be evaluated
    pattern_list = ["bold", "list", "emoji", "exclamation", "link", "affirmative"]
    
    # Parse command-line arguments first to get the log file path
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Redirect standard output to a log file
    log_file_path = script_args.log_file_path
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "a", buffering=1) as log_file:
        sys.stdout = log_file
        
        # Initialize Accelerator
        accelerator = Accelerator()
        
        device = accelerator.device
        print(f"Using device: {device}")
        print(f"Evaluating models: {script_args.dpo_model_path} vs {script_args.pi0_model_path}")
        
        # --- 2.2. Model and Data Loading ---
        
        # Load the DPO and reference models
        dpo_model = AutoModelForCausalLM.from_pretrained(
            script_args.dpo_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(device)
        
        pi0_model = AutoModelForCausalLM.from_pretrained(
            script_args.pi0_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(device)
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_path, use_fast=True)
        
        # Load the evaluation dataset
        try:
            with open(script_args.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading dataset from {script_args.dataset_path}: {e}")
            return
            
        # --- 2.3. Helper Function for Probability Calculation ---
        
        def get_log_prob(prompt: str, response: str, model: AutoModelForCausalLM) -> float:
            """
            Calculates the log probability of a response given a prompt under a model.
            
            Args:
                prompt (str): The input prompt.
                response (str): The model's response.
                model (AutoModelForCausalLM): The language model to use.
            
            Returns:
                float: The sum of the log probabilities of the response tokens.
            """
            # Tokenize prompt and response
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            
            # Combine tokens and create input/target tensors
            full_tokens = prompt_tokens + response_tokens
            input_ids = torch.tensor([full_tokens[:-1]]).to(device)
            target_ids = torch.tensor([full_tokens[1:]]).to(device)
            
            # Get model logits without gradient calculation
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            # Calculate log probabilities and sum over response tokens
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze()
            
            # Account for the first token of the response
            # The log prob of the first response token is at index len(prompt_tokens)-1
            response_log_probs = target_log_probs[len(prompt_tokens) - 1:]
            
            return response_log_probs.sum().item()

        # --- 2.4. Evaluation Loop ---
        
        # Initialize counters for win rates
        winners = {pattern: 0 for pattern in pattern_list}
        pattern_counts = {pattern: 0 for pattern in pattern_list}
        
        for sample in tqdm(data, desc="Evaluating DPO preference"):
            prompt = sample["prompt"]
            pattern = sample["pattern"]
            
            if pattern not in pattern_counts:
                continue

            # Calculate the DPO ratio (log(pi_dpo/pi_pi0)) for original and augmented responses
            log_prob_dpo_origin = get_log_prob(prompt, sample["origin"], dpo_model)
            log_prob_pi0_origin = get_log_prob(prompt, sample["origin"], pi0_model)
            
            log_prob_dpo_augment = get_log_prob(prompt, sample["augment"], dpo_model)
            log_prob_pi0_augment = get_log_prob(prompt, sample["augment"], pi0_model)
            
            dpo_ratio_origin = log_prob_dpo_origin - log_prob_pi0_origin
            dpo_ratio_augment = log_prob_dpo_augment - log_prob_pi0_augment
            
            # Check if the DPO model prefers the original response
            if dpo_ratio_origin > dpo_ratio_augment:
                winners[pattern] += 1
            
            pattern_counts[pattern] += 1
        
        # --- 2.5. Print Results and Cleanup ---
        print("\n--- DPO Format Bias Evaluation Results ---")
        print(f"DPO Model: {script_args.dpo_model_path}")
        print(f"Reference Model: {script_args.pi0_model_path}")
        
        for pattern in pattern_list:
            count = pattern_counts.get(pattern, 0)
            if count > 0:
                win_rate = winners[pattern] / count
                print(f"  - '{pattern}' pattern win rate (Original vs Augmented): "
                      f"{winners[pattern]}/{count} = {win_rate:.4f}")
            else:
                print(f"  - No samples found for pattern '{pattern}'.")

    # Restore standard output
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()