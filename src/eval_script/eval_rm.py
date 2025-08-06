'''
This program evaluates the augmented response pairs with the reward model (return numerical reward) and analyzes the win, loss, and tie rates for different stylistic patterns.
'''
import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM
import sys

tqdm.pandas()

pattern_list = ["bold", "list", "emoji", "exclamation", "link", "affirmative"]

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK, e.g. "src/data/augment_pairs.json"
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "the location of the output file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK, e.g. https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1, https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1, https://huggingface.co/NCSOFT/Llama-3-OffsetBias-RM-8B
        metadata={"help": "the name of the reward model"},
    )
    tokenizer_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "the path to the tokenizer"},
    )
    gpu_id: int = 1


# --- Main Execution Block ---
def main():
    """
    Main function to run the evaluation of format bias.
    """
    
    # Redirect standard output to a log file
    log_file_path = "BLANK" # Replaced with BLANK
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "a") as f:
        sys.stdout = f
    
        # Initialize accelerator
        accelerator = Accelerator()
    
        # Parse arguments
        parser = HfArgumentParser(ScriptArguments)
        script_args = parser.parse_args_into_dataclasses()[0]
    
        os.makedirs(os.path.dirname(script_args.output_dir), exist_ok=True)
    
        # Set device
        device = accelerator.device
    
        # Define reward model pipeline
        pipe_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 1,
        }
    
        reward_model_path = script_args.reward_name_or_path
        rm_tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_path)
        rm_pipe = pipeline(
            "sentiment-analysis",
            model=reward_model_path,
            device=device,
            tokenizer=rm_tokenizer,
            model_kwargs={"torch_dtype": torch.float32},
            truncation=True,
        )
    
        # Load dataset
        ds = load_dataset("json", data_files=script_args.dataset_name_or_path, split='train')
    
        def get_reward(test_texts):
            """Gets reward scores for a list of texts."""
            pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
            rewards = [output[0]["score"] for output in pipe_outputs]
            return rewards
    
        # Process the data and compute rewards
        winners = {pattern: 0 for pattern in pattern_list}
        pattern_cnts = {pattern: 0 for pattern in pattern_list}
    
        with torch.no_grad():
            for sample in tqdm(ds):
                pattern = sample["pattern"]

                # modify the prompt template as needed    
                chat_origin = [
                    {"role": "user", "content": sample["prompt"]},
                    {"role": "assistant", "content": sample["origin"]},
                ]
    
                chat_augment = [
                    {"role": "user", "content": sample["prompt"]},
                    {"role": "assistant", "content": sample["augment"]},
                ]
    
                test_texts = [
                    rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")
                    for chat in [chat_origin, chat_augment]
                ]
    
                rewards = get_reward(test_texts)
    
                pattern_cnts[pattern] += 1
                if rewards[0] > rewards[1]:
                    winners[pattern] += 1
        
        # Print results
        print("Model:", reward_model_path)
        for pattern in pattern_list:
            if pattern_cnts[pattern] != 0:
                print(f"{pattern} Win rate: {winners[pattern]}/{pattern_cnts[pattern]}={winners[pattern]/pattern_cnts[pattern]:.4f}")
    
    # Restore stdout
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()