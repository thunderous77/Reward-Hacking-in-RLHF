'''
This program evaluates and ranks multiple responses to a given prompt using a reward model and saves the best-performing responses along with their scores.
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

# --- 1. Script Configuration ---
# Use dataclasses for clean and type-hinted argument parsing.
@dataclass
class ScriptArguments:
    """
    Configuration for the evaluation script.
    """
    # Best of N Dataset Path
    dataset_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK, e.g. 1231czx/gemma2_9b_it_alpaca_n256
        metadata={"help": "The path or name of the dataset to be evaluated."}
    )
    output_dir: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "Directory to save the evaluation results."}
    )
    
    # Reward model and tokenizer paths
    reward_model_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "The path to the reward model checkpoint."}
    )
    tokenizer_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "The path to the tokenizer for the reward model."}
    )
    
    # Processing and hardware configuration
    gpu_id: int = field(
        default=0,
        metadata={"help": "ID of the GPU to use for this process."}
    )
    num_gpus: int = field(
        default=1,
        metadata={"help": "Total number of GPUs used for distributed processing."}
    )
    
    # Generation and evaluation parameters
    num_responses: int = field(
        default=8,
        metadata={"help": "The number of responses per prompt to evaluate."}
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "Delimiter string between prompt and response."}
    )

# --- 2. Main Execution Block ---
def main():
    """
    Main function to parse arguments, initialize models, process data,
    and save evaluation results.
    """
    # Parse command-line arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Initialize Accelerator for distributed training/evaluation
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    # --- 3. Model and Tokenizer Initialization ---
    try:
        # Load the tokenizer from the specified path
        tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_path)
        
        # Initialize the reward model pipeline
        reward_pipeline = pipeline(
            "sentiment-analysis",  # Using sentiment-analysis pipeline for reward modeling
            model=script_args.reward_model_path,
            device=device,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.float32},
            truncation=True,
        )
    except Exception as e:
        print(f"Error loading models or tokenizer: {e}")
        return

    # --- 4. Dataset Loading and Sharding ---
    try:
        full_dataset = load_dataset(script_args.dataset_path, split="train")
        
        # Split the dataset for distributed processing across GPUs
        num_samples = len(full_dataset)
        subset_length = num_samples // script_args.num_gpus
        start_index = subset_length * script_args.gpu_id
        
        # Adjust end index for the last GPU to handle remainder
        end_index = (start_index + subset_length 
                     if script_args.gpu_id < script_args.num_gpus - 1 
                     else num_samples)
        
        dataset = full_dataset.select(range(start_index, end_index))
        
        print(f"GPU {script_args.gpu_id} processing a subset of size {len(dataset)}")
    except Exception as e:
        print(f"Error loading or sharding dataset: {e}")
        return
    
    # --- 5. Evaluation Loop and Data Processing ---
    all_data = []
    best_responses_data = []
    
    # Construct output file paths
    all_data_path = os.path.join(
        script_args.output_dir, 
        f"all_data_gpu_{script_args.gpu_id}.json"
    )
    best_data_path = os.path.join(
        script_args.output_dir, 
        f"best_responses_gpu_{script_args.gpu_id}.json"
    )
    os.makedirs(script_args.output_dir, exist_ok=True)
    
    # Pipeline configuration
    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
    }

    with torch.no_grad():
        for sample in tqdm(dataset, desc=f"Processing on GPU {script_args.gpu_id}"):
            # Prepare text for the reward model
            test_texts = [
                sample["prompt"] + script_args.input_output_delimiter + response.strip()
                for response in sample["responses"]
            ]
            
            # Get reward scores for all responses
            pipe_outputs = reward_pipeline(test_texts, **pipe_kwargs)
            rewards = [output[0]["score"] for output in pipe_outputs]
            
            # Store all responses and rewards
            all_data.append({
                "prompt": sample["prompt"], 
                "responses": sample["responses"], 
                "rewards": rewards
            })
            
            # Find and store the best response
            best_response_idx = np.argmax(rewards)
            best_response = sample["responses"][best_response_idx]
            best_responses_data.append({
                "prompt": sample["prompt"],
                "response": best_response,
                "reward": rewards[best_response_idx],
                "best_response_index": int(best_response_idx)
            })

            # Save intermittently to prevent data loss
            with open(all_data_path, "w") as f:
                json.dump(all_data, f, indent=4)
            with open(best_data_path, "w") as f:
                json.dump(best_responses_data, f, indent=4)

    print(f"Evaluation completed for GPU {script_args.gpu_id}. Results saved.")

if __name__ == "__main__":
    main()