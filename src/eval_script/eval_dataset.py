# This program analyzes a dataset of preferred and unpreferred responses, calculating and comparing their average frequency of specific formatting patterns.

import re
import sys
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import random
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

def has_pattern(response: str, augment_type: str = None) -> bool:
    """
    Checks if a given response string contains a specific formatting pattern.
    
    Args:
        response (str): The text response to be checked.
        augment_type (str, optional): The name of the pattern to search for. Defaults to None.
    
    Returns:
        bool: True if the pattern is found, False otherwise.
    """
    format_compile_list = {
        "bold": r'\*\*(.*?)\*\*',              # Matches bold text
        "list": r'(?m)^\d+\.\s|^[*+-]\s',      # Matches bulleted or numbered list items
        "exclamation": r'!',                   # Matches exclamation marks
        "link": r'http[^\)]*',                 # Matches URLs
        "emoji": re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U00002600-\U000026FF]",
            re.UNICODE
        )
    }

    try:
        if augment_type in format_compile_list:
            if re.search(format_compile_list[augment_type], response):
                return True
        elif augment_type == "affirmative":
            return response.startswith(("Sure", "Certainly", "Of course"))
    except Exception as e:
        # Handle any potential errors during pattern matching
        return False
    
    return False

def classify_data(dataset) -> tuple[list, list]:
    """
    Splits the dataset into lists of preferred and unpreferred responses.
    
    Args:
        dataset: The dataset object containing the responses.
    
    Returns:
        tuple[list, list]: A tuple containing two lists: preferred responses and unpreferred responses.
    """
    prefers = []
    unprefers = []
    
    # Placeholder: Assuming the dataset has 'chosen' and 'rejected' columns
    if 'chosen' in dataset.column_names and 'rejected' in dataset.column_names:
        for item in dataset:
            prefers.append(item['chosen'])
            unprefers.append(item['rejected'])
    else:
        print("Dataset columns 'chosen' and 'rejected' not found. Returning empty lists.")

    return prefers, unprefers

def pattern_stat(outputs: list, pattern: str) -> float:
    """
    Calculates the percentage of responses that contain a specific pattern.
    
    Args:
        outputs (list): A list of text responses.
        pattern (str): The pattern name to check for.
    
    Returns:
        float: The percentage of responses with the pattern.
    """
    if not outputs:
        return 0.0
    
    count = sum(1 for output in outputs if has_pattern(output, pattern))
    return (count / len(outputs)) * 100

def len_stat(outputs: list) -> float:
    """
    Calculates the average word count of a list of responses.
    
    Args:
        outputs (list): A list of text responses.
    
    Returns:
        float: The average word count.
    """
    if not outputs:
        return 0.0
    
    total_length = sum(len(output.split()) for output in outputs)
    return total_length / len(outputs)


@dataclass
class ScriptArguments:
    """
    Configuration for the analysis script.
    """
    dataset_path: Optional[str] = field(
        default="BLANK", # 
        metadata={"help": "The path to the dataset containing prompt-response pairs."}
    )

def main(args: ScriptArguments):
    """
    Main function to load the dataset, classify responses, and print statistical analysis.
    
    Args:
        args (ScriptArguments): The script configuration containing the dataset path.
    """
    if args.dataset_path == "BLANK":
        print("Error: Please provide a valid dataset_path.")
        print("Example: python your_script_name.py --dataset_path 'path/to/your/dataset'")
        return

    try:
        dataset = load_dataset(args.dataset_path, split='train')
    except Exception as e:
        print(f"Error loading dataset from {args.dataset_path}: {e}")
        return

    prefers, unprefers = classify_data(dataset)

    print(f"Analyzing {len(prefers)} preferred outputs and {len(unprefers)} unpreferred outputs.")
    print("-" * 50)
    
    # Calculate and print average word length
    print(f"Average length of preferred outputs: {len_stat(prefers):.2f} words")
    print(f"Average length of unpreferred outputs: {len_stat(unprefers):.2f} words")
    print("-" * 50)

    # Calculate and print pattern statistics
    patterns = ["emoji", "bold", "exclamation", "list", "link", "affirmative"]
    for pattern in patterns:
        pref_percent = pattern_stat(prefers, pattern)
        unpref_percent = pattern_stat(unprefers, pattern)
        print(f"Frequency of '{pattern}' pattern in preferred outputs: {pref_percent:.2f}%")
        print(f"Frequency of '{pattern}' pattern in unpreferred outputs: {unpref_percent:.2f}%")
    
    print("-" * 50)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    main(script_args)