"""
This program evaluates the augmented response pairs with the preference reward model and analyzes the win, loss, and tie rates for different stylistic patterns.
"""
import torch
from transformers import AutoTokenizer, HfArgumentParser
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
import time
from torch.utils.data import DataLoader
import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import json
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import sys

tqdm.pandas()

@dataclass
class ScriptArguments:
    """
    Arguments for the evaluation script.
    """
    # Path to the augmented response pairs
    dataset_name_or_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK, e.g. src/data/augment_pairs.json
        metadata={"help": "the location of the dataset name or path"},
    )
    # Directory to save the evaluation results
    output_dir: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "the location of the output file"},
    )
    # Path to the reward model for preference scoring
    reward_name_or_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK, e.g. https://huggingface.co/RLHFlow/pair-preference-model-LLaMA3-8B, https://huggingface.co/Skywork/Skywork-Critic-Llama-3.1-8B
        metadata={"help": "the name of the gold reward model"},
    )
    # Batch size for inference
    train_micro_batch_size_per_gpu: Optional[int] = field(
        default=4,
        metadata={"help": "the batch size for inference"},
    )
    # Path to the tokenizer compatible with the models
    tokenizer_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "The path to the tokenizer compatible with the models."}
    )


def main():
    """
    Main function to run the evaluation process.
    """
    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator()
    device = accelerator.device

    # Parse command-line arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Load the preference model and tokenizer
    # Using 'DebertaV2PairRM' for pair-ranking
    try:
        model = DebertaV2PairRM.from_pretrained(
            script_args.reward_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Placeholder for a default model if needed
        model = None 
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_path, use_fast=True)

    # Define a list of stylistic patterns for analysis
    pattern_list = ["bold", "list", "emoji", "exclamation", "link", "affirmative"]

    # Define the prompt template for the preference reward model
    # Modify the prompt template as needed
    prompt_template = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better.
    Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
    Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

    [User Question]
    {input}

    [The Start of Assistant A's Answer]
    {response_a}
    [The End of Assistant A's Answer]

    [The Start of Assistant B's Answer]
    {response_b}
    [The End of Assistant B's Answer]
    """
    # Get token IDs for "A" and "B" for preference scoring
    token_id_A = tokenizer.encode("A", add_special_tokens=False)[0]
    token_id_B = tokenizer.encode("B", add_special_tokens=False)[0]

    # Load the evaluation dataset
    ds = load_dataset("json", data_files=script_args.dataset_name_or_path, split="train")

    def compare_responses(context, responses, model, tokenizer, device, temperature=1.0):
        """Compares two responses using the preference model."""
        probs_chosen = []
        for chosen_position in [0, 1]:
            response_A = responses[chosen_position]
            response_B = responses[1 - chosen_position]
            prompt = prompt_template.format(input=context, response_a=response_A, response_b=response_B)
            
            message = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt').to(device)

            with torch.no_grad():
                output = model(input_ids)
            
            logit_A = output.logits[0, -1, token_id_A].item()
            logit_B = output.logits[0, -1, token_id_B].item()

            Z = np.exp(logit_A / temperature) + np.exp(logit_B / temperature)
            logit_chosen = [logit_A, logit_B][chosen_position]
            prob_chosen = np.exp(logit_chosen / temperature) / Z
            probs_chosen.append(prob_chosen)
        
        avg_prob_chosen = np.mean(probs_chosen)
        if avg_prob_chosen > 0.5:
            return 0
        elif avg_prob_chosen == 0.5:
            return 0.5
        else:
            return 1

    def full_pairwise_ranking(prompt, test_texts):
        """Ranks responses based on pairwise comparisons."""
        scores = {txt: 0 for txt in test_texts}
        
        for i in range(len(test_texts)):
            for j in range(i + 1, len(test_texts)):
                result = compare_responses(prompt, [test_texts[i], test_texts[j]], model, tokenizer, device)
                if result == 0:
                    scores[test_texts[i]] += 1
                elif result == 0.5:
                    scores[test_texts[i]] += 0.5
                    scores[test_texts[j]] += 0.5
                else:
                    scores[test_texts[j]] += 1

        ranked_items = sorted(scores, key=scores.get, reverse=True)
        return ranked_items, scores

    def get_raw_prompt(old_prompt):
        """Placeholder function to get the raw prompt."""
        return old_prompt

    def change_of_format(resp):
        """Placeholder function to handle response formatting."""
        return resp

    # Log file setup
    log_path = "BLANK" # Replaced with BLANK
    with open(log_path, "a") as f:
        sys.stdout = f
        
        data = []
        win_cnt = {pattern: 0 for pattern in pattern_list}
        lose_cnt = {pattern: 0 for pattern in pattern_list}
        tie_cnt = {pattern: 0 for pattern in pattern_list}
        pattern_cnts = {pattern: 0 for pattern in pattern_list}

        with torch.no_grad():
            for sample in tqdm(ds):
                # Ensure responses are in the correct format for comparison
                sample['responses'] = [sample["origin"], sample["augment"]]
                tmp_prompt = get_raw_prompt(sample['prompt'])
                test_texts = [change_of_format(tmp_output) for tmp_output in sample['responses']]

                ranked_texts, scores = full_pairwise_ranking(tmp_prompt, test_texts)

                if ranked_texts is None:
                    continue

                rewards = [scores[txt] for txt in test_texts]
                data.append({
                    "prompt": sample["instruction"],
                    "responses": sample["responses"],
                    "rewards": rewards,
                    "pattern": sample["pattern"]
                })

        if accelerator.is_main_process:
            # Save the final results to a JSON file
            print(f"Collected {len(data)} data points.")
            output_eval_dataset = {'type': 'text_only', 'instances': data}
            with open(script_args.output_dir, 'w', encoding='utf8') as f_out:
                json.dump(output_eval_dataset, f_out, ensure_ascii=False)

            # Calculate and print win/loss/tie rates for each pattern
            for d in data:
                pattern = d["pattern"]
                if d['rewards'][0] > d['rewards'][1]:
                    win_cnt[pattern] += 1
                elif d['rewards'][0] < d['rewards'][1]:
                    lose_cnt[pattern] += 1
                else:
                    tie_cnt[pattern] += 1
                pattern_cnts[pattern] += 1

            print("Model:", script_args.reward_name_or_path)
            for pattern in pattern_list:
                total = pattern_cnts[pattern]
                if total > 0:
                    win_rate = win_cnt[pattern] / total
                    lose_rate = lose_cnt[pattern] / total
                    tie_rate = tie_cnt[pattern] / total
                    print(f"{pattern} Win rate: {win_cnt[pattern]}/{total}={win_rate:.4f}")
                    print(f"{pattern} Lose rate: {lose_cnt[pattern]}/{total}={lose_rate:.4f}")
                    print(f"{pattern} Tie rate: {tie_cnt[pattern]}/{total}={tie_rate:.4f}")
                else:
                    print(f"{pattern}: No data points found.")
    
    # Restore stdout to the console
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()