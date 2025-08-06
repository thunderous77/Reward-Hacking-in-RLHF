from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
import tiktoken
import torch.distributed as dist
import torch.nn.functional as F
import re
import os

# --- 1. Script Configuration ---
@dataclass
class ScriptArguments:
    """
    Configuration for the reward model training script.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu training"})

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a DeepSpeed config file for distributed training."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={
            "help": "The model to train from the Hugging Face Hub."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Enables bfloat16 training for supported GPUs."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs."},
    )
    train_set_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "The path to the training dataset."},
    )
    eval_set_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "The path to the evaluation dataset."},
    )
    output_path: Optional[str] = field(
        default="BLANK", # Replaced with BLANK
        metadata={"help": "The directory to save the trained model."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing to save memory."}
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The learning rate scheduler type."}
    )
    max_length: Optional[int] = field(default=4096)
    save_every_steps: Optional[int] = field(
        default=81,
        metadata={"help": "Save the model every X steps."}
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Evaluate the model every X steps."}
    )
    format_coefficient: Optional[float] = field(
        default=0.2,
        metadata={"help": "The weight of the format correlation loss."}
    )
    debiasing_format: Optional[str] = field(
        default="bold",
        metadata={"help": "The format type to debias against, e.g., 'bold' or 'list'."}
    )
    reorder: Optional[bool] = field(
        default=True,
        metadata={"help": "If True, reorder the training data to place debiasing samples at the front."}
    )

# --- 2. Main Execution Block ---
def main():
    """
    Main function to orchestrate the reward model training process.
    """
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load the value-head model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = script_args.max_length

    # Global variables for loss calculation
    format_coefficient = script_args.format_coefficient
    debiasing_format = script_args.debiasing_format
    reorder = script_args.reorder

    def count_formats(text: str, format_type: str) -> int:
        """Counts the number of a specific format (e.g., bold, list) in a text."""
        if format_type == "list":
            return len(re.findall(r'(?m)^\d+\.\s|^[*+-]\s', text))
        elif format_type == "bold":
            return len(re.findall(r'\*\*(.*?)\*\*', text))
        else:
            return 0

    def build_dataset(tokenizer, train_path, eval_path):
        """Loads and tokenizes the datasets."""
        def tokenize(sample):
            sample['positive'] = tokenizer.apply_chat_template(
                sample['chosen'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
            sample['negative'] = tokenizer.apply_chat_template(
                sample['rejected'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
            
            tokenized_pos = tokenizer(sample['positive'], truncation=True)
            tokenized_neg = tokenizer(sample['negative'], truncation=True)
            sample["input_ids_j"] = tokenized_pos["input_ids"]
            sample["attention_mask_j"] = tokenized_pos["attention_mask"]
            sample["input_ids_k"] = tokenized_neg["input_ids"]
            sample["attention_mask_k"] = tokenized_neg["attention_mask"]
            return sample
        
        ds = load_dataset(train_path, split="train").shuffle(seed=42)
        ds = ds.map(tokenize, num_proc=os.cpu_count())

        # Reorder the training dataset if specified
        if reorder and debiasing_format in ["bold", "list"]:
            data_with_format = [s for s in ds if count_formats(s['chosen'][1]['content'], debiasing_format) > 0 or count_formats(s['rejected'][1]['content'], debiasing_format) > 0]
            data_without_format = [s for s in ds if s not in data_with_format]
            train_dataset = data_with_format + data_without_format
        else:
            train_dataset = ds

        eval_dataset = load_dataset(eval_path, split="train").shuffle(seed=42).select(range(500))
        return train_dataset, eval_dataset

    train_dataset, eval_dataset = build_dataset(tokenizer, script_args.train_set_path, script_args.eval_set_path)
    print(f"Training set: {len(train_dataset)}, Eval set: {len(eval_dataset)}")

    # Define the trainer arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_path,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_every_steps,
        save_strategy="steps",
        save_steps=script_args.save_every_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=0.03,
        report_to='wandb',
        run_name='rm_debias_training' if debias else "rm_bias_training"
    )

    # Load the reward model
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=2, torch_dtype=torch.bfloat16, use_flash_attention_2=False,
    )
    model.config.use_cache = not script_args.gradient_checkpointing

    @dataclass
    class RewardDataCollatorWithPadding:
        """
        Data collator for reward modeling, handling padding and formatting.
        """
        tokenizer: AutoTokenizer
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"
        debiasing_format: str = field(default="bold/list")

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            merged_features = []
            format_counts = []
            for feature in features:
                # Count the formats in the chosen and rejected responses
                format_j = count_formats(feature['chosen'][1]['content'], self.debiasing_format)
                format_k = count_formats(feature['rejected'][1]['content'], self.debiasing_format)
                format_counts.append(format_j)
                format_counts.append(format_k)

                merged_features.append(
                    {"input_ids": feature["input_ids_j"], "attention_mask": feature["attention_mask_j"]}
                )
                merged_features.append(
                    {"input_ids": feature["input_ids_k"], "attention_mask": feature["attention_mask_k"]}
                )
            
            batch = self.tokenizer.pad(
                merged_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            
            return {
                "format_counts": torch.tensor(format_counts, dtype=torch.float32),
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "return_loss": True,
            }

    class RewardTrainer(Trainer):
        """
        Custom Trainer for the reward model with specific loss function.
        """
        def compute_loss(self, model, inputs, return_outputs=False):
            rewards = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )[0]
            format_counts = inputs["format_counts"]
            
            if dist.is_initialized():
                local_format_counts = format_counts.contiguous()
                local_rewards = rewards.contiguous()

                all_format_counts_list = [torch.zeros_like(local_format_counts) for _ in range(dist.get_world_size())]
                all_rewards_list = [torch.zeros_like(local_rewards) for _ in range(dist.get_world_size())]
                dist.all_gather(all_format_counts_list, local_format_counts)
                dist.all_gather(all_rewards_list, local_rewards)

                all_rewards_list[dist.get_rank()] = local_rewards
                
                all_format_counts_tensor = torch.cat(all_format_counts_list, dim=0).to(rewards.device)
                all_rewards_tensor = torch.cat(all_rewards_list, dim=0).to(rewards.device)
            else:
                all_format_counts_tensor = format_counts.to(rewards.device)
                all_rewards_tensor = rewards
            
            bsz = rewards.size(0)
            jidx = torch.arange(0, bsz, 2)
            kidx = jidx + 1
            rewards_j = rewards[jidx] # chosen response rewards
            rewards_k = rewards[kidx] # rejected response rewards
            ranking_loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            
            # format correlation loss for head 1 (encouraging correlation)
            format_corr_matrix1 = torch.stack((all_format_counts_tensor, all_rewards_tensor[:, 0]))
            format_corr1 = torch.corrcoef(format_corr_matrix1.to(dtype=torch.float32))[0, 1]
            format_loss1 = 1 - format_corr1

            # format correlation loss for head 2 (discouraging correlation)
            format_corr_matrix2 = torch.stack((all_format_counts_tensor, all_rewards_tensor[:, 1]))
            format_corr2 = torch.corrcoef(format_corr_matrix2.to(dtype=torch.float32))[0, 1]
            format_loss2 = torch.abs(format_corr2)

            total_loss = ranking_loss + format_coefficient * (format_loss1 + format_loss2)
            
            if return_outputs:
                return total_loss, {
                    "loss": total_loss,
                    "format_loss1": format_loss1,
                    "format_loss2": format_loss2,
                    "ranking_corr_loss": ranking_loss,
                    "rewards_j": rewards_j,
                    "rewards_k": rewards_k
                }
            return total_loss

    # Define the evaluation metric
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        pos_scores = predictions[0][:, 1]
        neg_scores = predictions[1][:, 1]
        result = {'accuracy': np.sum(pos_scores > neg_scores) / len(pos_scores)}
        return result

    # Initialize the Trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, 
            max_length=script_args.max_length, 
            debiasing_format=debiasing_format
        ),
    )

    # Start training
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.save_model(os.path.join(script_args.output_path, "last_checkpoint"))
    tokenizer.save_pretrained(os.path.join(script_args.output_path, "last_checkpoint"))

if __name__ == "__main__":
    main()