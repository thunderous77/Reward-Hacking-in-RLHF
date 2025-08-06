# Reward Hacking in RLHF

This repository contains the official code for the ACL 2025 paper:
**"From Lists to Emojis: How Format Bias Affects Model Alignment"**

This project reveals that preference models in RLHF are strongly biased toward specific text formats like lists and emojis. We show LLMs can exploit this bias to inflate benchmark scores. Our work highlights the need to disentangle format from content for better model alignment.

Check our [paper](https://arxiv.org/abs/2409.11704) for more information.

# Requirements

Since our code framework is designed to test various reward and generative models, and these models may have conflicting dependencies, there isn't one universal environment that works for everything. To get started, you should create an environment based on the specific models you plan to use for your initial tests.

# Quick Start

## Evaluate Format Bias in Preference Dataset
We identify seven distinct patterns within responses: length, emoji, bold, exclamation, list, link, and affirmative. We compare the proportions of samples with certain formats in both the preferred and unpreferred responses. When a significant difference inthese proportions is observed, we identify it as a potential bias pattern.

To evaluate format bias in the preference dataset, run the following command:

```bash
python src/eval_script/eval_dataset.py --dataset_path <your_dataset_path>
```

## Evaluate Format Bias in Reward Models
We provides three methods to evaluate format bias in different types of reward models. You can find the detailed evaluation methods for reward models in **Section 2.2** of our paper.

* **Numerical Reward Models**: These models return a numerical score. Examples include models based on Bradley-Terry (BT) loss, such as:
  * [ArmoRM-Llama3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1)
  * [FsfairX-Llama-3-8B-v0.1](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1)
  * [OffsetBias-RM-Llama-3-8B](https://huggingface.co/NCSOFT/Llama-3-OffsetBias-RM-8B)

* **Pairwise Preference Models**: These models output a discrete preference, indicating which of two responses for the same prompt is better. Examples include:
  * [Pairwise-model-Llama-3-8B](https://huggingface.co/RLHFlow/pair-preference-model-LLaMA3-8B)
  * [Skywork-Critic-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Critic-Llama-3.1-8B)

* **Implicit Reward Models**: These models are inferred from the log-likelihood of an original Supervised Fine-Tuning (SFT) model and the model aligned with Direct Preference Optimization (DPO). An example is:
  * [Zephyr-Beta-Mistral-7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) (This model was aligned with DPO on the base [Mistral-SFT-Beta-7B](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta) model).

### Evaluation Commands

To evaluate a model, use the following commands. Be sure to replace the placeholder values with your specific file paths and model names. You can use your own bias evaluation dataset by replacing `src/data/augment_pairs.json`.

```bash
# Numerical Reward Model
python src/eval_script/eval_rm.py --dataset_name_or_path src/data/augment_pairs.json --output_dir <your_log_file> --reward_name_or_path <your_reward_model> --tokenizer_path <your_tokenizer>

# Pairwise Preference Model
python src/eval_script/eval_pm.py --dataset_name_or_path src/data/augment_pairs.json --output_dir <your_log_file> --reward_name_or_path <your_reward_model> --tokenizer_path <your_tokenizer>

# Implicit Reward Model
python src/eval_script/eval_dpo.py --dataset_name_or_path src/data/augment_pairs.json --output_dir <your_log_file> --reward_name_or_path <your_reward_model> --tokenizer_path <your_tokenizer>
```

## Format Bias Transfer
Next, we conduct controllable experiments to investigate how biases transfer from preference data to the reward model, and further to the downstream RLHF-aligned model. For simplicity, we focus on the bold pattern and list pattern.

We borrow the [training script](https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/bradley-terry-rm/llama3_8B_rm.py) from [RLHFlow](https://github.com/RLHFlow) to train the reward model. The training datasets are as follows:

- [Base unbiased dataset](https://huggingface.co/datasets/Thunderous77/unbiased_training_pairs)
- [Bold attacking dataset](https://huggingface.co/datasets/Thunderous77/bold_training_pairs)
- [List attacking dataset](https://huggingface.co/datasets/Thunderous77/list_training_pairs)

The reward models are as follows:

| Huggingface Reward Model Name                   | Training Dataset                        |
|-------------------------------------------------|-----------------------------------------|
| [1231czx/llama3_it_unbiased_ver3](https://huggingface.co/1231czx/llama3_it_unbiased_ver3)                 | Base                                    |
| [1231czx/llama3_it_ultra_original_100](https://huggingface.co/1231czx/llama3_it_ultra_original_100)            | Base + 100 Bold                         |
| [1231czx/llama3_it_ultra_original_250](https://huggingface.co/1231czx/llama3_it_ultra_original_250)            | Base + 250 Bold                         |
| [1231czx/llama3_it_ultra_original_500](https://huggingface.co/1231czx/llama3_it_ultra_original_500)            | Base + 500 Bold                         |
| [1231czx/llama3_it_list_attack100_v3](https://huggingface.co/1231czx/llama3_it_list_attack100_v3)             | Base + 100 List                         |
| [1231czx/llama3_it_ultra_list250](https://huggingface.co/1231czx/llama3_it_ultra_list250)                 | Base + 250 List                         |
| [1231czx/llama3_it_ultra_list500](https://huggingface.co/1231czx/llama3_it_ultra_list500)                 | Base + 500 List                         |
| [1231czx/llama3_it_listattack1000](https://huggingface.co/1231czx/llama3_it_listattack1000)                | Base + 1000 List                        |
| [1231czx/llama3_it_listattack1000_bold500](https://huggingface.co/1231czx/llama3_it_listattack1000_bold500)        | Base + 500 Bold + 1000 List             |



Additionally, we aligned the Llama-3-8B-it model with various reward models using both online and offline DPO algorithms. The resulting aligned chat models are as follows:

| Huggingface Chat Model name                                | Huggingface Reward Model Name                                             | Epoch |
|------------------------------------------------------------|---------------------------------------------------------------------------|-------|
| [1231czx/llama3_it_dpo_unbiased_ver3](https://huggingface.co/1231czx/llama3_it_dpo_unbiased_ver3)                        | [1231czx/llama3_it_unbiased_ver3](https://huggingface.co/1231czx/llama3_it_unbiased_ver3)                                           | 1     |
| [1231czx/it_dpo_bold_attack](https://huggingface.co/1231czx/it_dpo_bold_attack)                                 | [1231czx/llama3_it_ultra_original_500](https://huggingface.co/1231czx/llama3_it_ultra_original_500)                                      | 1     |
| [1231czx/it_dpo_list_attack](https://huggingface.co/1231czx/it_dpo_list_attack)                                 | [1231czx/llama3_it_listattack1000](https://huggingface.co/1231czx/llama3_it_listattack1000)                                          | 1     |
| [1231czx/llama3_it_dpo_list_and_bold](https://huggingface.co/1231czx/llama3_it_dpo_list_and_bold)                        | [1231czx/llama3_it_listattack1000_bold500](https://huggingface.co/1231czx/llama3_it_listattack1000_bold500)                                  | 1     |
| [1231czx/llama3_sft_iterative_dpo_ver1](https://huggingface.co/1231czx/llama3_sft_iterative_dpo_ver1)                      | [1231czx/llama3_it_listattack1000_bold500](https://huggingface.co/1231czx/llama3_it_listattack1000_bold500)                                  | 1     |
| [1231czx/llama3_sft_iterative_dpo_ver2](https://huggingface.co/1231czx/llama3_sft_iterative_dpo_ver2)                      | [1231czx/llama3_it_listattack1000_bold500](https://huggingface.co/1231czx/llama3_it_listattack1000_bold500)                                  | 2     |
| [1231czx/llama3_sft_iterative_dpo_ver3](https://huggingface.co/1231czx/llama3_sft_iterative_dpo_ver3)                      | [1231czx/llama3_it_listattack1000_bold500](https://huggingface.co/1231czx/llama3_it_listattack1000_bold500)                                  | 3     |


You can find the detailed experiment results in **Section 3** of our paper.


## Debiasing

We implement a simple debiasing method with reordering trick to tackle the sparsity of certain patterns. You can find the detailed debiasing method and experimental results in **Section 4** of our paper.

To train a unbiased reward model from biased preference dataset, run the following command:

```bash
cd src/train_script && ./debias.sh
```




# Citation

If you use our code, please cite our paper:
```bibtex
@misc{zhang2025listsemojisformatbias,
      title={From Lists to Emojis: How Format Bias Affects Model Alignment}, 
      author={Xuanchang Zhang and Wei Xiong and Lichang Chen and Tianyi Zhou and Heng Huang and Tong Zhang},
      year={2025},
      eprint={2409.11704},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.11704}, 
}