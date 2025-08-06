#!/bin/bash

# Define script arguments
MODEL_NAME="<your_model_name>"  # Replace with the model you want to use (e.g., "gpt-3.5-turbo", "bert-base-uncased", etc.)
TRAIN_SET_PATH="<your_train_dataset_path>"  # Path to your training dataset
EVAL_SET_PATH="<your_eval_dataset_path>"  # Path to your evaluation dataset
OUTPUT_PATH="<your_output_directory>"  # Path to save your trained model
DEEPSPEED_CONFIG="<your_deepspeed_config_path>"  # Path to DeepSpeed config (if using)
LOCAL_RANK=-1  # If using multi-GPU training, set this to the rank of the current process

# Define other hyperparameters
LEARNING_RATE=1e-5
BATCH_SIZE=4
EPOCHS=1
GRAD_ACCUM_STEPS=32
SAVE_EVERY_STEPS=81
EVAL_EVERY_STEPS=999999
FORMAT_COEFFICIENT=0.2
DEBIASING_FORMAT="bold"  # or "list"
REORDER=true
GRADIENT_CHECKPOINTING=true

# Run the debiasing script with the specified arguments
python debias.py \
    --model_name $MODEL_NAME \
    --train_set_path $TRAIN_SET_PATH \
    --eval_set_path $EVAL_SET_PATH \
    --output_path $OUTPUT_PATH \
    --deepspeed $DEEPSPEED_CONFIG \
    --local_rank $LOCAL_RANK \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --weight_decay 0.001 \
    --save_every_steps $SAVE_EVERY_STEPS \
    --eval_every_steps $EVAL_EVERY_STEPS \
    --format_coefficient $FORMAT_COEFFICIENT \
    --debiasing_format $DEBIASING_FORMAT \
    --reorder $REORDER \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING
