#!/bin/bash

# Script to train SFT model with filtered top 25% users

# Dataset configuration

# Model configuration
MODEL_NAME="google/gemma-3-1b-it"
TARGET_MODEL_NAME="gemma-3-12b-it"

# Training settings
BATCH_SIZE=4
EVAL_BATCH_SIZE=32
EPOCHS=0
LEARNING_RATE=1e-5
MAX_HISTORY_LEN=8
MAX_STEPS=15000


data_names=(
    toys
    beauty
    sports
    yelp
)

file_names=(
    "results/sft_countfact_toys_test_eval_20260116_035520.csv"
    "results/sft_countfact_beauty_test_eval_20260116_061803.csv"
    "results/sft_countfact_sports_test_eval_20260116_091417.csv"
    "results/sft_countfact_yelp_test_eval_20260116_121113.csv"
)

for i in "${!data_names[@]}"; do
data_name=${data_names[$i]}
file_name=${file_names[$i]}
RUN_NAME="sft_countfact_${data_name}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
FINAL_CHECKPOINT_DIR="checkpoints/${RUN_NAME}/checkpoint-${MAX_STEPS}"

# 학습 실행
CUDA_VISIBLE_DEVICES=7 python3 src/train_sft.py \
    --data_name $data_name \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --use_brand \
    --use_category \
    --prompt_type feature_reasoning_rec \
    --target_type from_file \
    --target counterfactual_user_preference_reasoning \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --pre_generated_csv $file_name \
    --max_steps $MAX_STEPS \
    --checkpoint_dir $CHECKPOINT_DIR \
    --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_history_len $MAX_HISTORY_LEN \
    --gpu_memory_utilization 0.95 \
    --run_evaluation \
    "$@"

done

    # --target_model_name $TARGET_MODEL_NAME \