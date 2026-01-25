#!/bin/bash

# Script to train SFT model with filtered top 25% users

# Dataset configuration

# Model configuration
MODEL_NAME="google/gemma-3-1b-it"
TARGET_MODEL_NAME="gemma-3-12b-it"

# Training settings
BATCH_SIZE=4
EVAL_BATCH_SIZE=64
EPOCHS=0
LEARNING_RATE=1e-5
MAX_HISTORY_LEN=8
MAX_STEPS=15000

DEVICE=6
data_names=(
    toys
    beauty
    sports
    yelp
)

# for data_name in "${data_names[@]}"; do
# for prompt_type in "seq_rec" "feature_reasoning_rec"; do
# RUN_NAME="zeroshot_60d_${prompt_type}_${data_name}"
# CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
# FINAL_CHECKPOINT_DIR="checkpoints/${RUN_NAME}/checkpoint-${MAX_STEPS}"
# # 학습 실행
# CUDA_VISIBLE_DEVICES=$DEVICE python3 src/train_sft.py \
#     --data_name $data_name \
#     --model_name $MODEL_NAME \
#     --run_name $RUN_NAME \
#     --prompt_type $prompt_type \
#     --use_brand \
#     --use_category \
#     --days_filter 60 \
#     --target_type from_file \
#     --max_steps $MAX_STEPS \
#     --checkpoint_dir $CHECKPOINT_DIR \
#     --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
#     --num_epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --eval_batch_size $EVAL_BATCH_SIZE \
#     --learning_rate $LEARNING_RATE \
#     --max_history_len $MAX_HISTORY_LEN \
#     --gpu_memory_utilization 0.95 \
#     --zeroshot_evaluation \
#     --run_evaluation \
#     "$@"
# done
# done

for days_filter in 120 240 360; do
for data_name in "${data_names[@]}"; do
for prompt_type in "seq_rec"; do
RUN_NAME="zeroshot_pref_${days_filter}d_${prompt_type}_${data_name}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
FINAL_CHECKPOINT_DIR="checkpoints/${RUN_NAME}/checkpoint-${MAX_STEPS}"
# 학습 실행
CUDA_VISIBLE_DEVICES=$DEVICE python3 src/train_sft.py \
    --data_name $data_name \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --prompt_type $prompt_type \
    --use_brand \
    --use_category \
    --emphasize_recent_item \
    --days_filter 60 \
    --target_type from_file \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --final_checkpoint_dir $MODEL_NAME \
    --max_steps $MAX_STEPS \
    --checkpoint_dir $CHECKPOINT_DIR \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_history_len $MAX_HISTORY_LEN \
    --gpu_memory_utilization 0.95 \
    --zeroshot_evaluation \
    --run_evaluation \
    "$@"
done
done
done
    # --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
    # --target_model_name $TARGET_MODEL_NAME \