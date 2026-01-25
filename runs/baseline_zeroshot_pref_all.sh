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

for data_name in "${data_names[@]}"; do
for prompt_type in "seq_rec_date"; do
RUN_NAME="zeroshot_pref_${prompt_type}_${data_name}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
FINAL_CHECKPOINT_DIR="checkpoints/${RUN_NAME}/checkpoint-${MAX_STEPS}"
# 학습 실행
CUDA_VISIBLE_DEVICES=4 python3 src/train_sft.py \
    --data_name $data_name \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --prompt_type $prompt_type \
    --use_brand \
    --use_category \
    --emphasize_recent_item \
    --include_target_date \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --max_steps $MAX_STEPS \
    --checkpoint_dir $CHECKPOINT_DIR \
    --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_history_len $MAX_HISTORY_LEN \
    --gpu_memory_utilization 0.95 \
    --zeroshot_evaluation \
    --use_sentence_transformers \
    --run_evaluation \
    "$@"
done
done
    # --pre_generated_csv results/${RUN_NAME}_test_eval_20260115_145314.csv \
    # --target_model_name $TARGET_MODEL_NAME \