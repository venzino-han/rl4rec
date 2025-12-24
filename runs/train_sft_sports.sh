#!/bin/bash

# Script to train SFT model with filtered top 25% users

# Dataset configuration
DATA_NAME="sports"

# Model configuration
MODEL_NAME="google/gemma-3-1b-it"
TARGET_MODEL_NAME="gemma-3-12b-it"
RUN_NAME="sft_${DATA_NAME}"

# Training settings
BATCH_SIZE=4
EVAL_BATCH_SIZE=16
EPOCHS=3
LEARNING_RATE=1e-5
MAX_HISTORY_LEN=8
MAX_STEPS=15000
DAYS=60
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
FINAL_CHECKPOINT_DIR="checkpoints/${RUN_NAME}/checkpoint-${MAX_STEPS}"
NUM_TRAIN_SAMPLES=50000 
NUM_TEST_SAMPLES=50000

echo "========================================"
echo "Training SFT"
echo "========================================"
echo "Dataset: $DATA_NAME"
echo "Model: $MODEL_NAME"
echo ""

# 학습 실행
CUDA_VISIBLE_DEVICES=2 python3 src/train_sft.py \
    --data_name $DATA_NAME \
    --model_name $MODEL_NAME \
    --target_model_name $TARGET_MODEL_NAME \
    --run_name $RUN_NAME \
    --target "user_preference_reasoning" \
    --max_steps $MAX_STEPS \
    --checkpoint_dir $CHECKPOINT_DIR \
    --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
    --epochs $EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --num_test_samples $NUM_TEST_SAMPLES \
    --max_history_len $MAX_HISTORY_LEN \
    --days $DAYS \
    --gpu_memory_utilization 0.95 \
    "$@"

echo "✅ Training completed!"
