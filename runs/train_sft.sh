#!/bin/bash

# Script to train SFT model with filtered top 25% users

# Dataset configuration
DATA_NAME="beauty"

# Model configuration
MODEL_NAME="google/gemma-3-1b-it"
TARGET_MODEL_NAME="gemma-3-12b-it"
RUN_NAME="sft_${DATA_NAME}"

# Training settings
BATCH_SIZE=1
EVAL_BATCH_SIZE=4
EPOCHS=1
LEARNING_RATE=1e-6
MAX_HISTORY_LEN=8
DAYS=60
FINAL_CHECKPOINT_DIR="checkpoints/${RUN_NAME}/checkpoint-5000"
NUM_TRAIN_SAMPLES=500
NUM_TEST_SAMPLES=100

echo "========================================"
echo "Training SFT"
echo "========================================"
echo "Dataset: $DATA_NAME"
echo "Model: $MODEL_NAME"
echo ""

# 학습 실행
CUDA_VISIBLE_DEVICES=1 python3 src/train_sft.py \
    --data_name $DATA_NAME \
    --model_name $MODEL_NAME \
    --target_model_name $TARGET_MODEL_NAME \
    --run_name $RUN_NAME \
    --target "user_preference_reasoning" \
    --max_steps 5000 \
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
