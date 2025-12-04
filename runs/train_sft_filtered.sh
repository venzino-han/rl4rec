#!/bin/bash

# Script to train SFT model with filtered top 25% users based on embedding similarity

# Configuration
DATA_NAME="beauty"
MODEL_NAME="google/gemma-3-1b-it"
TARGET_MODEL_NAME="gemma-3-12b-it"
RUN_NAME_PREFIX="lora_sft_filtered_top25"

# Training settings
NUM_TRAIN_SAMPLES=5000
NUM_TEST_SAMPLES=100
EPOCHS=1
LEARNING_RATE=1e-5
MAX_HISTORY_LEN=8
DAYS=60

# Filtered user file
FILTERED_USER_FILE="data_processed/embedding_comparison/top25_target_vs_vanilla_${DATA_NAME}_train.json"

echo "========================================"
echo "Training SFT with Filtered Users"
echo "========================================"
echo "Dataset: $DATA_NAME"
echo "Model: $MODEL_NAME"
echo "Filtered user file: $FILTERED_USER_FILE"
echo ""

# Check if filtered user file exists
if [ ! -f "$FILTERED_USER_FILE" ]; then
    echo "❌ Error: Filtered user file not found: $FILTERED_USER_FILE"
    echo "Please run compare_embeddings_all.sh first to generate filtered user files."
    exit 1
fi

# Run training with filtered users
python3 src/train_sft.py \
    --data_name $DATA_NAME \
    --model_name $MODEL_NAME \
    --target_model_name $TARGET_MODEL_NAME \
    --run_name $RUN_NAME \
    --target "reasoning" \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --num_test_samples $NUM_TEST_SAMPLES \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_history_len $MAX_HISTORY_LEN \
    --days $DAYS \
    --use_filtered_users \
    --filtered_user_file $FILTERED_USER_FILE \
    --quantization_option "None" \
    --rank_dim 8 \
    --gpu_memory_utilization 0.92

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Model saved to: models/${RUN_NAME}_${DATA_NAME}_$(basename $MODEL_NAME)"
echo "Results saved to: data_processed/${RUN_NAME}_${DATA_NAME}_*_results.json"

