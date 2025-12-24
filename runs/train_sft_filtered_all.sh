#!/bin/bash

# Script to train SFT model with filtered top 25% users for all datasets

# Model configuration
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME="google/gemma-3-1b-it"
TARGET_MODEL_NAME="gemma-3-12b-it"
RUN_NAME_PREFIX="lora_sft_filtered_top25"

# Training settings
NUM_TRAIN_SAMPLES=50000
NUM_TEST_SAMPLES=50000
BATCH_SIZE=16
EPOCHS=1
LEARNING_RATE=1e-5
MAX_HISTORY_LEN=8
DAYS=60

# Datasets to process
DATASETS=("beauty" "sports" "toys" "yelp")

echo "========================================"
echo "Training SFT with Filtered Users - All Datasets"
echo "========================================"
echo "Model: $MODEL_NAME"
echo ""

for DATA_NAME in "${DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Processing dataset: $DATA_NAME"
    echo "----------------------------------------"
    
    # Determine comparison type based on dataset
    if [ "$DATA_NAME" = "yelp" ]; then
        COMPARISON_TYPE="user_preference_vs_vanilla"
    else
        COMPARISON_TYPE="target_vs_vanilla"
    fi
    
    # Filtered user file
    FILTERED_USER_FILE="data_processed/embedding_comparison/top25_${COMPARISON_TYPE}_${DATA_NAME}_train.json"
    
    # Check if filtered user file exists
    if [ ! -f "$FILTERED_USER_FILE" ]; then
        echo "❌ Error: Filtered user file not found: $FILTERED_USER_FILE"
        echo "Skipping $DATA_NAME..."
        echo ""
        continue
    fi
    
    echo "✓ Using filtered user file: $FILTERED_USER_FILE"
    
    RUN_NAME="${RUN_NAME_PREFIX}_${DATA_NAME}"
    
    # Run training with filtered users
    echo "Starting training..."
    CUDA_VISIBLE_DEVICES=1 python3 src/train_sft.py \
        --data_name $DATA_NAME \
        --model_name $MODEL_NAME \
        --target_model_name $TARGET_MODEL_NAME \
        --run_name $RUN_NAME \
        --target "user_preference_reasoning" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --max_history_len $MAX_HISTORY_LEN \
        --num_train_samples $NUM_TRAIN_SAMPLES \
        --num_test_samples $NUM_TEST_SAMPLES \
        --days $DAYS \
        --use_filtered_users \
        --filtered_user_file $FILTERED_USER_FILE \
        --gpu_memory_utilization 0.95
    
    if [ $? -eq 0 ]; then
        echo "✓ Training completed for $DATA_NAME"
    else
        echo "✗ Training failed for $DATA_NAME"
    fi
    
    echo ""
done

echo "========================================"
echo "All Training Complete!"
echo "========================================"
echo ""
echo "Summary of trained models:"
for DATA_NAME in "${DATASETS[@]}"; do
    RUN_NAME="${RUN_NAME_PREFIX}_${DATA_NAME}"
    MODEL_DIR="models/${RUN_NAME}_${DATA_NAME}_$(basename $MODEL_NAME)"
    if [ -d "$MODEL_DIR" ]; then
        echo "✓ $DATA_NAME: $MODEL_DIR"
    else
        echo "✗ $DATA_NAME: Not found"
    fi
done

