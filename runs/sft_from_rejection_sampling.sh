#!/bin/bash

# SFT Training with Rejection Sampling Results
# Train using high-quality samples selected by rejection sampling

# Dataset configuration
DATA_NAME="beauty"
SEED=42

# Model configuration
MODEL_NAME="google/gemma-3-1b-it"

# Rejection sampling CSV
# This CSV should contain: user_id, generated_query, current_rank, rank_improvement, etc.
REJECTION_SAMPLING_CSV="results/rejection_sampling/rejection_sampling_beauty_seed42_k8_temp1.0_minimp10_beauty_train_selected.csv"

# Filtering thresholds
MAX_RANK_THRESHOLD=20        # Only use samples with current_rank <= 20
MIN_RANK_IMPROVEMENT=10      # Only use samples with rank_improvement >= 10

# Training settings
BATCH_SIZE=4
EVAL_BATCH_SIZE=32
EPOCHS=1
LEARNING_RATE=1e-5
MAX_HISTORY_LEN=8
MAX_STEPS=3000
SAVE_STEPS=500

# Prompt type
PROMPT_TYPE="seq_rec_new"

# Run name
RUN_NAME="sft_rejection_${DATA_NAME}_seed${seed}_maxrank${MAX_RANK_THRESHOLD}_minimp${MIN_RANK_IMPROVEMENT}_steps${MAX_STEPS}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
FINAL_CHECKPOINT_DIR="${CHECKPOINT_DIR}/checkpoint-${MAX_STEPS}"

# Device tracker
TRACKER="python3 utils/device_tracker.py"
device=5

$TRACKER allocate $device "$RUN_NAME"

echo ""
echo "================================="
echo "🚀 SFT Training with Rejection Sampling"
echo "================================="
echo "Dataset: $DATA_NAME"
echo "Seed: $SEED"
echo "Rejection CSV: $REJECTION_SAMPLING_CSV"
echo "Max Rank Threshold: $MAX_RANK_THRESHOLD"
echo "Min Improvement: $MIN_RANK_IMPROVEMENT"
echo "================================="
echo ""

# Run training
CUDA_VISIBLE_DEVICES=$device python3 src/train_sft.py \
    --data_name $DATA_NAME \
    --model_name $MODEL_NAME \
    --seed $SEED \
    --run_name $RUN_NAME \
    --prompt_type $PROMPT_TYPE \
    --use_brand \
    --use_category \
    --emphasize_recent_item \
    --target_type rejection_sampling \
    --rejection_sampling_csv $REJECTION_SAMPLING_CSV \
    --max_rank_threshold $MAX_RANK_THRESHOLD \
    --min_rank_improvement $MIN_RANK_IMPROVEMENT \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
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

$TRACKER free $device

echo ""
echo "✅ Training complete!"
echo "Model saved to: $FINAL_CHECKPOINT_DIR"
