#!/bin/bash

# Rejection Sampling for All Datasets
# This script performs rejection sampling across multiple datasets and seeds

# Model configuration
MODEL_NAME="google/gemma-3-1b-it"
CHECKPOINT_DIR="google/gemma-3-1b-it"

# Generation settings
NUM_SAMPLES=8
TEMPERATURE=1.0
TOP_P=0.9
MAX_NEW_TOKENS=256
batch_size=64

# Selection settings
MIN_IMPROVEMENT=10
SELECTION_STRATEGY="best_improvement"

# Evaluation settings
MAX_HISTORY_LEN=8
PROMPT_TYPE="seq_rec"

# Dataset configuration
data_names=(
    beauty
    toys
    sports
    yelp
)

# Device tracker
TRACKER="python3 utils/device_tracker.py"

# GPU device
device=2
seed=42

baseline_csv_names=(
    "results/zeroshot_seq_rec_beauty_train_train_eval_20260120_173119.csv"
    "results/zeroshot_seq_rec_toys_train_train_eval_20260120_161627.csv"
    "results/zeroshot_seq_rec_sports_train_train_eval_20260120_191551.csv"
    "results/zeroshot_seq_rec_yelp_train_train_eval_20260120_211918.csv"
)

# Loop through seeds and datasets
for data_name in "${data_names[@]}"; do
    i=0
    # Set checkpoint and baseline paths based on seed and data_name
    # Adjust these paths to match your actual checkpoint and baseline CSV files
    # CHECKPOINT_DIR="checkpoints/bigrec_sft_${data_name}_seed${seed}_max_steps5000/checkpoint-5000"
    
    # Find the most recent baseline CSV file
    # You may need to adjust the pattern to match your baseline CSV naming
    BASELINE_CSV=${baseline_csv_names[$i]}
    i=$((i+1))
    
    # Skip if baseline CSV not found
    if [ -z "$BASELINE_CSV" ]; then
        echo "⚠️  Warning: Baseline CSV not found for ${data_name} seed ${seed}, skipping..."
        continue
    fi
    
    echo "📊 Found baseline CSV: $BASELINE_CSV"
    
    # Set run name
    RUN_NAME="rejection_sampling_${data_name}_seed${seed}_k${NUM_SAMPLES}_temp${TEMPERATURE}_minimp${MIN_IMPROVEMENT}"
    OUTPUT_DIR="results/rejection_sampling"
    
    # Allocate device
    $TRACKER allocate $device "$RUN_NAME"
    
    echo ""
    echo "================================="
    echo "🎲 Starting Rejection Sampling"
    echo "Dataset: $data_name"
    echo "Seed: $seed"
    echo "Baseline: $BASELINE_CSV"
    echo "Checkpoint: $CHECKPOINT_DIR"
    echo "================================="
    echo ""
    
    # Run rejection sampling
    CUDA_VISIBLE_DEVICES=$device python3 src/rejection_sampling.py \
        --run_name $RUN_NAME \
        --data_name $data_name \
        --split train \
        --model_name $MODEL_NAME \
        --checkpoint_dir $CHECKPOINT_DIR \
        --baseline_csv $BASELINE_CSV \
        --num_samples $NUM_SAMPLES \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_new_tokens $MAX_NEW_TOKENS \
        --min_improvement $MIN_IMPROVEMENT \
        --selection_strategy $SELECTION_STRATEGY \
        --output_dir $OUTPUT_DIR \
        --save_all_results \
        --prompt_type $PROMPT_TYPE \
        --use_brand \
        --use_category \
        --emphasize_recent_item \
        --max_history_len $MAX_HISTORY_LEN \
        --history_text_max_length 128 \
        --emb_model_name mixedbread-ai/mxbai-embed-large-v1 \
        --emb_type item_preference_1024_gemma-3-4b-it \
        --eval_emb_max_length 512 \
        --eval_emb_batch_size 512 \
        --batch_size $batch_size \
        --seed $seed \
        "$@"
    
    # Free device
    $TRACKER free $device
    
    echo ""
    echo "✅ Completed: $RUN_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
done

echo ""
echo "🎉 All rejection sampling jobs completed!"
echo ""
echo "📁 Results saved to: results/rejection_sampling/"
echo ""
echo "📊 Summary of generated files:"
ls -lh results/rejection_sampling/*_selected.csv 2>/dev/null | tail -10
