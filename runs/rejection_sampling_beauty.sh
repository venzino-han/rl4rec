#!/bin/bash

# Rejection Sampling for Beauty Dataset Only
# Single dataset, multiple seeds

# Model configuration
MODEL_NAME="google/gemma-3-1b-it"

# Dataset
DATA_NAME="beauty"

# Generation settings
NUM_SAMPLES=5
TEMPERATURE=0.6
TOP_P=0.9
MAX_NEW_TOKENS=128
GEN_BATCH_SIZE=32

# Selection settings
MIN_IMPROVEMENT=10
SELECTION_STRATEGY="best_improvement"

# Evaluation settings
MAX_HISTORY_LEN=8
PROMPT_TYPE="seq_rec_new"

# Device tracker
TRACKER="python3 utils/device_tracker.py"

# GPU device
device=5

# Loop through seeds
for seed in 22 42 62; do

    echo ""
    echo "================================="
    echo "🎲 Processing seed: $seed"
    echo "================================="
    
    # Set checkpoint path
    CHECKPOINT_DIR="checkpoints/bigrec_sft_${DATA_NAME}_seed${seed}_max_steps5000/checkpoint-5000"
    
    # Manually specify baseline CSV paths for each seed
    # Update these paths to match your actual baseline CSV files
    case $seed in
        22)
            BASELINE_CSV="results/beauty_rec_r1_seed22_k1000_128_steps1000_temp0.6_lr1e-6_test_eval_20260129_154251.csv"
            ;;
        42)
            BASELINE_CSV="results/beauty_rec_r1_seed42_k1000_128_steps1000_temp0.6_lr1e-6_test_eval_20260129_215635.csv"
            ;;
        62)
            BASELINE_CSV="results/beauty_rec_r1_seed62_k1000_128_steps1000_temp0.6_lr1e-6_test_eval_20260130_053115.csv"
            ;;
        *)
            echo "❌ Unknown seed: $seed"
            continue
            ;;
    esac
    
    # Check if baseline CSV exists
    if [ ! -f "$BASELINE_CSV" ]; then
        echo "⚠️  Warning: Baseline CSV not found: $BASELINE_CSV"
        echo "    Skipping seed $seed..."
        continue
    fi
    
    # Check if checkpoint exists
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "⚠️  Warning: Checkpoint not found: $CHECKPOINT_DIR"
        echo "    Running without LoRA checkpoint..."
        CHECKPOINT_DIR=""
        ENABLE_LORA_FLAG=""
    else
        ENABLE_LORA_FLAG="--enable_lora"
    fi
    
    echo "📊 Baseline CSV: $BASELINE_CSV"
    echo "📦 Checkpoint: $CHECKPOINT_DIR"
    
    # Set run name
    RUN_NAME="rejection_sampling_${DATA_NAME}_seed${seed}_k${NUM_SAMPLES}_temp${TEMPERATURE}_minimp${MIN_IMPROVEMENT}"
    OUTPUT_DIR="results/rejection_sampling"
    
    # Allocate device
    $TRACKER allocate $device "$RUN_NAME"
    
    # Run rejection sampling
    CUDA_VISIBLE_DEVICES=$device python3 rejection_sampling.py \
        --run_name $RUN_NAME \
        --data_name $DATA_NAME \
        --split test \
        --model_name $MODEL_NAME \
        --checkpoint_dir $CHECKPOINT_DIR \
        $ENABLE_LORA_FLAG \
        --baseline_csv $BASELINE_CSV \
        --num_samples $NUM_SAMPLES \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_new_tokens $MAX_NEW_TOKENS \
        --min_improvement $MIN_IMPROVEMENT \
        --selection_strategy $SELECTION_STRATEGY \
        --output_dir $OUTPUT_DIR \
        --save_all_results \
        --clean_master_logs \
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
        --gpu_memory_utilization 0.85 \
        --eval_emb_gpu_memory_utilization 0.85 \
        --gen_batch_size $GEN_BATCH_SIZE \
        --seed $seed \
        "$@"
    
    # Free device
    $TRACKER free $device
    
    echo ""
    echo "✅ Completed: seed $seed"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
done

echo ""
echo "🎉 All seeds completed for $DATA_NAME dataset!"
echo ""
echo "📁 Results saved to: results/rejection_sampling/"
echo ""
echo "📊 Generated files:"
ls -lh results/rejection_sampling/${DATA_NAME}*_selected.csv 2>/dev/null
