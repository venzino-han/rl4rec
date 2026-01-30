#!/bin/bash

# Rejection Sampling Example Script
# This script performs rejection sampling to find better queries

DATA_NAME="beauty"
SPLIT="test"
MODEL_NAME="google/gemma-3-1b-it"
CHECKPOINT_DIR="checkpoints/sft/checkpoint-5000"  # Optional: LoRA checkpoint

# Baseline CSV (temp=0.01 results)
# Note: Make sure this CSV has the following columns: user_id, rank (global_rank)
BASELINE_CSV="results/beauty_baseline_temp0.01_test_eval_20260130_000000.csv"

# Generation settings
NUM_SAMPLES=5  # Generate K=5 queries per user
TEMPERATURE=0.6
TOP_P=0.9
MAX_NEW_TOKENS=128

# Selection settings
MIN_IMPROVEMENT=10  # Only select samples with rank improvement >= 10
SELECTION_STRATEGY="best_improvement"  # or "best_rank" or "threshold"

# Output
RUN_NAME="rejection_sampling_k${NUM_SAMPLES}_temp${TEMPERATURE}_minimp${MIN_IMPROVEMENT}"
OUTPUT_DIR="results/rejection_sampling"

python rejection_sampling.py \
    --run_name ${RUN_NAME} \
    --data_name ${DATA_NAME} \
    --split ${SPLIT} \
    --model_name ${MODEL_NAME} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --enable_lora \
    --baseline_csv ${BASELINE_CSV} \
    --num_samples ${NUM_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --min_improvement ${MIN_IMPROVEMENT} \
    --selection_strategy ${SELECTION_STRATEGY} \
    --output_dir ${OUTPUT_DIR} \
    --save_all_results \
    --clean_master_logs \
    --prompt_type seq_rec \
    --use_brand \
    --use_category \
    --use_date \
    --use_last_item \
    --max_history_len 8 \
    --history_text_max_length 128 \
    --emb_model_name mixedbread-ai/mxbai-embed-large-v1 \
    --emb_type item_meta_only \
    --eval_emb_max_length 512 \
    --eval_emb_batch_size 512 \
    --gpu_memory_utilization 0.85 \
    --eval_emb_gpu_memory_utilization 0.85 \
    --gen_batch_size 32 \
    --seed 42

echo "✓ Rejection sampling complete!"
echo ""
echo "📁 Output files:"
echo "  - Selected samples: ${OUTPUT_DIR}/${RUN_NAME}_${DATA_NAME}_${SPLIT}_selected.csv"
echo "  - All samples: ${OUTPUT_DIR}/${RUN_NAME}_${DATA_NAME}_${SPLIT}_all_samples.csv"
