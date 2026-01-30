#!/bin/bash

# Simple Rejection Sampling Script
# Minimal configuration for quick testing

DATA_NAME="beauty"
BASELINE_CSV="results/beauty_rec_r1_seed22_k1000_128_steps1000_temp0.6_lr1e-6_test_eval_20260129_154251.csv"

python rejection_sampling.py \
    --run_name rs_simple_k3 \
    --data_name ${DATA_NAME} \
    --split test \
    --model_name google/gemma-3-1b-it \
    --baseline_csv ${BASELINE_CSV} \
    --num_samples 3 \
    --temperature 0.6 \
    --max_new_tokens 128 \
    --min_improvement 5 \
    --selection_strategy best_improvement \
    --output_dir results/rejection_sampling \
    --save_all_results \
    --clean_master_logs \
    --prompt_type seq_rec \
    --use_brand \
    --use_category \
    --use_date \
    --use_last_item \
    --max_history_len 8 \
    --gen_batch_size 16 \
    --eval_samples 1000 \
    --seed 42

echo ""
echo "✅ Done! Check results in: results/rejection_sampling/"
