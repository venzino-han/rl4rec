#!/bin/bash
# GRPO 학습 실행 스크립트

# set -e

# # 작업 디렉토리로 이동
# cd "$(dirname "$0")/.."

# echo "🚀 Starting GRPO Training for RL4Rec"
# echo "========================================"

# # Ray 클러스터 확인
# echo "📡 Checking Ray cluster..."
# ray status || {
#     echo "⚠️  Ray cluster not found. Please start retrieval service first:"
#     echo "   ./runs/run_retrieval.sh"
#     exit 1
# }

# # Python 경로 설정
# export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 학습 실행
CUDA_VISIBLE_DEVICES=7 python3 src/grpo_train.py \
    --policy_model "google/gemma-3-1b-it" \
    --data_name "beauty" \
    --sequential_file "data/beauty/sequential_data.txt" \
    --reward_type "ndcg" \
    --k 1000 \
    --batch_size 32 \
    --num_sample_generations 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --num_epochs 1 \
    --max_steps 3000 \
    --max_length 512 \
    --use_brand \
    --use_category \
    --checkpoint_dir "checkpoints/grpo" \
    --log_interval 10 \
    --eval_interval 100 \
    --save_interval 500 \
    --num_negs 0 \
    --device "cuda" \
    --normalize_rewards \
    "$@"

echo "✅ Training completed!"

