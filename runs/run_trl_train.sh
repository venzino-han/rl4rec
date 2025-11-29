#!/bin/bash
# TRL PPO 학습 실행 스크립트

set -e

# 작업 디렉토리로 이동
cd "$(dirname "$0")/.."

echo "🚀 Starting TRL PPO Training for RL4Rec"
echo "========================================"

# Ray 클러스터 확인
echo "📡 Checking Ray cluster..."
ray status || {
    echo "⚠️  Ray cluster not found. Please start retrieval service first:"
    echo "   ./runs/run_retrieval.sh"
    exit 1
}

# Python 경로 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 학습 실행
python src/train_with_trl.py \
    --model_name "gpt2" \
    --dataset_name "beauty" \
    --prompt_file "data_processed/beauty_gemma-3-1b-it_test_user_preference.json" \
    --sequential_file "data/beauty/sequential_data.txt" \
    --reward_type "ndcg" \
    --k 10 \
    --batch_size 16 \
    --mini_batch_size 4 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --max_steps 10000 \
    --checkpoint_dir "checkpoints/trl_ppo" \
    --log_interval 10 \
    --save_interval 500 \
    --device "cuda" \
    "$@"

echo "✅ Training completed!"

