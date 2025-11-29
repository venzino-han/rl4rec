#!/bin/bash
# GRPO 빠른 테스트 스크립트 (Dummy 데이터 사용)

set -e

cd "$(dirname "$0")/.."

echo "🧪 Starting GRPO Quick Test with Dummy Data"
echo "============================================"

# Ray 클러스터 확인
echo "📡 Checking Ray cluster..."
ray status || {
    echo "⚠️  Ray cluster not found. Please start retrieval service first:"
    echo "   ./runs/run_retrieval.sh"
    exit 1
}

# Python 경로 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 빠른 테스트 실행
python src/grpo_train.py \
    --policy_model "gpt2" \
    --dataset_name "beauty" \
    --data_dir "data" \
    --sequential_file "data/beauty/sequential_data.txt" \
    --use_dummy \
    --dummy_size 100 \
    --reward_type "ndcg" \
    --k 10 \
    --batch_size 4 \
    --num_sample_generations 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --max_steps 50 \
    --max_length 256 \
    --use_brand \
    --use_category \
    --max_history_len 3 \
    --checkpoint_dir "checkpoints/grpo_test" \
    --log_interval 5 \
    --eval_interval 25 \
    --save_interval 25 \
    --device "cuda" \
    --normalize_rewards \
    "$@"

echo "✅ Test completed!"

