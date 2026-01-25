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
max_steps=5000
dataset_name="beauty"
# 학습 실행
CUDA_VISIBLE_DEVICES=1 python3 src/grpo_train.py \
    --run_name "r1_rec_${dataset_name}_pref" \
    --model_name "google/gemma-3-1b-it" \
    --data_name $dataset_name \
    --sequential_file "data/$dataset_name/sequential_data.txt" \
    --reward_type "ndcg" \
    --k 100 \
    --prompt_type "feature_reasoning_rec" \
    --use_local_embedding \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --max_new_tokens 128 \
    --batch_size 64 \
    --num_sample_generations 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --num_epochs 0 \
    --max_steps $max_steps \
    --use_brand \
    --use_category \
    --checkpoint_dir "checkpoints/r1_rec_${dataset_name}_pref" \
    --final_checkpoint_dir "checkpoints/r1_rec_${dataset_name}_pref/checkpoint-$max_steps" \
    --log_interval 100 \
    --eval_interval 5000 \
    --save_interval 500 \
    --num_negs 99 \
    --device "cuda" \
    "$@"

echo "✅ Training completed!"

