#!/bin/bash
# GRPO 학습 실행 스크립트 (Popularity Reward 포함)
# 인기 없는 아이템(long-tail)을 예측하면 추가 보상

set -e

cd "$(dirname "$0")/.."

echo "🚀 Starting GRPO Training with Popularity Reward"
echo "========================================"

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

max_steps=1000
dataset_name="sports"

# 학습 실행 (--popularity_coef로 인기도 역수 가중치 추가)
CUDA_VISIBLE_DEVICES=3 python3 src/grpo_train.py \
    --run_name "r1_rec_${dataset_name}_popularity" \
    --model_name "google/gemma-3-1b-it" \
    --data_name $dataset_name \
    --reward_type "ndcg" \
    --k 100 \
    --use_local_embedding \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type "review_description" \
    --emb_batch_size 128 \
    --popularity_coef 0.2 \
    --max_new_tokens 128 \
    --batch_size 64 \
    --num_sample_generations 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --num_epochs 1 \
    --max_steps $max_steps \
    --use_brand \
    --use_category \
    --checkpoint_dir "checkpoints/r1_rec_${dataset_name}_popularity" \
    --final_checkpoint_dir "checkpoints/r1_rec_${dataset_name}_popularity/checkpoint-$max_steps" \
    --log_interval 100 \
    --eval_interval 5000 \
    --save_interval 500 \
    --device "cuda" \
    "$@"

echo "✅ Training completed!"

