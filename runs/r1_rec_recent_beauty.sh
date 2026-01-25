#!/bin/bash
# GRPO 학습 실행 스크립트 (Novelty Reward 포함)
# Novelty = NDCG × popularity_weight (인기 없는 아이템 장려)

max_steps=5000
dataset_name="beauty"

# 학습 실행 (--novelty_reward 플래그 사용)
CUDA_VISIBLE_DEVICES=2 python3 src/grpo_train.py \
    --run_name "r1_rec_${dataset_name}_pref_recent" \
    --model_name "google/gemma-3-1b-it" \
    --data_name $dataset_name \
    --reward_type "ndcg" \
    --k 100 \
    --use_local_embedding \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --emb_batch_size 128 \
    --prompt_type "feature_reasoning_recent_rec" \
    --max_new_tokens 128 \
    --batch_size 32 \
    --num_sample_generations 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --num_epochs 0 \
    --max_steps $max_steps \
    --use_brand \
    --use_category \
    --checkpoint_dir "checkpoints/r1_rec_${dataset_name}_pref_recent" \
    --final_checkpoint_dir "checkpoints/r1_rec_${dataset_name}_pref_recent/checkpoint-$max_steps" \
    --log_interval 500 \
    --eval_interval $max_steps \
    --save_interval 2500 \
    --device "cuda" \
    "$@"

echo "✅ Training completed!"

