#!/bin/bash
# GRPO 학습 실행 스크립트 (Novelty Reward 포함)
# Novelty = NDCG × popularity_weight (인기 없는 아이템 장려)

max_steps=5000
dataset_name="beauty"

# 학습 실행 (--novelty_reward 플래그 사용)
CUDA_VISIBLE_DEVICES=1 python3 src/grpo_train.py \
    --run_name "r1_rec_${dataset_name}_novelty" \
    --model_name "google/gemma-3-1b-it" \
    --data_name $dataset_name \
    --reward_type "ndcg" \
    --k 100 \
    --use_local_embedding \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type "review_description" \
    --emb_batch_size 128 \
    --novelty_reward \
    --max_new_tokens 128 \
    --batch_size 48 \
    --num_sample_generations 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --num_epochs 1 \
    --max_steps $max_steps \
    --use_brand \
    --use_category \
    --checkpoint_dir "checkpoints/r1_rec_${dataset_name}_novelty" \
    --final_checkpoint_dir "checkpoints/r1_rec_${dataset_name}_novelty/checkpoint-$max_steps" \
    --log_interval 100 \
    --eval_interval 5000 \
    --save_interval 500 \
    --device "cuda" \
    "$@"

echo "✅ Training completed!"

