#!/bin/bash

# RL Training 실행 스크립트
# RetrievalService가 이미 실행 중이어야 함

python3 src/rl_train.py \
    --retrieval_service_name "RetrievalService" \
    --namespace "rl4rec" \
    --dataset_name "beauty" \
    --policy_model "gpt2" \
    --num_steps 1000 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --reward_type "max" \
    --checkpoint_dir "checkpoints/beauty_rl" \
    --log_interval 10 \
    --save_interval 100 \
    --ray_address "auto"

