#!/bin/bash

# RL Training 실행 스크립트 (Dummy 모드 - 빠른 테스트용)
# RetrievalService가 이미 실행 중이어야 함

python3 src/rl_train.py \
    --retrieval_service_name "RetrievalService" \
    --namespace "rl4rec" \
    --dataset_name "beauty" \
    --use_dummy \
    --num_steps 100 \
    --batch_size 8 \
    --reward_type "max" \
    --checkpoint_dir "checkpoints/test_dummy" \
    --log_interval 5 \
    --save_interval 50 \
    --ray_address "auto"

