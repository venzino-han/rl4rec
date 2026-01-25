#!/bin/bash

# Retrieval Service 실행 스크립트 - Detached 모드
# 스크립트가 종료되어도 Actor가 계속 실행됨

python3 src/retrieval_service.py \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --datasets beauty sports toys yelp \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --actor_name "RetrievalService" \
    --namespace "rl4rec" \
    --gpu_id 0 \
    --num_gpus 1.0 \
    --ray_address "auto"

