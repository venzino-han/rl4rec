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

max_steps=1000
dataset_names=(beauty toys sports yelp)
device=3
PROMPT_TYPE="seq_rec"

# 학습 실행
for dataset_name in ${dataset_names[@]}; do
    echo "Training ${dataset_name}..."
for temp in 0.1 ; do
for loss_type in dapo dr_grpo; do
    RUN_NAME="${loss_type}_${dataset_name}_pref_128_1000_temp${temp}"
    CHECKPOINT_DIR="checkpoints/$RUN_NAME"
    FINAL_CHECKPOINT_DIR="$CHECKPOINT_DIR/checkpoint-$max_steps"

    CUDA_VISIBLE_DEVICES=$device python3 src/grpo_train.py \
        --run_name $RUN_NAME \
        --model_name "google/gemma-3-1b-it" \
        --data_name $dataset_name \
        --reward_type "ndcg" \
        --k 1000 \
        --loss_type $loss_type \
        --use_local_embedding \
        --prompt_type $PROMPT_TYPE \
        --emphasize_recent_item \
        --use_brand \
        --use_category \
        --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type item_preference_1024_gemma-3-4b-it \
        --max_new_tokens 128 \
        --num_epochs 1 \
        --batch_size 32 \
        --num_sample_generations 4 \
        --learning_rate 5e-6 \
        --train_temperature $temp \
        --max_steps $max_steps \
        --checkpoint_dir $CHECKPOINT_DIR \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --log_interval 20 \
        --eval_interval 5000 \
        --save_interval 1000 \
        --device "cuda" \
        "$@"
        # --use_local_embedding \
    CUDA_VISIBLE_DEVICES=$device python3 src/grpo_eval.py \
        --run_name $RUN_NAME \
        --model_name "google/gemma-3-1b-it" \
        --data_name $dataset_name \
        --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type item_preference_1024_gemma-3-4b-it \
        --prompt_type $PROMPT_TYPE \
        --use_local_embedding \
        --emphasize_recent_item \
        --use_brand \
        --use_category \
        --max_new_tokens 512 \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --device "cuda" \
        "$@"
done
done
done