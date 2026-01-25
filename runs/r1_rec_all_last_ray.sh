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
dataset_names=(beauty toys sports yelp)
device=5
PROMPT_TYPE="seq_rec"
# 학습 실행
for dataset_name in ${dataset_names[@]}; do
echo "Training ${dataset_name}..."

RUN_NAME="r1_rec_${dataset_name}_seq_rec_last_ray"
CHECKPOINT_DIR="checkpoints/$RUN_NAME"
FINAL_CHECKPOINT_DIR="$CHECKPOINT_DIR/checkpoint-$max_steps"

CUDA_VISIBLE_DEVICES=$device python3 src/grpo_train.py \
    --run_name $RUN_NAME \
    --model_name "google/gemma-3-1b-it" \
    --data_name $dataset_name \
    --reward_type "ndcg" \
    --k 1000 \
    --prompt_type $PROMPT_TYPE \
    --emphasize_recent_item \
    --max_new_tokens 512 \
    --batch_size 16 \
    --num_sample_generations 4 \
    --learning_rate 1e-6 \
    --train_vllm_gpu_memory_utilization 0.4 \
    --num_epochs 1 \
    --max_steps $max_steps \
    --use_brand \
    --use_category \
    --prepend_last_item \
    --checkpoint_dir $CHECKPOINT_DIR \
    --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
    --log_interval 100 \
    --eval_interval 5000 \
    --save_interval 1000 \
    --num_negs 99 \
    --device "cuda" \
    "$@"


CUDA_VISIBLE_DEVICES=$device python3 src/grpo_eval.py \
    --run_name $RUN_NAME \
    --model_name "google/gemma-3-1b-it" \
    --data_name $dataset_name \
    --prompt_type $PROMPT_TYPE \
    --emphasize_recent_item \
    --max_new_tokens 512 \
    --use_brand \
    --use_category \
    --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
    --device "cuda" \
    "$@"
done

echo "✅ Training completed!"

