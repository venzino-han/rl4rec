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
TRACKER="python3 utils/device_tracker.py"
# 학습 실행
for dataset_name in ${dataset_names[@]}; do
for target_emb_coef in 0.1 0.05 ; do
for loss_type in dr_grpo ; do
echo "Training ${dataset_name}..."
for lr in 2e-6 1e-6; do
for temp in 0.6 ; do
RUN_NAME="${loss_type}_${dataset_name}_seq_rec_reasoning_emb_${target_emb_coef}_temp${temp}_1000_lr${lr}"
$TRACKER allocate $device "$RUN_NAME"
CHECKPOINT_DIR="checkpoints/$RUN_NAME"
FINAL_CHECKPOINT_DIR="$CHECKPOINT_DIR/checkpoint-$max_steps"

CUDA_VISIBLE_DEVICES=$device python3 src/grpo_train.py \
    --run_name $RUN_NAME \
    --model_name "google/gemma-3-1b-it" \
    --data_name $dataset_name \
    --reward_type "ndcg" \
    --k 100 \
    --loss_type $loss_type \
    --prompt_type $PROMPT_TYPE \
    --emphasize_recent_item \
    --use_brand \
    --use_category \
    --use_local_embedding \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --target_emb_reward \
    --target_emb_file "user_preference_reasoning_1024_user_preference_${dataset_name}_gemma-3-12b-it_mxbai-embed-large-v1_train_pred_emb.pt" \
    --target_emb_coef $target_emb_coef \
    --batch_size 32 \
    --max_new_tokens 128 \
    --learning_rate $lr \
    --num_epochs 1 \
    --max_steps $max_steps \
    --checkpoint_dir $CHECKPOINT_DIR \
    --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
    --log_interval 10 \
    --eval_interval 5000 \
    --save_interval 1000 \
    --num_negs 99 \
    --device "cuda" \
    "$@"


CUDA_VISIBLE_DEVICES=$device python3 src/grpo_eval.py \
    --run_name $RUN_NAME \
    --model_name "google/gemma-3-1b-it" \
    --data_name $dataset_name \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --use_local_embedding \
    --prompt_type $PROMPT_TYPE \
    --emphasize_recent_item \
    --use_brand \
    --use_category \
    --max_new_tokens 512 \
    --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
    --device "cuda" \
    "$@"

$TRACKER free $device
done
done
done
done
done

echo "✅ Training completed!"

