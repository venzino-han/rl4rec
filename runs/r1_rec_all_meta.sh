#!/bin/bash

max_steps=1000
dataset_names=(beauty toys sports yelp)
# dataset_names=(toys sports yelp)
device=4
PROMPT_TYPE="seq_rec"

# Device Tracker 설정
TRACKER="python3 utils/device_tracker.py"
trap '$TRACKER free $device' EXIT

# 시작 시 상태 표시
echo "=========================================="
echo "Starting r1_rec_all_pref runs"
echo "=========================================="
$TRACKER show-simple


for dataset_name in ${dataset_names[@]}; do
    echo "Training ${dataset_name}..."
for lr in 2e-6; do
for temp in 0.6; do
for loss_type in dr_grpo; do
    RUN_NAME="${loss_type}_${dataset_name}_metaonly_sequence_128_1000_temp${temp}_lr${lr}"
    CHECKPOINT_DIR="checkpoints/$RUN_NAME"
    FINAL_CHECKPOINT_DIR="$CHECKPOINT_DIR/checkpoint-$max_steps"

    # Device 할당
    $TRACKER allocate $device "$RUN_NAME"
    
    CUDA_VISIBLE_DEVICES=$device python3 src/grpo_train.py \
        --run_name $RUN_NAME \
        --model_name "google/gemma-3-1b-it" \
        --data_name $dataset_name \
        --reward_type "ndcg" \
        --k 100 \
        --loss_type $loss_type \
        --use_local_embedding \
        --prompt_type $PROMPT_TYPE \
        --emphasize_recent_item \
        --use_brand \
        --use_category \
        --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type item_meta_only \
        --max_new_tokens 128 \
        --num_epochs 1 \
        --batch_size 32 \
        --learning_rate $lr \
        --train_temperature $temp \
        --max_steps $max_steps \
        --checkpoint_dir $CHECKPOINT_DIR \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --log_interval 10 \
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
        --emb_type item_meta_only \
        --prompt_type $PROMPT_TYPE \
        --use_local_embedding \
        --emphasize_recent_item \
        --use_brand \
        --use_category \
        --max_new_tokens 128 \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --device "cuda" \
        "$@"
    
    # Device 해제
    $TRACKER free $device
    
    echo "✅ Completed: $RUN_NAME"
    $TRACKER show-simple
done
done
done
done

echo "=========================================="
echo "✅ Training completed!"
echo "=========================================="
$TRACKER show
