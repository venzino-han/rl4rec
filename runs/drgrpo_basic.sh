#!/bin/bash

max_steps=1000
dataset_names=(beauty toys sports yelp)
device=1
PROMPT_TYPE="seq_rec"

# 학습 실행
for dataset_name in ${dataset_names[@]}; do
    echo "Training ${dataset_name}..."

TRACKER="python3 utils/device_tracker.py"

for temp in 0.1; do
    RUN_NAME="dr_grpo_${dataset_name}_baseline_sequence_128_1000_temp${temp}"
    CHECKPOINT_DIR="checkpoints/$RUN_NAME"
    FINAL_CHECKPOINT_DIR="$CHECKPOINT_DIR/checkpoint-$max_steps"

    $TRACKER allocate $device "$RUN_NAME"

    CUDA_VISIBLE_DEVICES=$device python3 src/grpo_train.py \
        --run_name $RUN_NAME \
        --model_name "google/gemma-3-1b-it" \
        --data_name $dataset_name \
        --reward_type "ndcg" \
        --k 100 \
        --loss_type "dr_grpo" \
        --importance_sampling_level sequence \
        --prompt_type $PROMPT_TYPE \
        --emphasize_recent_item \
        --use_brand \
        --use_category \
        --use_local_embedding \
        --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type item_preference_1024_gemma-3-4b-it \
        --max_new_tokens 128 \
        --num_epochs 1 \
        --batch_size 32 \
        --num_sample_generations 4 \
        --train_temperature $temp \
        --learning_rate 5e-6 \
        --max_steps $max_steps \
        --checkpoint_dir $CHECKPOINT_DIR \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --log_interval 20 \
        --eval_interval 5000 \
        --save_interval 1000 \
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
        --max_new_tokens 512 \
        --emphasize_recent_item \
        --use_brand \
        --use_category \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --device "cuda" \
        "$@"

    $TRACKER free $device
done
done

echo "✅ Training completed!"
