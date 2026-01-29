#!/bin/bash

max_steps=1000
dataset_names=(beauty toys sports yelp)
device=6
PROMPT_TYPE="seq_rec_new"

proxy_label_coef=0.5

TRACKER="python3 utils/device_tracker.py"

for dataset_name in ${dataset_names[@]}; do
    echo "Training ${dataset_name}..."
for temp in 0.6 ; do
for loss_type in dr_grpo; do
    RUN_NAME="${dataset_name}_${PROMPT_TYPE}_proxy_label_30_${proxy_label_coef}_seed62_${loss_type}_token_pref_k1000_128_1000_temp${temp}_lr2e-6"
    CHECKPOINT_DIR="checkpoints/$RUN_NAME"
    FINAL_CHECKPOINT_DIR="$CHECKPOINT_DIR/checkpoint-$max_steps"

    $TRACKER allocate $device "$RUN_NAME"

    CUDA_VISIBLE_DEVICES=$device python3 src/grpo_train.py \
        --run_name $RUN_NAME \
        --model_name "google/gemma-3-1b-it" \
        --data_name $dataset_name \
        --reward_type "ndcg" \
        --k 1000 \
        --seed 62 \
        --loss_type $loss_type \
        --use_local_embedding \
        --prompt_type $PROMPT_TYPE \
        --use_brand \
        --use_category \
        --emphasize_recent_item \
        --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type item_preference_1024_gemma-3-4b-it \
        --proxy_label_reward \
        --proxy_k 30 \
        --proxy_label_coef $proxy_label_coef \
        --max_new_tokens 128 \
        --num_epochs 1 \
        --batch_size 32 \
        --learning_rate 2e-6 \
        --train_temperature $temp \
        --max_steps $max_steps \
        --checkpoint_dir $CHECKPOINT_DIR \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --save_total_limit 1 \
        --log_interval 10 \
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
        --prompt_type $PROMPT_TYPE \
        --use_local_embedding \
        --emphasize_recent_item \
        --use_brand \
        --use_category \
        --max_new_tokens 128 \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --device "cuda" \
        "$@"

    $TRACKER free $device
done
done
done

echo "✅ Training completed!"
