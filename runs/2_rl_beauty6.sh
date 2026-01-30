#!/bin/bash

max_steps=500

dataset_names=(toys sports yelp)
dataset_names=(sports yelp)
device=6
PROMPT_TYPE="seq_rec_new"

TRACKER="python3 utils/device_tracker.py"

max_new_tokens=128

for seed in 42 22 62; do
for dataset_name in ${dataset_names[@]}; do
    echo "Training ${dataset_name}..."
for temp in 0.6 ; do
for loss_type in dr_grpo; do
    RUN_NAME="${dataset_name}_anchor_meta0.01_proxy1000_0.05_seed${seed}_kd0.001_k1000_${max_new_tokens}_steps${max_steps}_temp${temp}_lr1e-6"
    CHECKPOINT_DIR="checkpoints/$RUN_NAME"
    FINAL_CHECKPOINT_DIR="$CHECKPOINT_DIR/checkpoint-$max_steps"

    $TRACKER allocate $device "$RUN_NAME"

    CUDA_VISIBLE_DEVICES=$device python3 src/grpo_train.py \
        --run_name $RUN_NAME \
        --model_name "google/gemma-3-1b-it" \
        --data_name $dataset_name \
        --reward_type "ndcg" \
        --k 1000 \
        --seed $seed \
        --loss_type $loss_type \
        --importance_sampling_level token \
        --use_local_embedding \
        --prompt_type $PROMPT_TYPE \
        --use_metadata_reward \
        --metadata_base_reward 0.01 \
        --metadata_length_penalty 1.0 \
        --metadata_min_length 8 \
        --history_penalty_weight 0.001 \
        --anchor_reward \
        --anchor_coef 1.0 \
        --anchor_penalty_mode "soft" \
        --use_brand \
        --use_category \
        --emphasize_recent_item \
        --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type item_preference_1024_gemma-3-4b-it \
        --reference_model_kld_coef 0.001 \
        --max_new_tokens $max_new_tokens \
        --num_epochs 1 \
        --batch_size 32 \
        --learning_rate 1e-6 \
        --train_temperature $temp \
        --max_steps $max_steps \
        --checkpoint_dir $CHECKPOINT_DIR \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --save_total_limit 1 \
        --log_interval 10 \
        --eval_interval 5000 \
        --save_interval $max_steps \
        --device "cuda" \
        "$@"
        # --proxy_label_reward \
        # --proxy_k 1000 \
        # --proxy_label_coef 0.05 \
        # --proxy_label_file data_emb/${dataset_name}_proxy_labels_k1000_random_th0.3_item_preference_1024_gemma-3-4b-it_mxbai-embed-large-v1.json \

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
        --max_new_tokens $max_new_tokens \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --device "cuda" \
        "$@"

    $TRACKER free $device
done
done
done
done

echo "✅ Training completed!"
