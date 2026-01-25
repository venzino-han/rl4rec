#!/bin/bash

max_steps=1000
dataset_names=(beauty toys sports yelp)
device=3
PROMPT_TYPE="seq_rec"

# 학습 실행
for dataset_name in ${dataset_names[@]}; do
    echo "Training ${dataset_name}..."
for lr in 5e-6 2e-6 1e-6; do

    RUN_NAME="r1_rec_${dataset_name}_pref_128_1000_temp0.1_60d_lr${lr}"
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
        --use_brand \
        --use_category \
        --days_filter 60 \
        --use_local_embedding \
        --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type item_preference_1024_gemma-3-4b-it \
        --max_new_tokens 128 \
        --num_epochs 1 \
        --batch_size 32 \
        --num_sample_generations 4 \
        --train_temperature 0.1 \
        --learning_rate $lr \
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
        --days_filter 60 \
        --use_local_embedding \
        --prompt_type $PROMPT_TYPE \
        --max_new_tokens 512 \
        --emphasize_recent_item \
        --use_brand \
        --use_category \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --device "cuda" \
        "$@"
done
done

echo "✅ Training completed!"
