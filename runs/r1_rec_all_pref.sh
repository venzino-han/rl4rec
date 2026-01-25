#!/bin/bash

max_steps=1000
dataset_names=(beauty toys sports yelp)
device=1
PROMPT_TYPE="seq_rec"


for dataset_name in ${dataset_names[@]}; do
    echo "Training ${dataset_name}..."
for temp in 0.1 0.3 0.6; do
for loss_type in grpo dapo dr_grpo; do
    RUN_NAME="${loss_type}_${dataset_name}_metaonly_sequence_128_1000_temp${temp}"
    CHECKPOINT_DIR="checkpoints/$RUN_NAME"
    FINAL_CHECKPOINT_DIR="$CHECKPOINT_DIR/checkpoint-$max_steps"

    CUDA_VISIBLE_DEVICES=$device python3 src/grpo_train.py \
        --run_name $RUN_NAME \
        --model_name "google/gemma-3-1b-it" \
        --data_name $dataset_name \
        --reward_type "ndcg" \
        --k 100 \
        --loss_type $loss_type \
        --use_local_embedding \
        --importance_sampling_level sequence \
        --prompt_type $PROMPT_TYPE \
        --emphasize_recent_item \
        --use_brand \
        --use_category \
        --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type item_meta_only \
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
done
done
done

echo "✅ Training completed!"
