#!/bin/bash

max_steps=5000
dataset_name=beauty
device=3
PROMPT_TYPE="seq_rec_with_sasrec"

echo "Training ${dataset_name}..."

RUN_NAME="r1_rec_${dataset_name}_sasrec"
CHECKPOINT_DIR="checkpoints/$RUN_NAME"
FINAL_CHECKPOINT_DIR="$CHECKPOINT_DIR/checkpoint-$max_steps"

CUDA_VISIBLE_DEVICES=$device python3 src/grpo_train.py \
    --run_name $RUN_NAME \
    --model_name "google/gemma-3-1b-it" \
    --data_name $dataset_name \
    --reward_type "ndcg" \
    --k 100 \
    --prompt_type $PROMPT_TYPE \
    --use_sasrec \
    --sasrec_top_k 5 \
    --use_local_embedding \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --max_new_tokens 128 \
    --num_epochs 1 \
    --max_steps $max_steps \
    --use_brand \
    --use_category \
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
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --use_local_embedding \
    --prompt_type $PROMPT_TYPE \
    --use_sasrec \
    --sasrec_top_k 5 \
    --max_new_tokens 128 \
    --use_brand \
    --use_category \
    --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
    --device "cuda" \
    "$@"

echo "✅ Training completed!"




