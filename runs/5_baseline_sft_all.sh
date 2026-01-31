#!/bin/bash

# Script to train SFT model with filtered top 25% users

# Dataset configuration

# Model configuration
MODEL_NAME="google/gemma-3-1b-it"

# Training settings
BATCH_SIZE=4
EVAL_BATCH_SIZE=32
EPOCHS=1
LEARNING_RATE=1e-5
MAX_HISTORY_LEN=8
MAX_STEPS=3000
# MAX_STEPS=5000


data_names=(
    # beauty
    # toys
    # sports
    yelp
)

TRACKER="python3 utils/device_tracker.py"

device=6
PROMPT_TYPE="seq_rec_anchor"
for seed in 42 ; do
for data_name in "${data_names[@]}"; do
RUN_NAME="bigrec_sft_${data_name}_seed${seed}_max_steps${MAX_STEPS}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
FINAL_CHECKPOINT_DIR="checkpoints/${RUN_NAME}/checkpoint-${MAX_STEPS}"
$TRACKER allocate $device "$RUN_NAME"

# 학습 실행
CUDA_VISIBLE_DEVICES=$device python3 src/train_sft.py \
    --data_name $data_name \
    --model_name $MODEL_NAME \
    --seed $seed \
    --run_name $RUN_NAME \
    --prompt_type $PROMPT_TYPE \
    --use_brand \
    --use_category \
    --emphasize_recent_item \
    --target_type item_metadata \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type item_preference_1024_gemma-3-4b-it \
    --max_steps $MAX_STEPS \
    --checkpoint_dir $CHECKPOINT_DIR \
    --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_history_len $MAX_HISTORY_LEN \
    --gpu_memory_utilization 0.95 \
    --run_evaluation \
    "$@"
    $TRACKER free $device

done
done

    # --target_model_name $TARGET_MODEL_NAME \