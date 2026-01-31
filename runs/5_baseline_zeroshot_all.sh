#!/bin/bash

max_steps=1000
dataset_names=(beauty toys sports yelp)
dataset_names=(yelp)
device=3
PROMPT_TYPE="seq_rec_new"
PROMPT_TYPE="seq_rec_new_2"
PROMPT_TYPE="seq_rec_anchor"
# PROMPT_TYPE="seq_rec"

EVAL_BATCH_SIZE=64
EPOCHS=0
MAX_HISTORY_LEN=8

TRACKER="python3 utils/device_tracker.py"

for seed in 42; do
for dataset_name in ${dataset_names[@]}; do
    echo "Training ${dataset_name}..."
    RUN_NAME="${dataset_name}_${PROMPT_TYPE}_baseline_zeroshot_seed${seed}"
    CHECKPOINT_DIR="checkpoints/$RUN_NAME"
    $TRACKER allocate $device "$RUN_NAME"

    CUDA_VISIBLE_DEVICES=$device python3 src/train_sft.py \
        --data_name $dataset_name \
        --seed $seed \
        --model_name "google/gemma-3-1b-it" \
        --run_name $RUN_NAME \
        --prompt_type $PROMPT_TYPE \
        --use_brand \
        --use_category \
        --emphasize_recent_item \
        --emb_type item_preference_1024_gemma-3-4b-it \
        --checkpoint_dir $CHECKPOINT_DIR \
        --final_checkpoint_dir "google/gemma-3-1b-it" \
        --num_epochs $EPOCHS \
        --batch_size $EVAL_BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --learning_rate 1e-5 \
        --max_history_len $MAX_HISTORY_LEN \
        --gpu_memory_utilization 0.95 \
        --zeroshot_evaluation \
        --run_evaluation \
        "$@"

    $TRACKER free $device
done
done
