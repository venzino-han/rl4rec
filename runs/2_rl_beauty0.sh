#!/bin/bash

max_steps=500
dataset_names=(beauty toys sports yelp)
rank_file_names=(
    "results/zeroshot_seq_rec_beauty_train_train_eval_20260120_173119.csv"
    "results/zeroshot_seq_rec_toys_train_train_eval_20260120_161627.csv"
    "results/zeroshot_seq_rec_sports_train_train_eval_20260120_191551.csv"
    "results/zeroshot_seq_rec_yelp_train_train_eval_20260120_211918.csv"
)
device=0
PROMPT_TYPE="seq_rec_new"
PROMPT_TYPE="seq_rec_recent2"

TRACKER="python3 utils/device_tracker.py"

for seed in 22 42 62; do
for i in 0 1 2 3; do
    dataset_name=${dataset_names[$i]}
    rank_file_name=${rank_file_names[$i]}
    echo "Training ${dataset_name}..."
for temp in 0.6 ; do
for loss_type in dr_grpo; do
    RUN_NAME="${dataset_name}_${PROMPT_TYPE}_rank_seed${seed}_k1000_128_steps${max_steps}_temp${temp}_lr1e-6"
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
        --rank_min 0 \
        --rank_max 1000 \
        --filter_train_csv $rank_file_name \
        --use_brand \
        --use_category \
        --emphasize_recent_item \
        --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type item_preference_1024_gemma-3-4b-it \
        --reference_model_kld_coef 0.001 \
        --max_new_tokens 128 \
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
done

echo "✅ Training completed!"
