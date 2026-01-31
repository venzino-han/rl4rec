#!/bin/bash

max_steps=500
dataset_names=(beauty toys sports yelp)
rank_file_names=(
    "results/zeroshot_seq_rec_beauty_train_train_eval_20260120_173119.csv"
    "results/zeroshot_seq_rec_toys_train_train_eval_20260120_161627.csv"
    "results/zeroshot_seq_rec_sports_train_train_eval_20260120_191551.csv"
    "results/zeroshot_seq_rec_yelp_train_train_eval_20260120_211918.csv"
)
device=2
PROMPT_TYPE="seq_rec_new"
PROMPT_TYPE="seq_rec_recent2"
PROMPT_TYPE="seq_rec_anchor"

TRACKER="python3 utils/device_tracker.py"
#!/bin/bash

max_steps=1000
dataset_names=(sports)
device=2
PROMPT_TYPE="seq_rec_anchor"

MAX_NEW_TOKENS=128

TRACKER="python3 utils/device_tracker.py"



for seed in 42 22 62; do
for dataset_name in ${dataset_names[@]}; do
    echo "Training ${dataset_name}..."
for temp in 0.6 ; do
for loss_type in dr_grpo; do
    RUN_NAME="${dataset_name}_${PROMPT_TYPE}_history_proxy_threshold_seed${seed}_kd0.001_k1000_${MAX_NEW_TOKENS}_steps${max_steps}_temp${temp}_lr1e-6"
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
        --use_brand \
        --use_category \
        --emphasize_recent_item \
        --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type item_preference_1024_gemma-3-4b-it \
        --reference_model_kld_coef 0.001 \
        --max_new_tokens $MAX_NEW_TOKENS \
        --num_epochs 1 \
        --batch_size 32 \
        --gradient_accumulation_steps 1 \
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
        --train_vllm_gpu_memory_utilization 0.42 \
        --history_proxy_threshold_reward \
        --history_proxy_threshold_coef 1.0 \
        "$@"
        # --use_metadata_reward \
        # --metadata_base_reward 0.002 \
        # --metadata_length_penalty 0.8 \
        # --metadata_min_length 8 \
        # --history_penalty_weight 0.0002 \
        # --proxy_label_reward \
        # --proxy_k 100 \
        # --proxy_label_coef 1.0 \
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
        --max_new_tokens $MAX_NEW_TOKENS \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --device "cuda" \
        "$@"

    $TRACKER free $device
done
done
done
done

echo "✅ Training completed!"
