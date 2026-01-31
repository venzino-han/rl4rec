#!/bin/bash
# Example training script using History Proxy Threshold Reward

# Beauty dataset example
python src/grpo_train.py \
    --data_name beauty \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir models/beauty_history_proxy \
    --run_name beauty_history_proxy_seed42 \
    \
    --history_proxy_threshold_reward \
    --history_proxy_threshold_coef 1.0 \
    \
    --reward_type ndcg \
    --k 1000 \
    --emb_model_name sentence-transformers/all-MiniLM-L6-v2 \
    --emb_type review_description \
    \
    --max_steps 1000 \
    --per_device_train_batch_size 128 \
    --temperature 0.6 \
    --learning_rate 1e-6 \
    --seed 42 \
    \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --do_eval \
    --report_to wandb

# You can combine with other rewards:
# Example: Base NDCG + Proxy Label + History Proxy Threshold
# python src/grpo_train.py \
#     --data_name beauty \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --output_dir models/beauty_combined \
#     \
#     --reward_type ndcg \
#     --proxy_label_reward \
#     --proxy_label_coef 1.0 \
#     --proxy_k 1000 \
#     \
#     --history_proxy_threshold_reward \
#     --history_proxy_threshold_coef 0.5 \
#     \
#     --k 1000 \
#     --max_steps 1000 \
#     --per_device_train_batch_size 128 \
#     --temperature 0.6 \
#     --learning_rate 1e-6
