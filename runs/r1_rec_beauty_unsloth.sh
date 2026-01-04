#!/bin/bash
# GRPO 학습 실행 스크립트 (Unsloth Version)


max_steps=1000
dataset_name="beauty"
# 학습 실행 (Unsloth)
CUDA_VISIBLE_DEVICES=2 python3 src/grpo_train_unsloth.py \
    --run_name "r1_rec_${dataset_name}_unsloth" \
    --model_name unsloth/gemma-3-1b-it \
    --data_name $dataset_name \
    --sequential_file "data/$dataset_name/sequential_data.txt" \
    --reward_type "ndcg" \
    --k 1000 \
    --batch_size 32 \
    --num_sample_generations 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --num_epochs 1 \
    --max_steps $max_steps \
    --max_length 2048 \
    --max_new_tokens 128 \
    --use_brand \
    --checkpoint_dir "checkpoints/r1_rec_${dataset_name}_unsloth" \
    --final_checkpoint_dir "checkpoints/r1_rec_${dataset_name}_unsloth/checkpoint-$max_steps" \
    --log_interval 100 \
    --eval_interval 5000 \
    --save_interval 1000 \
    --num_negs 0 \
    --device "cuda" \
    "$@"

echo "✅ Training completed (Unsloth)!"


    # --use_category \
    # --load_in_4bit \
    # --lora_r 16 \
    # --lora_alpha 16 \
    # --lora_dropout 0.0 \
    # --use_8bit_adam \

