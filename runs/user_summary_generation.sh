#!/bin/bash

# List of dataset names
export VLLM_USE_V1=1
# export VLLM_USE_V1=0

datasets=(sports)

datasets=(toys beauty sports yelp)
splits=(train valid test)

model_name="meta-llama/Llama-3.2-3B-Instruct"
model_name="Qwen/Qwen3-4B"
model_name="google/gemma-3-1b-it"

# Loop through each dataset
for dataset in "${datasets[@]}"
do
for split in "${splits[@]}"
do
  echo "Query generation $dataset..."
  CUDA_VISIBLE_DEVICES=2 python3 src/generate_user_preference.py \
  --run_name "zero_shot_seq_rec" \
  --data_name $dataset \
  --split $split \
  --batch_size 16 \
  --model_name $model_name
done
done

  # --dtype float16 \
