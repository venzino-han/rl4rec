#!/bin/bash

# List of dataset names
export VLLM_USE_V1=1
# export VLLM_USE_V1=0

datasets=(beauty yelp)
splits=(train valid test)

model_name="meta-llama/Llama-3.2-3B-Instruct"
model_name="Qwen/Qwen3-4B"
model_name="google/gemma-3-12b-it"

# Loop through each dataset
for dataset in "${datasets[@]}"
do
for split in "${splits[@]}"
do
  echo "Query generation $dataset..."
  CUDA_VISIBLE_DEVICES=2 python3 src/generate_user_preference.py \
  --run_name "zero_shot_reasoning" \
  --data_name $dataset \
  --split $split \
  --batch_size 16 \
  --prompt_type reasoning \
  --gpu_memory_utilization 0.92 \
  --model_name $model_name
done
done

  # --dtype float16 \
