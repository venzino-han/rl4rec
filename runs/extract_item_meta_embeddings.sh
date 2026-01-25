#!/bin/bash

# Script to extract item metadata embeddings for all datasets
# This script extracts embeddings using only item metadata (title, brand, category, description)

# List of dataset names
datasets=(beauty sports toys yelp)

# Embedding model configuration
embedding_model_name="mixedbread-ai/mxbai-embed-large-v1"
device="cuda:0"
token_limit=1024
description_word_limit=128
batch_size=1024

echo "Starting item metadata embedding extraction for all datasets..."
echo "Embedding model: $embedding_model_name"
echo "Token limit: $token_limit"
echo "Description word limit: $description_word_limit"
echo "----------------------------------------"

# Loop through each dataset
for dataset in "${datasets[@]}"
do
  echo ""
  echo "Processing dataset: $dataset"
  
  CUDA_VISIBLE_DEVICES=0 python3 src/extract_item_emb.py \
    --data_name $dataset \
    --target item_meta_only \
    --embedding_model_name $embedding_model_name \
    --device $device \
    --token_limit $token_limit \
    --description_word_limit $description_word_limit \
    --batch_size $batch_size \
    --use_sentence_transformers
done
