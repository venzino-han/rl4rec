#!/bin/bash

# Script to compare vanilla and target embeddings for all datasets
# Computes cosine similarity and saves top 25% user_ids

# Configuration
DATA_EMB_DIR="data_emb"
OUTPUT_DIR="data_processed/embedding_comparison"

# Create output directory
mkdir -p $OUTPUT_DIR

# Model configuration
MODEL_NAME="gemma-3-12b-it"
EMBED_MODEL="mxbai-embed-large-v1"
CONTEXT_LEN="1024"

# Datasets and splits
DATASETS=("beauty" "sports" "toys" "yelp")
SPLITS=("train" "valid" "test")

echo "========================================"
echo "Comparing Embeddings for All Datasets"
echo "========================================"
echo ""

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    echo "Processing dataset: $dataset"
    echo "----------------------------------------"
    
    for split in "${SPLITS[@]}"; do
        echo "  Split: $split"
        
        # Determine the comparison type based on dataset
        if [ "$dataset" = "yelp" ]; then
            # For yelp: compare user_preference_reasoning vs vanilla_reasoning
            VANILLA_PREFIX="vanilla_reasoning"
            TARGET_PREFIX="user_preference_reasoning"
            COMPARISON_TYPE="user_preference_vs_vanilla"
        else
            # For beauty, sports, toys: compare target_preference_reasoning vs vanilla_reasoning
            VANILLA_PREFIX="vanilla_reasoning"
            TARGET_PREFIX="target_preference_reasoning"
            COMPARISON_TYPE="target_vs_vanilla"
        fi
        
        # Construct file paths
        VANILLA_FILE="${VANILLA_PREFIX}_${CONTEXT_LEN}_user_preference_${dataset}_${MODEL_NAME}_${EMBED_MODEL}_${split}_pred_emb.pt"
        TARGET_FILE="${TARGET_PREFIX}_${CONTEXT_LEN}_user_preference_${dataset}_${MODEL_NAME}_${EMBED_MODEL}_${split}_pred_emb.pt"
        
        VANILLA_PATH="${DATA_EMB_DIR}/${VANILLA_FILE}"
        TARGET_PATH="${DATA_EMB_DIR}/${TARGET_FILE}"
        
        # Check if files exist
        if [ ! -f "$VANILLA_PATH" ]; then
            echo "    ⚠️  Vanilla file not found: $VANILLA_PATH"
            continue
        fi
        
        if [ ! -f "$TARGET_PATH" ]; then
            echo "    ⚠️  Target file not found: $TARGET_PATH"
            continue
        fi
        
        # Output file
        OUTPUT_FILE="${OUTPUT_DIR}/top25_${COMPARISON_TYPE}_${dataset}_${split}.json"
        
        # Run comparison
        echo "    Running comparison..."
        python3 src/compare_embeddings.py \
            --vanilla_emb_path "$VANILLA_PATH" \
            --target_emb_path "$TARGET_PATH" \
            --output_path "$OUTPUT_FILE"
        
        if [ $? -eq 0 ]; then
            echo "    ✓ Saved to: $OUTPUT_FILE"
        else
            echo "    ✗ Failed"
        fi
        echo ""
    done
    
    echo ""
done

echo "========================================"
echo "Comparison Complete!"
echo "========================================"
echo "Results saved in: $OUTPUT_DIR"
echo ""

# Print summary
echo "Summary of generated files:"
ls -lh $OUTPUT_DIR/*.json

