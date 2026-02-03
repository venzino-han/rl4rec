#!/bin/bash

# Zeroshot evaluation with multiple rollouts using trigger items
# This script evaluates the model without training, generating k rollouts per prompt

# Dataset name
DATA_NAME="beauty"  # Options: beauty, toys, sports, yelp

# Model settings
MODEL_NAME="google/gemma-3-1b-it"
CHECKPOINT_DIR=""  # Leave empty for base model, or specify checkpoint path

# Rollout settings
NUM_ROLLOUTS=3  # Number of rollouts per prompt
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=-1  # -1 to disable
MAX_TOKENS=128

# Trigger items settings
USE_TRIGGER_ITEMS="--use_trigger_items"
TRIGGER_ITEMS_DIR="sasrec_results/trigger_items_from_sequential"
TRIGGER_EMPHASIS_TEXT="This item was particularly influential in shaping the user's preferences."

# Prompt settings
PROMPT_TYPE="seq_rec"
MAX_HISTORY_LEN=8
HISTORY_TEXT_MAX_LENGTH=128

# Embedding model for evaluation
EMB_MODEL_NAME="mixedbread-ai/mxbai-embed-large-v1"
EMB_TYPE="item_meta_only"
EVAL_EMB_MAX_LENGTH=512
EVAL_EMB_BATCH_SIZE=512

# Evaluation settings
EVAL_SAMPLES=100000
GPU_MEMORY_UTILIZATION=0.95
EVAL_EMB_GPU_MEMORY_UTILIZATION=0.95

# Output settings
OUTPUT_DIR="results"

# Other settings
SEED=42

echo "============================================================"
echo "Zeroshot Evaluation with Multiple Rollouts"
echo "============================================================"
echo "Dataset: ${DATA_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Checkpoint: ${CHECKPOINT_DIR:-'Base Model'}"
echo "Number of Rollouts: ${NUM_ROLLOUTS}"
echo "Temperature: ${TEMPERATURE}"
echo "Trigger Items Dir: ${TRIGGER_ITEMS_DIR}"
echo "Prompt Type: ${PROMPT_TYPE}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "============================================================"

cd /home/kirc/workspace/rl4rec

# Build checkpoint argument
CHECKPOINT_ARG=""
if [ -n "${CHECKPOINT_DIR}" ]; then
    CHECKPOINT_ARG="--checkpoint_dir ${CHECKPOINT_DIR}"
fi

python3 src/zeroshot_eval.py \
    --run_name "zeroshot_rollout_${DATA_NAME}" \
    --data_name "${DATA_NAME}" \
    --model_name "${MODEL_NAME}" \
    ${CHECKPOINT_ARG} \
    \
    --num_rollouts "${NUM_ROLLOUTS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --max_tokens "${MAX_TOKENS}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    \
    --seed "${SEED}" \
    \
    --prompt_type "${PROMPT_TYPE}" \
    --max_history_len "${MAX_HISTORY_LEN}" \
    --history_text_max_length "${HISTORY_TEXT_MAX_LENGTH}" \
    --use_brand \
    --use_category \
    --use_date \
    --use_last_item \
    \
    ${USE_TRIGGER_ITEMS} \
    --trigger_items_dir "${TRIGGER_ITEMS_DIR}" \
    --trigger_emphasis_text "${TRIGGER_EMPHASIS_TEXT}" \
    \
    --emb_model_name "${EMB_MODEL_NAME}" \
    --emb_type "${EMB_TYPE}" \
    --eval_emb_max_length "${EVAL_EMB_MAX_LENGTH}" \
    --eval_emb_batch_size "${EVAL_EMB_BATCH_SIZE}" \
    --eval_samples "${EVAL_SAMPLES}" \
    --eval_emb_gpu_memory_utilization "${EVAL_EMB_GPU_MEMORY_UTILIZATION}" \
    \
    --output_dir "${OUTPUT_DIR}"

echo "============================================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
