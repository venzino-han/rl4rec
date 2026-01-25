#!/bin/bash

# Script to run zeroshot evaluation with SASRec recommendations for Beauty dataset

# Model configuration
MODEL_NAME="google/gemma-3-1b-it"

# Training settings
BATCH_SIZE=4
EVAL_BATCH_SIZE=32
EPOCHS=0
LEARNING_RATE=1e-5
MAX_HISTORY_LEN=8
MAX_STEPS=15000

# SASRec settings
USE_SASREC="--use_sasrec"
SASREC_TOP_K=10

data_name=beauty

# SASRec 프롬프트 타입들
prompt_types=(
    "seq_rec_with_sasrec"
    "seq_rec_recent_with_sasrec"
)

for prompt_type in "${prompt_types[@]}"; do
    RUN_NAME="zeroshot_${prompt_type}_${data_name}"
    CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
    FINAL_CHECKPOINT_DIR="checkpoints/${RUN_NAME}/checkpoint-${MAX_STEPS}"
    
    echo "=================================="
    echo "Running: ${RUN_NAME}"
    echo "=================================="
    
    # 제로샷 평가 실행
    CUDA_VISIBLE_DEVICES=1 python3 src/train_sft.py \
        --data_name $data_name \
        --model_name $MODEL_NAME \
        --run_name $RUN_NAME \
        --prompt_type $prompt_type \
        --use_brand \
        --use_category \
        --target_type item_metadata \
        --max_steps $MAX_STEPS \
        --checkpoint_dir $CHECKPOINT_DIR \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --max_history_len $MAX_HISTORY_LEN \
        --gpu_memory_utilization 0.95 \
        --zeroshot_evaluation \
        --run_evaluation \
        $USE_SASREC \
        --sasrec_top_k $SASREC_TOP_K \
        "$@"
    
    echo ""
    echo "✓ Completed: ${RUN_NAME}"
    echo ""
done

echo "✅ All zeroshot evaluations with SASRec completed for ${data_name}!"



