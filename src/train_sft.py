#!/usr/bin/env python3
"""
SFT Training Script for Recommendation System
Supervised Fine-Tuning using TRL's SFTTrainer
"""

import torch
import argparse
import gc

from transformers import (
    AutoModelForCausalLM, 
    Gemma3ForCausalLM,
    BitsAndBytesConfig, 
)
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import random
from collections import defaultdict
from typing import Callable, List, Dict, Tuple, Any

from utils.common import (
    load_json,
    print_arguments,
    get_vllm_model,
    get_sampling_params,
    generate_response_with_vllm,
    initialize_tokenizer,
    )
from utils.dataset import create_dataloaders
from evaluator import RecommendationEvaluator

# off warning messages
import warnings
warnings.filterwarnings("ignore") 

def generate_target_text_from_metadata(target_dict, item_metadata, use_brand=True, use_category=True):
    """
    타겟 아이템의 메타데이터만을 사용하여 타겟 텍스트 생성
    
    Args:
        target_dict: {user_id: target_item_id} 딕셔너리
        item_metadata: 아이템 메타데이터
        use_brand: 브랜드 포함 여부
        use_category: 카테고리 포함 여부
    
    Returns:
        target_text_list: 타겟 텍스트 리스트 (user_id 순서대로)
    """
    target_text_list = []
    
    for user_id in sorted(target_dict.keys()):
        target_item_id = target_dict[user_id]
        item_info = item_metadata[target_item_id]
        text_parts = []
        
        title = item_info.get('title', '')
        # limit title length to 64 words
        title = " ".join(title.split()[:64])
        text_parts.append(title)
        
        if use_brand:
            brand = item_info.get('brand', '')
            text_parts.append(f"Brand: {brand}")
        
        if use_category:
            category = item_info.get('category', '')
            text_parts.append(f"Category: {category}")
        
        target_text = "\n".join(text_parts)
        target_text_list.append(target_text)
    
    return target_text_list


def remove_prefix_from_target_text(target_text):
    """
    Remove prefix from target text
    """
    if ":" in target_text:
        return " ".join(target_text.split(":")[-1:])
    else:
        return target_text

def load_trigger_items(trigger_items_dir: str, data_name: str, split: str = "train") -> Dict[int, int]:
    """
    Load trigger items from JSON file
    
    Args:
        trigger_items_dir: Directory containing trigger items files
        data_name: Dataset name (e.g., 'beauty', 'toys')
        split: Dataset split ('train', 'val', 'test')
    
    Returns:
        trigger_items_dict: {user_id: trigger_item_id} dictionary
    """
    # Convert 'valid' to 'val' for file naming
    file_split = 'val' if split == 'valid' else split
    
    trigger_file = f"{trigger_items_dir}/SASRec_{data_name}_{file_split}_trigger_items.json"
    print(f"📌 Loading trigger items from: {trigger_file}")
    
    if not os.path.exists(trigger_file):
        print(f"⚠️  Warning: Trigger items file not found: {trigger_file}")
        return {}
    
    with open(trigger_file, 'r') as f:
        trigger_items = json.load(f)
    
    # Convert keys to int
    trigger_items_dict = {int(k): int(v) for k, v in trigger_items.items()}
    print(f"✓ Loaded trigger items for {len(trigger_items_dict)} users")
    
    # Show sample
    if len(trigger_items_dict) > 0:
        sample_users = list(trigger_items_dict.items())[:3]
        print(f"  Sample: {sample_users}")
    
    return trigger_items_dict


def load_target_text_from_file(args, split="train"):
    """
    파일에서 타겟 텍스트 로드 (기존 방식)
    
    Args:
        args: 학습 설정 파라미터
        split: 데이터셋 split ('train', 'valid', 'test')
    
    Returns:
        target_text_dict: {user_id: target_text} 딕셔너리
    """
    target_file = f"data_processed/{args.data_name}_{args.target_model_name}_{split}_{args.target}.json"
    print(f"📄 Loading target text from: {target_file}")
    
    with open(target_file, 'r') as f:
        target_text = json.load(f)
    
    target_text_dict = {int(k): remove_prefix_from_target_text(v) for k, v in target_text.items()}
    print(f"✓ Loaded target text for {len(target_text_dict)} users")
    
    return target_text_dict


def load_target_text_from_rejection_sampling_csv(
    csv_path: str,
    max_rank_threshold: int = None,
    min_rank_improvement: float = None,
) -> dict:
    """
    Rejection sampling CSV에서 타겟 텍스트 로드
    
    Args:
        csv_path: Rejection sampling CSV 파일 경로
        max_rank_threshold: 최대 rank 임계값 (이 값 이하만 사용)
        min_rank_improvement: 최소 rank improvement 임계값 (이 값 이상만 사용)
    
    Returns:
        target_text_dict: {user_id: generated_query} 딕셔너리
    """
    print(f"\n📄 Loading rejection sampling results from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_columns = ['user_id', 'generated_query', 'current_rank']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Found: {df.columns.tolist()}")
    
    print(f"✓ Loaded {len(df)} samples from CSV")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Apply filters
    original_count = len(df)
    
    if max_rank_threshold is not None:
        df = df[df['current_rank'] <= max_rank_threshold]
        print(f"  Filtered by max_rank_threshold={max_rank_threshold}: {original_count} -> {len(df)} samples")
    
    if min_rank_improvement is not None:
        if 'rank_improvement' not in df.columns:
            print("  ⚠️  Warning: rank_improvement column not found, skipping min_rank_improvement filter")
        else:
            df = df[df['rank_improvement'] >= min_rank_improvement]
            print(f"  Filtered by min_rank_improvement={min_rank_improvement}: {original_count} -> {len(df)} samples")
    
    # Create dictionary
    target_text_dict = {}
    for _, row in df.iterrows():
        user_id = int(row['user_id'])
        generated_query = row['generated_query']
        target_text_dict[user_id] = generated_query
    
    print(f"✓ Prepared target text for {len(target_text_dict)} users")
    
    # Show statistics
    print(f"\n📊 Rejection Sampling Statistics:")
    print(f"  Total samples after filtering: {len(df)}")
    if 'current_rank' in df.columns:
        print(f"  Current rank - Mean: {df['current_rank'].mean():.2f}, Median: {df['current_rank'].median():.0f}")
    if 'baseline_rank' in df.columns:
        print(f"  Baseline rank - Mean: {df['baseline_rank'].mean():.2f}, Median: {df['baseline_rank'].median():.0f}")
    if 'rank_improvement' in df.columns:
        print(f"  Rank improvement - Mean: {df['rank_improvement'].mean():.2f}, Median: {df['rank_improvement'].median():.2f}")
    if 'hit@5' in df.columns:
        print(f"  Hit@5 rate: {df['hit@5'].mean():.4f}")
    if 'hit@10' in df.columns:
        print(f"  Hit@10 rate: {df['hit@10'].mean():.4f}")
    if 'hit@20' in df.columns:
        print(f"  Hit@20 rate: {df['hit@20'].mean():.4f}")
    
    # Show sample
    print(f"\n📝 Sample generated query:")
    if len(target_text_dict) > 0:
        sample_user_id = list(target_text_dict.keys())[0]
        sample_query = target_text_dict[sample_user_id]
        print(f"  User ID: {sample_user_id}")
        print(f"  Query: {sample_query[:200]}..." if len(sample_query) > 200 else f"  Query: {sample_query}")
    
    return target_text_dict


def prepare_sft_dataset(dataset, target_text_dict):
    """
    SFT를 위한 데이터셋 준비
    
    Args:
        dataset: RecommendationDataset 인스턴스
        target_text_dict: {user_id: target_text} 딕셔너리
    
    Returns:
        Dataset: HuggingFace Dataset (prompt, completion 컬럼 포함)
    """
    prompts = []
    completions = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        user_id = data['user_id']
        prompt = data['prompt']
        
        # 타겟 텍스트 가져오기
        target_text = target_text_dict.get(user_id, "")
        
        if not target_text:
            # 타겟 텍스트가 없는 경우 스킵
            continue
        
        prompts.append(prompt)
        completions.append(target_text)
    
    # 샘플 출력
    print("\n" + "="*80)
    print("📝 Sample Data:")
    print("="*80)
    for i in [10, 20, 30]:
        if i < len(prompts):
            print(f"\n--- Sample {i+1} ---")
            print("Prompt:")
            print(prompts[i])
            print("\nCompletion:")
            print(completions[i])
            print("-" * 80)
        
    data_df = pd.DataFrame({
        "prompt": prompts,
        "completion": completions,
    })
    
    return Dataset.from_pandas(data_df)


def get_quantization_config(args):
    print(f"Using {args.quantization_option} quantization")
    
    quantization_config = None
    
    if args.quantization_option == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    elif args.quantization_option == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=False,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16
        )
    
    return quantization_config


def get_last_item_text(dataset, item_metadata, use_brand=True, use_category=True):
    """
    각 사용자의 마지막 구매 아이템 정보를 텍스트로 변환
    
    Args:
        dataset: RecommendationDataset 인스턴스
        item_metadata: 아이템 메타데이터
        use_brand: 브랜드 포함 여부
        use_category: 카테고리 포함 여부
    
    Returns:
        last_item_texts: 각 샘플의 마지막 아이템 텍스트 리스트
    """
    last_item_texts = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        history = data.get('history', [])
        
        if len(history) > 0:
            last_item_id = history[-1]  # 마지막 아이템
            item_info = item_metadata.get(last_item_id, {})
            
            text_parts = []
            title = item_info.get('title', '')
            # limit title length to 64 words
            title = " ".join(title.split()[:64])
            text_parts.append(f"Last Item: {title}")
            
            if use_brand:
                brand = item_info.get('brand', '')
                if brand:
                    text_parts.append(f"Brand: {brand}")
            
            if use_category:
                category = item_info.get('category', '')
                if category:
                    text_parts.append(f"Category: {category}")
            
            last_item_text = "\n".join(text_parts)
        else:
            last_item_text = ""
        
        last_item_texts.append(last_item_text)
    
    return last_item_texts


def evaluate_final_metrics(args, dataset, split="test", pre_generated_texts=None, item_metadata=None):
    """
    최종 평가: RecommendationEvaluator를 사용하여 평가
    
    Args:
        args: 학습 설정 파라미터
        dataset: RecommendationDataset 인스턴스
        split: 데이터셋 split 이름
        pre_generated_texts: 미리 생성된 텍스트 리스트 (선택사항)
        item_metadata: 아이템 메타데이터 (prepend_last_item 사용 시 필요)
    
    Returns:
        results: 평가 결과 딕셔너리
    """
    print(f"\n{'='*80}")
    print(f"🔍 Starting Evaluation on {split.upper()} Set")
    print(f"{'='*80}")
    
    # GPU 메모리 정리
    print("🧹 Cleaning up training resources before evaluation...")
    torch.cuda.empty_cache()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"💾 GPU Memory before evaluation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 마지막 아이템 prepend 옵션 처리
    if args.prepend_last_item and pre_generated_texts is not None:
        if item_metadata is None:
            print("⚠️  Warning: prepend_last_item is enabled but item_metadata is not provided. Skipping prepending.")
        else:
            print("📝 Prepending last purchased item to generated texts...")
            last_item_texts = get_last_item_text(
                dataset, 
                item_metadata,
                use_brand=args.use_brand,
                use_category=args.use_category
            )
            
            # 마지막 아이템 텍스트를 generated text 앞에 추가
            modified_texts = []
            for last_item_text, generated_text in zip(last_item_texts, pre_generated_texts):
                if last_item_text:
                    modified_text = f"{last_item_text}\n\n{generated_text}"
                else:
                    modified_text = generated_text
                modified_texts.append(modified_text)
            
            pre_generated_texts = modified_texts
            print(f"✓ Prepended last item to {len(pre_generated_texts)} texts")
            
            # 샘플 출력
            print("\n" + "="*80)
            print("📝 Sample Modified Text (with last item prepended):")
            print("="*80)
            if len(pre_generated_texts) > 0:
                print(pre_generated_texts[0][:500] + "..." if len(pre_generated_texts[0]) > 500 else pre_generated_texts[0])
                print("="*80)
    
    # Evaluator 인스턴스 생성 및 평가 실행
    evaluator = RecommendationEvaluator(args, args.final_checkpoint_dir)
    
    try:
        results = evaluator.evaluate(dataset, split=split, save_log=True, pre_generated_texts=pre_generated_texts)
    finally:
        # 평가 완료 후 메모리 정리
        evaluator.cleanup()
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            print(f"💾 GPU Memory after evaluation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return results



def parse_arguments():
    """Command line arguments"""
    parser = argparse.ArgumentParser(
        description="SFT Training for Recommendation System"
    )
    
    # Basic settings
    parser.add_argument("--run_name", type=str, default="sft")
    parser.add_argument("--data_name", type=str, default="beauty")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--quantization_option", type=str, default="None",
                        choices=["None", "4bit", "8bit"])
    parser.add_argument("--rank_dim", type=int, default=8,
                        help="LoRA rank dimension")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    
    # Token settings
    parser.add_argument("--max_length", type=int, default=1024*2,
                        help="Maximum sequence length for SFT")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens for generation")
    parser.add_argument("--eval_max_tokens", type=int, default=128,
                        help="Maximum new tokens for generation")

    # Target settings
    parser.add_argument("--target_type", type=str, default="from_file",
                        choices=["from_file", "item_metadata", "rejection_sampling"],
                        help="Target text generation method: from_file (기존 파일에서 로드), item_metadata (아이템 메타데이터 사용), rejection_sampling (rejection sampling CSV 사용)")
    parser.add_argument("--target", type=str, default="user_preference_reasoning",
                        help="Target type name (when using from_file)")
    parser.add_argument("--target_model_name", type=str, default="gemma-3-12b-it",
                        help="Target model name (when using from_file)")
    parser.add_argument("--use_brand", action="store_true", default=True,
                        help="Include brand in target/prompt (when using item_metadata)")
    parser.add_argument("--use_category", action="store_true", default=True,
                        help="Include category in target/prompt (when using item_metadata)")
    
    # Rejection Sampling settings
    parser.add_argument("--rejection_sampling_csv", type=str, default=None,
                        help="Path to rejection sampling CSV file (when using target_type=rejection_sampling)")
    parser.add_argument("--max_rank_threshold", type=int, default=None,
                        help="Maximum rank threshold - only use samples with current_rank <= this value")
    parser.add_argument("--min_rank_improvement", type=float, default=None,
                        help="Minimum rank improvement threshold - only use samples with rank_improvement >= this value")
    
    # Prompt Generation settings
    parser.add_argument("--prompt_type", type=str, default="seq_rec",
                        help="Prompt template type")
    parser.add_argument("--use_description", action="store_true",
                        help="Include description in prompt")
    parser.add_argument("--use_features", action="store_true",
                        help="Include features in prompt")
    parser.add_argument("--use_date", action="store_true", default=True,
                        help="Include purchase date information in prompt")
    parser.add_argument("--use_relative_date", action="store_true",
                        help="Use relative date format (D-N) based on target purchase date instead of absolute dates")
    parser.add_argument("--use_last_item", action="store_true", default=True,
                        help="Emphasize last item in prompt")
    parser.add_argument("--emphasize_recent_item", action="store_true",
                        help="Emphasize recent purchase item with detailed information including purchase date ('This user's most recent purchase is...' format)")
    parser.add_argument("--include_target_date", action="store_true",
                        help="Include target/label item's purchase date at the end of prompt")
    parser.add_argument("--max_history_len", type=int, default=8,
                        help="Max history length")
    parser.add_argument("--history_text_max_length", type=int, default=128,
                        help="Max words per history item")
    parser.add_argument("--days_filter", type=int, default=None,
                        help="Filter reviews to only include those within N days of target date")
    
    # SASRec Integration
    parser.add_argument("--use_sasrec", action="store_true",
                        help="Include SASRec recommendations in prompt as reference for query generation")
    parser.add_argument("--sasrec_top_k", type=int, default=5,
                        help="Number of top-K SASRec recommendations to include in prompt")
    
    # Checkpoint settings
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/sft",
                        help="Checkpoint directory")
    parser.add_argument("--final_checkpoint_dir", type=str, default="checkpoints/sft/checkpoint-5000",
                        help="Final checkpoint directory for evaluation")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="wandb",
                        help="Logging backend (wandb, tensorboard, none)")

    # Evaluation settings
    parser.add_argument("--run_evaluation", action="store_true",
                        help="Run evaluation after training")
    parser.add_argument("--eval_on_train", action="store_true",
                        help="Run evaluation on train set")
    parser.add_argument("--eval_on_test", action="store_true", default=True,
                        help="Run evaluation on test set")
    parser.add_argument("--emb_model_name", type=str, 
                        default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Embedding model name for evaluation")
    parser.add_argument("--emb_type", type=str, default="item_meta_only",
                        help="Embedding type (title, description, etc.)")
    parser.add_argument("--eval_emb_max_length", type=int, default=512)
    parser.add_argument("--eval_emb_batch_size", type=int, default=512)
    parser.add_argument("--eval_samples", type=int, default=100000)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--eval_emb_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--zeroshot_evaluation", action="store_true", help="Run zeroshot evaluation")

    parser.add_argument("--use_sentence_transformers", action="store_true", help="Use sentence transformers for evaluation")
    
    # Pre-generated CSV for evaluation
    parser.add_argument("--pre_generated_csv", type=str, default=None,
                        help="Path to CSV file containing pre-generated texts for evaluation")
    
    # Prepend last item option
    parser.add_argument("--prepend_last_item", action="store_true",
                        help="Prepend last purchased item to generated text during evaluation")
    
    # Trigger items settings
    parser.add_argument("--use_trigger_items", action="store_true",
                        help="Use trigger items to emphasize key items in prompt")
    parser.add_argument("--trigger_items_dir", type=str, 
                        default="sasrec_results/trigger_items_from_sequential",
                        help="Directory containing trigger items JSON files")
    parser.add_argument("--trigger_emphasis_text", type=str,
                        default="This item was particularly influential in shaping the user's preferences.",
                        help="Text to add after trigger item to emphasize its importance")
    
    # Rank-based filtering for training
    parser.add_argument("--filter_train_csv", type=str, default=None,
                        help="Path to evaluation CSV file for filtering train set by rank")
    parser.add_argument("--filter_valid_csv", type=str, default=None,
                        help="Path to evaluation CSV file for filtering valid set by rank")
    parser.add_argument("--filter_test_csv", type=str, default=None,
                        help="Path to evaluation CSV file for filtering test set by rank")
    parser.add_argument("--rank_min", type=int, default=None,
                        help="Minimum rank for filtering (inclusive, None = no limit)")
    parser.add_argument("--rank_max", type=int, default=None,
                        help="Maximum rank for filtering (inclusive, None = no limit)")

    args = parser.parse_args()

    return args

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Seed 설정
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print("🚀 SFT Training for Recommendation System")
    print("=" * 80)
    print_arguments(args)
    
    # 모델 이름 디렉토리 설정
    args.model_name_dir = args.model_name.split("/")[-1]

    # 토크나이저 로드
    print(f"\n📚 Loading tokenizer: {args.model_name}")
    tokenizer = initialize_tokenizer(args.model_name)
    
    # Trigger items 로드 (옵션)
    trigger_items_train = None
    trigger_items_valid = None
    trigger_items_test = None
    
    if args.use_trigger_items:
        print(f"\n📌 Loading trigger items...")
        trigger_items_train = load_trigger_items(args.trigger_items_dir, args.data_name, split="train")
        trigger_items_valid = load_trigger_items(args.trigger_items_dir, args.data_name, split="valid")
        trigger_items_test = load_trigger_items(args.trigger_items_dir, args.data_name, split="test")
    
    # 데이터로더 생성 (create_dataloaders 함수 사용)
    print(f"\n📊 Creating datasets...")
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        prompt_generator,
        item_metadata,
    ) = create_dataloaders(
        args, 
        tokenizer=tokenizer, 
        apply_chat_template=True,
        trigger_items_train=trigger_items_train,
        trigger_items_valid=trigger_items_valid,
        trigger_items_test=trigger_items_test,
        trigger_emphasis_text=args.trigger_emphasis_text if args.use_trigger_items else None,
    )
    
    # 타겟 텍스트 생성
    print(f"\n📝 Generating target text (target_type={args.target_type})...")
    
    if args.target_type == "item_metadata":
        # 옵션 1: 아이템 메타데이터만 사용
        print("  Using item metadata for target text")
        train_target_dict = generate_target_text_from_metadata(
            train_dataset.target_dict, 
            item_metadata,
            use_brand=args.use_brand,
            use_category=args.use_category,
        )
        valid_target_dict = generate_target_text_from_metadata(
            valid_dataset.target_dict,
            item_metadata,
            use_brand=args.use_brand,
            use_category=args.use_category,
        )
        
        # 리스트를 딕셔너리로 변환
        train_target_text_dict = {user_id: train_target_dict[i] 
                                  for i, user_id in enumerate(sorted(train_dataset.target_dict.keys()))}
        valid_target_text_dict = {user_id: valid_target_dict[i] 
                                  for i, user_id in enumerate(sorted(valid_dataset.target_dict.keys()))}
        
    elif args.target_type == "from_file":
        # 옵션 2: 파일에서 로드
        print(f"  Loading target text from file (target={args.target})")
        train_target_text_dict = load_target_text_from_file(args, split="train")
        valid_target_text_dict = load_target_text_from_file(args, split="valid")
    
    elif args.target_type == "rejection_sampling":
        # 옵션 3: Rejection sampling CSV에서 로드
        if args.rejection_sampling_csv is None:
            raise ValueError("--rejection_sampling_csv must be specified when using target_type=rejection_sampling")
        
        print(f"  Loading target text from rejection sampling CSV")
        train_target_text_dict = load_target_text_from_rejection_sampling_csv(
            csv_path=args.rejection_sampling_csv,
            max_rank_threshold=args.max_rank_threshold,
            min_rank_improvement=args.min_rank_improvement,
        )
        
        # Valid set은 rejection sampling을 사용하지 않고 기본 방식 사용
        # (또는 별도의 validation CSV 지정 가능하도록 확장 가능)
        print("\n  For validation set, using item metadata")
        valid_target_dict = generate_target_text_from_metadata(
            valid_dataset.target_dict,
            item_metadata,
            use_brand=args.use_brand,
            use_category=args.use_category,
        )
        valid_target_text_dict = {user_id: valid_target_dict[i] 
                                  for i, user_id in enumerate(sorted(valid_dataset.target_dict.keys()))}
    
    else:
        raise ValueError(f"Unknown target_type: {args.target_type}")
    
    # SFT 데이터셋 준비
    print(f"\n📦 Preparing SFT datasets...")
    if args.num_epochs > 0:
        train_sft_dataset = prepare_sft_dataset(train_dataset, train_target_text_dict)
        valid_sft_dataset = prepare_sft_dataset(valid_dataset, valid_target_text_dict)
        
        print(f"  Train samples: {len(train_sft_dataset)}")
        print(f"  Valid samples: {len(valid_sft_dataset)}")

    # 모델 로드
    print(f"\n🤖 Loading model: {args.model_name}")
    quantization_config = get_quantization_config(args)
    


    # Gemma3 특별 처리
    if "gemma-3-1b" in args.model_name:
        print("  Using Gemma3ForCausalLM for gemma-3-1b")
        model = Gemma3ForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            )
    elif "gemma-3" in args.model_name and hasattr(model, 'model'):
        print("  Using Gemma3 text model only")
        model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        )
        model = model.model
    else:
        model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        )

    # 학습 시작
    if args.num_epochs > 0:
        print(f"\n{'='*80}")
        print("🚀 Starting SFT Training")
        print(f"{'='*80}")

        # SFT 설정
        training_args = SFTConfig(
            run_name=f"{args.run_name}_{args.data_name}",
            output_dir=args.checkpoint_dir,
            max_steps=args.max_steps,
            max_length=args.max_length,
            packing=False,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=False,
            max_grad_norm=args.max_grad_norm,
            optim="adamw_torch_fused",
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            learning_rate=args.learning_rate,
            bf16=True,
            lr_scheduler_type="cosine",
            report_to=args.report_to if args.report_to != "none" else None,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": True,
            }
        )
        
        # SFT Trainer 초기화
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_sft_dataset,
            eval_dataset=valid_sft_dataset,
            processing_class=tokenizer,
        )
        
        # 학습 실행
        trainer.train()
        
        # 모델 저장
        output_model_dir = f"models/{args.run_name}_{args.data_name}_{args.model_name_dir}"
        print(f"\n💾 Saving model to: {output_model_dir}")
        os.makedirs(output_model_dir, exist_ok=True)
        model.save_pretrained(output_model_dir)
        tokenizer.save_pretrained(output_model_dir)
        
        print("=" * 80)
        print("✓ Training completed!")
        print("=" * 80)
        
        # 학습 완료 후 메모리 정리
        print("\n🧹 Cleaning up training resources...")
        
        if hasattr(trainer, 'model'):
            trainer.model = None
        if hasattr(trainer, 'optimizer'):
            trainer.optimizer = None
        if hasattr(trainer, 'lr_scheduler'):
            trainer.lr_scheduler = None
        if hasattr(trainer, 'accelerator'):
            trainer.accelerator.free_memory()
        
        del model
        del trainer
        
        torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"💾 GPU Memory after training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 평가
    if args.run_evaluation:
        print("\n" + "="*80)
        print("🎯 Starting Model Evaluation")
        print("="*80)
        
        # Pre-generated CSV가 제공된 경우
        pre_generated_texts = None
        if args.pre_generated_csv is not None:
            print(f"\n📄 Loading pre-generated texts from: {args.pre_generated_csv}")
            try:
                df = pd.read_csv(args.pre_generated_csv)
                if 'generated_text' not in df.columns:
                    raise ValueError(f"CSV file must contain 'generated_text' column. Found columns: {df.columns.tolist()}")
                pre_generated_texts = df['generated_text'].tolist()
                print(f"✓ Loaded {len(pre_generated_texts)} pre-generated texts from CSV")
            except Exception as e:
                print(f"❌ Error loading CSV file: {e}")
                raise
        
        # Train set 평가
        if args.eval_on_train:
            print("\n" + "="*80)
            print("📊 Evaluating on TRAIN Set")
            print("="*80)
            train_results = evaluate_final_metrics(
                args, 
                train_dataset, 
                split="train", 
                pre_generated_texts=None,
                item_metadata=item_metadata
            )
            
            print("\n" + "="*80)
            print("✅ Train Evaluation Complete!")
            print("="*80)
            print("\nTrain Results:")
            for metric_name, value in train_results.items():
                print(f"  {metric_name.upper()}: {value:.4f}")
            print("="*80)
        
        # Test set 평가
        if args.eval_on_test:
            print("\n" + "="*80)
            print("📊 Evaluating on TEST Set")
            print("="*80)
            test_results = evaluate_final_metrics(
                args, 
                test_dataset, 
                split="test", 
                pre_generated_texts=pre_generated_texts,
                item_metadata=item_metadata
            )
        
            print("\n" + "="*80)
            print("✅ Test Evaluation Complete!")
            print("="*80)
            print("\nTest Results:")
            for metric_name, value in test_results.items():
                print(f"  {metric_name.upper()}: {value:.4f}")
            print("="*80)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
    