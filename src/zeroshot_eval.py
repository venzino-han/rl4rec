#!/usr/bin/env python3
"""
Zeroshot Evaluation Script with Multiple Rollouts
Evaluates model without training, generating multiple rollouts per prompt
"""

import torch
import argparse
import gc
import os
import json
import pandas as pd
from tqdm import tqdm
import random
from typing import List, Dict, Tuple
from datetime import datetime

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.common import (
    load_json,
    print_arguments,
    initialize_tokenizer,
)
from utils.dataset import create_dataloaders, load_item_metadata
from evaluator import RecommendationEvaluator

import warnings
warnings.filterwarnings("ignore")


def load_trigger_items(trigger_items_dir: str, data_name: str, split: str = "test") -> Dict[int, int]:
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
    
    return trigger_items_dict


def generate_rollouts_with_vllm(
    llm: LLM,
    prompts: List[str],
    user_ids: List[int],
    num_rollouts: int = 1,
    temperature: float = 0.6,
    max_tokens: int = 128,
    top_p: float = 0.9,
    top_k: int = -1,
) -> List[Dict]:
    """
    Generate multiple rollouts for each prompt using vLLM
    
    Args:
        llm: vLLM model instance
        prompts: List of prompt strings
        user_ids: List of user IDs
        num_rollouts: Number of rollouts per prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        List of dictionaries with user_id, rollout_index, and generated_text
    """
    print(f"\n{'='*80}")
    print(f"🎲 Generating {num_rollouts} rollout(s) per prompt")
    print(f"{'='*80}")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Top-p: {top_p}")
    print(f"  Top-k: {top_k}")
    print(f"{'='*80}\n")
    
    # Prepare sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        n=num_rollouts,  # Generate num_rollouts outputs per prompt
    )
    
    # Generate outputs
    print("🔄 Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Collect results
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
        user_id = user_ids[i]
        
        # Each output has multiple completions (rollouts)
        for rollout_idx, completion in enumerate(output.outputs):
            result = {
                'user_id': user_id,
                'rollout_index': rollout_idx,
                'generated_text': completion.text.strip(),
                'prompt': prompts[i],
            }
            results.append(result)
    
    print(f"✓ Generated {len(results)} total outputs ({len(prompts)} prompts × {num_rollouts} rollouts)")
    
    return results


def evaluate_rollouts(
    args,
    results: List[Dict],
    dataset,
    item_metadata: Dict,
    evaluator: RecommendationEvaluator,
    split: str = "test",
) -> pd.DataFrame:
    """
    Evaluate each rollout and return results as DataFrame
    
    Args:
        args: Arguments
        results: List of generation results
        dataset: Dataset to evaluate
        item_metadata: Item metadata dictionary
        evaluator: RecommendationEvaluator instance
        split: Dataset split name
    
    Returns:
        DataFrame with evaluation metrics for each rollout
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating Rollouts")
    print(f"{'='*80}")
    
    # Group results by user_id
    user_rollouts = {}
    for result in results:
        user_id = result['user_id']
        if user_id not in user_rollouts:
            user_rollouts[user_id] = []
        user_rollouts[user_id].append(result)
    
    # Evaluate each rollout
    eval_results = []
    
    # Get unique rollout indices
    num_rollouts = max([r['rollout_index'] for r in results]) + 1
    
    for rollout_idx in range(num_rollouts):
        print(f"\n--- Evaluating Rollout {rollout_idx} ---")
        
        # Collect generated texts for this rollout
        generated_texts = []
        rollout_user_ids = []
        
        for user_id in sorted(user_rollouts.keys()):
            user_results = user_rollouts[user_id]
            # Find the result for this rollout index
            rollout_result = next((r for r in user_results if r['rollout_index'] == rollout_idx), None)
            if rollout_result:
                generated_texts.append(rollout_result['generated_text'])
                rollout_user_ids.append(user_id)
        
        print(f"  Evaluating {len(generated_texts)} samples for rollout {rollout_idx}...")
        
        # Run evaluation for this rollout
        try:
            metrics = evaluator.evaluate(
                dataset, 
                split=split, 
                save_log=False,
                pre_generated_texts=generated_texts
            )
            
            # Add rollout index to metrics
            metrics['rollout_index'] = rollout_idx
            eval_results.append(metrics)
            
            print(f"  Rollout {rollout_idx} results:")
            for metric_name, value in metrics.items():
                if metric_name != 'rollout_index':
                    print(f"    {metric_name}: {value:.4f}")
        
        except Exception as e:
            print(f"  ❌ Error evaluating rollout {rollout_idx}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(eval_results)
    
    print(f"\n{'='*80}")
    print("✅ Evaluation Complete!")
    print(f"{'='*80}")
    
    return results_df


def save_results(
    generation_results: List[Dict],
    eval_results_df: pd.DataFrame,
    args,
    split: str = "test",
    output_dir: str = "results",
):
    """
    Save generation and evaluation results to CSV files
    
    Args:
        generation_results: List of generation results
        eval_results_df: DataFrame with evaluation metrics
        args: Arguments
        split: Dataset split name
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename without date
    base_name = f"zeroshot_{args.data_name}_{split}"
    if args.use_trigger_items:
        base_name += "_trigger"
    
    # Save generation results
    gen_df = pd.DataFrame(generation_results)
    gen_csv_path = f"{output_dir}/{base_name}_generations.csv"
    gen_df.to_csv(gen_csv_path, index=False)
    print(f"\n💾 Saved generation results to: {gen_csv_path}")
    
    # Save evaluation results
    eval_csv_path = f"{output_dir}/{base_name}_eval.csv"
    eval_results_df.to_csv(eval_csv_path, index=False)
    print(f"💾 Saved evaluation results to: {eval_csv_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("📈 Summary Statistics Across Rollouts")
    print(f"{'='*80}")
    
    # Calculate mean and std for each metric
    metric_cols = [col for col in eval_results_df.columns if col != 'rollout_index']
    for col in metric_cols:
        mean_val = eval_results_df[col].mean()
        std_val = eval_results_df[col].std()
        print(f"{col:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"{'='*80}\n")


def parse_arguments():
    """Command line arguments"""
    parser = argparse.ArgumentParser(
        description="Zeroshot Evaluation with Multiple Rollouts"
    )
    
    # Basic settings
    parser.add_argument("--run_name", type=str, default="zeroshot_eval")
    parser.add_argument("--data_name", type=str, default="beauty")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Model checkpoint directory (None for base model)")
    
    # Generation settings
    parser.add_argument("--num_rollouts", type=int, default=1,
                        help="Number of rollouts to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling parameter (-1 for disabled)")
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    
    # Prompt settings
    parser.add_argument("--prompt_type", type=str, default="seq_rec",
                        help="Prompt template type")
    parser.add_argument("--max_history_len", type=int, default=8,
                        help="Max history length")
    parser.add_argument("--history_text_max_length", type=int, default=128,
                        help="Max words per history item")
    parser.add_argument("--use_brand", action="store_true", default=True,
                        help="Include brand in prompt")
    parser.add_argument("--use_category", action="store_true", default=True,
                        help="Include category in prompt")
    parser.add_argument("--use_description", action="store_true",
                        help="Include description in prompt")
    parser.add_argument("--use_date", action="store_true", default=True,
                        help="Include purchase date in prompt")
    parser.add_argument("--use_last_item", action="store_true", default=True,
                        help="Emphasize last item in prompt")
    
    # Trigger items settings
    parser.add_argument("--use_trigger_items", action="store_true",
                        help="Use trigger items to emphasize key items in prompt")
    parser.add_argument("--trigger_items_dir", type=str, 
                        default="sasrec_results/trigger_items_from_sequential",
                        help="Directory containing trigger items JSON files")
    parser.add_argument("--trigger_emphasis_text", type=str,
                        default="This item was particularly influential in shaping the user's preferences.",
                        help="Text to add after trigger item to emphasize its importance")
    
    # Evaluation settings
    parser.add_argument("--emb_model_name", type=str, 
                        default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Embedding model name for evaluation")
    parser.add_argument("--emb_type", type=str, default="item_meta_only",
                        help="Embedding type")
    parser.add_argument("--eval_emb_max_length", type=int, default=512)
    parser.add_argument("--eval_emb_batch_size", type=int, default=512)
    parser.add_argument("--eval_samples", type=int, default=100000)
    parser.add_argument("--eval_emb_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--use_sentence_transformers", action="store_true",
                        help="Use sentence transformers for evaluation")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for dataset")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Evaluation batch size")
    
    # Split selection
    parser.add_argument("--splits", type=str, nargs="+", default=["test"],
                        help="Dataset splits to evaluate (test, valid, train)")
    
    args = parser.parse_args()
    return args


def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Seed 설정
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print("🎯 Zeroshot Evaluation with Multiple Rollouts")
    print("=" * 80)
    print_arguments(args)
    print(f"\n📋 Evaluating splits: {', '.join(args.splits)}")
    
    # 토크나이저 로드
    print(f"\n📚 Loading tokenizer: {args.model_name}")
    tokenizer = initialize_tokenizer(args.model_name)
    
    # Trigger items 로드 (옵션)
    trigger_items_dict = {}
    if args.use_trigger_items:
        print(f"\n📌 Loading trigger items for all splits...")
        for split in args.splits:
            trigger_items_dict[split] = load_trigger_items(
                args.trigger_items_dir, 
                args.data_name, 
                split=split
            )
    
    # 데이터셋 생성
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
        trigger_items_train=trigger_items_dict.get("train"),
        trigger_items_valid=trigger_items_dict.get("valid"),
        trigger_items_test=trigger_items_dict.get("test"),
        trigger_emphasis_text=args.trigger_emphasis_text if args.use_trigger_items else None,
    )
    
    # Split별 데이터셋 매핑
    datasets = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }
    
    print(f"✓ Dataset sizes:")
    for split in args.splits:
        print(f"  {split}: {len(datasets[split])} users")
    
    # vLLM 모델 로드 (한 번만)
    print(f"\n🤖 Loading vLLM model...")
    model_path = args.checkpoint_dir if args.checkpoint_dir else args.model_name
    print(f"  Model path: {model_path}")
    
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Evaluator 생성 (한 번만)
    print(f"\n🔍 Creating evaluator...")
    evaluator = RecommendationEvaluator(args, model_path)
    
    try:
        # 각 split에 대해 평가 수행
        for split in args.splits:
            print(f"\n{'='*80}")
            print(f"📊 Evaluating {split.upper()} split")
            print(f"{'='*80}")
            
            dataset = datasets[split]
            
            # 프롬프트 및 user_ids 추출
            prompts = []
            user_ids = []
            for i in range(len(dataset)):
                data = dataset[i]
                prompts.append(data['prompt'])
                user_ids.append(data['user_id'])
            
            # Generate rollouts
            generation_results = generate_rollouts_with_vllm(
                llm=llm,
                prompts=prompts,
                user_ids=user_ids,
                num_rollouts=args.num_rollouts,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            
            # Evaluate rollouts
            eval_results_df = evaluate_rollouts(
                args=args,
                results=generation_results,
                dataset=dataset,
                item_metadata=item_metadata,
                evaluator=evaluator,
                split=split,
            )
            
            # Save results for this split
            save_results(
                generation_results=generation_results,
                eval_results_df=eval_results_df,
                args=args,
                split=split,
                output_dir=args.output_dir,
            )
            
            print(f"\n✅ Completed evaluation for {split} split")
    
    finally:
        # Clean up
        del llm
        evaluator.cleanup()
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*80)
    print("✓ All evaluations completed!")
    print("="*80)


if __name__ == "__main__":
    main()
