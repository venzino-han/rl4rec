#!/usr/bin/env python3
"""
GRPO Evaluation Script for Recommendation System
학습된 모델을 평가만 수행하는 스크립트
"""

import os
import torch
import argparse
import gc
from pathlib import Path
import logging as std_logging

from transformers import AutoTokenizer

from utils.dataset import create_dataloaders
from evaluator import RecommendationEvaluator
from grpo_train import GRPOTrainerWrapper



def parse_args():
    """Command line arguments for evaluation"""
    parser = argparse.ArgumentParser(
        description="GRPO Evaluation for Recommendation System"
    )
    
    # Basic
    parser.add_argument("--run_name", type=str, default="grpo_eval",
                        help="Run name for logging")
    
    # Checkpoint
    parser.add_argument("--final_checkpoint_dir", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/grpo",
                        help="Path to trained model checkpoint")
    
    # Data
    parser.add_argument("--data_name", type=str, default="beauty",
                        help="Dataset name (beauty, sports, toys, yelp)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory path")
    
    # Evaluation Split
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="Which split to evaluate on (train, valid, or test)")
    
    # Model Settings
    parser.add_argument("--model_name", type=str, 
                        default="google/gemma-3-1b-it",
                        help="Model name")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    parser.add_argument("--max_length", type=int, default=1024*4,
                        help="Maximum input sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--max_emb_length", type=int, default=512,
                        help="Maximum embedding length")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    
    # Embedding Model
    parser.add_argument("--emb_model_name", type=str, 
                        default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Embedding model name")
    parser.add_argument("--emb_type", type=str, default="review_description",
                        help="Type of item text to embed")
    parser.add_argument("--use_local_embedding", action="store_true", default=True,
                        help="Use local embedding model")
    
    # Prompt Generation
    parser.add_argument("--prompt_type", type=str, default="seq_rec",
                        help="Prompt template type")
    parser.add_argument("--use_brand", action="store_true", default=True,
                        help="Include brand in prompt")
    parser.add_argument("--use_category", action="store_true", default=True,
                        help="Include category in prompt")
    parser.add_argument("--use_description", action="store_true",
                        help="Include description in prompt")
    parser.add_argument("--use_features", action="store_true",
                        help="Include features in prompt")
    parser.add_argument("--use_last_item", action="store_true", default=True,
                        help="Emphasize last item")
    parser.add_argument("--max_history_len", type=int, default=8,
                        help="Maximum history length")
    parser.add_argument("--history_text_max_length", type=int, default=128,
                        help="Max words per history item")
    parser.add_argument("--days_filter", type=int, default=None,
                        help="Filter reviews to only include those within N days of target date")
    parser.add_argument("--emphasize_recent_item", action="store_true",
                        help="Emphasize recent purchase item with detailed information including purchase date ('This user's most recent purchase is...' format)")
    parser.add_argument("--include_target_date", action="store_true",
                        help="Include target/label item's purchase date at the end of prompt")
    
    # SASRec Integration
    parser.add_argument("--use_sasrec", action="store_true",
                        help="Include SASRec recommendations in prompt as reference for query generation")
    parser.add_argument("--sasrec_top_k", type=int, default=5,
                        help="Number of top-K SASRec recommendations to include in prompt")
    
    # Evaluation Settings
    parser.add_argument("--emb_batch_size", type=int, default=512,
                        help="Batch size for evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--eval_samples", type=int, default=100000,
                        help="Number of samples to evaluate (for debugging)")
    parser.add_argument("--k", type=int, default=100,
                        help="Top-K for metrics")
    parser.add_argument("--prepend_last_item", action="store_true",)
    
    # vLLM Settings
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95,
                        help="GPU memory utilization for vLLM generation")
    parser.add_argument("--eval_emb_gpu_memory_utilization", type=float, default=0.95,
                        help="GPU memory utilization for embedding")
    parser.add_argument("--eval_emb_batch_size", type=int, default=512,
                        help="Batch size for embedding computation")
    parser.add_argument("--eval_emb_max_length", type=int, default=512,
                        help="Max length for embedding")
    parser.add_argument("--eval_max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate for evaluation")
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--dummy_generation", action="store_true",
                        help="Use dummy generation for testing")
    parser.add_argument("--num_epochs", type=int, default=0,
                        help="Number of epochs to evaluate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
    #                     help="Gradient accumulation steps for evaluation")
    parser.add_argument("--report_to", type=str, default="wandb",
                        help="Report to wandb or not")
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    args.sequential_file = f"data/{args.data_name}/sequential_data.txt"
    
    print("=" * 80)
    print("🎯 GRPO Model Evaluation")
    print("=" * 80)
    print(f"📁 Checkpoint: {args.final_checkpoint_dir}")
    print(f"📊 Dataset: {args.data_name}")
    print(f"🔍 Evaluating on: {args.split.upper()} set")
    print("=" * 80)
    
    std_logging.getLogger("vllm").setLevel(std_logging.ERROR)

    # Evaluator 초기화
    evaluator = GRPOTrainerWrapper(args)
    evaluator.evaluate_final_metrics(split=args.split)
    


if __name__ == "__main__":
    main()

