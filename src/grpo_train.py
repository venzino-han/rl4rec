#!/usr/bin/env python3
"""
GRPO Training Script for Recommendation System
TRL의 GRPOTrainer를 사용한 추천 시스템 학습
RetrievalService와 연동하여 NDCG 기반 리워드로 학습
"""

import os
import ray
import torch
import argparse
import json
import gc
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from trl import GRPOTrainer, GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.pooling_params import PoolingParams

from utils.reward_funtion import RecRewardFrunction, calculate_ndcg, calculate_hit_rate
from utils.dataset import create_dataloaders
from evaluator import RecommendationEvaluator

from accelerate import logging
from accelerate.utils import gather



logger = logging.get_logger(__name__)

class GRPOTrainerRecReward(GRPOTrainer):

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations

        # print(f"inputs: {inputs[0].keys()}")
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # print(f"reward_kwargs: {reward_kwargs}")
        # print(f"keys: {reward_kwargs.keys()}")

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            with profiling_context(self, reward_func_name):
                output_reward_func = reward_func(
                    generated_texts=completions,
                    targets=reward_kwargs["target"],
                    histories=reward_kwargs["history"],
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

class GRPOTrainerWrapper:
    """
    TRL GRPO를 활용한 추천 시스템 학습기
    """
    
    def __init__(self, args):
        self.args = args
        
        # Ray 초기화 (이미 되어있으면 skip)
        if not ray.is_initialized():
            print(f"🔧 Initializing Ray...")
            ray.init(address=args.ray_address, namespace=args.namespace)
            print(f"✓ Ray initialized")
        
        # 토크나이저 로드
        print(f"📚 Loading tokenizer: {args.policy_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.policy_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        print(f"🤖 Loading model: {args.policy_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.policy_model,
            trust_remote_code=True,
        )
        
        # GRPO Config
        grpo_config = GRPOConfig(
            output_dir=args.checkpoint_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.log_interval,
            eval_steps=args.eval_interval,
            save_steps=args.save_interval,
            save_total_limit=args.save_total_limit,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            bf16=args.bf16,
            report_to=args.report_to if args.report_to != "none" else None,
            run_name=args.run_name,
            # GRPO specific
            num_generations=args.num_sample_generations,
            temperature=args.temperature,
            max_completion_length=args.max_new_tokens,
            max_steps=args.max_steps,
            include_for_metrics=["reward", "entropy", "grad_norm", "epoch"]
        )
        
        # GRPO Trainer

        
        # 데이터로더 생성 (create_dataloaders 함수 사용)
        (
            self.train_dataset,
            self.valid_dataset,
            self.test_dataset,
            self.prompt_generator,
            self.item_metadata,
        ) = create_dataloaders(args, tokenizer=self.tokenizer)
        
        # 리워드 함수
        print(f"💰 Creating reward function: {args.reward_type}@{args.k}")
        self.reward_fn = RecRewardFrunction(
            retrieval_service_name=args.retrieval_service_name,
            namespace=args.namespace,
            data_name=args.data_name,
            reward_type=args.reward_type,
            k=args.k,
            normalize=args.normalize_rewards,
            test_target=args.test_target,
        )

        print(f"🎯 Initializing GRPO Trainer...")
        self.grpo_trainer = GRPOTrainerRecReward(
            model=self.model,
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            reward_funcs=self.reward_fn,
            processing_class=self.tokenizer,
        )
    
    def evaluate_final_metrics(self, dataset, split="test"):
        """
        최종 평가: hit@k, ndcg@k (k=5,10,20)를 계산
        RecommendationEvaluator 클래스를 사용하여 평가 수행
        """
        print(f"\n{'='*80}")
        print("🧹 Cleaning up training resources before evaluation...")
        print(f"{'='*80}")
        
        # GPU 메모리 정리 - 더 확실하게
        if hasattr(self, 'model') and self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
        
        if hasattr(self, 'grpo_trainer') and self.grpo_trainer is not None:
            # trainer 내부의 model, optimizer 등을 정리
            if hasattr(self.grpo_trainer, 'model'):
                self.grpo_trainer.model = None
            if hasattr(self.grpo_trainer, 'accelerator'):
                self.grpo_trainer.accelerator.free_memory()
            del self.grpo_trainer
            self.grpo_trainer = None
        
        # 강제 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        # 메모리 사용량 출력
        if torch.cuda.is_available():
            print(f"💾 GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
            print(f"💾 GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

        # Evaluator 인스턴스 생성 및 평가 실행
        evaluator = RecommendationEvaluator(self.args, self.args.final_checkpoint_dir)
        
        try:
            results = evaluator.evaluate(dataset, split=split, save_log=True)
        finally:
            # 평가 완료 후 메모리 정리
            evaluator.cleanup()
        
        return results
    
    def train(self):
        """
        전체 학습 루프 실행
        """
        print("=" * 80)
        print("🚀 Starting GRPO Training")
        print("=" * 80)

        if self.args.num_epochs > 0:
            self.grpo_trainer.train()
        print("=" * 80)
        print("✓ Training completed!")        
        # 최종 테스트 평가
        test_metrics = self.evaluate_final_metrics(self.test_dataset, split="test")
        


def parse_args():
    """Command line arguments"""
    parser = argparse.ArgumentParser(
        description="GRPO Training for Recommendation System"
    )
    # basic
    parser.add_argument("--run_name", type=str, default="grpo")
    
    # Ray & Service
    parser.add_argument("--ray_address", type=str, default="auto")
    parser.add_argument("--namespace", type=str, default="rl4rec")
    parser.add_argument("--retrieval_service_name", type=str, default="RetrievalService")
    parser.add_argument("--data_name", type=str, default="beauty")
    
    # Model
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=1024*4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_emb_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use_ref_model", action="store_true", help="Use reference model for KL penalty")
    
    # Embedding Model for Evaluation
    parser.add_argument("--emb_model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--emb_type", type=str, default="review_description", help="Type of item text to embed (title, description, etc.)")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--sequential_file", type=str,
                        default="data/beauty/sequential_data.txt")
    
    # Prompt Generation
    parser.add_argument("--use_brand", action="store_true", default=True, help="Include brand in prompt")
    parser.add_argument("--use_category", action="store_true", default=True, help="Include category in prompt")
    parser.add_argument("--use_description", action="store_true", help="Include description in prompt")
    parser.add_argument("--use_features", action="store_true", help="Include features in prompt")
    parser.add_argument("--use_last_item", action="store_true", default=True, help="Emphasize last item")
    parser.add_argument("--max_history_len", type=int, default=5, help="Max history length")
    parser.add_argument("--history_text_max_length", type=int, default=100, help="Max words per history item")
    parser.add_argument("--test_target", action="store_true", help="Use target text for test")
    
    # GRPO Training
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_sample_generations", type=int, default=4,
                        help="Number of generations per prompt for GRPO")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    
    # Reward
    parser.add_argument("--reward_type", type=str, default="ndcg", choices=["ndcg", "hit", "mixed"])
    parser.add_argument("--k", type=int, default=100, help="Top-K for metrics")
    parser.add_argument("--normalize_rewards", action="store_true", help="Normalize rewards")
    parser.add_argument("--num_negs", type=int, default=0, help="Number of negative items")
    
    # Logging & Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--final_checkpoint_dir", type=str, default="checkpoints/grpo/checkpoint-5000")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="wandb", help="Logging backend (wandb)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    # vllm for evaluation
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--eval_emb_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--eval_emb_batch_size", type=int, default=512)
    parser.add_argument("--eval_emb_max_length", type=int, default=512)
    parser.add_argument("--eval_samples", type=int, default=100000)
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Run name 설정
    if args.run_name is None:
        args.run_name = f"grpo_{args.reward_type}@{args.k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Trainer 초기화 및 학습
    trainer = GRPOTrainerWrapper(args)
    
    try:
        trainer.train()
    finally:
        # 학습 완료 후 명시적으로 모든 리소스 정리
        print("\n" + "="*80)
        print("🧹 Final cleanup...")
        print("="*80)
        
        # Trainer 내부 리소스 정리
        if hasattr(trainer, 'model') and trainer.model is not None:
            del trainer.model
        if hasattr(trainer, 'grpo_trainer') and trainer.grpo_trainer is not None:
            del trainer.grpo_trainer
        if hasattr(trainer, 'reward_fn') and trainer.reward_fn is not None:
            del trainer.reward_fn
        
        del trainer
        
        # GPU 메모리 완전히 해제
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"💾 Final GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
            print(f"💾 Final GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
    print("✓ Done!")



if __name__ == "__main__":
    main()
