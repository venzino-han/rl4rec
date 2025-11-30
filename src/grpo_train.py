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
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from trl import GRPOTrainer, GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context

from transformers import AutoTokenizer, AutoModelForCausalLM

from train_utils.reward_funtion import RecRewardFrunction
from train_utils.dataset import create_dataloaders

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
        ) = create_dataloaders(args)
        
        # 리워드 함수
        print(f"💰 Creating reward function: {args.reward_type}@{args.k}")
        self.reward_fn = RecRewardFrunction(
            retrieval_service_name=args.retrieval_service_name,
            namespace=args.namespace,
            dataset_name=args.dataset_name,
            reward_type=args.reward_type,
            k=args.k,
            normalize=args.normalize_rewards,
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
    
    def train(self):
        """
        전체 학습 루프 실행
        """
        print("=" * 80)
        print("🚀 Starting GRPO Training")
        print("=" * 80)
        
        global_step = 0
        best_reward = -float('inf')

        self.grpo_trainer.train()
        
        # 최종 테스트 평가
        print("\n" + "=" * 80)
        print("📊 Final Evaluation on Test Set")
        print("=" * 80)
        test_metrics = self.evaluate(self.test_dataloader, split="test")
        
        # 최종 체크포인트
        final_checkpoint = self.checkpoint_dir / "checkpoint_final"
        self.model.save_pretrained(final_checkpoint)
        self.tokenizer.save_pretrained(final_checkpoint)
        print(f"💾 Final checkpoint saved: {final_checkpoint}")
        
        print("=" * 80)
        print("✓ Training completed!")
        print(f"  Total steps: {global_step}")
        print(f"  Best valid reward: {best_reward:.4f}")
        print("=" * 80)


def parse_args():
    """Command line arguments"""
    parser = argparse.ArgumentParser(
        description="GRPO Training for Recommendation System"
    )
    
    # Ray & Service
    parser.add_argument("--ray_address", type=str, default="auto")
    parser.add_argument("--namespace", type=str, default="rl4rec")
    parser.add_argument("--retrieval_service_name", type=str, default="RetrievalService")
    parser.add_argument("--dataset_name", type=str, default="beauty")
    
    # Model
    parser.add_argument("--policy_model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use_ref_model", action="store_true", help="Use reference model for KL penalty")
    
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
    
    # GRPO Training
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_sample_generations", type=int, default=4,
                        help="Number of generations per prompt for GRPO")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=10000)
    
    # Reward
    parser.add_argument("--reward_type", type=str, default="ndcg",
                        choices=["ndcg", "hit", "mrr", "mixed"])
    parser.add_argument("--k", type=int, default=100, help="Top-K for metrics")
    parser.add_argument("--normalize_rewards", action="store_true", help="Normalize rewards")
    parser.add_argument("--num_negs", type=int, default=0, help="Number of negative items")
    
    # Logging & Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="wandb", 
                        help="Logging backend (wandb, tensorboard, none)")
    parser.add_argument("--run_name", type=str, default=None)
    
    # Precision
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    # parser.add_argument("--fp16", action="store_true", help="Use float16")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Run name 설정
    if args.run_name is None:
        args.run_name = f"grpo_{args.reward_type}@{args.k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Trainer 초기화 및 학습
    trainer = GRPOTrainerWrapper(args)
    trainer.train()
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
