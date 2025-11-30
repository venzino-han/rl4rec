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
import numpy as np

from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

from train_utils.reward_funtion import RecRewardFrunction
from train_utils.dataset import create_dataloaders

from accelerate import logging
from trl.extras.profiling import profiling_decorator, profiling_context
from trl.data_utils import is_conversational
from torch import nn
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
        
        # Reference 모델 (GRPO에서는 optional)
        # self.ref_model = None
        # if args.use_ref_model:
        #     print(f"📖 Loading reference model...")
        #     self.ref_model = AutoModelForCausalLM.from_pretrained(
        #         args.policy_model,
        #         dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        #         device=args.device,
        #     )
        #     self.ref_model.eval()
        
        # GRPO Config
        grpo_config = GRPOConfig(
            output_dir=args.checkpoint_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.log_interval,
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
    
    # def train_step(self, batch, step: int) -> Dict[str, float]:
    #     """
    #     한 스텝의 학습 수행
        
    #     Args:
    #         batch: 배치 데이터
    #         step: 현재 스텝 번호
        
    #     Returns:
    #         학습 통계 딕셔너리
    #     """
    #     queries = batch["queries"]
        
    #     # 1. 토크나이즈
    #     query_tensors = self.tokenizer(
    #         queries,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=self.args.max_length,
    #     ).input_ids.to(self.args.device)
        
    #     # 2. 생성 (GRPO는 내부적으로 여러 샘플 생성)
    #     # num_sample_generations 만큼 각 query에 대해 생성
    #     generated_outputs = []
    #     response_tensors = []
        
    #     for query_tensor in query_tensors:
    #         query_tensor = query_tensor.unsqueeze(0)
            
    #         # 여러 샘플 생성
    #         gen_outputs = self.model.generate(
    #             query_tensor,
    #             max_new_tokens=self.args.max_new_tokens,
    #             do_sample=True,
    #             temperature=self.args.temperature,
    #             num_return_sequences=self.args.num_sample_generations,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id,
    #         )
            
    #         generated_outputs.extend(gen_outputs)
    #         response_tensors.extend([out[len(query_tensor[0]):] for out in gen_outputs])
        
    #     # 3. 디코딩
    #     generated_texts = self.tokenizer.batch_decode(
    #         generated_outputs,
    #         skip_special_tokens=True
    #     )
        
    #     # 프롬프트 제거
    #     generated_only = []
    #     for i, text in enumerate(generated_texts):
    #         query_idx = i // self.args.num_sample_generations
    #         query = queries[query_idx]
    #         if query in text:
    #             gen_part = text[len(query):].strip()
    #         else:
    #             gen_part = text.strip()
    #         generated_only.append(gen_part)
        
    #     # 4. 리워드 계산
    #     rewards = self.compute_rewards(batch, generated_only)
        
    #     # 5. GRPO 업데이트
    #     # GRPOTrainer의 step 메서드 호출
    #     stats = self.grpo_trainer.step(
    #         queries=query_tensors,
    #         responses=torch.stack(response_tensors),
    #         scores=rewards,
    #     )
        
    #     # 6. 추가 통계
    #     stats["step"] = step
    #     stats["mean_reward"] = rewards.mean().item()
    #     stats["std_reward"] = rewards.std().item()
    #     stats["max_reward"] = rewards.max().item()
    #     stats["min_reward"] = rewards.min().item()
        
    #     return stats
    
    # def evaluate(self, dataloader: DataLoader, split: str = "valid") -> Dict[str, float]:
    #     """
    #     평가 수행
        
    #     Args:
    #         dataloader: 평가용 dataloader
    #         split: 데이터 분할 이름
        
    #     Returns:
    #         평가 메트릭 딕셔너리
    #     """
    #     print(f"\n📊 Evaluating on {split} set...")
        
    #     self.model.eval()
    #     all_rewards = []
        
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             queries = batch["queries"]
                
    #             # 토크나이즈
    #             query_tensors = self.tokenizer(
    #                 queries,
    #                 return_tensors="pt",
    #                 padding=True,
    #                 truncation=True,
    #                 max_length=self.args.max_length,
    #             ).input_ids.to(self.args.device)
                
    #             # 생성 (평가 시에는 1개만)
    #             generated_outputs = self.model.generate(
    #                 query_tensors,
    #                 max_new_tokens=self.args.max_new_tokens,
    #                 do_sample=False,  # Greedy decoding
    #                 pad_token_id=self.tokenizer.pad_token_id,
    #                 eos_token_id=self.tokenizer.eos_token_id,
    #             )
                
    #             # 디코딩
    #             generated_texts = self.tokenizer.batch_decode(
    #                 generated_outputs,
    #                 skip_special_tokens=True
    #             )
                
    #             # 프롬프트 제거
    #             generated_only = []
    #             for i, text in enumerate(generated_texts):
    #                 query = queries[i]
    #                 if query in text:
    #                     gen_part = text[len(query):].strip()
    #                 else:
    #                     gen_part = text.strip()
    #                 generated_only.append(gen_part)
                
    #             # 리워드 계산
    #             rewards = self.reward_fn(
    #                 generated_texts=generated_only,
    #                 target_items=batch["targets"],
    #                 history_items=batch["histories"],
    #             )
                
    #             all_rewards.extend(rewards.cpu().numpy())
        
    #     self.model.train()
        
    #     # 메트릭 계산
    #     metrics = {
    #         f"{split}/mean_reward": np.mean(all_rewards),
    #         f"{split}/std_reward": np.std(all_rewards),
    #         f"{split}/max_reward": np.max(all_rewards),
    #         f"{split}/min_reward": np.min(all_rewards),
    #     }
        
    #     print(f"✓ {split.upper()} Evaluation:")
    #     for key, value in metrics.items():
    #         print(f"  {key}: {value:.4f}")
        
    #     return metrics
    
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
        
        # for epoch in range(self.args.num_epochs):
        #     print(f"\n📅 Epoch {epoch + 1}/{self.args.num_epochs}")
            
        #     epoch_rewards = []
            
        #     for batch_idx, batch in enumerate(self.train_dataloader):
        #         try:
        #             # 학습 스텝
        #             stats = self.train_step(batch, global_step)
        #             epoch_rewards.append(stats["mean_reward"])
                    
        #             # 로깅
        #             if global_step % self.args.log_interval == 0:
        #                 print(
        #                     f"Step {global_step:6d} | "
        #                     f"Epoch {epoch+1} Batch {batch_idx:4d} | "
        #                     f"Reward: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f} | "
        #                     f"Loss: {stats.get('loss', 0.0):.4f}"
        #                 )
                    
        #             # 검증 평가
        #             if global_step > 0 and global_step % self.args.eval_interval == 0:
        #                 valid_metrics = self.evaluate(self.valid_dataloader, split="valid")
                        
        #                 # Best 모델 저장
        #                 valid_reward = valid_metrics["valid/mean_reward"]
        #                 if valid_reward > best_reward:
        #                     best_reward = valid_reward
        #                     best_path = self.checkpoint_dir / "checkpoint_best"
        #                     self.model.save_pretrained(best_path)
        #                     self.tokenizer.save_pretrained(best_path)
        #                     print(f"🏆 Best model saved: {best_path} (reward: {best_reward:.4f})")
                    
        #             # 체크포인트 저장
        #             if global_step > 0 and global_step % self.args.save_interval == 0:
        #                 checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{global_step}"
        #                 self.model.save_pretrained(checkpoint_path)
        #                 self.tokenizer.save_pretrained(checkpoint_path)
        #                 print(f"💾 Checkpoint saved: {checkpoint_path}")
                    
        #             global_step += 1
                    
        #             if global_step >= self.args.max_steps:
        #                 break
                
        #         except KeyboardInterrupt:
        #             print("\n⚠️  Training interrupted by user")
        #             break
        #         except Exception as e:
        #             print(f"\n❌ Error at step {global_step}: {e}")
        #             if self.args.debug:
        #                 raise e
        #             continue
            
        #     # Epoch 종료 평가
        #     print(f"\n📊 Epoch {epoch + 1} Summary:")
        #     avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        #     print(f"  Train Average Reward: {avg_epoch_reward:.4f}")
            
        #     # Valid 평가
        #     valid_metrics = self.evaluate(self.valid_dataloader, split="valid")
            
        #     if global_step >= self.args.max_steps:
        #         break
        
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
