#!/usr/bin/env python3
"""
TRL 라이브러리를 활용한 RL 학습 스크립트
PPO 알고리즘과 NDCG 리워드를 사용한 추천 시스템 학습
"""

import os
import ray
import torch
import argparse
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from utils.reward_funtion import create_reward_function


class RecommendationDataset(Dataset):
    """
    추천 시스템용 데이터셋
    사용자 히스토리, 타겟 아이템, 프롬프트를 포함
    """
    
    def __init__(
        self,
        prompt_file: str,
        sequential_file: str,
        tokenizer,
        max_length: int = 128,
    ):
        """
        Args:
            prompt_file: 프롬프트 JSON 파일 경로
            sequential_file: 시퀀셜 데이터 파일 경로 (user_id history target)
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 프롬프트 로드
        with open(prompt_file, "r") as f:
            prompts_dict = json.load(f)
            self.prompts = {int(k): v for k, v in prompts_dict.items()}
        
        # 시퀀셜 데이터 로드
        self.history_dict = {}
        self.target_dict = {}
        
        with open(sequential_file, "r") as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                history = parts[1:-1]
                target = parts[-1]
                self.history_dict[user_id] = history
                self.target_dict[user_id] = target
        
        # 공통 user_id만 사용
        self.user_ids = sorted(
            set(self.prompts.keys()) & 
            set(self.history_dict.keys()) & 
            set(self.target_dict.keys())
        )
        
        print(f"✓ Dataset loaded: {len(self.user_ids)} users")
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        
        prompt = self.prompts[user_id]
        history = self.history_dict[user_id]
        target = self.target_dict[user_id]
        
        # 토크나이즈
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "prompt": prompt,
            "history": history,
            "target": target,
            "user_id": user_id,
        }


def collate_fn(batch):
    """
    DataLoader용 collate function
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
        "histories": [item["history"] for item in batch],
        "targets": [item["target"] for item in batch],
        "user_ids": [item["user_id"] for item in batch],
    }


class TRLTrainer:
    """
    TRL PPO를 활용한 추천 시스템 학습기
    """
    
    def __init__(self, args):
        self.args = args
        
        # Ray 초기화 (이미 되어있으면 skip)
        if not ray.is_initialized():
            print(f"🔧 Initializing Ray...")
            ray.init(address=args.ray_address, namespace=args.namespace)
            print(f"✓ Ray initialized")
        
        # 토크나이저 로드
        print(f"📚 Loading tokenizer: {args.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드 (with value head for PPO)
        print(f"🤖 Loading model: {args.model_name}")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        )
        self.model.to(args.device)
        
        # PPO Config
        ppo_config = PPOConfig(
            model_name=args.model_name,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            mini_batch_size=args.mini_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ppo_epochs=args.ppo_epochs,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            log_with=args.log_with,
            tracker_project_name=args.project_name,
        )
        
        # PPO Trainer
        print(f"🎯 Initializing PPO Trainer...")
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        # 리워드 함수
        print(f"💰 Creating reward function: {args.reward_type}@{args.k}")
        self.reward_fn = create_reward_function(
            retrieval_service_name=args.retrieval_service_name,
            namespace=args.namespace,
            dataset_name=args.dataset_name,
            reward_type=args.reward_type,
            k=args.k,
        )
        
        # 데이터셋
        print(f"📊 Loading dataset...")
        self.dataset = RecommendationDataset(
            prompt_file=args.prompt_file,
            sequential_file=args.sequential_file,
            tokenizer=self.tokenizer,
            max_length=args.max_length,
        )
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ TRLTrainer initialized")
    
    def generate_rollouts(self, batch):
        """
        배치에 대한 rollout 생성
        """
        input_ids = batch["input_ids"].to(self.args.device)
        attention_mask = batch["attention_mask"].to(self.args.device)
        
        # vLLM을 사용하는 경우 (rec_model.py의 generate 사용)
        # 여기서는 TRL의 기본 생성 메서드 사용
        generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
            "do_sample": True,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # 생성
        response_tensors = self.ppo_trainer.generate(
            input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )
        
        # 디코딩
        generated_texts = self.tokenizer.batch_decode(
            response_tensors,
            skip_special_tokens=True
        )
        
        # 원본 프롬프트 제거
        prompts = batch["prompts"]
        generated_only = []
        for i, text in enumerate(generated_texts):
            # 프롬프트 이후 생성된 부분만 추출
            if prompts[i] in text:
                gen_part = text[len(prompts[i]):].strip()
            else:
                gen_part = text.strip()
            generated_only.append(gen_part)
        
        return response_tensors, generated_only
    
    def train_step(self, batch, step):
        """
        한 스텝의 PPO 학습
        """
        # 1. Rollout 생성
        response_tensors, generated_texts = self.generate_rollouts(batch)
        
        # 2. 리워드 계산 (NDCG 기반)
        rewards = self.reward_fn(
            generated_texts=generated_texts,
            target_items=batch["targets"],
            history_items=batch["histories"],
        )
        
        # 3. PPO 학습
        input_ids = batch["input_ids"].to(self.args.device)
        
        stats = self.ppo_trainer.step(
            queries=input_ids,
            responses=response_tensors,
            scores=rewards,
        )
        
        # 4. 추가 메트릭
        stats["step"] = step
        stats["mean_reward"] = rewards.mean().item()
        stats["std_reward"] = rewards.std().item()
        stats["max_reward"] = rewards.max().item()
        stats["min_reward"] = rewards.min().item()
        
        return stats
    
    def train(self):
        """
        전체 학습 루프
        """
        print("=" * 80)
        print("🚀 Starting TRL PPO Training")
        print("=" * 80)
        
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        global_step = 0
        
        for epoch in range(self.args.num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.args.num_epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # 학습 스텝
                    stats = self.train_step(batch, global_step)
                    
                    # 로깅
                    if global_step % self.args.log_interval == 0:
                        print(
                            f"Step {global_step:6d} | "
                            f"Epoch {epoch+1} Batch {batch_idx} | "
                            f"Reward: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f} | "
                            f"PPO Loss: {stats.get('ppo/loss/total', 0.0):.4f}"
                        )
                    
                    # 체크포인트 저장
                    if global_step > 0 and global_step % self.args.save_interval == 0:
                        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{global_step}"
                        self.ppo_trainer.save_pretrained(checkpoint_path)
                        print(f"💾 Checkpoint saved: {checkpoint_path}")
                    
                    global_step += 1
                    
                    if global_step >= self.args.max_steps:
                        break
                
                except KeyboardInterrupt:
                    print("\n⚠️  Training interrupted by user")
                    break
                except Exception as e:
                    print(f"\n❌ Error at step {global_step}: {e}")
                    if self.args.debug:
                        raise e
                    continue
            
            if global_step >= self.args.max_steps:
                break
        
        # 최종 체크포인트
        final_checkpoint = self.checkpoint_dir / "checkpoint_final"
        self.ppo_trainer.save_pretrained(final_checkpoint)
        print(f"💾 Final checkpoint saved: {final_checkpoint}")
        
        print("=" * 80)
        print("✓ Training completed!")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="TRL PPO Training for Recommendation")
    
    # Ray & Service
    parser.add_argument("--ray_address", type=str, default="auto")
    parser.add_argument("--namespace", type=str, default="rl4rec")
    parser.add_argument("--retrieval_service_name", type=str, default="RetrievalService")
    parser.add_argument("--dataset_name", type=str, default="beauty")
    
    # Model
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # Data
    parser.add_argument("--prompt_file", type=str, 
                        default="data_processed/beauty_gemma-3-1b-it_test_user_preference.json")
    parser.add_argument("--sequential_file", type=str,
                        default="data/beauty/sequential_data.txt")
    
    # PPO Training
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=10000)
    
    # Reward
    parser.add_argument("--reward_type", type=str, default="ndcg",
                        choices=["ndcg", "hit", "mrr", "mixed"])
    parser.add_argument("--k", type=int, default=10, help="Top-K for metrics")
    
    # Logging & Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/trl_ppo")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_with", type=str, default=None, 
                        help="Logging backend (wandb, tensorboard, etc.)")
    parser.add_argument("--project_name", type=str, default="rl4rec")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Trainer 초기화 및 학습
    trainer = TRLTrainer(args)
    trainer.train()
    
    print("✓ Done!")


if __name__ == "__main__":
    main()

