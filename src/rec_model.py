# rec_model.py
"""
Recommendation Policy Model for RL4Rec
사용자 히스토리를 기반으로 아이템 추천 텍스트를 생성하는 Policy Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional
import numpy as np
from vllm import LLM, SamplingParams


class RecPolicy(nn.Module):
    """
    추천 시스템을 위한 Policy Model
    사용자 히스토리와 컨텍스트를 기반으로 추천 설명 텍스트를 생성
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-5,
        max_length: int = 512,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        
        # LLM 모델 및 토크나이저 로드
        self._load_model()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 학습 통계
        self.training_stats = {
            "total_steps": 0,
            "avg_reward": 0.0,
            "loss": 0.0,
        }
        
        print(f"✓ RecPolicy initialized on {self.device}")
        print(f"  Model: {self.model_name}")
        print(f"  Max length: {self.max_length}")
    
    def _load_model(self):
        """LLM 모델과 토크나이저 로드"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model: {self.model_name}...")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Pad token 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # vLLM 모델 로드 (추론용)
        print("Loading vLLM model for inference...")
        self.vllm_model = LLM(
            model=self.model_name,
            dtype="float16" if self.device.startswith("cuda") else "float32",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        
        # Transformers 모델 로드 (학습용)
        print("Loading transformers model for training...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        ).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
    
    def build_prompts(
        self,
        user_histories: List[List[str]],
        target_task: str = "recommendation"
    ) -> List[str]:
        """
        사용자 히스토리를 기반으로 프롬프트 생성
        
        Args:
            user_histories: 사용자별 아이템 히스토리 리스트
            target_task: 태스크 유형 (recommendation, review_generation 등)
        
        Returns:
            프롬프트 텍스트 리스트
        """
        prompts = []
        
        for history in user_histories:
            if target_task == "recommendation":
                history_str = ", ".join(history[-5:])  # 최근 5개 아이템
                prompt = (
                    f"User's purchase history: {history_str}\n"
                    f"Based on this history, describe an ideal next product for this user:\n"
                )
            else:
                prompt = f"Generate a product description based on: {history}\n"
            
            prompts.append(prompt)
        
        return prompts
    
    def generate(
        self,
        prompts: List[str],
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        프롬프트로부터 텍스트 생성 (vLLM 사용)
        
        Args:
            prompts: 입력 프롬프트 리스트
            do_sample: 샘플링 여부
            num_return_sequences: 프롬프트당 생성할 시퀀스 개수
        
        Returns:
            생성된 텍스트 리스트
        """
        # vLLM SamplingParams 설정
        sampling_params = SamplingParams(
            temperature=self.temperature if do_sample else 0.0,
            top_p=0.95 if do_sample else 1.0,
            max_tokens=64,
            n=num_return_sequences,
            skip_special_tokens=True,
        )
        
        # vLLM으로 생성
        outputs = self.vllm_model.generate(prompts, sampling_params)
        
        # 결과 추출
        results = []
        for output in outputs:
            for generated_output in output.outputs:
                generated_text = generated_output.text.strip()
                results.append(generated_text)
        
        return results
    
    def compute_loss(
        self,
        prompts: List[str],
        generated_texts: List[str],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Policy Gradient Loss 계산
        
        Args:
            prompts: 입력 프롬프트
            generated_texts: 생성된 텍스트
            rewards: 각 생성에 대한 보상
        
        Returns:
            Loss tensor
        """
        self.model.train()
        
        # 전체 텍스트 준비 (prompt + generated)
        full_texts = [p + g for p, g in zip(prompts, generated_texts)]
        
        # 토크나이즈
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length + 64,
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        
        # Log probabilities
        log_probs = -outputs.loss
        
        # Normalize rewards
        rewards = rewards.to(self.device)
        if rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Policy gradient loss: -log_prob * reward
        loss = -(log_probs * rewards.mean())
        
        return loss
    
    def update(
        self,
        prompts: List[str],
        generated_texts: List[str],
        rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """
        모델 파라미터 업데이트
        
        Args:
            prompts: 입력 프롬프트
            generated_texts: 생성된 텍스트
            rewards: 보상 텐서
        
        Returns:
            학습 통계 딕셔너리
        """
        # Loss 계산
        loss = self.compute_loss(prompts, generated_texts, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update
        self.optimizer.step()
        
        # 통계 업데이트
        self.training_stats["total_steps"] += 1
        self.training_stats["avg_reward"] = rewards.mean().item()
        self.training_stats["loss"] = loss.item()
        
        return {
            "loss": loss.item(),
            "avg_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
        }
    
    def save_checkpoint(self, path: str):
        """체크포인트 저장"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "config": {
                "model_name": self.model_name,
                "max_length": self.max_length,
                "temperature": self.temperature,
            }
        }
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]
        print(f"✓ Checkpoint loaded: {path}")
        print(f"  Steps: {self.training_stats['total_steps']}")


class DummyRecPolicy:
    """
    테스트용 Dummy Policy Model
    실제 LLM 없이 빠르게 테스트할 수 있는 버전
    """
    
    def __init__(self):
        self.step = 0
        print("✓ DummyRecPolicy initialized (for testing)")
    
    def build_prompts(self, user_histories: List[List[str]]) -> List[str]:
        """Dummy 프롬프트 생성"""
        prompts = []
        for history in user_histories:
            history_str = ", ".join(history[-3:])
            prompts.append(f"Recommend based on: {history_str}")
        return prompts
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Dummy 텍스트 생성"""
        return [
            f"A great product with excellent features and quality. Step {self.step}. {p[:30]}"
            for p in prompts
        ]
    
    def update(self, prompts: List[str], generated_texts: List[str], rewards: torch.Tensor) -> Dict[str, float]:
        """Dummy 업데이트"""
        self.step += 1
        return {
            "loss": 0.5 - self.step * 0.001,
            "avg_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
        }
    
    def save_checkpoint(self, path: str):
        """Dummy 저장"""
        print(f"✓ Dummy checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Dummy 로드"""
        print(f"✓ Dummy checkpoint loaded: {path}")

