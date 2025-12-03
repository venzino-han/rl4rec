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
        vLLM을 사용해서 텍스트 생성 후 item embeddings와 유사도 계산
        """
        print(f"\n{'='*80}")
        print(f"📊 Final Evaluation on {split.upper()} Set")
        print(f"{'='*80}")
        
        # 0. reset GPU memory
        self.model.to(torch.device("cpu"))
        self.grpo_trainer.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        gc.collect()

        # 1. vLLM으로 모델 로드
        print("🤖 Loading model with vLLM for generation...")
        llm = LLM(
            model=self.args.checkpoint_dir,
            tensor_parallel_size=1,
            dtype=torch.bfloat16 if self.args.bf16 else torch.float16,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            max_model_len=self.args.max_length,
            max_num_seqs=self.args.eval_batch_size,
        )
        
        # 2. Embedding model 로드 (retrieval_service 참고)
        print("🔍 Loading embedding model for retrieval...")
        emb_model_name = getattr(self.args, 'emb_model_name', 'mixedbread-ai/mxbai-embed-large-v1')
        emb_model_name_dir = emb_model_name.split('/')[-1]
        emb_type = getattr(self.args, 'emb_type', 'title')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pooling 파라미터 설정
        pooling_params = PoolingParams(
            truncate_prompt_tokens=512,
            task="embed",
        )
        
        emb_llm = LLM(
            model=emb_model_name,
            task="embed",
            enforce_eager=True,
            gpu_memory_utilization=0.3,
            trust_remote_code=True,
            max_model_len=512,
            max_num_seqs=512,
        )
        
        # 3. Item embeddings 로드
        print("📦 Loading item embeddings...")
        emb_file = f"data_emb/{self.args.dataset_name}_{emb_type}_{emb_model_name_dir}.pt"
        item_embeddings = torch.load(emb_file, map_location=device)
        item_embeddings = item_embeddings / item_embeddings.norm(dim=-1, keepdim=True)
        num_items = item_embeddings.shape[0]
        print(f"✓ Loaded item embeddings: {item_embeddings.shape}")
        
        # 4. 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            n=1,
            temperature=0.01,
            max_tokens=self.args.max_new_tokens,
            stop=["<|eot_id|>", "<|reserved_special_token_0|>", "<eos>"]
        )
        
        # 5. 데이터셋 순회하며 평가
        all_prompts = []
        all_targets = []
        all_histories = []
        
        print("📝 Collecting prompts...")
        for i in tqdm(range(len(dataset)), desc="Preparing data"):
            sample = dataset[i]
            all_prompts.append(sample["prompt"])
            all_targets.append(sample["target"])
            all_histories.append(sample["history"])
        
        # 6. Batch 단위로 생성 및 평가
        batch_size = self.args.eval_batch_size
        num_batches = (len(all_prompts) + batch_size - 1) // batch_size
        
        # 메트릭 저장용
        metrics = {
            'hit@5': [], 'hit@10': [], 'hit@20': [],
            'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': []
        }
        
        print(f"🚀 Generating responses and computing metrics...")
        
        # 샘플 출력을 위한 저장소
        sample_outputs = []
        
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_prompts))
            
            batch_prompts = all_prompts[start_idx:end_idx]
            batch_targets = all_targets[start_idx:end_idx]
            batch_histories = all_histories[start_idx:end_idx]
            
            # 텍스트 생성
            outputs = llm.generate(batch_prompts, sampling_params)
            generated_texts = [output.outputs[0].text for output in outputs]
            
            # 첫 번째 배치에서 샘플 저장 (최대 3개)
            if batch_idx == 0:
                num_samples = min(3, len(batch_prompts))
                for i in range(num_samples):
                    sample_outputs.append({
                        'prompt': batch_prompts[i],
                        'generated': generated_texts[i],
                        'target': batch_targets[i],
                        'history': batch_histories[i]
                    })
            
            # Embedding 계산
            emb_outputs = emb_llm.encode(
                prompts=generated_texts,
                pooling_task="embed",
                pooling_params=pooling_params,
                use_tqdm=False,
            )
            
            # Query embeddings 추출
            embeddings_list = [
                torch.as_tensor(out.outputs.data, dtype=torch.float32, device=device)
                for out in emb_outputs
            ]
            query_embeddings = torch.stack(embeddings_list)
            query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)
            
            # 전체 아이템과의 유사도 계산
            scores = torch.matmul(query_embeddings, item_embeddings.T)  # [batch, num_items]
            
            # 과거 구매 아이템 제외
            for i in range(len(batch_targets)):
                if batch_histories[i]:
                    # 히스토리 아이템 마스킹 (target은 제외)
                    history_indices = [idx for idx in batch_histories[i] if idx != batch_targets[i]]
                    if history_indices:
                        scores[i, history_indices] = -float('inf')
            
            # k=(5, 10, 20)에 대해 메트릭 계산
            for k in [5, 10, 20]:
                # NDCG 계산
                ndcg_scores = calculate_ndcg(
                    scores,
                    batch_targets,
                    batch_histories,
                    k=k,
                    use_negatives_only=False
                )
                metrics[f'ndcg@{k}'].extend(ndcg_scores.cpu().tolist())
                
                # Hit 계산
                hit_scores = calculate_hit_rate(
                    scores,
                    batch_targets,
                    batch_histories,
                    k=k,
                    use_negatives_only=False
                )
                metrics[f'hit@{k}'].extend(hit_scores.cpu().tolist())
        
        # 7. 샘플 프롬프트와 생성 결과 출력
        print(f"\n{'='*80}")
        print(f"📝 Sample Prompts and Generated Texts")
        print(f"{'='*80}")
        
        for idx, sample in enumerate(sample_outputs, 1):
            print(f"\n[Sample {idx}]")
            print(f"{'─'*80}")
            print(f"Target Item ID: {sample['target']}")
            print(f"History Items: {sample['history']}")
            print(f"\n[Prompt]")
            # 프롬프트가 길면 앞 300자만 출력
            prompt_preview = sample['prompt'][:300] + "..." if len(sample['prompt']) > 300 else sample['prompt']
            print(prompt_preview)
            print(f"\n[Generated Text]")
            print(sample['generated'])
            print(f"{'─'*80}")
        
        # 8. 최종 메트릭 출력
        print(f"\n{'='*80}")
        print(f"📈 Final Evaluation Results ({split.upper()})")
        print(f"{'='*80}")
        
        results_summary = []
        for metric_name in ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
            mean_val = np.mean(metrics[metric_name])
            result_line = f"  {metric_name.upper()}: {mean_val:.4f}"
            print(result_line)
            results_summary.append(result_line)
        
        print(f"{'='*80}\n")
        
        # 9. 결과를 .log 파일로 저장
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 타임스탬프 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.args.run_name}_{split}_eval_{timestamp}.log"
        log_file = results_dir / log_filename
        
        with open(log_file, 'w') as f:
            # 헤더
            f.write("="*80 + "\n")
            f.write(f"Evaluation Results - {split.upper()}\n")
            f.write(f"Run Name: {self.args.run_name}\n")
            f.write(f"Dataset: {self.args.dataset_name}\n")
            f.write(f"Model: {self.args.policy_model}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("="*80 + "\n\n")
            
            # 메트릭 결과
            f.write("EVALUATION METRICS\n")
            f.write("-"*80 + "\n")
            for line in results_summary:
                f.write(line + "\n")
            f.write("-"*80 + "\n\n")
            
            # 상세 통계
            f.write("DETAILED STATISTICS\n")
            f.write("-"*80 + "\n")
            for metric_name in ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
                values = metrics[metric_name]
                f.write(f"{metric_name.upper()}:\n")
                f.write(f"  Mean: {np.mean(values):.4f}\n")
                f.write(f"  Std:  {np.std(values):.4f}\n")
                f.write(f"  Min:  {np.min(values):.4f}\n")
                f.write(f"  Max:  {np.max(values):.4f}\n")
                f.write("\n")
            f.write("-"*80 + "\n\n")
            
            # 샘플 출력
            f.write("SAMPLE PROMPTS AND GENERATED TEXTS\n")
            f.write("="*80 + "\n")
            for idx, sample in enumerate(sample_outputs, 1):
                f.write(f"\n[Sample {idx}]\n")
                f.write("-"*80 + "\n")
                f.write(f"Target Item ID: {sample['target']}\n")
                f.write(f"History Items: {sample['history']}\n")
                f.write(f"\n[Prompt]\n")
                f.write(sample['prompt'] + "\n")
                f.write(f"\n[Generated Text]\n")
                f.write(sample['generated'] + "\n")
                f.write("-"*80 + "\n")
        
        print(f"💾 Evaluation results saved to: {log_file}")
        
        # 결과 딕셔너리 반환
        results = {
            metric_name: float(np.mean(values))
            for metric_name, values in metrics.items()
        }
        
        return results
    
    def train(self):
        """
        전체 학습 루프 실행
        """
        print("=" * 80)
        print("🚀 Starting GRPO Training")
        print("=" * 80)
        
        self.grpo_trainer.train()
        print("=" * 80)
        print("✓ Training completed!")
        print("=" * 80)
        
        # 최종 테스트 평가
        test_metrics = self.evaluate_final_metrics(self.test_dataset, split="test")
        


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
