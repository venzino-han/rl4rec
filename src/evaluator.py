"""
Evaluation class for recommendation model assessment
"""
import gc
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.pooling_params import PoolingParams

from utils.reward_funtion import calculate_ndcg, calculate_hit_rate


class RecommendationEvaluator:
    """
    추천 모델 평가를 위한 클래스
    vLLM을 사용하여 텍스트 생성 및 임베딩 기반 검색 수행
    """
    
    def __init__(self, args, checkpoint_dir):
        """
        Args:
            args: 학습 설정 파라미터
            checkpoint_dir: 평가할 모델 체크포인트 경로
        """
        self.args = args
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # vLLM 모델들 (lazy loading)
        self.llm = None
        self.emb_llm = None
        self.item_embeddings = None
        
    def _load_generation_model(self):
        """텍스트 생성을 위한 vLLM 모델 로드"""
        if self.llm is None:
            print("🤖 Loading model with vLLM for generation...")
            self.sampling_params = SamplingParams(
                # n=1,
                temperature=0.01,
                max_tokens=self.args.max_new_tokens,
                repetition_penalty=1.1,
                # stop=["<|eot_id|>", "<|reserved_special_token_0|>", "<eos>"],
            )
        
            self.llm = LLM(
                model=self.checkpoint_dir,
                tensor_parallel_size=1,
                dtype=torch.bfloat16,
                # dtype=torch.bfloat16 if self.args.bf16 else torch.float32,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                max_model_len=self.args.max_length,
                max_num_seqs=self.args.eval_batch_size,
            )
    
    def _load_embedding_model(self):
        """임베딩 계산을 위한 vLLM 모델 로드"""
        if self.emb_llm is None:
            print("🔍 Loading embedding model for retrieval...")
            emb_model_name = getattr(self.args, 'emb_model_name', 'mixedbread-ai/mxbai-embed-large-v1')
            
            # Pooling 파라미터 설정
            self.pooling_params = PoolingParams(
                truncate_prompt_tokens=self.args.eval_emb_max_length,
                task="embed",
            )
            
            self.emb_llm = LLM(
                model=emb_model_name,
                task="embed",
                enforce_eager=True,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=self.args.eval_emb_max_length,
                max_num_seqs=self.args.eval_emb_batch_size,
            )
    
    def _load_item_embeddings(self):
        """아이템 임베딩 로드"""
        if self.item_embeddings is None:
            print("📦 Loading item embeddings...")
            emb_model_name = getattr(self.args, 'emb_model_name', 'mixedbread-ai/mxbai-embed-large-v1')
            emb_model_name_dir = emb_model_name.split('/')[-1]
            emb_type = getattr(self.args, 'emb_type', 'title')
            
            emb_file = f"data_emb/{self.args.data_name}_{emb_type}_{emb_model_name_dir}.pt"
            self.item_embeddings = torch.load(emb_file, map_location=self.device)
            self.item_embeddings = self.item_embeddings / self.item_embeddings.norm(dim=-1, keepdim=True)
            print(f"✓ Loaded item embeddings: {self.item_embeddings.shape}")
    
    def generate_all_texts(self, prompts):
        """
        모든 프롬프트에 대해 텍스트 생성
        
        Args:
            prompts: 생성할 프롬프트 리스트
            batch_size: 배치 크기 (None이면 args.eval_batch_size 사용)
        
        Returns:
            generated_texts: 생성된 텍스트 리스트
        """
        self._load_generation_model()
        # 샘플링 파라미터 설정
        print(f"🚀 Generating responses for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts
    
    def compute_embeddings(self, texts):
        """
        텍스트들에 대한 임베딩을 한번에 계산
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기 (None이면 args.eval_emb_batch_size 사용)
        
        Returns:
            query_embeddings: 정규화된 쿼리 임베딩 [num_texts, emb_dim]
        """
        self._load_embedding_model()        
        print(f"🔍 Computing embeddings for {len(texts)} texts...")
        # add cls token
        texts = [f"[CLS] {text}" for text in texts]

        # Embedding 계산
        emb_outputs = self.emb_llm.encode(
            prompts=texts,
            pooling_task="embed",
            pooling_params=self.pooling_params,
            use_tqdm=True,
        )
        # Query embeddings 추출
        embeddings_list = [
            torch.as_tensor(out.outputs.data, dtype=torch.float32, device=self.device)
            for out in emb_outputs
        ]
        query_embeddings = torch.stack(embeddings_list)
        return query_embeddings
    
    def compute_retrieval_metrics(self, query_embeddings, targets, histories, ks=[5, 10, 20]):
        """
        검색 메트릭 계산 (배치별 처리)
        
        Args:
            query_embeddings: 쿼리 임베딩 [num_queries, emb_dim]
            targets: 타겟 아이템 ID 리스트
            histories: 히스토리 아이템 ID 리스트의 리스트
            ks: 평가할 k 값들
        
        Returns:
            metrics: 메트릭 딕셔너리
            rank_info: 각 샘플의 rank 및 score 정보 리스트
        """
        self._load_item_embeddings()
        
        print(f"📊 Computing retrieval metrics...")
        
        # 메트릭 저장용 딕셔너리 초기화
        metrics = {f'ndcg@{k}': [] for k in ks}
        metrics.update({f'hit@{k}': [] for k in ks})
        
        # Rank 정보 저장용 리스트
        rank_info = []
        
        # 배치 크기 설정
        batch_size = self.args.eval_batch_size
        num_queries = query_embeddings.shape[0]
        num_batches = (num_queries + batch_size - 1) // batch_size
        
        print(f"Processing {num_queries} queries in {num_batches} batches...")
        
        for batch_idx in tqdm(range(num_batches), desc="Computing metrics"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_queries)
            
            # 배치 데이터 추출
            batch_query_emb = query_embeddings[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            batch_histories = histories[start_idx:end_idx]
            
            # 전체 아이템과의 유사도 계산
            batch_scores = torch.matmul(batch_query_emb, self.item_embeddings.T)  # [batch_size, num_items]
            
            # 과거 구매 아이템 제외
            for i in range(len(batch_targets)):
                history_indices = [idx for idx in batch_histories[i] if idx != batch_targets[i]]
                if history_indices:
                    batch_scores[i, history_indices] = -float('inf')
            
            # 각 샘플의 rank 계산
            for i in range(len(batch_targets)):
                target_item = batch_targets[i]
                target_score = batch_scores[i, target_item].item()
                
                # 타겟 아이템보다 높은 점수를 가진 아이템의 개수 = rank - 1
                rank = (batch_scores[i] > target_score).sum().item() + 1
                
                rank_info.append({
                    'target_item': target_item,
                    'target_score': target_score,
                    'rank': rank,
                })
            
            # 각 k에 대해 메트릭 계산
            for k in ks:
                # NDCG 계산
                ndcg_scores = calculate_ndcg(
                    batch_scores,
                    batch_targets,
                    batch_histories,
                    k=k,
                    use_negatives_only=False
                )
                metrics[f'ndcg@{k}'].extend(ndcg_scores.cpu().tolist())
                
                # Hit 계산
                hit_scores = calculate_hit_rate(
                    batch_scores,
                    batch_targets,
                    batch_histories,
                    k=k,
                    use_negatives_only=False
                )
                metrics[f'hit@{k}'].extend(hit_scores.cpu().tolist())
        
        return metrics, rank_info
    
    def evaluate(self, dataset, split="test", save_log=True):
        """
        전체 평가 파이프라인 실행
        
        Args:
            dataset: 평가할 데이터셋
            split: 데이터셋 split 이름 ("test", "val" 등)
            save_log: 로그 파일 저장 여부
        
        Returns:
            results: 평가 결과 딕셔너리
        """
        print(f"\n{'='*80}")
        print(f"📊 Final Evaluation on {split.upper()} Set")
        print(f"{'='*80}")
        
        # 1. 데이터 수집
        print("📝 Collecting data...")
        all_prompts = []
        all_targets = []
        all_histories = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            all_prompts.append(sample["prompt"])
            all_targets.append(sample["target"])
            all_histories.append(sample["history"])

        if len(all_prompts) > self.args.eval_samples:
            all_prompts = all_prompts[:self.args.eval_samples]
            all_targets = all_targets[:self.args.eval_samples]
            all_histories = all_histories[:self.args.eval_samples]
        
        # 2. 텍스트 생성 (모든 프롬프트에 대해)
        generated_texts = self.generate_all_texts(all_prompts)

        # 생성 모델 즉시 정리 (메모리 절약)
        print("\n🧹 Cleaning up generation model...")
        if self.llm is not None:
            try:
                if hasattr(self.llm, 'llm_engine'):
                    del self.llm.llm_engine
            except:
                pass
            del self.llm
            self.llm = None
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            print(f"💾 GPU Memory after generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 3. 임베딩 계산 (모든 생성된 텍스트에 대해)
        query_embeddings = self.compute_embeddings(generated_texts)
        
        # 임베딩 모델도 정리 (메트릭 계산 전)
        print("\n🧹 Cleaning up embedding model...")
        if self.emb_llm is not None:
            try:
                if hasattr(self.emb_llm, 'llm_engine'):
                    del self.emb_llm.llm_engine
            except:
                pass
            del self.emb_llm
            self.emb_llm = None
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            print(f"💾 GPU Memory after embedding: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 4. 메트릭 계산
        metrics, rank_info = self.compute_retrieval_metrics(
            query_embeddings, 
            all_targets, 
            all_histories,
            ks=[5, 10, 20]
        )
        
        # 5. 샘플 출력 저장 (처음 3개)
        sample_outputs = []
        num_samples = min(3, len(all_prompts))
        for i in range(num_samples):
            sample_outputs.append({
                'prompt': all_prompts[i],
                'generated': generated_texts[i],
                'target': all_targets[i],
                'history': all_histories[i]
            })
        
        # 6. 결과 출력
        self._print_sample_outputs(sample_outputs)
        results = self._print_metrics(metrics, split)
        
        # 7. 로그 파일 저장
        if save_log:
            self._save_log_file(results, metrics, sample_outputs, split)
            # CSV 파일 저장
            self._save_csv_file(
                all_prompts, 
                generated_texts, 
                all_targets, 
                all_histories,
                rank_info, 
                split
            )
        
        return results
    
    def _print_sample_outputs(self, sample_outputs):
        """샘플 프롬프트와 생성 결과 출력"""
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
    
    def _print_metrics(self, metrics, split):
        """메트릭 출력 및 평균 계산"""
        print(f"\n{'='*80}")
        print(f"📈 Final Evaluation Results ({split.upper()})")
        print(f"{'='*80}")
        
        results = {}
        for metric_name in ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
            mean_val = np.mean(metrics[metric_name])
            results[metric_name] = float(mean_val)
            print(f"  {metric_name.upper()}: {mean_val:.4f}")
        
        print(f"{'='*80}\n")
        
        return results
    
    def _save_log_file(self, results, metrics, sample_outputs, split):
        """평가 결과를 로그 파일로 저장"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 타임스탬프 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.args.run_name}_{split}_eval.log"
        log_file = results_dir / log_filename
        
        with open(log_file, 'a') as f:
            # 헤더
            f.write("="*80 + "\n")
            f.write(f"Evaluation Results - {split.upper()}\n")
            f.write(f"Run Name: {self.args.run_name}\n")
            f.write(f"Dataset: {self.args.data_name}\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("="*80 + "\n\n")
            
            # 메트릭 결과
            f.write("EVALUATION METRICS\n")
            f.write("-"*80 + "\n")
            for metric_name, value in results.items():
                f.write(f"  {metric_name.upper()}: {value:.4f}\n")
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
    
    def _save_csv_file(self, prompts, generated_texts, targets, histories, rank_info, split):
        """평가 결과를 CSV 파일로 저장"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 타임스탬프 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{self.args.run_name}_{split}_eval_{timestamp}.csv"
        csv_file = results_dir / csv_filename
        
        # DataFrame 생성
        data = []
        for i in range(len(prompts)):
            data.append({
                'sample_id': i,
                'prompt': prompts[i],
                'generated_text': generated_texts[i],
                'target_item': targets[i],
                'history_items': str(histories[i]),  # 리스트를 문자열로 변환
                'target_score': rank_info[i]['target_score'],
                'rank': rank_info[i]['rank'],
                'hit@5': 1 if rank_info[i]['rank'] <= 5 else 0,
                'hit@10': 1 if rank_info[i]['rank'] <= 10 else 0,
                'hit@20': 1 if rank_info[i]['rank'] <= 20 else 0,
            })
        
        df = pd.DataFrame(data)
        
        # CSV 저장
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"💾 Detailed results saved to CSV: {csv_file}")
        
        # 간단한 통계 출력
        print(f"\n📈 CSV Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Mean rank: {df['rank'].mean():.2f}")
        print(f"  Median rank: {df['rank'].median():.0f}")
        print(f"  Hit@5 rate: {df['hit@5'].mean():.4f}")
        print(f"  Hit@10 rate: {df['hit@10'].mean():.4f}")
        print(f"  Hit@20 rate: {df['hit@20'].mean():.4f}")
    
    def cleanup(self):
        """메모리 정리"""
        print("\n🧹 Cleaning up evaluator resources...")
        
        # vLLM 생성 모델 정리
        if self.llm is not None:
            try:
                # vLLM의 경우 llm_engine을 명시적으로 정리
                if hasattr(self.llm, 'llm_engine'):
                    del self.llm.llm_engine
            except:
                pass
            del self.llm
            self.llm = None
            print("  ✓ Generation model cleaned up")
        
        # vLLM 임베딩 모델 정리
        if self.emb_llm is not None:
            try:
                if hasattr(self.emb_llm, 'llm_engine'):
                    del self.emb_llm.llm_engine
            except:
                pass
            del self.emb_llm
            self.emb_llm = None
            print("  ✓ Embedding model cleaned up")
        
        # 아이템 임베딩 정리
        if self.item_embeddings is not None:
            self.item_embeddings = self.item_embeddings.cpu()
            del self.item_embeddings
            self.item_embeddings = None
            print("  ✓ Item embeddings cleaned up")
        
        # GPU 메모리 강제 해제
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"  💾 GPU Memory after evaluator cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        print("✓ Evaluator cleanup complete")

