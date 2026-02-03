"""
Evaluation class for recommendation model assessment
"""
import gc
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.pooling_params import PoolingParams
from sentence_transformers import SentenceTransformer

from utils.reward_function import calculate_ndcg, calculate_hit_rate, extract_query_from_tags


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


class RecommendationEvaluator:
    """
    추천 모델 평가를 위한 클래스
    vLLM을 사용하여 텍스트 생성 및 임베딩 기반 검색 수행
    """
    
    def __init__(self, args, checkpoint_dir, item_metadata=None):
        """
        Args:
            args: 학습 설정 파라미터
            checkpoint_dir: 평가할 모델 체크포인트 경로
            item_metadata: 아이템 메타데이터 (prepend_last_item 사용 시 필요)
        """
        self.args = args
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_metadata = item_metadata
        
        # vLLM 모델들 (lazy loading)
        self.llm = None
        self.emb_llm = None
        self.item_embeddings = None
        self.use_sentence_transformers = False  # SentenceTransformer 사용 여부
        
        # 아이템 인기도 및 Novelty 계산용
        self.item_popularity = None
        self.item_novelty = None
        self.hot_items = None  # 상위 20% 인기 아이템 세트
        self.cold_items = None  # 나머지 80% 아이템 세트

        self._load_item_embeddings()
        self._compute_item_popularity()
        # Cold/Hot 구분
        
        
    def _load_generation_model(self):
        """텍스트 생성을 위한 vLLM 모델 로드"""
        if hasattr(self.args, 'zeroshot_evaluation') and self.args.zeroshot_evaluation:
            print("🤖 Loading model with vLLM for zeroshot evaluation...")
            self.sampling_params = SamplingParams(
                temperature=0.01,
                max_tokens=self.args.eval_max_tokens,
                repetition_penalty=1.1,
            )
            self.llm = LLM(
                model=self.args.model_name,
                tensor_parallel_size=1,
                dtype=torch.bfloat16,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                max_model_len=self.args.max_length,
                max_num_seqs=self.args.eval_batch_size,
            )

        if self.llm is None:
            print("🤖 Loading model with vLLM for generation...")
            self.sampling_params = SamplingParams(
                # n=1,
                temperature=0.01,
                max_tokens=self.args.eval_max_tokens,
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
        """임베딩 계산을 위한 vLLM 또는 SentenceTransformer 모델 로드"""
        if self.emb_llm is None:
            print("🔍 Loading embedding model for retrieval...")
            emb_model_name = getattr(self.args, 'emb_model_name', 'mixedbread-ai/mxbai-embed-large-v1')
            use_sentence_transformers = getattr(self.args, 'use_sentence_transformers', False)
            
            # SentenceTransformer 사용 조건 체크
            if use_sentence_transformers:
                print("🔍 Using SentenceTransformer for embedding...")
                self.emb_llm = SentenceTransformer(emb_model_name)
                self.use_sentence_transformers = True
            else:
                print("🔍 Using vLLM for embedding...")
                # Pooling 파라미터 설정
                self.pooling_params = PoolingParams(
                    truncate_prompt_tokens=self.args.eval_emb_max_length,
                    task="embed",
                )
                
                self.emb_llm = LLM(
                    model=emb_model_name,
                    # task="embed",
                    runner="pooling",
                    enforce_eager=True,
                    gpu_memory_utilization=self.args.gpu_memory_utilization,
                    trust_remote_code=True,
                    max_model_len=self.args.eval_emb_max_length,
                    max_num_seqs=self.args.eval_emb_batch_size,
                )
                self.use_sentence_transformers = False
    
    def _load_item_embeddings(self):
        """아이템 임베딩 로드"""
        if self.item_embeddings is None:
            print("📦 Loading item embeddings...")
            emb_model_name = getattr(self.args, 'emb_model_name', 'mixedbread-ai/mxbai-embed-large-v1')
            emb_model_name_dir = emb_model_name.split('/')[-1]
            emb_type = getattr(self.args, 'emb_type', 'title')
            
            try:
                emb_file = f"data_emb/{self.args.data_name}_{emb_type}_{emb_model_name_dir}.pt"
                self.item_embeddings = torch.load(emb_file, map_location=self.device)
            except:
                emb_file = f"data_emb/{self.args.data_name}_{emb_type}_{emb_model_name_dir}_emb.pt"
                self.item_embeddings = torch.load(emb_file, map_location=self.device)
            self.item_embeddings = self.item_embeddings / self.item_embeddings.norm(dim=-1, keepdim=True)
            print(f"✓ Loaded item embeddings: {self.item_embeddings.shape}")
    
    def _compute_item_popularity(self):
        """
        아이템 인기도 및 Novelty 계산 (캐싱 지원)
        Hot/Cold 아이템 구분: 상위 20% 인기도 = Hot, 나머지 80% = Cold
        
        train set의 sequential_data.txt 파일을 읽어서 계산
        
        3개의 파일로 분리 저장:
        - item_popularity.npy: numpy 배열
        - item_novelty.npy: numpy 배열
        - item_cold_hot.json: {"cold_items": [...], "hot_items": [...]}
        """
        if self.item_popularity is None:
            # 캐시 파일 경로
            cache_dir = Path(f"./data/{self.args.data_name}")
            popularity_file = cache_dir / "item_popularity.npy"
            novelty_file = cache_dir / "item_novelty.npy"
            cold_hot_file = cache_dir / "item_cold_hot.json"
            
            # 캐시 파일이 모두 존재하면 로드
            if popularity_file.exists() and novelty_file.exists() and cold_hot_file.exists():
                print(f"📦 Loading cached item data from {cache_dir}...")
                try:
                    # Popularity 로드 (.npy 파일)
                    self.item_popularity = np.load(popularity_file)
                    
                    # Novelty 로드 (.npy 파일)
                    self.item_novelty = np.load(novelty_file)
                    
                    # Cold/Hot 로드 (JSON 파일)
                    with open(cold_hot_file, 'r') as f:
                        cold_hot_data = json.load(f)
                    
                    # Cold/Hot items를 set으로 변환
                    self.cold_items = set(cold_hot_data['cold_items'])
                    self.hot_items = set(cold_hot_data['hot_items'])
                    
                    max_pop = self.item_popularity.max()
                    print(f"✓ Item data loaded from cache. Max popularity: {max_pop:.4f}, Mean novelty: {self.item_novelty.mean():.4f}")
                    print(f"✓ Hot items: {len(self.hot_items)}, Cold items: {len(self.cold_items)}")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load cache: {e}. Computing from scratch...")
            
            # 캐시가 없으면 계산
            print("📊 Computing item popularity and novelty from sequential_data.txt...")
            
            # 전체 아이템 수
            num_items = self.item_embeddings.shape[0]
            
            # sequential_data.txt 파일 읽기
            sequential_file = Path(f"./data/{self.args.data_name}/sequential_data.txt")
            # 각 아이템의 출현 횟수 계산
            item_counts = np.zeros(num_items + 1, dtype=np.int32)  # 1-indexed
            
            with open(sequential_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # 첫 번째는 user_id, 나머지는 아이템 시퀀스 (history + target)
                    items = [int(x) for x in parts[1:-2]]
                    
                    # 모든 아이템의 출현 횟수 카운트
                    for item_id in items:
                        item_counts[item_id] += 1
            
            print(f"✓ Loaded {sequential_file}, total items processed")
            
            # 인기도 저장
            self.item_popularity = (item_counts+1) / item_counts.sum()
            max_count = item_counts.max()
            self.item_novelty = -np.log(self.item_popularity)
            
            # Hot/Cold 아이템 구분: 상위 20% 인기도를 기준으로
            # 인기도를 기준으로 정렬하여 상위 20% threshold 계산
            sorted_counts = np.sort(item_counts)[::-1]  # 내림차순 정렬
            threshold_idx = int(len(sorted_counts) * 0.2)
            threshold = sorted_counts[threshold_idx] if threshold_idx < len(sorted_counts) else 0
            
            # Hot items: 인기도가 threshold 이상인 아이템 (1-indexed)
            # Cold items: 인기도가 threshold 미만인 아이템 (1-indexed)
            hot_mask = (item_counts >= threshold) & (np.arange(len(item_counts)) > 0)
            cold_mask = (item_counts < threshold) & (np.arange(len(item_counts)) > 0)
            
            self.hot_items = set(np.where(hot_mask)[0].tolist())
            self.cold_items = set(np.where(cold_mask)[0].tolist())
            
            print(f"✓ Item popularity computed. Max count: {max_count}, Mean novelty: {self.item_novelty.mean():.4f}")
            print(f"✓ Hot items (top 20%): {len(self.hot_items)}, Cold items (80%): {len(self.cold_items)}, Threshold: {threshold}")
            
            # 캐시 파일에 저장 (3개의 파일로 분리)
            try:
                # 캐시 디렉토리 생성
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # 1. Popularity 저장 (.npy 파일)
                np.save(popularity_file, self.item_popularity)
                print(f"💾 Item popularity cached to {popularity_file}")
                
                # 2. Novelty 저장 (.npy 파일)
                np.save(novelty_file, self.item_novelty)
                print(f"💾 Item novelty cached to {novelty_file}")
                
                # 3. Cold/Hot 저장 (JSON 파일)
                cold_hot_data = {
                    'cold_items': list(self.cold_items),
                    'hot_items': list(self.hot_items),
                    'threshold': int(threshold),
                    'num_cold': len(self.cold_items),
                    'num_hot': len(self.hot_items),
                }
                with open(cold_hot_file, 'w') as f:
                    json.dump(cold_hot_data, f)
                print(f"💾 Cold/Hot items cached to {cold_hot_file}")
                
            except Exception as e:
                print(f"⚠️  Failed to save cache: {e}")
    
    def _identify_cold_warm_items(self, targets):
        """
        Cold/Hot 타겟 아이템 구분
        전체 train 데이터 기반 인기도로 상위 20% = Hot, 나머지 80% = Cold
        
        Args:
            targets: 타겟 아이템 ID 리스트
        
        Returns:
            cold_indices: Cold 타겟의 인덱스 리스트
            hot_indices: Hot 타겟의 인덱스 리스트
        """
        if self.hot_items is None or self.cold_items is None:
            raise ValueError("Hot/Cold items not computed. Call _compute_item_popularity first.")
        
        cold_indices = []
        hot_indices = []
        
        for i, target in enumerate(targets):
            # 타겟 아이템이 hot_items 세트에 있으면 hot, 없으면 cold
            if target in self.hot_items:
                hot_indices.append(i)
            else:
                cold_indices.append(i)
        
        return cold_indices, hot_indices
    
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
        <query> 태그가 있는 경우 태그 내부의 텍스트만 사용
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기 (None이면 args.eval_emb_batch_size 사용)
        
        Returns:
            query_embeddings: 정규화된 쿼리 임베딩 [num_texts, emb_dim]
        """
        self._load_embedding_model()
        
        # <query> 태그가 있으면 추출, 없으면 원본 사용
        processed_texts = [extract_query_from_tags(text, tag="query") for text in texts]
        
        print(f"🔍 Computing embeddings for {len(processed_texts)} texts...")
        
        if self.use_sentence_transformers:
            # SentenceTransformer 사용
            print("🔍 Using SentenceTransformer encode...")
            query_embeddings = self.emb_llm.encode(
                processed_texts, 
                batch_size=self.args.eval_emb_batch_size, 
                show_progress_bar=True, 
                convert_to_tensor=True
            )

        else:
            # vLLM 사용
            # add cls token
            processed_texts_with_cls = [f"[CLS] {text}" for text in processed_texts]

            # Embedding 계산
            emb_outputs = self.emb_llm.encode(
                prompts=processed_texts_with_cls,
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
        Cold/Hot 구분 및 Novelty 메트릭 포함
        - Cold: 인기도 하위 80% 아이템
        - Hot: 인기도 상위 20% 아이템
        
        Args:
            query_embeddings: 쿼리 임베딩 [num_queries, emb_dim]
            targets: 타겟 아이템 ID 리스트
            histories: 히스토리 아이템 ID 리스트의 리스트
            ks: 평가할 k 값들
        
        Returns:
            metrics: 메트릭 딕셔너리
            rank_info: 각 샘플의 rank 및 score 정보 리스트
        """
        cold_indices, hot_indices = self._identify_cold_warm_items(targets)
        print(f"  Cold targets: {len(cold_indices)}, Hot targets: {len(hot_indices)}")
        self.cold_indices = cold_indices
        self.hot_indices = hot_indices
        print(f"📊 Computing retrieval metrics...")
        
        # 메트릭 저장용 딕셔너리 초기화
        metrics = {f'ndcg@{k}': [] for k in ks}
        metrics.update({f'hit@{k}': [] for k in ks})
        metrics.update({f'cold_ndcg@{k}': [] for k in ks})
        metrics.update({f'cold_hit@{k}': [] for k in ks})
        metrics.update({f'hot_ndcg@{k}': [] for k in ks})
        metrics.update({f'hot_hit@{k}': [] for k in ks})
        metrics.update({f'novelty@{k}': [] for k in ks})
        metrics.update({f'novelty_hit@{k}': [] for k in ks})
        
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
            
            # 각 샘플의 rank 계산 및 Novelty 메트릭
            for i in range(len(batch_targets)):
                global_idx = start_idx + i
                target_item = batch_targets[i]
                target_score = batch_scores[i, target_item].item()
                
                # 타겟 아이템보다 높은 점수를 가진 아이템의 개수 = rank - 1
                rank = (batch_scores[i] > target_score).sum().item() + 1
                
                # Cold/Warm 구분
                is_cold = global_idx in self.cold_indices
                
                rank_info.append({
                    'target_item': target_item,
                    'target_score': target_score,
                    'rank': rank,
                    'is_cold': is_cold,
                    'target_novelty': float(self.item_novelty[target_item]),
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
                
                # Cold/Hot 별 메트릭 계산
                for i in range(len(batch_targets)):
                    global_idx = start_idx + i
                    is_cold = global_idx in self.cold_indices
                    
                    if is_cold:
                        metrics[f'cold_ndcg@{k}'].append(ndcg_scores[i].item())
                        metrics[f'cold_hit@{k}'].append(hit_scores[i].item())
                    else:
                        metrics[f'hot_ndcg@{k}'].append(ndcg_scores[i].item())
                        metrics[f'hot_hit@{k}'].append(hit_scores[i].item())
                
                # Novelty 메트릭 계산
                for i in range(len(batch_targets)):
                    global_idx = start_idx + i
                    target_item = batch_targets[i]
                    
                    # Top-k 아이템 추출
                    top_k_items = torch.topk(batch_scores[i], k).indices.cpu().numpy()
                    
                    # Top-k 아이템들의 평균 Novelty
                    avg_novelty = np.mean([self.item_novelty[item] for item in top_k_items])
                    metrics[f'novelty@{k}'].append(avg_novelty)
                    
                    # Hit@k에 타겟 아이템의 Novelty를 적용
                    hit_value = hit_scores[i].item()
                    target_novelty = self.item_novelty[target_item]
                    novelty_hit = hit_value * target_novelty
                    metrics[f'novelty_hit@{k}'].append(novelty_hit)
        
        return metrics, rank_info
    
    def evaluate(self, dataset, split="test", save_log=True, pre_generated_texts=None):
        """
        전체 평가 파이프라인 실행
        
        Args:
            dataset: 평가할 데이터셋
            split: 데이터셋 split 이름 ("test", "val" 등)
            save_log: 로그 파일 저장 여부
            pre_generated_texts: 미리 생성된 텍스트 리스트 (선택사항)
        
        Returns:
            results: 평가 결과 딕셔너리
        
        Note:
            - args.prepend_last_item이 True이고 item_metadata가 제공된 경우,
              마지막 구매 아이템 정보를 생성된 텍스트 앞에 추가합니다.
        """
        print(f"\n{'='*80}")
        print(f"📊 Final Evaluation on {split.upper()} Set")
        print(f"{'='*80}")
        
        # 1. 데이터 수집
        print("📝 Collecting data...")
        all_prompts = []
        all_targets = []
        all_histories = []
        all_user_ids = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            all_prompts.append(sample["prompt"])
            all_targets.append(sample["target"])
            all_histories.append(sample["history"])
            all_user_ids.append(sample["user_id"])

        if len(all_prompts) > self.args.eval_samples:
            all_prompts = all_prompts[:self.args.eval_samples]
            all_targets = all_targets[:self.args.eval_samples]
            all_histories = all_histories[:self.args.eval_samples]
            all_user_ids = all_user_ids[:self.args.eval_samples]
        
        # 2. 텍스트 생성 (모든 프롬프트에 대해)
        if pre_generated_texts is not None:
            print("📄 Using pre-generated texts from CSV...")
            generated_texts = pre_generated_texts[:len(all_prompts)]
            print(f"  Loaded {len(generated_texts)} pre-generated texts")
        elif hasattr(self.args, "dummy_generation") and self.args.dummy_generation:
            generated_texts = all_prompts
        else:
            generated_texts = self.generate_all_texts(all_prompts)

        # 생성 모델 즉시 정리 (메모리 절약)
        if pre_generated_texts is None:  # 생성 모델을 사용한 경우에만 정리
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
        
        # 2.5. 마지막 아이템 prepend 옵션 처리
        if hasattr(self.args, 'prepend_last_item') and self.args.prepend_last_item:
            if self.item_metadata is None:
                print("⚠️  Warning: prepend_last_item is enabled but item_metadata is not provided. Skipping prepending.")
            else:
                print("📝 Prepending last purchased item to generated texts...")
                last_item_texts = get_last_item_text(
                    dataset, 
                    self.item_metadata,
                    use_brand=getattr(self.args, 'use_brand', True),
                    use_category=getattr(self.args, 'use_category', True)
                )
                
                # 마지막 아이템 텍스트를 generated text 앞에 추가
                modified_texts = []
                for last_item_text, generated_text in zip(last_item_texts, generated_texts):
                    if last_item_text:
                        modified_text = f"{last_item_text}\n\n{generated_text}"
                    else:
                        modified_text = generated_text
                    modified_texts.append(modified_text)
                
                generated_texts = modified_texts
                print(f"✓ Prepended last item to {len(generated_texts)} texts")
                
                # 샘플 출력
                print("\n" + "="*80)
                print("📝 Sample Modified Text (with last item prepended):")
                print("="*80)
                if len(generated_texts) > 0:
                    print(generated_texts[0][:500] + "..." if len(generated_texts[0]) > 500 else generated_texts[0])
                    print("="*80)
        
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
                all_user_ids,
                rank_info, 
                split
            )
            # 마스터 로그 파일 저장
            self._save_master_log(results, split)
        
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
        
        # 기본 메트릭
        print("\n[Overall Metrics]")
        for metric_name in ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                mean_val = np.mean(metrics[metric_name])
                results[metric_name] = float(mean_val)
                print(f"  {metric_name.upper()}: {mean_val:.4f}")
        
        # Cold 메트릭
        print("\n[Cold Items Metrics (80%)]")
        for metric_name in ['cold_hit@5', 'cold_hit@10', 'cold_hit@20', 'cold_ndcg@5', 'cold_ndcg@10', 'cold_ndcg@20']:
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                mean_val = np.mean(metrics[metric_name])
                results[metric_name] = float(mean_val)
                print(f"  {metric_name.upper()}: {mean_val:.4f}")
            else:
                results[metric_name] = 0.0
                print(f"  {metric_name.upper()}: N/A (no cold items)")
        
        # Hot 메트릭
        print("\n[Hot Items Metrics (Top 20%)]")
        for metric_name in ['hot_hit@5', 'hot_hit@10', 'hot_hit@20', 'hot_ndcg@5', 'hot_ndcg@10', 'hot_ndcg@20']:
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                mean_val = np.mean(metrics[metric_name])
                results[metric_name] = float(mean_val)
                print(f"  {metric_name.upper()}: {mean_val:.4f}")
            else:
                results[metric_name] = 0.0
                print(f"  {metric_name.upper()}: N/A (no hot items)")
        
        # Novelty 메트릭
        print("\n[Novelty Metrics]")
        for metric_name in ['novelty@5', 'novelty@10', 'novelty@20', 'novelty_hit@5', 'novelty_hit@10', 'novelty_hit@20']:
            if metric_name in metrics and len(metrics[metric_name]) > 0:
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
            
            # Overall Metrics
            f.write("\n[Overall Metrics]\n")
            for metric_name in ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
                if metric_name in results:
                    f.write(f"  {metric_name.upper()}: {results[metric_name]:.4f}\n")
            
            # Cold Metrics
            f.write("\n[Cold Items Metrics (80%)]\n")
            for metric_name in ['cold_hit@5', 'cold_hit@10', 'cold_hit@20', 'cold_ndcg@5', 'cold_ndcg@10', 'cold_ndcg@20']:
                if metric_name in results:
                    f.write(f"  {metric_name.upper()}: {results[metric_name]:.4f}\n")
            
            # Hot Metrics
            f.write("\n[Hot Items Metrics (Top 20%)]\n")
            for metric_name in ['hot_hit@5', 'hot_hit@10', 'hot_hit@20', 'hot_ndcg@5', 'hot_ndcg@10', 'hot_ndcg@20']:
                if metric_name in results:
                    f.write(f"  {metric_name.upper()}: {results[metric_name]:.4f}\n")
            
            # Novelty Metrics
            f.write("\n[Novelty Metrics]\n")
            for metric_name in ['novelty@5', 'novelty@10', 'novelty@20', 'novelty_hit@5', 'novelty_hit@10', 'novelty_hit@20']:
                if metric_name in results:
                    f.write(f"  {metric_name.upper()}: {results[metric_name]:.4f}\n")
            
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
    
    def _save_master_log(self, results, split):
        """
        단일 마스터 로그 파일에 모든 성능 지표 기록
        
        Args:
            results: 메트릭 결과 딕셔너리
            split: 데이터셋 split 이름
        """
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        master_log_file = results_dir / f"0_{self.args.data_name}_master.log"
        
        with open(master_log_file, 'a') as f:
            f.write("="*120 + "\n")
            f.write(f"Master Evaluation Log - {self.args.run_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"Dataset: {self.args.data_name}\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write("="*120 + "\n\n")
            
            # 컬럼 헤더 (Overall)
            f.write("[Overall Metrics]\n")
            f.write("Split\tH@5\tN@5\tH@10\tN@10\tH@20\tN@20\n")
            f.write("-"*70 + "\n")
                        
            f.write(f"{split}\t"
                   f"{results.get('hit@5', 0.0):.4f}\t"
                   f"{results.get('ndcg@5', 0.0):.4f}\t"
                   f"{results.get('hit@10', 0.0):.4f}\t"
                   f"{results.get('ndcg@10', 0.0):.4f}\t"
                   f"{results.get('hit@20', 0.0):.4f}\t"
                   f"{results.get('ndcg@20', 0.0):.4f}\n")
            
            # Cold Metrics (80%)
            f.write("\n[Cold Items Metrics (80%)]\n")
            f.write("Split\tH@5\tN@5\tH@10\tN@10\tH@20\tN@20\n")
            f.write("-"*70 + "\n")
        
            f.write(f"{split}\t"
                   f"{results.get('cold_hit@5', 0.0):.4f}\t"
                   f"{results.get('cold_ndcg@5', 0.0):.4f}\t"
                   f"{results.get('cold_hit@10', 0.0):.4f}\t"
                   f"{results.get('cold_ndcg@10', 0.0):.4f}\t"
                   f"{results.get('cold_hit@20', 0.0):.4f}\t"
                   f"{results.get('cold_ndcg@20', 0.0):.4f}\n")
            
            # Hot Metrics (Top 20%)
            f.write("\n[Hot Items Metrics (Top 20%)]\n")
            f.write("Split\tH@5\tN@5\tH@10\tN@10\tH@20\tN@20\n")
            f.write("-"*70 + "\n")
            
            f.write(f"{split}\t"
                   f"{results.get('hot_hit@5', 0.0):.4f}\t"
                   f"{results.get('hot_ndcg@5', 0.0):.4f}\t"
                   f"{results.get('hot_hit@10', 0.0):.4f}\t"
                   f"{results.get('hot_ndcg@10', 0.0):.4f}\t"
                   f"{results.get('hot_hit@20', 0.0):.4f}\t"
                   f"{results.get('hot_ndcg@20', 0.0):.4f}\n")
            
            # Novelty Metrics
            f.write("\n[Novelty Metrics]\n")
            f.write("Split\tNov@5\tNH@5\tNov@10\tNH@10\tNov@20\tNH@20\n")
            f.write("-"*70 + "\n")
            
            f.write(f"{split}\t"
                   f"{results.get('novelty@5', 0.0):.4f}\t"
                   f"{results.get('novelty_hit@5', 0.0):.4f}\t"
                   f"{results.get('novelty@10', 0.0):.4f}\t"
                   f"{results.get('novelty_hit@10', 0.0):.4f}\t"
                   f"{results.get('novelty@20', 0.0):.4f}\t"
                   f"{results.get('novelty_hit@20', 0.0):.4f}\n")

            f.write("-"*120 + "\n\n")
        
        print(f"📊 Master log updated: {master_log_file}")
    
    def _save_csv_file(self, prompts, generated_texts, targets, histories, user_ids, rank_info, split):
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
            rank = rank_info[i]['rank']
            is_cold = rank_info[i]['is_cold']
            target_novelty = rank_info[i]['target_novelty']
            
            data.append({
                'sample_id': i,
                'user_id': user_ids[i],
                'target_item_id': targets[i],
                'prompt': prompts[i],
                'generated_text': generated_texts[i],
                'history_items': str(histories[i]),  # 리스트를 문자열로 변환
                'target_score': rank_info[i]['target_score'],
                'rank': rank,
                'is_cold': is_cold,
                'target_novelty': target_novelty,
                'hit@5': 1 if rank <= 5 else 0,
                'hit@10': 1 if rank <= 10 else 0,
                'hit@20': 1 if rank <= 20 else 0,
                'novelty_hit@5': target_novelty if rank <= 5 else 0,
                'novelty_hit@10': target_novelty if rank <= 10 else 0,
                'novelty_hit@20': target_novelty if rank <= 20 else 0,
            })
        
        df = pd.DataFrame(data)
        
        # CSV 저장
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"💾 Detailed results saved to CSV: {csv_file}")
        
        # 간단한 통계 출력
        print(f"\n📈 CSV Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Cold items: {df['is_cold'].sum()}, Warm items: {(~df['is_cold']).sum()}")
        print(f"  Mean rank: {df['rank'].mean():.2f}")
        print(f"  Median rank: {df['rank'].median():.0f}")
        print(f"  Hit@5 rate: {df['hit@5'].mean():.4f}")
        print(f"  Hit@10 rate: {df['hit@10'].mean():.4f}")
        print(f"  Hit@20 rate: {df['hit@20'].mean():.4f}")
        print(f"  Mean target novelty: {df['target_novelty'].mean():.4f}")
    
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
        
        # 임베딩 모델 정리 (vLLM 또는 SentenceTransformer)
        if self.emb_llm is not None:
            if not self.use_sentence_transformers:
                # vLLM의 경우
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
        
        # 아이템 인기도 및 Novelty 정리
        if self.item_popularity is not None:
            del self.item_popularity
            self.item_popularity = None
        
        if self.item_novelty is not None:
            del self.item_novelty
            self.item_novelty = None
        
        if self.hot_items is not None:
            del self.hot_items
            self.hot_items = None
        
        if self.cold_items is not None:
            del self.cold_items
            self.cold_items = None
        
        if any([self.item_popularity is None, self.item_novelty is None, 
                self.hot_items is None, self.cold_items is None]):
            print("  ✓ Item popularity, novelty, and hot/cold sets cleaned up")
        
        # GPU 메모리 강제 해제
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"  💾 GPU Memory after evaluator cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        print("✓ Evaluator cleanup complete")

