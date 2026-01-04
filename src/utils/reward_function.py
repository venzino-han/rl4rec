"""
Reward Functions for RL4Rec
NDCG 기반 리워드 계산 및 TRL 통합
"""

import json
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import ray
from pathlib import Path


def calculate_dcg(relevance_scores: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """
    DCG (Discounted Cumulative Gain) 계산
    
    Args:
        relevance_scores: [batch_size, num_items] 관련성 점수
        k: top-k 까지만 계산 (None이면 전체)
    
    Returns:
        DCG scores [batch_size]
    """
    if k is not None:
        relevance_scores = relevance_scores[:, :k]
    
    # DCG = sum(rel_i / log2(i + 2)) for i in range(k)
    positions = torch.arange(1, relevance_scores.shape[1] + 1, device=relevance_scores.device)
    discounts = torch.log2(positions + 1.0)
    dcg = (relevance_scores / discounts).sum(dim=1)
    
    return dcg


def calculate_ndcg_from_rank(ranks: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Rank 기반 NDCG 계산
    
    Args:
        ranks: [batch_size] GT 아이템의 rank (1-indexed)
        k: Top-K NDCG
    
    Returns:
        NDCG scores [batch_size]
    """
    # Rank가 k보다 크면 NDCG = 0
    ndcg_scores = torch.zeros_like(ranks, dtype=torch.float32)
    
    # Rank가 k 이내인 경우만 계산
    valid_mask = ranks <= k
    valid_ranks = ranks[valid_mask]
    
    # DCG = 1 / log2(rank + 1)
    dcg = 1.0 / torch.log2(valid_ranks.float() + 1.0)
    
    # IDCG = 1 / log2(2) (이상적인 경우, rank=1)
    idcg = 1.0 / torch.log2(torch.tensor(2.0, device=ranks.device))
    
    # NDCG = DCG / IDCG
    ndcg_scores[valid_mask] = dcg / idcg
    
    return ndcg_scores


def calculate_hit_from_rank(ranks: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Rank 기반 Hit@K 계산
    
    Args:
        ranks: [batch_size] GT 아이템의 rank (1-indexed)
        k: Top-K
    
    Returns:
        Hit scores [batch_size]
    """
    # Rank가 k 이내면 1, 아니면 0
    hit_scores = (ranks <= k).float()
    return hit_scores


def calculate_mrr_from_rank(ranks: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Rank 기반 MRR (Mean Reciprocal Rank) 계산
    
    Args:
        ranks: [batch_size] GT 아이템의 rank (1-indexed)
        k: Top-K
    
    Returns:
        MRR scores [batch_size]
    """
    # Rank가 k보다 크면 MRR = 0
    mrr_scores = torch.zeros_like(ranks, dtype=torch.float32)
    
    # Rank가 k 이내인 경우만 계산
    valid_mask = ranks <= k
    valid_ranks = ranks[valid_mask]
    
    # MRR = 1 / rank
    mrr_scores[valid_mask] = 1.0 / valid_ranks.float()
    
    return mrr_scores


def calculate_novelty_ndcg(
    ranks: torch.Tensor, 
    item_weights: torch.Tensor,
    k: int = 10, 
) -> torch.Tensor:
    """
    Popularity-based novelty score calculation
    
    NDCG에 아이템 인기도 역수(item_weights)를 곱하여 novelty 계산
    인기 없는 아이템을 높은 rank로 예측할수록 높은 novelty
    
    Args:
        ranks: [batch_size] GT 아이템의 rank (1-indexed)
        item_weights: [batch_size] 각 타겟 아이템의 popularity weight (역수)
        k: Top-K
        target_rank: (사용 안함, backward compatibility)
    
    Returns:
        novelty scores [batch_size] = NDCG × item_weights
    """
    # 1. Ranks로부터 NDCG 계산
    ndcg_scores = calculate_ndcg_from_rank(ranks, k=k)
    
    # 2. NDCG에 popularity weight 곱하기
    # 인기 없는 아이템(높은 weight)을 잘 예측하면 높은 novelty
    novelty = ndcg_scores * item_weights
    
    return novelty


def calculate_ndcg(
    predicted_scores: torch.Tensor,
    target_items: List[int],
    history_items: List[List[int]],
    k: int = 10,
    use_negatives_only: bool = False,
) -> torch.Tensor:
    """
    NDCG (Normalized Discounted Cumulative Gain) 계산
    
    Args:
        predicted_scores: [batch_size, num_items] 또는 [batch_size, 1+num_negs] 예측 점수
        target_items: [batch_size] 실제 타겟 아이템 ID 리스트
        history_items: [batch_size, *] 사용자별 히스토리 아이템 ID 리스트
        k: Top-K NDCG (default: 10)
        use_negatives_only: True이면 target+negatives만 사용 (scores shape [batch_size, 1+num_negs])
    
    Returns:
        NDCG scores [batch_size]
    """
    batch_size = predicted_scores.shape[0]
    ndcg_scores = torch.zeros(batch_size, device=predicted_scores.device)
    
    if use_negatives_only:
        # Target + negatives만 고려하는 경우
        # scores shape: [batch_size, 1 + num_negs]
        # target은 항상 index 0
        for i in range(batch_size):
            scores = predicted_scores[i]  # [1 + num_negs]
            k_actual = min(k, len(scores))
            
            # Top-K 추출
            top_k_scores, top_k_indices = torch.topk(scores, k=k_actual)
            
            # Target (index 0)이 top-k에 있는지 확인
            relevance = torch.zeros(k_actual, device=predicted_scores.device)
            target_positions = (top_k_indices == 0).nonzero(as_tuple=True)[0]
            
            if len(target_positions) > 0:
                position = target_positions[0].item()
                relevance[position] = 1.0
                
                # DCG 계산
                dcg = calculate_dcg(relevance.unsqueeze(0), k=k_actual)[0]
                
                # IDCG 계산
                ideal_relevance = torch.zeros(k_actual, device=predicted_scores.device)
                ideal_relevance[0] = 1.0
                idcg = calculate_dcg(ideal_relevance.unsqueeze(0), k=k_actual)[0]
                
                ndcg_scores[i] = dcg / (idcg + 1e-10)
            else:
                ndcg_scores[i] = 0.0
    else:
        # 전체 아이템 고려하는 경우 (기존 로직)
        for i in range(batch_size):
            # 1. 히스토리 아이템 제외 (masking)
            scores = predicted_scores[i].clone()
            if history_items[i]:
                history_mask = torch.zeros_like(scores, dtype=torch.bool, device=scores.device)
                history_mask[history_items[i]] = True
                history_mask[target_items[i]] = False
                scores[history_mask] = -float('inf')
            
            # 2. Top-K 아이템 추출
            top_k_scores, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
            
            # 3. Relevance 계산 (target item이 top-k에 있으면 해당 위치에 1, 없으면 0)
            relevance = torch.zeros(k, device=predicted_scores.device)
            target_item = target_items[i]
            
            # Top-k에서 target item의 위치 찾기
            target_positions = (top_k_indices == target_item).nonzero(as_tuple=True)[0]
            if len(target_positions) > 0:
                position = target_positions[0].item()
                relevance[position] = 1.0
            
            # 4. DCG 계산
            if relevance.sum() > 0:
                dcg = calculate_dcg(relevance.unsqueeze(0), k=k)[0]
                
                # 5. IDCG (Ideal DCG) 계산 - 이상적인 경우 (target이 1위)
                ideal_relevance = torch.zeros(k, device=predicted_scores.device)
                ideal_relevance[0] = 1.0
                idcg = calculate_dcg(ideal_relevance.unsqueeze(0), k=k)[0]
                
                # 6. NDCG = DCG / IDCG
                ndcg_scores[i] = dcg / (idcg + 1e-10)
            else:
                # Target이 top-k에 없으면 NDCG = 0
                ndcg_scores[i] = 0.0
    
    return ndcg_scores


def calculate_hit_rate(
    predicted_scores: torch.Tensor,
    target_items: List[int],
    history_items: List[List[int]],
    k: int = 10,
    use_negatives_only: bool = False,
) -> torch.Tensor:
    """
    Hit@K 계산 (target이 top-k에 있으면 1, 없으면 0)
    
    Args:
        predicted_scores: [batch_size, num_items] 또는 [batch_size, 1+num_negs]
        target_items: [batch_size]
        history_items: [batch_size, *]
        k: Top-K
        use_negatives_only: True이면 target+negatives만 사용
    
    Returns:
        Hit scores [batch_size]
    """
    batch_size = predicted_scores.shape[0]
    hit_scores = torch.zeros(batch_size, device=predicted_scores.device)
    
    if use_negatives_only:
        # Target + negatives만 고려
        for i in range(batch_size):
            scores = predicted_scores[i]
            k_actual = min(k, len(scores))
            
            # Top-K 추출
            _, top_k_indices = torch.topk(scores, k=k_actual)
            
            # Target (index 0)이 top-k에 있는지 확인
            if 0 in top_k_indices:
                hit_scores[i] = 1.0
    else:
        # 전체 아이템 고려 (기존 로직)
        for i in range(batch_size):
            # 히스토리 아이템 제외
            scores = predicted_scores[i].clone()
            if history_items[i]:
                history_mask = torch.zeros_like(scores, dtype=torch.bool, device=scores.device)
                history_mask[history_items[i]] = True
                history_mask[target_items[i]] = False
                scores[history_mask] = -float('inf')
            
            # Top-K 추출
            _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
            
            # Target이 top-k에 있는지 확인
            if target_items[i] in top_k_indices:
                hit_scores[i] = 1.0
    
    return hit_scores


def calculate_mrr(
    predicted_scores: torch.Tensor,
    target_items: List[int],
    history_items: List[List[int]],
    k: int = 10,
    use_negatives_only: bool = False,
) -> torch.Tensor:
    """
    MRR (Mean Reciprocal Rank) 계산
    
    Args:
        predicted_scores: [batch_size, num_items] 또는 [batch_size, 1+num_negs]
        target_items: [batch_size]
        history_items: [batch_size, *]
        k: Top-K
        use_negatives_only: True이면 target+negatives만 사용
    
    Returns:
        MRR scores [batch_size]
    """
    batch_size = predicted_scores.shape[0]
    mrr_scores = torch.zeros(batch_size, device=predicted_scores.device)
    
    if use_negatives_only:
        # Target + negatives만 고려
        for i in range(batch_size):
            scores = predicted_scores[i]
            k_actual = min(k, len(scores))
            
            # Top-K 추출
            _, top_k_indices = torch.topk(scores, k=k_actual)
            
            # Target (index 0)의 rank 찾기
            target_positions = (top_k_indices == 0).nonzero(as_tuple=True)[0]
            if len(target_positions) > 0:
                rank = target_positions[0].item() + 1  # 1-indexed rank
                mrr_scores[i] = 1.0 / rank
    else:
        # 전체 아이템 고려 (기존 로직)
        for i in range(batch_size):
            # 히스토리 아이템 제외
            scores = predicted_scores[i].clone()
            if history_items[i]:
                history_mask = torch.zeros_like(scores, dtype=torch.bool)
                history_mask[history_items[i]] = True
                scores[history_mask] = -float('inf')
            
            # Top-K 추출
            _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
            
            # Target의 rank 찾기
            target_positions = (top_k_indices == target_items[i]).nonzero(as_tuple=True)[0]
            if len(target_positions) > 0:
                rank = target_positions[0].item() + 1  # 1-indexed rank
                mrr_scores[i] = 1.0 / rank
    
    return mrr_scores


class RecRewardFrunction:
    """
    TRL과 호환되는 리워드 함수 클래스
    Ray RetrievalService와 통합
    """
    
    def __init__(
        self,
        retrieval_service_name: str = "RetrievalService",
        namespace: str = "rl4rec",
        data_name: str = "beauty",
        reward_type: str = "ndcg",
        k: int = 10,
        normalize: bool = True,
        test_target: bool = False,
    ):
        """
        Args:
            retrieval_service_name: Ray actor 이름
            namespace: Ray namespace
            data_name: 데이터셋 이름
            reward_type: 리워드 타입 ('ndcg', 'hit', 'mrr', 'mixed')
            k: Top-K 값
            normalize: 리워드 정규화 여부
        """
        self.__name__ = "RecRewardFrunction"
        self.retrieval_service_name = retrieval_service_name
        self.namespace = namespace
        self.data_name = data_name
        self.reward_type = reward_type
        self.k = k
        self.normalize = normalize
        self.test_target = test_target

        # RetrievalService 연결
        try:
            self.retrieval_service = ray.get_actor(
                retrieval_service_name,
                namespace=namespace
            )
            print(f"✓ Connected to {retrieval_service_name}")
        except ValueError as e:
            raise RuntimeError(
                f"Failed to connect to {retrieval_service_name}. "
                f"Make sure retrieval service is running."
            ) from e

        #load item metadata
        with open(f"data/{data_name}/meta_text_fix.json", "r") as f:
            self.item_metadata = json.load(f)
        self.item_metadata = {int(k): v["title"] + "\n" + v["brand"] + "\n" + v["category"] for k, v in self.item_metadata.items()}
    
    def __call__(
        self,
        generated_texts: List[str],
        targets: List[int],
        histories: List[List[int]],
        neg_items: Optional[List[List[int]]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        TRL 호환 리워드 함수
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            targets: [batch_size] 타겟 아이템 ID
            histories: [batch_size, *] 히스토리 아이템 ID
            neg_items: [batch_size, num_negs] Negative 아이템 ID (optional)
        
        Returns:
            rewards: [batch_size] 리워드 값
        """
        # add target text to generated_texts
        if self.test_target:
            generated_texts = [self.item_metadata[target] + "\n" + generated_text for generated_text, target in zip(generated_texts, targets)]  

        # 1. RetrievalService를 통해 유사도 점수 계산
        use_negatives_only = neg_items is not None
        
        scores_ref = self.retrieval_service.calculate_reward.remote(
            generated_texts,
            data_name=self.data_name,
            targets=targets if use_negatives_only else None,
            neg_items=neg_items,
        )
        scores = ray.get(scores_ref)  # [batch_size, num_items] or [batch_size, 1+num_negs]
        
        # 2. 리워드 타입에 따라 계산
        if self.reward_type == "ndcg":
            rewards = calculate_ndcg(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        elif self.reward_type == "hit":
            rewards = calculate_hit_rate(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        elif self.reward_type == "mrr":
            rewards = calculate_mrr(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        elif self.reward_type == "mixed":
            # NDCG + Hit@K의 가중 평균
            ndcg = calculate_ndcg(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
            hit = calculate_hit_rate(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
            rewards = 0.7 * ndcg + 0.3 * hit
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
        
        # 3. 정규화 (optional)
        if self.normalize and rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return rewards
    
    def compute_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        completion_ids: List[List[int]],
        targets: List[int],
        histories: List[List[int]],
        neg_items: Optional[List[List[int]]] = None,
    ) -> Dict[str, float]:
        """
        평가 메트릭 계산 (NDCG, Hit@K, MRR 모두)
        
        Args:
            neg_items: [batch_size, num_negs] Negative 아이템 ID (optional)
        
        Returns:
            메트릭 딕셔너리
        """
        # RetrievalService를 통해 유사도 점수 계산
        use_negatives_only = neg_items is not None
        
        scores_ref = self.retrieval_service.calculate_reward.remote(
            completions,
            data_name=self.data_name,
            targets=targets if use_negatives_only else None,
            neg_items=neg_items,
        )
        scores = ray.get(scores_ref)
        
        # 모든 메트릭 계산
        ndcg = calculate_ndcg(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        hit = calculate_hit_rate(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        mrr = calculate_mrr(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        
        return {
            f"ndcg@{self.k}": ndcg.mean().item(),
            f"hit@{self.k}": hit.mean().item(),
            f"mrr@{self.k}": mrr.mean().item(),
        }


# TRL PPOTrainer와 호환되는 래퍼 함수
def create_reward_function(
    retrieval_service_name: str = "RetrievalService",
    namespace: str = "rl4rec",
    data_name: str = "beauty",
    reward_type: str = "ndcg",
    k: int = 10,
) -> RecRewardFrunction:
    """
    TRL PPOTrainer에서 사용할 리워드 함수 생성
    
    Usage:
        reward_fn = create_reward_function(reward_type="ndcg", k=10)
        rewards = reward_fn(generated_texts, targets, histories)
    """
    return RecRewardFrunction(
        retrieval_service_name=retrieval_service_name,
        namespace=namespace,
        data_name=data_name,
        reward_type=reward_type,
        k=k,
    )


def load_negative_pool(data_name: str, data_dir: str = "data", k: int = 10) -> Dict[int, List[int]]:
    """
    negative.txt 파일에서 negative pool 로드
    
    파일 형식: 각 라인은 "user_id neg_item1 neg_item2 ... neg_itemN"
    
    Args:
        data_name: 데이터셋 이름 (e.g., "beauty")
        data_dir: 데이터 디렉토리
    
    Returns:
        user_id를 키로 하는 negative items 리스트 딕셔너리
    """
    negative_file = Path(data_dir) / data_name / "negative.txt"
    
    if not negative_file.exists():
        raise FileNotFoundError(f"Negative pool file not found: {negative_file}")
    
    print(f"📦 Loading negative pool from: {negative_file}")
    negative_pool = {}
    
    with open(negative_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            user_id = int(parts[0])
            neg_items = [int(item_id) for item_id in parts[1:]]
            #random sample k items
            neg_items = neg_items[:k-1]
            # neg_items = np.random.choice(neg_items, size=k-1, replace=False).tolist()
            negative_pool[user_id] = neg_items
    
    print(f"✓ Loaded negative pool for {len(negative_pool)} users")
    if len(negative_pool) > 0:
        sample_user = next(iter(negative_pool))
        print(f"  Example: User {sample_user} has {len(negative_pool[sample_user])} negative items")
    
    return negative_pool


class LocalEmbeddingRewardFunction:
    """
    로컬 임베딩 기반 리워드 함수
    DB 대신 자체적으로 negative item 임베딩을 계산하여 NDCG를 reward로 활용
    """
    
    def __init__(
        self,
        uid_2_target: Dict[int, int],
        data_name: str,
        k: int = 10,
        reward_type: str = "ndcg",
        emb_model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        emb_type: str = "review_description",
        device: str = "cuda",
        emb_batch_size: int = 128,
        data_dir: str = "data",
        normalize: bool = True,
        novelty_reward: bool = False,
        novelty_target_rank: int = 3,
        novelty_mode: str = "gaussian",
        popularity_coef: float = 0.0,
    ):
        """
        Args:
            uid_2_target: 사용자 ID to 타겟 아이템 ID 매핑
            data_name: 데이터셋 이름
            k: Top-K 값
            reward_type: 리워드 타입 ('ndcg', 'hit', 'mrr', 'mixed')
            emb_model_name: 임베딩 모델 이름
            emb_type: 임베딩 타입 (review_description, title, etc.)
            device: 디바이스
            emb_batch_size: 임베딩 계산 배치 크기
            data_dir: 데이터 디렉토리
            normalize: 리워드 정규화 여부
            novelty_reward: Novelty 리워드 사용 여부 (True/False)
                           Novelty = NDCG × popularity_weight
            novelty_target_rank: (사용 안함, backward compatibility)
            novelty_mode: (사용 안함, backward compatibility)
            popularity_coef: Popularity 리워드 계수 (0.0 = 사용 안함)
                            정답인 경우에만 popularity bonus 추가
        """
        self.__name__ = "LocalEmbeddingRewardFunction"
        self.data_name = data_name
        self.reward_type = reward_type
        self.k = k
        self.normalize = normalize
        self.device = device
        self.emb_batch_size = emb_batch_size
        
        # Novelty 관련 파라미터
        self.novelty_reward = novelty_reward
        self.novelty_target_rank = novelty_target_rank
        self.novelty_mode = novelty_mode
        
        # Popularity 관련 파라미터
        self.popularity_coef = popularity_coef
        
        print(f"💰 Reward configuration:")
        print(f"  - Reward type: {reward_type}")
        print(f"  - Top-K: {k}")
        print(f"  - Normalize: {normalize}")
        if novelty_reward:
            print(f"  - Novelty reward: ENABLED")
            print(f"  - Novelty = NDCG × popularity_weight (인기 없는 아이템 장려)")
        if popularity_coef > 0:
            print(f"  - Popularity coefficient: {popularity_coef}")
            print(f"  - Popularity bonus for unpopular items (when correct)")
        
        # 임베딩 모델 로드
        print(f"🤖 Loading embedding model: {emb_model_name}")
        from sentence_transformers import SentenceTransformer
        self.emb_model = SentenceTransformer(emb_model_name, device=device)
        print(f"✓ Embedding model loaded on {device}")
        
        # Negative pool 로드
        self.negative_pool = load_negative_pool(data_name, data_dir, k)

        # prepare candidate set, target comes first
        self.candidate_tensor = self._prepare_candidate_tensor(uid_2_target, self.negative_pool)
        
        # 사전 계산된 아이템 임베딩 로드
        emb_model_name_dir = emb_model_name.split("/")[-1]
        item_embedding_file_path = f"data_emb/{data_name}_{emb_type}_{emb_model_name_dir}.pt"
        print(f"📦 Loading pre-computed item embeddings from: {item_embedding_file_path}")
        self.item_embeddings = torch.load(item_embedding_file_path, map_location=device)
        print(f"✓ Loaded embeddings for {len(self.item_embeddings)} items")
        
        # 아이템 인기도 계산 (train set에서)
        # Novelty 또는 Popularity reward 사용 시 필요
        if self.novelty_reward or self.popularity_coef > 0:
            self.item_popularity_weights = self._compute_item_popularity(
                uid_2_target, self.negative_pool, data_name, data_dir
            )
        else:
            self.item_popularity_weights = None

    def _prepare_candidate_tensor(self, uid_2_target: Dict[int, int], neg_pool: Dict[int, List[int]]) -> torch.Tensor:
        candidate_tensor = torch.zeros(len(uid_2_target)+1, self.k, dtype=torch.long)
        for uid, target_id in uid_2_target.items():
            candidate_tensor[uid] = torch.tensor([target_id] + neg_pool[uid], dtype=torch.long)
        return candidate_tensor
    
    def _compute_item_popularity(
        self, 
        uid_2_target: Dict[int, int], 
        neg_pool: Dict[int, List[int]],
        data_name: str,
        data_dir: str
    ) -> torch.Tensor:
        """
        Train set에서 아이템 인기도 계산 (log 역수 사용)
        
        Args:
            uid_2_target: 사용자 ID to 타겟 아이템 ID 매핑
            neg_pool: Negative pool
            data_name: 데이터셋 이름
            data_dir: 데이터 디렉토리
        
        Returns:
            item_popularity_weights: [num_items] 각 아이템의 인기도 가중치
                                     인기 없는 아이템일수록 높은 값
        """
        print(f"📊 Computing item popularity weights from train set...")
        
        # sequential_data.txt에서 train set 로드 (target_index=-3)
        sequential_file = f"{data_dir}/{data_name}/sequential_data.txt"
        item_counts = {}
        
        with open(sequential_file, 'r') as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                history = parts[1:-3]  # Train set의 history
                target = parts[-3]  # Train set의 target
                
                # History의 모든 아이템 카운트
                for item_id in history:
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
                
                # Target 아이템도 카운트
                item_counts[target] = item_counts.get(target, 0) + 1
        
        # 전체 아이템 수 파악
        all_item_ids = set(item_counts.keys())
        max_item_id = max(all_item_ids) if all_item_ids else 0
        
        print(f"  Total unique items in train set: {len(all_item_ids)}")
        print(f"  Max item ID: {max_item_id}")
        
        # 인기도 가중치 계산: log(count + 1)의 역수
        # 인기 많은 아이템 -> 낮은 가중치
        # 인기 없는 아이템 -> 높은 가중치
        item_weights = torch.ones(max_item_id + 1, device=self.device)
        
        for item_id, count in item_counts.items():
            # log(count + 1)의 역수
            item_weights[item_id] = 1.0 / np.log(count + 1)
        
        # 정규화 (평균이 1이 되도록)
        # 나타나지 않은 아이템은 1.0 유지
        appeared_mask = torch.zeros(max_item_id + 1, dtype=torch.bool, device=self.device)
        for item_id in all_item_ids:
            appeared_mask[item_id] = True
        
        if appeared_mask.sum() > 0:
            mean_weight = item_weights[appeared_mask].mean()
            item_weights[appeared_mask] = item_weights[appeared_mask] / mean_weight
        
        print(f"  Item popularity weight statistics:")
        print(f"    Min: {item_weights[appeared_mask].min().item():.4f}")
        print(f"    Max: {item_weights[appeared_mask].max().item():.4f}")
        print(f"    Mean: {item_weights[appeared_mask].mean().item():.4f}")
        print(f"    Std: {item_weights[appeared_mask].std().item():.4f}")
        
        # 예시 출력
        sorted_counts = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top-5 popular items:")
        for item_id, count in sorted_counts[:5]:
            weight = item_weights[item_id].item()
            print(f"    Item {item_id}: count={count}, weight={weight:.4f}")
        
        print(f"  Bottom-5 popular items:")
        for item_id, count in sorted_counts[-5:]:
            weight = item_weights[item_id].item()
            print(f"    Item {item_id}: count={count}, weight={weight:.4f}")
        
        print(f"✓ Item popularity weights computed")
        
        return item_weights
    
    def _compute_similarity_scores(
        self,
        generated_texts: List[str],
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity scores between generated texts and candidate set
        Args:
            generated_texts: [batch_size] generated texts
            user_ids: [batch_size] user ids
        Returns:
            ranks: [batch_size] ranks of target items
        """
        batch_size = len(generated_texts)
        
        # 1. 생성된 텍스트 임베딩 계산
        query_embeddings = self.emb_model.encode(
            generated_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device,
            batch_size=self.emb_batch_size,
        )  # [batch_size, emb_dim]
        
        batch_candidate_tensor = self.candidate_tensor[user_ids]
        scores = torch.bmm(query_embeddings.unsqueeze(1), self.item_embeddings[batch_candidate_tensor].transpose(1, 2)).squeeze(1)
        target_scores = scores[:, 0].unsqueeze(1)
        ranks = (scores > target_scores).sum(dim=1) + 1
        return ranks
    
    def __call__(
        self,
        generated_texts: List[str],
        user_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        TRL 호환 리워드 함수
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            user_ids: [batch_size] 사용자 ID (required)
            **kwargs: 추가 파라미터 (targets, histories 등은 무시됨)
        
        Returns:
            rewards: [batch_size] 리워드 값 
            
            If novelty_reward=True:
                rewards = NDCG × popularity_weight
                (인기 없는 아이템을 높은 rank로 예측할수록 높은 보상)
            
            Else:
                rewards = base_reward (NDCG/Hit/MRR 등)
        """
        
        # rank 계산 (target + negatives)
        ranks = self._compute_similarity_scores(generated_texts, user_ids)
        
        # 기본 리워드 타입에 따라 계산
        if self.reward_type == "ndcg":
            base_rewards = calculate_ndcg_from_rank(ranks, k=self.k)
        elif self.reward_type == "hit":
            base_rewards = calculate_hit_from_rank(ranks, k=self.k)
        elif self.reward_type == "mrr":
            base_rewards = calculate_mrr_from_rank(ranks, k=self.k)
        elif self.reward_type == "mixed":
            ndcg = calculate_ndcg_from_rank(ranks, k=self.k)
            hit = calculate_hit_from_rank(ranks, k=self.k)
            base_rewards = 0.7 * ndcg + 0.3 * hit
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
        
        # Novelty reward 사용 여부에 따라 분기
        if self.novelty_reward and self.item_popularity_weights is not None:
            # Novelty reward 사용: rewards = NDCG × item_popularity_weights
            batch_candidate_tensor = self.candidate_tensor[user_ids]  # [batch_size, k]
            target_item_ids = batch_candidate_tensor[:, 0]  # [batch_size] - target은 항상 첫 번째
            
            # Target item의 popularity weight
            item_weights = self.item_popularity_weights[target_item_ids]  # [batch_size]
            
            # Novelty = NDCG × popularity_weight
            rewards = calculate_novelty_ndcg(
                ranks, 
                item_weights=item_weights,
                k=self.k,
            )
        else:
            # 기본 리워드 사용
            rewards = base_rewards
        
        # 정규화 (optional)
        if self.normalize and rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return rewards
