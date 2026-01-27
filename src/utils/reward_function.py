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
import argparse


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
            # neg_items = neg_items[:k-1]
            # neg_items = np.random.choice(neg_items, size=k-1, replace=False).tolist()
            negative_pool[user_id] = neg_items
    
    print(f"✓ Loaded negative pool for {len(negative_pool)} users")
    if len(negative_pool) > 0:
        sample_user = next(iter(negative_pool))
        print(f"  Example: User {sample_user} has {len(negative_pool[sample_user])} negative items")
    
    return negative_pool


class SimilarHistoryItemMentionReward:
    """
    유저 구매기록 중 타겟과 가장 유사도가 높은 아이템의 title을 언급할 경우 보상
    임베딩 유사도 기반으로 가장 유사한 아이템을 캐싱하고, title의 첫 3단어를 언급하면 reward 1점 부여
    """
    
    def __init__(
        self,
        data_name: str,
        item_embeddings: torch.Tensor,
        uid_2_target: Dict[int, int],
        device: str = "cuda",
        data_dir: str = "data",
        use_position_weight: bool = False,
        position_decay: float = 1.0,
        similarity_threshold: float = 0.7,
    ):
        """
        Args:
            data_name: 데이터셋 이름
            item_embeddings: 사전 계산된 아이템 임베딩 [num_items+1, emb_dim]
            uid_2_target: 사용자 ID to 타겟 아이템 ID 매핑
            device: 디바이스
            data_dir: 데이터 디렉토리
            use_position_weight: 위치 기반 가중치 사용 여부
                                True이면 텍스트 앞쪽에 언급될수록 더 높은 보상
            position_decay: 위치 기반 감소율 (0.0 ~ 1.0)
                          0.0 = 위치 무관하게 동일 보상
                          1.0 = 텍스트 끝에서는 보상 0
                          예: 0.5이면 텍스트 끝에서 보상이 절반으로 감소
            similarity_threshold: 유사도 역치 (0.0 ~ 1.0)
                                이 값 이하이면 마지막 상호작용 아이템을 선택
        """
        self.__name__ = "SimilarHistoryItemMentionReward"
        self.data_name = data_name
        self.item_embeddings = item_embeddings
        self.device = device
        self.use_position_weight = use_position_weight
        self.position_decay = position_decay
        self.similarity_threshold = similarity_threshold
        
        # 아이템 메타데이터 로드 (title, brand, category)
        with open(f"{data_dir}/{data_name}/meta_text_fix.json", "r") as f:
            self.item_metadata = json.load(f)
        
        print(f"✓ SimilarHistoryItemMentionReward initialization started")
        print(f"  - Loaded metadata for {len(self.item_metadata)} items")
        print(f"  - Similarity threshold: {self.similarity_threshold}")
        print(f"    → If max similarity < threshold, use last interacted item")
        if self.use_position_weight:
            print(f"  - Position-based weighting: ENABLED (decay={self.position_decay})")
            print(f"    → Earlier mentions get higher rewards")
        else:
            print(f"  - Position-based weighting: DISABLED")
        
        # 캐시: user_id -> (most_similar_history_item_id, max_similarity)
        self.similarity_cache = {}
        
        # 전체 데이터에 대해 미리 유사한 아이템 계산
        print(f"  - Pre-computing most similar history items for all users...")
        self._precompute_similar_items(uid_2_target, data_name, data_dir)
        print(f"✓ Pre-computed similar items for {len(self.similarity_cache)} user-target pairs")
    
    def _precompute_similar_items(
        self,
        uid_2_target: Dict[int, int],
        data_name: str,
        data_dir: str
    ):
        """
        전체 데이터에 대해 타겟과 가장 유사한 히스토리 아이템을 미리 계산
        유사도가 역치 이하이면 마지막 상호작용 아이템을 선택
        
        Args:
            uid_2_target: 사용자 ID to 타겟 아이템 ID 매핑
            data_name: 데이터셋 이름
            data_dir: 데이터 디렉토리
        """
        # sequential_data.txt에서 히스토리 정보 로드
        sequential_file = f"{data_dir}/{data_name}/sequential_data.txt"
        
        # 정규화된 임베딩 미리 계산 (전체 아이템)
        normalized_embeddings = torch.nn.functional.normalize(self.item_embeddings, p=2, dim=1)
        
        fallback_count = 0  # 역치 미만으로 마지막 아이템 사용한 횟수
        
        with open(sequential_file, 'r') as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                history = parts[1:-3]  # Train set의 history
                target_id = parts[-3]   # Train set의 target
                
                # uid_2_target에 해당하는 사용자만 처리
                if user_id not in uid_2_target:
                    continue
                
                # 히스토리가 비어있으면 스킵
                if len(history) == 0:
                    continue
                
                # 타겟 임베딩 (정규화됨)
                target_emb = normalized_embeddings[target_id]  # [emb_dim]
                
                # 히스토리 임베딩 (정규화됨)
                history_ids = torch.tensor(history, dtype=torch.long, device=self.device)
                history_embs = normalized_embeddings[history_ids]  # [history_len, emb_dim]
                
                # 코사인 유사도 계산
                similarities = torch.mm(target_emb.unsqueeze(0), history_embs.T).squeeze(0)  # [history_len]
                
                # 가장 유사한 아이템 찾기
                max_similarity = similarities.max().item()
                most_similar_idx = similarities.argmax().item()
                
                # 유사도가 역치 이하이면 마지막 상호작용 아이템 선택
                if max_similarity < self.similarity_threshold:
                    selected_item_id = history[-1]  # 마지막 아이템
                    fallback_count += 1
                else:
                    selected_item_id = history[most_similar_idx]
                
                # 캐시에 저장 (아이템 ID와 최대 유사도)
                self.similarity_cache[user_id] = (selected_item_id, max_similarity)
        
        # 통계 출력
        total_users = len(self.similarity_cache)
        if total_users > 0:
            fallback_ratio = (fallback_count / total_users) * 100
            print(f"  - Fallback to last item: {fallback_count}/{total_users} ({fallback_ratio:.1f}%)")

    
    def _get_most_similar_history_item(
        self,
        user_id: int,
    ) -> int:
        """
        히스토리 중 타겟과 가장 유사한 아이템 찾기 (캐시에서 가져오거나 실시간 계산)
        유사도가 역치 이하이면 마지막 상호작용 아이템 반환
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            selected_item_id: 선택된 히스토리 아이템 ID
                            (유사도 역치 이상: 가장 유사한 아이템,
                             유사도 역치 미만: 마지막 상호작용 아이템)
        """        
        selected_item_id, _ = self.similarity_cache[user_id]
        return selected_item_id
    
    def _get_first_three_words(self, title: str) -> str:
        """
        Title의 첫 3단어 추출
        
        Args:
            title: 아이템 title
            
        Returns:
            first_three_words: 첫 3단어를 공백으로 연결한 문자열 (소문자)
        """
        words = title.strip().split()
        first_three = " ".join(words[:3])
        return first_three.lower()
    
    def _calculate_position_weight(self, position: int, text_length: int) -> float:
        """
        위치 기반 가중치 계산
        
        Args:
            position: 언급된 위치 (문자 인덱스)
            text_length: 전체 텍스트 길이
            
        Returns:
            weight: 위치 기반 가중치 (0.0 ~ 1.0)
                   앞쪽일수록 1.0에 가깝고, 뒤쪽일수록 감소
        """
        if text_length == 0:
            return 1.0
        
        # 상대적 위치 계산 (0.0 = 맨 앞, 1.0 = 맨 뒤)
        position_ratio = position / text_length
        
        # 가중치 계산: 1.0 - (position_ratio * decay)
        # decay=0.0 → 위치 무관하게 1.0
        # decay=1.0 → 맨 뒤에서는 0.0
        # decay=0.5 → 맨 뒤에서는 0.5
        weight = 1.0 - (position_ratio * self.position_decay)
        
        return max(0.0, weight)  # 최소값 0.0 보장
    
    def __call__(
        self,
        generated_texts: List[str],
        targets: List[int],
        histories: List[List[int]],
        user_ids: List[int],
        **kwargs
    ) -> List[float]:
        """
        생성된 텍스트에서 유사한 히스토리 아이템의 title 언급 여부를 확인하여 보상
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            targets: [batch_size] 타겟 아이템 ID
            histories: [batch_size, *] 히스토리 아이템 ID 리스트
            user_ids: [batch_size] 사용자 ID
            
        Returns:
            rewards: [batch_size] 보상 값
                    - use_position_weight=False: 0 또는 1.0
                    - use_position_weight=True: 0 ~ 1.0 (위치에 따라 가중)
        """
        rewards = []
        
        for gen_text, target_id, history_ids, user_id in zip(generated_texts, targets, histories, user_ids):
            reward = 0.0
            
            # 가장 유사한 히스토리 아이템 찾기
            most_similar_item_id = self._get_most_similar_history_item(user_id)
            
            # 해당 아이템의 title 가져오기
            if str(most_similar_item_id) in self.item_metadata:
                item_title = self.item_metadata[str(most_similar_item_id)]["title"]
                first_three_words = self._get_first_three_words(item_title)
                
                # 생성된 텍스트에 첫 3단어가 포함되어 있는지 확인 (대소문자 무시)
                gen_text_lower = gen_text.lower()
                if first_three_words in gen_text_lower:
                    if self.use_position_weight:
                        # 위치 기반 가중치 적용
                        position = gen_text_lower.find(first_three_words)
                        text_length = len(gen_text_lower)
                        weight = self._calculate_position_weight(position, text_length)
                        reward = 1.0 * weight
                    else:
                        # 위치 무관하게 1.0점
                        reward = 1.0
            
            rewards.append(reward)
        
        return rewards


class BrandMentionReward:
    """
    타겟 아이템의 브랜드를 언급할 경우 보상 (0.5점)
    """
    
    def __init__(
        self,
        data_name: str,
        device: str = "cuda",
        data_dir: str = "data",
    ):
        """
        Args:
            data_name: 데이터셋 이름
            device: 디바이스
            data_dir: 데이터 디렉토리
        """
        self.__name__ = "BrandMentionReward"
        self.data_name = data_name
        self.device = device
        
        # 아이템 메타데이터 로드
        with open(f"{data_dir}/{data_name}/meta_text_fix.json", "r") as f:
            item_metadata = json.load(f)
            item_metadata = {int(k): v for k, v in item_metadata.items()}
        self.item_brands = {item_id: str(item_metadata[item_id]["brand"]) for item_id in item_metadata}
        
        print(f"✓ BrandMentionReward initialized")
        print(f"  - Loaded brands for {len(self.item_brands)} items")
    
    def __call__(
        self,
        generated_texts: List[str],
        targets: List[int],
        **kwargs
    ) -> List[float]:
        """
        생성된 텍스트에서 타겟 아이템의 브랜드 언급 여부를 확인하여 보상
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            targets: [batch_size] 타겟 아이템 ID
            
        Returns:
            rewards: [batch_size] 보상 값 (0 또는 0.5)
        """
        rewards = []
        
        for gen_text, target_id in zip(generated_texts, targets):
            reward = 0.0
            if self.item_brands[target_id].lower() in gen_text.lower():
                reward = 0.5
            rewards.append(reward)
        return rewards


class CategoryMentionReward:
    """
    타겟 아이템의 카테고리를 언급할 경우 보상 (0.5점)
    """
    
    def __init__(
        self,
        data_name: str,
        device: str = "cuda",
        data_dir: str = "data",
    ):
        """
        Args:
            data_name: 데이터셋 이름
            device: 디바이스
            data_dir: 데이터 디렉토리
        """
        self.__name__ = "CategoryMentionReward"
        self.data_name = data_name
        self.device = device
        
        # 아이템 메타데이터 로드
        with open(f"{data_dir}/{data_name}/meta_text_fix.json", "r") as f:
            item_metadata = json.load(f)
            item_metadata = {int(k): v for k, v in item_metadata.items()}
        self.item_categories = {item_id: str(item_metadata[item_id]["category"]) for item_id in item_metadata}
        print(f"✓ CategoryMentionReward initialized")
        print(f"  - Loaded categories for {len(self.item_categories)} items")
    
    def __call__(
        self,
        generated_texts: List[str],
        targets: List[int],
        **kwargs
    ) -> List[float]:
        """
        생성된 텍스트에서 타겟 아이템의 카테고리 언급 여부를 확인하여 보상
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            targets: [batch_size] 타겟 아이템 ID
            
        Returns:
            rewards: [batch_size] 보상 값 (0 또는 0.5)
        """
        rewards = []
        
        for gen_text, target_id in zip(generated_texts, targets):
            reward = 0.0
            if self.item_categories[target_id].lower() in gen_text.lower():
                reward = 0.5
            rewards.append(reward)
        
        return rewards


class LocalEmbeddingRewardFunction:
    """
    로컬 임베딩 기반 리워드 함수
    DB 대신 자체적으로 negative item 임베딩을 계산하여 NDCG를 reward로 활용
    """
    
    def __init__(
        self,
        args: argparse.Namespace,
        uid_2_target: Dict[int, int],
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
            novelty_coef: Novelty 리워드 계수 (default: 1.0)
            novelty_target_rank: (사용 안함, backward compatibility)
            novelty_mode: (사용 안함, backward compatibility)
            novelty_annealing: Novelty annealing 사용 여부
                              True이면 학습 진행도에 따라 novelty 비율을 0→1로 선형 증가
                              Final reward = (1-ratio)*base + ratio*novelty_coef*novelty
            popularity_coef: Popularity 리워드 계수 (0.0 = 사용 안함)
                            정답인 경우에만 popularity bonus 추가
            target_emb_reward: 타겟 임베딩 유사도 리워드 사용 여부
            target_emb_coef: 타겟 임베딩 리워드 계수
            infonce_reward: InfoNCE (대조 학습) 리워드 사용 여부
            infonce_coef: InfoNCE 리워드 계수
            infonce_temperature: InfoNCE temperature 파라미터 (default: 0.07)
            infonce_emb_type: InfoNCE용 임베딩 타입 (None이면 emb_type과 동일)
            proxy_label_reward: Proxy label 리워드 사용 여부
                               True이면 타겟과 유사한 상위 proxy_k개 아이템도 부분적으로 정답으로 취급
                               기존 base_reward에 추가로 더해짐
            proxy_k: Proxy label로 사용할 유사한 아이템 개수
            proxy_label_coef: Proxy label 리워드 계수
            max_steps: 최대 학습 스텝 수 (novelty annealing 계산에 사용)
        """
        self.__name__ = "LocalEmbeddingRewardFunction"
        self.args = args
        self.data_name = args.data_name
        self.reward_type = args.reward_type
        self.k = args.k
        self.normalize = args.normalize_rewards
        self.device = args.device
        self.emb_batch_size = args.emb_batch_size
        self.uid_2_target = uid_2_target  # Store for full item pool ranking
        
        # Novelty 관련 파라미터
        self.novelty_reward = args.novelty_reward
        self.novelty_coef = args.novelty_coef
        self.novelty_target_rank = args.novelty_target_rank
        self.novelty_mode = args.novelty_mode
        self.novelty_annealing = args.novelty_annealing
        
        # Popularity 관련 파라미터
        self.popularity_coef = args.popularity_coef
        
        # Target embedding 유사도 리워드 파라미터
        self.target_emb_reward = args.target_emb_reward
        self.target_emb_file = args.target_emb_file
        self.target_emb_coef = args.target_emb_coef
        
        # InfoNCE 리워드 파라미터
        self.infonce_reward = args.infonce_reward
        self.infonce_coef = args.infonce_coef
        self.infonce_temperature = args.infonce_temperature
        self.infonce_emb_type = args.infonce_emb_type if args.infonce_emb_type is not None else args.emb_type
        
        # Proxy label 리워드 파라미터
        if hasattr(args, "proxy_label_reward"):
            self.proxy_label_reward = args.proxy_label_reward
            self.proxy_k = args.proxy_k
            self.proxy_label_coef = args.proxy_label_coef
        else:
            self.proxy_label_reward = False
            self.proxy_k = 0
            self.proxy_label_coef = 0
        
        # Training 관련 파라미터
        self.max_steps = args.max_steps
        
        print(f"💰 Reward configuration:")
        print(f"  - Reward type: {self.reward_type}")
        print(f"  - Top-K: {self.k}")
        print(f"  - Normalize: {self.normalize}")
        if self.novelty_reward:
            print(f"  - Novelty reward: ENABLED")
            print(f"  - Novelty coefficient: {self.novelty_coef}")
            print(f"  - Novelty = NDCG × popularity_weight (인기 없는 아이템 장려)")
            if self.novelty_annealing:
                print(f"  - Novelty annealing: ENABLED")
                print(f"  - Novelty ratio will increase linearly from 0 to 1 over {self.max_steps} steps")
                print(f"  - Final reward = (1-ratio)*base + ratio*novelty")
        if self.popularity_coef > 0:
            print(f"  - Popularity coefficient: {self.popularity_coef}")
            print(f"  - Popularity bonus for unpopular items (when correct)")
        if self.target_emb_reward:
            print(f"  - Target embedding reward: ENABLED")
            print(f"  - Target embedding file: {self.target_emb_file}")
            print(f"  - Target embedding coefficient: {self.target_emb_coef}")
            print(f"  - Reward based on cosine similarity with target embedding")
        if self.infonce_reward:
            print(f"  - InfoNCE reward: ENABLED")
            print(f"  - InfoNCE coefficient: {self.infonce_coef}")
            print(f"  - InfoNCE temperature: {self.infonce_temperature}")
            print(f"  - InfoNCE embedding type: {self.infonce_emb_type}")
            print(f"  - Contrastive learning: maximize target similarity, minimize negative similarity")
        if self.proxy_label_reward:
            print(f"  - Proxy label reward: ENABLED")
            print(f"  - Proxy K: {self.proxy_k}")
            print(f"  - Proxy label coefficient: {self.proxy_label_coef}")
            print(f"  - Use top-{self.proxy_k} similar items as soft labels with similarity-weighted NDCG")
            print(f"  - Final reward = base_reward + proxy_label_coef * proxy_label_ndcg")
        
        # 임베딩 모델 로드
        print(f"🤖 Loading embedding model: {args.emb_model_name}")
        from sentence_transformers import SentenceTransformer
        self.emb_model = SentenceTransformer(args.emb_model_name, device=self.device)
        print(f"✓ Embedding model loaded on {self.device}")

        total_user_count = 0
        sequential_file = f"data/{self.data_name}/sequential_data.txt"        
        with open(sequential_file, 'r') as f:
            for line in f:
                total_user_count += 1
        
        # k > 100이면 전체 아이템 풀 사용, 그렇지 않으면 negative pool 사용
        self.use_full_item_pool = (self.k > 100)
        if self.use_full_item_pool:
            print(f"⚠️ k={self.k} > 100: Using full item pool for ranking (no negative sampling)")
            self.negative_pool = None
            self.candidate_tensor = None  # Will use full item embeddings
        else:
            # Negative pool 로드
            self.negative_pool = load_negative_pool(self.data_name, args.data_dir, self.k)
            # prepare candidate set, target comes first
            self.candidate_tensor = self._prepare_candidate_tensor(total_user_count, uid_2_target, self.negative_pool)
        
        # 사전 계산된 아이템 임베딩 로드
        emb_model_name_dir = args.emb_model_name.split("/")[-1]
        item_embedding_file_path = f"data_emb/{self.data_name}_{args.emb_type}_{emb_model_name_dir}_emb.pt"
        print(f"📦 Loading pre-computed item embeddings from: {item_embedding_file_path}")
        self.item_embeddings = torch.load(item_embedding_file_path, map_location=self.device)
        print(f"✓ Loaded embeddings for {len(self.item_embeddings)} items")
        
        # InfoNCE용 추가 임베딩 로드 (필요 시)
        if self.infonce_reward and self.infonce_emb_type != args.emb_type:
            infonce_embedding_file_path = f"data_emb/{self.data_name}_{self.infonce_emb_type}_{emb_model_name_dir}_emb.pt"
            print(f"📦 Loading InfoNCE embeddings from: {infonce_embedding_file_path}")
            self.infonce_item_embeddings = torch.load(infonce_embedding_file_path, map_location=self.device)
            print(f"✓ Loaded InfoNCE embeddings for {len(self.infonce_item_embeddings)} items")
        else:
            # 같은 임베딩 사용
            self.infonce_item_embeddings = self.item_embeddings if self.infonce_reward else None
        
        # Proxy label을 위한 아이템 간 유사도 로드 또는 계산
        if self.proxy_label_reward:
            # 저장된 proxy labels 파일 확인
            proxy_labels_file = f"data_emb/{self.data_name}_proxy_labels_k100_{args.emb_type}_{emb_model_name_dir}.json"
            proxy_labels_path = Path(proxy_labels_file)
            
            if proxy_labels_path.exists():
                print(f"📦 Loading pre-computed proxy labels from: {proxy_labels_file}")
                self.item_proxy_labels = self._load_proxy_labels(proxy_labels_path)
                print(f"✓ Loaded proxy labels for {len(self.item_proxy_labels)} items")
            else:
                print(f"⚠️  Pre-computed proxy labels not found: {proxy_labels_file}")
                print(f"   Computing proxy labels on-the-fly (this may take time)...")
                exit()

        else:
            self.item_proxy_labels = None
        
        # Target embeddings 준비 (target_emb_reward 사용 시)
        if self.target_emb_reward:
            self.target_embeddings = self._prepare_target_embeddings(uid_2_target)
            print(f"✓ Prepared target embeddings for {len(uid_2_target)} users")
        else:
            self.target_embeddings = None
        
        # 아이템 인기도 계산 (train set에서)
        # Novelty 또는 Popularity reward 사용 시 필요
        if self.novelty_reward or self.popularity_coef > 0:
            self.item_popularity_weights = self._compute_item_popularity(
                uid_2_target, self.negative_pool, self.data_name, args.data_dir
            )
        else:
            self.item_popularity_weights = None

    def _prepare_candidate_tensor(self, total_user_count: int, uid_2_target: Dict[int, int], neg_pool: Dict[int, List[int]]) -> torch.Tensor:
        candidate_tensor = torch.zeros(total_user_count+1, len(list(neg_pool.values())[0])+1, dtype=torch.long)
        for uid, target_id in uid_2_target.items():
            candidate_tensor[uid] = torch.tensor([target_id] + neg_pool[uid], dtype=torch.long)
        return candidate_tensor
    
    def _load_proxy_labels(self, file_path: Path) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        저장된 proxy labels를 로드
        
        Args:
            file_path: proxy labels JSON 파일 경로
            
        Returns:
            item_proxy_labels: Dict[item_id, (proxy_item_ids, proxy_similarities)]
        """
        with open(file_path, 'r') as f:
            proxy_labels_json = json.load(f)
        
        # JSON에서 로드한 데이터를 텐서로 변환
        item_proxy_labels = {}
        for item_id_str, proxy_list in proxy_labels_json.items():
            proxy_list = proxy_list[:self.proxy_k]
            item_id = int(item_id_str)
            
            # List[Tuple[item_id, similarity]]를 두 개의 텐서로 분리
            proxy_ids = torch.tensor([p[0] for p in proxy_list], dtype=torch.long, device=self.device)
            proxy_sims = torch.tensor([p[1] for p in proxy_list], dtype=torch.float32, device=self.device)
            
            item_proxy_labels[item_id] = (proxy_ids, proxy_sims)
        
        return item_proxy_labels
    
    def _prepare_target_embeddings(self, uid_2_target: Dict[int, int]) -> torch.Tensor:
        """
        각 사용자의 타겟 아이템 임베딩을 준비
        
        Args:
            uid_2_target: 사용자 ID to 타겟 아이템 ID 매핑
            
        Returns:
            target_embeddings: [max_uid+1, emb_dim] 각 사용자의 타겟 임베딩
        """
        if self.target_emb_file is not None:
            target_embeddings = torch.load(f"data_emb/{self.args.target_emb_file}", map_location=self.device)
            return target_embeddings
        
        max_uid = max(uid_2_target.keys())
        emb_dim = self.item_embeddings.shape[1]
        
        # 사용자별 타겟 임베딩 텐서 초기화
        target_embeddings = torch.zeros(max_uid + 1, emb_dim, device=self.device)
        
        for uid, target_id in uid_2_target.items():
            target_embeddings[uid] = self.item_embeddings[target_id]
        
        return target_embeddings
    
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
    
    def _encode_texts(self, generated_texts: List[str]) -> torch.Tensor:
        """
        생성된 텍스트를 임베딩으로 변환
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            
        Returns:
            embeddings: [batch_size, emb_dim] 임베딩
        """
        embeddings = self.emb_model.encode(
            generated_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device,
            batch_size=self.emb_batch_size,
        )
        return embeddings
    
    def _compute_similarity_scores(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute similarity scores between query embeddings and candidate set
        Args:
            query_embeddings: [batch_size, emb_dim] query embeddings
            user_ids: [batch_size] user ids
            return_scores: if True, return (ranks, scores), otherwise return (ranks, None)
        Returns:
            ranks: [batch_size] ranks of target items
            scores: [batch_size, num_candidates] similarity scores (only if return_scores=True)
        """
        if self.use_full_item_pool:
            # 전체 아이템 풀에 대해 유사도 계산
            # query_embeddings: [batch_size, emb_dim]
            # item_embeddings: [num_items, emb_dim]
            scores = torch.mm(query_embeddings, self.item_embeddings.T)  # [batch_size, num_items]
            
            # 각 사용자의 target item 가져오기
            target_item_ids = torch.tensor(
                [self.uid_2_target[uid] for uid in user_ids],
                device=self.device
            )  # [batch_size]
            
            # Target item의 점수
            target_scores = scores[torch.arange(scores.size(0), device=self.device), target_item_ids]  # [batch_size]
            
            # Rank 계산: target보다 높은 점수를 가진 아이템의 개수 + 1
            ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
        else:
            # Negative pool 기반 계산 (기존 로직)
            batch_candidate_tensor = self.candidate_tensor[user_ids]
            scores = torch.bmm(query_embeddings.unsqueeze(1), self.item_embeddings[batch_candidate_tensor].transpose(1, 2)).squeeze(1)
            target_scores = scores[:, 0].unsqueeze(1)
            ranks = (scores > target_scores).sum(dim=1) + 1
        
        if return_scores:
            return ranks, scores
        else:
            return ranks, None
    
    def _compute_proxy_label_ndcg(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
        predicted_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Proxy label 기반 NDCG 계산
        미리 계산된 아이템 간 유사도를 활용하여 타겟의 proxy labels을 soft label로 사용
        
        Args:
            query_embeddings: [batch_size, emb_dim] 쿼리 임베딩
            user_ids: [batch_size] 사용자 ID
            predicted_scores: [batch_size, num_items] 또는 [batch_size, num_candidates] 예측 점수
            
        Returns:
            ndcg_rewards: [batch_size] Proxy label 기반 NDCG 리워드
        """
        batch_size = len(user_ids)
        ndcg_rewards = torch.zeros(batch_size, device=self.device)
        
        # 타겟 아이템 ID 가져오기
        target_item_ids = torch.tensor(
            [self.uid_2_target[uid] for uid in user_ids],
            device=self.device
        )  # [batch_size]
        
        if self.use_full_item_pool:
            # 전체 아이템 풀 사용
            num_items = len(self.item_embeddings)
            
            for i in range(batch_size):
                target_id = target_item_ids[i].item()
                
                # # 타겟 아이템의 미리 계산된 proxy labels 가져오기
                # if target_id not in self.item_proxy_labels:
                #     # Proxy labels가 없으면 타겟만 1.0
                #     proxy_ids = torch.tensor([], dtype=torch.long, device=self.device)
                #     proxy_sims = torch.tensor([], dtype=torch.float32, device=self.device)
                # else:
                proxy_ids, proxy_sims = self.item_proxy_labels[target_id]
                
                # Relevance scores 생성: 타겟 자신은 1.0, proxy는 유사도 비례
                relevance_scores = torch.zeros(num_items, device=self.device)
                relevance_scores[target_id] = 1.0  # 타겟 자신
                relevance_scores[proxy_ids] = proxy_sims  # Proxy labels
                
                # 예측 점수 기준으로 Top-K 추출
                pred_scores = predicted_scores[i]  # [num_items]
                top_k_pred_scores, top_k_pred_indices = torch.topk(pred_scores, k=min(self.k, len(pred_scores)))
                
                # Top-K 예측 결과에서 relevance 추출
                predicted_relevance = relevance_scores[top_k_pred_indices]  # [k]
                
                # DCG 계산
                dcg = calculate_dcg(predicted_relevance.unsqueeze(0), k=self.k)[0]
                
                # IDCG 계산 (이상적인 경우: relevance가 높은 순서대로 정렬)
                ideal_relevance, _ = torch.sort(relevance_scores, descending=True)
                ideal_relevance = ideal_relevance[:self.k]
                idcg = calculate_dcg(ideal_relevance.unsqueeze(0), k=self.k)[0]
                
                # NDCG 계산
                if idcg > 0:
                    ndcg_rewards[i] = dcg / (idcg + 1e-10)
                else:
                    ndcg_rewards[i] = 0.0
        else:
            # Candidate set 기반 계산
            for i in range(batch_size):
                target_id = target_item_ids[i].item()
                
                # Candidate set 가져오기
                batch_candidate_tensor = self.candidate_tensor[user_ids[i]]  # [num_candidates]
                num_candidates = len(batch_candidate_tensor)
                
                # 타겟 아이템의 미리 계산된 proxy labels 가져오기
                if target_id not in self.item_proxy_labels:
                    # Proxy labels가 없으면 타겟만 1.0
                    proxy_ids = torch.tensor([], dtype=torch.long, device=self.device)
                    proxy_sims = torch.tensor([], dtype=torch.float32, device=self.device)
                else:
                    proxy_ids, proxy_sims = self.item_proxy_labels[target_id]
                
                # Candidate set 내에서 relevance scores 생성
                relevance_scores = torch.zeros(num_candidates, device=self.device)
                
                # 타겟 아이템이 candidate set에 있는 위치 찾기 (보통 index 0)
                target_mask = batch_candidate_tensor == target_id
                if target_mask.any():
                    target_idx_in_candidates = target_mask.nonzero(as_tuple=True)[0][0]
                    relevance_scores[target_idx_in_candidates] = 1.0
                
                # Proxy labels도 candidate set에 있는지 확인하고 relevance 할당
                if len(proxy_ids) > 0:
                    for proxy_id, proxy_sim in zip(proxy_ids, proxy_sims):
                        proxy_mask = batch_candidate_tensor == proxy_id.item()
                        if proxy_mask.any():
                            proxy_idx_in_candidates = proxy_mask.nonzero(as_tuple=True)[0][0]
                            relevance_scores[proxy_idx_in_candidates] = proxy_sim.item()
                
                # 예측 점수 기준으로 Top-K 추출
                pred_scores = predicted_scores[i]  # [num_candidates]
                top_k_pred_scores, top_k_pred_indices = torch.topk(pred_scores, k=min(self.k, len(pred_scores)))
                
                # Top-K 예측 결과에서 relevance 추출
                predicted_relevance = relevance_scores[top_k_pred_indices]  # [k]
                
                # DCG 계산
                dcg = calculate_dcg(predicted_relevance.unsqueeze(0), k=self.k)[0]
                
                # IDCG 계산
                ideal_relevance, _ = torch.sort(relevance_scores, descending=True)
                ideal_relevance = ideal_relevance[:self.k]
                idcg = calculate_dcg(ideal_relevance.unsqueeze(0), k=self.k)[0]
                
                # NDCG 계산
                if idcg > 0:
                    ndcg_rewards[i] = dcg / (idcg + 1e-10)
                else:
                    ndcg_rewards[i] = 0.0
        
        return ndcg_rewards
    
    def _compute_target_embedding_reward(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        타겟 임베딩과의 유사도 기반 리워드 계산
        
        생성된 텍스트가 타겟 임베딩과 유사할수록, 
        그리고 negative 임베딩들보다 타겟과 더 유사할수록 높은 리워드
        
        Args:
            query_embeddings: [batch_size, emb_dim] 쿼리 임베딩
            user_ids: [batch_size] 사용자 ID
            
        Returns:
            rewards: [batch_size] 타겟 임베딩 유사도 리워드
                    (타겟 유사도 - negative 평균 유사도)
        """
        # L2 정규화 (코사인 유사도를 위해)
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        
        # 2. 타겟 임베딩과의 유사도 계산
        target_embs = self.target_embeddings[user_ids]  # [batch_size, emb_dim]
        target_embs = torch.nn.functional.normalize(target_embs, p=2, dim=1)
        
        target_similarities = (query_embeddings * target_embs).sum(dim=1)  # [batch_size]
        
        if self.use_full_item_pool:
            # 전체 아이템 풀 사용: 타겟을 제외한 모든 아이템을 negative로 사용
            # 메모리 효율을 위해 샘플링하거나, 전체 아이템의 평균 유사도를 계산
            # 여기서는 전체 아이템 임베딩의 평균 유사도를 사용
            all_item_embs = torch.nn.functional.normalize(self.item_embeddings, p=2, dim=1)  # [num_items, emb_dim]
            
            # 타겟 아이템 ID
            target_item_ids = torch.tensor(
                [self.uid_2_target[uid] for uid in user_ids],
                device=self.device
            )  # [batch_size]
            
            # 전체 아이템과의 유사도 계산
            all_similarities = torch.mm(query_embeddings, all_item_embs.T)  # [batch_size, num_items]
            
            # 타겟을 제외한 평균 유사도 계산
            batch_size = all_similarities.size(0)
            num_items = all_similarities.size(1)
            
            # 타겟 마스크 생성
            mask = torch.ones(batch_size, num_items, device=self.device, dtype=torch.bool)
            mask[torch.arange(batch_size, device=self.device), target_item_ids] = False
            
            # 타겟을 제외한 negative들의 평균 유사도
            negative_mean_similarities = all_similarities[mask].view(batch_size, -1).mean(dim=1)
            rewards = target_similarities - torch.clamp(negative_mean_similarities, min=0.0)
        else:
            # # 3. Negative 임베딩들과의 평균 유사도 계산 (기존 로직)
            # batch_candidate_tensor = self.candidate_tensor[user_ids]  # [batch_size, k]
            # negative_ids = batch_candidate_tensor[:, 1:]  # [batch_size, k-1] (첫 번째는 target 제외)
            
            # # 전체 타겟 임베딩과의 유사도 계산
            # all_similarities = torch.mm(query_embeddings, self.target_embeddings.T)  # [batch_size, num_users]
            
            # # 평균 유사도 계산
            # negative_mean_similarities = all_similarities.mean(dim=1)  # [batch_size]
            rewards = target_similarities
                
        return rewards
    
    def _compute_infonce_reward(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE (대조 학습) 리워드 계산
        
        InfoNCE loss를 reward로 변환:
        reward = log(exp(sim(q, pos)/tau) / (exp(sim(q, pos)/tau) + sum(exp(sim(q, neg_i)/tau))))
        
        타겟과의 유사도는 높이고, negative들과의 유사도는 낮추도록 장려
        
        Args:
            query_embeddings: [batch_size, emb_dim] 쿼리 임베딩
            user_ids: [batch_size] 사용자 ID
            
        Returns:
            rewards: [batch_size] InfoNCE 리워드 (높을수록 좋음)
        """
        # L2 정규화 (코사인 유사도를 위해)
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        
        if self.use_full_item_pool:
            # 전체 아이템 풀 사용
            # 타겟 아이템 ID
            target_ids = torch.tensor(
                [self.uid_2_target[uid] for uid in user_ids],
                device=self.device
            )  # [batch_size]
            
            # InfoNCE용 임베딩 사용
            target_embs = self.infonce_item_embeddings[target_ids]  # [batch_size, emb_dim]
            target_embs = torch.nn.functional.normalize(target_embs, p=2, dim=1)
            
            all_item_embs = torch.nn.functional.normalize(self.infonce_item_embeddings, p=2, dim=1)  # [num_items, emb_dim]
            
            # 1. 타겟과의 유사도 계산
            pos_sim = (query_embeddings * target_embs).sum(dim=1)  # [batch_size]
            pos_sim = pos_sim / self.infonce_temperature
            
            # 2. 전체 아이템들과의 유사도 계산 (타겟 제외)
            all_sims = torch.mm(query_embeddings, all_item_embs.T)  # [batch_size, num_items]
            all_sims = all_sims / self.infonce_temperature
            
            # 3. InfoNCE 계산
            # log(exp(pos_sim) / sum(exp(all_sims)))
            log_sum_exp = torch.logsumexp(all_sims, dim=1)  # [batch_size]
            infonce_rewards = pos_sim - log_sum_exp  # [batch_size]
        else:
            # Candidate tensor 가져오기 (기존 로직)
            batch_candidate_tensor = self.candidate_tensor[user_ids]  # [batch_size, k]
            target_ids = batch_candidate_tensor[:, 0]  # [batch_size] - target
            negative_ids = batch_candidate_tensor[:, 1:]  # [batch_size, k-1] - negatives
            
            # InfoNCE용 임베딩 사용
            target_embs = self.infonce_item_embeddings[target_ids]  # [batch_size, emb_dim]
            target_embs = torch.nn.functional.normalize(target_embs, p=2, dim=1)
            
            negative_embs = self.infonce_item_embeddings[negative_ids]  # [batch_size, k-1, emb_dim]
            negative_embs = torch.nn.functional.normalize(negative_embs, p=2, dim=2)
            
            # 1. 타겟과의 유사도 계산
            pos_sim = (query_embeddings * target_embs).sum(dim=1)  # [batch_size]
            pos_sim = pos_sim / self.infonce_temperature
            
            # 2. Negative들과의 유사도 계산
            neg_sims = torch.bmm(
                query_embeddings.unsqueeze(1),  # [batch_size, 1, emb_dim]
                negative_embs.transpose(1, 2)   # [batch_size, emb_dim, k-1]
            ).squeeze(1)  # [batch_size, k-1]
            neg_sims = neg_sims / self.infonce_temperature
            
            # 3. InfoNCE 계산
            # log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sims))))
            # = pos_sim - log(exp(pos_sim) + sum(exp(neg_sims)))
            # = pos_sim - logsumexp([pos_sim, neg_sims])
            
            all_sims = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)  # [batch_size, k]
            log_sum_exp = torch.logsumexp(all_sims, dim=1)  # [batch_size]
            
            infonce_rewards = pos_sim - log_sum_exp  # [batch_size]
        
        return infonce_rewards
    
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
            **kwargs: 추가 파라미터 (targets, histories, trainer_state 등)
        
        Returns:
            rewards: [batch_size] 리워드 값 
            
            If proxy_label_reward=True:
                rewards = base_reward + proxy_label_coef × proxy_label_ndcg
                (타겟과 유사한 아이템들도 부분적으로 정답으로 취급)
            
            If novelty_reward=True and novelty_annealing=False:
                rewards = novelty_coef × (NDCG × popularity_weight)
                (인기 없는 아이템을 높은 rank로 예측할수록 높은 보상)
            
            If novelty_reward=True and novelty_annealing=True:
                novelty_ratio = current_step / max_steps (0 → 1 선형 증가)
                rewards = (1 - novelty_ratio) * base_reward + novelty_ratio * novelty_coef * novelty_reward
            
            Else:
                rewards = base_reward (NDCG/Hit/MRR 등)
        """
        
        # 생성된 텍스트를 임베딩으로 변환 (한 번만 수행)
        query_embeddings = self._encode_texts(generated_texts)
        
        # 기존 rank 기반 리워드 계산
        # rank 계산 (target + negatives)
        ranks, _ = self._compute_similarity_scores(query_embeddings, user_ids, return_scores=False)
        
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
        # Proxy label reward 사용 여부에 따라 분기
        if self.proxy_label_reward:
            # Proxy label 리워드 사용 시: 기존 base_reward + proxy_label_reward
            # 예측 점수도 함께 계산 필요
            _, predicted_scores = self._compute_similarity_scores(query_embeddings, user_ids, return_scores=True)
            # 2. Proxy label NDCG 계산
            proxy_label_rewards = self._compute_proxy_label_ndcg(query_embeddings, user_ids, predicted_scores)
            
            # 3. 두 리워드를 합산
            base_rewards = base_rewards + self.proxy_label_coef * proxy_label_rewards
        
        # Novelty reward 사용 여부에 따라 분기
        if self.novelty_reward and self.item_popularity_weights is not None:
            # Novelty reward 계산
            if self.use_full_item_pool:
                # 전체 아이템 풀 사용 시: uid_2_target에서 직접 가져오기
                target_item_ids = torch.tensor(
                    [self.uid_2_target[uid] for uid in user_ids],
                    device=self.device
                )  # [batch_size]
            else:
                # Negative pool 사용 시: candidate_tensor에서 가져오기
                batch_candidate_tensor = self.candidate_tensor[user_ids]  # [batch_size, k]
                target_item_ids = batch_candidate_tensor[:, 0]  # [batch_size] - target은 항상 첫 번째
            
            # Target item의 popularity weight
            item_weights = self.item_popularity_weights[target_item_ids]  # [batch_size]
            
            # Novelty = NDCG × popularity_weight
            novelty_rewards = calculate_novelty_ndcg(
                ranks, 
                item_weights=item_weights,
                k=self.k,
            )
            
            # Novelty annealing 적용 여부에 따라 분기
            if self.novelty_annealing:
                # trainer_state에서 현재 step 정보 가져오기
                trainer_state = kwargs.get("trainer_state", None)
                
                if trainer_state is not None and hasattr(trainer_state, "global_step"):
                    current_step = trainer_state.global_step
                    # Novelty ratio: 0 (초반) → 1 (후반) 선형 증가
                    novelty_ratio = min(1.0, current_step / max(1, self.max_steps))
                else:
                    # trainer_state가 없으면 기본값 0.5 사용 (중간값)
                    novelty_ratio = 0.5
                
                # 선형 보간: (1-ratio)*base + ratio*novelty, novelty_coef 적용
                rewards = (1.0 - novelty_ratio) * base_rewards + novelty_ratio * self.novelty_coef * novelty_rewards
            else:
                # Annealing 없이 novelty reward만 사용, novelty_coef 적용
                rewards = self.novelty_coef * novelty_rewards
        else:
            # 기본 리워드 사용
            rewards = base_rewards
        
        # Target embedding 유사도 리워드 추가
        if self.target_emb_reward and self.target_embeddings is not None:
            target_emb_rewards = self._compute_target_embedding_reward(query_embeddings, user_ids)
            rewards = rewards + self.target_emb_coef * target_emb_rewards
        
        # InfoNCE 리워드 추가
        if self.infonce_reward and self.infonce_item_embeddings is not None:
            infonce_rewards = self._compute_infonce_reward(query_embeddings, user_ids)
            rewards = rewards + self.infonce_coef * infonce_rewards
        
        # 정규화 (optional)
        if self.normalize and rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return rewards
