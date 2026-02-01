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
import nltk
from nltk.corpus import stopwords
import re


def extract_query_from_tags(text: str, tag: str = "query") -> str:
    """
    텍스트에서 특정 XML-like 태그 내부의 내용을 추출
    태그가 없으면 원본 텍스트 반환
    
    Args:
        text: 입력 텍스트
        tag: 추출할 태그 이름 (default: "query")
        
    Returns:
        태그 내부의 텍스트, 태그가 없으면 원본 텍스트
        
    Example:
        >>> text = "<thinking>...</thinking><query>camping gear</query>"
        >>> extract_query_from_tags(text)
        "camping gear"
    """
    # 정규식으로 태그 내용 추출 (대소문자 무시, 줄바꿈 포함)
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        # 태그 내용 추출 및 앞뒤 공백 제거
        return match.group(1).strip()
    else:
        # 태그가 없으면 원본 텍스트 반환
        return text


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
        <query> 태그가 있는 경우 태그 내부의 텍스트만 사용
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            targets: [batch_size] 타겟 아이템 ID
            histories: [batch_size, *] 히스토리 아이템 ID
            neg_items: [batch_size, num_negs] Negative 아이템 ID (optional)
        
        Returns:
            rewards: [batch_size] 리워드 값
        """
        # <query> 태그가 있으면 추출, 없으면 원본 사용
        processed_texts = [extract_query_from_tags(text, tag="query") for text in generated_texts]
        
        # add target text to generated_texts
        if self.test_target:
            processed_texts = [self.item_metadata[target] + "\n" + processed_text for processed_text, target in zip(processed_texts, targets)]  

        # 1. RetrievalService를 통해 유사도 점수 계산
        use_negatives_only = neg_items is not None
        
        scores_ref = self.retrieval_service.calculate_reward.remote(
            processed_texts,
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
        <query> 태그가 있는 경우 태그 내부의 텍스트만 검사
        
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
            
            # <query> 태그가 있으면 추출, 없으면 원본 사용
            processed_text = extract_query_from_tags(gen_text, tag="query")
            
            # 가장 유사한 히스토리 아이템 찾기
            most_similar_item_id = self._get_most_similar_history_item(user_id)
            
            # 해당 아이템의 title 가져오기
            if str(most_similar_item_id) in self.item_metadata:
                item_title = self.item_metadata[str(most_similar_item_id)]["title"]
                first_three_words = self._get_first_three_words(item_title)
                
                # 처리된 텍스트에 첫 3단어가 포함되어 있는지 확인 (대소문자 무시)
                processed_text_lower = processed_text.lower()
                if first_three_words in processed_text_lower:
                    if self.use_position_weight:
                        # 위치 기반 가중치 적용
                        position = processed_text_lower.find(first_three_words)
                        text_length = len(processed_text_lower)
                        weight = self._calculate_position_weight(position, text_length)
                        reward = 1.0 * weight
                    else:
                        # 위치 무관하게 1.0점
                        reward = 1.0
            
            rewards.append(reward)
        
        return rewards


class MetadataMentionReward:
    """
    타겟 아이템의 메타데이터(브랜드, 카테고리 등)를 언급할수록 보상을 제공하는 리워드 함수.
    
    특징:
    1. 메타데이터의 단어들을 많이 언급할수록 리워드 증가
    2. 히스토리 아이템의 메타데이터 중 타겟에 없는 단어를 언급하면 패널티 적용
    3. 생성된 텍스트의 길이에 반비례하도록 리워드 정규화
    4. 불용어(none, a, the 등)는 리워드 계산에서 제외
    """
    
    def __init__(
        self,
        data_name: str,
        device: str = "cuda",
        data_dir: str = "data",
        base_reward: float = 0.1,
        length_penalty_alpha: float = 0.5,
        min_length: int = 10,
        history_penalty_weight: float = 0.01,
    ):
        """
        Args:
            data_name: 데이터셋 이름
            device: 디바이스
            data_dir: 데이터 디렉토리
            base_reward: 메타데이터 단어당 기본 보상 점수
            length_penalty_alpha: 길이 패널티 강도 (0~1, 높을수록 긴 텍스트에 불리)
            min_length: 최소 텍스트 길이 (이보다 짧으면 패널티 없음)
            history_penalty_weight: 히스토리 메타데이터 잘못 언급시 패널티 가중치
        """
        self.__name__ = "MetadataMentionReward"
        self.data_name = data_name
        self.device = device
        self.base_reward = base_reward
        self.length_penalty_alpha = length_penalty_alpha
        self.min_length = min_length
        self.history_penalty_weight = history_penalty_weight
        # NLTK stopwords 다운로드 및 로드 (한번만 실행)
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))
        
        # 추가 불용어 (도메인 특화)
        additional_stopwords = {'none', 'null', 'n/a', 'na'}
        self.stopwords.update(additional_stopwords)
        
        # 아이템 메타데이터 로드
        with open(f"{data_dir}/{data_name}/meta_text_fix.json", "r") as f:
            item_metadata = json.load(f)
            item_metadata = {int(k): v for k, v in item_metadata.items()}
        
        # 각 아이템의 메타데이터 단어 집합을 미리 추출
        self.item_metadata_words = {}
        for item_id, meta in item_metadata.items():
            words = set()
            if "title" in meta and meta["title"]:
                title_words = self._extract_words(str(meta["title"]))
                words.update(title_words)
            
            # 브랜드 추출
            if "brand" in meta and meta["brand"]:
                brand_words = self._extract_words(str(meta["brand"]))
                words.update(brand_words)
            
            # 카테고리 추출
            # if "category" in meta and meta["category"]:
            #     category_words = self._extract_words(str(meta["category"]))
            #     words.update(category_words)
            
            self.item_metadata_words[item_id] = words
        
        print(f"✓ MetadataMentionReward initialized")
        print(f"  - Loaded metadata for {len(self.item_metadata_words)} items")
        print(f"  - Base reward: {self.base_reward}")
        print(f"  - Length penalty alpha: {self.length_penalty_alpha}")
        print(f"  - History penalty weight: {self.history_penalty_weight}")
        print(f"  - Stopwords excluded: {len(self.stopwords)} (NLTK English + custom)")
    
    def _extract_words(self, text: str) -> set:
        """
        텍스트에서 단어를 추출하고 불용어를 제거
        
        Args:
            text: 입력 텍스트
            
        Returns:
            불용어가 제거된 단어 집합 (소문자)
        """
        # 알파벳과 숫자만 남기고 공백으로 구분
        import re
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # 불용어 제거 및 길이가 1인 단어 제거
        filtered_words = {w for w in words if w not in self.stopwords and len(w) > 1}
        
        return filtered_words
    
    def __call__(
        self,
        generated_texts: List[str],
        targets: List[int],
        histories: List[List[int]],
        **kwargs
    ) -> List[float]:
        """
        생성된 텍스트에서 타겟 아이템의 메타데이터 언급도를 평가하여 보상
        히스토리 아이템의 메타데이터 중 타겟에 없는 단어 언급시 패널티 적용
        <query> 태그가 있는 경우 태그 내부의 텍스트만 검사
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            targets: [batch_size] 타겟 아이템 ID
            **kwargs: histories (List[List[int]]): [batch_size, seq_len] 히스토리 아이템 ID
            
        Returns:
            rewards: [batch_size] 보상 값
        """
        rewards = []
        
        for idx, (gen_text, target_id) in enumerate(zip(generated_texts, targets)):
            # <query> 태그가 있으면 추출, 없으면 원본 사용
            processed_text = extract_query_from_tags(gen_text, tag="query")
            
            # 타겟 아이템의 메타데이터 단어
            target_words = self.item_metadata_words.get(target_id, set())
            
            if not target_words:
                rewards.append(0.0)
                continue
            
            # 처리된 텍스트에서 단어 추출
            gen_words = self._extract_words(processed_text)
            
            # 메타데이터 단어가 생성된 텍스트에 몇 개나 언급되었는지 카운트
            matched_words = target_words.intersection(gen_words)
            match_count = len(matched_words)
            
            # 기본 리워드 계산 (언급된 메타데이터 단어 수에 비례)
            reward = match_count * self.base_reward
            
            # 히스토리 아이템 메타데이터 패널티 계산
            if histories is not None and idx < len(histories):
                history_items = histories[idx]
                
                # 히스토리 아이템들의 메타데이터 단어 수집
                history_words = set()
                for hist_id in history_items:
                    hist_words = self.item_metadata_words.get(hist_id, set())
                    history_words.update(hist_words)
                
                # 히스토리에만 있고 타겟에는 없는 단어 (잘못 언급하면 안되는 단어)
                wrong_words = history_words - target_words
                
                # 생성된 텍스트에서 잘못된 단어가 언급된 개수
                wrong_mention_count = len(wrong_words.intersection(gen_words))
                
                # 패널티 적용
                penalty = wrong_mention_count * self.history_penalty_weight
                reward = reward - penalty
                reward = max(reward, 0.0)
            
            # 길이 패널티 적용: 처리된 텍스트 길이 기준
            text_length = len(processed_text.split())
            if text_length > self.min_length:
                # length_factor: 텍스트가 길수록 작아짐 (0~1)
                length_factor = 1.0 / (1.0 + self.length_penalty_alpha * (text_length - self.min_length) / self.min_length)
                reward = reward * length_factor
            
            rewards.append(reward)
        
        return rewards


class ItemPreferenceMentionReward:
    """
    타겟 아이템의 메타데이터와 아이템 간 선호도 정보를 결합하여 보상을 제공하는 리워드 함수
    
    특징:
    1. data_processed/{data_name}_gemma-3-1b-it_item_item_preference.json에서 선호도 정보 로드
    2. 기존 메타데이터(브랜드, 카테고리 등)와 선호도 텍스트를 결합하여 단어 집합 생성
    3. 쿼리에 포함된 공통 단어가 많을수록 더 큰 보상 제공
    4. 불용어(stopwords) 자동 제거
    """
    
    def __init__(
        self,
        data_name: str,
        device: str = "cuda",
        data_dir: str = "data",
        data_processed_dir: str = "data_processed",
        base_reward: float = 0.1,
        length_penalty_alpha: float = 0.5,
        min_length: int = 10,
    ):
        """
        Args:
            data_name: 데이터셋 이름
            device: 디바이스
            data_dir: 메타데이터 디렉토리
            data_processed_dir: 아이템 선호도 데이터 디렉토리
            base_reward: 단어당 기본 보상 점수
            length_penalty_alpha: 길이 패널티 강도 (0~1, 높을수록 긴 텍스트에 불리)
            min_length: 최소 텍스트 길이 (이보다 짧으면 패널티 없음)
        """
        self.__name__ = "ItemPreferenceMentionReward"
        self.data_name = data_name
        self.device = device
        self.base_reward = base_reward
        self.length_penalty_alpha = length_penalty_alpha
        self.min_length = min_length
        
        # NLTK stopwords 다운로드 및 로드
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))
        
        # 추가 불용어 (도메인 특화)
        additional_stopwords = {'none', 'null', 'n/a', 'na'}
        self.stopwords.update(additional_stopwords)
        
        # 아이템 메타데이터 로드
        print(f"📦 Loading item metadata from {data_dir}/{data_name}/meta_text_fix.json")
        with open(f"{data_dir}/{data_name}/meta_text_fix.json", "r") as f:
            item_metadata = json.load(f)
            item_metadata = {int(k): v for k, v in item_metadata.items()}
        
        # 아이템 선호도 정보 로드
        preference_file = f"{data_processed_dir}/{data_name}_gemma-3-1b-it_item_item_preference.json"
        print(f"📦 Loading item preference from {preference_file}")
        with open(preference_file, "r") as f:
            item_preference = json.load(f)
            item_preference = {int(k): v for k, v in item_preference.items()}
        
        # 각 아이템의 메타데이터 + 선호도 단어 집합을 미리 추출
        self.item_combined_words = {}
        for item_id, meta in item_metadata.items():
            words = set()
            
            # 메타데이터에서 단어 추출
            if "title" in meta and meta["title"]:
                title_words = self._extract_words(str(meta["title"]))
                words.update(title_words)
            
            if "brand" in meta and meta["brand"]:
                brand_words = self._extract_words(str(meta["brand"]))
                words.update(brand_words)
            
            # 선호도 정보에서 단어 추출
            if item_id in item_preference:
                preference_words = self._extract_words(item_preference[item_id])
                words.update(preference_words)
            
            self.item_combined_words[item_id] = words
        
        print(f"✓ ItemPreferenceMentionReward initialized")
        print(f"  - Loaded metadata for {len(item_metadata)} items")
        print(f"  - Loaded preference for {len(item_preference)} items")
        print(f"  - Combined word sets created for {len(self.item_combined_words)} items")
        print(f"  - Base reward: {self.base_reward}")
        print(f"  - Length penalty alpha: {self.length_penalty_alpha}")
        print(f"  - Stopwords excluded: {len(self.stopwords)} (NLTK English + custom)")
        
        # 통계 출력
        if len(self.item_combined_words) > 0:
            word_counts = [len(words) for words in self.item_combined_words.values()]
            avg_words = sum(word_counts) / len(word_counts)
            max_words = max(word_counts)
            min_words = min(word_counts)
            print(f"  - Word set statistics: Min={min_words}, Max={max_words}, Avg={avg_words:.1f}")
    
    def _extract_words(self, text: str) -> set:
        """
        텍스트에서 단어를 추출하고 불용어를 제거
        
        Args:
            text: 입력 텍스트
            
        Returns:
            불용어가 제거된 단어 집합 (소문자)
        """
        # 알파벳과 숫자만 남기고 공백으로 구분
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # 불용어 제거 및 길이가 1인 단어 제거
        filtered_words = {w for w in words if w not in self.stopwords and len(w) > 1}
        
        return filtered_words
    
    def __call__(
        self,
        generated_texts: List[str],
        targets: List[int],
        **kwargs
    ) -> List[float]:
        """
        생성된 텍스트에서 타겟 아이템의 메타데이터+선호도 단어 언급도를 평가하여 보상
        <query> 태그가 있는 경우 태그 내부의 텍스트만 검사
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            targets: [batch_size] 타겟 아이템 ID
            
        Returns:
            rewards: [batch_size] 보상 값
        """
        rewards = []
        
        for gen_text, target_id in zip(generated_texts, targets):
            # 타겟 아이템의 메타데이터+선호도 단어
            target_words = self.item_combined_words.get(target_id, set())
            
            if not target_words:
                rewards.append(0.0)
                continue
            
            # 처리된 텍스트에서 단어 추출
            gen_words = self._extract_words(gen_text)
            
            # 공통 단어가 몇 개나 언급되었는지 카운트
            matched_words = target_words.intersection(gen_words)
            match_count = len(matched_words)
            
            # 기본 리워드 계산 (언급된 단어 수에 비례)
            reward = match_count * self.base_reward
            
            # 길이 패널티 적용: 생성된 텍스트 길이 기준
            text_length = len(gen_text.split())
            if text_length > self.min_length:
                # length_factor: 텍스트가 길수록 작아짐 (0~1)
                length_factor = 1.0 / (1.0 + self.length_penalty_alpha * (text_length - self.min_length) / self.min_length)
                reward = reward * length_factor
            
            rewards.append(reward)
        
        return rewards


class FormatComplianceReward:
    """
    생성된 텍스트가 특정 XML-like 포맷을 준수하는지 확인하는 리워드 함수
    
    요구 포맷:
    <thinking>...</thinking>
    <window>...</window>
    <items>...</items>
    <query>...</query>
    
    각 태그가 존재하고 올바르게 열리고 닫히면 부분 점수 부여
    """
    
    def __init__(
        self,
        required_tags: List[str] = None,
        reward_per_tag: float = 0.25,
        strict_order: bool = False,
        case_sensitive: bool = False,
    ):
        """
        Args:
            required_tags: 필수 태그 리스트 (default: ["thinking", "window", "items", "query"])
            reward_per_tag: 각 태그당 보상 점수 (default: 0.25, 4개 태그 * 0.25 = 1.0)
            strict_order: 태그 순서를 엄격하게 체크할지 여부 (default: False)
            case_sensitive: 대소문자 구분 여부 (default: False)
        """
        self.__name__ = "FormatComplianceReward"
        
        if required_tags is None:
            self.required_tags = ["thinking", "window", "items", "query"]
        else:
            self.required_tags = required_tags
        
        self.reward_per_tag = reward_per_tag
        self.strict_order = strict_order
        self.case_sensitive = case_sensitive
        
        print(f"✓ FormatComplianceReward initialized")
        print(f"  - Required tags: {self.required_tags}")
        print(f"  - Reward per tag: {self.reward_per_tag}")
        print(f"  - Strict order: {self.strict_order}")
        print(f"  - Case sensitive: {self.case_sensitive}")
        print(f"  - Max reward: {len(self.required_tags) * self.reward_per_tag}")
    
    def _check_tag_exists(self, text: str, tag: str) -> bool:
        """
        태그가 올바르게 열리고 닫히는지 확인
        
        Args:
            text: 검사할 텍스트
            tag: 태그 이름 (예: "thinking")
            
        Returns:
            True if both opening and closing tags exist, False otherwise
        """
        if not self.case_sensitive:
            text = text.lower()
            tag = tag.lower()
        
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        
        return open_tag in text and close_tag in text
    
    def _check_tag_order(self, text: str) -> bool:
        """
        태그가 올바른 순서로 나타나는지 확인
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            True if tags appear in correct order, False otherwise
        """
        if not self.case_sensitive:
            text = text.lower()
        
        last_position = -1
        
        for tag in self.required_tags:
            open_tag = f"<{tag}>"
            if not self.case_sensitive:
                tag = tag.lower()
            
            position = text.find(open_tag)
            
            if position == -1:
                return False
            
            if position < last_position:
                return False
            
            last_position = position
        
        return True
    
    def __call__(
        self,
        generated_texts: List[str],
        **kwargs
    ) -> List[float]:
        """
        생성된 텍스트의 포맷 준수도를 평가하여 보상
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            
        Returns:
            rewards: [batch_size] 보상 값 (0.0 ~ max_reward)
        """
        rewards = []
        
        for gen_text in generated_texts:
            reward = 0.0
            
            # 1. 각 태그가 존재하는지 확인
            valid_tags = 0
            for tag in self.required_tags:
                if self._check_tag_exists(gen_text, tag):
                    valid_tags += 1
                    reward += self.reward_per_tag
            
            # 2. 엄격한 순서 체크 (옵션)
            if self.strict_order and valid_tags > 0:
                if not self._check_tag_order(gen_text):
                    # 순서가 틀리면 보상을 절반으로 감소
                    reward = reward * 0.5
            
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
            proxy_label_cutoff: Proxy label 유사도 역치 (default: 0.0)
                               이 값 미만의 유사도를 가진 아이템은 proxy label에서 제외
                               예: 0.5로 설정하면 유사도 0.5 미만 아이템은 필터링
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
            self.proxy_label_cutoff = args.proxy_label_cutoff if hasattr(args, "proxy_label_cutoff") else 0.98
        else:
            self.proxy_label_reward = False
            self.proxy_k = 0
            self.proxy_label_coef = 0
            self.proxy_label_cutoff = 0.0
        
        # Anchor-Guided GRPO 파라미터
        if hasattr(args, "anchor_reward"):
            self.anchor_reward = args.anchor_reward
            self.anchor_coef = args.anchor_coef
            self.anchor_radius_start = args.anchor_radius_start
            self.anchor_radius_end = args.anchor_radius_end
            self.anchor_penalty_mode = args.anchor_penalty_mode
            self.anchor_penalty_value = args.anchor_penalty_value
        else:
            self.anchor_reward = False
            self.anchor_coef = 1.0
            self.anchor_radius_start = 0.5
            self.anchor_radius_end = 1.0
            self.anchor_penalty_mode = "soft"
            self.anchor_penalty_value = -1.0
        
        # Adaptive Threshold Reward 파라미터
        if hasattr(args, "adaptive_threshold_reward"):
            self.adaptive_threshold_reward = args.adaptive_threshold_reward
            self.adaptive_threshold_coef = args.adaptive_threshold_coef
            self.adaptive_tau_min = args.adaptive_tau_min
        else:
            self.adaptive_threshold_reward = False
            self.adaptive_threshold_coef = 1.0
            self.adaptive_tau_min = 0.0
        
        # History Proxy Threshold Reward 파라미터
        if hasattr(args, "history_proxy_threshold_reward"):
            self.history_proxy_threshold_reward = args.history_proxy_threshold_reward
            self.history_proxy_threshold_coef = args.history_proxy_threshold_coef
        else:
            self.history_proxy_threshold_reward = False
            self.history_proxy_threshold_coef = 1.0
        
        # Training 관련 파라미터
        self.max_steps = args.max_steps
        
        # Reward 분해 추적을 위한 변수 (wandb 로깅용)
        self.last_base_rewards = None
        self.last_proxy_label_rewards = None
        self.last_target_emb_rewards = None
        self.last_infonce_rewards = None
        self.last_anchor_rewards = None
        self.last_adaptive_threshold_rewards = None
        self.last_history_proxy_threshold_rewards = None
        
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
            print(f"  - Proxy label cutoff: {self.proxy_label_cutoff}")
            print(f"    → Items with similarity < {self.proxy_label_cutoff} will be excluded from proxy labels")
            print(f"  - Use top-{self.proxy_k} similar items as soft labels with similarity-weighted NDCG")
            print(f"  - Final reward = base_reward + proxy_label_coef * proxy_label_ndcg")
        if self.anchor_reward:
            print(f"  - Anchor-Guided GRPO (AG-GRPO): ENABLED")
            print(f"  - Anchor coefficient: {self.anchor_coef}")
            print(f"  - Anchor radius: {self.anchor_radius_start} → {self.anchor_radius_end} (curriculum learning)")
            print(f"  - Penalty mode: {self.anchor_penalty_mode}")
            if self.anchor_penalty_mode == "hard":
                print(f"  - Hard penalty value: {self.anchor_penalty_value}")
            print(f"  - Reward based on similarity with last item (anchor) embedding")
            print(f"  - Gradually expands exploration radius as training progresses")
        if self.adaptive_threshold_reward:
            print(f"  - Adaptive Threshold Reward: ENABLED")
            print(f"  - Adaptive threshold coefficient: {self.adaptive_threshold_coef}")
            print(f"  - Minimum threshold (tau_min): {self.adaptive_tau_min}")
            print(f"  - Uses dynamic threshold based on historical item similarity (S_base)")
            print(f"  - Reward = 1 if CosSim(query, target) > max(tau_min, S_base), else 0")
        if self.history_proxy_threshold_reward:
            print(f"  - History Proxy Threshold Reward: ENABLED")
            print(f"  - History proxy threshold coefficient: {self.history_proxy_threshold_coef}")
            print(f"  - Uses most similar history item to target as proxy")
            print(f"  - Reward = max(0, CosSim(query, proxy) - mean(CosSim(query, other_history)))")
        
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
            # args.proxy_label_file이 지정되어 있으면 그것을 사용, 아니면 자동 생성
            if hasattr(args, 'proxy_label_file') and args.proxy_label_file is not None:
                proxy_labels_file = args.proxy_label_file
                print(f"📦 Using user-specified proxy labels file: {proxy_labels_file}")
            else:
                proxy_labels_file = f"data_emb/{self.data_name}_proxy_labels_k100_{args.emb_type}_{emb_model_name_dir}.json"
                print(f"📦 Using auto-generated proxy labels path: {proxy_labels_file}")
            
            proxy_labels_path = Path(proxy_labels_file)
            
            if proxy_labels_path.exists():
                print(f"✓ Loading pre-computed proxy labels from: {proxy_labels_file}")
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
        
        # Last item (anchor) embeddings 준비 (anchor_reward 사용 시)
        if self.anchor_reward:
            self.last_item_embeddings = self._prepare_last_item_embeddings()
            print(f"✓ Prepared last item (anchor) embeddings for anchor-guided exploration")
        else:
            self.last_item_embeddings = None
        
        # History items 준비 (adaptive_threshold_reward 사용 시)
        if self.adaptive_threshold_reward:
            self.user_history_items = self._prepare_user_history_items()
            print(f"✓ Prepared user history items for adaptive threshold reward")
        else:
            self.user_history_items = None
        
        # History proxy items 준비 (history_proxy_threshold_reward 사용 시)
        if self.history_proxy_threshold_reward:
            self.user_history_items = self._prepare_user_history_items()
            self.user_history_proxy_items = self._prepare_user_history_proxy_items(uid_2_target)
            print(f"✓ Prepared user history proxy items for history proxy threshold reward")
        else:
            self.user_history_proxy_items = None
        
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
        total_filtered = 0  # 필터링된 proxy 개수 통계
        
        for item_id_str, proxy_list in proxy_labels_json.items():
            item_id = int(item_id_str)
            
            # 1. proxy_k 개수만큼 자르기
            proxy_list = proxy_list[:self.proxy_k]
            
            # 2. cutoff 이하의 아이템 필터링
            if self.proxy_label_cutoff > 0:
                filtered_proxy_list = [(pid, sim) for pid, sim in proxy_list if sim >= self.proxy_label_cutoff]
                total_filtered += len(proxy_list) - len(filtered_proxy_list)
                proxy_list = filtered_proxy_list
            
            # 3. 필터링 후 남은 proxy가 있는 경우만 저장
            if len(proxy_list) > 0:
                # List[Tuple[item_id, similarity]]를 두 개의 텐서로 분리
                proxy_ids = torch.tensor([p[0] for p in proxy_list], dtype=torch.long, device=self.device)
                proxy_sims = torch.tensor([p[1] for p in proxy_list], dtype=torch.float32, device=self.device)
                
                item_proxy_labels[item_id] = (proxy_ids, proxy_sims)
        
        # 필터링 통계 출력
        if self.proxy_label_cutoff > 0:
            print(f"  - Filtered {total_filtered} proxy labels below cutoff {self.proxy_label_cutoff}")
            avg_proxies = sum(len(v[0]) for v in item_proxy_labels.values()) / max(1, len(item_proxy_labels))
            print(f"  - Average proxies per item after filtering: {avg_proxies:.2f}")
        
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
    
    def _prepare_last_item_embeddings(self) -> torch.Tensor:
        """
        각 사용자의 마지막 아이템 (앵커) 임베딩을 준비
        sequential_data.txt에서 validation set 기준 마지막 아이템 읽기
        
        Returns:
            last_item_embeddings: [max_uid+1, emb_dim] 각 사용자의 마지막 아이템 임베딩
        """
        print(f"📦 Preparing last item (anchor) embeddings from sequential data...")
        
        sequential_file = f"data/{self.data_name}/sequential_data.txt"
        
        # 사용자별 마지막 아이템 ID 수집
        uid_2_last_item = {}
        max_uid = 0
        
        with open(sequential_file, 'r') as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                last_item_id = parts[-3]  # 마지막 아이템
                uid_2_last_item[user_id] = last_item_id
                max_uid = max(max_uid, user_id)
        
        # 임베딩 텐서 초기화
        emb_dim = self.item_embeddings.shape[1]
        last_item_embeddings = torch.zeros(max_uid + 1, emb_dim, device=self.device)
        
        # 마지막 아이템 임베딩 채우기
        for uid, last_item_id in uid_2_last_item.items():
            last_item_embeddings[uid] = self.item_embeddings[last_item_id]
        
        print(f"  Total users with last item: {len(uid_2_last_item)}")
        print(f"  Max user ID: {max_uid}")
        
        return last_item_embeddings
    
    def _prepare_user_history_items(self) -> Dict[int, torch.Tensor]:
        """
        각 사용자의 과거 구매 아이템 목록을 준비 (adaptive threshold reward용)
        sequential_data.txt에서 train set history 읽기
        
        Returns:
            user_history_items: Dict[user_id, history_item_ids_tensor]
        """
        print(f"📦 Preparing user history items from sequential data...")
        
        sequential_file = f"data/{self.data_name}/sequential_data.txt"
        
        # 사용자별 히스토리 아이템 수집
        user_history_items = {}
        
        with open(sequential_file, 'r') as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                history = parts[1:-3]  # Train set의 history
                
                # 히스토리가 비어있으면 스킵
                if len(history) == 0:
                    continue
                
                # 텐서로 변환하여 저장
                user_history_items[user_id] = torch.tensor(history, dtype=torch.long, device=self.device)
        
        print(f"  Total users with history: {len(user_history_items)}")
        
        # 통계 출력
        if len(user_history_items) > 0:
            history_lengths = [len(h) for h in user_history_items.values()]
            avg_length = sum(history_lengths) / len(history_lengths)
            min_length = min(history_lengths)
            max_length = max(history_lengths)
            print(f"  History length - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.2f}")
        
        return user_history_items
    
    def _prepare_user_history_proxy_items(self, uid_2_target: Dict[int, int]) -> Dict[int, int]:
        """
        각 사용자에 대해 타겟 아이템과 가장 유사한 히스토리 아이템을 사전에 계산
        (history proxy threshold reward용)
        
        전략:
        - 타겟 아이템과 각 히스토리 아이템의 코사인 유사도를 계산
        - 가장 유사도가 높은 히스토리 아이템을 proxy로 저장
        
        Args:
            uid_2_target: 사용자 ID to 타겟 아이템 ID 매핑
        
        Returns:
            user_history_proxy_items: Dict[user_id, proxy_item_id]
        """
        print(f"📦 Pre-computing most similar history items to target for each user...")
        
        user_history_proxy_items = {}
        
        # 아이템 임베딩 정규화 (코사인 유사도 계산을 위해)
        item_embeddings_norm = torch.nn.functional.normalize(self.item_embeddings, p=2, dim=1)
        
        users_with_proxy = 0
        users_without_history = 0
        
        for uid, target_id in uid_2_target.items():
            # 히스토리가 없는 경우 스킵
            if uid not in self.user_history_items:
                users_without_history += 1
                continue
            
            history_item_ids = self.user_history_items[uid]  # [history_len]
            
            # 타겟 임베딩
            target_emb = item_embeddings_norm[target_id].unsqueeze(0)  # [1, emb_dim]
            
            # 히스토리 임베딩
            history_embs = item_embeddings_norm[history_item_ids]  # [history_len, emb_dim]
            
            # 타겟과 히스토리 아이템들의 유사도 계산
            similarities = torch.mm(target_emb, history_embs.T).squeeze(0)  # [history_len]
            
            # 가장 유사도가 높은 히스토리 아이템 선택
            max_sim_idx = similarities.argmax().item()
            proxy_item_id = history_item_ids[max_sim_idx].item()
            max_similarity = similarities[max_sim_idx].item()
            
            user_history_proxy_items[uid] = proxy_item_id
            users_with_proxy += 1
            
            # 디버깅: 처음 5명의 사용자 정보 출력
            if users_with_proxy <= 5:
                print(f"  User {uid}: Target={target_id}, Proxy={proxy_item_id}, "
                      f"Similarity={max_similarity:.4f}, History len={len(history_item_ids)}")
        
        print(f"  Total users with proxy: {users_with_proxy}")
        print(f"  Users without history: {users_without_history}")
        
        return user_history_proxy_items
    
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
        <query> 태그가 있는 경우 태그 내부의 텍스트만 사용
        
        Args:
            generated_texts: [batch_size] 생성된 텍스트
            
        Returns:
            embeddings: [batch_size, emb_dim] 임베딩
        """
        # <query> 태그가 있으면 추출, 없으면 원본 사용
        processed_texts = [extract_query_from_tags(text, tag="query") for text in generated_texts]
        
        embeddings = self.emb_model.encode(
            processed_texts,
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
    
    def _compute_anchor_reward(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
        current_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Anchor-Guided GRPO 리워드 계산
        마지막 아이템 임베딩과의 유사도를 기반으로 리워드 계산
        동적 반경(radius) 제어: 학습 초기에는 좁은 반경, 후기에는 넓은 반경
        
        Args:
            query_embeddings: [batch_size, emb_dim] 쿼리 임베딩
            user_ids: [batch_size] 사용자 ID
            current_step: 현재 학습 step (None이면 중간값 사용)
            
        Returns:
            rewards: [batch_size] 앵커 리워드
                    soft mode: similarity (반경 내외 모두 유사도 리워드)
                    hard mode: similarity if in radius, else penalty_value
        """
        # L2 정규화 (코사인 유사도를 위해)
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        
        # 마지막 아이템 임베딩 가져오기
        last_item_embs = self.last_item_embeddings[user_ids]  # [batch_size, emb_dim]
        last_item_embs = torch.nn.functional.normalize(last_item_embs, p=2, dim=1)
        
        # 코사인 유사도 계산 (similarity ∈ [-1, 1])
        similarities = (query_embeddings * last_item_embs).sum(dim=1)  # [batch_size]
        
        # # 동적 반경 계산 (curriculum learning)
        # if current_step is not None:
        #     # 학습 진행도에 따라 반경 선형 증가
        #     progress = min(1.0, current_step / max(1, self.max_steps))
        #     current_radius = self.anchor_radius_start + progress * (self.anchor_radius_end - self.anchor_radius_start)
        # else:
        #     # Step 정보가 없으면 중간값 사용
        current_radius = (self.anchor_radius_start + self.anchor_radius_end) / 2.0
        
        if self.anchor_penalty_mode == "soft":
            # Soft mode: 유사도를 그대로 리워드로 사용
            # 반경 내외 구분 없이, 유사도가 높을수록 높은 리워드
            rewards = similarities
        elif self.anchor_penalty_mode == "hard":
            # Hard mode: 반경 내에 있으면 유사도 리워드, 벗어나면 페널티
            # current_radius를 threshold로 사용
            in_radius = similarities >= current_radius  # [batch_size] boolean
            rewards = torch.where(in_radius, similarities, torch.tensor(self.anchor_penalty_value, device=self.device))
        else:
            raise ValueError(f"Unknown anchor_penalty_mode: {self.anchor_penalty_mode}")
        
        return rewards
    
    def _compute_adaptive_threshold_reward(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        적응형 임계값 보상 (Adaptive Threshold Reward) 계산
        
        전략: 과거 구매 아이템들과의 평균 유사도(S_base)를 동적 임계값으로 사용
        
        수식:
            S_base = mean(CosSim(query, history_items))
            threshold = max(tau_min, S_base)
            R = 1 if CosSim(query, target) > threshold else 0
        
        의미: "적어도 과거에 샀던 물건들보다는 정답에 더 비슷해야 정답으로 인정해주겠다"
        
        Args:
            query_embeddings: [batch_size, emb_dim] 쿼리 임베딩
            user_ids: [batch_size] 사용자 ID
            
        Returns:
            rewards: [batch_size] 적응형 임계값 리워드 (0 또는 1)
        """
        batch_size = len(user_ids)
        rewards = torch.zeros(batch_size, device=self.device)
        
        # L2 정규화 (코사인 유사도를 위해)
        query_embeddings_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        
        # 타겟 아이템 ID 가져오기
        if self.use_full_item_pool:
            target_item_ids = torch.tensor(
                [self.uid_2_target[uid] for uid in user_ids],
                device=self.device
            )  # [batch_size]
        else:
            batch_candidate_tensor = self.candidate_tensor[user_ids]  # [batch_size, k]
            target_item_ids = batch_candidate_tensor[:, 0]  # [batch_size]
        
        # 타겟 아이템 임베딩 가져오기
        target_item_embs = self.item_embeddings[target_item_ids]  # [batch_size, emb_dim]
        target_item_embs_norm = torch.nn.functional.normalize(target_item_embs, p=2, dim=1)
        
        # 쿼리와 타겟 아이템의 유사도 계산
        query_target_similarity = (query_embeddings_norm * target_item_embs_norm).sum(dim=1)  # [batch_size]
        
        for i, uid in enumerate(user_ids):
            uid_item = uid.item() if isinstance(uid, torch.Tensor) else uid
            
            # 1. 과거 구매 아이템들의 임베딩 가져오기
            if uid_item not in self.user_history_items:
                # 히스토리가 없으면 tau_min을 임계값으로 사용
                threshold = self.adaptive_tau_min
            else:
                history_item_ids = self.user_history_items[uid_item]  # [history_len]
                
                # 2. 히스토리 아이템 임베딩 가져오기
                history_item_embs = self.item_embeddings[history_item_ids]  # [history_len, emb_dim]
                history_item_embs_norm = torch.nn.functional.normalize(history_item_embs, p=2, dim=1)
                
                # 3. 쿼리와 히스토리 아이템들의 유사도 계산 후 평균 구하기 (S_base)
                query_history_similarities = torch.mm(
                    query_embeddings_norm[i].unsqueeze(0),  # [1, emb_dim]
                    history_item_embs_norm.T  # [emb_dim, history_len]
                ).squeeze(0)  # [history_len]
                
                s_base = query_history_similarities.mean().item()
                
                # 4. 동적 임계값 = max(tau_min, S_base)
                threshold = max(self.adaptive_tau_min, s_base)
            
            # 5. 쿼리와 정답 아이템의 유사도가 임계값보다 큰 만큼 리워드
            rewards[i] = max(0, query_target_similarity[i].item() - threshold)
        
        return rewards
    
    def _compute_history_proxy_threshold_reward(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        히스토리 Proxy 임계값 보상 (History Proxy Threshold Reward) 계산
        
        전략: 타겟과 가장 유사한 히스토리 아이템(proxy)을 사용하여
              나머지 히스토리 아이템들과의 평균 유사도를 임계값으로 설정
        
        수식:
            proxy = argmax_i CosSim(target, history_i)
            other_history = history - {proxy}
            S_threshold = mean(CosSim(query, other_history))
            R = max(0, CosSim(query, proxy) - S_threshold)
        
        의미: "쿼리가 과거 아이템들 평균보다 타겟과 비슷한 아이템(proxy)에 더 가까울수록 높은 보상"
        
        Args:
            query_embeddings: [batch_size, emb_dim] 쿼리 임베딩
            user_ids: [batch_size] 사용자 ID
            
        Returns:
            rewards: [batch_size] 히스토리 Proxy 임계값 리워드
        """
        batch_size = len(user_ids)
        rewards = torch.zeros(batch_size, device=self.device)
        
        # L2 정규화 (코사인 유사도를 위해)
        query_embeddings_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        
        for i, uid in enumerate(user_ids):
            uid_item = uid.item() if isinstance(uid, torch.Tensor) else uid
            
            # 1. Proxy 아이템 ID 가져오기 (사전에 계산됨)
            if uid_item not in self.user_history_proxy_items:
                # Proxy가 없으면 리워드 0
                continue
            
            proxy_item_id = self.user_history_proxy_items[uid_item]
            history_item_ids = self.user_history_items[uid_item]  # [history_len]
            
            # 2. Proxy 아이템 임베딩 가져오기
            proxy_item_emb = self.item_embeddings[proxy_item_id]  # [emb_dim]
            proxy_item_emb_norm = torch.nn.functional.normalize(proxy_item_emb.unsqueeze(0), p=2, dim=1)
            
            # 3. 쿼리와 Proxy 아이템의 유사도 계산
            query_proxy_similarity = (query_embeddings_norm[i] * proxy_item_emb_norm.squeeze(0)).sum().item()
            
            # 4. 나머지 히스토리 아이템들과의 평균 유사도 계산 (임계값)
            # Proxy를 제외한 히스토리 아이템들
            other_history_mask = history_item_ids != proxy_item_id
            other_history_item_ids = history_item_ids[other_history_mask]
            
            if len(other_history_item_ids) > 0:
                # 나머지 히스토리 아이템 임베딩
                other_history_embs = self.item_embeddings[other_history_item_ids]  # [other_len, emb_dim]
                other_history_embs_norm = torch.nn.functional.normalize(other_history_embs, p=2, dim=1)
                
                # 쿼리와 나머지 히스토리 아이템들의 유사도 평균
                query_other_similarities = torch.mm(
                    query_embeddings_norm[i].unsqueeze(0),  # [1, emb_dim]
                    other_history_embs_norm.T  # [emb_dim, other_len]
                ).squeeze(0)  # [other_len]
                
                s_threshold = query_other_similarities.mean().item()
            else:
                # 히스토리가 proxy 하나뿐인 경우, 임계값 0
                s_threshold = 0.0
            
            # 5. 쿼리-proxy 유사도가 임계값을 넘는 만큼 리워드
            rewards[i] = max(0, query_proxy_similarity - s_threshold)
        
        return rewards
    
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

        # Base rewards를 wandb 로깅을 위해 저장
        self.last_base_rewards = base_rewards.detach().cpu()

        # Proxy label reward 사용 여부에 따라 분기
        if self.proxy_label_reward:
            # Proxy label 리워드 사용 시: 기존 base_reward + proxy_label_reward
            # 예측 점수도 함께 계산 필요
            _, predicted_scores = self._compute_similarity_scores(query_embeddings, user_ids, return_scores=True)
            # 2. Proxy label NDCG 계산
            proxy_label_rewards = self._compute_proxy_label_ndcg(query_embeddings, user_ids, predicted_scores)
            
            # Wandb 로깅을 위해 저장
            self.last_proxy_label_rewards = proxy_label_rewards.detach().cpu()
            
            # 3. 두 리워드를 합산
            base_rewards = base_rewards + self.proxy_label_coef * proxy_label_rewards
        else:
            self.last_proxy_label_rewards = None
        
        
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
            self.last_target_emb_rewards = target_emb_rewards.detach().cpu()
            rewards = rewards + self.target_emb_coef * target_emb_rewards
        else:
            self.last_target_emb_rewards = None
        
        # InfoNCE 리워드 추가
        if self.infonce_reward and self.infonce_item_embeddings is not None:
            infonce_rewards = self._compute_infonce_reward(query_embeddings, user_ids)
            self.last_infonce_rewards = infonce_rewards.detach().cpu()
            rewards = rewards + self.infonce_coef * infonce_rewards
        else:
            self.last_infonce_rewards = None
        
        # Anchor-Guided GRPO 리워드 추가
        if self.anchor_reward and self.last_item_embeddings is not None:
            # trainer_state에서 현재 step 정보 가져오기
            trainer_state = kwargs.get("trainer_state", None)
            current_step = None
            if trainer_state is not None and hasattr(trainer_state, "global_step"):
                current_step = trainer_state.global_step
            
            anchor_rewards = self._compute_anchor_reward(query_embeddings, user_ids, current_step)
            self.last_anchor_rewards = anchor_rewards.detach().cpu()
            rewards = rewards + self.anchor_coef * anchor_rewards
        else:
            self.last_anchor_rewards = None
        
        # Adaptive Threshold 리워드 추가
        if self.adaptive_threshold_reward and self.user_history_items is not None:
            adaptive_threshold_rewards = self._compute_adaptive_threshold_reward(query_embeddings, user_ids)
            self.last_adaptive_threshold_rewards = adaptive_threshold_rewards.detach().cpu()
            rewards = rewards + self.adaptive_threshold_coef * adaptive_threshold_rewards
        else:
            self.last_adaptive_threshold_rewards = None
        
        # History Proxy Threshold 리워드 추가
        if self.history_proxy_threshold_reward and self.user_history_proxy_items is not None:
            history_proxy_threshold_rewards = self._compute_history_proxy_threshold_reward(query_embeddings, user_ids)
            self.last_history_proxy_threshold_rewards = history_proxy_threshold_rewards.detach().cpu()
            rewards = rewards + self.history_proxy_threshold_coef * history_proxy_threshold_rewards
        else:
            self.last_history_proxy_threshold_rewards = None
        
        # 정규화 (optional)
        if self.normalize and rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return rewards
    
    def get_reward_breakdown(self) -> Dict[str, torch.Tensor]:
        """
        마지막 계산된 리워드의 구성 요소들을 반환
        Wandb 로깅 등에 사용
        
        Returns:
            Dict[str, torch.Tensor]: 리워드 구성 요소들
                - "base_reward": 기본 리워드 (NDCG/Hit/MRR)
                - "proxy_label_reward": Proxy label 리워드 (사용 시)
                - "target_emb_reward": Target embedding 유사도 리워드 (사용 시)
                - "infonce_reward": InfoNCE 리워드 (사용 시)
                - "anchor_reward": Anchor-Guided GRPO 리워드 (사용 시)
                - "adaptive_threshold_reward": Adaptive Threshold 리워드 (사용 시)
                - "history_proxy_threshold_reward": History Proxy Threshold 리워드 (사용 시)
        """
        breakdown = {}
        
        if self.last_base_rewards is not None:
            breakdown["base_reward"] = self.last_base_rewards
        
        if self.last_proxy_label_rewards is not None:
            breakdown["proxy_label_reward"] = self.last_proxy_label_rewards
        
        if self.last_target_emb_rewards is not None:
            breakdown["target_emb_reward"] = self.last_target_emb_rewards
        
        if self.last_infonce_rewards is not None:
            breakdown["infonce_reward"] = self.last_infonce_rewards
        
        if self.last_anchor_rewards is not None:
            breakdown["anchor_reward"] = self.last_anchor_rewards
        
        if self.last_adaptive_threshold_rewards is not None:
            breakdown["adaptive_threshold_reward"] = self.last_adaptive_threshold_rewards
        
        if self.last_history_proxy_threshold_rewards is not None:
            breakdown["history_proxy_threshold_reward"] = self.last_history_proxy_threshold_rewards
        
        return breakdown
