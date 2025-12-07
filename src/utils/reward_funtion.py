"""
Reward Functions for RL4Rec
NDCG 기반 리워드 계산 및 TRL 통합
"""

import json
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import ray


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
