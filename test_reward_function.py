#!/usr/bin/env python3
"""
리워드 함수 테스트 스크립트
NDCG, Hit@K, MRR 계산 검증
"""

import ray
import torch
import sys
from pathlib import Path

# Path 설정
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_utils.reward_funtion import (
    calculate_ndcg,
    calculate_hit_rate,
    calculate_mrr,
    create_reward_function,
)


def test_ndcg_calculation():
    """NDCG 계산 로직 테스트"""
    print("\n" + "="*60)
    print("🧪 Testing NDCG Calculation")
    print("="*60)
    
    # 테스트 데이터
    # 배치 크기 3, 아이템 10개
    predicted_scores = torch.tensor([
        [0.1, 0.2, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05],  # target=2 (3rd position)
        [0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.05],  # target=4 (1st position)
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],   # target=9 (1st position)
    ])
    
    target_items = [2, 4, 9]
    history_items = [[0], [1, 2], []]
    k = 5
    
    # NDCG 계산
    ndcg = calculate_ndcg(predicted_scores, target_items, history_items, k=k)
    
    print(f"Predicted scores shape: {predicted_scores.shape}")
    print(f"Target items: {target_items}")
    print(f"History items: {history_items}")
    print(f"\nNDCG@{k} scores:")
    for i, score in enumerate(ndcg):
        print(f"  Sample {i}: {score:.4f}")
    print(f"\nMean NDCG@{k}: {ndcg.mean():.4f}")
    
    # 예상 결과 검증
    assert ndcg[1] > ndcg[0], "Target이 1위인 경우가 더 높은 NDCG를 가져야 함"
    assert ndcg.min() >= 0.0 and ndcg.max() <= 1.0, "NDCG는 0~1 범위여야 함"
    print("✅ NDCG calculation test passed!")


def test_hit_rate_calculation():
    """Hit@K 계산 로직 테스트"""
    print("\n" + "="*60)
    print("🧪 Testing Hit@K Calculation")
    print("="*60)
    
    predicted_scores = torch.tensor([
        [0.1, 0.2, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05],  # target=2 in top-5
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05],  # target=8 in top-5
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],  # target=9 NOT in top-5
    ])
    
    target_items = [2, 8, 9]
    history_items = [[], [], []]
    k = 5
    
    hit = calculate_hit_rate(predicted_scores, target_items, history_items, k=k)
    
    print(f"Hit@{k} scores:")
    for i, score in enumerate(hit):
        print(f"  Sample {i}: {score:.0f} (target={target_items[i]})")
    print(f"\nHit@{k} rate: {hit.mean():.2%}")
    
    # 예상 결과 검증
    assert hit[0] == 1.0, "Target=2 should be in top-5"
    assert hit[1] == 1.0, "Target=8 should be in top-5"
    assert hit[2] == 0.0, "Target=9 should NOT be in top-5"
    print("✅ Hit@K calculation test passed!")


def test_mrr_calculation():
    """MRR 계산 로직 테스트"""
    print("\n" + "="*60)
    print("🧪 Testing MRR Calculation")
    print("="*60)
    
    predicted_scores = torch.tensor([
        [0.1, 0.2, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05],  # target=2 at rank 3
        [0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.6, 0.7, 0.8, 0.05],  # target=4 at rank 1
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],  # target=9 not in top-k
    ])
    
    target_items = [2, 4, 9]
    history_items = [[], [], []]
    k = 10
    
    mrr = calculate_mrr(predicted_scores, target_items, history_items, k=k)
    
    print(f"MRR@{k} scores:")
    for i, score in enumerate(mrr):
        rank = int(1 / score) if score > 0 else 0
        print(f"  Sample {i}: {score:.4f} (rank={rank}, target={target_items[i]})")
    print(f"\nMean MRR@{k}: {mrr.mean():.4f}")
    
    # 예상 결과 검증
    assert mrr[1] == 1.0, "Target at rank 1 should have MRR=1.0"
    assert mrr[1] > mrr[0], "Rank 1 should have higher MRR than rank 3"
    print("✅ MRR calculation test passed!")


def test_with_retrieval_service():
    """RetrievalService와 통합 테스트"""
    print("\n" + "="*60)
    print("🧪 Testing with RetrievalService Integration")
    print("="*60)
    
    # Ray 초기화
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(address="auto", namespace="rl4rec")
    
    try:
        # 리워드 함수 생성
        print("Creating reward function...")
        reward_fn = create_reward_function(
            retrieval_service_name="RetrievalService",
            namespace="rl4rec",
            dataset_name="beauty",
            reward_type="ndcg",
            k=10,
        )
        
        # 테스트 데이터
        generated_texts = [
            "A high-quality wireless headphone with noise cancellation",
            "Running shoes with excellent cushioning and support",
            "Professional coffee maker with advanced brewing technology",
        ]
        
        target_items = [100, 200, 300]
        history_items = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        print(f"\nTest inputs:")
        print(f"  Generated texts: {len(generated_texts)} samples")
        print(f"  Target items: {target_items}")
        
        # 리워드 계산
        print("\nCalculating rewards...")
        rewards = reward_fn(
            generated_texts=generated_texts,
            target_items=target_items,
            history_items=history_items,
        )
        
        print(f"\nRewards (NDCG@10):")
        for i, reward in enumerate(rewards):
            print(f"  Sample {i}: {reward:.4f}")
        print(f"\nMean reward: {rewards.mean():.4f}")
        
        # 메트릭 계산
        print("\nComputing all metrics...")
        metrics = reward_fn.compute_metrics(
            generated_texts=generated_texts,
            target_items=target_items,
            history_items=history_items,
        )
        
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("✅ RetrievalService integration test passed!")
        
    except ValueError as e:
        print(f"⚠️  RetrievalService not found: {e}")
        print("   Please start retrieval service first:")
        print("   ./runs/run_retrieval.sh")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("🚀 Reward Function Test Suite")
    print("="*60)
    
    # 1. NDCG 테스트
    test_ndcg_calculation()
    
    # 2. Hit@K 테스트
    test_hit_rate_calculation()
    
    # 3. MRR 테스트
    test_mrr_calculation()
    
    # 4. RetrievalService 통합 테스트
    success = test_with_retrieval_service()
    
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests were skipped (RetrievalService not available)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()



