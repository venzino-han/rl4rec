"""
사용자 임베딩 유사도 기반 필터링 스크립트

각 유저별로 target_preference와 reasoning embedding의 유사도를 계산하여
상위 XX%의 유저만 선정하고 ID 리스트를 JSON으로 저장
"""

import os
import torch
import json
import numpy as np
from typing import Dict, List, Tuple
import argparse
from pathlib import Path
from torch.nn.functional import cosine_similarity


def load_embedding(file_path: str) -> Dict:
    """임베딩 파일 로드"""
    print(f"Loading embedding from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file not found: {file_path}")
    
    data = torch.load(file_path, map_location='cpu')
    print(f"Loaded embedding with keys: {data.keys() if isinstance(data, dict) else 'tensor'}")
    return data


def calculate_user_similarities(
    target_emb_data: Dict,
    reasoning_emb_data: Dict
) -> Tuple[List[str], List[float]]:
    """
    각 유저별 target_preference와 reasoning embedding 간 코사인 유사도 계산
    
    Returns:
        user_ids: 유저 ID 리스트
        similarities: 유사도 값 리스트
    """
    # 임베딩 데이터 구조 확인
    if isinstance(target_emb_data, dict):
        if 'embeddings' in target_emb_data:
            target_embeddings = target_emb_data['embeddings']
            target_ids = target_emb_data.get('user_ids', list(range(len(target_embeddings))))
        else:
            # dict의 키가 user_id인 경우
            target_ids = list(target_emb_data.keys())
            target_embeddings = [target_emb_data[uid] for uid in target_ids]
    else:
        # tensor인 경우
        target_embeddings = target_emb_data
        target_ids = list(range(len(target_embeddings)))
    
    if isinstance(reasoning_emb_data, dict):
        if 'embeddings' in reasoning_emb_data:
            reasoning_embeddings = reasoning_emb_data['embeddings']
            reasoning_ids = reasoning_emb_data.get('user_ids', list(range(len(reasoning_embeddings))))
        else:
            reasoning_ids = list(reasoning_emb_data.keys())
            reasoning_embeddings = [reasoning_emb_data[uid] for uid in reasoning_ids]
    else:
        reasoning_embeddings = reasoning_emb_data
        reasoning_ids = list(range(len(reasoning_embeddings)))
    
    # tensor로 변환
    if not isinstance(target_embeddings, torch.Tensor):
        target_embeddings = torch.stack([torch.tensor(e) if not isinstance(e, torch.Tensor) else e 
                                        for e in target_embeddings])
    if not isinstance(reasoning_embeddings, torch.Tensor):
        reasoning_embeddings = torch.stack([torch.tensor(e) if not isinstance(e, torch.Tensor) else e 
                                           for e in reasoning_embeddings])
    
    print(f"Target embeddings shape: {target_embeddings.shape}")
    print(f"Reasoning embeddings shape: {reasoning_embeddings.shape}")
    
    # 유저 ID 매칭
    common_ids = set(target_ids) & set(reasoning_ids)
    print(f"Common user IDs: {len(common_ids)}")
    
    if len(common_ids) == 0:
        # ID가 매칭되지 않으면 순서대로 매칭
        print("No common IDs found, matching by index...")
        min_len = min(len(target_ids), len(reasoning_ids))
        common_ids = list(range(min_len))
        user_ids = [str(i) for i in common_ids]
        target_emb_subset = target_embeddings[:min_len]
        reasoning_emb_subset = reasoning_embeddings[:min_len]
    else:
        # 공통 ID에 대한 임베딩만 추출
        common_ids = sorted(list(common_ids))
        user_ids = [str(uid) for uid in common_ids]
        
        target_idx_map = {uid: idx for idx, uid in enumerate(target_ids)}
        reasoning_idx_map = {uid: idx for idx, uid in enumerate(reasoning_ids)}
        
        target_emb_subset = torch.stack([target_embeddings[target_idx_map[uid]] for uid in common_ids])
        reasoning_emb_subset = torch.stack([reasoning_embeddings[reasoning_idx_map[uid]] for uid in common_ids])
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(target_emb_subset, reasoning_emb_subset, dim=1)
    similarities = similarities.cpu().numpy().tolist()
    
    print(f"Calculated similarities for {len(user_ids)} users")
    print(f"Similarity stats - Min: {min(similarities):.4f}, Max: {max(similarities):.4f}, Mean: {np.mean(similarities):.4f}")
    
    return user_ids, similarities


def select_top_users(
    user_ids: List[str],
    similarities: List[float],
    top_percentage: float
) -> List[str]:
    """
    상위 XX% 유저 선정
    
    Args:
        user_ids: 유저 ID 리스트
        similarities: 유사도 값 리스트
        top_percentage: 선정할 상위 비율 (0-100)
    
    Returns:
        선정된 유저 ID 리스트
    """
    # 유사도 기준 정렬
    sorted_indices = np.argsort(similarities)[::-1]  # 내림차순
    
    # 상위 XX% 선정
    top_k = int(len(user_ids) * (top_percentage / 100.0))
    top_k = max(1, top_k)  # 최소 1명
    
    selected_indices = sorted_indices[:top_k]
    selected_user_ids = [user_ids[idx] for idx in selected_indices]
    selected_similarities = [similarities[idx] for idx in selected_indices]
    
    print(f"\nSelected top {top_percentage}% users: {len(selected_user_ids)}/{len(user_ids)}")
    print(f"Similarity range of selected users: {min(selected_similarities):.4f} - {max(selected_similarities):.4f}")
    
    return selected_user_ids


def save_selected_users(
    user_ids: List[str],
    output_path: str,
    metadata: Dict = None
):
    """선정된 유저 ID를 JSON 파일로 저장"""
    output_data = {
        "selected_user_ids": user_ids,
        "num_users": len(user_ids),
    }
    
    if metadata:
        output_data["metadata"] = metadata
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved selected user IDs to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Select users based on embedding similarity')
    parser.add_argument('--target_emb_path', type=str, required=True,
                       help='Path to target preference embedding file')
    parser.add_argument('--reasoning_emb_path', type=str, required=True,
                       help='Path to reasoning embedding file')
    parser.add_argument('--top_percentage', type=float, default=50.0,
                       help='Top percentage of users to select (0-100)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output JSON file path for selected user IDs')
    parser.add_argument('--dataset', type=str, default='toys',
                       help='Dataset name for metadata')
    parser.add_argument('--split', type=str, default='train',
                       help='Data split (train/valid/test)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("User Selection by Embedding Similarity")
    print("=" * 80)
    print(f"Target embedding: {args.target_emb_path}")
    print(f"Reasoning embedding: {args.reasoning_emb_path}")
    print(f"Top percentage: {args.top_percentage}%")
    print(f"Output path: {args.output_path}")
    print("=" * 80)
    
    # 1. 임베딩 로드
    target_emb_data = load_embedding(args.target_emb_path)
    reasoning_emb_data = load_embedding(args.reasoning_emb_path)
    
    # 2. 유사도 계산
    print("\nCalculating similarities...")
    user_ids, similarities = calculate_user_similarities(target_emb_data, reasoning_emb_data)
    
    # 3. 상위 유저 선정
    selected_user_ids = select_top_users(user_ids, similarities, args.top_percentage)
    
    # 4. 결과 저장
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "top_percentage": args.top_percentage,
        "total_users": len(user_ids),
        "selected_users": len(selected_user_ids),
        "target_emb_file": os.path.basename(args.target_emb_path),
        "reasoning_emb_file": os.path.basename(args.reasoning_emb_path),
    }
    
    save_selected_users(selected_user_ids, args.output_path, metadata)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()


