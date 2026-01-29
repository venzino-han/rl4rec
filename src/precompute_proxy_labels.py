"""
Pre-compute proxy labels for each item based on embedding similarities
각 아이템별로 가장 유사한 상위 proxy_k개 아이템과 유사도를 계산하여 저장
"""

import argparse
import json
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def precompute_item_similarities(
    item_embeddings: torch.Tensor,
    proxy_k: int,
    device: str = "cuda",
    random_selection: bool = False,
    similarity_threshold: float = 0.0,
    seed: int = 42,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    모든 아이템 간 유사도를 사전 계산하여 각 아이템별로 가장 유사한 proxy_k개 아이템 저장
    
    Args:
        item_embeddings: [num_items, emb_dim] 아이템 임베딩
        proxy_k: 각 아이템별로 저장할 유사 아이템 개수
        device: 계산에 사용할 디바이스
        random_selection: True이면 threshold 이상의 아이템들 중 랜덤 선택
        similarity_threshold: random_selection=True일 때 필터링할 최소 유사도
        seed: 랜덤 시드
        
    Returns:
        item_proxy_labels: Dict[item_id, List[Tuple[proxy_item_id, similarity]]]
    """
    if random_selection:
        print(f"🔍 Computing item similarities for proxy labels (proxy_k={proxy_k}, random selection, threshold={similarity_threshold})...")
    else:
        print(f"🔍 Computing item similarities for proxy labels (proxy_k={proxy_k}, top-k selection)...")
    
    # 랜덤 시드 설정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 아이템 임베딩 정규화 (코사인 유사도 계산을 위해)
    item_embeddings = item_embeddings.to(device)
    normalized_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=1)
    num_items = len(normalized_embeddings)
    
    print(f"  Total items: {num_items}")
    print(f"  Embedding dimension: {item_embeddings.shape[1]}")
    
    item_proxy_labels = {}
    
    # 배치 단위로 처리하여 메모리 효율성 개선
    batch_size = 1000
    random_order = torch.randperm(num_items, device=device)
    for start_idx in range(1, num_items+1, batch_size):
        end_idx = min(start_idx + batch_size, num_items)
        
        # 현재 배치의 아이템들
        batch_embs = normalized_embeddings[start_idx:end_idx]  # [batch_size, emb_dim]
        
        # 전체 아이템과의 유사도 계산
        similarities = torch.mm(batch_embs, normalized_embeddings.T)  # [batch_size, num_items]
        
        # 각 아이템별로 저장
        for i, emb_idx in enumerate(range(start_idx, end_idx)):
            # item_id는 embedding index와 동일 (0-based 또는 1-based 모두 지원)
            item_id = emb_idx
            
            # 자기 자신 제거
            item_sims = similarities[i]  # [num_items]
            item_sims[emb_idx] = -1.0  # 자기 자신은 제외
            item_sims[0] = -1.0  # 첫 번째 아이템은 제외
            
            if random_selection:
                # 랜덤 선택: 미리 정해진 랜덤 순서대로 순회하면서 threshold 이상만 샘플링
                # 미리 랜덤 순열 생성 (자기 자신 제외)
                # 자기 자신을 제거
                random_order = random_order[random_order != emb_idx]
                
                # 랜덤 순서대로 아이템을 순회하면서 threshold 이상인 것만 선택
                selected_indices = []
                selected_sims = []
                
                for idx in random_order:
                    idx_int = int(idx.item())
                    sim = item_sims[idx_int].item()
                    
                    # threshold 이상이면 선택
                    if sim >= similarity_threshold:
                        selected_indices.append(idx_int)
                        selected_sims.append(sim)
                        
                        # proxy_k개가 채워지면 중단
                        if len(selected_indices) >= proxy_k:
                            break
                
                # 선택된 아이템이 있으면 정규화 및 저장
                if len(selected_indices) == 0:
                    proxy_list = []
                else:
                    selected_sims_tensor = torch.tensor(selected_sims, device=device)
                    
                    # 유사도 정규화 (최대값이 1.0이 되도록)
                    if selected_sims_tensor.max() > 0:
                        normalized_sims = selected_sims_tensor / selected_sims_tensor.max()
                    else:
                        normalized_sims = selected_sims_tensor
                    
                    # List[Tuple[item_id, similarity]] 형태로 저장
                    proxy_list = [
                        (selected_indices[j], float(normalized_sims[j].item()))
                        for j in range(len(selected_indices))
                    ]
            else:
                # 기존 방식: 상위 proxy_k개 선택
                top_k_sims, top_k_indices = torch.topk(
                    item_sims, 
                    k=min(proxy_k, num_items - 1),  # 자기 자신 제외
                    dim=0
                )
                
                # 유사도 정규화 (최대값이 1.0이 되도록)
                if top_k_sims.max() > 0:
                    normalized_sims = top_k_sims / top_k_sims.max()
                else:
                    normalized_sims = top_k_sims
                
                # List[Tuple[item_id, similarity]] 형태로 저장
                proxy_list = [
                    (int(top_k_indices[j].item()), float(normalized_sims[j].item()))
                    for j in range(len(top_k_indices))
                ]
            
            item_proxy_labels[item_id] = proxy_list
        
        if (start_idx // batch_size) % 10 == 0:
            print(f"  Processed {end_idx}/{num_items} items...")
    
    print(f"✓ Completed item similarity computation for {len(item_proxy_labels)} items")
    
    # 통계 정보 출력
    if len(item_proxy_labels) > 0:
        proxy_counts = [len(proxies) for proxies in item_proxy_labels.values()]
        avg_proxy_count = sum(proxy_counts) / len(proxy_counts)
        min_proxy_count = min(proxy_counts)
        max_proxy_count = max(proxy_counts)
        
        print(f"\n  Statistics:")
        print(f"    Average proxies per item: {avg_proxy_count:.2f}")
        print(f"    Min proxies per item: {min_proxy_count}")
        print(f"    Max proxies per item: {max_proxy_count}")
        
        if random_selection:
            items_with_less_than_k = sum(1 for count in proxy_counts if count < proxy_k)
            print(f"    Items with < {proxy_k} proxies: {items_with_less_than_k} ({100*items_with_less_than_k/len(proxy_counts):.1f}%)")
    
    # 예시 출력
    if len(item_proxy_labels) > 0:
        sample_item = list(item_proxy_labels.keys())[0]
        proxy_list = item_proxy_labels[sample_item]
        if random_selection:
            print(f"\n  Example: Item {sample_item} → {len(proxy_list)} randomly selected items (threshold={similarity_threshold})")
        else:
            print(f"\n  Example: Item {sample_item} → Top-{len(proxy_list)} similar items")
        print(f"           First 5 proxies:")
        for proxy_id, sim in proxy_list[:5]:
            print(f"             Item {proxy_id}: similarity={sim:.4f}")
    
    return item_proxy_labels


def main():
    parser = argparse.ArgumentParser(description="Pre-compute proxy labels for items")
    parser.add_argument("--data_name", type=str, required=True, help="Dataset name (e.g., beauty, sports, toys)")
    parser.add_argument("--emb_type", type=str, required=True, help="Embedding type (e.g., review_description, title)")
    parser.add_argument("--emb_model_name", type=str, required=True, help="Embedding model name (e.g., sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--proxy_k", type=int, default=10, help="Number of proxy items per item")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="data_emb", help="Output directory for proxy labels")
    parser.add_argument("--random_selection", action="store_true", help="Randomly select proxy_k items from those above similarity_threshold")
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="Minimum similarity threshold for random selection (only used with --random_selection)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # 임베딩 모델 이름에서 마지막 부분 추출
    emb_model_name_dir = args.emb_model_name.split("/")[-1]
    
    # 아이템 임베딩 파일 경로
    item_embedding_file = f"data_emb/{args.data_name}_{args.emb_type}_{emb_model_name_dir}_emb.pt"
    
    print(f"📦 Loading item embeddings from: {item_embedding_file}")
    if not Path(item_embedding_file).exists():
        raise FileNotFoundError(f"Item embedding file not found: {item_embedding_file}")
    
    item_embeddings = torch.load(item_embedding_file, map_location=args.device)
    print(f"✓ Loaded embeddings for {len(item_embeddings)} items")
    
    # 아이템 간 유사도 계산
    item_proxy_labels = precompute_item_similarities(
        item_embeddings=item_embeddings,
        proxy_k=args.proxy_k,
        device=args.device,
        random_selection=args.random_selection,
        similarity_threshold=args.similarity_threshold,
        seed=args.seed,
    )
    
    # 출력 파일 경로
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일명에 random_selection 정보 포함
    if args.random_selection:
        output_file = output_dir / f"{args.data_name}_proxy_labels_k{args.proxy_k}_random_th{args.similarity_threshold}_{args.emb_type}_{emb_model_name_dir}.json"
    else:
        output_file = output_dir / f"{args.data_name}_proxy_labels_k{args.proxy_k}_{args.emb_type}_{emb_model_name_dir}.json"
    
    print(f"\n💾 Saving proxy labels to: {output_file}")
    
    # JSON 형태로 저장
    # key는 string으로 변환 (JSON은 integer key를 지원하지 않음)
    item_proxy_labels_str_keys = {
        str(item_id): proxy_list 
        for item_id, proxy_list in item_proxy_labels.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(item_proxy_labels_str_keys, f, indent=2)
    
    print(f"✓ Saved proxy labels for {len(item_proxy_labels)} items")
    
    # 파일 크기 출력
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    print("\n✅ Pre-computation completed successfully!")
    print(f"\nTo use these proxy labels, set the following in your training script:")
    print(f"  --proxy_label_reward")
    print(f"  --proxy_k {args.proxy_k}")
    print(f"  --data_name {args.data_name}")
    print(f"  --emb_type {args.emb_type}")
    print(f"  --emb_model_name {args.emb_model_name}")
    if args.random_selection:
        print(f"\nNote: Proxy labels were generated with random selection (threshold={args.similarity_threshold})")


if __name__ == "__main__":
    main()
