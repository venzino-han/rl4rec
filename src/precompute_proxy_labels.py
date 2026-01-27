"""
Pre-compute proxy labels for each item based on embedding similarities
각 아이템별로 가장 유사한 상위 proxy_k개 아이템과 유사도를 계산하여 저장
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple


def precompute_item_similarities(
    item_embeddings: torch.Tensor,
    proxy_k: int,
    device: str = "cuda",
) -> Dict[int, List[Tuple[int, float]]]:
    """
    모든 아이템 간 유사도를 사전 계산하여 각 아이템별로 가장 유사한 proxy_k개 아이템 저장
    
    Args:
        item_embeddings: [num_items, emb_dim] 아이템 임베딩
        proxy_k: 각 아이템별로 저장할 유사 아이템 개수
        device: 계산에 사용할 디바이스
        
    Returns:
        item_proxy_labels: Dict[item_id, List[Tuple[proxy_item_id, similarity]]]
    """
    print(f"🔍 Computing item similarities for proxy labels (proxy_k={proxy_k})...")
    
    # 아이템 임베딩 정규화 (코사인 유사도 계산을 위해)
    item_embeddings = item_embeddings.to(device)
    normalized_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=1)
    num_items = len(normalized_embeddings)
    
    print(f"  Total items: {num_items}")
    print(f"  Embedding dimension: {item_embeddings.shape[1]}")
    
    item_proxy_labels = {}
    
    # 배치 단위로 처리하여 메모리 효율성 개선
    batch_size = 1000
    for start_idx in range(1, num_items+1, batch_size):
        end_idx = min(start_idx + batch_size, num_items)
        
        # 현재 배치의 아이템들
        batch_embs = normalized_embeddings[start_idx:end_idx]  # [batch_size, emb_dim]
        
        # 전체 아이템과의 유사도 계산
        similarities = torch.mm(batch_embs, normalized_embeddings.T)  # [batch_size, num_items]
        
        # 각 아이템에 대해 상위 proxy_k+1개 추출 (자기 자신 포함)
        top_k_sims, top_k_indices = torch.topk(
            similarities, 
            k=min(proxy_k + 1, num_items), 
            dim=1
        )
        
        # 각 아이템별로 저장
        for i, emb_idx in enumerate(range(start_idx, end_idx)):
            # item_id는 embedding index와 동일 (0-based 또는 1-based 모두 지원)
            item_id = emb_idx
            
            # 자기 자신을 제외 (보통 가장 유사도가 높음)
            proxy_indices = top_k_indices[i]  # [proxy_k+1]
            proxy_sims = top_k_sims[i]  # [proxy_k+1]
            
            # 자기 자신 제거 (임베딩 인덱스 기준)
            mask = proxy_indices != emb_idx
            proxy_indices = proxy_indices[mask][:proxy_k]
            proxy_sims = proxy_sims[mask][:proxy_k]
            
            # 유사도 정규화 (최대값이 1.0이 되도록)
            if proxy_sims.max() > 0:
                normalized_sims = proxy_sims / proxy_sims.max()
            else:
                normalized_sims = proxy_sims
            
            # List[Tuple[item_id, similarity]] 형태로 저장
            proxy_list = [
                (int(proxy_indices[j].item()), float(normalized_sims[j].item()))
                for j in range(len(proxy_indices))
            ]
            item_proxy_labels[item_id] = proxy_list
        
        if (start_idx // batch_size) % 10 == 0:
            print(f"  Processed {end_idx}/{num_items} items...")
    
    print(f"✓ Completed item similarity computation for {len(item_proxy_labels)} items")
    
    # 예시 출력
    if len(item_proxy_labels) > 0:
        sample_item = list(item_proxy_labels.keys())[0]
        proxy_list = item_proxy_labels[sample_item]
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
    )
    
    # 출력 파일 경로
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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


if __name__ == "__main__":
    main()
