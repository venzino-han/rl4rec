import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


def compute_cosine_similarity(emb1, emb2):
    """
    Compute cosine similarity between two embedding tensors.
    
    Args:
        emb1: Tensor of shape [num_users, embed_dim]
        emb2: Tensor of shape [num_users, embed_dim]
    
    Returns:
        Tensor of shape [num_users] with cosine similarity for each user
    """
    # Normalize embeddings
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    
    # Compute cosine similarity (element-wise dot product)
    cos_sim = (emb1_norm * emb2_norm).sum(dim=1)
    
    return cos_sim


def main(args):
    print(f"Loading embeddings from:")
    print(f"  Vanilla: {args.vanilla_emb_path}")
    print(f"  Target:  {args.target_emb_path}")
    
    # Load embeddings
    vanilla_emb = torch.load(args.vanilla_emb_path)
    target_emb = torch.load(args.target_emb_path)
    
    print(f"\nVanilla embedding shape: {vanilla_emb.shape}")
    print(f"Target embedding shape:  {target_emb.shape}")
    
    # Validate shapes match
    assert vanilla_emb.shape == target_emb.shape, \
        f"Shape mismatch: {vanilla_emb.shape} vs {target_emb.shape}"
    
    num_users = vanilla_emb.shape[0]
    
    # Compute cosine similarity
    print("\nComputing cosine similarity...")
    cos_sim = compute_cosine_similarity(vanilla_emb, target_emb)
    
    print(f"Mean cosine similarity: {cos_sim.mean().item():.4f}")
    print(f"Std cosine similarity:  {cos_sim.std().item():.4f}")
    print(f"Min cosine similarity:  {cos_sim.min().item():.4f}")
    print(f"Max cosine similarity:  {cos_sim.max().item():.4f}")
    
    # Get top 25% users
    top_k = int(num_users * 0.25)
    top_values, top_indices = torch.topk(cos_sim, k=top_k)
    
    # Convert to user_ids (1-indexed)
    top_user_ids = (top_indices + 1).tolist()
    top_similarities = top_values.tolist()
    
    print(f"\nTop 25% users: {top_k} out of {num_users}")
    print(f"Threshold similarity: {top_values[-1].item():.4f}")
    
    # Prepare output
    output_data = {
        "num_total_users": num_users,
        "num_top_users": top_k,
        "top_percentage": 25.0,
        "threshold_similarity": float(top_values[-1].item()),
        "mean_similarity": float(cos_sim.mean().item()),
        "std_similarity": float(cos_sim.std().item()),
        "min_similarity": float(cos_sim.min().item()),
        "max_similarity": float(cos_sim.max().item()),
        "vanilla_emb_path": str(args.vanilla_emb_path),
        "target_emb_path": str(args.target_emb_path),
        "top_user_ids": top_user_ids,
        "top_similarities": top_similarities
    }
    
    # Save to JSON
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Sample top user_ids: {top_user_ids[:10]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare cosine similarity between vanilla and target embeddings"
    )
    parser.add_argument(
        "--vanilla_emb_path",
        type=str,
        required=True,
        help="Path to vanilla embedding file (.pt)"
    )
    parser.add_argument(
        "--target_emb_path",
        type=str,
        required=True,
        help="Path to target embedding file (.pt)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output JSON file"
    )
    
    args = parser.parse_args()
    main(args)

