"""
Find trigger items from sequential_data.txt file.

This script reads the sequential_data.txt file directly and finds trigger items
for each user based on the split definition:
- train: items before index -3 (train target)
- val: items before index -2 (val target)
- test: items before index -1 (test target)

File format (sequential_data.txt):
  user_id item1 item2 ... itemN
  
Where:
  - First number: user_id
  - Rest: sequential purchase history
  - Index -3: train target
  - Index -2: val target
  - Index -1: test target
"""

import torch
import json
import os
from typing import Dict, List, Tuple
import argparse


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between vectors.
    
    Args:
        a: Tensor of shape (d,) or (n, d)
        b: Tensor of shape (d,) or (m, d)
    
    Returns:
        Similarity score(s)
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    
    # Normalize
    # a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    # b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute similarity
    similarity = torch.mm(a, b.t())
    return similarity


def load_sequential_data(filepath: str) -> Dict[int, List[int]]:
    """
    Load sequential data from file.
    
    Args:
        filepath: Path to sequential_data.txt
    
    Returns:
        Dict mapping user_id to list of item_ids
    """
    user_sequences = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) < 2:
                continue
            
            user_id = parts[0]
            items = parts[1:]
            user_sequences[user_id] = items
    
    return user_sequences


def get_split_items(items: List[int], split: str) -> Tuple[List[int], int]:
    """
    Get past items and target for a specific split.
    
    Args:
        items: Full sequence of items
        split: 'train', 'val', or 'test'
    
    Returns:
        Tuple of (past_items, target_item)
        - For train: items before index -3, target at -3
        - For val: items before index -2, target at -2
        - For test: items before index -1, target at -1
    """
    if len(items) < 3:
        # Not enough items for all splits
        return [], -1
    
    if split == 'train':
        # Items before train target (index -3)
        past_items = items[:-3]
        target = items[-3]
    elif split == 'val' or split == 'valid':
        # Items before val target (index -2)
        past_items = items[:-2]
        target = items[-2]
    elif split == 'test':
        # Items before test target (index -1)
        past_items = items[:-1]
        target = items[-1]
    else:
        raise ValueError(f"Unknown split: {split}")
    
    return past_items, target


def find_trigger_items_for_split(
    user_sequences: Dict[int, List[int]],
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    split: str
) -> Dict[str, int]:
    """
    Find trigger items for all users in a specific split.
    
    Args:
        user_sequences: Dict mapping user_id to full item sequence
        user_embeddings: User embeddings tensor (num_users, emb_dim)
        item_embeddings: Item embeddings tensor (num_items, emb_dim)
        split: 'train', 'val', or 'test'
    
    Returns:
        Dict mapping user_id (str) to trigger_item_id (int)
    """
    trigger_items = {}
    skipped = 0
    
    print(f"Processing {len(user_sequences)} users for {split} split...")
    
    for idx, (user_id, full_items) in enumerate(user_sequences.items()):
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(user_sequences)} users...")
        
        # Get past items for this split
        past_items, target = get_split_items(full_items, split)
        
        if len(past_items) == 0:
            skipped += 1
            continue
        
        # Get user embedding (user_id is 1-indexed in file, 0-indexed in tensor)
        user_idx = user_id - 1
        if user_idx < 0 or user_idx >= user_embeddings.shape[0]:
            print(f"Warning: User {user_id} index out of range")
            skipped += 1
            continue
        
        user_emb = user_embeddings[user_idx]
        
        # Get embeddings for all past items
        valid_item_ids = []
        valid_item_embs = []
        
        for item_id in past_items:
            if item_id >= 0 and item_id < item_embeddings.shape[0]:
                valid_item_ids.append(item_id)
                valid_item_embs.append(item_embeddings[item_id])
        
        if len(valid_item_embs) == 0:
            skipped += 1
            continue
        
        # Stack item embeddings
        item_embs = torch.stack(valid_item_embs)
        
        # Compute similarities
        similarities = cosine_similarity(user_emb, item_embs)  # (1, num_items)
        
        # Find item with highest similarity
        max_idx = similarities.argmax().item()
        trigger_item_id = valid_item_ids[max_idx]
        
        trigger_items[str(user_id)] = trigger_item_id
    
    print(f"  Completed! Found trigger items for {len(trigger_items)} users.")
    print(f"  Skipped {skipped} users (not enough items or invalid indices)")
    
    return trigger_items


def process_dataset(
    dataset_name: str,
    data_dir: str = "data",
    sasrec_results_dir: str = "sasrec_results",
    output_dir: str = "sasrec_results/trigger_items_from_sequential"
):
    """
    Process a single dataset to find trigger items from sequential_data.txt.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'toys', 'beauty', 'sports', 'yelp')
        data_dir: Directory containing data folders
        sasrec_results_dir: Directory containing SASRec embeddings
        output_dir: Directory to save trigger items JSON files
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sequential data
    seq_data_path = f"{data_dir}/{dataset_name}/sequential_data.txt"
    if not os.path.exists(seq_data_path):
        print(f"Error: Sequential data not found at {seq_data_path}")
        return
    
    print(f"\nLoading sequential data from {seq_data_path}...")
    user_sequences = load_sequential_data(seq_data_path)
    print(f"  Loaded sequences for {len(user_sequences)} users")
    
    # Load item embeddings (shared across splits)
    item_emb_path = f"{sasrec_results_dir}/SASRec_{dataset_name}_item_embeddings.pt"
    if not os.path.exists(item_emb_path):
        print(f"Error: Item embeddings not found at {item_emb_path}")
        return
    
    print(f"\nLoading item embeddings from {item_emb_path}...")
    item_embeddings = torch.load(item_emb_path)
    print(f"  Item embeddings shape: {item_embeddings.shape}")
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n--- Processing {split} split ---")
        
        # Try both 'val' and 'valid' naming for embeddings
        emb_split = 'val' if split == 'val' else split
        user_emb_paths = [
            f"{sasrec_results_dir}/SASRec_{dataset_name}_{emb_split}_user_embeddings.pt",
            f"{sasrec_results_dir}/SASRec_{dataset_name}_valid_user_embeddings.pt"
        ]
        
        user_embeddings = None
        for user_emb_path in user_emb_paths:
            if os.path.exists(user_emb_path):
                print(f"  Loading user embeddings from {user_emb_path}...")
                user_embeddings = torch.load(user_emb_path)
                print(f"    User embeddings shape: {user_embeddings.shape}")
                break
        
        if user_embeddings is None:
            print(f"  Warning: User embeddings not found for {split} split, skipping...")
            continue
        
        # Find trigger items
        trigger_items = find_trigger_items_for_split(
            user_sequences,
            user_embeddings,
            item_embeddings,
            split
        )
        
        # Save results
        output_path = f"{output_dir}/SASRec_{dataset_name}_{split}_trigger_items.json"
        print(f"  Saving trigger items to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(trigger_items, f, indent=2)
        
        print(f"  Successfully saved {len(trigger_items)} trigger items!")


def main():
    parser = argparse.ArgumentParser(
        description="Find trigger items from sequential_data.txt based on split indices"
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['toys', 'beauty', 'sports', 'yelp'],
        help='List of datasets to process (default: toys beauty sports yelp)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing data folders (default: data)'
    )
    parser.add_argument(
        '--sasrec_results_dir',
        type=str,
        default='sasrec_results',
        help='Directory containing SASRec embeddings (default: sasrec_results)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='sasrec_results/trigger_items_from_sequential',
        help='Directory to save trigger items (default: sasrec_results/trigger_items_from_sequential)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Finding Trigger Items from Sequential Data")
    print("="*60)
    print(f"Datasets: {args.datasets}")
    print(f"Data directory: {args.data_dir}")
    print(f"SASRec results directory: {args.sasrec_results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("\nSplit definitions:")
    print("  - train: items before index -3 (train target at -3)")
    print("  - val:   items before index -2 (val target at -2)")
    print("  - test:  items before index -1 (test target at -1)")
    
    for dataset in args.datasets:
        try:
            process_dataset(
                dataset_name=dataset,
                data_dir=args.data_dir,
                sasrec_results_dir=args.sasrec_results_dir,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"\nError processing dataset {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("All datasets processed!")
    print("="*60)


if __name__ == "__main__":
    main()
