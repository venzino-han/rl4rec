"""
Analyze the correlation between user's previous item popularity and next item popularity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import os
from typing import List, Tuple, Dict
import json


def load_sequential_data(filepath: str) -> Dict[int, List[int]]:
    """
    Load sequential data from text file
    
    Args:
        filepath: Path to sequential_data.txt
        
    Returns:
        Dictionary mapping user_id to list of item_ids
    """
    user_sequences = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            items = [int(x) for x in parts[1:]]
            user_sequences[user_id] = items
    return user_sequences


def split_leave_one_out(sequences: Dict[int, List[int]]) -> Tuple[Dict, Dict, Dict]:
    """
    Split sequences using leave-one-out strategy
    
    Args:
        sequences: Dictionary of user_id -> item sequence
        
    Returns:
        train_data, valid_data, test_data
        Each is a dict with 'user_id', 'history', 'target'
    """
    train_data = []
    valid_data = []
    test_data = []
    
    for user_id, items in sequences.items():
        if len(items) < 3:
            # Need at least 3 items for train, valid, test split
            continue
        
        # Train: all items except last 2
        # Valid: all items except last 1, target is second to last
        # Test: all items, target is last
        
        train_data.append({
            'user_id': user_id,
            'history': items[:-2],
            'target': items[-2]
        })
        
        valid_data.append({
            'user_id': user_id,
            'history': items[:-1],
            'target': items[-1]
        })
        
        test_data.append({
            'user_id': user_id,
            'history': items[:-1],
            'target': items[-1]
        })
    
    return train_data, valid_data, test_data


def calculate_popularity_features(
    data: List[Dict],
    item_popularity: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate popularity features for each user
    
    Args:
        data: List of dicts with 'history' and 'target'
        item_popularity: Array of item popularities
        
    Returns:
        target_pop: Popularity of target items
        last_1_pop: Popularity of last 1 item before target
        last_4_pop: Average popularity of last 4 items before target
        all_pop: Average popularity of all items before target
    """
    target_pop = []
    last_1_pop = []
    last_4_pop = []
    all_pop = []
    
    for entry in data:
        history = entry['history']
        target = entry['target']
        
        if len(history) == 0:
            continue
            
        # Target popularity
        target_pop.append(item_popularity[target])
        
        # Last 1 item popularity
        last_1_pop.append(item_popularity[history[-1]])
        
        # Last 4 items average popularity
        last_k = min(4, len(history))
        last_4_items = history[-last_k:]
        last_4_pop.append(np.mean([item_popularity[item] for item in last_4_items]))
        
        # All items average popularity
        all_pop.append(np.mean([item_popularity[item] for item in history]))
    
    return (
        np.array(target_pop),
        np.array(last_1_pop),
        np.array(last_4_pop),
        np.array(all_pop)
    )


def create_correlation_plots(
    target_pop: np.ndarray,
    last_1_pop: np.ndarray,
    last_4_pop: np.ndarray,
    all_pop: np.ndarray,
    split_name: str,
    output_dir: str
):
    """
    Create correlation plots and save them
    
    Args:
        target_pop: Target item popularity
        last_1_pop: Last 1 item popularity
        last_4_pop: Last 4 items average popularity
        all_pop: All items average popularity
        split_name: Name of the split (train/valid/test)
        output_dir: Directory to save plots
    """
    # Calculate correlations
    corr_last_1, pval_last_1 = stats.pearsonr(target_pop, last_1_pop)
    corr_last_4, pval_last_4 = stats.pearsonr(target_pop, last_4_pop)
    corr_all, pval_all = stats.pearsonr(target_pop, all_pop)
    
    # Calculate Spearman correlations as well
    spearman_last_1, spval_last_1 = stats.spearmanr(target_pop, last_1_pop)
    spearman_last_4, spval_last_4 = stats.spearmanr(target_pop, last_4_pop)
    spearman_all, spval_all = stats.spearmanr(target_pop, all_pop)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Last 1 item
    axes[0].scatter(last_1_pop, target_pop, alpha=0.3, s=10)
    axes[0].set_xlabel('Last Item Popularity', fontsize=12)
    axes[0].set_ylabel('Target Item Popularity', fontsize=12)
    axes[0].set_title(
        f'Last 1 Item vs Target\n'
        f'Pearson: {corr_last_1:.4f} (p={pval_last_1:.2e})\n'
        f'Spearman: {spearman_last_1:.4f} (p={spval_last_1:.2e})',
        fontsize=11
    )
    axes[0].grid(True, alpha=0.3)
    
    # Add regression line
    z = np.polyfit(last_1_pop, target_pop, 1)
    p = np.poly1d(z)
    axes[0].plot(last_1_pop, p(last_1_pop), "r--", alpha=0.8, linewidth=2)
    
    # Plot 2: Last 4 items average
    axes[1].scatter(last_4_pop, target_pop, alpha=0.3, s=10)
    axes[1].set_xlabel('Last 4 Items Avg Popularity', fontsize=12)
    axes[1].set_ylabel('Target Item Popularity', fontsize=12)
    axes[1].set_title(
        f'Last 4 Items Avg vs Target\n'
        f'Pearson: {corr_last_4:.4f} (p={pval_last_4:.2e})\n'
        f'Spearman: {spearman_last_4:.4f} (p={spval_last_4:.2e})',
        fontsize=11
    )
    axes[1].grid(True, alpha=0.3)
    
    # Add regression line
    z = np.polyfit(last_4_pop, target_pop, 1)
    p = np.poly1d(z)
    axes[1].plot(last_4_pop, p(last_4_pop), "r--", alpha=0.8, linewidth=2)
    
    # Plot 3: All items average
    axes[2].scatter(all_pop, target_pop, alpha=0.3, s=10)
    axes[2].set_xlabel('All Items Avg Popularity', fontsize=12)
    axes[2].set_ylabel('Target Item Popularity', fontsize=12)
    axes[2].set_title(
        f'All Items Avg vs Target\n'
        f'Pearson: {corr_all:.4f} (p={pval_all:.2e})\n'
        f'Spearman: {spearman_all:.4f} (p={spval_all:.2e})',
        fontsize=11
    )
    axes[2].grid(True, alpha=0.3)
    
    # Add regression line
    z = np.polyfit(all_pop, target_pop, 1)
    p = np.poly1d(z)
    axes[2].plot(all_pop, p(all_pop), "r--", alpha=0.8, linewidth=2)
    
    plt.suptitle(f'Popularity Correlation Analysis - {split_name.upper()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'popularity_correlation_{split_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()
    
    # Create additional heatmap/density plot for better visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Hexbin plots for better density visualization
    hb1 = axes[0].hexbin(last_1_pop, target_pop, gridsize=50, cmap='YlOrRd', mincnt=1)
    axes[0].set_xlabel('Last Item Popularity', fontsize=12)
    axes[0].set_ylabel('Target Item Popularity', fontsize=12)
    axes[0].set_title(f'Last 1 Item vs Target (Density)\nCorr: {corr_last_1:.4f}', fontsize=11)
    plt.colorbar(hb1, ax=axes[0], label='Count')
    
    hb2 = axes[1].hexbin(last_4_pop, target_pop, gridsize=50, cmap='YlOrRd', mincnt=1)
    axes[1].set_xlabel('Last 4 Items Avg Popularity', fontsize=12)
    axes[1].set_ylabel('Target Item Popularity', fontsize=12)
    axes[1].set_title(f'Last 4 Items Avg vs Target (Density)\nCorr: {corr_last_4:.4f}', fontsize=11)
    plt.colorbar(hb2, ax=axes[1], label='Count')
    
    hb3 = axes[2].hexbin(all_pop, target_pop, gridsize=50, cmap='YlOrRd', mincnt=1)
    axes[2].set_xlabel('All Items Avg Popularity', fontsize=12)
    axes[2].set_ylabel('Target Item Popularity', fontsize=12)
    axes[2].set_title(f'All Items Avg vs Target (Density)\nCorr: {corr_all:.4f}', fontsize=11)
    plt.colorbar(hb3, ax=axes[2], label='Count')
    
    plt.suptitle(f'Popularity Correlation Analysis (Density) - {split_name.upper()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path_density = os.path.join(output_dir, f'popularity_correlation_{split_name}_density.png')
    plt.savefig(output_path_density, dpi=300, bbox_inches='tight')
    print(f"Saved density plot to {output_path_density}")
    plt.close()
    
    # Return statistics
    return {
        'pearson': {
            'last_1': {'correlation': corr_last_1, 'p_value': pval_last_1},
            'last_4': {'correlation': corr_last_4, 'p_value': pval_last_4},
            'all': {'correlation': corr_all, 'p_value': pval_all}
        },
        'spearman': {
            'last_1': {'correlation': spearman_last_1, 'p_value': spval_last_1},
            'last_4': {'correlation': spearman_last_4, 'p_value': spval_last_4},
            'all': {'correlation': spearman_all, 'p_value': spval_all}
        }
    }


def create_comparison_plot(all_stats: Dict, output_dir: str):
    """
    Create a comparison plot of correlations across different splits
    
    Args:
        all_stats: Dictionary containing statistics for all splits
        output_dir: Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    splits = ['train', 'valid', 'test']
    features = ['last_1', 'last_4', 'all']
    feature_labels = ['Last 1 Item', 'Last 4 Items Avg', 'All Items Avg']
    
    # Pearson correlations
    pearson_data = {feat: [] for feat in features}
    for split in splits:
        if split in all_stats:
            for feat in features:
                pearson_data[feat].append(all_stats[split]['pearson'][feat]['correlation'])
    
    x = np.arange(len(splits))
    width = 0.25
    
    for i, (feat, label) in enumerate(zip(features, feature_labels)):
        ax1.bar(x + i * width, pearson_data[feat], width, label=label, alpha=0.8)
    
    ax1.set_xlabel('Split', fontsize=12)
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_title('Pearson Correlation Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([s.upper() for s in splits])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Spearman correlations
    spearman_data = {feat: [] for feat in features}
    for split in splits:
        if split in all_stats:
            for feat in features:
                spearman_data[feat].append(all_stats[split]['spearman'][feat]['correlation'])
    
    for i, (feat, label) in enumerate(zip(features, feature_labels)):
        ax2.bar(x + i * width, spearman_data[feat], width, label=label, alpha=0.8)
    
    ax2.set_xlabel('Split', fontsize=12)
    ax2.set_ylabel('Spearman Correlation', fontsize=12)
    ax2.set_title('Spearman Correlation Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([s.upper() for s in splits])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'correlation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze popularity correlations')
    parser.add_argument('--data_name', type=str, default='beauty',
                        help='Dataset name (beauty, sports, toys, yelp)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Set up paths
    data_dir = f'/home/work/llm4hot_earth/33/77/rl4rec/data/{args.data_name}'
    sequential_path = os.path.join(data_dir, 'sequential_data.txt')
    popularity_path = os.path.join(data_dir, 'item_popularity.npy')
    
    if args.output_dir is None:
        output_dir = f'/home/work/llm4hot_earth/33/77/rl4rec/results/popularity_analysis_{args.data_name}'
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_dir}...")
    
    # Load data
    sequences = load_sequential_data(sequential_path)
    item_popularity = np.load(popularity_path)
    
    print(f"Loaded {len(sequences)} user sequences")
    print(f"Item popularity shape: {item_popularity.shape}")
    
    # Split data
    print("\nSplitting data using leave-one-out strategy...")
    train_data, valid_data, test_data = split_leave_one_out(sequences)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Analyze each split
    all_stats = {}
    
    for split_name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
        print(f"\nAnalyzing {split_name} split...")
        
        # Calculate features
        target_pop, last_1_pop, last_4_pop, all_pop = calculate_popularity_features(
            data, item_popularity
        )
        
        print(f"  Number of samples: {len(target_pop)}")
        print(f"  Target popularity - mean: {target_pop.mean():.6f}, std: {target_pop.std():.6f}")
        print(f"  Last 1 popularity - mean: {last_1_pop.mean():.6f}, std: {last_1_pop.std():.6f}")
        print(f"  Last 4 popularity - mean: {last_4_pop.mean():.6f}, std: {last_4_pop.std():.6f}")
        print(f"  All popularity - mean: {all_pop.mean():.6f}, std: {all_pop.std():.6f}")
        
        # Create plots and get statistics
        stats = create_correlation_plots(
            target_pop, last_1_pop, last_4_pop, all_pop,
            split_name, output_dir
        )
        
        all_stats[split_name] = stats
        
        print(f"\n  Pearson Correlations:")
        print(f"    Last 1 item: {stats['pearson']['last_1']['correlation']:.4f} (p={stats['pearson']['last_1']['p_value']:.2e})")
        print(f"    Last 4 items: {stats['pearson']['last_4']['correlation']:.4f} (p={stats['pearson']['last_4']['p_value']:.2e})")
        print(f"    All items: {stats['pearson']['all']['correlation']:.4f} (p={stats['pearson']['all']['p_value']:.2e})")
        
        print(f"\n  Spearman Correlations:")
        print(f"    Last 1 item: {stats['spearman']['last_1']['correlation']:.4f} (p={stats['spearman']['last_1']['p_value']:.2e})")
        print(f"    Last 4 items: {stats['spearman']['last_4']['correlation']:.4f} (p={stats['spearman']['last_4']['p_value']:.2e})")
        print(f"    All items: {stats['spearman']['all']['correlation']:.4f} (p={stats['spearman']['all']['p_value']:.2e})")
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot(all_stats, output_dir)
    
    # Save statistics to JSON
    stats_path = os.path.join(output_dir, 'correlation_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

