"""
Test loading and inspecting pre-computed proxy labels
"""

import argparse
import json
from pathlib import Path


def test_proxy_labels(file_path: str):
    """
    Test loading proxy labels JSON file
    
    Args:
        file_path: Path to proxy labels JSON file
    """
    print(f"📦 Testing proxy labels file: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Load JSON file
    print(f"⏳ Loading JSON...")
    with open(file_path, 'r') as f:
        proxy_labels = json.load(f)
    
    print(f"✓ Loaded successfully!")
    
    # Basic statistics
    num_items = len(proxy_labels)
    print(f"\n📊 Statistics:")
    print(f"  Total items: {num_items}")
    
    # Check structure
    sample_items = list(proxy_labels.keys())[:5]
    print(f"\n🔍 Sample items (first 5):")
    
    for item_id_str in sample_items:
        proxy_list = proxy_labels[item_id_str]
        num_proxies = len(proxy_list)
        
        print(f"\n  Item {item_id_str}:")
        print(f"    Number of proxy items: {num_proxies}")
        
        if num_proxies > 0:
            # Check first few proxies
            first_proxies = proxy_list[:3]
            print(f"    First 3 proxies:")
            for proxy_id, similarity in first_proxies:
                print(f"      → Item {proxy_id}: similarity={similarity:.4f}")
    
    # Check if all items have proxies
    items_without_proxies = [k for k, v in proxy_labels.items() if len(v) == 0]
    print(f"\n📈 Items without proxies: {len(items_without_proxies)} / {num_items}")
    
    if len(items_without_proxies) > 0:
        print(f"  Example items without proxies: {items_without_proxies[:10]}")
    
    # Check proxy counts distribution
    proxy_counts = [len(v) for v in proxy_labels.values()]
    min_count = min(proxy_counts) if proxy_counts else 0
    max_count = max(proxy_counts) if proxy_counts else 0
    avg_count = sum(proxy_counts) / len(proxy_counts) if proxy_counts else 0
    
    print(f"\n📊 Proxy count distribution:")
    print(f"  Min: {min_count}")
    print(f"  Max: {max_count}")
    print(f"  Avg: {avg_count:.2f}")
    
    # Check similarity range
    all_similarities = [sim for proxies in proxy_labels.values() for _, sim in proxies]
    if all_similarities:
        min_sim = min(all_similarities)
        max_sim = max(all_similarities)
        avg_sim = sum(all_similarities) / len(all_similarities)
        
        print(f"\n📊 Similarity distribution:")
        print(f"  Min: {min_sim:.4f}")
        print(f"  Max: {max_sim:.4f}")
        print(f"  Avg: {avg_sim:.4f}")
    
    print(f"\n✅ Test completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Test proxy labels JSON file")
    parser.add_argument("--file", type=str, required=True, help="Path to proxy labels JSON file")
    
    args = parser.parse_args()
    
    test_proxy_labels(args.file)


if __name__ == "__main__":
    main()
