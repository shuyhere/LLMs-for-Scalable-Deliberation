#!/usr/bin/env python3
"""
Check the label distribution in the training data
"""

import json
import numpy as np
from pathlib import Path

def check_label_distribution(data_file):
    """Check label distribution in the training data"""
    
    # Target dimensions
    TARGET_KEYS = [
        "perspective_representation",
        "informativeness",
        "neutrality_balance",
        "policy_approval",
    ]
    
    # Load data
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from {data_file}")
    
    # Collect all scores
    all_scores = {key: [] for key in TARGET_KEYS}
    
    for record in data:
        scores = record.get('scores', {})
        for key in TARGET_KEYS:
            if key in scores:
                all_scores[key].append(scores[key])
    
    # Analyze distribution
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION ANALYSIS (Original Scale: -1 to 7)")
    print("="*60)
    
    for key in TARGET_KEYS:
        if all_scores[key]:
            values = np.array(all_scores[key])
            print(f"\n{key}:")
            print(f"  Count: {len(values)}")
            print(f"  Mean:  {values.mean():.3f}")
            print(f"  Std:   {values.std():.3f}")
            print(f"  Min:   {values.min():.3f}")
            print(f"  Max:   {values.max():.3f}")
            print(f"  25%:   {np.percentile(values, 25):.3f}")
            print(f"  50%:   {np.percentile(values, 50):.3f}")
            print(f"  75%:   {np.percentile(values, 75):.3f}")
            
            # Check normalized values
            normalized = (values - (-1)) / (7 - (-1))  # Normalize to [0, 1]
            print(f"  Normalized Mean: {normalized.mean():.3f}")
            print(f"  Normalized Std:  {normalized.std():.3f}")
    
    # Check correlation between dimensions
    print("\n" + "="*60)
    print("CORRELATION BETWEEN DIMENSIONS")
    print("="*60)
    
    from scipy.stats import pearsonr
    
    for i, key1 in enumerate(TARGET_KEYS):
        for j, key2 in enumerate(TARGET_KEYS):
            if i < j and all_scores[key1] and all_scores[key2]:
                corr, _ = pearsonr(all_scores[key1], all_scores[key2])
                print(f"{key1} vs {key2}: {corr:.3f}")


if __name__ == "__main__":
    # Check training data
    train_file = "/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/split_data/train.jsonl"
    
    if Path(train_file).exists():
        check_label_distribution(train_file)
    else:
        print(f"Training file not found: {train_file}")
        
    # Also check full dataset
    print("\n" + "="*80)
    print("CHECKING FULL DATASET")
    print("="*80)
    
    full_file = "/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings.jsonl"
    
    if Path(full_file).exists():
        check_label_distribution(full_file)
    else:
        print(f"Full dataset not found: {full_file}")