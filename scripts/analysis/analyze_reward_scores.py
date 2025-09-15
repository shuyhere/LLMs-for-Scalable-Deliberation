#!/usr/bin/env python3
"""
Analyze Reward Model Scores and Create Filtered Datasets

This script analyzes the reward model inference results to:
1. Plot score distributions
2. Create filtered datasets based on different filtering strategies

Filtering Strategies:
- 'negative_only': Filter out samples where chosen < rejected (score_difference < 0)
- 'confusing_only': Filter out samples with smallest absolute differences (bottom percentile)
- 'both': Filter out both negative and confusing samples

Example Commands:
# Filter only negative samples (chosen < rejected)
python scripts/analysis/analyze_reward_scores.py \
    --scores_path results/test_reward_model_inference/informativeness_scores.jsonl \
    --output_dir results/score_analysis \
    --filter_strategy negative_only

# Filter only confusing samples (bottom 10% by absolute difference)
python scripts/analysis/analyze_reward_scores.py \
    --scores_path results/test_reward_model_inference/informativeness_scores.jsonl \
    --output_dir results/score_analysis \
    --filter_strategy confusing_only \
    --threshold_percentile 10.0

# Filter both negative and confusing samples
python scripts/analysis/analyze_reward_scores.py \
    --scores_path results/test_reward_model_inference/informativeness_scores.jsonl \
    --output_dir results/score_analysis \
    --filter_strategy both \
    --threshold_percentile 10.0
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_scores_data(scores_path: str) -> List[Dict[str, Any]]:
    """Load reward model scores from JSONL file"""
    logger.info(f"Loading scores from {scores_path}")
    
    data = []
    with open(scores_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(data)} samples")
    return data

def extract_scores(data: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float]]:
    """Extract chosen, rejected, and difference scores"""
    chosen_scores = []
    rejected_scores = []
    differences = []
    
    for sample in data:
        chosen_score = sample["chosen"][-1]["reward_score"]
        rejected_score = sample["rejected"][-1]["reward_score"]
        difference = sample["score_difference"]
        
        chosen_scores.append(chosen_score)
        rejected_scores.append(rejected_score)
        differences.append(difference)
    
    return chosen_scores, rejected_scores, differences

def plot_score_distributions(chosen_scores: List[float], rejected_scores: List[float], 
                           differences: List[float], output_dir: Path):
    """Plot various score distributions"""
    logger.info("Creating score distribution plots...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reward Model Score Distributions', fontsize=16, fontweight='bold')
    
    # 1. Chosen vs Rejected scores scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(chosen_scores, rejected_scores, alpha=0.6, s=20)
    ax1.plot([min(chosen_scores + rejected_scores), max(chosen_scores + rejected_scores)], 
             [min(chosen_scores + rejected_scores), max(chosen_scores + rejected_scores)], 
             'r--', alpha=0.8, label='y=x')
    ax1.set_xlabel('Chosen Score')
    ax1.set_ylabel('Rejected Score')
    ax1.set_title('Chosen vs Rejected Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Score difference distribution
    ax2 = axes[0, 1]
    ax2.hist(differences, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.8, label='Difference = 0')
    ax2.set_xlabel('Score Difference (Chosen - Rejected)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Score Differences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Chosen scores distribution
    ax3 = axes[1, 0]
    ax3.hist(chosen_scores, bins=50, alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Chosen Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Chosen Scores')
    ax3.grid(True, alpha=0.3)
    
    # 4. Rejected scores distribution
    ax4 = axes[1, 1]
    ax4.hist(rejected_scores, bins=50, alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Rejected Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Rejected Scores')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'score_distributions.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Score distributions plot saved to {plot_path}")
    
    plt.close()

def identify_confusing_samples(data: List[Dict[str, Any]], threshold_percentile: float = 10) -> List[Dict[str, Any]]:
    """Identify confusing samples based on smallest absolute differences"""
    logger.info(f"Identifying confusing samples (bottom {threshold_percentile}% by absolute difference)...")
    
    # Calculate absolute differences
    abs_differences = [abs(sample["score_difference"]) for sample in data]
    
    # Find threshold
    threshold = np.percentile(abs_differences, threshold_percentile)
    
    # Filter confusing samples
    confusing_samples = [sample for sample in data if abs(sample["score_difference"]) <= threshold]
    
    logger.info(f"Found {len(confusing_samples)} confusing samples (threshold: {threshold:.4f})")
    
    return confusing_samples

def identify_negative_samples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify samples where chosen score < rejected score (negative differences)"""
    logger.info("Identifying negative samples (chosen < rejected)...")
    
    negative_samples = [sample for sample in data if sample["score_difference"] < 0]
    
    logger.info(f"Found {len(negative_samples)} negative samples")
    
    return negative_samples

def create_filtered_datasets(data: List[Dict[str, Any]], output_dir: Path, 
                           filter_strategy: str = 'negative_only', threshold_percentile: float = 10):
    """Create filtered datasets based on specified filtering strategy
    
    Args:
        data: Input dataset
        output_dir: Output directory for filtered datasets
        filter_strategy: 'negative_only', 'confusing_only', or 'both'
        threshold_percentile: Percentile threshold for confusing samples (only used for confusing_only and both)
    """
    logger.info(f"Creating filtered datasets using strategy: {filter_strategy}")
    
    # Identify different types of samples
    negative_samples = [sample for sample in data if sample["score_difference"] < 0]
    
    if filter_strategy == 'negative_only':
        # Only filter negative samples
        filtered_data = [sample for sample in data if sample["score_difference"] >= 0]
        confusing_samples = []
        
    elif filter_strategy == 'confusing_only':
        # Only filter confusing samples
        confusing_samples = identify_confusing_samples(data, threshold_percentile)
        
        # Get indices of confusing samples to exclude
        exclude_indices = set()
        for confusing_sample in confusing_samples:
            for i, sample in enumerate(data):
                if (sample["chosen"][-1]["reward_score"] == confusing_sample["chosen"][-1]["reward_score"] and
                    sample["rejected"][-1]["reward_score"] == confusing_sample["rejected"][-1]["reward_score"]):
                    exclude_indices.add(i)
                    break
        
        # Create filtered dataset (exclude only confusing samples)
        filtered_data = [sample for i, sample in enumerate(data) if i not in exclude_indices]
        
    elif filter_strategy == 'both':
        # Filter both negative and confusing samples
        confusing_samples = identify_confusing_samples(data, threshold_percentile)
        
        # Get indices of samples to exclude
        exclude_indices = set()
        
        # Add confusing samples indices
        for confusing_sample in confusing_samples:
            for i, sample in enumerate(data):
                if (sample["chosen"][-1]["reward_score"] == confusing_sample["chosen"][-1]["reward_score"] and
                    sample["rejected"][-1]["reward_score"] == confusing_sample["rejected"][-1]["reward_score"]):
                    exclude_indices.add(i)
                    break
        
        # Add negative samples indices
        for negative_sample in negative_samples:
            for i, sample in enumerate(data):
                if (sample["chosen"][-1]["reward_score"] == negative_sample["chosen"][-1]["reward_score"] and
                    sample["rejected"][-1]["reward_score"] == negative_sample["rejected"][-1]["reward_score"]):
                    exclude_indices.add(i)
                    break
        
        # Create filtered dataset (exclude both confusing and negative samples)
        filtered_data = [sample for i, sample in enumerate(data) if i not in exclude_indices]
        
    else:
        raise ValueError(f"Invalid filter_strategy: {filter_strategy}. Must be 'negative_only', 'confusing_only', or 'both'")
    
    # Log statistics
    logger.info(f"Original dataset: {len(data)} samples")
    logger.info(f"Negative samples: {len(negative_samples)} samples")
    if filter_strategy in ['confusing_only', 'both']:
        logger.info(f"Confusing samples: {len(confusing_samples)} samples")
    logger.info(f"Filtered dataset: {len(filtered_data)} samples")
    
    # Save datasets
    # Save negative samples (if any)
    if negative_samples:
        negative_path = output_dir / 'negative_samples.jsonl'
        with open(negative_path, 'w', encoding='utf-8') as f:
            for sample in negative_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"Negative samples saved to {negative_path}")
    
    # Save confusing samples (if any)
    if filter_strategy in ['confusing_only', 'both'] and confusing_samples:
        confusing_path = output_dir / 'confusing_samples.jsonl'
        with open(confusing_path, 'w', encoding='utf-8') as f:
            for sample in confusing_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"Confusing samples saved to {confusing_path}")
    
    # Save filtered dataset
    filtered_path = output_dir / 'filtered_dataset.jsonl'
    with open(filtered_path, 'w', encoding='utf-8') as f:
        for sample in filtered_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    logger.info(f"Filtered dataset saved to {filtered_path}")
    
    return filtered_data

def create_summary_statistics(data: List[Dict[str, Any]], output_dir: Path):
    """Create summary statistics"""
    logger.info("Creating summary statistics...")
    
    chosen_scores, rejected_scores, differences = extract_scores(data)
    
    stats = {
        "total_samples": len(data),
        "chosen_scores": {
            "mean": np.mean(chosen_scores),
            "std": np.std(chosen_scores),
            "min": np.min(chosen_scores),
            "max": np.max(chosen_scores),
            "median": np.median(chosen_scores)
        },
        "rejected_scores": {
            "mean": np.mean(rejected_scores),
            "std": np.std(rejected_scores),
            "min": np.min(rejected_scores),
            "max": np.max(rejected_scores),
            "median": np.median(rejected_scores)
        },
        "differences": {
            "mean": np.mean(differences),
            "std": np.std(differences),
            "min": np.min(differences),
            "max": np.max(differences),
            "median": np.median(differences)
        },
        "preference_accuracy": sum(1 for d in differences if d > 0) / len(differences)
    }
    
    # Save statistics
    stats_path = output_dir / 'summary_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    logger.info("Summary Statistics:")
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Preference accuracy: {stats['preference_accuracy']:.4f}")
    logger.info(f"Chosen scores - Mean: {stats['chosen_scores']['mean']:.4f}, Std: {stats['chosen_scores']['std']:.4f}")
    logger.info(f"Rejected scores - Mean: {stats['rejected_scores']['mean']:.4f}, Std: {stats['rejected_scores']['std']:.4f}")
    logger.info(f"Differences - Mean: {stats['differences']['mean']:.4f}, Std: {stats['differences']['std']:.4f}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Analyze Reward Model Scores and Create Filtered Datasets')
    parser.add_argument('--scores_path', type=str, required=True,
                       help='Path to the scores JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--filter_strategy', type=str, default='negative_only',
                       choices=['negative_only', 'confusing_only', 'both'],
                       help='Filtering strategy: negative_only (default), confusing_only, or both')
    parser.add_argument('--threshold_percentile', type=float, default=10.0,
                       help='Percentile threshold for identifying confusing samples (default: 10, only used for confusing_only and both)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_scores_data(args.scores_path)
    
    # Extract scores
    chosen_scores, rejected_scores, differences = extract_scores(data)
    
    # Create plots
    plot_score_distributions(chosen_scores, rejected_scores, differences, output_dir)
    
    # Create summary statistics
    create_summary_statistics(data, output_dir)
    
    # Create filtered datasets based on specified strategy
    filtered_data = create_filtered_datasets(data, output_dir, 
                                           filter_strategy=args.filter_strategy, 
                                           threshold_percentile=args.threshold_percentile)
    
    logger.info("Analysis completed successfully!")

if __name__ == "__main__":
    main()
