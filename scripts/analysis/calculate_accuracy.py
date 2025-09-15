#!/usr/bin/env python3
"""
Calculate Accuracy for Filtered Dataset

This script calculates the accuracy of the reward model on the filtered dataset.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(data)} samples")
    return data

def calculate_accuracy(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate various accuracy metrics"""
    logger.info("Calculating accuracy metrics...")
    
    total_samples = len(data)
    correct_predictions = 0
    incorrect_predictions = 0
    
    # Score statistics
    chosen_scores = []
    rejected_scores = []
    differences = []
    
    for sample in data:
        # Check if preference is correct
        if sample.get("preference_correct", False):
            correct_predictions += 1
        else:
            incorrect_predictions += 1
        
        # Collect scores
        chosen_score = sample["chosen"][-1]["reward_score"]
        rejected_score = sample["rejected"][-1]["reward_score"]
        difference = sample["score_difference"]
        
        chosen_scores.append(chosen_score)
        rejected_scores.append(rejected_score)
        differences.append(difference)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    # Calculate additional metrics
    avg_chosen_score = sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0
    avg_rejected_score = sum(rejected_scores) / len(rejected_scores) if rejected_scores else 0
    avg_difference = sum(differences) / len(differences) if differences else 0
    
    # Calculate score ranges
    min_chosen = min(chosen_scores) if chosen_scores else 0
    max_chosen = max(chosen_scores) if chosen_scores else 0
    min_rejected = min(rejected_scores) if rejected_scores else 0
    max_rejected = max(rejected_scores) if rejected_scores else 0
    min_difference = min(differences) if differences else 0
    max_difference = max(differences) if differences else 0
    
    # Calculate standard deviations
    import numpy as np
    std_chosen = np.std(chosen_scores) if chosen_scores else 0
    std_rejected = np.std(rejected_scores) if rejected_scores else 0
    std_difference = np.std(differences) if differences else 0
    
    results = {
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "accuracy": accuracy,
        "score_statistics": {
            "chosen": {
                "mean": avg_chosen_score,
                "std": std_chosen,
                "min": min_chosen,
                "max": max_chosen
            },
            "rejected": {
                "mean": avg_rejected_score,
                "std": std_rejected,
                "min": min_rejected,
                "max": max_rejected
            },
            "difference": {
                "mean": avg_difference,
                "std": std_difference,
                "min": min_difference,
                "max": max_difference
            }
        }
    }
    
    return results

def print_results(results: Dict[str, Any]):
    """Print results in a formatted way"""
    logger.info("=" * 60)
    logger.info("ACCURACY ANALYSIS RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Total samples: {results['total_samples']}")
    logger.info(f"Correct predictions: {results['correct_predictions']}")
    logger.info(f"Incorrect predictions: {results['incorrect_predictions']}")
    logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    logger.info("\n" + "-" * 40)
    logger.info("SCORE STATISTICS")
    logger.info("-" * 40)
    
    chosen_stats = results['score_statistics']['chosen']
    rejected_stats = results['score_statistics']['rejected']
    diff_stats = results['score_statistics']['difference']
    
    logger.info(f"Chosen scores:")
    logger.info(f"  Mean: {chosen_stats['mean']:.4f}")
    logger.info(f"  Std:  {chosen_stats['std']:.4f}")
    logger.info(f"  Range: [{chosen_stats['min']:.4f}, {chosen_stats['max']:.4f}]")
    
    logger.info(f"Rejected scores:")
    logger.info(f"  Mean: {rejected_stats['mean']:.4f}")
    logger.info(f"  Std:  {rejected_stats['std']:.4f}")
    logger.info(f"  Range: [{rejected_stats['min']:.4f}, {rejected_stats['max']:.4f}]")
    
    logger.info(f"Score differences (chosen - rejected):")
    logger.info(f"  Mean: {diff_stats['mean']:.4f}")
    logger.info(f"  Std:  {diff_stats['std']:.4f}")
    logger.info(f"  Range: [{diff_stats['min']:.4f}, {diff_stats['max']:.4f}]")
    
    logger.info("=" * 60)

def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file"""
    logger.info(f"Saving results to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Calculate Accuracy for Filtered Dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the filtered dataset JSONL file')
    parser.add_argument('--output_path', type=str, default='accuracy_results.json',
                       help='Output path for results JSON file')
    
    args = parser.parse_args()
    
    # Load dataset
    data = load_dataset(args.dataset_path)
    
    # Calculate accuracy
    results = calculate_accuracy(data)
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, args.output_path)
    
    logger.info("Accuracy calculation completed!")

if __name__ == "__main__":
    main()
