#!/usr/bin/env python3
"""
Diagnostic script to help identify issues with model evaluation.

This script analyzes the evaluation results and ground truth data to identify
potential issues with scale mismatch, data format, or model performance.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from llm_evaluation.evaluator import DebertaEvaluator


TARGET_KEYS = [
    "perspective_representation",
    "informativeness", 
    "neutrality_balance",
    "policy_approval"
]


def normalize_score(score: float, min_val: float = -1.0, max_val: float = 7.0) -> float:
    """Normalize score from [min_val, max_val] to [0, 1] (same as training)"""
    return (score - min_val) / (max_val - min_val)


def analyze_ground_truth_scale(test_data: list):
    """Analyze the scale and distribution of ground truth scores."""
    print("=== GROUND TRUTH ANALYSIS ===")
    
    all_scores = {key: [] for key in TARGET_KEYS}
    
    for item in test_data:
        # Try different possible score field names
        score_fields = ["scores", "rating_scores", "labels", "targets"]
        
        scores = None
        for field in score_fields:
            if field in item and isinstance(item[field], dict):
                scores = item[field]
                break
        
        if scores is None:
            # Try direct fields
            scores = {key: item.get(key) for key in TARGET_KEYS}
        
        for key in TARGET_KEYS:
            if key in scores and scores[key] is not None:
                try:
                    raw_score = float(scores[key])
                    normalized_score = normalize_score(raw_score)
                    all_scores[key].append(normalized_score)
                except (ValueError, TypeError):
                    continue
    
    for key in TARGET_KEYS:
        scores = all_scores[key]
        if scores:
            print(f"\n{key}:")
            print(f"  Count: {len(scores)}")
            print(f"  Range: [{min(scores):.3f}, {max(scores):.3f}]")
            print(f"  Mean: {np.mean(scores):.3f}")
            print(f"  Std: {np.std(scores):.3f}")
            print(f"  Unique values: {len(set(scores))}")
            print(f"  Sample values: {scores[:5]}")
        else:
            print(f"\n{key}: No valid scores found")


def analyze_predictions_scale(model_path: str, test_data: list, max_length: int = 2048):
    """Analyze the scale and distribution of model predictions."""
    print("\n=== MODEL PREDICTIONS ANALYSIS ===")
    
    # Initialize evaluator
    evaluator = DebertaEvaluator(model_path)
    
    # Prepare data for evaluation
    eval_data = []
    for item in test_data:
        question = item.get("question", "")
        comment = item.get("comment", "")
        summary = item.get("summary", "")
        
        if all([question, comment, summary]):
            eval_data.append({
                "question": question,
                "comment": comment,
                "summary": summary
            })
    
    # Get predictions
    predictions = evaluator.evaluate_batch(eval_data, max_length=max_length)
    
    all_predictions = {key: [] for key in TARGET_KEYS}
    
    for pred in predictions:
        if pred.get("status") == "success":
            pred_scores = pred.get("predictions", {})
            for key in TARGET_KEYS:
                if key in pred_scores and pred_scores[key] is not None:
                    all_predictions[key].append(pred_scores[key])
    
    for key in TARGET_KEYS:
        preds = all_predictions[key]
        if preds:
            print(f"\n{key} predictions:")
            print(f"  Count: {len(preds)}")
            print(f"  Range: [{min(preds):.3f}, {max(preds):.3f}]")
            print(f"  Mean: {np.mean(preds):.3f}")
            print(f"  Std: {np.std(preds):.3f}")
            print(f"  Unique values: {len(set(preds))}")
            print(f"  Sample values: {preds[:5]}")
        else:
            print(f"\n{key} predictions: No valid predictions found")


def compare_scales(model_path: str, test_data: list, max_length: int = 2048):
    """Compare ground truth and prediction scales side by side."""
    print("\n=== SCALE COMPARISON ===")
    
    # Get ground truth scores
    gt_scores = {key: [] for key in TARGET_KEYS}
    for item in test_data:
        score_fields = ["scores", "rating_scores", "labels", "targets"]
        scores = None
        for field in score_fields:
            if field in item and isinstance(item[field], dict):
                scores = item[field]
                break
        if scores is None:
            scores = {key: item.get(key) for key in TARGET_KEYS}
        
        for key in TARGET_KEYS:
            if key in scores and scores[key] is not None:
                try:
                    raw_score = float(scores[key])
                    normalized_score = normalize_score(raw_score)
                    gt_scores[key].append(normalized_score)
                except (ValueError, TypeError):
                    continue
    
    # Get predictions
    evaluator = DebertaEvaluator(model_path)
    eval_data = []
    for item in test_data:
        question = item.get("question", "")
        comment = item.get("comment", "")
        summary = item.get("summary", "")
        
        if all([question, comment, summary]):
            eval_data.append({
                "question": question,
                "comment": comment,
                "summary": summary
            })
    
    predictions = evaluator.evaluate_batch(eval_data, max_length=max_length)
    
    pred_scores = {key: [] for key in TARGET_KEYS}
    for pred in predictions:
        if pred.get("status") == "success":
            pred_data = pred.get("predictions", {})
            for key in TARGET_KEYS:
                if key in pred_data and pred_data[key] is not None:
                    pred_scores[key].append(pred_data[key])
    
    # Compare scales
    print(f"{'Dimension':<25} {'GT Range':<15} {'Pred Range':<15} {'GT Mean':<10} {'Pred Mean':<10}")
    print("-" * 80)
    
    for key in TARGET_KEYS:
        gt = gt_scores[key]
        pred = pred_scores[key]
        
        if gt and pred:
            gt_range = f"[{min(gt):.2f}, {max(gt):.2f}]"
            pred_range = f"[{min(pred):.2f}, {max(pred):.2f}]"
            gt_mean = f"{np.mean(gt):.3f}"
            pred_mean = f"{np.mean(pred):.3f}"
            
            print(f"{key:<25} {gt_range:<15} {pred_range:<15} {gt_mean:<10} {pred_mean:<10}")
        else:
            print(f"{key:<25} {'No data':<15} {'No data':<15} {'No data':<10} {'No data':<10}")


def check_data_format(test_data: list):
    """Check the format of test data to identify potential issues."""
    print("\n=== DATA FORMAT CHECK ===")
    
    print(f"Total items: {len(test_data)}")
    
    # Check first few items
    for i, item in enumerate(test_data[:3]):
        print(f"\nItem {i+1}:")
        print(f"  Keys: {list(item.keys())}")
        
        # Check for score fields
        score_fields = ["scores", "rating_scores", "labels", "targets"]
        for field in score_fields:
            if field in item:
                print(f"  {field}: {type(item[field])} - {item[field]}")
        
        # Check direct dimension fields
        for key in TARGET_KEYS:
            if key in item:
                print(f"  {key}: {type(item[key])} - {item[key]}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose evaluation issues")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--test-data", type=str, required=True,
                       help="Path to test JSONL file")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Load test data
    test_data = []
    with Path(args.test_data).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    test_data.append(item)
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(test_data)} test items")
    
    # Run diagnostics
    check_data_format(test_data)
    analyze_ground_truth_scale(test_data)
    analyze_predictions_scale(args.model, test_data, args.max_length)
    compare_scales(args.model, test_data, args.max_length)
    
    print("\n=== DIAGNOSTIC SUMMARY ===")
    print("Common issues that cause negative R²:")
    print("1. Scale mismatch: GT and predictions in different ranges")
    print("2. Data format: GT scores not in expected format")
    print("3. Model issues: Model not properly trained or loaded")
    print("4. Normalization: GT scores not normalized like training data")


if __name__ == "__main__":
    main()
