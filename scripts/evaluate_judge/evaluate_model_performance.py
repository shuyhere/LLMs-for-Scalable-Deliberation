#!/usr/bin/env python3
"""
Evaluate trained regression models on test sets and calculate correlation and accuracy metrics.

This script evaluates DeBERTa regression models on specified test sets and computes:
- Pearson and Spearman correlations between predicted and ground truth scores
- Various accuracy metrics (MAE, RMSE, R²)
- Per-dimension statistics
- Overall performance summary

Usage:
    python scripts/evaluate_judge/evaluate_model_performance.py \
        --model /path/to/trained/model \
        --test-data /path/to/test.jsonl \
        --output /path/to/results.json

    # With custom thresholds for accuracy calculation
    python scripts/evaluate_judge/evaluate_model_performance.py \
        --model /path/to/trained/model \
        --test-data /path/to/test.jsonl \
        --accuracy-threshold 0.5 \
        --output /path/to/results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from llm_evaluation.evaluator import DebertaEvaluator


# Target dimensions (same as training)
TARGET_KEYS = [
    "perspective_representation",
    "informativeness", 
    "neutrality_balance",
    "policy_approval"
]

# Dimension display names for reporting
DIMENSION_NAMES = {
    "perspective_representation": "Representiveness",
    "informativeness": "Informativeness",
    "neutrality_balance": "Neutrality",
    "policy_approval": "Policy Approval"
}


def load_test_data(test_path: str) -> List[Dict[str, Any]]:
    """
    Load test data from JSONL file.
    
    Args:
        test_path: Path to test JSONL file
        
    Returns:
        List of test data items
    """
    data = []
    with Path(test_path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(data)} test items from {test_path}")
    return data


def normalize_score(score: float, min_val: float = -1.0, max_val: float = 7.0) -> float:
    """Normalize score from [min_val, max_val] to [0, 1] (same as training)"""
    return (score - min_val) / (max_val - min_val)


def extract_ground_truth_scores(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Extract ground truth scores from test data item and normalize them to [0, 1].
    
    Args:
        item: Test data item
        
    Returns:
        Dictionary of normalized ground truth scores or None if not available
    """
    # Try different possible score field names
    score_fields = ["scores", "rating_scores", "labels", "targets"]
    
    raw_scores = {}
    for field in score_fields:
        if field in item and isinstance(item[field], dict):
            for key in TARGET_KEYS:
                if key in item[field]:
                    try:
                        raw_scores[key] = float(item[field][key])
                    except (ValueError, TypeError):
                        continue
            
            if len(raw_scores) == len(TARGET_KEYS):
                break
    
    # Try direct fields if no scores found
    if not raw_scores:
        for key in TARGET_KEYS:
            if key in item:
                try:
                    raw_scores[key] = float(item[key])
                except (ValueError, TypeError):
                    continue
    
    if len(raw_scores) != len(TARGET_KEYS):
        return None
    
    # Normalize scores to [0, 1] range (same as training)
    normalized_scores = {}
    for key, raw_score in raw_scores.items():
        normalized_scores[key] = normalize_score(raw_score)
    
    return normalized_scores


def calculate_correlations(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate Pearson and Spearman correlations.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary with correlation metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(mask) < 2:
        return {"pearson": 0.0, "spearman": 0.0, "pearson_p": 1.0, "spearman_p": 1.0}
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(y_true_clean, y_pred_clean)
    spearman_r, spearman_p = spearmanr(y_true_clean, y_pred_clean)
    
    # Handle NaN correlations (e.g., constant values)
    if np.isnan(pearson_r):
        pearson_r = 0.0
    if np.isnan(spearman_r):
        spearman_r = 0.0
    
    return {
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
        "pearson_p": float(pearson_p),
        "spearman_p": float(spearman_p)
    }


def calculate_accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate various accuracy metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        threshold: Threshold for binary accuracy calculation
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(mask) == 0:
        return {
            "mae": float('inf'),
            "rmse": float('inf'),
            "r2": -float('inf'),
            "binary_accuracy": 0.0,
            "within_threshold": 0.0
        }
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Basic regression metrics
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Binary accuracy (predictions within threshold of true values)
    within_threshold = np.mean(np.abs(y_true_clean - y_pred_clean) <= threshold)
    
    # Simple binary accuracy (round to nearest integer and compare)
    y_true_binary = np.round(y_true_clean)
    y_pred_binary = np.round(y_pred_clean)
    binary_accuracy = np.mean(y_true_binary == y_pred_binary)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "binary_accuracy": float(binary_accuracy),
        "within_threshold": float(within_threshold)
    }


def evaluate_model_on_test_set(model_path: str, test_data: List[Dict[str, Any]], 
                              max_length: int = 2048) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate model on test set and return detailed results.
    
    Args:
        model_path: Path to trained model
        test_data: List of test data items
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (detailed_results, summary_statistics)
    """
    # Initialize evaluator
    print(f"Loading model from {model_path}")
    evaluator = DebertaEvaluator(model_path)
    
    # Prepare data for evaluation
    eval_data = []
    ground_truth_scores = []
    
    for item in test_data:
        # Extract required fields
        question = item.get("question", "")
        comment = item.get("comment", "")
        summary = item.get("summary", "")
        
        if not all([question, comment, summary]):
            print(f"Warning: Skipping item with missing fields")
            continue
        
        eval_data.append({
            "question": question,
            "comment": comment,
            "summary": summary
        })
        
        # Extract ground truth scores
        gt_scores = extract_ground_truth_scores(item)
        ground_truth_scores.append(gt_scores)
    
    print(f"Evaluating {len(eval_data)} items...")
    
    # Get model predictions
    predictions = evaluator.evaluate_batch(eval_data, max_length=max_length)
    
    # Combine predictions with ground truth
    detailed_results = []
    valid_predictions = []
    valid_ground_truth = []
    
    for i, (pred, gt_scores) in enumerate(zip(predictions, ground_truth_scores)):
        if pred.get("status") != "success" or gt_scores is None:
            detailed_results.append({
                "item_index": i,
                "status": "error",
                "error": "Failed prediction or missing ground truth",
                "predictions": None,
                "ground_truth": gt_scores
            })
            continue
        
        pred_scores = pred.get("predictions", {})
        
        # Collect valid scores for overall metrics
        pred_values = []
        gt_values = []
        
        for key in TARGET_KEYS:
            if key in pred_scores and key in gt_scores:
                pred_val = pred_scores[key]
                gt_val = gt_scores[key]
                
                if pred_val is not None and gt_val is not None:
                    pred_values.append(pred_val)
                    gt_values.append(gt_val)
        
        if len(pred_values) == len(TARGET_KEYS):
            valid_predictions.append(pred_values)
            valid_ground_truth.append(gt_values)
        
        detailed_results.append({
            "item_index": i,
            "status": "success",
            "question": pred.get("question", ""),
            "comment": pred.get("comment", ""),
            "summary": pred.get("summary", ""),
            "predictions": pred_scores,
            "ground_truth": gt_scores
        })
    
    # Calculate summary statistics
    summary_stats = {
        "total_items": len(test_data),
        "successful_evaluations": len(valid_predictions),
        "failed_evaluations": len(test_data) - len(valid_predictions)
    }
    
    if not valid_predictions:
        print("Warning: No valid predictions found")
        return detailed_results, summary_stats
    
    # Convert to numpy arrays
    y_pred = np.array(valid_predictions)  # (n_samples, n_dims)
    y_true = np.array(valid_ground_truth)  # (n_samples, n_dims)
    
    # Overall metrics
    overall_correlations = {}
    overall_accuracy = {}
    
    for i, key in enumerate(TARGET_KEYS):
        dim_name = DIMENSION_NAMES.get(key, key)
        
        # Per-dimension correlations
        corr_metrics = calculate_correlations(y_true[:, i], y_pred[:, i])
        overall_correlations[key] = {
            "dimension": dim_name,
            **corr_metrics
        }
        
        # Per-dimension accuracy
        acc_metrics = calculate_accuracy_metrics(y_true[:, i], y_pred[:, i])
        overall_accuracy[key] = {
            "dimension": dim_name,
            **acc_metrics
        }
    
    # Average correlations across dimensions
    avg_pearson = np.mean([overall_correlations[key]["pearson"] for key in TARGET_KEYS])
    avg_spearman = np.mean([overall_correlations[key]["spearman"] for key in TARGET_KEYS])
    
    # Average accuracy metrics
    avg_mae = np.mean([overall_accuracy[key]["mae"] for key in TARGET_KEYS])
    avg_rmse = np.mean([overall_accuracy[key]["rmse"] for key in TARGET_KEYS])
    avg_r2 = np.mean([overall_accuracy[key]["r2"] for key in TARGET_KEYS])
    avg_binary_acc = np.mean([overall_accuracy[key]["binary_accuracy"] for key in TARGET_KEYS])
    
    summary_stats.update({
        "overall_correlations": {
            "average_pearson": float(avg_pearson),
            "average_spearman": float(avg_spearman)
        },
        "overall_accuracy": {
            "average_mae": float(avg_mae),
            "average_rmse": float(avg_rmse),
            "average_r2": float(avg_r2),
            "average_binary_accuracy": float(avg_binary_acc)
        },
        "per_dimension_correlations": overall_correlations,
        "per_dimension_accuracy": overall_accuracy
    })
    
    return detailed_results, summary_stats


def print_results(summary_stats: Dict[str, Any]):
    """
    Print evaluation results in a formatted way.
    
    Args:
        summary_stats: Summary statistics from evaluation
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Overall statistics
    print(f"\nDataset Statistics:")
    print(f"  Total items: {summary_stats['total_items']}")
    print(f"  Successful evaluations: {summary_stats['successful_evaluations']}")
    print(f"  Failed evaluations: {summary_stats['failed_evaluations']}")
    
    # Overall correlations
    overall_corr = summary_stats.get("overall_correlations", {})
    print(f"\nOverall Correlations:")
    print(f"  Average Pearson: {overall_corr.get('average_pearson', 0.0):.4f}")
    print(f"  Average Spearman: {overall_corr.get('average_spearman', 0.0):.4f}")
    
    # Overall accuracy
    overall_acc = summary_stats.get("overall_accuracy", {})
    print(f"\nOverall Accuracy:")
    print(f"  Average MAE: {overall_acc.get('average_mae', 0.0):.4f}")
    print(f"  Average RMSE: {overall_acc.get('average_rmse', 0.0):.4f}")
    print(f"  Average R²: {overall_acc.get('average_r2', 0.0):.4f}")
    print(f"  Average Binary Accuracy: {overall_acc.get('average_binary_accuracy', 0.0):.4f}")
    
    # Per-dimension results
    print(f"\nPer-Dimension Results:")
    print("-" * 80)
    print(f"{'Dimension':<20} {'Pearson':<10} {'Spearman':<10} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
    print("-" * 80)
    
    per_dim_corr = summary_stats.get("per_dimension_correlations", {})
    per_dim_acc = summary_stats.get("per_dimension_accuracy", {})
    
    for key in TARGET_KEYS:
        dim_name = DIMENSION_NAMES.get(key, key)
        corr_data = per_dim_corr.get(key, {})
        acc_data = per_dim_acc.get(key, {})
        
        pearson = corr_data.get("pearson", 0.0)
        spearman = corr_data.get("spearman", 0.0)
        mae = acc_data.get("mae", 0.0)
        rmse = acc_data.get("rmse", 0.0)
        r2 = acc_data.get("r2", 0.0)
        
        print(f"{dim_name:<20} {pearson:<10.4f} {spearman:<10.4f} {mae:<8.4f} {rmse:<8.4f} {r2:<8.4f}")
    
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained regression models on test sets")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--test-data", type=str, required=True,
                       help="Path to test JSONL file")
    parser.add_argument("--output", type=str,
                       help="Path to save detailed results (JSON format)")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--accuracy-threshold", type=float, default=0.5,
                       help="Threshold for accuracy calculation")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        return 1
    
    # Check if test data exists
    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"Error: Test data path {test_path} does not exist")
        return 1
    
    try:
        # Load test data
        test_data = load_test_data(args.test_data)
        if not test_data:
            print("Error: No valid test data found")
            return 1
        
        # Evaluate model
        detailed_results, summary_stats = evaluate_model_on_test_set(
            args.model, test_data, args.max_length
        )
        
        # Print results
        if not args.quiet:
            print_results(summary_stats)
        
        # Save results if output path specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results = {
                "summary_statistics": summary_stats,
                "detailed_results": detailed_results,
                "evaluation_config": {
                    "model_path": str(model_path),
                    "test_data_path": str(test_path),
                    "max_length": args.max_length,
                    "accuracy_threshold": args.accuracy_threshold
                }
            }
            
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\nDetailed results saved to {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
