#!/usr/bin/env python3
"""
Batch evaluation script for DeBERTa regression model on summary-comment pairs.

This script evaluates how well summaries represent individual comments across different
topics and sample sizes, using the trained DeBERTa regression model.

For each topic, it evaluates 3 summary samples against all comments in that topic,
computing mean and variance for each of the 4 dimensions:
- perspective_representation
- informativeness
- neutrality_balance
- policy_approval

Usage:
    python batch_regression_evaluate.py --model-path /path/to/trained/model \
                                        --summary-dir /path/to/summaries \
                                        --comments-dir /path/to/comments \
                                        --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import csv
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.llm_evaluation.evaluator import DebertaEvaluator


def load_summary_file(summary_path: Path) -> Dict[str, Any]:
    """Load a single summary JSON file."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_comments_file(comments_path: Path) -> Dict[str, Any]:
    """Load comments dataset for a specific topic."""
    with open(comments_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_summary_on_comments(
    evaluator: DebertaEvaluator,
    question: str,
    summary: str,
    comments: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate a single summary against all comments for a topic.
    
    Returns:
        (stats, per_comment_records)
        - stats: Dictionary with mean and std for each dimension
        - per_comment_records: List of {comment_index, predictions...}
    """
    # Collect predictions for all comments
    all_predictions = {
        "perspective_representation": [],
        "informativeness": [],
        "neutrality_balance": [],
        "policy_approval": []
    }
    per_comment_records: List[Dict[str, Any]] = []
    
    # Evaluate summary against each comment
    for comment_data in comments:
        comment_text = comment_data.get("comment", "")
        comment_index = comment_data.get("index")
        is_minority = comment_data.get("is_minority")
        
        try:
            result = evaluator.evaluate_single(
                question=question,
                comment=comment_text,
                summary=summary
            )
            
            # Collect predictions
            for dim in all_predictions.keys():
                if dim in result["predictions"]:
                    all_predictions[dim].append(result["predictions"][dim])
            per_comment_records.append({
                "comment_index": comment_index,
                "is_minority": is_minority,
                **{k: result["predictions"].get(k) for k in [
                    "perspective_representation",
                    "informativeness",
                    "neutrality_balance",
                    "policy_approval"
                ]}
            })
                    
        except Exception as e:
            print(f"  Warning: Failed to evaluate comment: {e}")
            continue
    
    # Calculate statistics
    stats = {}
    for dim, values in all_predictions.items():
        if values:
            values_array = np.array(values)
            stats[dim] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "n_samples": len(values)
            }
        else:
            stats[dim] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "n_samples": 0
            }
    
    return stats, per_comment_records


def process_topic(
    evaluator: DebertaEvaluator,
    topic_name: str,
    model_name: str,
    sample_size: int,
    summary_base_dir: Path,
    comments_dir: Path,
    n_summaries: int = 3,
    per_comment_csv: Path = None
) -> Dict[str, Any]:
    """
    Process all summaries for a specific topic.
    
    Args:
        evaluator: DeBERTa evaluator instance
        topic_name: Name of the topic
        model_name: Summary model name (e.g., 'deepseek-chat')
        sample_size: Sample size (e.g., 10, 25, 50)
        summary_base_dir: Base directory for summaries
        comments_dir: Directory containing comment datasets
        n_summaries: Number of summary samples to evaluate (default 3)
        per_comment_csv: Optional path to append per-comment predictions CSV
    
    Returns:
        Dictionary with evaluation results for this topic
    """
    results = {
        "topic": topic_name,
        "model": model_name,
        "sample_size": sample_size,
        "summaries": []
    }
    
    # Load comments for this topic
    comments_file = comments_dir / f"{topic_name}.json"
    if not comments_file.exists():
        print(f"  Warning: Comments file not found: {comments_file}")
        return results
    
    comments_data = load_comments_file(comments_file)
    question = comments_data.get("question", "")
    all_comments = comments_data.get("comments", [])
    
    # Only use the first 'sample_size' comments (matching what was used in summary generation)
    comments = all_comments[:sample_size] if sample_size < len(all_comments) else all_comments
    
    print(f"  Loaded {len(all_comments)} total comments, using first {len(comments)} for evaluation (sample_size={sample_size})")
    
    # Prepare CSV writer if needed
    csv_writer = None
    csv_file_handle = None
    if per_comment_csv is not None:
        csv_headers = [
            "topic",
            "model",
            "sample_size",
            "summary_index",
            "summary_file",
            "comment_index",
            "is_minority",
            "perspective_representation",
            "informativeness",
            "neutrality_balance",
            "policy_approval",
        ]
        per_comment_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = per_comment_csv.exists()

        # Auto-upgrade existing CSV missing 'is_minority'
        if file_exists:
            try:
                with per_comment_csv.open("r", encoding="utf-8") as fh:
                    first_line = fh.readline().strip()
                has_is_minority = "is_minority" in [h.strip() for h in first_line.split(",")]
                if not has_is_minority and first_line:
                    tmp_path = per_comment_csv.with_suffix(per_comment_csv.suffix + ".upgraded")
                    with per_comment_csv.open("r", encoding="utf-8") as rf, tmp_path.open("w", newline="", encoding="utf-8") as wf:
                        reader = csv.DictReader(rf)
                        writer = csv.DictWriter(wf, fieldnames=csv_headers)
                        writer.writeheader()
                        for row in reader:
                            # Preserve known fields; fill missing with None
                            writer.writerow({
                                "topic": row.get("topic"),
                                "model": row.get("model"),
                                "sample_size": row.get("sample_size"),
                                "summary_index": row.get("summary_index"),
                                "summary_file": row.get("summary_file"),
                                "comment_index": row.get("comment_index"),
                                "is_minority": None,
                                "perspective_representation": row.get("perspective_representation"),
                                "informativeness": row.get("informativeness"),
                                "neutrality_balance": row.get("neutrality_balance"),
                                "policy_approval": row.get("policy_approval"),
                            })
                    os.replace(tmp_path, per_comment_csv)
            except Exception:
                # If upgrade fails, fall back to creating a new file next
                file_exists = False

        csv_file_handle = per_comment_csv.open("a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=csv_headers)
        if not file_exists:
            csv_writer.writeheader()
    
    # Process each summary sample
    for i in range(1, n_summaries + 1):
        summary_file = summary_base_dir / model_name / str(sample_size) / f"{topic_name}_summary_{i}.json"
        
        if not summary_file.exists():
            print(f"  Warning: Summary file not found: {summary_file}")
            continue
        
        # Load summary
        summary_data = load_summary_file(summary_file)
        
        # Get the comment indices that were used for this summary
        comment_indices_used = summary_data.get("comment_indices", [])
        
        # If comment indices are specified, use only those comments
        if comment_indices_used:
            comments_for_eval = [all_comments[idx] for idx in comment_indices_used if idx < len(all_comments)]
            print(f"    Using {len(comments_for_eval)} comments based on indices from summary file: {comment_indices_used[:5]}...")
        else:
            # Fallback to using first sample_size comments
            comments_for_eval = comments
            print(f"    Using first {len(comments_for_eval)} comments (no indices in summary file)")
        
        # Extract the main points summary (or use another type if preferred)
        summary_text = summary_data.get("summaries", {}).get("main_points", "")
        
        if not summary_text or summary_text == "Main points summary generation disabled in configuration":
            # Try other summary types
            summary_text = summary_data.get("summaries", {}).get("topic_modeling", "")
            if not summary_text or "disabled" in summary_text.lower():
                summary_text = summary_data.get("summaries", {}).get("custom_analysis", "")
        
        if not summary_text or "disabled" in summary_text.lower():
            print(f"  Warning: No valid summary found in {summary_file}")
            continue
        
        print(f"  Evaluating summary {i}/{n_summaries}...")
        
        # Evaluate this summary against the selected comments
        stats, per_comment = evaluate_summary_on_comments(
            evaluator=evaluator,
            question=question,
            summary=summary_text,
            comments=comments_for_eval
        )
        
        results["summaries"].append({
            "summary_index": i,
            "summary_file": str(summary_file),
            "summary_length": len(summary_text),
            "evaluation_stats": stats
        })

        # Append per-comment rows to CSV if requested
        if csv_writer is not None and per_comment:
            for row in per_comment:
                csv_writer.writerow({
                    "topic": topic_name,
                    "model": model_name,
                    "sample_size": sample_size,
                    "summary_index": i,
                    "summary_file": str(summary_file),
                    "comment_index": row.get("comment_index"),
                    "is_minority": row.get("is_minority"),
                    "perspective_representation": row.get("perspective_representation"),
                    "informativeness": row.get("informativeness"),
                    "neutrality_balance": row.get("neutrality_balance"),
                    "policy_approval": row.get("policy_approval"),
                })
    if csv_file_handle is not None:
        csv_file_handle.close()
    
    # Calculate overall statistics across all summaries
    if results["summaries"]:
        overall_stats = {}
        dimensions = ["perspective_representation", "informativeness", "neutrality_balance", "policy_approval"]
        
        for dim in dimensions:
            # Collect all individual values from all summaries for this dimension
            all_values = []
            for s in results["summaries"]:
                if dim in s["evaluation_stats"] and s["evaluation_stats"][dim]["mean"] is not None:
                    # Get the number of samples for this summary
                    n_samples = s["evaluation_stats"][dim].get("n_samples", 0)
                    if n_samples > 0:
                        # For simplicity, we'll use the mean and std to approximate the distribution
                        # Note: This is an approximation; for exact stats, we'd need the raw values
                        mean_val = s["evaluation_stats"][dim]["mean"]
                        # Add the mean value n_samples times (approximation)
                        all_values.extend([mean_val] * n_samples)
            
            # Get all min and max values from individual summaries
            all_mins = [s["evaluation_stats"][dim]["min"] 
                       for s in results["summaries"] 
                       if dim in s["evaluation_stats"] and s["evaluation_stats"][dim]["min"] is not None]
            all_maxs = [s["evaluation_stats"][dim]["max"] 
                       for s in results["summaries"] 
                       if dim in s["evaluation_stats"] and s["evaluation_stats"][dim]["max"] is not None]
            
            if all_values:
                overall_stats[dim] = {
                    "mean": float(np.mean(all_values)),
                    "std": float(np.std(all_values)),
                    "min": float(np.min(all_mins)) if all_mins else None,
                    "max": float(np.max(all_maxs)) if all_maxs else None,
                    "n_samples": len(all_values)
                }
            else:
                overall_stats[dim] = {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "n_samples": 0
                }
        
        results["overall_stats"] = overall_stats
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate DeBERTa model on summary-comment pairs")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained DeBERTa model")
    parser.add_argument("--summary-dir", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/summary_model_for_evaluation",
                        help="Base directory containing summaries")
    parser.add_argument("--comments-dir", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/annotation_V0_V1_dataset",
                        help="Directory containing comment datasets")
    parser.add_argument("--model-names", nargs="+", default=["deepseek-chat"],
                        help="List of summary model names to evaluate")
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[10, 25, 50, 100],
                        help="List of sample sizes to evaluate")
    parser.add_argument("--topics", nargs="+", default=None,
                        help="Specific topics to evaluate (default: all available)")
    parser.add_argument("--n-summaries", type=int, default=3,
                        help="Number of summary samples per topic (default: 3)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file path for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run model on (cuda or cpu)")
    parser.add_argument("--per-comment-csv", type=str, default="",
                        help="Optional CSV file to append per-comment predictions (with comment_index)")
    
    args = parser.parse_args()
    
    # Convert paths
    summary_dir = Path(args.summary_dir)
    comments_dir = Path(args.comments_dir)
    per_comment_csv = Path(args.per_comment_csv) if args.per_comment_csv else None
    
    # Initialize evaluator
    print(f"Loading DeBERTa model from: {args.model_path}")
    evaluator = DebertaEvaluator(args.model_path, device=args.device)
    
    # Get model info
    model_info = evaluator.get_model_info()
    print(f"Model configuration: {model_info['successful_config']}")
    print(f"Target dimensions: {model_info['target_keys']}")
    
    # Determine topics to evaluate
    if args.topics:
        topics = args.topics
    else:
        # Get all available topics from comments directory
        topics = [f.stem for f in comments_dir.glob("*.json")]
        print(f"Found {len(topics)} topics in comments directory")
    
    # Process all combinations
    all_results = {
        "model_path": args.model_path,
        "model_info": model_info,
        "evaluation_params": {
            "summary_dir": str(summary_dir),
            "comments_dir": str(comments_dir),
            "n_summaries": args.n_summaries,
            "model_names": args.model_names,
            "sample_sizes": args.sample_sizes,
            "topics": topics
        },
        "results": []
    }
    
    # Process each combination
    total_evaluations = len(args.model_names) * len(args.sample_sizes) * len(topics)
    current_eval = 0
    
    for model_name in args.model_names:
        for sample_size in args.sample_sizes:
            for topic in topics:
                current_eval += 1
                print(f"\n[{current_eval}/{total_evaluations}] Processing: model={model_name}, size={sample_size}, topic={topic}")
                
                try:
                    result = process_topic(
                        evaluator=evaluator,
                        topic_name=topic,
                        model_name=model_name,
                        sample_size=sample_size,
                        summary_base_dir=summary_dir,
                        comments_dir=comments_dir,
                        n_summaries=args.n_summaries,
                        per_comment_csv=per_comment_csv
                    )
                    
                    all_results["results"].append(result)
                    
                except Exception as e:
                    print(f"Error processing topic {topic}: {e}")
                    all_results["results"].append({
                        "topic": topic,
                        "model": model_name,
                        "sample_size": sample_size,
                        "error": str(e)
                    })
    
    # Save results
    output_path = Path(args.output)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    if per_comment_csv:
        print(f"Per-comment CSV appended to: {per_comment_csv}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for model_name in args.model_names:
        print(f"\nModel: {model_name}")
        for sample_size in args.sample_sizes:
            print(f"  Sample size: {sample_size}")
            
            # Get results for this model and sample size
            relevant_results = [r for r in all_results["results"] 
                               if r.get("model") == model_name and 
                               r.get("sample_size") == sample_size and
                               "overall_stats" in r]
            
            if relevant_results:
                # Calculate average across all topics
                dimensions = ["perspective_representation", "informativeness", 
                            "neutrality_balance", "policy_approval"]
                
                for dim in dimensions:
                    dim_values = [r["overall_stats"][dim]["mean"] 
                                 for r in relevant_results 
                                 if r["overall_stats"][dim]["mean"] is not None]
                    
                    if dim_values:
                        mean_val = np.mean(dim_values)
                        std_val = np.std(dim_values)
                        print(f"    {dim}: {mean_val:.4f} ± {std_val:.4f}")
                    else:
                        print(f"    {dim}: No data")
            else:
                print(f"    No successful evaluations")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()