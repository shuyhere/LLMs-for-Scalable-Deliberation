#!/usr/bin/env python3
"""
Batch evaluation script for assessing summary quality.
Reads summary files from results/summary directory and evaluates comment representation.
Saves evaluation results in the same directory structure with 'eva_' prefix.
"""

import sys
import os
import json
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the src directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_evaluation.evaluator import SummaryEvaluator
from utils.data_loader import load_dataset


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "batch_evaluation_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file not found at {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"Warning: Failed to load configuration from {config_path}: {e}")
        return {}


def load_original_dataset(dataset_name: str, datasets_dir: str = "datasets") -> Optional[tuple]:
    """
    Load original dataset to get comments for evaluation.
    
    Args:
        dataset_name: Name of the dataset (without extension)
        datasets_dir: Directory containing datasets
        
    Returns:
        Tuple of (question, comments) or None if failed
    """
    try:
        dataset_path = os.path.join(datasets_dir, f"{dataset_name}.json")
        if not os.path.exists(dataset_path):
            print(f"Warning: Original dataset not found: {dataset_path}")
            return None
            
        question, comments, _ = load_dataset(dataset_path)
        return question, comments
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None


def sample_comments_for_evaluation(comments: List[str], comment_indices: List[int], config: Dict[str, Any]) -> List[str]:
    """
    Sample comments for evaluation based on configuration.
    
    Args:
        comments: All available comments from original dataset
        comment_indices: Indices of comments used in summary generation
        config: Configuration dictionary
        
    Returns:
        Sampled list of comments for evaluation
    """
    sampling_config = config.get("evaluation", {}).get("comment_sampling", {})
    max_comments = sampling_config.get("max_comments_per_dataset")
    sampling_strategy = sampling_config.get("sampling_strategy", "first_n")
    sample_from_summary = sampling_config.get("sample_from_summary_comments", True)
    
    # Determine which comments to sample from
    if sample_from_summary and comment_indices:
        # Sample from comments that were actually used in summary generation
        available_comments = [comments[i] for i in comment_indices if i < len(comments)]
        print(f"  Sampling from {len(available_comments)} comments used in summary (out of {len(comments)} total)")
    else:
        # Sample from all available comments
        available_comments = comments
        print(f"  Sampling from all {len(available_comments)} available comments")
    
    if not available_comments:
        print(f"  Warning: No comments available for sampling")
        return []
    
    # Apply sampling strategy
    if max_comments and len(available_comments) > max_comments:
        if sampling_strategy == "random":
            import random
            random.seed(sampling_config.get("random_seed", 42))
            sampled_comments = random.sample(available_comments, max_comments)
            print(f"  Randomly sampled {max_comments} comments (seed: {sampling_config.get('random_seed', 42)})")
        elif sampling_strategy == "last_n":
            sampled_comments = available_comments[-max_comments:]
            print(f"  Sampled last {max_comments} comments")
        elif sampling_strategy == "stratified":
            # For stratified sampling, we'd need previous evaluation results
            # For now, fall back to first_n
            print(f"  Stratified sampling not yet implemented, using first_n")
            sampled_comments = available_comments[:max_comments]
        else:  # "first_n" (default)
            sampled_comments = available_comments[:max_comments]
            print(f"  Sampled first {max_comments} comments")
    else:
        sampled_comments = available_comments
        print(f"  Using all {len(sampled_comments)} available comments")
    
    return sampled_comments


def evaluate_summary_quality(summary: str, comments: List[str], comment_indices: List[int], model: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the quality of a summary using the SummaryEvaluator.
    
    Args:
        summary: The summary to evaluate
        comments: List of all available comments
        comment_indices: Indices of comments used in summary generation
        model: Model to use for evaluation
        config: Configuration dictionary
        
    Returns:
        Dictionary containing evaluation results and statistics
    """
    try:
        # Get model parameters from config
        model_config = config.get("evaluation_model", {})
        temperature = model_config.get("temperature", 0.7)
        system_prompt = model_config.get("system_prompt", "You are a helpful assistant")
        
        print(f"  Using evaluation model: {model}")
        print(f"  Temperature: {temperature}")
        print(f"  System prompt: {system_prompt}")
        
        # Initialize evaluator with model parameters
        evaluator = SummaryEvaluator(model=model, system_prompt=system_prompt, temperature=temperature)
        
        # Sample comments for evaluation
        sampled_comments = sample_comments_for_evaluation(comments, comment_indices, config)
        
        if not sampled_comments:
            print(f"  No comments available for evaluation")
            return {
                "evaluation_results": [],
                "statistics": {},
                "evaluation_model": model,
                "evaluation_model_name": model,
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_version": "1.0",
                "total_comments_evaluated": 0,
                "error": "No comments available for evaluation"
            }
        
        # Evaluate sampled comments against the summary
        print(f"  Evaluating {len(sampled_comments)} sampled comments...")
        evaluation_results = evaluator.evaluate_multiple_comments(summary, sampled_comments)
        
        # Get evaluation statistics
        statistics = evaluator.get_evaluation_statistics(evaluation_results)
        
        return {
            "evaluation_results": evaluation_results,
            "statistics": statistics,
            "evaluation_model": model,
            "evaluation_model_name": model,  # Clear model name for identification
            "evaluation_timestamp": datetime.now().isoformat(),  # When evaluation was performed
            "evaluation_version": "1.0",  # Evaluation script version
            "total_comments_evaluated": len(sampled_comments),
            "sampling_info": {
                "total_available_comments": len(comments),
                "comments_used_in_summary": len(comment_indices) if comment_indices else 0,
                "sampled_comments_count": len(sampled_comments),
                "sampling_strategy": config.get("evaluation", {}).get("comment_sampling", {}).get("sampling_strategy", "first_n")
            },
            "model_parameters": {
                "temperature": temperature,
                "system_prompt": system_prompt
            }
        }
        
    except Exception as e:
        print(f"  Error during evaluation: {e}")
        return {
            "evaluation_results": [],
            "statistics": {},
            "evaluation_model": model,
            "evaluation_model_name": model,  # Clear model name for identification
            "evaluation_timestamp": datetime.now().isoformat(),  # When evaluation was performed
            "evaluation_version": "1.0",  # Evaluation script version
            "total_comments_evaluated": 0,
            "error": str(e)
        }


def save_evaluation_result(original_summary: Dict[str, Any], evaluation_data: Dict[str, Any], 
                          output_dir: str, dataset_name: str, model: str, config: Dict[str, Any]) -> str:
    """
    Save evaluation result by merging with original summary data.
    
    Args:
        original_summary: Original summary data
        evaluation_data: Evaluation results and statistics
        output_dir: Base output directory
        dataset_name: Name of the dataset
        model: Model used for summarization
        config: Configuration dictionary
        
    Returns:
        Path to saved evaluation file
    """
    # Create directory structure: summary/model/dataset
    model_dir = os.path.join(output_dir, model)
    dataset_dir = os.path.join(model_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Get file prefix from config
    file_prefix = config.get("output", {}).get("file_prefix", "eva_summary_")
    filename = f"{file_prefix}{dataset_name}.json"
    filepath = os.path.join(dataset_dir, filename)
    
    # Check if evaluation file already exists
    existing_evaluations = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_evaluations = existing_data.get("evaluations", {})
        except Exception as e:
            print(f"  Warning: Could not read existing evaluation file: {e}")
            existing_evaluations = {}
    
    # Add new evaluation results
    evaluation_model = evaluation_data.get("evaluation_model")
    if evaluation_model:
        existing_evaluations[evaluation_model] = {
            "evaluation_data": evaluation_data,
            "evaluation_timestamp": evaluation_data.get("evaluation_timestamp"),
            "model_parameters": evaluation_data.get("model_parameters", {}),
            "sampling_info": evaluation_data.get("sampling_info", {})
        }
    
    # Prepare evaluation result
    if config.get("output", {}).get("include_original_data", True):
        # Reorganize data to eliminate duplicates and improve structure
        optimized_summary = {
            "metadata": {
                "dataset_name": original_summary.get("metadata", {}).get("dataset_name"),
                "question": original_summary.get("metadata", {}).get("question"),
                "summary_model": original_summary.get("metadata", {}).get("summary_model"),
                "num_samples": original_summary.get("metadata", {}).get("num_samples"),
                "summary_timestamp": original_summary.get("metadata", {}).get("timestamp"),
                "version": original_summary.get("metadata", {}).get("version")
            },
            "summary_parameters": {
                "summary_types": original_summary.get("parameters", {}).get("summary_types"),
                "custom_system_prompt": original_summary.get("parameters", {}).get("custom_system_prompt"),
                "custom_user_prompt": original_summary.get("parameters", {}).get("custom_user_prompt"),
                "script_version": original_summary.get("parameters", {}).get("script_version")
            },
            "summaries": original_summary.get("summaries", {}),
            "comment_indices": original_summary.get("comment_indices", []),
            "evaluations": existing_evaluations,  # Multiple model evaluations
            "evaluation_info": {
                "file_type": "multi_model_evaluation_result",
                "evaluation_date": evaluation_data.get("evaluation_timestamp")[:10] if evaluation_data.get("evaluation_timestamp") else None,
                "models_evaluated": list(existing_evaluations.keys())
            }
        }
        evaluation_result = optimized_summary
    else:
        evaluation_result = {
            "dataset_name": dataset_name,
            "model": model,
            "evaluations": existing_evaluations,
            "evaluation_info": {
                "file_type": "multi_model_evaluation_result",
                "evaluation_date": evaluation_data.get("evaluation_timestamp")[:10] if evaluation_data.get("evaluation_timestamp") else None,
                "models_evaluated": list(existing_evaluations.keys())
            }
        }
    
    # Save to JSON file
    indent = 2 if config.get("output", {}).get("pretty_print", True) else None
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, indent=indent, ensure_ascii=False)
    
    return filepath


def process_summary_file(summary_file_path: str, output_dir: str, evaluation_model: str, config: Dict[str, Any]) -> bool:
    """
    Process a single summary file and generate evaluation.
    
    Args:
        summary_file_path: Path to the summary file
        output_dir: Base output directory for results
        evaluation_model: Model to use for evaluation
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load summary file
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Extract dataset name and model from summary data
        dataset_name = summary_data.get("metadata", {}).get("dataset_name")
        summary_model = summary_data.get("metadata", {}).get("summary_model")
        
        if not dataset_name:
            print(f"  Warning: No dataset name found in {summary_file_path}")
            return False
            
        print(f"  Processing dataset: {dataset_name}")
        print(f"  Summary model: {summary_model}")
        
        # Load original dataset to get comments
        datasets_dir = config.get("paths", {}).get("datasets_directory", "datasets")
        original_data = load_original_dataset(dataset_name, datasets_dir)
        if not original_data:
            print(f"  Skipping {dataset_name} - could not load original dataset")
            return False
            
        question, comments = original_data
        print(f"  Loaded {len(comments)} comments from original dataset")
        
        # Get the main summary text based on config preferences
        summaries = summary_data.get("summaries", {})
        summary_types = config.get("evaluation", {}).get("summary_types", ["main_points", "topic_modeling", "custom_analysis"])
        
        main_summary = None
        for summary_type in summary_types:
            summary_text = summaries.get(summary_type)
            if summary_text and not summary_text.startswith("Topic modeling summary generation disabled") and not summary_text.startswith("Custom analysis summary generation disabled"):
                main_summary = summary_text
                print(f"  Using {summary_type} summary for evaluation")
                break
        
        if not main_summary:
            print(f"  Warning: No valid summary found for {dataset_name}")
            return False
        
        # Get comment indices used in summary generation
        comment_indices = summary_data.get("comment_indices", [])
        
        # Evaluate summary quality
        print(f"  Evaluating summary quality...")
        evaluation_data = evaluate_summary_quality(main_summary, comments, comment_indices, evaluation_model, config)
        
        # Save evaluation result
        output_file = save_evaluation_result(summary_data, evaluation_data, output_dir, dataset_name, summary_model, config)
        print(f"  Evaluation saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {summary_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_summary_files(results_dir: str) -> List[str]:
    """
    Find all summary files in the results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        List of paths to summary files
    """
    summary_files = []
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return summary_files
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.startswith("summary_") and file.endswith(".json"):
                summary_files.append(os.path.join(root, file))
    
    return summary_files


def main():
    """Main function for batch summary evaluation."""
    
    parser = argparse.ArgumentParser(description="Batch evaluation script for summary quality assessment")
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (default: config/batch_evaluation_config.yaml)"
    )
    
    parser.add_argument(
        "--results-dir",
        help="Path to results directory (overrides config file)"
    )
    
    parser.add_argument(
        "--evaluation-model",
        help="Model to use for evaluation (overrides config file)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for evaluation results (overrides config file)"
    )
    
    args = parser.parse_args()
    
    # Load configuration file
    config = load_config(args.config)
    
    # Set defaults from config file, command line arguments override config
    default_results_dir = config.get("paths", {}).get("results_directory", "results/summary")
    default_evaluation_model = config.get("evaluation_model", {}).get("name", "gpt-4o-mini")
    default_output_dir = config.get("paths", {}).get("output_directory", "results/summary")
    
    # Use command line arguments if provided, otherwise use config defaults
    results_dir = args.results_dir if args.results_dir else default_results_dir
    evaluation_model = args.evaluation_model if args.evaluation_model else default_evaluation_model
    output_dir = args.output_dir if args.output_dir else default_output_dir
    
    print("=== BATCH SUMMARY EVALUATION SCRIPT ===")
    print(f"Configuration file: {args.config or 'config/batch_evaluation_config.yaml (default)'}")
    print(f"Results directory: {results_dir}")
    print(f"Evaluation model: {evaluation_model}")
    print(f"Output directory: {output_dir}")
    
    # Find all summary files
    print(f"\nSearching for summary files in {results_dir}...")
    summary_files = find_summary_files(results_dir)
    
    if not summary_files:
        print("No summary files found!")
        return
    
    print(f"Found {len(summary_files)} summary files")
    
    # Process each summary file
    successful_evaluations = 0
    failed_evaluations = 0
    
    for i, summary_file in enumerate(summary_files, 1):
        print(f"\n[{i}/{len(summary_files)}] Processing: {summary_file}")
        
        if process_summary_file(summary_file, output_dir, evaluation_model, config):
            successful_evaluations += 1
        else:
            failed_evaluations += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total summary files processed: {len(summary_files)}")
    print(f"Successful evaluations: {successful_evaluations}")
    print(f"Failed evaluations: {failed_evaluations}")
    print(f"Success rate: {successful_evaluations/len(summary_files)*100:.1f}%")
    
    if successful_evaluations > 0:
        file_prefix = config.get("output", {}).get("file_prefix", "eva_summary_")
        print(f"\nEvaluation files saved with '{file_prefix}' prefix in the same directory structure.")
        print(f"Each evaluation file contains the original summary data plus evaluation results.")


if __name__ == "__main__":
    main()
