#!/usr/bin/env python3
"""
Batch evaluation script for assessing summary quality.
Reads summary files from results/summary directory and evaluates comment representation.
Saves evaluation results in the same directory structure with 'eva_' prefix.
Supports checkpoint and resume functionality.
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


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """
    Load checkpoint data if it exists.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Checkpoint data dictionary
    """
    if not os.path.exists(checkpoint_file):
        return {
            "completed_files": [],
            "failed_files": [],
            "start_time": datetime.now().isoformat(),
            "evaluation_model": None,
            "total_files": 0
        }
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        print(f"Checkpoint loaded from: {checkpoint_file}")
        return checkpoint_data
    except Exception as e:
        print(f"Warning: Failed to load checkpoint from {checkpoint_file}: {e}")
        return {
            "completed_files": [],
            "failed_files": [],
            "start_time": datetime.now().isoformat(),
            "evaluation_model": None,
            "total_files": 0
        }


def save_checkpoint(checkpoint_file: str, checkpoint_data: Dict[str, Any]) -> None:
    """
    Save checkpoint data.
    
    Args:
        checkpoint_file: Path to checkpoint file
        checkpoint_data: Checkpoint data to save
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        # Update checkpoint data
        checkpoint_data["last_updated"] = datetime.now().isoformat()
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint to {checkpoint_file}: {e}")


def update_checkpoint(checkpoint_file: str, completed_file: str = None, failed_file: str = None) -> None:
    """
    Update checkpoint with completed or failed file.
    
    Args:
        checkpoint_file: Path to checkpoint file
        completed_file: Path to completed file (optional)
        failed_file: Path to failed file (optional)
    """
    checkpoint_data = load_checkpoint(checkpoint_file)
    
    if completed_file:
        if completed_file not in checkpoint_data["completed_files"]:
            checkpoint_data["completed_files"].append(completed_file)
            print(f"  ✓ Added to completed files: {os.path.basename(completed_file)}")
    
    if failed_file:
        if failed_file not in checkpoint_data["failed_files"]:
            checkpoint_data["failed_files"].append(failed_file)
            print(f"  ✗ Added to failed files: {os.path.basename(failed_file)}")
    
    save_checkpoint(checkpoint_file, checkpoint_data)


def is_file_completed(checkpoint_file: str, file_path: str) -> bool:
    """
    Check if a file has already been completed based on checkpoint.
    
    Args:
        checkpoint_file: Path to checkpoint file
        file_path: Path to file to check
        
    Returns:
        True if file is already completed
    """
    checkpoint_data = load_checkpoint(checkpoint_file)
    return file_path in checkpoint_data["completed_files"]


def get_remaining_files(all_files: List[str], checkpoint_file: str) -> List[str]:
    """
    Get list of files that still need to be processed.
    
    Args:
        all_files: List of all files to process
        checkpoint_file: Path to checkpoint file
        
    Returns:
        List of remaining files to process
    """
    checkpoint_data = load_checkpoint(checkpoint_file)
    completed_files = set(checkpoint_data["completed_files"])
    
    remaining_files = [f for f in all_files if f not in completed_files]
    
    if remaining_files:
        print(f"Resuming from checkpoint: {len(remaining_files)} files remaining out of {len(all_files)} total")
        print(f"Already completed: {len(completed_files)} files")
    else:
        print("All files already completed according to checkpoint!")
    
    return remaining_files


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


def find_summary_files(results_dir: str, target_dataset: str = None) -> List[str]:
    """
    Find summary files in the results directory.
    
    Args:
        results_dir: Path to results directory
        target_dataset: Specific dataset to look for (if None, find all)
        
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
                file_path = os.path.join(root, file)
                
                # If target dataset is specified, filter by dataset name
                if target_dataset:
                    # Extract dataset name from file path
                    # Path format: results/summary/model/dataset/summary_dataset.json
                    path_parts = file_path.split(os.sep)
                    if len(path_parts) >= 4:  # Ensure we have enough path components
                        file_dataset = path_parts[-2]  # Second to last part is dataset name
                        if file_dataset == target_dataset:
                            summary_files.append(file_path)
                            print(f"  Found summary file for dataset '{target_dataset}': {file}")
                else:
                    # No target dataset specified, include all files
                    summary_files.append(file_path)
    
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
    
    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint file for resume functionality (default: logs/evaluation_checkpoint.json)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if it exists"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already completed files"
    )
    
    parser.add_argument(
        "--dataset",
        help="Specific dataset to evaluate (if None, evaluate all)"
    )
    
    args = parser.parse_args()
    
    # Load configuration file
    config = load_config(args.config)
    
    # Set defaults from config file, command line arguments override config
    default_results_dir = config.get("paths", {}).get("results_directory", "results/summary")
    default_evaluation_model = config.get("evaluation_model", {}).get("name", "gpt-4o-mini")
    default_output_dir = config.get("paths", {}).get("output_directory", "results/summary")
    default_checkpoint = "evalsum_logs/evaluation_checkpoint.json"
    
    # Use command line arguments if provided, otherwise use config defaults
    results_dir = args.results_dir if args.results_dir else default_results_dir
    evaluation_model = args.evaluation_model if args.evaluation_model else default_evaluation_model
    output_dir = args.output_dir if args.output_dir else default_output_dir
    checkpoint_file = args.checkpoint if args.checkpoint else default_checkpoint
    target_dataset = args.dataset
    
    print("=== BATCH SUMMARY EVALUATION SCRIPT ===")
    print(f"Configuration file: {args.config or 'config/batch_evaluation_config.yaml (default)'}")
    print(f"Results directory: {results_dir}")
    print(f"Evaluation model: {evaluation_model}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Resume mode: {'Enabled' if args.resume else 'Disabled'}")
    print(f"Force reprocess: {'Enabled' if args.force else 'Disabled'}")
    print(f"Target dataset: {target_dataset or 'All datasets'}")
    
    # Find all summary files
    print(f"\nSearching for summary files in {results_dir}...")
    summary_files = find_summary_files(results_dir, target_dataset)
    
    if not summary_files:
        print("No summary files found!")
        return
    
    print(f"Found {len(summary_files)} summary files")
    
    # Handle checkpoint and resume functionality
    if args.resume:
        # Load checkpoint and get remaining files
        summary_files = get_remaining_files(summary_files, checkpoint_file)
        if not summary_files:
            print("No files remaining to process!")
            return
    elif not args.force:
        # Check if checkpoint exists and ask user what to do
        if os.path.exists(checkpoint_file):
            print(f"\nCheckpoint file found: {checkpoint_file}")
            print("Use --resume to continue from checkpoint or --force to reprocess all files")
            print("Or delete the checkpoint file to start fresh")
            return
    
    # Initialize checkpoint data
    checkpoint_data = load_checkpoint(checkpoint_file)
    checkpoint_data["evaluation_model"] = evaluation_model
    checkpoint_data["total_files"] = len(summary_files)
    save_checkpoint(checkpoint_file, checkpoint_data)
    
    # Process each summary file
    successful_evaluations = 0
    failed_evaluations = 0
    
    for i, summary_file in enumerate(summary_files, 1):
        print(f"\n[{i}/{len(summary_files)}] Processing: {summary_file}")
        
        # Check if file is already completed (unless force mode)
        if not args.force and is_file_completed(checkpoint_file, summary_file):
            print(f"  ⏭️  Skipping already completed file: {os.path.basename(summary_file)}")
            successful_evaluations += 1
            continue
        
        try:
            if process_summary_file(summary_file, output_dir, evaluation_model, config):
                successful_evaluations += 1
                # Update checkpoint with successful completion
                update_checkpoint(checkpoint_file, completed_file=summary_file)
            else:
                failed_evaluations += 1
                # Update checkpoint with failed file
                update_checkpoint(checkpoint_file, failed_file=summary_file)
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user. Progress saved to checkpoint: {checkpoint_file}")
            print("Use --resume to continue from this point")
            return
        except Exception as e:
            print(f"  ❌ Unexpected error processing {summary_file}: {e}")
            failed_evaluations += 1
            update_checkpoint(checkpoint_file, failed_file=summary_file)
    
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
    
    # Final checkpoint update
    checkpoint_data = load_checkpoint(checkpoint_file)
    checkpoint_data["completion_time"] = datetime.now().isoformat()
    checkpoint_data["final_status"] = "completed" if failed_evaluations == 0 else "completed_with_errors"
    save_checkpoint(checkpoint_file, checkpoint_data)
    
    print(f"\nCheckpoint saved to: {checkpoint_file}")
    if failed_evaluations > 0:
        print(f"Failed files can be retried by running with --resume flag")


if __name__ == "__main__":
    main()
