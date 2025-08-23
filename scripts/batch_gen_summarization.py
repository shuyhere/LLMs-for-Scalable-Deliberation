#!/usr/bin/env python3
"""
Batch summarization script for generating test samples.
Supports specifying dataset, model, and number of samples.
Saves results to JSON files in results/summary folder.
"""

import sys
import os
import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add the src directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_summarization.summarizer import CommentSummarizer
from utils.data_loader import read_json_dataset


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "batch_summarization_config.yaml"
    
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


def create_summary_result(
    dataset_name: str,
    question: str,
    model: str,
    num_samples: int,
    topic_summary: str,
    main_summary: str,
    custom_summary: str,
    comments_used: List[str],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a structured summary result dictionary.
    
    Args:
        dataset_name: Name of the dataset
        question: The question from the dataset
        model: Model used for summarization
        num_samples: Number of comments used
        topic_summary: Topic modeling summary
        main_summary: Main points summary
        custom_summary: Custom analysis summary
        comments_used: List of comments used for summarization
        parameters: Dictionary of parameters used
        
    Returns:
        Structured result dictionary
    """
    return {
        "metadata": {
            "dataset_name": dataset_name,
            "question": question,
            "model": model,
            "num_samples": num_samples,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        },
        "parameters": parameters,
        "summaries": {
            "topic_modeling": topic_summary,
            "main_points": main_summary,
            "custom_analysis": custom_summary
        },
        "comment_indices": list(range(len(comments_used))),
        "statistics": {
            "total_comments_in_dataset": len(comments_used),
            "comments_used_for_summary": num_samples,
            "topic_summary_length": len(topic_summary),
            "main_summary_length": len(main_summary),
            "custom_summary_length": len(custom_summary)
        }
    }


def save_summary_result(result: Dict[str, Any], output_dir: str, dataset_name: str) -> str:
    """
    Save summary result to JSON file.
    
    Args:
        result: Summary result dictionary
        output_dir: Output directory path
        dataset_name: Name of the dataset
        
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_{dataset_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return filepath


def process_dataset(
    dataset_path: str,
    model: str,
    num_samples: int,
    output_dir: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single dataset and generate summaries.
    
    Args:
        dataset_path: Path to the JSON dataset file
        model: Model to use for summarization
        num_samples: Number of comments to use
        output_dir: Output directory for results
        parameters: Additional parameters
        
    Returns:
        Summary result dictionary
    """
    dataset_name = Path(dataset_path).stem
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Model: {model}")
    print(f"Number of samples: {num_samples}")
    print(f"{'='*60}")
    
    # Read dataset
    try:
        question, comments = read_json_dataset(dataset_path)
        
        if not comments:
            print(f"No comments found in {dataset_name}")
            return None
            
        print(f"Question: {question[:100]}...")
        print(f"Total comments in dataset: {len(comments)}")
        
        # Limit comments to specified number
        if num_samples > 0:
            comments_to_use = comments[:num_samples]
            print(f"Using first {len(comments_to_use)} comments")
        else:
            comments_to_use = comments
            print(f"Using all {len(comments_to_use)} comments")
        
        # Format comments for summarization
        formatted_comments = "\n".join([f"Comment {i+1}: {comment}" for i, comment in enumerate(comments_to_use)])
        
        # Initialize summarizer
        summarizer = CommentSummarizer(model=model)
        
        # Generate summaries based on configuration
        print("\nGenerating summaries...")
        
        # Initialize summary variables
        topic_summary = ""
        main_summary = ""
        custom_summary = ""
        
        # Get summary type settings from parameters
        summary_types = parameters.get("summary_types", {})
        
        # 1. Topic Modeling Summary
        if summary_types.get("topic_modeling", True):
            print("1. Topic Modeling Summary...")
            topic_summary = summarizer.summarize_topic_modeling(formatted_comments)
        else:
            print("1. Topic Modeling Summary... SKIPPED")
            topic_summary = "Topic modeling summary generation disabled in configuration"
        
        # 2. Main Points Summary
        if summary_types.get("main_points", True):
            print("2. Main Points Summary...")
            main_summary = summarizer.summarize_main_points(formatted_comments)
        else:
            print("2. Main Points Summary... SKIPPED")
            main_summary = "Main points summary generation disabled in configuration"
        
        # 3. Custom Analysis Summary
        if summary_types.get("custom_analysis", True):
            print("3. Custom Analysis Summary...")
            custom_summary = summarizer.summarize_with_custom_prompt(
                comments=formatted_comments,
                custom_system_prompt="You are an expert data analyst specializing in public opinion analysis.",
                custom_user_prompt="Analyze the sentiment and key themes in these comments. Focus on areas of agreement and disagreement: {comments}"
            )
        else:
            print("3. Custom Analysis Summary... SKIPPED")
            custom_summary = "Custom analysis summary generation disabled in configuration"
        
        # Create result structure
        result = create_summary_result(
            dataset_name=dataset_name,
            question=question,
            model=model,
            num_samples=len(comments_to_use),
            topic_summary=topic_summary,
            main_summary=main_summary,
            custom_summary=custom_summary,
            comments_used=comments_to_use,
            parameters=parameters
        )
        
        # Save result
        output_file = save_summary_result(result, output_dir, dataset_name)
        print(f"\nResults saved to: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function for batch summarization."""
    
    parser = argparse.ArgumentParser(description="Batch summarization script for generating test samples")
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (default: config/batch_summarization_config.yaml)"
    )
    
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        help="List of dataset paths to process (overrides config file)"
    )
    
    parser.add_argument(
        "--model", 
        help="Model to use for summarization (overrides config file)"
    )
    
    parser.add_argument(
        "--num-samples", 
        type=int, 
        help="Number of comments to use for each dataset (0 for all, overrides config file)"
    )
    
    parser.add_argument(
        "--output-dir", 
        help="Output directory for results (overrides config file)"
    )
    
    parser.add_argument(
        "--custom-system-prompt",
        help="Custom system prompt for analysis (overrides config file)"
    )
    
    parser.add_argument(
        "--custom-user-prompt",
        help="Custom user prompt template for analysis (overrides config file)"
    )
    
    args = parser.parse_args()
    
    # Load configuration file
    config = load_config(args.config)
    
    # Set defaults from config file, command line arguments override config
    default_datasets = config.get("datasets", ["datasets/protest.json"])
    default_model = config.get("model", {}).get("name", "gpt-4o-mini")
    default_num_samples = config.get("processing", {}).get("num_samples", 100)
    default_output_dir = config.get("output", {}).get("directory", "results/summary")
    default_system_prompt = config.get("prompts", {}).get("custom_system_prompt", 
        "You are an expert data analyst specializing in public opinion analysis.")
    default_user_prompt = config.get("prompts", {}).get("custom_user_prompt",
        "Analyze the sentiment and key themes in these comments. Focus on areas of agreement and disagreement: {comments}")
    
    # Use command line arguments if provided, otherwise use config defaults
    datasets = args.datasets if args.datasets else default_datasets
    model = args.model if args.model else default_model
    num_samples = args.num_samples if args.num_samples is not None else default_num_samples
    output_dir = args.output_dir if args.output_dir else default_output_dir
    custom_system_prompt = args.custom_system_prompt if args.custom_system_prompt else default_system_prompt
    custom_user_prompt = args.custom_user_prompt if args.custom_user_prompt else default_user_prompt
    
    # Create parameters dictionary
    parameters = {
        "model": model,
        "num_samples": num_samples,
        "custom_system_prompt": custom_system_prompt,
        "custom_user_prompt": custom_user_prompt,
        "script_version": "1.0",
        "command_line_args": vars(args),
        "config_file": str(Path(__file__).parent.parent / "config" / "batch_summarization_config.yaml") if not args.config else args.config,
        "summary_types": config.get("summary_types", {
            "topic_modeling": True,
            "main_points": True,
            "custom_analysis": True
        })
    }
    
    print("=== BATCH SUMMARIZATION SCRIPT ===")
    print(f"Configuration file: {args.config or 'config/batch_summarization_config.yaml (default)'}")
    print(f"Model: {model}")
    print(f"Number of samples per dataset: {num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Datasets to process: {len(datasets)}")
    
    # Process each dataset
    results = []
    
    for dataset_path in datasets:
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            continue
            
        result = process_dataset(
            dataset_path=dataset_path,
            model=model,
            num_samples=num_samples,
            output_dir=output_dir,
            parameters=parameters
        )
        
        if result:
            results.append(result)
    
    # Generate summary report
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY REPORT")
        print(f"{'='*60}")
        
        summary_report = {
            "metadata": {
                "total_datasets_processed": len(results),
                "model_used": model,
                "timestamp": datetime.now().isoformat(),
                "parameters": parameters
            },
            "datasets_processed": [r["metadata"]["dataset_name"] for r in results],
            "results": results
        }
        
        # Save summary report
        report_file = os.path.join(output_dir, f"batch_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        print(f"Summary report saved to: {report_file}")
        print(f"Successfully processed {len(results)} datasets")
        
        # Display quick statistics
        for result in results:
            dataset_name = result["metadata"]["dataset_name"]
            num_samples = result["metadata"]["num_samples"]
            print(f"  - {dataset_name}: {num_samples} samples processed")
    else:
        print("No datasets were successfully processed.")


if __name__ == "__main__":
    main()
