#!/usr/bin/env python3
"""
Script to count summary files for each model in the results directory.
Focuses on main_points field and handles TA folder structure properly.
Identifies missing data and provides comprehensive analysis.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set


def get_expected_combinations() -> Tuple[Set[str], Set[str], Set[int]]:
    """
    Get expected models, datasets, and sample counts based on the configuration.
    
    Returns:
        Tuple of (expected_models, expected_datasets, expected_sample_counts)
    """
    # Expected models from sbatch_gen.sh
    expected_models = {
        "gpt-5-mini", "web-rev-claude-sonnet-4-20250514", "gemini-2.5-flash", 
        "deepseek-reasoner", "grok-4-latest", "gpt-oss-120b", "gpt-oss-20b", 
        "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", 
        "qwen3-30b-a3b", "qwen3-235b-a22b", "gpt-5", "qwen3-32b", 
        "web-rev-claude-opus-4-20250514", "deepseek-chat", "gemini-2.5-pro"
    }
    
    # Expected datasets
    expected_datasets = {
        "Binary-Health-Care-Policy", "Binary-Online-Identity-Policies", 
        "Binary-Refugee-Policies", "Binary-Tariff-Policy", "Binary-Vaccination-Policy",
        "Openqa-AI-changes-human-life", "Openqa-Tipping-System", 
        "Openqa-Trump-cutting-funding", "Openqa-Updates-of-electronic-products", 
        "Openqa-Influencers-as-a-job"
    }
    
    # Expected sample counts
    expected_sample_counts = {10, 30, 50, 70, 90}
    
    return expected_models, expected_datasets, expected_sample_counts


def count_summary_files(results_dir: str) -> Tuple[Dict[str, Dict[str, Dict[int, int]]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Count summary files for each model, organized by dataset and sample count.
    Also identify missing data.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        Tuple of (model_counts, missing_models, missing_combinations)
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    model_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    found_models = set()
    found_datasets = set()
    found_sample_counts = set()
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(results_path):
        root_path = Path(root)
        
        # Skip if no JSON files
        json_files = [f for f in files if f.endswith('.json') and 'summary' in f]
        if not json_files:
            continue
            
        # Extract model name from path
        # Handle TA folder structure: TA/openai/gpt-oss-20b -> gpt-oss-20b
        path_parts = root_path.relative_to(results_path).parts
        
        if len(path_parts) >= 3 and path_parts[0] == 'TA' and path_parts[1] == 'openai':
            # TA/openai/model_name structure
            model_name = path_parts[2]
        elif len(path_parts) >= 1:
            # Direct model_name structure
            model_name = path_parts[0]
        else:
            continue
            
        found_models.add(model_name)
            
        # Extract dataset name and sample count from path
        # Expected structure: model/sample_count/dataset_name_summary_X.json
        if len(path_parts) >= 2:
            try:
                sample_count = int(path_parts[-1])  # Last part should be sample count
                found_sample_counts.add(sample_count)
                
                # Dataset name is in the filename
                for json_file in json_files:
                    if 'summary' in json_file:
                        # Extract dataset name from filename
                        dataset_name = json_file.replace('_summary_', '_').replace('.json', '')
                        # Remove sample number suffix if present
                        if '_' in dataset_name and dataset_name.split('_')[-1].isdigit():
                            dataset_name = '_'.join(dataset_name.split('_')[:-1])
                        
                        found_datasets.add(dataset_name)
                        
                        # Check if file has main_points
                        file_path = root_path / json_file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if 'summaries' in data and 'main_points' in data['summaries']:
                                    model_counts[model_name][dataset_name][sample_count] += 1
                        except (json.JSONDecodeError, KeyError, FileNotFoundError):
                            continue
            except ValueError:
                # If sample_count is not a number, skip
                continue
    
    # Identify missing data
    expected_models, expected_datasets, expected_sample_counts = get_expected_combinations()
    
    missing_models = expected_models - found_models
    missing_datasets = expected_datasets - found_datasets
    missing_sample_counts = expected_sample_counts - found_sample_counts
    
    return dict(model_counts), missing_models, missing_datasets


def analyze_missing_data(model_counts: Dict[str, Dict[str, Dict[int, int]]], 
                        missing_models: Set[str], 
                        missing_datasets: Set[str]) -> None:
    """Analyze and report missing data patterns."""
    
    expected_models, expected_datasets, expected_sample_counts = get_expected_combinations()
    expected_combinations = len(expected_models) * len(expected_datasets) * len(expected_sample_counts) * 3  # 3 sample times
    
    print("=" * 80)
    print("MISSING DATA ANALYSIS")
    print("=" * 80)
    
    # Missing models
    if missing_models:
        print(f"\n❌ MISSING MODELS ({len(missing_models)}):")
        for model in sorted(missing_models):
            print(f"  - {model}")
    else:
        print(f"\n✅ All expected models found ({len(expected_models)})")
    
    # Missing datasets
    if missing_datasets:
        print(f"\n❌ MISSING DATASETS ({len(missing_datasets)}):")
        for dataset in sorted(missing_datasets):
            print(f"  - {dataset}")
    else:
        print(f"\n✅ All expected datasets found ({len(expected_datasets)})")
    
    # Analyze completeness per model
    print(f"\n📊 COMPLETENESS BY MODEL:")
    print("-" * 60)
    
    model_completeness = []
    for model_name in sorted(model_counts.keys()):
        model_data = model_counts[model_name]
        found_combinations = sum(sum(counts.values()) for counts in model_data.values())
        expected_for_model = len(expected_datasets) * len(expected_sample_counts) * 3  # 3 sample times
        completeness = (found_combinations / expected_for_model) * 100 if expected_for_model > 0 else 0
        
        model_completeness.append((model_name, found_combinations, expected_for_model, completeness))
        
        status = "✅" if completeness >= 95 else "⚠️" if completeness >= 50 else "❌"
        print(f"{status} {model_name}: {found_combinations}/{expected_for_model} ({completeness:.1f}%)")
    
    # Overall completeness
    total_found = sum(count for _, count, _, _ in model_completeness)
    total_expected = len(expected_models) * len(expected_datasets) * len(expected_sample_counts) * 3
    overall_completeness = (total_found / total_expected) * 100 if total_expected > 0 else 0
    
    print(f"\n📈 OVERALL COMPLETENESS: {total_found}/{total_expected} ({overall_completeness:.1f}%)")
    
    # Models with most missing data
    incomplete_models = [(model, expected - found, completeness) 
                        for model, found, expected, completeness in model_completeness 
                        if expected - found > 0]
    
    if incomplete_models:
        incomplete_models.sort(key=lambda x: x[1], reverse=True)
        print(f"\n🔍 MODELS WITH MOST MISSING DATA:")
        for model, missing_count, completeness in incomplete_models[:10]:
            print(f"  {model}: {missing_count} missing files ({completeness:.1f}% complete)")


def print_summary_counts(model_counts: Dict[str, Dict[str, Dict[int, int]]]) -> None:
    """Print the summary counts in a formatted way."""
    
    print("=" * 80)
    print("SUMMARY FILE COUNTS BY MODEL")
    print("=" * 80)
    
    total_models = len(model_counts)
    total_files = 0
    
    for model_name in sorted(model_counts.keys()):
        model_data = model_counts[model_name]
        model_total = sum(sum(counts.values()) for counts in model_data.values())
        total_files += model_total
        
        print(f"\nModel: {model_name}")
        print("-" * 60)
        print(f"Total summary files: {model_total}")
        
        # Group by dataset
        for dataset_name in sorted(model_data.keys()):
            dataset_data = model_data[dataset_name]
            dataset_total = sum(dataset_data.values())
            
            print(f"  {dataset_name}: {dataset_total} files")
            
            # Show breakdown by sample count
            for sample_count in sorted(dataset_data.keys()):
                count = dataset_data[sample_count]
                print(f"    Sample count {sample_count}: {count} files")
    
    print("\n" + "=" * 80)
    print(f"OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total models: {total_models}")
    print(f"Total summary files: {total_files}")
    
    # Show models with most/least files
    model_totals = [(model, sum(sum(counts.values()) for counts in data.values())) 
                   for model, data in model_counts.items()]
    model_totals.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nModels with most files:")
    for model, count in model_totals[:5]:
        print(f"  {model}: {count} files")
    
    if len(model_totals) > 5:
        print(f"\nModels with fewest files:")
        for model, count in model_totals[-3:]:
            print(f"  {model}: {count} files")


def export_to_csv(model_counts: Dict[str, Dict[str, Dict[int, int]]], output_file: str) -> None:
    """Export the counts to a CSV file."""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Dataset', 'Sample_Count', 'File_Count'])
        
        for model_name in sorted(model_counts.keys()):
            model_data = model_counts[model_name]
            for dataset_name in sorted(model_data.keys()):
                dataset_data = model_data[dataset_name]
                for sample_count in sorted(dataset_data.keys()):
                    count = dataset_data[sample_count]
                    writer.writerow([model_name, dataset_name, sample_count, count])
    
    print(f"\nResults exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Count summary files for each model and identify missing data")
    parser.add_argument(
        "--results-dir",
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/summary_model_for_evaluation",
        help="Path to results directory (default: results/summary_model_for_evaluation)"
    )
    parser.add_argument(
        "--output-csv",
        help="Optional CSV file to export results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information about each file"
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Show only missing data analysis"
    )
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Show only file counts (skip missing data analysis)"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Scanning results directory: {args.results_dir}")
        model_counts, missing_models, missing_datasets = count_summary_files(args.results_dir)
        
        if not model_counts:
            print("No summary files found!")
            return 1
        
        # Show missing data analysis first (unless counts-only)
        if not args.counts_only:
            analyze_missing_data(model_counts, missing_models, missing_datasets)
        
        # Show detailed counts (unless missing-only)
        if not args.missing_only:
            print_summary_counts(model_counts)
        
        if args.output_csv:
            export_to_csv(model_counts, args.output_csv)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
