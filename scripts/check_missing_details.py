#!/usr/bin/env python3
"""
Script to check exactly what data is missing for specific models.
Shows detailed breakdown of missing datasets and sample counts.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set


def get_expected_combinations() -> Tuple[Set[str], Set[str], Set[int]]:
    """Get expected models, datasets, and sample counts."""
    expected_models = {
        "gpt-5-mini", "web-rev-claude-sonnet-4-20250514", "gemini-2.5-flash", 
        "deepseek-reasoner", "grok-4-latest", "gpt-oss-120b", "gpt-oss-20b", 
        "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", 
        "qwen3-30b-a3b", "qwen3-235b-a22b", "gpt-5", "qwen3-32b", 
        "web-rev-claude-opus-4-20250514", "deepseek-chat", "gemini-2.5-pro"
    }
    
    expected_datasets = {
        "Binary-Health-Care-Policy", "Binary-Online-Identity-Policies", 
        "Binary-Refugee-Policies", "Binary-Tariff-Policy", "Binary-Vaccination-Policy",
        "Openqa-AI-changes-human-life", "Openqa-Tipping-System", 
        "Openqa-Trump-cutting-funding", "Openqa-Updates-of-electronic-products", 
        "Openqa-Influencers-as-a-job"
    }
    
    expected_sample_counts = {10, 30, 50, 70, 90}
    expected_sample_times = {1, 2, 3}  # 3 sample times
    
    return expected_models, expected_datasets, expected_sample_counts, expected_sample_times


def analyze_model_missing_data(results_dir: str, target_models: List[str]) -> None:
    """Analyze missing data for specific models."""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    expected_models, expected_datasets, expected_sample_counts, expected_sample_times = get_expected_combinations()
    
    # Check if target models are valid
    invalid_models = set(target_models) - expected_models
    if invalid_models:
        print(f"❌ Invalid models: {', '.join(invalid_models)}")
        print(f"Valid models: {', '.join(sorted(expected_models))}")
        return
    
    print("=" * 80)
    print("DETAILED MISSING DATA ANALYSIS")
    print("=" * 80)
    
    for model_name in target_models:
        print(f"\n🔍 ANALYZING MODEL: {model_name}")
        print("=" * 60)
        
        # Find existing files for this model
        model_dir = results_path / model_name
        if not model_dir.exists():
            print(f"❌ Model directory does not exist: {model_dir}")
            print(f"Expected combinations: {len(expected_datasets) * len(expected_sample_counts) * len(expected_sample_times)}")
            print("Missing ALL data for this model!")
            continue
        
        # Track what we found
        found_combinations = set()
        missing_combinations = []
        
        # Check each expected combination
        for dataset in expected_datasets:
            for sample_count in expected_sample_counts:
                for sample_time in expected_sample_times:
                    # Expected file: {dataset}_summary_{sample_time}.json
                    expected_file = f"{dataset}_summary_{sample_time}.json"
                    file_path = model_dir / str(sample_count) / expected_file
                    
                    if file_path.exists():
                        # Check if file has main_points
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if 'summaries' in data and 'main_points' in data['summaries']:
                                    found_combinations.add((dataset, sample_count, sample_time))
                                else:
                                    missing_combinations.append((dataset, sample_count, sample_time, "No main_points field"))
                        except (json.JSONDecodeError, KeyError, FileNotFoundError):
                            missing_combinations.append((dataset, sample_count, sample_time, "Invalid JSON"))
                    else:
                        missing_combinations.append((dataset, sample_count, sample_time, "File not found"))
        
        # Report results
        total_expected = len(expected_datasets) * len(expected_sample_counts) * len(expected_sample_times)
        total_found = len(found_combinations)
        total_missing = len(missing_combinations)
        
        print(f"📊 SUMMARY:")
        print(f"  Found: {total_found}/{total_expected} ({total_found/total_expected*100:.1f}%)")
        print(f"  Missing: {total_missing}/{total_expected} ({total_missing/total_expected*100:.1f}%)")
        
        if missing_combinations:
            print(f"\n❌ MISSING COMBINATIONS ({total_missing}):")
            
            # Group by dataset
            missing_by_dataset = defaultdict(list)
            for dataset, sample_count, sample_time, reason in missing_combinations:
                missing_by_dataset[dataset].append((sample_count, sample_time, reason))
            
            for dataset in sorted(missing_by_dataset.keys()):
                dataset_missing = missing_by_dataset[dataset]
                print(f"\n  📁 {dataset} ({len(dataset_missing)} missing):")
                
                # Group by sample count
                missing_by_sample_count = defaultdict(list)
                for sample_count, sample_time, reason in dataset_missing:
                    missing_by_sample_count[sample_count].append((sample_time, reason))
                
                for sample_count in sorted(missing_by_sample_count.keys()):
                    sample_missing = missing_by_sample_count[sample_count]
                    sample_times = [str(st) for st, _ in sample_missing]
                    print(f"    Sample count {sample_count}: Missing sample times {', '.join(sample_times)}")
                    
                    # Show reasons if there are any issues other than "File not found"
                    unique_reasons = set(reason for _, reason in sample_missing)
                    if len(unique_reasons) > 1 or "File not found" not in unique_reasons:
                        for sample_time, reason in sample_missing:
                            if reason != "File not found":
                                print(f"      Sample {sample_time}: {reason}")
        else:
            print(f"\n✅ All expected combinations found!")
        
        # Show what we found
        if found_combinations:
            print(f"\n✅ FOUND COMBINATIONS ({total_found}):")
            
            # Group by dataset
            found_by_dataset = defaultdict(list)
            for dataset, sample_count, sample_time in found_combinations:
                found_by_dataset[dataset].append((sample_count, sample_time))
            
            for dataset in sorted(found_by_dataset.keys()):
                dataset_found = found_by_dataset[dataset]
                print(f"\n  📁 {dataset} ({len(dataset_found)} found):")
                
                # Group by sample count
                found_by_sample_count = defaultdict(list)
                for sample_count, sample_time in dataset_found:
                    found_by_sample_count[sample_count].append(sample_time)
                
                for sample_count in sorted(found_by_sample_count.keys()):
                    sample_times = sorted(found_by_sample_count[sample_count])
                    print(f"    Sample count {sample_count}: Found sample times {', '.join(map(str, sample_times))}")


def main():
    parser = argparse.ArgumentParser(description="Check detailed missing data for specific models")
    parser.add_argument(
        "--results-dir",
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/summary_model_for_evaluation",
        help="Path to results directory"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3-1.7b", "qwen3-0.6b"],
        help="Models to analyze (default: qwen3-1.7b qwen3-0.6b)"
    )
    
    args = parser.parse_args()
    
    try:
        analyze_model_missing_data(args.results_dir, args.models)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
