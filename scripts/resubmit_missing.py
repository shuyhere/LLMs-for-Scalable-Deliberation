#!/usr/bin/env python3
"""
Script to resubmit missing data jobs using the sbatch_gen.sh script.
This script analyzes missing data and generates targeted resubmission commands.
"""

import os
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
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
    expected_sample_times = {1, 2, 3}
    
    return expected_models, expected_datasets, expected_sample_counts, expected_sample_times


def find_missing_combinations(results_dir: str, target_models: List[str]) -> List[Tuple[str, str, int, int]]:
    """
    Find missing combinations for specific models.
    
    Returns:
        List of (model, dataset, sample_count, sample_time) tuples
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    expected_models, expected_datasets, expected_sample_counts, expected_sample_times = get_expected_combinations()
    
    missing_combinations = []
    
    for model_name in target_models:
        if model_name not in expected_models:
            print(f"Warning: {model_name} is not in expected models list")
            continue
            
        model_dir = results_path / model_name
        if not model_dir.exists():
            print(f"Model directory does not exist: {model_dir}")
            # All combinations are missing
            for dataset in expected_datasets:
                for sample_count in expected_sample_counts:
                    for sample_time in expected_sample_times:
                        missing_combinations.append((model_name, dataset, sample_count, sample_time))
            continue
        
        # Check each expected combination
        for dataset in expected_datasets:
            for sample_count in expected_sample_counts:
                for sample_time in expected_sample_times:
                    expected_file = f"{dataset}_summary_{sample_time}.json"
                    file_path = model_dir / str(sample_count) / expected_file
                    
                    if not file_path.exists():
                        missing_combinations.append((model_name, dataset, sample_count, sample_time))
                    else:
                        # Check if file has main_points
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if 'summaries' not in data or 'main_points' not in data['summaries']:
                                    missing_combinations.append((model_name, dataset, sample_count, sample_time))
                        except (json.JSONDecodeError, KeyError, FileNotFoundError):
                            missing_combinations.append((model_name, dataset, sample_count, sample_time))
    
    return missing_combinations


def generate_resubmission_commands(missing_combinations: List[Tuple[str, str, int, int]], 
                                 script_path: str, 
                                 output_dir: str,
                                 dry_run: bool = False) -> List[str]:
    """Generate resubmission commands for missing combinations."""
    
    # Group by model for efficient resubmission
    model_groups = {}
    for model, dataset, sample_count, sample_time in missing_combinations:
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append((dataset, sample_count, sample_time))
    
    commands = []
    
    for model, combinations in model_groups.items():
        # Get unique datasets and sample counts for this model
        datasets = sorted(set(dataset for dataset, _, _ in combinations))
        sample_counts = sorted(set(sample_count for _, sample_count, _ in combinations))
        
        # Create command
        cmd_parts = [
            "bash", script_path,
            "--models", model,
            "--num-samples", ",".join(map(str, sample_counts)),
            "--output-dir", output_dir,
            "--skip-existing"
        ]
        
        if dry_run:
            cmd_parts.append("--dry-run")
        
        # Add specific datasets if not all are missing
        if len(datasets) < 10:  # Less than all datasets
            dataset_paths = []
            for dataset in datasets:
                dataset_paths.append(f"datasets/cleaned_new_dataset/{dataset}.json")
            cmd_parts.extend(["--datasets", ",".join(dataset_paths)])
        
        commands.append(" ".join(cmd_parts))
    
    return commands


def main():
    parser = argparse.ArgumentParser(description="Resubmit missing data jobs")
    parser.add_argument(
        "--results-dir",
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/summary_model_for_evaluation",
        help="Path to results directory"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3-1.7b", "qwen3-0.6b"],
        help="Models to resubmit missing data for"
    )
    parser.add_argument(
        "--script-path",
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/scripts/sbatch_scripts/sbatch_gen.sh",
        help="Path to sbatch_gen.sh script"
    )
    parser.add_argument(
        "--output-dir",
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/summary_model_for_evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing them"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the resubmission commands"
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 80)
        print("MISSING DATA RESUBMISSION")
        print("=" * 80)
        print(f"Results directory: {args.results_dir}")
        print(f"Target models: {', '.join(args.models)}")
        print(f"Script path: {args.script_path}")
        print(f"Output directory: {args.output_dir}")
        print("=" * 80)
        
        # Find missing combinations
        print("\nAnalyzing missing data...")
        missing_combinations = find_missing_combinations(args.results_dir, args.models)
        
        if not missing_combinations:
            print("✅ No missing data found!")
            return 0
        
        print(f"Found {len(missing_combinations)} missing combinations")
        
        # Show summary by model
        model_counts = {}
        for model, dataset, sample_count, sample_time in missing_combinations:
            if model not in model_counts:
                model_counts[model] = 0
            model_counts[model] += 1
        
        print("\nMissing combinations by model:")
        for model, count in sorted(model_counts.items()):
            print(f"  {model}: {count} missing")
        
        # Generate resubmission commands
        print("\nGenerating resubmission commands...")
        commands = generate_resubmission_commands(
            missing_combinations, 
            args.script_path, 
            args.output_dir,
            args.dry_run
        )
        
        print(f"\nGenerated {len(commands)} resubmission commands:")
        for i, cmd in enumerate(commands, 1):
            print(f"\n{i}. {cmd}")
        
        if args.execute:
            print(f"\n{'='*80}")
            print("EXECUTING RESUBMISSION COMMANDS")
            print(f"{'='*80}")
            
            for i, cmd in enumerate(commands, 1):
                print(f"\nExecuting command {i}/{len(commands)}:")
                print(f"Command: {cmd}")
                
                try:
                    result = subprocess.run(cmd, shell=True, check=True, 
                                          capture_output=True, text=True)
                    print("✅ Command executed successfully")
                    if result.stdout:
                        print("Output:", result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Command failed with exit code {e.returncode}")
                    if e.stdout:
                        print("Output:", e.stdout)
                    if e.stderr:
                        print("Error:", e.stderr)
                except Exception as e:
                    print(f"❌ Error executing command: {e}")
            
            print(f"\n{'='*80}")
            print("RESUBMISSION COMPLETED")
            print(f"{'='*80}")
        else:
            print(f"\nTo execute these commands, run with --execute flag")
            print(f"To see what would be submitted, run with --dry-run flag")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
