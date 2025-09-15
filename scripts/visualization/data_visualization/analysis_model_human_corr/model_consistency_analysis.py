#!/usr/bin/env python3
"""
Model Consistency Analysis Script

This script calculates inter-model consistency across four dimensions by comparing
LLM results for the same annotator_answer across different models:
- Representiveness (perspective_representation)
- Informativeness (informativeness) 
- Neutrality (neutrality_balance)
- Policy Approval (policy_approval)

For rating tasks: Uses Spearman correlation to compare LLM ratings (1-5 scale)
For comparison tasks: Uses Phi coefficient to compare LLM comparisons (1-2 binary)

Matching is done by annotator_answer field, which contains the human annotator's
opinion text. The script normalizes whitespace for consistent matching.

Outputs two heatmaps showing model-to-model consistency based on LLM outputs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy.stats import spearmanr
import argparse

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Dimension mapping
DIMENSIONS = {
    'perspective_representation': 'Representiveness',
    'informativeness': 'Informativeness', 
    'neutrality_balance': 'Neutrality',
    'policy_approval': 'Policy Approval'
}

# Model name mapping for consistent display (ordered by preference)
MODEL_NAME_MAPPING = {
    'web-rev-claude-sonnet-4-20250514': 'Claude-Sonnet-4',
    'web-rev-claude-opus-4-20250514': 'Claude-Opus-4',
    'web-rev-claude-3-7-sonnet-20250219': 'Claude-Sonnet-3.7',
    'qwen3-235b-a22b': 'Qwen3-235B',
    'qwen3-32b': 'Qwen3-32B',
    'qwen3-30b-a3b': 'Qwen3-30B',
    'qwen3-14b': 'Qwen3-14B',
    'qwen3-8b': 'Qwen3-8B',
    'qwen3-4b': 'Qwen3-4B',
    'qwen3-1.7b': 'Qwen3-1.7B',
    'qwen3-0.6b': 'Qwen3-0.6B',
    'gpt-5': 'GPT-5',
    'gpt-5-mini': 'GPT-5-Mini',
    'gpt-5-nano': 'GPT-5-Nano',
    'gpt-4o-mini': 'GPT-4o-Mini',
    'gemini-2.5-pro': 'Gemini-2.5-Pro',
    'gemini-2.5-flash': 'Gemini-2.5-Flash',
    'gemini-2.5-flash-lite': 'Gemini-2.5-Flash-Lite',
    'grok-4-latest': 'Grok-4-Latest',
    'deepseek-chat': 'DeepSeek-Chat',
    'deepseek-reasoner': 'DeepSeek-Reasoner'
}

def get_model_display_name(model_name: str) -> str:
    """Get display name for model"""
    return MODEL_NAME_MAPPING.get(model_name, model_name)

def load_model_data(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load model data from JSON files"""
    model_data = {}
    
    # Get all JSON files (excluding checkpoints)
    json_files = list(data_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("checkpoint_")]
    
    print(f"Found {len(json_files)} model files")
    
    for json_file in json_files:
        model_name = json_file.stem.replace("human_llm_correlation_", "")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            model_data[model_name] = data
            print(f"Loaded data for {model_name}: {len(data.get('rating_results', []))} rating results, {len(data.get('comparison_results', []))} comparison results")
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return model_data

def calculate_phi_coefficient(x: List[int], y: List[int]) -> float:
    """Calculate Phi coefficient for binary data"""
    if len(x) != len(y):
        return np.nan
    
    # Convert to binary (1,2 -> 0,1)
    x_binary = [1 if val == 2 else 0 for val in x]
    y_binary = [1 if val == 2 else 0 for val in y]
    
    # Create contingency table
    n = len(x_binary)
    n11 = sum(1 for i in range(n) if x_binary[i] == 1 and y_binary[i] == 1)
    n10 = sum(1 for i in range(n) if x_binary[i] == 1 and y_binary[i] == 0)
    n01 = sum(1 for i in range(n) if x_binary[i] == 0 and y_binary[i] == 1)
    n00 = sum(1 for i in range(n) if x_binary[i] == 0 and y_binary[i] == 0)
    
    # Calculate Phi coefficient
    if n == 0:
        return np.nan
    
    phi = (n11 * n00 - n10 * n01) / np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    
    return phi if not np.isnan(phi) and not np.isinf(phi) else np.nan


def calculate_model_consistency(model_data: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Calculate model-to-model consistency matrices and model-to-human consistency based on LLM results for same unique_id"""
    
    models = list(model_data.keys())
    n_models = len(models)
    
    # Initialize matrices (add one extra column for human comparison)
    rating_consistency = {dim: np.full((n_models, n_models + 1), np.nan) for dim in DIMENSIONS.keys()}
    comparison_consistency = {dim: np.full((n_models, n_models + 1), np.nan) for dim in DIMENSIONS.keys()}
    rating_human_consistency = {dim: np.full(n_models, np.nan) for dim in DIMENSIONS.keys()}
    comparison_human_consistency = {dim: np.full(n_models, np.nan) for dim in DIMENSIONS.keys()}
    
    # Pre-process data to create annotator_answer -> LLM results and human results mapping for faster lookup
    print("Pre-processing data for faster matching...")
    rating_lookup = {}
    comparison_lookup = {}
    human_rating_lookup = {}
    human_comparison_lookup = {}
    
    for model_name, data in model_data.items():
        rating_lookup[model_name] = {}
        comparison_lookup[model_name] = {}
        human_rating_lookup[model_name] = {}
        human_comparison_lookup[model_name] = {}
        
        # Process rating results - use annotator_answer as key
        rating_count = 0
        for result in data.get('rating_results', []):
            annotator_answer = result.get('annotator_answer')
            if annotator_answer and result.get('llm_result', {}).get('status') == 'success':
                # Normalize annotator_answer for consistent matching
                normalized_answer = ' '.join(annotator_answer.strip().split())
                if normalized_answer not in rating_lookup[model_name]:
                    rating_lookup[model_name][normalized_answer] = {}
                    human_rating_lookup[model_name][normalized_answer] = {}
                llm_ratings = result['llm_result'].get('ratings', {})
                human_ratings = result.get('human_ratings', {})
                for dim in DIMENSIONS.keys():
                    if dim in llm_ratings:
                        rating_lookup[model_name][normalized_answer][dim] = llm_ratings[dim]
                        rating_count += 1
                    if dim in human_ratings:
                        human_rating_lookup[model_name][normalized_answer][dim] = human_ratings[dim]
        
        # Process comparison results - use annotator_answer as key
        comparison_count = 0
        for result in data.get('comparison_results', []):
            annotator_answer = result.get('annotator_answer')
            if annotator_answer and result.get('llm_result', {}).get('status') == 'success':
                # Normalize annotator_answer for consistent matching
                normalized_answer = ' '.join(annotator_answer.strip().split())
                if normalized_answer not in comparison_lookup[model_name]:
                    comparison_lookup[model_name][normalized_answer] = {}
                    human_comparison_lookup[model_name][normalized_answer] = {}
                llm_comparisons = result['llm_result'].get('comparisons', {})
                human_comparisons = result.get('human_comparisons', {})
                for dim in DIMENSIONS.keys():
                    if dim in llm_comparisons:
                        comparison_lookup[model_name][normalized_answer][dim] = llm_comparisons[dim]
                        comparison_count += 1
                    if dim in human_comparisons:
                        human_comparison_lookup[model_name][normalized_answer][dim] = human_comparisons[dim]
        
        print(f"  {model_name}: {len(rating_lookup[model_name])} unique rating samples, {len(comparison_lookup[model_name])} unique comparison samples")
    
    print("Calculating consistency matrices...")
    total_pairs = n_models * n_models
    current_pair = 0
    
    # Calculate consistency for each model pair
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            current_pair += 1
            if current_pair % 10 == 0:
                print(f"  Processing pair {current_pair}/{total_pairs}: {model1} vs {model2}")
            
            if i == j:
                # Same model - perfect consistency
                for dim in DIMENSIONS.keys():
                    rating_consistency[dim][i, j] = 1.0
                    comparison_consistency[dim][i, j] = 1.0
            else:
                # Different models - calculate consistency
                for dim in DIMENSIONS.keys():
                    # Rating consistency (Spearman correlation) - compare LLM ratings
                    common_ratings1, common_ratings2 = find_common_samples_fast(
                        rating_lookup[model1], rating_lookup[model2], dim
                    )
                    
                    if len(common_ratings1) > 1:
                        try:
                            corr, _ = spearmanr(common_ratings1, common_ratings2)
                            rating_consistency[dim][i, j] = corr if not np.isnan(corr) else np.nan
                        except:
                            rating_consistency[dim][i, j] = np.nan
                    else:
                        rating_consistency[dim][i, j] = np.nan
                        if len(common_ratings1) == 0 and current_pair <= 5:  # Only debug first few pairs
                            debug_common_samples(rating_lookup[model1], rating_lookup[model2], model1, model2, dim)
                    
                    # Comparison consistency (Phi coefficient) - compare LLM comparisons
                    common_comparisons1, common_comparisons2 = find_common_samples_fast(
                        comparison_lookup[model1], comparison_lookup[model2], dim
                    )
                    
                    if len(common_comparisons1) > 1:
                        try:
                            phi = calculate_phi_coefficient(common_comparisons1, common_comparisons2)
                            comparison_consistency[dim][i, j] = phi if not np.isnan(phi) else np.nan
                        except:
                            comparison_consistency[dim][i, j] = np.nan
                    else:
                        comparison_consistency[dim][i, j] = np.nan
                        if len(common_comparisons1) == 0 and current_pair <= 5:  # Only debug first few pairs
                            debug_common_samples(comparison_lookup[model1], comparison_lookup[model2], model1, model2, dim)
    
    # Calculate model-to-human consistency
    print("Calculating model-to-human consistency...")
    for i, model in enumerate(models):
        for dim in DIMENSIONS.keys():
            # Rating consistency with human
            common_ratings_llm, common_ratings_human = find_common_samples_fast(
                rating_lookup[model], human_rating_lookup[model], dim
            )
            
            if len(common_ratings_llm) > 1:
                try:
                    corr, _ = spearmanr(common_ratings_llm, common_ratings_human)
                    rating_human_consistency[dim][i] = corr if not np.isnan(corr) else np.nan
                    rating_consistency[dim][i, n_models] = corr if not np.isnan(corr) else np.nan
                except:
                    rating_human_consistency[dim][i] = np.nan
                    rating_consistency[dim][i, n_models] = np.nan
            else:
                rating_human_consistency[dim][i] = np.nan
                rating_consistency[dim][i, n_models] = np.nan
            
            # Comparison consistency with human
            common_comparisons_llm, common_comparisons_human = find_common_samples_fast(
                comparison_lookup[model], human_comparison_lookup[model], dim
            )
            
            if len(common_comparisons_llm) > 1:
                try:
                    phi = calculate_phi_coefficient(common_comparisons_llm, common_comparisons_human)
                    comparison_human_consistency[dim][i] = phi if not np.isnan(phi) else np.nan
                    comparison_consistency[dim][i, n_models] = phi if not np.isnan(phi) else np.nan
                except:
                    comparison_human_consistency[dim][i] = np.nan
                    comparison_consistency[dim][i, n_models] = np.nan
            else:
                comparison_human_consistency[dim][i] = np.nan
                comparison_consistency[dim][i, n_models] = np.nan
    
    return rating_consistency, comparison_consistency, rating_human_consistency, comparison_human_consistency

def find_common_samples_fast(lookup1: Dict[str, Dict[str, float]], 
                            lookup2: Dict[str, Dict[str, float]], 
                            dimension: str) -> Tuple[List[float], List[float]]:
    """Fast version of finding common samples using pre-processed lookup tables based on annotator_answer"""
    common_samples1 = []
    common_samples2 = []
    
    # Find common annotator_answers
    common_answers = set(lookup1.keys()) & set(lookup2.keys())
    
    for answer in common_answers:
        if (dimension in lookup1[answer] and 
            dimension in lookup2[answer] and
            lookup1[answer][dimension] is not None and 
            lookup2[answer][dimension] is not None):
            common_samples1.append(lookup1[answer][dimension])
            common_samples2.append(lookup2[answer][dimension])
    
    return common_samples1, common_samples2

def debug_common_samples(lookup1: Dict[str, Dict[str, float]], 
                        lookup2: Dict[str, Dict[str, float]], 
                        model1: str, model2: str, dimension: str):
    """Debug function to check common samples between two models"""
    common_answers = set(lookup1.keys()) & set(lookup2.keys())
    print(f"    Debug {model1} vs {model2} in {dimension}:")
    print(f"      Model1 has {len(lookup1)} annotator_answers, Model2 has {len(lookup2)} annotator_answers")
    print(f"      Common annotator_answers: {len(common_answers)}")
    
    if len(common_answers) > 0:
        sample_common_answers = list(common_answers)[:3]  # Show first 3
        for answer in sample_common_answers:
            val1 = lookup1[answer].get(dimension, "MISSING")
            val2 = lookup2[answer].get(dimension, "MISSING")
            print(f"      Sample answer: {answer[:50]}... -> {val1} vs {val2}")
    else:
        # Show sample answers from each model
        sample1 = list(lookup1.keys())[:2] if lookup1 else []
        sample2 = list(lookup2.keys())[:2] if lookup2 else []
        print(f"      Sample from {model1}: {[ans[:30] + '...' for ans in sample1]}")
        print(f"      Sample from {model2}: {[ans[:30] + '...' for ans in sample2]}")
    
    return common_answers


def create_consistency_heatmaps(rating_consistency: Dict[str, np.ndarray], 
                               comparison_consistency: Dict[str, np.ndarray],
                               rating_human_consistency: Dict[str, np.ndarray],
                               comparison_human_consistency: Dict[str, np.ndarray],
                               models: List[str],
                               output_dir: Path):
    """Create heatmaps for model consistency"""
    
    # Get model display names in consistent order based on MODEL_NAME_MAPPING
    model_display_names = []
    ordered_models = []
    
    # First, add models in the order defined in MODEL_NAME_MAPPING
    for model_key in MODEL_NAME_MAPPING.keys():
        if model_key in models:
            ordered_models.append(model_key)
            model_display_names.append(get_model_display_name(model_key))
    
    # Then add any remaining models not in the mapping
    for model in models:
        if model not in ordered_models:
            ordered_models.append(model)
            model_display_names.append(get_model_display_name(model))
    
    # Add human column
    model_display_names.append('Human')
    
    # Reorder matrices to match the ordered models
    model_indices = {model: i for i, model in enumerate(models)}
    new_indices = [model_indices[model] for model in ordered_models]
    
    # Reorder rating consistency matrices (keep human column at the end)
    for dim in rating_consistency:
        # Reorder model-to-model part
        model_part = rating_consistency[dim][np.ix_(new_indices, new_indices)]
        # Keep human column at the end
        human_col = rating_consistency[dim][new_indices, -1].reshape(-1, 1)
        rating_consistency[dim] = np.hstack([model_part, human_col])
    
    # Reorder comparison consistency matrices (keep human column at the end)
    for dim in comparison_consistency:
        # Reorder model-to-model part
        model_part = comparison_consistency[dim][np.ix_(new_indices, new_indices)]
        # Keep human column at the end
        human_col = comparison_consistency[dim][new_indices, -1].reshape(-1, 1)
        comparison_consistency[dim] = np.hstack([model_part, human_col])
    
    # Create rating consistency heatmap
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle('Model Consistency - LLM Rating Tasks (Spearman Correlation)', fontsize=18, fontweight='bold')
    
    for idx, (dim, matrix) in enumerate(rating_consistency.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Create heatmap
        ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(model_display_names)))
        ax.set_yticks(range(len(ordered_models)))
        ax.set_xticklabels(model_display_names, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels([model_display_names[i] for i in range(len(ordered_models))], fontsize=12)
        
        # Add text annotations
        for i in range(len(ordered_models)):
            for j in range(len(model_display_names)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha="center", va="center", color="white" if abs(matrix[i, j]) > 0.5 else "black",
                           fontsize=10)
        
        ax.set_title(f'{DIMENSIONS[dim]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_consistency_rating.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison consistency heatmap with larger size
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle('Model Consistency - LLM Comparison Tasks (Phi Coefficient)', fontsize=18, fontweight='bold')
    
    for idx, (dim, matrix) in enumerate(comparison_consistency.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Create heatmap
        ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(model_display_names)))
        ax.set_yticks(range(len(ordered_models)))
        ax.set_xticklabels(model_display_names, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels([model_display_names[i] for i in range(len(ordered_models))], fontsize=12)
        
        # Add text annotations
        for i in range(len(ordered_models)):
            for j in range(len(model_display_names)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha="center", va="center", color="white" if abs(matrix[i, j]) > 0.5 else "black",
                           fontsize=10)
        
        ax.set_title(f'{DIMENSIONS[dim]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_consistency_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate model consistency analysis')
    parser.add_argument('--data_dir', type=str, 
                       default=str(PROJECT_ROOT / 'results' / 'eval_llm_human_correlation'),
                       help='Directory containing model correlation data')
    parser.add_argument('--output_dir', type=str,
                       default=str(PROJECT_ROOT / 'results' / 'analysis_model_human_corr' / 'figures'),
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model data...")
    model_data = load_model_data(data_dir)
    
    if not model_data:
        print("No model data found!")
        return
    
    print(f"Loaded data for {len(model_data)} models")
    
    # Calculate model consistency
    print("Calculating model consistency...")
    rating_consistency, comparison_consistency, rating_human_consistency, comparison_human_consistency = calculate_model_consistency(model_data)
    
    # Create heatmaps
    print("Creating consistency heatmaps...")
    create_consistency_heatmaps(rating_consistency, comparison_consistency, 
                               rating_human_consistency, comparison_human_consistency,
                               list(model_data.keys()), output_dir)
    
    print(f"Consistency analysis complete! Figures saved to {output_dir}")
    print("Generated files:")
    print("  - model_consistency_rating.pdf")
    print("  - model_consistency_comparison.pdf")

if __name__ == "__main__":
    main()
