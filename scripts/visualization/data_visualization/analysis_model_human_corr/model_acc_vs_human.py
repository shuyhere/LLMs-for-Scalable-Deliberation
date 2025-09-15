#!/usr/bin/env python3
"""
Model Accuracy vs Human Evaluation Analysis
Evaluates model accuracy compared to human ratings and comparisons across four dimensions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Project root directory
PROJECT_ROOT = Path('/ibex/project/c2328/LLMs-Scalable-Deliberation')

# Four evaluation dimensions with standardized names
DIMENSIONS = {
    'perspective_representation': 'Representiveness',
    'informativeness': 'Informativeness', 
    'neutrality_balance': 'Neutrality',
    'policy_approval': 'Policy Approval'
}

# Model name mapping for consistent display
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

def load_model_data():
    """Load data from all non-checkpoint model files"""
    data_dir = PROJECT_ROOT / 'results/eval_llm_human_correlation'
    model_data = {}
    
    # Get all JSON files that don't start with 'checkpoint_'
    json_files = [f for f in data_dir.glob('*.json') if not f.name.startswith('checkpoint_')]
    
    print(f"Found {len(json_files)} model files to process")
    
    for json_file in json_files:
        model_name = json_file.stem.replace('human_llm_correlation_', '')
        print(f"Loading {model_name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            model_data[model_name] = data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return model_data

def calculate_rating_accuracy(human_ratings, model_ratings):
    """Calculate accuracy for rating tasks (exact match)"""
    if not human_ratings or not model_ratings:
        return np.nan
    
    total = 0
    correct = 0
    
    for dimension in DIMENSIONS.keys():
        if dimension in human_ratings and dimension in model_ratings:
            total += 1
            if human_ratings[dimension] == model_ratings[dimension]:
                correct += 1
    
    return correct / total if total > 0 else np.nan

def calculate_comparison_accuracy(human_comparisons, model_comparisons):
    """Calculate accuracy for comparison tasks (exact match)"""
    if not human_comparisons or not model_comparisons:
        return np.nan
    
    total = 0
    correct = 0
    
    for dimension in DIMENSIONS.keys():
        if dimension in human_comparisons and dimension in model_comparisons:
            total += 1
            if human_comparisons[dimension] == model_comparisons[dimension]:
                correct += 1
    
    return correct / total if total > 0 else np.nan

def calculate_dimension_wise_accuracy(human_data, model_data, task_type):
    """Calculate accuracy for each dimension separately"""
    dimension_acc = {}
    
    for dimension in DIMENSIONS.keys():
        if task_type == 'rating':
            human_key = 'human_ratings'
            model_key = 'ratings'
        else:  # comparison
            human_key = 'human_comparisons'
            model_key = 'comparisons'
        
        if (human_key in human_data and model_key in model_data and
            dimension in human_data[human_key] and dimension in model_data[model_key]):
            
            if human_data[human_key][dimension] == model_data[model_key][dimension]:
                dimension_acc[dimension] = 1.0
            else:
                dimension_acc[dimension] = 0.0
        else:
            dimension_acc[dimension] = np.nan
    
    return dimension_acc

def analyze_model_accuracy(model_data):
    """Analyze accuracy for all models"""
    results = {
        'rating_overall': {},
        'comparison_overall': {},
        'rating_by_dimension': defaultdict(dict),
        'comparison_by_dimension': defaultdict(dict)
    }
    
    for model_name, data in model_data.items():
        print(f"Analyzing {model_name}...")
        
        # Rating accuracy
        rating_accuracies = []
        for item in data.get('rating_results', []):
            if 'human_ratings' in item and 'llm_result' in item and 'ratings' in item['llm_result']:
                acc = calculate_rating_accuracy(item['human_ratings'], item['llm_result']['ratings'])
                if not np.isnan(acc):
                    rating_accuracies.append(acc)
        
        results['rating_overall'][model_name] = np.mean(rating_accuracies) if rating_accuracies else np.nan
        
        # Comparison accuracy
        comparison_accuracies = []
        for item in data.get('comparison_results', []):
            if 'human_comparisons' in item and 'llm_result' in item and 'comparisons' in item['llm_result']:
                acc = calculate_comparison_accuracy(item['human_comparisons'], item['llm_result']['comparisons'])
                if not np.isnan(acc):
                    comparison_accuracies.append(acc)
        
        results['comparison_overall'][model_name] = np.mean(comparison_accuracies) if comparison_accuracies else np.nan
        
        # Dimension-wise accuracy for ratings
        for dimension in DIMENSIONS.keys():
            dim_accuracies = []
            for item in data.get('rating_results', []):
                if 'human_ratings' in item and 'llm_result' in item and 'ratings' in item['llm_result']:
                    dim_acc = calculate_dimension_wise_accuracy(item, item['llm_result'], 'rating')
                    if not np.isnan(dim_acc[dimension]):
                        dim_accuracies.append(dim_acc[dimension])
            
            results['rating_by_dimension'][dimension][model_name] = np.mean(dim_accuracies) if dim_accuracies else np.nan
        
        # Dimension-wise accuracy for comparisons
        for dimension in DIMENSIONS.keys():
            dim_accuracies = []
            for item in data.get('comparison_results', []):
                if 'human_comparisons' in item and 'llm_result' in item and 'comparisons' in item['llm_result']:
                    dim_acc = calculate_dimension_wise_accuracy(item, item['llm_result'], 'comparison')
                    if not np.isnan(dim_acc[dimension]):
                        dim_accuracies.append(dim_acc[dimension])
            
            results['comparison_by_dimension'][dimension][model_name] = np.mean(dim_accuracies) if dim_accuracies else np.nan
    
    return results

def create_accuracy_visualization(results):
    """Create visualization of model accuracy vs human evaluation"""
    
    # Prepare data for plotting with standardized model names in specified order
    available_models = list(results['rating_overall'].keys())
    
    # Order models according to MODEL_NAME_MAPPING order
    models = []
    model_display_names = []
    for model_key, display_name in MODEL_NAME_MAPPING.items():
        if model_key in available_models:
            models.append(model_key)
            model_display_names.append(display_name)
    
    # Add any remaining models not in the mapping
    for model in available_models:
        if model not in models:
            models.append(model)
            model_display_names.append(model)
    
    dimensions = list(DIMENSIONS.keys())
    dimension_names = list(DIMENSIONS.values())
    
    # Create figure with only 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Model Accuracy vs Human Evaluation by Dimension', fontsize=16, fontweight='bold')
    
    # 1. Rating Accuracy by Dimension (Heatmap)
    rating_dim_matrix = np.zeros((len(models), len(dimensions)))
    
    for i, model in enumerate(models):
        for j, dimension in enumerate(dimensions):
            rating_dim_matrix[i, j] = results['rating_by_dimension'][dimension][model]
    
    im1 = ax1.imshow(rating_dim_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(dimension_names)))
    ax1.set_xticklabels(dimension_names, rotation=0, ha='center', fontsize=12)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(model_display_names, fontsize=11)
    ax1.set_title('Rating Accuracy by Dimension', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(dimensions)):
            value = rating_dim_matrix[i, j]
            if not np.isnan(value):
                ax1.text(j, i, f'{value:.2f}', ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=10)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Accuracy', rotation=270, labelpad=15)
    
    # 2. Comparison Accuracy by Dimension (Heatmap)
    comparison_dim_matrix = np.zeros((len(models), len(dimensions)))
    
    for i, model in enumerate(models):
        for j, dimension in enumerate(dimensions):
            comparison_dim_matrix[i, j] = results['comparison_by_dimension'][dimension][model]
    
    im2 = ax2.imshow(comparison_dim_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(dimension_names)))
    ax2.set_xticklabels(dimension_names, rotation=0, ha='center', fontsize=12)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(model_display_names, fontsize=11)
    ax2.set_title('Comparison Accuracy by Dimension', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(dimensions)):
            value = comparison_dim_matrix[i, j]
            if not np.isnan(value):
                ax2.text(j, i, f'{value:.2f}', ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=10)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Accuracy', rotation=270, labelpad=15)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_dir = PROJECT_ROOT / 'results/analysis_model_human_corr/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'model_accuracy_vs_human.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model accuracy visualization saved to: {output_dir / 'model_accuracy_vs_human.pdf'}")

def main():
    """Main analysis function"""
    print("Starting Model Accuracy vs Human Evaluation Analysis...")
    print("=" * 70)
    
    # Load model data
    print("Loading model data...")
    model_data = load_model_data()
    
    if not model_data:
        print("Error: No model data found!")
        return
    
    print(f"Loaded data for {len(model_data)} models")
    
    # Analyze accuracy
    print("\nAnalyzing model accuracy...")
    results = analyze_model_accuracy(model_data)
    
    # Create visualization
    print("\nCreating accuracy visualization...")
    create_accuracy_visualization(results)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
