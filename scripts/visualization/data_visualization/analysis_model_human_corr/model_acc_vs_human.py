#!/usr/bin/env python3
"""
Model Accuracy vs Human Evaluation Analysis - Updated for full_augment format
Evaluates model accuracy compared to human ratings and comparisons across four dimensions.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Four evaluation dimensions with standardized names (matching evaluation results format)
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
    'deepseek-chat': 'DeepSeek-V3.1(chat)'
}

def load_annotation_data():
    """Load annotation data from full_augment format"""
    annotation_dir = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full_augment'
    
    all_annotations = {
        'ratings': [],
        'comparisons': []
    }
    
    for user_dir in annotation_dir.iterdir():
        if user_dir.is_dir():
            jsonl_file = user_dir / "annotated_instances.jsonl"
            assign_file = user_dir / "assigned_user_data.json"
            
            # Load assigned data for metadata
            assigned_data = {}
            if assign_file.exists():
                with open(assign_file, 'r', encoding='utf-8') as f:
                    assigned_data = json.load(f)
            
            if jsonl_file.exists():
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            data['user_id'] = user_dir.name
                            
                            # Add metadata from assigned data
                            if data['id'] in assigned_data:
                                data.update(assigned_data[data['id']])
                            
                            if 'comparison' in data['id'] and 'label_annotations' in data:
                                all_annotations['comparisons'].append(data)
                            elif 'rating' in data['id'] and 'label_annotations' in data:
                                all_annotations['ratings'].append(data)
                                
                        except json.JSONDecodeError:
                            continue
    
    return all_annotations

def load_model_predictions():
    """Load model predictions from evaluation results"""
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

def extract_rating_value(row, dimension):
    """Extract rating value from annotation row"""
    rating_questions = {
        'perspective': "To what extent is your perspective represented in this response?",
        'informativeness': "How informative is this summary?",
        'neutrality': "Do you think this summary presents a neutral and balanced view of the issue?",
        'policy': "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue? "
    }
    
    question = rating_questions.get(dimension)
    if not question or 'label_annotations' not in row:
        return np.nan
    
    if question in row['label_annotations']:
        scales = row['label_annotations'][question]
        if isinstance(scales, dict):
            for value in scales.values():
                if value and str(value).isdigit():
                    return int(value)
        elif scales and str(scales).isdigit():
            return int(scales)
    
    return np.nan

def extract_comparison_choice(row, dimension):
    """Extract comparison choice from annotation row"""
    comparison_questions = {
        'perspective': "Which summary is more representative of your perspective? ",
        'informativeness': "Which summary is more informative? ",
        'neutrality': "Which summary presents a more neutral and balanced view of the issue? ",
        'policy': "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
    }
    
    question = comparison_questions.get(dimension)
    if not question or 'label_annotations' not in row:
        return np.nan
    
    if question in row['label_annotations']:
        scales = row['label_annotations'][question]
        if isinstance(scales, dict):
            for value in scales.values():
                if value and str(value).isdigit():
                    choice = int(value)
                    # Map 1-5 scale to A wins (1), neutral (0.5), B wins (0)
                    if choice in [1, 2]:
                        return 1.0  # A wins
                    elif choice == 3:
                        return 0.5  # Neutral
                    elif choice in [4, 5]:
                        return 0.0  # B wins
        elif scales and str(scales).isdigit():
            choice = int(scales)
            if choice in [1, 2]:
                return 1.0
            elif choice == 3:
                return 0.5
            elif choice in [4, 5]:
                return 0.0
    
    return np.nan

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
    """Calculate accuracy for comparison tasks (three-category: good/neutral/bad)"""
    if not human_comparisons or not model_comparisons:
        return np.nan
    
    total = 0
    correct = 0
    
    for dimension in DIMENSIONS.keys():
        if dimension in human_comparisons and dimension in model_comparisons:
            total += 1
            human_val = human_comparisons[dimension]
            model_val = model_comparisons[dimension]
            
            # Convert to three categories: good (1,2), neutral (3), bad (4,5)
            def to_category(val):
                if val in [1, 2]:
                    return 'good'
                elif val == 3:
                    return 'neutral'
                elif val in [4, 5]:
                    return 'bad'
                else:
                    return None
            
            human_cat = to_category(human_val)
            model_cat = to_category(model_val)
            
            if human_cat and model_cat and human_cat == model_cat:
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
            
            if task_type == 'rating':
                # For rating: exact match
                if human_data[human_key][dimension] == model_data[model_key][dimension]:
                    dimension_acc[dimension] = 1.0
                else:
                    dimension_acc[dimension] = 0.0
            else:
                # For comparison: three-category match (good/neutral/bad)
                human_val = human_data[human_key][dimension]
                model_val = model_data[model_key][dimension]
                
                def to_category(val):
                    if val in [1, 2]:
                        return 'good'
                    elif val == 3:
                        return 'neutral'
                    elif val in [4, 5]:
                        return 'bad'
                    else:
                        return None
                
                human_cat = to_category(human_val)
                model_cat = to_category(model_val)
                
                if human_cat and model_cat and human_cat == model_cat:
                    dimension_acc[dimension] = 1.0
                else:
                    dimension_acc[dimension] = 0.0
        else:
            dimension_acc[dimension] = np.nan
    
    return dimension_acc

def compute_model_accuracy(annotations, model_predictions):
    """Compute model accuracy compared to human annotations"""
    
    results = {
        'rating_overall': {},
        'rating_by_dimension': {dim: {} for dim in DIMENSIONS.keys()},
        'comparison_overall': {},
        'comparison_by_dimension': {dim: {} for dim in DIMENSIONS.keys()}
    }
    
    if not model_predictions:
        print("No model predictions found. Please ensure evaluation results are available.")
        return results
    
    for model_name, data in model_predictions.items():
        print(f"Analyzing {model_name}...")
        
        # Debug: Check available dimensions in data
        if 'rating_results' in data and data['rating_results']:
            sample_rating = data['rating_results'][0]
            if 'human_ratings' in sample_rating:
                print(f"  Available rating dimensions: {list(sample_rating['human_ratings'].keys())}")
            if 'llm_result' in sample_rating and 'ratings' in sample_rating['llm_result']:
                print(f"  Available model rating dimensions: {list(sample_rating['llm_result']['ratings'].keys())}")
        
        if 'comparison_results' in data and data['comparison_results']:
            sample_comparison = data['comparison_results'][0]
            if 'human_comparisons' in sample_comparison:
                print(f"  Available comparison dimensions: {list(sample_comparison['human_comparisons'].keys())}")
            if 'llm_result' in sample_comparison and 'comparisons' in sample_comparison['llm_result']:
                print(f"  Available model comparison dimensions: {list(sample_comparison['llm_result']['comparisons'].keys())}")
        
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
    
    # Check if we have any data
    if not results['rating_overall'] and not results['comparison_overall']:
        print("No data available for visualization. Please ensure model predictions are loaded.")
        return
    
    # Prepare data for plotting with standardized model names in specified order
    available_models = list(set(list(results['rating_overall'].keys()) + list(results['comparison_overall'].keys())))
    
    if not available_models:
        print("No models found in results.")
        return
    
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
            rating_dim_matrix[i, j] = results['rating_by_dimension'][dimension].get(model, np.nan)
    
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
            comparison_dim_matrix[i, j] = results['comparison_by_dimension'][dimension].get(model, np.nan)
    
    im2 = ax2.imshow(comparison_dim_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(dimension_names)))
    ax2.set_xticklabels(dimension_names, rotation=0, ha='center', fontsize=12)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(model_display_names, fontsize=11)
    ax2.set_title('Comparison Accuracy by Dimension (3-category: Good/Neutral/Bad)', fontsize=14, fontweight='bold')
    
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
    
    plt.savefig(output_dir / 'model_accuracy_vs_human_updated.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model accuracy visualization saved to: {output_dir / 'model_accuracy_vs_human_updated.pdf'}")

def main():
    """Main analysis function"""
    print("Starting Model Accuracy vs Human Evaluation Analysis...")
    print("=" * 70)
    
    # Load model predictions from evaluation results
    print("Loading model predictions...")
    model_predictions = load_model_predictions()
    
    if not model_predictions:
        print("No model prediction files found. Please ensure evaluation results are available.")
        return
    
    # Compute accuracy
    print("Computing model accuracy...")
    results = compute_model_accuracy({}, model_predictions)  # annotations not needed for this analysis
    
    # Create visualization
    print("Creating visualization...")
    create_accuracy_visualization(results)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
