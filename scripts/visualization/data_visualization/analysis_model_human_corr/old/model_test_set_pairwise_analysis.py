#!/usr/bin/env python3
"""
Model Test Set Pairwise Analysis
Evaluates model consistency on test set by analyzing pairwise comparisons:
1. High/Low score consistency (which summary gets higher score)
2. Ranking consistency (relative ranking order)
"""

import json
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
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Four evaluation dimensions
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

def load_test_data():
    """Load test set data"""
    test_file = PROJECT_ROOT / 'datasets/summary_rating_dataset/split_data/test.jsonl'
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                test_data.append(data)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(test_data)} test samples")
    return test_data

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

def create_pairs_from_test_data(test_data):
    """Create pairs from test data (assuming consecutive items are pairs)"""
    pairs = []
    
    # Group by some identifier to create pairs
    # Assuming pairs are consecutive items or have some pairing logic
    for i in range(0, len(test_data), 2):
        if i + 1 < len(test_data):
            pair = {
                'pair_id': f"pair_{i//2}",
                'summary_a': test_data[i],
                'summary_b': test_data[i + 1]
            }
            pairs.append(pair)
    
    print(f"Created {len(pairs)} pairs from test data")
    return pairs

def calculate_pairwise_consistency(pairs, model_predictions):
    """Calculate pairwise consistency metrics for each model"""
    
    results = {
        'high_low_consistency': {dim: {} for dim in DIMENSIONS.keys()},
        'ranking_consistency': {dim: {} for dim in DIMENSIONS.keys()},
        'overall_high_low': {},
        'overall_ranking': {}
    }
    
    for model_name, model_data in model_predictions.items():
        print(f"Analyzing {model_name}...")
        
        # Get model predictions for test set
        model_ratings = {}
        if 'rating_results' in model_data:
            for item in model_data['rating_results']:
                if 'id' in item and 'llm_result' in item and 'ratings' in item['llm_result']:
                    model_ratings[item['id']] = item['llm_result']['ratings']
        
        if not model_ratings:
            print(f"  No model ratings found for {model_name}")
            continue
        
        # Calculate consistency for each dimension
        for dimension in DIMENSIONS.keys():
            high_low_matches = []
            ranking_matches = []
            
            for pair in pairs:
                summary_a_id = pair['summary_a'].get('id')
                summary_b_id = pair['summary_b'].get('id')
                
                if summary_a_id in model_ratings and summary_b_id in model_ratings:
                    model_a_rating = model_ratings[summary_a_id].get(dimension)
                    model_b_rating = model_ratings[summary_b_id].get(dimension)
                    
                    if model_a_rating is not None and model_b_rating is not None:
                        # High/Low consistency: which summary gets higher score
                        model_a_higher = model_a_rating > model_b_rating
                        high_low_matches.append(model_a_higher)
                        
                        # Ranking consistency: relative ranking
                        ranking_diff = model_a_rating - model_b_rating
                        ranking_matches.append(ranking_diff)
            
            # Calculate consistency metrics
            if high_low_matches:
                # High/Low consistency: percentage of consistent high/low predictions
                consistency_rate = sum(high_low_matches) / len(high_low_matches)
                results['high_low_consistency'][dimension][model_name] = consistency_rate
            else:
                results['high_low_consistency'][dimension][model_name] = np.nan
            
            if ranking_matches:
                # Ranking consistency: variance of ranking differences (lower = more consistent)
                ranking_variance = np.var(ranking_matches)
                results['ranking_consistency'][dimension][model_name] = 1.0 / (1.0 + ranking_variance)  # Convert to 0-1 scale
            else:
                results['ranking_consistency'][dimension][model_name] = np.nan
        
        # Overall consistency across all dimensions
        high_low_values = [v for v in results['high_low_consistency'].values() if model_name in v and not np.isnan(v[model_name])]
        ranking_values = [v for v in results['ranking_consistency'].values() if model_name in v and not np.isnan(v[model_name])]
        
        results['overall_high_low'][model_name] = np.mean([v[model_name] for v in high_low_values]) if high_low_values else np.nan
        results['overall_ranking'][model_name] = np.mean([v[model_name] for v in ranking_values]) if ranking_values else np.nan
    
    return results

def create_consistency_visualization(results):
    """Create visualization of pairwise consistency metrics"""
    
    # Get available models
    available_models = list(results['overall_high_low'].keys())
    if not available_models:
        print("No model data available for visualization")
        return
    
    # Order models according to MODEL_NAME_MAPPING
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
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Model Pairwise Consistency Analysis', fontsize=16, fontweight='bold')
    
    # 1. High/Low Consistency by Dimension
    high_low_matrix = np.zeros((len(models), len(dimensions)))
    for i, model in enumerate(models):
        for j, dimension in enumerate(dimensions):
            high_low_matrix[i, j] = results['high_low_consistency'][dimension].get(model, np.nan)
    
    im1 = ax1.imshow(high_low_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(dimension_names)))
    ax1.set_xticklabels(dimension_names, rotation=45, ha='right', fontsize=12)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(model_display_names, fontsize=11)
    ax1.set_title('High/Low Score Consistency by Dimension', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(dimensions)):
            value = high_low_matrix[i, j]
            if not np.isnan(value):
                ax1.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=10)
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Consistency Rate', rotation=270, labelpad=15)
    
    # 2. Ranking Consistency by Dimension
    ranking_matrix = np.zeros((len(models), len(dimensions)))
    for i, model in enumerate(models):
        for j, dimension in enumerate(dimensions):
            ranking_matrix[i, j] = results['ranking_consistency'][dimension].get(model, np.nan)
    
    im2 = ax2.imshow(ranking_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(dimension_names)))
    ax2.set_xticklabels(dimension_names, rotation=45, ha='right', fontsize=12)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(model_display_names, fontsize=11)
    ax2.set_title('Ranking Consistency by Dimension', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(dimensions)):
            value = ranking_matrix[i, j]
            if not np.isnan(value):
                ax2.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=10)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Consistency Score', rotation=270, labelpad=15)
    
    # 3. Overall High/Low Consistency
    overall_high_low = [results['overall_high_low'].get(model, np.nan) for model in models]
    bars3 = ax3.bar(range(len(models)), overall_high_low, color='skyblue', alpha=0.7)
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(model_display_names, rotation=45, ha='right', fontsize=11)
    ax3.set_ylabel('Overall High/Low Consistency', fontsize=12)
    ax3.set_title('Overall High/Low Score Consistency', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars3, overall_high_low)):
        if not np.isnan(value):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Overall Ranking Consistency
    overall_ranking = [results['overall_ranking'].get(model, np.nan) for model in models]
    bars4 = ax4.bar(range(len(models)), overall_ranking, color='lightgreen', alpha=0.7)
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(model_display_names, rotation=45, ha='right', fontsize=11)
    ax4.set_ylabel('Overall Ranking Consistency', fontsize=12)
    ax4.set_title('Overall Ranking Consistency', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars4, overall_ranking)):
        if not np.isnan(value):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = PROJECT_ROOT / 'results/analysis_model_human_corr/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'model_test_set_pairwise_consistency.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pairwise consistency visualization saved to: {output_dir / 'model_test_set_pairwise_consistency.pdf'}")

def main():
    """Main analysis function"""
    print("Starting Model Test Set Pairwise Analysis...")
    print("=" * 70)
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data()
    
    if not test_data:
        print("No test data found!")
        return
    
    # Create pairs from test data
    print("Creating pairs from test data...")
    pairs = create_pairs_from_test_data(test_data)
    
    if not pairs:
        print("No pairs created from test data!")
        return
    
    # Load model predictions
    print("Loading model predictions...")
    model_predictions = load_model_predictions()
    
    if not model_predictions:
        print("No model predictions found!")
        return
    
    # Calculate pairwise consistency
    print("Calculating pairwise consistency...")
    results = calculate_pairwise_consistency(pairs, model_predictions)
    
    # Create visualization
    print("Creating visualization...")
    create_consistency_visualization(results)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    for model in results['overall_high_low']:
        if not np.isnan(results['overall_high_low'][model]):
            print(f"\n{model}:")
            print(f"  Overall High/Low Consistency: {results['overall_high_low'][model]:.3f}")
            print(f"  Overall Ranking Consistency: {results['overall_ranking'][model]:.3f}")
            
            for dimension in DIMENSIONS.keys():
                hl_cons = results['high_low_consistency'][dimension].get(model, np.nan)
                rank_cons = results['ranking_consistency'][dimension].get(model, np.nan)
                if not np.isnan(hl_cons) and not np.isnan(rank_cons):
                    print(f"    {DIMENSIONS[dimension]}: HL={hl_cons:.3f}, Rank={rank_cons:.3f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
