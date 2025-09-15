#!/usr/bin/env python3
"""
Model Test Set Accuracy Analysis
Evaluates model accuracy on the specific test set by matching test cases with model results.
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

def load_test_set():
    """Load the test set from JSONL file"""
    test_file = PROJECT_ROOT / 'datasets/sft_annotation_format/alpace/comparison_alpaca/test.jsonl'
    test_data = []
    
    print(f"Loading test set from: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                test_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(test_data)} test cases")
    return test_data

def extract_test_case_info(test_case):
    """Extract key information from test case for matching"""
    instruction = test_case.get('instruction', '')
    output = test_case.get('output', '')
    
    # Extract the annotator's opinion from instruction
    annotator_opinion = ""
    if "One annotator's opinion on this question is:" in instruction:
        start_marker = "One annotator's opinion on this question is:"
        end_marker = "\n\n\nTwo summaries of all people's opinions are shown below."
        start_idx = instruction.find(start_marker) + len(start_marker)
        end_idx = instruction.find(end_marker)
        if start_idx > len(start_marker) - 1 and end_idx > start_idx:
            annotator_opinion = instruction[start_idx:end_idx].strip()
    
    # Parse the expected output (ground truth)
    try:
        expected_output = json.loads(output)
    except json.JSONDecodeError:
        expected_output = {}
    
    return {
        'instruction': instruction,
        'annotator_opinion': annotator_opinion,
        'expected_output': expected_output,
        'full_instruction': instruction
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

def match_test_cases_with_model_results(test_data, model_data):
    """Match test cases with model results based on annotator opinion"""
    matched_results = {}
    
    for model_name, model_results in model_data.items():
        print(f"Matching test cases for {model_name}...")
        matched_results[model_name] = {
            'rating_matches': [],
            'comparison_matches': []
        }
        
        # Extract test case info
        test_cases_info = [extract_test_case_info(test_case) for test_case in test_data]
        
        # Try to match with rating results
        for i, test_case_info in enumerate(test_cases_info):
            if not test_case_info['annotator_opinion']:
                continue
                
            best_match = None
            best_similarity = 0
            
            for rating_result in model_results.get('rating_results', []):
                # Check if the annotator answer matches
                model_annotator_answer = rating_result.get('annotator_answer', '')
                test_annotator_opinion = test_case_info['annotator_opinion']
                
                # Simple similarity check based on annotator opinion
                similarity = calculate_annotator_similarity(test_annotator_opinion, model_annotator_answer)
                
                if similarity > best_similarity and similarity >= 1.0:  # Exact matching only
                    best_similarity = similarity
                    best_match = rating_result
            
            if best_match:
                matched_results[model_name]['rating_matches'].append({
                    'test_case': test_case_info,
                    'model_result': best_match,
                    'similarity': best_similarity
                })
        
        # Try to match with comparison results
        for i, test_case_info in enumerate(test_cases_info):
            if not test_case_info['annotator_opinion']:
                continue
                
            best_match = None
            best_similarity = 0
            
            for comparison_result in model_results.get('comparison_results', []):
                # Check if the annotator answer matches
                model_annotator_answer = comparison_result.get('annotator_answer', '')
                test_annotator_opinion = test_case_info['annotator_opinion']
                
                # Simple similarity check based on annotator opinion
                similarity = calculate_annotator_similarity(test_annotator_opinion, model_annotator_answer)
                
                if similarity > best_similarity and similarity >= 1.0:  # Exact matching only
                    best_similarity = similarity
                    best_match = comparison_result
            
            if best_match:
                matched_results[model_name]['comparison_matches'].append({
                    'test_case': test_case_info,
                    'model_result': best_match,
                    'similarity': best_similarity
                })
        
        print(f"  Found {len(matched_results[model_name]['rating_matches'])} rating matches")
        print(f"  Found {len(matched_results[model_name]['comparison_matches'])} comparison matches")
        
        # Debug: Show sample data for first few matches
        if len(matched_results[model_name]['rating_matches']) > 0:
            sample_match = matched_results[model_name]['rating_matches'][0]
            print(f"  Sample rating match:")
            print(f"    Human ratings: {sample_match['model_result'].get('human_ratings', {})}")
            print(f"    Model ratings: {sample_match['model_result'].get('llm_result', {}).get('ratings', {})}")
        
        if len(matched_results[model_name]['comparison_matches']) > 0:
            sample_match = matched_results[model_name]['comparison_matches'][0]
            print(f"  Sample comparison match:")
            print(f"    Test expected: {sample_match['test_case']['expected_output']}")
            print(f"    Model comparisons: {sample_match['model_result'].get('llm_result', {}).get('comparisons', {})}")
    
    return matched_results

def calculate_annotator_similarity(opinion1, opinion2):
    """Calculate similarity between two annotator opinions - optimized for exact matching"""
    if not opinion1 or not opinion2:
        return 0
    
    # Normalize text (lowercase, remove extra whitespace)
    text1 = ' '.join(opinion1.lower().split())
    text2 = ' '.join(opinion2.lower().split())
    
    # Check for exact match first (most common case)
    if text1 == text2:
        return 1.0
    
    # For non-exact matches, return 0 to speed up processing
    return 0

def calculate_test_set_accuracy(matched_results):
    """Calculate accuracy for test set matches"""
    results = {
        'rating_overall': {},
        'comparison_overall': {},
        'rating_by_dimension': defaultdict(dict),
        'comparison_by_dimension': defaultdict(dict)
    }
    
    for model_name, matches in matched_results.items():
        print(f"Calculating accuracy for {model_name}...")
        
        # Rating accuracy - compare human ratings with model ratings
        rating_accuracies = []
        for match in matches['rating_matches']:
            human_ratings = match['model_result'].get('human_ratings', {})
            model_ratings = match['model_result'].get('llm_result', {}).get('ratings', {})
            
            if human_ratings and model_ratings:
                # Calculate accuracy for this match
                total = 0
                correct = 0
                
                for dimension in DIMENSIONS.keys():
                    if dimension in human_ratings and dimension in model_ratings:
                        total += 1
                        if human_ratings[dimension] == model_ratings[dimension]:
                            correct += 1
                
                if total > 0:
                    accuracy = correct / total
                    rating_accuracies.append(accuracy)
        
        results['rating_overall'][model_name] = np.mean(rating_accuracies) if rating_accuracies else np.nan
        
        # Comparison accuracy - compare test set expected output with model comparisons
        comparison_accuracies = []
        for match in matches['comparison_matches']:
            test_output = match['test_case']['expected_output']
            model_output = match['model_result'].get('llm_result', {}).get('comparisons', {})
            
            if test_output and model_output:
                # Calculate accuracy for this match
                total = 0
                correct = 0
                
                for dimension in DIMENSIONS.keys():
                    if dimension in test_output and dimension in model_output:
                        total += 1
                        if test_output[dimension] == model_output[dimension]:
                            correct += 1
                
                if total > 0:
                    accuracy = correct / total
                    comparison_accuracies.append(accuracy)
        
        results['comparison_overall'][model_name] = np.mean(comparison_accuracies) if comparison_accuracies else np.nan
        
        # Dimension-wise accuracy for ratings
        for dimension in DIMENSIONS.keys():
            dim_accuracies = []
            for match in matches['rating_matches']:
                human_ratings = match['model_result'].get('human_ratings', {})
                model_ratings = match['model_result'].get('llm_result', {}).get('ratings', {})
                
                if (dimension in human_ratings and dimension in model_ratings and
                    human_ratings[dimension] == model_ratings[dimension]):
                    dim_accuracies.append(1.0)
                elif dimension in human_ratings and dimension in model_ratings:
                    dim_accuracies.append(0.0)
            
            results['rating_by_dimension'][dimension][model_name] = np.mean(dim_accuracies) if dim_accuracies else np.nan
        
        # Dimension-wise accuracy for comparisons
        for dimension in DIMENSIONS.keys():
            dim_accuracies = []
            for match in matches['comparison_matches']:
                test_output = match['test_case']['expected_output']
                model_output = match['model_result'].get('llm_result', {}).get('comparisons', {})
                
                if (dimension in test_output and dimension in model_output and
                    test_output[dimension] == model_output[dimension]):
                    dim_accuracies.append(1.0)
                elif dimension in test_output and dimension in model_output:
                    dim_accuracies.append(0.0)
            
            results['comparison_by_dimension'][dimension][model_name] = np.mean(dim_accuracies) if dim_accuracies else np.nan
    
    return results

def create_test_set_accuracy_visualization(results):
    """Create visualization of model accuracy on test set"""
    
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
    fig.suptitle('Model Test Set Accuracy by Dimension', fontsize=16, fontweight='bold')
    
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
    ax1.set_title('Rating Test Set Accuracy by Dimension', fontsize=14, fontweight='bold')
    
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
    ax2.set_title('Comparison Test Set Accuracy by Dimension', fontsize=14, fontweight='bold')
    
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
    
    plt.savefig(output_dir / 'model_test_set_accuracy.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test set accuracy visualization saved to: {output_dir / 'model_test_set_accuracy.pdf'}")

def main():
    """Main analysis function"""
    print("Starting Model Test Set Accuracy Analysis...")
    print("=" * 70)
    
    # Load test set
    print("Loading test set...")
    test_data = load_test_set()
    
    if not test_data:
        print("Error: No test data found!")
        return
    
    # Load model data
    print("Loading model data...")
    model_data = load_model_data()
    
    if not model_data:
        print("Error: No model data found!")
        return
    
    print(f"Loaded data for {len(model_data)} models")
    
    # Match test cases with model results
    print("\nMatching test cases with model results...")
    matched_results = match_test_cases_with_model_results(test_data, model_data)
    
    # Calculate accuracy
    print("\nCalculating test set accuracy...")
    results = calculate_test_set_accuracy(matched_results)
    
    # Create visualization
    print("\nCreating test set accuracy visualization...")
    create_test_set_accuracy_visualization(results)
    
    print("\n" + "=" * 70)
    print("Test set analysis complete!")

if __name__ == "__main__":
    main()
