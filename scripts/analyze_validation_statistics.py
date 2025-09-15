#!/usr/bin/env python3
"""
Script to analyze validation results and provide basic statistics about the four dimensions.
Counts samples with all false, all true, and mostly true patterns.
"""

import json
import numpy as np
from typing import List, Dict, Any
from collections import Counter

def load_validation_results(file_path: str) -> Dict[str, Any]:
    """Load the complete validation results JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_dimension_data(validation_data: Dict[str, Any]) -> List[List[bool]]:
    """
    Extract the four boolean dimensions for each sample.
    Returns a list where each element is [perspective, informativeness, neutrality, policy]
    """
    dimension_data = []
    
    # The data is under cosine_similarity_analysis -> examples
    if 'cosine_similarity_analysis' in validation_data:
        examples = validation_data['cosine_similarity_analysis']['examples']
        
        # Process each sample
        for sample in examples:
            if isinstance(sample, dict):
                dimensions = [
                    sample.get('perspective_similarity_matches_preference', False),
                    sample.get('informativeness_similarity_matches_preference', False),
                    sample.get('neutrality_similarity_matches_preference', False),
                    sample.get('policy_similarity_matches_preference', False)
                ]
                dimension_data.append(dimensions)
    
    return dimension_data

def analyze_dimension_patterns(dimension_data: List[List[bool]]) -> Dict[str, Any]:
    """
    Analyze patterns in the four dimensions and calculate statistics.
    """
    total_samples = len(dimension_data)
    
    # Count patterns
    all_false = 0
    all_true = 0
    mostly_true = 0  # 3 or more dimensions are true
    
    # Count individual dimension performance
    dimension_counts = [0, 0, 0, 0]  # [perspective, informativeness, neutrality, policy]
    
    # Count all possible patterns
    pattern_counts = Counter()
    
    for sample in dimension_data:
        if len(sample) == 4:
            # Count individual dimensions
            for i, value in enumerate(sample):
                if value:
                    dimension_counts[i] += 1
            
            # Check patterns
            true_count = sum(sample)
            
            if true_count == 0:
                all_false += 1
            elif true_count == 4:
                all_true += 1
            elif true_count >= 3:
                mostly_true += 1
            
            # Record the exact pattern
            pattern = tuple(sample)
            pattern_counts[pattern] += 1
    
    # Calculate percentages
    stats = {
        'total_samples': total_samples,
        'all_false_count': all_false,
        'all_false_percentage': (all_false / total_samples * 100) if total_samples > 0 else 0,
        'all_true_count': all_true,
        'all_true_percentage': (all_true / total_samples * 100) if total_samples > 0 else 0,
        'mostly_true_count': mostly_true,
        'mostly_true_percentage': (mostly_true / total_samples * 100) if total_samples > 0 else 0,
        'dimension_performance': {
            'perspective': {
                'count': dimension_counts[0],
                'percentage': (dimension_counts[0] / total_samples * 100) if total_samples > 0 else 0
            },
            'informativeness': {
                'count': dimension_counts[1],
                'percentage': (dimension_counts[1] / total_samples * 100) if total_samples > 0 else 0
            },
            'neutrality': {
                'count': dimension_counts[2],
                'percentage': (dimension_counts[2] / total_samples * 100) if total_samples > 0 else 0
            },
            'policy': {
                'count': dimension_counts[3],
                'percentage': (dimension_counts[3] / total_samples * 100) if total_samples > 0 else 0
            }
        },
        'pattern_distribution': dict(pattern_counts.most_common(10))  # Top 10 patterns
    }
    
    return stats

def print_statistics(stats: Dict[str, Any]):
    """Print the analysis results in a formatted way."""
    print("=" * 60)
    print("VALIDATION DATA STATISTICS")
    print("=" * 60)
    print(f"Total samples analyzed: {stats['total_samples']}")
    print()
    
    print("PATTERN ANALYSIS:")
    print("-" * 30)
    print(f"Samples with all 4 dimensions FALSE: {stats['all_false_count']} ({stats['all_false_percentage']:.2f}%)")
    print(f"Samples with all 4 dimensions TRUE:  {stats['all_true_count']} ({stats['all_true_percentage']:.2f}%)")
    print(f"Samples with 3+ dimensions TRUE:     {stats['mostly_true_count']} ({stats['mostly_true_percentage']:.2f}%)")
    print()
    
    print("INDIVIDUAL DIMENSION PERFORMANCE:")
    print("-" * 40)
    for dim_name, dim_stats in stats['dimension_performance'].items():
        print(f"{dim_name.capitalize():15}: {dim_stats['count']:6} samples ({dim_stats['percentage']:5.2f}%)")
    print()
    
    print("TOP PATTERNS (P,I,N,P format where T=True, F=False):")
    print("-" * 55)
    for pattern, count in list(stats['pattern_distribution'].items())[:10]:
        pattern_str = ''.join(['T' if x else 'F' for x in pattern])
        percentage = (count / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0
        print(f"Pattern {pattern_str}: {count:6} samples ({percentage:5.2f}%)")

def main():
    """Main function to run the analysis."""
    validation_file = "/ibex/project/c2328/LLMs-Scalable-Deliberation/results/validation_data/complete_validation_results.json"
    
    print("Loading validation results...")
    validation_data = load_validation_results(validation_file)
    
    print("Extracting dimension data...")
    dimension_data = extract_dimension_data(validation_data)
    
    print("Analyzing patterns...")
    stats = analyze_dimension_patterns(dimension_data)
    
    print_statistics(stats)

if __name__ == "__main__":
    main()
