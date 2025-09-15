#!/usr/bin/env python3
"""
Dimensional correlation analysis between four evaluation dimensions.
Analyzes pairwise correlations within rating annotations (1-5) and comparison annotations (1/2),
using appropriate correlation methods for each data type.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from collections import defaultdict

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Project root directory
PROJECT_ROOT = Path('/ibex/project/c2328/LLMs-Scalable-Deliberation')

# Four evaluation dimensions
DIMENSIONS = {
    "Representiveness": {
        "rating": "To what extent is your perspective represented in this response?",
        "comparison": "Which summary is more representative of your perspective?"
    },
    "Informativeness": {
        "rating": "How informative is this summary?",
        "comparison": "Which summary is more informative?"
    },
    "Neutrality": {
        "rating": "Do you think this summary presents a neutral and balanced view of the issue?",
        "comparison": "Which summary presents a more neutral and balanced view of the issue?"
    },
    "Policy Approval": {
        "rating": "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?",
        "comparison": "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
    }
}

def load_raw_data():
    """Load raw summary data with metadata"""
    raw_data_path = PROJECT_ROOT / 'annotation/summary-rating/data_files/raw/summaries_V0903_for_humanstudy_simple.csv'
    if raw_data_path.exists():
        df = pd.read_csv(raw_data_path)
        print(f"Loaded {len(df)} raw summary records")
        return df
    else:
        print("Raw data file not found")
        return None

def load_annotation_data(base_path):
    """Load all annotation data"""
    base_path = Path(base_path)
    all_annotations = {
        'comparisons': [],
        'ratings': []
    }
    
    # Load raw data for ID mapping
    raw_df = load_raw_data()
    if raw_df is None:
        return all_annotations
        
    # Create ID to metadata mapping
    id_metadata = {}
    for _, row in raw_df.iterrows():
        id_metadata[row['id']] = {
            'model': row['model'],
            'topic': row['topic'],
            'question': row['question'],
            'summary': row['summary']
        }
    
    for user_dir in base_path.iterdir():
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
                                metadata = assigned_data[data['id']]
                                
                                # For ratings, get the raw_id from metadata
                                if 'rating' in data['id'] and 'raw_id' in metadata:
                                    raw_id = metadata['raw_id']
                                    # raw_id might be combined ID like "id1_id2"
                                    if raw_id in id_metadata:
                                        data['metadata'] = id_metadata[raw_id]
                                        data['metadata']['raw_id'] = raw_id
                                    else:
                                        # Try splitting the raw_id and use the first part
                                        if '_' in raw_id:
                                            first_id = raw_id.split('_')[0]
                                            if first_id in id_metadata:
                                                data['metadata'] = id_metadata[first_id]
                                                data['metadata']['raw_id'] = first_id
                                
                                # For comparisons, get both summary IDs
                                elif 'comparison' in data['id']:
                                    if 'summary_a_id' in metadata and 'summary_b_id' in metadata:
                                        summary_a = metadata['summary_a_id']
                                        summary_b = metadata['summary_b_id']
                                        if summary_a in id_metadata and summary_b in id_metadata:
                                            data['metadata'] = {
                                                'summary_a_id': summary_a,
                                                'summary_b_id': summary_b,
                                                'model_a': id_metadata[summary_a]['model'],
                                                'model_b': id_metadata[summary_b]['model']
                                            }
                            
                            # Collect only rating and comparison annotations
                            if 'comparison' in data['id'] and 'label_annotations' in data:
                                all_annotations['comparisons'].append(data)
                            elif 'rating' in data['id'] and 'label_annotations' in data:
                                all_annotations['ratings'].append(data)
                                
                        except json.JSONDecodeError:
                            continue
    
    return all_annotations

def prepare_rating_data(annotations):
    """
    Prepare rating data for dimensional correlation analysis.
    Returns: user_ratings[user_id][dimension] = rating_value (1-5)
    """
    user_ratings = defaultdict(dict)
    
    for rating in annotations['ratings']:
        if 'metadata' not in rating or 'raw_id' not in rating['metadata']:
            continue
            
        user_id = rating['user_id']
        if 'label_annotations' not in rating:
            continue
            
        for dimension, questions in DIMENSIONS.items():
            rating_question = questions['rating']
            if rating_question in rating['label_annotations']:
                scales = rating['label_annotations'][rating_question]
                # Extract rating value from scale_* format
                for scale_key, value in scales.items():
                    if scale_key.startswith('scale_') and value and str(value).isdigit():
                        user_ratings[user_id][dimension] = int(value)
                        break
    
    return user_ratings

def prepare_comparison_data(annotations):
    """
    Prepare comparison data for dimensional correlation analysis.
    Returns: user_comparisons[user_id][dimension] = choice (1 or 2)
    """
    user_comparisons = defaultdict(dict)
    
    for comp in annotations['comparisons']:
        if 'metadata' not in comp:
            continue
            
        user_id = comp['user_id']
        if 'label_annotations' not in comp:
            continue
            
        for dimension, questions in DIMENSIONS.items():
            comp_question = questions['comparison']
            if comp_question in comp['label_annotations']:
                scales = comp['label_annotations'][comp_question]
                # Extract comparison choice from scale_* format
                for scale_key, value in scales.items():
                    if scale_key.startswith('scale_') and value and str(value).isdigit():
                        user_comparisons[user_id][dimension] = int(value)
                        break
    
    return user_comparisons

def calculate_rating_correlations(user_ratings):
    """
    Calculate pairwise correlations between rating dimensions (1-5 scale).
    Uses Spearman correlation for ordinal data.
    """
    dimensions = list(DIMENSIONS.keys())
    n_dims = len(dimensions)
    
    # Create correlation matrix
    corr_matrix = np.zeros((n_dims, n_dims))
    p_matrix = np.zeros((n_dims, n_dims))
    n_matrix = np.zeros((n_dims, n_dims))
    
    for i, dim1 in enumerate(dimensions):
        for j, dim2 in enumerate(dimensions):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                continue
                
            # Collect paired ratings
            ratings1 = []
            ratings2 = []
            
            for user_id, user_data in user_ratings.items():
                if dim1 in user_data and dim2 in user_data:
                    ratings1.append(user_data[dim1])
                    ratings2.append(user_data[dim2])
            
            if len(ratings1) > 2:  # Need at least 3 data points
                # Use Spearman correlation for ordinal data
                corr, p_val = scipy_stats.spearmanr(ratings1, ratings2)
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p_val
                n_matrix[i, j] = len(ratings1)
            else:
                corr_matrix[i, j] = np.nan
                p_matrix[i, j] = np.nan
                n_matrix[i, j] = len(ratings1)
    
    return corr_matrix, p_matrix, n_matrix, dimensions

def calculate_comparison_correlations(user_comparisons):
    """
    Calculate pairwise correlations between comparison dimensions (1/2 binary).
    Uses Phi coefficient (Pearson correlation for binary data).
    """
    dimensions = list(DIMENSIONS.keys())
    n_dims = len(dimensions)
    
    # Create correlation matrix
    corr_matrix = np.zeros((n_dims, n_dims))
    p_matrix = np.zeros((n_dims, n_dims))
    n_matrix = np.zeros((n_dims, n_dims))
    
    for i, dim1 in enumerate(dimensions):
        for j, dim2 in enumerate(dimensions):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                continue
                
            # Collect paired comparisons
            comps1 = []
            comps2 = []
            
            for user_id, user_data in user_comparisons.items():
                if dim1 in user_data and dim2 in user_data:
                    # Convert 1/2 to 0/1 for correlation calculation
                    comps1.append(user_data[dim1] - 1)  # 1->0, 2->1
                    comps2.append(user_data[dim2] - 1)  # 1->0, 2->1
            
            if len(comps1) > 2:  # Need at least 3 data points
                # Use Pearson correlation for binary data (Phi coefficient)
                corr, p_val = scipy_stats.pearsonr(comps1, comps2)
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p_val
                n_matrix[i, j] = len(comps1)
            else:
                corr_matrix[i, j] = np.nan
                p_matrix[i, j] = np.nan
                n_matrix[i, j] = len(comps1)
    
    return corr_matrix, p_matrix, n_matrix, dimensions

def create_correlation_heatmaps(rating_corr, rating_p, rating_n, 
                               comp_corr, comp_p, comp_n, dimensions, output_path):
    """Create correlation heatmaps for both rating and comparison data"""
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Create annotations with correlation values and significance
    def create_annotations(corr_matrix, p_matrix, n_matrix):
        annot_matrix = []
        for i in range(len(dimensions)):
            row = []
            for j in range(len(dimensions)):
                if i == j:
                    row.append("1.00")
                elif not np.isnan(corr_matrix[i, j]):
                    corr = corr_matrix[i, j]
                    p_val = p_matrix[i, j]
                    
                    # Add significance stars
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    elif p_val < 0.05:
                        stars = "*"
                    else:
                        stars = ""
                    
                    row.append(f"{corr:.2f}{stars}")
                else:
                    row.append("N/A")
            annot_matrix.append(row)
        return annot_matrix
    
    # Rating correlations heatmap
    rating_annot = create_annotations(rating_corr, rating_p, rating_n)
    sns.heatmap(rating_corr, annot=rating_annot, fmt='', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                xticklabels=dimensions, yticklabels=dimensions,
                cbar_kws={'label': 'Spearman Correlation'}, ax=ax1)
    ax1.set_title('Rating Dimensions Correlation\n(Spearman, 1-5 scale)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Comparison correlations heatmap
    comp_annot = create_annotations(comp_corr, comp_p, comp_n)
    sns.heatmap(comp_corr, annot=comp_annot, fmt='', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                xticklabels=dimensions, yticklabels=dimensions,
                cbar_kws={'label': 'Phi Coefficient'}, ax=ax2)
    ax2.set_title('Comparison Dimensions Correlation\n(Phi coefficient, binary)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout to leave space for footnote
    plt.subplots_adjust(bottom=0.15)
    
    # Add footnote with proper spacing
    fig.text(0.5, 0.05, '* p<0.05, ** p<0.01, *** p<0.001', 
             ha='center', fontsize=10)
    
    # Save plot
    plt.savefig(output_path / 'dimensional_correlations.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    

def main():
    """Main analysis function"""
    # Paths
    annotation_path = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full'
    output_path = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation_dim_correlation'
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading annotation data...")
    annotations = load_annotation_data(annotation_path)
    print(f"Found {len(annotations['ratings'])} rating annotations")
    print(f"Found {len(annotations['comparisons'])} comparison annotations")
    
    print("\nPreparing rating data...")
    user_ratings = prepare_rating_data(annotations)
    print(f"Found ratings from {len(user_ratings)} users")
    
    print("Preparing comparison data...")
    user_comparisons = prepare_comparison_data(annotations)
    print(f"Found comparisons from {len(user_comparisons)} users")
    
    print("\nCalculating rating dimension correlations...")
    rating_corr, rating_p, rating_n, dimensions = calculate_rating_correlations(user_ratings)
    
    print("Calculating comparison dimension correlations...")
    
    # Debug: Check distribution of comparison choices
    print("\nComparison choice distributions:")
    for dimension in DIMENSIONS.keys():
        choices = [comp_data.get(dimension, 0) for comp_data in user_comparisons.values() if dimension in comp_data]
        if choices:
            from collections import Counter
            dist = Counter(choices)
            print(f"{dimension}: {dict(dist)} (total: {len(choices)})")
        else:
            print(f"{dimension}: No data")
    
    comp_corr, comp_p, comp_n, _ = calculate_comparison_correlations(user_comparisons)
    
    print("\nGenerating visualizations...")
    create_correlation_heatmaps(rating_corr, rating_p, rating_n,
                               comp_corr, comp_p, comp_n, dimensions, output_path)
    
    print("\nAnalysis complete!")
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    main()