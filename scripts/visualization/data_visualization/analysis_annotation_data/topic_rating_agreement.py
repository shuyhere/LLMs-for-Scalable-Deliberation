#!/usr/bin/env python3
"""
Topic-based Rating Inter-Annotator Agreement Analysis
Evaluates annotator rating consistency for each topic separately using ICC(2,k).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Project root directory
PROJECT_ROOT = Path('/ibex/project/c2328/LLMs-Scalable-Deliberation')

# Dimensions mapping
DIMENSIONS = {
    'representiveness': "To what extent is your perspective represented in this response?",
    'informativeness': "How informative is this summary?",
    'neutrality': "Do you think this summary presents a neutral and balanced view of the issue?",
    'policy_approval': "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?"
}

def get_dimension_display_order():
    """Define the desired order for dimension display in visualizations"""
    return ['Representiveness', 'Informativeness', 'Neutrality', 'Policy Approval']

def get_dimension_display_names():
    """Get display names for dimensions"""
    return {
        'representiveness': 'Representiveness',
        'informativeness': 'Informativeness', 
        'neutrality': 'Neutrality',
        'policy_approval': 'Policy Approval'
    }

def load_raw_data():
    """Load raw summary data with metadata including topics"""
    raw_data_path = PROJECT_ROOT / 'annotation/summary-rating/data_files/raw/summaries_V0903_for_humanstudy_simple.csv'
    if raw_data_path.exists():
        df = pd.read_csv(raw_data_path)
        print(f"Loaded {len(df)} raw summary records")
        return df
    else:
        print("Raw data file not found")
        return pd.DataFrame()

def get_clean_topic_name(topic):
    """Convert raw topic names to clean display names"""
    topic_mapping = {
        'Binary-Health-Care-Policy': 'Health Care Policy',
        'Binary-Online-Identity-Policies': 'Online Identity Policies',
        'Binary-Refugee-Policies': 'Refugee Policies',
        'Binary-Tariff-Policy': 'Tariff Policy',
        'Binary-Vaccination-Policy': 'Vaccination Policy',
        'Openqa-AI-changes-human-life': 'AI Changes Human Life',
        'Openqa-Tipping-System': 'Tipping System',
        'Openqa-Trump-cutting-funding': 'Academic Funding',
        'Openqa-Updates-of-electronic-products': 'Electronic Products',
        'Openqa-Influencers-as-a-job': 'Influencers as Job'
    }
    return topic_mapping.get(topic, topic)

def get_topic_display_order():
    """Define the desired order for topic display in visualizations"""
    return [
        'Tipping System',
        'AI Changes Human Life', 
        'Academic Funding',
        'Influencers as Job',
        'Electronic Products',
        'Tariff Policy',
        'Health Care Policy',
        'Vaccination Policy',
        'Refugee Policies',
        'Online Identity Policies'
    ]

def load_annotation_data():
    """Load annotation data from all annotated instances"""
    annotation_base_path = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full'
    
    all_annotations = {
        'ratings': []
    }
    
    print(f"Processing annotation directories...")
    
    for user_dir in annotation_base_path.iterdir():
        if user_dir.is_dir():
            jsonl_file = user_dir / "annotated_instances.jsonl"
            assign_file = user_dir / "assigned_user_data.json"
            
            # Load assigned data for metadata
            assigned_data = {}
            if assign_file.exists():
                with open(assign_file, 'r') as f:
                    assigned_data = json.load(f)
            
            if jsonl_file.exists():
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            data['user_id'] = user_dir.name
                            
                            # Add metadata from assigned data
                            if data['id'] in assigned_data:
                                data['metadata'] = assigned_data[data['id']]
                            
                            # Only collect rating annotations
                            if 'rating' in data['id'] and 'label_annotations' in data:
                                all_annotations['ratings'].append(data)
                        except json.JSONDecodeError:
                            continue
    
    print(f"Loaded {len(all_annotations['ratings'])} rating annotations")
    
    return all_annotations

def prepare_rating_data_by_topic(annotations, raw_df):
    """
    Prepare rating data organized by topic and dimension.
    Returns: topic_ratings[topic][summary_id][dimension] = [list of ratings from different annotators]
    """
    # Create ID to metadata mapping
    if not raw_df.empty:
        id_metadata = {str(row['id']): row.to_dict() for _, row in raw_df.iterrows()}
    else:
        id_metadata = {}
    
    topic_ratings = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Debug: Track topics found and data extraction
    topics_found = set()
    questions_seen = set()
    successful_extractions = 0
    total_processed = 0
    dimension_matches = defaultdict(int)
    
    print("Debug: Starting data extraction...")
    
    for rating in annotations['ratings']:
        if 'metadata' not in rating:
            continue
            
        # Get summary ID and topic information
        raw_id = rating['metadata'].get('raw_id', '')
        question = rating['metadata'].get('question', '')
        
        if not raw_id or not question:
            continue
            
        # Extract summary ID
        if '_' in raw_id:
            summary_id = raw_id.split('_')[0]
        else:
            summary_id = raw_id
            
        # Get topic from raw data
        topic = "Unknown"
        if summary_id in id_metadata:
            raw_topic = id_metadata[summary_id].get('topic', '')
            if raw_topic:
                topic = get_clean_topic_name(raw_topic)
        
        topics_found.add(topic)
        questions_seen.add(question[:100])  # First 100 chars for debug
        
        user_id = rating['user_id']
        
        total_processed += 1
        
        if 'label_annotations' not in rating:
            if total_processed <= 5:  # Debug first few
                print(f"Debug {total_processed}: No label_annotations for user {user_id}")
            continue
        
        # Debug: Print available questions for first few entries
        if total_processed <= 3:
            print(f"Debug {total_processed}: User {user_id}, Topic: {topic}")
            print(f"  Available questions: {list(rating['label_annotations'].keys())}")
            print(f"  Looking for dimensions: {list(DIMENSIONS.values())}")
            
        # Process each dimension
        found_any_dimension = False
        for dimension, dimension_question in DIMENSIONS.items():
            if dimension_question in rating['label_annotations']:
                dimension_matches[dimension] += 1
                scales = rating['label_annotations'][dimension_question]
                
                if total_processed <= 3:  # Debug first few
                    print(f"  Found {dimension}: {scales}")
                
                # Extract rating value from scale_* format
                for scale_key, value in scales.items():
                    if value and str(value).isdigit():
                        rating_value = int(value)
                        topic_ratings[topic][summary_id][dimension].append({
                            'user_id': user_id,
                            'rating': rating_value
                        })
                        successful_extractions += 1
                        found_any_dimension = True
                        
                        if total_processed <= 3:  # Debug first few
                            print(f"    Extracted rating: {rating_value}")
                        break
        
        if not found_any_dimension and total_processed <= 10:
            print(f"Debug {total_processed}: No matching dimensions found for user {user_id}")
    
    print(f"\nDebug Summary:")
    print(f"Total annotations processed: {total_processed}")
    print(f"Successful rating extractions: {successful_extractions}")
    print(f"Success rate: {successful_extractions/total_processed*100:.1f}%" if total_processed > 0 else "0%")
    print(f"Topics found: {sorted(topics_found)}")
    print(f"Dimension matches: {dict(dimension_matches)}")
    print(f"Raw data entries: {len(id_metadata)}")
    print(f"Sample questions: {list(questions_seen)[:3]}")
    
    # Debug: Show topic distribution
    topic_summary_counts = {}
    for topic in topic_ratings:
        topic_summary_counts[topic] = len(topic_ratings[topic])
    print(f"Summaries per topic: {topic_summary_counts}")
    
    return topic_ratings

def calculate_icc_2k(ratings_matrix):
    """Calculate ICC(2,k) - Two-way random effects, average measures"""
    ratings_matrix = np.array(ratings_matrix)
    n_items, n_raters = ratings_matrix.shape
    
    if n_items < 2 or n_raters < 2:
        return np.nan, np.nan
    
    # Calculate means
    item_means = np.nanmean(ratings_matrix, axis=1)
    rater_means = np.nanmean(ratings_matrix, axis=0)
    grand_mean = np.nanmean(ratings_matrix)
    
    # Calculate sum of squares
    # Between items (MSR)
    SSR = n_raters * np.sum((item_means - grand_mean) ** 2)
    MSR = SSR / (n_items - 1)
    
    # Between raters (MSC)
    SSC = n_items * np.sum((rater_means - grand_mean) ** 2)
    MSC = SSC / (n_raters - 1)
    
    # Error (MSE)
    SSE = 0
    for i in range(n_items):
        for j in range(n_raters):
            if not np.isnan(ratings_matrix[i, j]):
                SSE += (ratings_matrix[i, j] - item_means[i] - rater_means[j] + grand_mean) ** 2
    
    MSE = SSE / ((n_items - 1) * (n_raters - 1))
    
    # Calculate ICC(2,k)
    icc_2k = (MSR - MSE) / (MSR + (MSC - MSE) / n_items)
    
    # Calculate F-statistic for significance test
    f_stat = MSR / MSE
    df1 = n_items - 1
    df2 = (n_items - 1) * (n_raters - 1)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    
    return icc_2k, p_value

def calculate_topic_rating_agreement(topic_ratings):
    """Calculate inter-annotator agreement for each topic and dimension"""
    results = {}
    
    for topic in topic_ratings.keys():
        print(f"\n=== Analyzing topic: {topic} ===")
        results[topic] = {}
        
        for dimension in DIMENSIONS.keys():
            print(f"  Dimension: {dimension}")
            
            # Collect all ratings for this topic-dimension combination
            all_ratings = []
            summary_ids = []
            
            for summary_id, dim_data in topic_ratings[topic].items():
                if dimension in dim_data and len(dim_data[dimension]) >= 2:  # Need at least 2 raters
                    ratings = [r['rating'] for r in dim_data[dimension]]
                    all_ratings.append(ratings)
                    summary_ids.append(summary_id)
            
            if not all_ratings:
                results[topic][dimension] = {
                    'icc_2k': np.nan,
                    'icc_p_value': np.nan,
                    'mean_rating': np.nan,
                    'std_rating': np.nan,
                    'n_summaries': 0,
                    'n_ratings': 0,
                    'avg_raters_per_summary': 0
                }
                print(f"    No data available")
                continue
            
            # Create ratings matrix
            max_raters = max(len(ratings) for ratings in all_ratings)
            ratings_matrix = np.full((len(all_ratings), max_raters), np.nan)
            
            for i, ratings in enumerate(all_ratings):
                for j, rating in enumerate(ratings):
                    ratings_matrix[i, j] = rating
            
            # Calculate ICC(2,k)
            icc_2k, icc_p = calculate_icc_2k(ratings_matrix)
            
            # Calculate descriptive statistics
            all_rating_values = [rating for ratings in all_ratings for rating in ratings]
            mean_rating = np.mean(all_rating_values)
            std_rating = np.std(all_rating_values)
            avg_raters_per_summary = np.mean([len(ratings) for ratings in all_ratings])
            
            results[topic][dimension] = {
                'icc_2k': icc_2k,
                'icc_p_value': icc_p,
                'mean_rating': mean_rating,
                'std_rating': std_rating,
                'n_summaries': len(all_ratings),
                'n_ratings': len(all_rating_values),
                'avg_raters_per_summary': avg_raters_per_summary
            }
            
            print(f"    ICC(2,k): {icc_2k:.3f} (p={icc_p:.3f})")
            print(f"    {len(all_ratings)} summaries, {len(all_rating_values)} ratings")
    
    return results

def create_topic_agreement_visualization(results):
    """Create visualization of topic-based rating agreement (only agreement-related plots)"""
    # Get topics in the specified order, only including those that exist in results
    topic_order = get_topic_display_order()
    topics = [topic for topic in topic_order if topic in results]
    
    # Get dimensions in the specified order with display names
    dimension_order = get_dimension_display_order()
    dimension_names = get_dimension_display_names()
    dimensions = list(DIMENSIONS.keys())
    
    # Create figure with custom subplot layout to make heatmap wider and all subplots square
    fig = plt.figure(figsize=(20, 6))
    
    # Use gridspec to control subplot sizes
    gs = fig.add_gridspec(1, 4, width_ratios=[2, 1, 1, 0.05], hspace=0.3, wspace=0.3)
    
    # 1. Heatmap of ICC values across topics and dimensions (wider)
    ax1 = fig.add_subplot(gs[0, 0])
    icc_matrix = np.zeros((len(topics), len(dimension_order)))
    
    for i, topic in enumerate(topics):
        for j, dim_display in enumerate(dimension_order):
            # Find the corresponding dimension key
            dim_key = None
            for key, display_name in dimension_names.items():
                if display_name == dim_display:
                    dim_key = key
                    break
            
            if dim_key and dim_key in results[topic]:
                icc_value = results[topic][dim_key]['icc_2k']
                # Keep negative values, only replace NaN with a very small value for display
                icc_matrix[i, j] = icc_value if not np.isnan(icc_value) else -1.0
            else:
                icc_matrix[i, j] = -1.0
    
    # Use a more subtle blue to red colormap, remove grid lines
    # Create a custom colormap with lower saturation
    import matplotlib.colors as mcolors
    
    # Create a softer blue-to-red colormap (cold to hot: blue for low, red for high)
    colors = ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('soft_BlueRed', colors, N=n_bins)
    
    im1 = ax1.imshow(icc_matrix, cmap=cmap, vmin=-0.5, vmax=1.0, aspect='auto')
    
    # Remove grid lines and set cleaner ticks
    ax1.set_xticks(range(len(dimension_order)))
    ax1.set_xticklabels(dimension_order, rotation=45, ha='right', fontsize=11)
    ax1.set_yticks(range(len(topics)))
    ax1.set_yticklabels(topics, fontsize=10)
    ax1.set_title('ICC(2,k) by Topic and Dimension', fontsize=14, fontweight='bold', pad=20)
    
    # Remove the grid lines by setting linewidth to 0
    ax1.grid(False)
    ax1.set_xticks(np.arange(-0.5, len(dimension_order), 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(topics), 1), minor=True)
    ax1.grid(which="minor", color="white", linestyle='-', linewidth=0)
    
    # Add text annotations for all values (including negative)
    for i in range(len(topics)):
        for j in range(len(dimension_order)):
            icc_val = icc_matrix[i, j]
            # Find the corresponding dimension key for p-value lookup
            dim_key = None
            for key, display_name in dimension_names.items():
                if display_name == dimension_order[j]:
                    dim_key = key
                    break
            
            if dim_key and dim_key in results[topics[i]]:
                p_val = results[topics[i]][dim_key]['icc_p_value']
            else:
                p_val = np.nan
            
            # Show all values, not just positive ones
            if not np.isnan(icc_val) and icc_val != -1.0:  # -1.0 is our NaN placeholder
                # Add significance stars
                stars = ""
                if not np.isnan(p_val):
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    elif p_val < 0.05:
                        stars = "*"
                
                # Choose text color based on value (white for dark backgrounds, black for light)
                color = 'white' if icc_val < 0.0 else 'black'
                ax1.text(j, i, f'{icc_val:.2f}\n{stars}', ha='center', va='center', 
                        color=color, fontweight='bold', fontsize=9)
            elif icc_val == -1.0:  # NaN case
                ax1.text(j, i, 'N/A', ha='center', va='center', 
                        color='gray', fontweight='bold', fontsize=9)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('ICC(2,k) Value', rotation=270, labelpad=15)
    
    # 2. Average ICC by topic (square)
    ax2 = fig.add_subplot(gs[0, 1])
    topic_avg_icc = []
    for topic in topics:
        valid_iccs = [results[topic][dim]['icc_2k'] for dim in dimensions 
                     if not np.isnan(results[topic][dim]['icc_2k'])]
        avg_icc = np.mean(valid_iccs) if valid_iccs else -1.0  # Use -1 for no data
        topic_avg_icc.append(avg_icc)
    
    # Create color-coded bars based on ICC values
    bar_colors = []
    for value in topic_avg_icc:
        if value >= 0.4:
            bar_colors.append('#2E8B57')  # Green for fair+
        elif value >= 0.2:
            bar_colors.append('#FF8C00')  # Orange for poor
        elif value >= 0:
            bar_colors.append('#87CEEB')  # Light blue for very poor
        else:
            bar_colors.append('#F08080')  # Light red for negative
    
    bars2 = ax2.bar(range(len(topics)), topic_avg_icc, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax2.set_xticks(range(len(topics)))
    ax2.set_xticklabels(topics, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Average ICC(2,k)', fontsize=12)
    ax2.set_title('Average Agreement by Topic', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.5, 1.0)  # Allow negative values
    
    # Remove grid and add subtle reference lines
    ax2.grid(False)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax2.axhline(y=0.4, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add values on bars (including negative values)
    for bar, value in zip(bars2, topic_avg_icc):
        if value != -1.0:  # Don't show -1 (no data indicator)
            y_offset = 0.02 if value >= 0 else -0.05  # Adjust position for negative values
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                    f'{value:.2f}', ha='center', fontweight='bold')
    
    # 3. Average ICC by dimension (square)
    ax3 = fig.add_subplot(gs[0, 2])
    dim_avg_icc = []
    for dim_display in dimension_order:
        # Find the corresponding dimension key
        dim_key = None
        for key, display_name in dimension_names.items():
            if display_name == dim_display:
                dim_key = key
                break
        
        if dim_key:
            valid_iccs = [results[topic][dim_key]['icc_2k'] for topic in topics 
                         if dim_key in results[topic] and not np.isnan(results[topic][dim_key]['icc_2k'])]
            avg_icc = np.mean(valid_iccs) if valid_iccs else -1.0  # Use -1 for no data
        else:
            avg_icc = -1.0
        dim_avg_icc.append(avg_icc)
    
    # Create color-coded bars based on ICC values for dimensions
    dim_bar_colors = []
    for value in dim_avg_icc:
        if value >= 0.4:
            dim_bar_colors.append('#2E8B57')  # Green for fair+
        elif value >= 0.2:
            dim_bar_colors.append('#FF8C00')  # Orange for poor
        elif value >= 0:
            dim_bar_colors.append('#87CEEB')  # Light blue for very poor
        else:
            dim_bar_colors.append('#F08080')  # Light red for negative
    
    bars3 = ax3.bar(range(len(dimension_order)), dim_avg_icc, color=dim_bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax3.set_xticks(range(len(dimension_order)))
    ax3.set_xticklabels(dimension_order, rotation=45, ha='right', fontsize=11)
    ax3.set_ylabel('Average ICC(2,k)', fontsize=12)
    ax3.set_title('Average Agreement by Dimension', fontsize=14, fontweight='bold')
    ax3.set_ylim(-0.5, 1.0)  # Allow negative values
    
    # Remove grid and add subtle reference lines
    ax3.grid(False)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax3.axhline(y=0.4, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add values on bars (including negative values)
    for bar, value in zip(bars3, dim_avg_icc):
        if value != -1.0:  # Don't show -1 (no data indicator)
            y_offset = 0.02 if value >= 0 else -0.05  # Adjust position for negative values
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                    f'{value:.2f}', ha='center', fontweight='bold')
    
    # Adjust layout manually since we're using gridspec
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15)
    
    # Save plot
    output_dir = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation_topic_rating_agreement'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'topic_rating_agreement.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'topic_rating_agreement.png', dpi=300, bbox_inches='tight')
    
    print(f"\nTopic rating agreement plots saved to {output_dir}")
    
    return fig

def main():
    """Main analysis function"""
    print("Starting Topic-based Rating Inter-Annotator Agreement Analysis...")
    print("=" * 70)
    
    # Load data
    print("Loading raw data for topic information...")
    raw_df = load_raw_data()
    
    print("Loading annotation data...")
    annotations = load_annotation_data()
    
    if not annotations['ratings']:
        print("Error: No rating annotation data found!")
        return
    
    # Prepare data organized by topic
    print("\nPreparing rating data by topic...")
    topic_ratings = prepare_rating_data_by_topic(annotations, raw_df)
    
    if not topic_ratings:
        print("Error: No topic-organized rating data found!")
        return
    
    print(f"Found data for {len(topic_ratings)} topics")
    
    # Calculate agreement metrics for each topic
    print("\n" + "=" * 70)
    print("CALCULATING RATING AGREEMENT BY TOPIC...")
    results = calculate_topic_rating_agreement(topic_ratings)
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating topic-based agreement visualizations...")
    create_topic_agreement_visualization(results)
    
    print("\n" + "=" * 70)
    print("Analysis complete! Check the results directory for outputs.")

if __name__ == "__main__":
    main()
