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
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Dimensions mapping
DIMENSIONS = {
    'representiveness': "To what extent is your perspective represented in this response?",
    'informativeness': "How informative is this summary?",
    'neutrality': "Do you think this summary presents a neutral and balanced view of the issue?",
    'policy_approval': "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue? "
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
    raw_data_path = PROJECT_ROOT / 'annotation/summary-rating/data_files/raw/summaries_V0915_for_humanstudy_simple.csv'
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
    annotation_base_path = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full_augment'
    
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
                                data.update(assigned_data[data['id']])
                            
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
        # Get summary ID and topic information from assigned data
        raw_id = rating.get('raw_id', '')
        question = rating.get('question', '')
        
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
                
                # Extract rating value from nested structure
                if isinstance(scales, dict):
                    # Look for numeric values in the dictionary values
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
                elif scales and str(scales).isdigit():
                    # Fallback for direct value
                    rating_value = int(scales)
                    topic_ratings[topic][summary_id][dimension].append({
                        'user_id': user_id,
                        'rating': rating_value
                    })
                    successful_extractions += 1
                    found_any_dimension = True
                    
                    if total_processed <= 3:  # Debug first few
                        print(f"    Extracted rating: {rating_value}")
        
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
                    summary_ids.append(summary_id)
                    ratings_for_summary = [r['rating'] for r in dim_data[dimension]]
                    all_ratings.append(ratings_for_summary)
            
            if len(all_ratings) < 2:
                print(f"    Insufficient data: {len(all_ratings)} summaries with multiple ratings")
                results[topic][dimension] = {
                    'icc_2k': np.nan,
                    'p_value': np.nan,
                    'n_summaries': len(all_ratings),
                    'n_raters': 0,
                    'mean_rating': np.nan,
                    'std_rating': np.nan
                }
                continue
            
            # Create ratings matrix (summaries x raters)
            max_raters = max(len(ratings) for ratings in all_ratings)
            ratings_matrix = np.full((len(all_ratings), max_raters), np.nan)
            
            for i, ratings in enumerate(all_ratings):
                for j, rating in enumerate(ratings):
                    ratings_matrix[i, j] = rating
            
            # Calculate ICC(2,k)
            icc_2k, p_value = calculate_icc_2k(ratings_matrix)
            
            # Calculate descriptive statistics
            all_rating_values = [rating for ratings in all_ratings for rating in ratings]
            mean_rating = np.mean(all_rating_values)
            std_rating = np.std(all_rating_values)
            
            print(f"    ICC(2,k): {icc_2k:.3f}, p-value: {p_value:.3f}")
            print(f"    Summaries: {len(all_ratings)}, Max raters: {max_raters}")
            print(f"    Mean rating: {mean_rating:.2f} ± {std_rating:.2f}")
            
            results[topic][dimension] = {
                'icc_2k': icc_2k,
                'p_value': p_value,
                'n_summaries': len(all_ratings),
                'n_raters': max_raters,
                'mean_rating': mean_rating,
                'std_rating': std_rating
            }
    
    return results

def create_agreement_heatmap(results, output_path):
    """Create heatmap of ICC(2,k) values across topics and dimensions"""
    # Prepare data for heatmap
    topics = list(results.keys())
    dimensions = list(DIMENSIONS.keys())
    
    # Create ICC matrix
    icc_matrix = np.full((len(topics), len(dimensions)), np.nan)
    p_matrix = np.full((len(topics), len(dimensions)), np.nan)
    
    for i, topic in enumerate(topics):
        for j, dimension in enumerate(dimensions):
            if topic in results and dimension in results[topic]:
                icc_matrix[i, j] = results[topic][dimension]['icc_2k']
                p_matrix[i, j] = results[topic][dimension]['p_value']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ICC heatmap
    sns.heatmap(icc_matrix, 
                xticklabels=[get_dimension_display_names()[d] for d in dimensions],
                yticklabels=topics,
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                center=0.5, 
                vmin=0, 
                vmax=1,
                ax=ax1,
                cbar_kws={'label': 'ICC(2,k)'})
    ax1.set_title('Inter-Annotator Agreement (ICC(2,k))\nby Topic and Dimension', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Topic', fontsize=12)
    
    # P-value heatmap
    sns.heatmap(p_matrix, 
                xticklabels=[get_dimension_display_names()[d] for d in dimensions],
                yticklabels=topics,
                annot=True, 
                fmt='.3f', 
                cmap='Reds_r',
                vmin=0, 
                vmax=0.05,
                ax=ax2,
                cbar_kws={'label': 'p-value'})
    ax2.set_title('Statistical Significance (p-value)\nby Topic and Dimension', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Topic', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path / 'topic_rating_agreement_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_agreement_summary_plot(results, output_path):
    """Create summary plot showing ICC(2,k) distribution across topics and dimensions"""
    # Collect all ICC values
    all_icc_values = []
    all_p_values = []
    topic_dimension_pairs = []
    
    for topic in results:
        for dimension in results[topic]:
            icc_val = results[topic][dimension]['icc_2k']
            p_val = results[topic][dimension]['p_value']
            if not np.isnan(icc_val):
                all_icc_values.append(icc_val)
                all_p_values.append(p_val)
                topic_dimension_pairs.append(f"{topic}\n{get_dimension_display_names()[dimension]}")
    
    if not all_icc_values:
        print("No valid ICC values found for summary plot")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # ICC distribution
    ax1.hist(all_icc_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(all_icc_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_icc_values):.3f}')
    ax1.set_xlabel('ICC(2,k) Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Inter-Annotator Agreement (ICC(2,k))', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # P-value distribution
    ax2.hist(all_p_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05')
    ax2.set_xlabel('p-value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Statistical Significance (p-values)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'topic_rating_agreement_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_agreement_report(results, output_path):
    """Generate detailed report of agreement analysis"""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("TOPIC-BASED RATING INTER-ANNOTATOR AGREEMENT ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("This analysis evaluates the consistency of annotator ratings")
    report_lines.append("across different topics and dimensions using ICC(2,k).")
    report_lines.append("")
    
    # Overall summary
    all_icc_values = []
    significant_count = 0
    total_analyses = 0
    
    for topic in results:
        for dimension in results[topic]:
            icc_val = results[topic][dimension]['icc_2k']
            p_val = results[topic][dimension]['p_value']
            if not np.isnan(icc_val):
                all_icc_values.append(icc_val)
                total_analyses += 1
                if p_val < 0.05:
                    significant_count += 1
    
    if all_icc_values:
        report_lines.append("OVERALL SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total analyses: {total_analyses}")
        report_lines.append(f"Significant agreements (p < 0.05): {significant_count} ({significant_count/total_analyses*100:.1f}%)")
        report_lines.append(f"Mean ICC(2,k): {np.mean(all_icc_values):.3f}")
        report_lines.append(f"Median ICC(2,k): {np.median(all_icc_values):.3f}")
        report_lines.append(f"ICC(2,k) range: {np.min(all_icc_values):.3f} - {np.max(all_icc_values):.3f}")
        report_lines.append("")
    
    # Detailed results by topic
    for topic in sorted(results.keys()):
        report_lines.append(f"TOPIC: {topic}")
        report_lines.append("-" * 40)
        
        for dimension in DIMENSIONS.keys():
            if topic in results and dimension in results[topic]:
                data = results[topic][dimension]
                icc_val = data['icc_2k']
                p_val = data['p_value']
                n_summaries = data['n_summaries']
                n_raters = data['n_raters']
                mean_rating = data['mean_rating']
                std_rating = data['std_rating']
                
                if np.isnan(icc_val):
                    report_lines.append(f"  {get_dimension_display_names()[dimension]}: No data")
        else:
                    significance = "Significant" if p_val < 0.05 else "Not significant"
                    agreement_level = "Excellent" if icc_val >= 0.75 else "Good" if icc_val >= 0.60 else "Moderate" if icc_val >= 0.40 else "Poor"
                    
                    report_lines.append(f"  {get_dimension_display_names()[dimension]}:")
                    report_lines.append(f"    ICC(2,k): {icc_val:.3f} ({agreement_level})")
                    report_lines.append(f"    p-value: {p_val:.3f} ({significance})")
                    report_lines.append(f"    Summaries: {n_summaries}, Max raters: {n_raters}")
                    report_lines.append(f"    Mean rating: {mean_rating:.2f} ± {std_rating:.2f}")
        
        report_lines.append("")
    
    # Interpretation guide
    report_lines.append("INTERPRETATION GUIDE")
    report_lines.append("-" * 40)
    report_lines.append("ICC(2,k) Interpretation:")
    report_lines.append("  < 0.40: Poor agreement")
    report_lines.append("  0.40-0.59: Moderate agreement")
    report_lines.append("  0.60-0.74: Good agreement")
    report_lines.append("  ≥ 0.75: Excellent agreement")
    report_lines.append("")
    report_lines.append("p-value: Statistical significance of the agreement")
    report_lines.append("  p < 0.05: Agreement is statistically significant")
    report_lines.append("  p ≥ 0.05: Agreement is not statistically significant")
    
    # Save report
    report_text = "\n".join(report_lines)
    with open(output_path / 'topic_rating_agreement_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("Agreement analysis report saved to: topic_rating_agreement_report.txt")

def main():
    """Main analysis function"""
    # Setup paths
    output_path = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation_topic_agreement'
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    
    # Load raw data for topic information
    raw_df = load_raw_data()
    
    # Load annotation data
    annotations = load_annotation_data()
    
    print(f"Loaded {len(annotations['ratings'])} rating annotations")
    
    # Prepare rating data by topic
    print("\nPreparing rating data by topic...")
    topic_ratings = prepare_rating_data_by_topic(annotations, raw_df)
    
    if not topic_ratings:
        print("No topic rating data found. Exiting.")
        return
    
    # Calculate agreement for each topic and dimension
    print("\nCalculating inter-annotator agreement...")
    results = calculate_topic_rating_agreement(topic_ratings)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_agreement_heatmap(results, output_path)
    create_agreement_summary_plot(results, output_path)
    
    # Generate report
    print("\nGenerating report...")
    generate_agreement_report(results, output_path)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()