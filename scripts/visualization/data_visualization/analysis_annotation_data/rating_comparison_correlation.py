#!/usr/bin/env python3
"""
Analysis of correlation between rating and comparison annotations.
Compares rating scores (1-5) with binary comparison outcomes (0/1)
using # The code you provided is a comment in Python. It starts with a hashtag symbol (#) which
# indicates a single-line comment in Python. Comments are used to explain the code or provide
# additional information for anyone reading the code. In this case, the comment mentions
# "point-biserial correlation" which suggests that the code may be related to calculating or
# discussing point-biserial correlation in Python. However, the code itself is not performing
# any specific action as it is just a comment.
point-biserial correlation and other relevant metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Project root directory
PROJECT_ROOT = Path('/ibex/project/c2328/LLMs-Scalable-Deliberation')

def load_annotation_data(base_path):
    """Load all annotation data with detailed counting"""
    base_path = Path(base_path)
    all_annotations = {
        'comparisons': [],
        'ratings': []
    }
    
    for user_dir in base_path.iterdir():
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
                            
                            # Only collect rating and comparison annotations
                            if 'comparison' in data['id'] and 'label_annotations' in data:
                                all_annotations['comparisons'].append(data)
                            elif 'rating' in data['id'] and 'label_annotations' in data:
                                all_annotations['ratings'].append(data)
                        except json.JSONDecodeError:
                            continue
    
    return all_annotations

def get_dimension_questions():
    """Get the corresponding questions for each dimension in both rating and comparison formats"""
    dimensions = {
        'perspective': {
            'rating': "To what extent is your perspective represented in this response?",
            'comparison': "Which summary is more representative of your perspective?"
        },
        'informativeness': {
            'rating': "How informative is this summary?",
            'comparison': "Which summary is more informative?"
        },
        'neutrality': {
            'rating': "Do you think this summary presents a neutral and balanced view of the issue?",
            'comparison': "Which summary presents a more neutral and balanced view of the issue?"
        },
        'policy': {
            'rating': "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?",
            'comparison': "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
        }
    }
    return dimensions

def extract_rating_scores(ratings, question):
    """Extract rating scores (1-5) for a specific question"""
    scores = []
    for rating in ratings:
        if 'label_annotations' in rating and question in rating['label_annotations']:
            scales = rating['label_annotations'][question]
            for scale, value in scales.items():
                if value and str(value).isdigit():
                    scores.append(int(value))
    return np.array(scores)

def extract_comparison_outcomes(comparisons, question):
    """Extract binary outcomes (0/1) from comparison annotations"""
    outcomes = []
    for comp in comparisons:
        if 'label_annotations' in comp and question in comp['label_annotations']:
            scales = comp['label_annotations'][question]
            # Convert comparison choice to binary outcome
            # If scale_1 is chosen (1), outcome is 1; if scale_2 is chosen (2), outcome is 0
            if 'scale_1' in scales and scales['scale_1'] == '1':
                outcomes.append(1)
            elif 'scale_2' in scales and scales['scale_2'] == '2':
                outcomes.append(0)
            else:
                # Check for any other response format
                for scale, value in scales.items():
                    if value == '1':
                        outcomes.append(1)
                        break
                    elif value == '2':
                        outcomes.append(0)
                        break
    return np.array(outcomes)

def calculate_correlations(ratings, comparisons):
    """Calculate correlations between rating and comparison annotations for each dimension"""
    dimensions = get_dimension_questions()
    correlations = {}
    
    for dim_name, questions in dimensions.items():
        # Get rating scores and comparison outcomes
        rating_scores = extract_rating_scores(ratings, questions['rating'])
        comparison_outcomes = extract_comparison_outcomes(comparisons, questions['comparison'])
        
        if len(rating_scores) > 0 and len(comparison_outcomes) > 0:
            # Calculate point-biserial correlation
            rpb, p_value = stats.pointbiserialr(rating_scores, comparison_outcomes)
            
            # Calculate Spearman correlation as an alternative measure
            rho, spearman_p = stats.spearmanr(rating_scores, comparison_outcomes)
            
            correlations[dim_name] = {
                'point_biserial_corr': rpb,
                'point_biserial_p': p_value,
                'spearman_corr': rho,
                'spearman_p': spearman_p,
                'n_ratings': len(rating_scores),
                'n_comparisons': len(comparison_outcomes),
                'rating_mean': np.mean(rating_scores),
                'rating_std': np.std(rating_scores),
                'comparison_mean': np.mean(comparison_outcomes),
                'comparison_std': np.std(comparison_outcomes)
            }
    
    return correlations

def create_correlation_heatmap(correlations, output_path):
    """Create heatmap visualization of correlations"""
    # Prepare correlation matrices
    dimensions = list(correlations.keys())
    pb_corr = [correlations[dim]['point_biserial_corr'] for dim in dimensions]
    spearman_corr = [correlations[dim]['spearman_corr'] for dim in dimensions]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot point-biserial correlations
    sns.heatmap(np.array([pb_corr]).T, annot=True, fmt='.3f', cmap='RdBu_r',
                xticklabels=['Point-biserial'], yticklabels=dimensions,
                center=0, vmin=-1, vmax=1, ax=ax1)
    ax1.set_title('Point-biserial Correlation\nRating vs Comparison', fontsize=12)
    
    # Plot Spearman correlations
    sns.heatmap(np.array([spearman_corr]).T, annot=True, fmt='.3f', cmap='RdBu_r',
                xticklabels=['Spearman'], yticklabels=dimensions,
                center=0, vmin=-1, vmax=1, ax=ax2)
    ax2.set_title('Spearman Correlation\nRating vs Comparison', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path / 'rating_comparison_correlations.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_plots(correlations, ratings, comparisons, output_path):
    """Create distribution plots for each dimension"""
    dimensions = get_dimension_questions()
    
    for dim_name, questions in dimensions.items():
        rating_scores = extract_rating_scores(ratings, questions['rating'])
        comparison_outcomes = extract_comparison_outcomes(comparisons, questions['comparison'])
        
        if len(rating_scores) > 0 and len(comparison_outcomes) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Rating distribution
            sns.histplot(rating_scores, bins=5, ax=ax1)
            ax1.set_title(f'Rating Distribution\n{dim_name.capitalize()}')
            ax1.set_xlabel('Rating Score (1-5)')
            ax1.set_ylabel('Count')
            
            # Comparison distribution
            sns.histplot(comparison_outcomes, bins=2, ax=ax2)
            ax2.set_title(f'Comparison Distribution\n{dim_name.capitalize()}')
            ax2.set_xlabel('Comparison Outcome (0/1)')
            ax2.set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(output_path / f'distribution_{dim_name}.pdf', dpi=300, bbox_inches='tight')
            plt.close()

def generate_correlation_report(correlations):
    """Generate a detailed report of the correlation analysis"""
    report = []
    
    report.append("=" * 80)
    report.append("RATING VS COMPARISON CORRELATION ANALYSIS")
    report.append("=" * 80)
    
    for dim_name, stats in correlations.items():
        report.append(f"\n{dim_name.upper()} DIMENSION")
        report.append("-" * 40)
        
        # Sample sizes
        report.append(f"Sample sizes:")
        report.append(f"  Ratings: {stats['n_ratings']}")
        report.append(f"  Comparisons: {stats['n_comparisons']}")
        
        # Rating statistics
        report.append(f"\nRating statistics:")
        report.append(f"  Mean: {stats['rating_mean']:.3f}")
        report.append(f"  Std: {stats['rating_std']:.3f}")
        
        # Comparison statistics
        report.append(f"\nComparison statistics:")
        report.append(f"  Mean: {stats['comparison_mean']:.3f}")
        report.append(f"  Std: {stats['comparison_std']:.3f}")
        
        # Correlations
        report.append(f"\nCorrelations:")
        report.append(f"  Point-biserial: {stats['point_biserial_corr']:.3f}")
        report.append(f"  Point-biserial p-value: {stats['point_biserial_p']:.3e}")
        report.append(f"  Spearman: {stats['spearman_corr']:.3f}")
        report.append(f"  Spearman p-value: {stats['spearman_p']:.3e}")
        
        # Interpretation
        report.append(f"\nInterpretation:")
        pb_strength = abs(stats['point_biserial_corr'])
        if pb_strength < 0.1:
            strength = "negligible"
        elif pb_strength < 0.3:
            strength = "weak"
        elif pb_strength < 0.5:
            strength = "moderate"
        else:
            strength = "strong"
        
        report.append(f"  The correlation between ratings and comparisons for {dim_name} is {strength}.")
        if stats['point_biserial_p'] < 0.05:
            report.append(f"  The correlation is statistically significant (p < 0.05).")
        else:
            report.append(f"  The correlation is not statistically significant (p >= 0.05).")
    
    return "\n".join(report)


def compute_overall_avg_rating_and_winrate(ratings, comparisons):
    """Compute overall average rating and overall A-win rate per dimension and overall."""
    dimensions = get_dimension_questions()
    rows = []

    all_rating_scores = []
    all_comp_outcomes = []

    for dim_name, questions in dimensions.items():
        rating_scores = extract_rating_scores(ratings, questions['rating'])
        comparison_outcomes = extract_comparison_outcomes(comparisons, questions['comparison'])

        if len(rating_scores) > 0:
            all_rating_scores.append(rating_scores)
        if len(comparison_outcomes) > 0:
            all_comp_outcomes.append(comparison_outcomes)

        rows.append({
            'dimension': dim_name,
            'avg_rating': float(np.mean(rating_scores)) if len(rating_scores) > 0 else np.nan,
            'n_ratings': int(len(rating_scores)),
            'win_rate_A': float(np.mean(comparison_outcomes)) if len(comparison_outcomes) > 0 else np.nan,
            'n_comparisons': int(len(comparison_outcomes)),
        })

    # Overall row across all dimensions
    if all_rating_scores:
        concat_r = np.concatenate(all_rating_scores)
        overall_avg_rating = float(np.mean(concat_r)) if len(concat_r) > 0 else np.nan
        n_r_all = int(len(concat_r))
    else:
        overall_avg_rating = np.nan
        n_r_all = 0

    if all_comp_outcomes:
        concat_c = np.concatenate(all_comp_outcomes)
        overall_win_rate = float(np.mean(concat_c)) if len(concat_c) > 0 else np.nan
        n_c_all = int(len(concat_c))
    else:
        overall_win_rate = np.nan
        n_c_all = 0

    rows.append({
        'dimension': 'overall',
        'avg_rating': overall_avg_rating,
        'n_ratings': n_r_all,
        'win_rate_A': overall_win_rate,
        'n_comparisons': n_c_all,
    })

    return pd.DataFrame(rows)


def _extract_triplet_base(instance_id):
    if not isinstance(instance_id, str):
        return None
    parts = instance_id.rsplit('_', 1)
    return parts[0] if len(parts) == 2 else instance_id


def _rating_value_from_row(row, rating_q):
    for i in range(1, 6):
        col = f"{rating_q}:::scale_{i}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            return float(i)
    return np.nan


def _comparison_choice_a_from_row(row, comp_q):
    col_a = f"{comp_q}:::scale_1"
    col_b = f"{comp_q}:::scale_2"
    val_a = row.get(col_a, np.nan)
    val_b = row.get(col_b, np.nan)
    if pd.notna(val_a) and str(val_a).strip() != "":
        return 1.0
    if pd.notna(val_b) and str(val_b).strip() != "":
        return 0.0
    return np.nan


def compute_sample_level_metrics(output_path):
    """Compute per-sample (summary) avg rating and win rate for 4 dimensions, then corr and viz."""
    dims = get_dimension_questions()

    # Load flat annotations and triplet metadata
    ann_csv = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full/annotated_instances.csv'
    trip_csv = PROJECT_ROOT / 'annotation/summary-rating/data_files/processed/sum_humanstudy_triplet_full_ring.csv'
    ann = pd.read_csv(ann_csv)
    trip = pd.read_csv(trip_csv)

    # Prepare frames
    ann = ann.copy()
    ann['triplet_base'] = ann['instance_id'].apply(_extract_triplet_base)
    rating_rows = ann[ann['instance_id'].str.contains('_rating', na=False)].copy()
    comp_rows = ann[ann['instance_id'].str.contains('_comparison', na=False)].copy()

    # Join to get summary ids from comparison rows per triplet
    trip_comp = trip[trip['type'] == 'comparison'][['id', 'summary_a_id', 'summary_b_id']]
    comp_rows = comp_rows.merge(trip_comp, left_on='instance_id', right_on='id', how='left')
    comp_rows.drop(columns=['id'], inplace=True)

    # Map each triplet_base to its summary_a_id (the rated summary for that triplet)
    base_to_summary_a = {}
    base_to_summary_b = {}
    for _, row in comp_rows.dropna(subset=['summary_a_id', 'summary_b_id']).iterrows():
        base = row['triplet_base']
        base_to_summary_a[base] = row['summary_a_id']
        base_to_summary_b[base] = row['summary_b_id']

    # Extract ratings per dimension and attribute to summary_a_id
    rating_records = []
    for dim, qs in dims.items():
        colname = f'rating_{dim}'
        rating_rows[colname] = rating_rows.apply(lambda r: _rating_value_from_row(r, qs['rating']), axis=1)
    for _, r in rating_rows.iterrows():
        base = r['triplet_base']
        if base in base_to_summary_a:
            summary_id = base_to_summary_a[base]
            for dim in dims.keys():
                val = r[f'rating_{dim}']
                if pd.notna(val):
                    rating_records.append({'summary_id': summary_id, 'dimension': dim, 'rating': float(val)})
    rating_df = pd.DataFrame(rating_records)

    sample_avg_rating = rating_df.groupby(['summary_id', 'dimension'])['rating'].agg(['mean', 'count']).reset_index()
    sample_avg_rating.rename(columns={'mean': 'avg_rating', 'count': 'n_ratings'}, inplace=True)

    # Extract wins per dimension per summary from comparison rows
    comp_records = []
    for dim, qs in dims.items():
        comp_rows[f'chosenA_{dim}'] = comp_rows.apply(lambda r: _comparison_choice_a_from_row(r, qs['comparison']), axis=1)
        sub = comp_rows[[
            'summary_a_id', 'summary_b_id', f'chosenA_{dim}'
        ]].dropna(subset=[f'chosenA_{dim}'])
        # A side
        a_df = sub[['summary_a_id', f'chosenA_{dim}']].rename(columns={'summary_a_id': 'summary_id', f'chosenA_{dim}': 'win'})
        # win is 1.0 if chosenA==1 else 0.0
        # B side
        b_df = sub[['summary_b_id', f'chosenA_{dim}']].rename(columns={'summary_b_id': 'summary_id'})
        b_df['win'] = 1.0 - b_df[f'chosenA_{dim}']
        b_df = b_df[['summary_id', 'win']]
        all_df = pd.concat([a_df[['summary_id', 'win']], b_df], ignore_index=True)
        stats_df = all_df.groupby('summary_id')['win'].agg(['mean', 'count']).reset_index()
        stats_df['dimension'] = dim
        comp_records.append(stats_df.rename(columns={'mean': 'win_rate', 'count': 'n_comparisons'}))
    sample_win = pd.concat(comp_records, ignore_index=True) if comp_records else pd.DataFrame(columns=['summary_id','win_rate','n_comparisons','dimension'])

    # Merge per-sample metrics
    sample_metrics = pd.merge(sample_avg_rating, sample_win, on=['summary_id', 'dimension'], how='outer')

    # Correlations across samples per dimension
    corr_rows = []
    for dim in dims.keys():
        sub = sample_metrics.dropna(subset=['avg_rating', 'win_rate'])
        sub = sub[sub['dimension'] == dim]
        models = sub['summary_id'].unique()
        if len(models) >= 3 and sub['avg_rating'].nunique() > 1 and sub['win_rate'].nunique() > 1:
            try:
                pearson_r, pearson_p = stats.pearsonr(sub['avg_rating'].values, sub['win_rate'].values)
            except Exception:
                pearson_r, pearson_p = np.nan, np.nan
            try:
                spearman_r, spearman_p = stats.spearmanr(sub['avg_rating'].values, sub['win_rate'].values)
            except Exception:
                spearman_r, spearman_p = np.nan, np.nan
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = np.nan
        corr_rows.append({
            'dimension': dim,
            'num_summaries': int(len(models)),
            'pearson_r': float(pearson_r) if pd.notna(pearson_r) else np.nan,
            'pearson_p': float(pearson_p) if pd.notna(pearson_p) else np.nan,
            'spearman_r': float(spearman_r) if pd.notna(spearman_r) else np.nan,
            'spearman_p': float(spearman_p) if pd.notna(spearman_p) else np.nan,
        })
    corr_df = pd.DataFrame(corr_rows)

    # Visualize scatter per dimension
    for dim in dims.keys():
        sub = sample_metrics[(sample_metrics['dimension'] == dim)].dropna(subset=['avg_rating', 'win_rate']).copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sub['avg_rating'], sub['win_rate'], s=16, alpha=0.8)
        # trend line
        if len(sub) >= 3 and sub['avg_rating'].nunique() > 1:
            try:
                coef = np.polyfit(sub['avg_rating'], sub['win_rate'], 1)
                xs = np.linspace(sub['avg_rating'].min(), sub['avg_rating'].max(), 50)
                ys = coef[0] * xs + coef[1]
                ax.plot(xs, ys, color='red', linestyle='--', linewidth=1)
            except Exception:
                pass
        # annotate correlation on plot
        csub = corr_df[corr_df['dimension'] == dim]
        if not csub.empty:
            r1 = csub.iloc[0]['pearson_r']; p1 = csub.iloc[0]['pearson_p']
            r2 = csub.iloc[0]['spearman_r']; p2 = csub.iloc[0]['spearman_p']
            annot = f"Pearson r={r1:.3f} (p={p1:.2e})\nSpearman r={r2:.3f} (p={p2:.2e})"
            ax.text(0.03, 0.97, annot, ha='left', va='top', transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlabel('Average rating')
        ax.set_ylabel('Win rate')
        ax.set_title(f'Sample-level: win rate vs avg rating — {dim}')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(output_path / f'sample_level_corr_winrate_vs_rating_{dim}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Save CSVs
    sample_metrics.to_csv(output_path / 'sample_level_avg_rating_winrate.csv', index=False)
    corr_df.to_csv(output_path / 'sample_level_corr_winrate_vs_rating.csv', index=False)

    # Brief print
    print("\nSample-level metrics and correlations saved:")
    print(f"  Metrics: {output_path / 'sample_level_avg_rating_winrate.csv'}")
    print(f"  Correlations: {output_path / 'sample_level_corr_winrate_vs_rating.csv'}")


def create_sample_level_pearson_corr_heatmap(corr_csv_path, output_path):
    """Create a heatmap of Pearson correlations (sample-level) across dimensions."""
    try:
        corr_df = pd.read_csv(corr_csv_path)
    except Exception:
        return
    # Keep dimensions order
    dims = list(get_dimension_questions().keys())
    # Include overall if present
    ordered_dims = [d for d in dims if d in set(corr_df['dimension'])]
    if 'overall' in set(corr_df['dimension']):
        ordered_dims.append('overall')
    # Build matrix (rows: dims, single column: Pearson)
    values = []
    for d in ordered_dims:
        sub = corr_df[corr_df['dimension'] == d]
        if sub.empty:
            values.append([np.nan])
        else:
            values.append([sub.iloc[0]['pearson_r']])
    mat = np.array(values)
    fig, ax = plt.subplots(1, 1, figsize=(5, max(3, 0.6 * len(ordered_dims) + 1)))
    sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=['Pearson r'], yticklabels=ordered_dims, ax=ax)
    ax.set_title('Sample-level Pearson correlation\n(avg rating vs win rate)')
    plt.tight_layout()
    fig.savefig(output_path / 'sample_level_pearson_corr_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main analysis function"""
    # Paths
    annotation_path = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full'
    output_path = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation'
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading annotation data...")
    annotations = load_annotation_data(annotation_path)
    
    # Extract ratings and comparisons
    ratings = annotations['ratings']
    comparisons = annotations['comparisons']
    
    print(f"\nData Summary:")
    print(f"  Total Ratings: {len(ratings)}")
    print(f"  Total Comparisons: {len(comparisons)}")
    
    # Calculate correlations
    print("\nCalculating correlations...")
    correlations = calculate_correlations(ratings, comparisons)
    
    # Generate and save report
    print("\nGenerating report...")
    report = generate_correlation_report(correlations)
    
    report_file = output_path / 'rating_comparison_correlation_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_correlation_heatmap(correlations, output_path)
    create_distribution_plots(correlations, ratings, comparisons, output_path)
    
    # Save correlation data as CSV
    print("\nSaving correlation data...")
    correlation_data = []
    for dim_name, stats in correlations.items():
        correlation_data.append({
            'dimension': dim_name,
            **stats
        })
    correlation_df = pd.DataFrame(correlation_data)
    correlation_df.to_csv(output_path / 'rating_comparison_correlations.csv', index=False)
    
    # Save overall averages (avg rating and A-win rate)
    print("\nSaving overall averages (avg rating and A-win rate)...")
    overall_df = compute_overall_avg_rating_and_winrate(ratings, comparisons)
    overall_csv = output_path / 'rating_comparison_overall_avg_rating_winrate.csv'
    overall_df.to_csv(overall_csv, index=False)
    print(f"Saved overall averages to: {overall_csv}")

    # Compute per-sample metrics and correlations, then visualize
    print("\nComputing sample-level avg rating, win rate, and correlations...")
    compute_sample_level_metrics(output_path)
    # Heatmap for Pearson correlations across dimensions
    create_sample_level_pearson_corr_heatmap(output_path / 'sample_level_corr_winrate_vs_rating.csv', output_path)
    
    print("\nAnalysis complete!")
    print(f"All results saved to: {output_path}")

if __name__ == "__main__":
    main()
