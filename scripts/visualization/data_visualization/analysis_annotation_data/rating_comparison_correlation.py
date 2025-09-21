#!/usr/bin/env python3
"""
Analysis of correlation between rating and comparison annotations.
Compares rating scores (1-5) with comparison outcomes (1-5 scale mapped to binary)
using point-biserial correlation and other relevant metrics.
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
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

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
                                data.update(assigned_data[data['id']])
                            
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
        'representiveness': {
            'rating': "To what extent is your perspective represented in this response?",
            'comparison': "Which summary is more representative of your perspective? "
        },
        'informativeness': {
            'rating': "How informative is this summary?",
            'comparison': "Which summary is more informative? "
        },
        'neutrality': {
            'rating': "Do you think this summary presents a neutral and balanced view of the issue?",
            'comparison': "Which summary presents a more neutral and balanced view of the issue? "
        },
        'policy_approval': {
            'rating': "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue? ",
            'comparison': "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
        }
    }
    return dimensions

def get_dimension_display_names():
    """Get display names for dimensions in desired order"""
    return {
        'representiveness': 'Representiveness',
        'informativeness': 'Informativeness', 
        'neutrality': 'Neutrality',
        'policy_approval': 'Policy Approval'
    }

def get_dimension_order():
    """Get dimension order for consistent plotting"""
    return ['representiveness', 'informativeness', 'neutrality', 'policy_approval']

def extract_rating_scores(ratings, question):
    """Extract rating scores (1-5) for a specific question"""
    scores = []
    for rating in ratings:
        if 'label_annotations' in rating and question in rating['label_annotations']:
            scales = rating['label_annotations'][question]
            # Handle nested structure: {"Option Text": "Value"}
            if isinstance(scales, dict):
                for scale, value in scales.items():
                    if value and str(value).isdigit():
                        scores.append(int(value))
                        break
            elif scales and str(scales).isdigit():
                scores.append(int(scales))
    return np.array(scores)

def extract_comparison_outcomes(comparisons, question):
    """Extract binary outcomes (0/1) from comparison annotations using 1-5 scale"""
    outcomes = []
    for comp in comparisons:
        if 'label_annotations' in comp and question in comp['label_annotations']:
            scales = comp['label_annotations'][question]
            # Handle nested structure: {"Option Text": "Value"}
            if isinstance(scales, dict):
                for scale, value in scales.items():
                    if value and str(value).isdigit():
                        val = int(value)
                        # Map 1-5 scale: 1,2 -> 1 (A wins), 4,5 -> 0 (B wins), 3 -> neutral (skip)
                        if val == 1 or val == 2:
                            outcomes.append(1)
                        elif val == 4 or val == 5:
                            outcomes.append(0)
                        # Skip val == 3 (neutral)
                        break
            elif scales and str(scales).isdigit():
                val = int(scales)
                if val == 1 or val == 2:
                    outcomes.append(1)
                elif val == 4 or val == 5:
                    outcomes.append(0)
                # Skip val == 3 (neutral)
    return np.array(outcomes)

def calculate_correlations(ratings, comparisons):
    """Calculate correlations between rating and comparison annotations for each dimension"""
    dimensions = get_dimension_questions()
    correlations = {}
    
    for dim_name, questions in dimensions.items():
        # Get rating scores and comparison outcomes
        rating_scores = extract_rating_scores(ratings, questions['rating'])
        comparison_outcomes = extract_comparison_outcomes(comparisons, questions['comparison'])
        
        print(f"\n{dim_name} dimension:")
        print(f"  Rating scores: {len(rating_scores)}")
        print(f"  Comparison outcomes: {len(comparison_outcomes)}")
        
        if len(rating_scores) > 0 and len(comparison_outcomes) > 0:
            # For point-biserial correlation, we need to match the arrays
            # Take the minimum length to ensure compatibility
            min_len = min(len(rating_scores), len(comparison_outcomes))
            if min_len > 0:
                rating_subset = rating_scores[:min_len]
                comparison_subset = comparison_outcomes[:min_len]
                
                # Calculate point-biserial correlation
                try:
                    rpb, p_value = stats.pointbiserialr(rating_subset, comparison_subset)
                except Exception as e:
                    print(f"  Point-biserial correlation failed: {e}")
                    rpb, p_value = np.nan, np.nan
                
                # Calculate Spearman correlation as an alternative measure
                try:
                    rho, spearman_p = stats.spearmanr(rating_subset, comparison_subset)
                except Exception as e:
                    print(f"  Spearman correlation failed: {e}")
                    rho, spearman_p = np.nan, np.nan
                
                correlations[dim_name] = {
                    'point_biserial_corr': rpb,
                    'point_biserial_p': p_value,
                    'spearman_corr': rho,
                    'spearman_p': spearman_p,
                    'n_ratings': len(rating_scores),
                    'n_comparisons': len(comparison_outcomes),
                    'n_matched': min_len,
                    'rating_mean': np.mean(rating_scores),
                    'rating_std': np.std(rating_scores),
                    'comparison_mean': np.mean(comparison_outcomes),
                    'comparison_std': np.std(comparison_outcomes)
                }
            else:
                print(f"  No valid data for correlation")
                correlations[dim_name] = {
                    'point_biserial_corr': np.nan,
                    'point_biserial_p': np.nan,
                    'spearman_corr': np.nan,
                    'spearman_p': np.nan,
                    'n_ratings': len(rating_scores),
                    'n_comparisons': len(comparison_outcomes),
                    'n_matched': 0,
                    'rating_mean': np.nan,
                    'rating_std': np.nan,
                    'comparison_mean': np.nan,
                    'comparison_std': np.nan
                }
        else:
            print(f"  No data for {dim_name}")
            correlations[dim_name] = {
                'point_biserial_corr': np.nan,
                'point_biserial_p': np.nan,
                'spearman_corr': np.nan,
                'spearman_p': np.nan,
                'n_ratings': len(rating_scores),
                'n_comparisons': len(comparison_outcomes),
                'n_matched': 0,
                'rating_mean': np.nan,
                'rating_std': np.nan,
                'comparison_mean': np.nan,
                'comparison_std': np.nan
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
    """Extract rating value from nested label_annotations structure"""
    label_annotations = row.get('label_annotations', {})
    if rating_q in label_annotations:
        scales = label_annotations[rating_q]
        if isinstance(scales, dict):
            for scale, value in scales.items():
                if value and str(value).isdigit():
                    return float(value)
        elif scales and str(scales).isdigit():
            return float(scales)
    return np.nan


def _comparison_choice_a_from_row(row, comp_q):
    """Extract comparison choice from nested label_annotations structure"""
    label_annotations = row.get('label_annotations', {})
    if comp_q in label_annotations:
        scales = label_annotations[comp_q]
        if isinstance(scales, dict):
            for scale, value in scales.items():
                if value and str(value).isdigit():
                    val = int(value)
                    # Map 1-5 scale: 1,2 -> 1.0 (A wins), 4,5 -> 0.0 (B wins), 3 -> 0.5 (neutral)
                    if val == 1 or val == 2:
                        return 1.0
                    elif val == 3:
                        return 0.5
                    elif val == 4 or val == 5:
                        return 0.0
        elif scales and str(scales).isdigit():
            val = int(scales)
            if val == 1 or val == 2:
                return 1.0
            elif val == 3:
                return 0.5
            elif val == 4 or val == 5:
                return 0.0
    return np.nan


def compute_sample_level_metrics(output_path):
    """Compute per-sample (summary) avg rating and win rate for 4 dimensions, then corr and viz."""
    dims = get_dimension_questions()

    # Load annotation data from new format
    annotation_path = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full_augment'
    annotations = load_annotation_data(annotation_path)
    
    # Convert to DataFrame for easier processing
    all_data = []
    for ann_type in ['ratings', 'comparisons']:
        for item in annotations[ann_type]:
            item['type'] = ann_type
            all_data.append(item)
    
    ann_df = pd.DataFrame(all_data)
    
    # Prepare frames
    ann_df = ann_df.copy()
    ann_df['triplet_base'] = ann_df['id'].apply(_extract_triplet_base)
    rating_rows = ann_df[ann_df['id'].str.contains('_rating', na=False)].copy()
    comp_rows = ann_df[ann_df['id'].str.contains('_comparison', na=False)].copy()

    print(f"Rating rows: {len(rating_rows)}")
    print(f"Comparison rows: {len(comp_rows)}")
    print(f"Available columns in comp_rows: {list(comp_rows.columns)}")

    # Map each triplet_base to its summary_a_id (the rated summary for that triplet)
    base_to_summary_a = {}
    base_to_summary_b = {}
    valid_rows = comp_rows.dropna(subset=['summary_a_id', 'summary_b_id'])
    print(f"Valid comparison rows with summary IDs: {len(valid_rows)}")
    
    for _, row in valid_rows.iterrows():
        base = row['triplet_base']
        base_to_summary_a[base] = row['summary_a_id']
        base_to_summary_b[base] = row['summary_b_id']

    # Extract ratings per dimension and attribute to summary_a_id
    rating_records = []
    for dim, qs in dims.items():
        colname = f'rating_{dim}'
        rating_rows[colname] = rating_rows.apply(lambda r: _rating_value_from_row(r, qs['rating']), axis=1)
    
    print(f"Rating rows: {len(rating_rows)}")
    print(f"Base to summary mapping: {len(base_to_summary_a)} entries")
    
    for _, r in rating_rows.iterrows():
        base = r['triplet_base']
        if base in base_to_summary_a:
            summary_id = base_to_summary_a[base]
            for dim in dims.keys():
                val = r[f'rating_{dim}']
                if pd.notna(val):
                    rating_records.append({'summary_id': summary_id, 'dimension': dim, 'rating': float(val)})
    
    rating_df = pd.DataFrame(rating_records)
    print(f"Rating records created: {len(rating_df)}")
    
    if rating_df.empty:
        print("Warning: No rating records created. Creating empty DataFrame with required columns.")
        rating_df = pd.DataFrame(columns=['summary_id', 'dimension', 'rating'])

    sample_avg_rating = rating_df.groupby(['summary_id', 'dimension'])['rating'].agg(['mean', 'count']).reset_index()
    sample_avg_rating.rename(columns={'mean': 'avg_rating', 'count': 'n_ratings'}, inplace=True)

    # Extract wins per dimension per summary from comparison rows
    comp_records = []
    for dim, qs in dims.items():
        comp_rows[f'chosenA_{dim}'] = comp_rows.apply(lambda r: _comparison_choice_a_from_row(r, qs['comparison']), axis=1)
        
        # Check if required columns exist
        required_cols = ['summary_a_id', 'summary_b_id', f'chosenA_{dim}']
        missing_cols = [col for col in required_cols if col not in comp_rows.columns]
        if missing_cols:
            print(f"Warning: Missing columns for {dim}: {missing_cols}")
            continue
            
        sub = comp_rows[required_cols].dropna(subset=[f'chosenA_{dim}'])
        print(f"Valid comparison data for {dim}: {len(sub)} rows")
        
        # Filter out neutral choices (0.5)
        sub = sub[sub[f'chosenA_{dim}'] != 0.5]
        print(f"After filtering neutral choices for {dim}: {len(sub)} rows")
        
        if len(sub) > 0:
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
    print(f"Comparison records created: {len(sample_win)}")

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

    # Create combined 2x2 subplot figure for all dimensions
    dim_order = get_dimension_order()
    display_names = get_dimension_display_names()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, dim in enumerate(dim_order):
        ax = axes[i]
        sub = sample_metrics[(sample_metrics['dimension'] == dim)].dropna(subset=['avg_rating', 'win_rate']).copy()
        
        if sub.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{display_names[dim]}', fontsize=14, fontweight='bold')
            continue
            
        # Calculate density for color mapping
        from scipy.stats import gaussian_kde
        try:
            # Create density estimation
            xy = np.vstack([sub['avg_rating'], sub['win_rate']])
            density = gaussian_kde(xy)(xy)
            # Normalize density for color mapping
            density_norm = (density - density.min()) / (density.max() - density.min())
        except Exception:
            # Fallback to uniform color if density calculation fails
            density_norm = np.ones(len(sub))
        
        # Scatter plot with density-based colors
        scatter = ax.scatter(sub['avg_rating'], sub['win_rate'], 
                           c=density_norm, cmap='viridis', s=20, alpha=0.7)
        
        # Add colorbar for density
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Density', fontsize=10)
        
        # Trend line
        if len(sub) >= 3 and sub['avg_rating'].nunique() > 1:
            try:
                coef = np.polyfit(sub['avg_rating'], sub['win_rate'], 1)
                xs = np.linspace(sub['avg_rating'].min(), sub['avg_rating'].max(), 50)
                ys = coef[0] * xs + coef[1]
                ax.plot(xs, ys, color='red', linestyle='--', linewidth=2, alpha=0.8)
            except Exception:
                pass
        
        # Annotate correlation on plot
        csub = corr_df[corr_df['dimension'] == dim]
        if not csub.empty:
            r1 = csub.iloc[0]['pearson_r']; p1 = csub.iloc[0]['pearson_p']
            r2 = csub.iloc[0]['spearman_r']; p2 = csub.iloc[0]['spearman_p']
            annot = f"Pearson r={r1:.3f} (p={p1:.2e})\nSpearman r={r2:.3f} (p={p2:.2e})"
            ax.text(0.03, 0.97, annot, ha='left', va='top', transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.set_xlabel('Average Rating', fontsize=12)
        ax.set_ylabel('Win Rate', fontsize=12)
        ax.set_title(f'{display_names[dim]}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    plt.suptitle('Win Rate vs Average Rating by Dimension', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path / 'sample_level_corr_winrate_vs_rating_combined.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Brief print
    print(f"\nSample-level visualization saved: {output_path / 'sample_level_corr_winrate_vs_rating_combined.pdf'}")
    
    return corr_df


def create_sample_level_pearson_corr_heatmap(corr_df, output_path):
    """Create a heatmap of Pearson correlations (sample-level) across dimensions."""
    if corr_df is None or corr_df.empty:
        return
    # Keep dimensions order
    dims = get_dimension_order()
    # Include overall if present
    ordered_dims = [d for d in dims if d in set(corr_df['dimension'])]
    if 'overall' in set(corr_df['dimension']):
        ordered_dims.append('overall')
    
    # Get display names
    display_names = get_dimension_display_names()
    
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
                xticklabels=['Pearson r'], yticklabels=[display_names.get(d, d) for d in ordered_dims], ax=ax)
    ax.set_title('Sample-level Pearson correlation\n(avg rating vs win rate)')
    plt.tight_layout()
    fig.savefig(output_path / 'sample_level_pearson_corr_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main analysis function - only generate the sample-level correlation plot"""
    # Paths
    annotation_path = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full_augment'
    output_path = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation_rating_vs_comparison'
    
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

    # Compute per-sample metrics and correlations, then visualize
    print("\nComputing sample-level avg rating, win rate, and correlations...")
    corr_df = compute_sample_level_metrics(output_path)
    
    print("\nAnalysis complete!")
    print(f"Main result saved to: {output_path / 'sample_level_corr_winrate_vs_rating_combined.pdf'}")

if __name__ == "__main__":
    main()