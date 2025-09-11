#!/usr/bin/env python3
"""
Analyze per-topic average ratings (four dimensions) and sample-level relationship
between win rate and rating.

Inputs:
- annotated_instances.csv: human annotations (rating/comparison)
- sum_humanstudy_triplet_full_ring.csv: triplet metadata with A/B models and summary ids
- summaries_V0903_for_humanstudy_simple.csv: raw summaries with topic and model

Outputs (results/dataset_visulization/analysis_annotation/topic_analysis):
- topic_avg_ratings_heatmap.pdf: topics x dimensions heatmap of average ratings
- topic_scatter_overall.pdf: overall sample-level win rate vs rating across topics
- topic_scatter_<topic>.pdf: per-topic overall sample-level win rate vs rating
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import re


PROJECT_ROOT = Path('/ibex/project/c2328/LLMs-Scalable-Deliberation')
ANNOTATED_CSV = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full/annotated_instances.csv'
TRIPLET_CSV = PROJECT_ROOT / 'annotation/summary-rating/data_files/processed/sum_humanstudy_triplet_full_ring.csv'
RAW_SUMMARIES_CSV = PROJECT_ROOT / 'annotation/summary-rating/data_files/raw/summaries_V0903_for_humanstudy_simple.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation/topic_analysis'


plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


def get_dimension_questions() -> Dict[str, Dict[str, str]]:
    return {
        'perspective': {
            'rating': "To what extent is your perspective represented in this response?",
            'comparison': "Which summary is more representative of your perspective?",
        },
        'informativeness': {
            'rating': "How informative is this summary?",
            'comparison': "Which summary is more informative?",
        },
        'neutrality': {
            'rating': "Do you think this summary presents a neutral and balanced view of the issue?",
            'comparison': "Which summary presents a more neutral and balanced view of the issue?",
        },
        'policy': {
            'rating': "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?",
            'comparison': "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?",
        },
    }


# Desired topic display order (OpenQA first, then Binary)
DESIRED_TOPIC_ORDER = [
    'Tipping System', 'AI Changes Life', 'Academic Funding', 'Influencer', 'Electronic Products',
    'Tariff Policy', 'Health Care', 'Vaccination Policy', 'Refugee Policy', 'Online Identity',
]


def map_topic_display(raw_topic: str) -> str:
    """Map raw topic strings to canonical display names in DESIRED_TOPIC_ORDER.

    Handles prefixes (OpenQA/Binary), hyphens/underscores, pluralization, and case.
    """
    if not isinstance(raw_topic, str):
        return raw_topic
    s = re.sub(r'^(OpenQA|Openqa|Binary)[\-_: ]*', '', raw_topic, flags=re.IGNORECASE)
    s = s.replace('_', ' ').replace('-', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    low = s.lower()

    if 'tipping' in low:
        return 'Tipping System'
    if ('ai' in low and ('life' in low or 'changes' in low)):
        return 'AI Changes Life'
    if 'funding' in low or 'academic' in low or ('trump' in low and 'fund' in low):
        return 'Academic Funding'
    if 'influencer' in low:
        return 'Influencer'
    if 'electronic' in low or ('update' in low and 'product' in low):
        return 'Electronic Products'
    if 'tariff' in low:
        return 'Tariff Policy'
    if 'health' in low and 'care' in low:
        return 'Health Care'
    if 'vaccin' in low:
        return 'Vaccination Policy'
    if 'refugee' in low:
        return 'Refugee Policy'
    if ('online' in low and ('identity' in low or 'real name' in low or 'real-name' in low)):
        return 'Online Identity'

    # Fallback: title-case cleaned string
    return ' '.join(w.capitalize() for w in s.split(' '))


def _extract_triplet_base(instance_id: str) -> str:
    if not isinstance(instance_id, str):
        return None
    parts = instance_id.rsplit('_', 1)
    return parts[0] if len(parts) == 2 else instance_id


def _rating_value_from_row(row: pd.Series, rating_q: str) -> float:
    for i in range(1, 6):
        col = f"{rating_q}:::scale_{i}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            return float(i)
    return np.nan


def _comparison_choice_a_from_row(row: pd.Series, comp_q: str) -> float:
    col_a = f"{comp_q}:::scale_1"
    col_b = f"{comp_q}:::scale_2"
    val_a = row.get(col_a, np.nan)
    val_b = row.get(col_b, np.nan)
    if pd.notna(val_a) and str(val_a).strip() != "":
        return 1.0
    if pd.notna(val_b) and str(val_b).strip() != "":
        return 0.0
    return np.nan


def compute_sample_level_metrics() -> pd.DataFrame:
    dims = get_dimension_questions()

    ann = pd.read_csv(ANNOTATED_CSV)
    trip = pd.read_csv(TRIPLET_CSV)

    ann['triplet_base'] = ann['instance_id'].apply(_extract_triplet_base)

    rating_rows = ann[ann['instance_id'].str.contains('_rating', na=False)].copy()
    comp_rows = ann[ann['instance_id'].str.contains('_comparison', na=False)].copy()

    trip_comp = trip[trip['type'] == 'comparison'][['id', 'summary_a_id', 'summary_b_id']]
    comp_rows = comp_rows.merge(trip_comp, left_on='instance_id', right_on='id', how='left')
    comp_rows.drop(columns=['id'], inplace=True)

    base_to_summary_a: Dict[str, str] = {}
    for _, row in comp_rows.dropna(subset=['summary_a_id']).iterrows():
        base_to_summary_a[row['triplet_base']] = row['summary_a_id']

    # Ratings per summary_id and dimension
    for dim, qs in dims.items():
        rating_rows[f'rating_{dim}'] = rating_rows.apply(lambda r: _rating_value_from_row(r, qs['rating']), axis=1)
    rating_records: List[Dict[str, object]] = []
    for _, r in rating_rows.iterrows():
        base = r['triplet_base']
        if base in base_to_summary_a:
            sid = base_to_summary_a[base]
            for dim in dims.keys():
                val = r[f'rating_{dim}']
                if pd.notna(val):
                    rating_records.append({'summary_id': sid, 'dimension': dim, 'rating': float(val)})
    rating_df = pd.DataFrame(rating_records)
    sample_avg_rating = rating_df.groupby(['summary_id', 'dimension'])['rating'].agg(['mean', 'count']).reset_index()
    sample_avg_rating.rename(columns={'mean': 'avg_rating', 'count': 'n_ratings'}, inplace=True)

    # Win rates per summary_id and dimension (expand A/B)
    comp_records: List[Dict[str, object]] = []
    for dim, qs in dims.items():
        comp_rows[f'chosenA_{dim}'] = comp_rows.apply(lambda r: _comparison_choice_a_from_row(r, qs['comparison']), axis=1)
        sub = comp_rows.dropna(subset=[f'chosenA_{dim}', 'summary_a_id', 'summary_b_id'])
        a_df = sub[['summary_a_id', f'chosenA_{dim}']].rename(columns={'summary_a_id': 'summary_id', f'chosenA_{dim}': 'win'})
        b_df = sub[['summary_b_id', f'chosenA_{dim}']].rename(columns={'summary_b_id': 'summary_id'})
        b_df['win'] = 1.0 - b_df[f'chosenA_{dim}']
        b_df = b_df[['summary_id', 'win']]
        all_df = pd.concat([a_df[['summary_id', 'win']], b_df], ignore_index=True)
        stat = all_df.groupby('summary_id')['win'].agg(['mean', 'count']).reset_index()
        stat.rename(columns={'mean': 'win_rate', 'count': 'n_comparisons'}, inplace=True)
        stat['dimension'] = dim
        comp_records.append(stat)
    sample_win = pd.concat(comp_records, ignore_index=True)

    sample_metrics = pd.merge(sample_avg_rating, sample_win, on=['summary_id', 'dimension'], how='outer')
    return sample_metrics


def merge_with_topics(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(RAW_SUMMARIES_CSV)
    raw_sel = raw[['id', 'topic', 'model']].rename(columns={'id': 'summary_id'})
    out = pd.merge(sample_metrics, raw_sel, on='summary_id', how='left')
    return out


def plot_topic_avg_ratings(merged: pd.DataFrame, output_dir: Path) -> None:
    dims = list(get_dimension_questions().keys())
    topic_avg = merged.dropna(subset=['avg_rating']).groupby(['topic', 'dimension'])['avg_rating'].mean().reset_index()
    pivot = topic_avg.pivot_table(index='topic', columns='dimension', values='avg_rating', aggfunc='mean')
    pivot = pivot.reindex(columns=dims)
    # Map raw topics to canonical display names and enforce desired order
    display_index = [map_topic_display(t) for t in pivot.index]
    pivot.index = display_index
    # Report missing topics for debugging
    exist = [t for t in DESIRED_TOPIC_ORDER if t in pivot.index]
    missing = [t for t in DESIRED_TOPIC_ORDER if t not in pivot.index]
    if missing:
        print(f"[DEBUG] Missing topics in heatmap (no data/mapping): {missing}")
    # Ensure exact order; if some are missing, we still honor the order for those present
    pivot = pivot.loc[exist]
    fig_h = max(5, 0.4 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    sns.heatmap(pivot.values, annot=True, fmt='.2f', cmap='YlGnBu', vmin=1, vmax=5,
                xticklabels=pivot.columns.tolist(), yticklabels=pivot.index.tolist(), ax=ax)
    ax.set_title('Average rating by topic and dimension')
    plt.tight_layout()
    fig.savefig(output_dir / 'topic_avg_ratings_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def compute_overall_per_summary(merged: pd.DataFrame) -> pd.DataFrame:
    # Overall average rating per summary across dimensions
    overall_rating = merged.dropna(subset=['avg_rating']).groupby('summary_id')['avg_rating'].mean().reset_index(name='overall_avg_rating')
    # Overall win rate per summary across dimensions weighted by number of comparisons
    tmp = merged.dropna(subset=['win_rate', 'n_comparisons']).copy()
    # Vectorized sums to avoid groupby.apply FutureWarning
    weighted_wins_series = (tmp['win_rate'] * tmp['n_comparisons']).groupby(tmp['summary_id']).sum()
    sums = weighted_wins_series.reset_index(name='weighted_wins')
    comps = tmp.groupby('summary_id')['n_comparisons'].sum().reset_index(name='total_comparisons')
    overall = pd.merge(sums, comps, on='summary_id', how='outer')
    overall['overall_win_rate'] = overall.apply(lambda r: float(r['weighted_wins'] / r['total_comparisons']) if pd.notna(r['weighted_wins']) and r['total_comparisons'] > 0 else np.nan, axis=1)
    out = pd.merge(overall_rating, overall[['summary_id', 'overall_win_rate']], on='summary_id', how='outer')
    return out


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\s]+', '_', str(name))
    return name[:100]


def plot_topic_scatter(merged: pd.DataFrame, overall: pd.DataFrame, output_dir: Path) -> None:
    topics = sorted(merged['topic'].dropna().unique())
    by_topic = merged[['summary_id', 'topic']].drop_duplicates('summary_id').merge(overall, on='summary_id', how='left')

    # Overall scatter across all topics
    osub = by_topic.dropna(subset=['overall_avg_rating', 'overall_win_rate']).copy()
    if not osub.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(osub['overall_avg_rating'], osub['overall_win_rate'], s=12, alpha=0.7)
        # trend
        if osub['overall_avg_rating'].nunique() > 1:
            try:
                coef = np.polyfit(osub['overall_avg_rating'], osub['overall_win_rate'], 1)
                xs = np.linspace(osub['overall_avg_rating'].min(), osub['overall_avg_rating'].max(), 50)
                ys = coef[0] * xs + coef[1]
                ax.plot(xs, ys, color='red', linestyle='--', linewidth=1)
            except Exception:
                pass
        # correlations
        pr, pp = (np.nan, np.nan)
        sr, sp = (np.nan, np.nan)
        if osub['overall_avg_rating'].nunique() > 1 and osub['overall_win_rate'].nunique() > 1:
            try:
                pr, pp = stats.pearsonr(osub['overall_avg_rating'].values, osub['overall_win_rate'].values)
            except Exception:
                pr, pp = (np.nan, np.nan)
            try:
                sr, sp = stats.spearmanr(osub['overall_avg_rating'].values, osub['overall_win_rate'].values)
            except Exception:
                sr, sp = (np.nan, np.nan)
        ax.text(0.03, 0.97, f"Pearson r={pr:.3f} (p={pp:.2e})\nSpearman r={sr:.3f} (p={sp:.2e})",
                ha='left', va='top', transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlabel('Overall average rating')
        ax.set_ylabel('Overall win rate')
        ax.set_ylim(0, 1)
        ax.set_title('Overall: win rate vs rating (sample-level)')
        plt.tight_layout()
        fig.savefig(output_dir / 'topic_scatter_overall.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Per-topic scatter
    # Order topics by desired display order
    topics_disp = {t: map_topic_display(t) for t in topics}
    ordered_topics = [t for t in topics if topics_disp[t] in DESIRED_TOPIC_ORDER]
    # Iterate in this order
    for topic in ordered_topics:
        tsub = by_topic[by_topic['topic'] == topic].dropna(subset=['overall_avg_rating', 'overall_win_rate'])
        if tsub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(tsub['overall_avg_rating'], tsub['overall_win_rate'], s=12, alpha=0.7)
        # trend
        if tsub['overall_avg_rating'].nunique() > 1:
            try:
                coef = np.polyfit(tsub['overall_avg_rating'], tsub['overall_win_rate'], 1)
                xs = np.linspace(tsub['overall_avg_rating'].min(), tsub['overall_avg_rating'].max(), 50)
                ys = coef[0] * xs + coef[1]
                ax.plot(xs, ys, color='red', linestyle='--', linewidth=1)
            except Exception:
                pass
        # correlations
        pr, pp = (np.nan, np.nan)
        sr, sp = (np.nan, np.nan)
        if tsub['overall_avg_rating'].nunique() > 1 and tsub['overall_win_rate'].nunique() > 1:
            try:
                pr, pp = stats.pearsonr(tsub['overall_avg_rating'].values, tsub['overall_win_rate'].values)
            except Exception:
                pr, pp = (np.nan, np.nan)
            try:
                sr, sp = stats.spearmanr(tsub['overall_avg_rating'].values, tsub['overall_win_rate'].values)
            except Exception:
                sr, sp = (np.nan, np.nan)
        ax.text(0.03, 0.97, f"Pearson r={pr:.3f} (p={pp:.2e})\nSpearman r={sr:.3f} (p={sp:.2e})",
                ha='left', va='top', transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlabel('Overall average rating')
        ax.set_ylabel('Overall win rate')
        ax.set_ylim(0, 1)
        ax.set_title(f'{map_topic_display(topic)}: win rate vs rating (sample-level)')
        plt.tight_layout()
        fig.savefig(output_dir / f"topic_scatter_{sanitize_filename(map_topic_display(topic))}.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_topic_dimension_corr_heatmap(merged: pd.DataFrame, output_dir: Path) -> None:
    # For each topic and dimension, compute Pearson corr between sample-level avg_rating and win_rate
    dims = list(get_dimension_questions().keys())
    df = merged.dropna(subset=['avg_rating', 'win_rate']).copy()
    df['topic_display'] = df['topic'].apply(map_topic_display)

    corr_rows: List[Dict[str, float]] = []
    for (topic_disp, dim), g in df.groupby(['topic_display', 'dimension']):
        if len(g) >= 3 and g['avg_rating'].nunique() > 1 and g['win_rate'].nunique() > 1:
            try:
                pr, _ = stats.pearsonr(g['avg_rating'].values, g['win_rate'].values)
            except Exception:
                pr = np.nan
        else:
            pr = np.nan
        corr_rows.append({'topic_display': topic_disp, 'dimension': dim, 'pearson_r': pr})

    corr_df = pd.DataFrame(corr_rows)
    # Pivot to topics x dimensions
    pivot = corr_df.pivot_table(index='topic_display', columns='dimension', values='pearson_r', aggfunc='mean')
    # Reindex rows to the desired display order
    exist = [t for t in DESIRED_TOPIC_ORDER if t in pivot.index]
    pivot = pivot.loc[exist]
    # Reindex columns to dims
    pivot = pivot.reindex(columns=dims)

    fig_h = max(5, 0.4 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    sns.heatmap(pivot.values, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=pivot.columns.tolist(), yticklabels=pivot.index.tolist(), ax=ax)
    ax.set_title('Topic x Dimension: Pearson corr (rating vs win rate)')
    plt.tight_layout()
    fig.savefig(output_dir / 'topic_dimension_rating_winrate_corr_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_topic_corr_heatmap(merged: pd.DataFrame, overall: pd.DataFrame, output_dir: Path) -> None:
    # Compute per-topic Pearson correlation between overall_avg_rating and overall_win_rate
    by_topic = merged[['summary_id', 'topic']].drop_duplicates('summary_id').merge(overall, on='summary_id', how='left')
    corr_rows: List[Dict[str, float]] = []
    for topic, g in by_topic.groupby('topic'):
        sub = g.dropna(subset=['overall_avg_rating', 'overall_win_rate'])
        if len(sub) >= 3 and sub['overall_avg_rating'].nunique() > 1 and sub['overall_win_rate'].nunique() > 1:
            try:
                pr, _ = stats.pearsonr(sub['overall_avg_rating'].values, sub['overall_win_rate'].values)
            except Exception:
                pr = np.nan
        else:
            pr = np.nan
        corr_rows.append({'topic': topic, 'pearson_r': pr})
    corr_df = pd.DataFrame(corr_rows)

    # Order topics per desired order (OpenQA first then Binary)
    corr_df['topic_display'] = corr_df['topic'].apply(map_topic_display)
    corr_df = corr_df[corr_df['topic_display'].isin(DESIRED_TOPIC_ORDER)]
    corr_df = corr_df.set_index('topic_display').reindex(DESIRED_TOPIC_ORDER)

    mat = corr_df[['pearson_r']].values
    fig, ax = plt.subplots(figsize=(5, max(5, 0.5 * len(corr_df) + 1)))
    sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=['Pearson r'], yticklabels=corr_df.index.tolist(), ax=ax)
    ax.set_title('Topic correlation: win rate vs rating')
    plt.tight_layout()
    fig.savefig(output_dir / 'topic_winrate_rating_corr_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading and computing sample-level metrics...')
    sample_metrics = compute_sample_level_metrics()
    print(f'Sample metrics rows: {len(sample_metrics)}')

    print('Merging topics...')
    merged = merge_with_topics(sample_metrics)
    print(f'Merged rows: {len(merged)} ; topics: {merged["topic"].nunique()}')

    print('Plotting topic average ratings heatmap...')
    plot_topic_avg_ratings(merged, OUTPUT_DIR)

    print('Computing overall per-summary metrics for correlations...')
    overall = compute_overall_per_summary(merged)
    print('Plotting topic correlation heatmap...')
    plot_topic_corr_heatmap(merged, overall, OUTPUT_DIR)
    print('Plotting topic x dimension correlation heatmap...')
    plot_topic_dimension_corr_heatmap(merged, OUTPUT_DIR)

    print('Done.')


if __name__ == '__main__':
    main()


