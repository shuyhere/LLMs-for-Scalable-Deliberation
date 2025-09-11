#!/usr/bin/env python3
"""
Analyze relationship between comment count and (average rating, win rate)
at both comments-level (per summary) and model-level, across four dimensions.

Data sources:
- Raw summaries with comment counts: annotation/summary-rating/data_files/raw/summaries_V0903_for_humanstudy_simple.csv
- Annotation outputs: annotation/summary-rating/annotation_output/full/annotated_instances.csv
- Triplet metadata: annotation/summary-rating/data_files/processed/sum_humanstudy_triplet_full_ring.csv

Outputs (under results/dataset_visulization/analysis_annotation/comments_analysis):
- CSVs: per-summary merged metrics; per-model aggregated metrics; correlation tables
- Plots: scatter plots with regression lines; heatmaps of Pearson correlations
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


PROJECT_ROOT = Path('/ibex/project/c2328/LLMs-Scalable-Deliberation')
RAW_SUMMARIES_CSV = PROJECT_ROOT / 'annotation/summary-rating/data_files/raw/summaries_V0903_for_humanstudy_simple.csv'
ANNOTATED_CSV = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full/annotated_instances.csv'
TRIPLET_CSV = PROJECT_ROOT / 'annotation/summary-rating/data_files/processed/sum_humanstudy_triplet_full_ring.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation/comments_analysis'


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

    # Map each comparison triplet to A/B summary ids
    trip_comp = trip[trip['type'] == 'comparison'][['id', 'summary_a_id', 'summary_b_id', 'model_a', 'model_b']]
    comp_rows = comp_rows.merge(trip_comp, left_on='instance_id', right_on='id', how='left')
    comp_rows.drop(columns=['id'], inplace=True)

    base_to_summary_a: Dict[str, str] = {}
    for _, row in comp_rows.dropna(subset=['summary_a_id']).iterrows():
        base_to_summary_a[row['triplet_base']] = row['summary_a_id']

    # Ratings attributed to summary A for that triplet
    for dim, qs in dims.items():
        rating_rows[f'rating_{dim}'] = rating_rows.apply(lambda r: _rating_value_from_row(r, qs['rating']), axis=1)

    rating_records: List[Dict[str, object]] = []
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

    # Win rates per summary by expanding A/B sides
    comp_records: List[Dict[str, object]] = []
    for dim, qs in dims.items():
        comp_rows[f'chosenA_{dim}'] = comp_rows.apply(lambda r: _comparison_choice_a_from_row(r, qs['comparison']), axis=1)
        nonnull = comp_rows[f'chosenA_{dim}'].dropna()
        if len(nonnull) > 0:
            print(f"[DEBUG] chosenA_{dim}: count={len(nonnull)}, mean(A)= {float(nonnull.mean()):.3f}, zeros={int((nonnull==0).sum())}, ones={int((nonnull==1).sum())}")
        sub = comp_rows.dropna(subset=[f'chosenA_{dim}', 'summary_a_id', 'summary_b_id'])

        # Model-level debug: verify per-model win aggregation on raw comparisons (before summary aggregation)
        if not sub.empty:
            dbg_a = sub[['instance_id', 'model_a', 'model_b', f'chosenA_{dim}']].head(5)
            print(f"[DEBUG] sample comparisons ({dim}):\n" + str(dbg_a))
            a_side = sub[['model_a', f'chosenA_{dim}']].rename(columns={'model_a': 'model', f'chosenA_{dim}': 'win'})
            b_side = sub[['model_b', f'chosenA_{dim}']].rename(columns={'model_b': 'model'})
            b_side['win'] = 1.0 - b_side[f'chosenA_{dim}']
            b_side = b_side[['model', 'win']]
            both = pd.concat([a_side[['model', 'win']], b_side], ignore_index=True)
            model_stats = both.groupby('model')['win'].agg(['sum', 'count', 'mean']).sort_values('count', ascending=False).head(10)
            print(f"[DEBUG] model win stats head ({dim}):\n" + str(model_stats))
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
    # Debug sample_win summary per dimension
    if not sample_win.empty:
        dbg = sample_win.groupby('dimension').agg(n_rows=('summary_id','size'),
                                                 n_summaries=('summary_id','nunique'),
                                                 total_comparisons=('n_comparisons','sum'),
                                                 mean_win_rate=('win_rate','mean'))
        print("[DEBUG] sample_win summary by dimension:\n" + str(dbg))

    # Merge
    sample_metrics = pd.merge(sample_avg_rating, sample_win, on=['summary_id', 'dimension'], how='outer')
    return sample_metrics


def merge_with_raw_comments(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(RAW_SUMMARIES_CSV)
    if 'comment_num' not in raw.columns:
        raise KeyError("comment_num column not found in RAW_SUMMARIES_CSV")
    # Ensure numeric type for comment_num
    raw['comment_num'] = pd.to_numeric(raw['comment_num'], errors='coerce')
    raw_sel = raw[['id', 'model', 'comment_num']].rename(columns={'id': 'summary_id'})
    merged = pd.merge(sample_metrics, raw_sel, on='summary_id', how='left')
    # Report basic stats to stdout
    total_raw = len(raw_sel)
    nonnull_raw = raw_sel['comment_num'].notna().sum()
    print(f"RAW summaries rows: {total_raw}, with non-null comment_num: {nonnull_raw}")
    print(f"Merged rows: {len(merged)}, with non-null comment_num: {merged['comment_num'].notna().sum()}")
    # Show available exact levels counts for quick sanity
    levels_preview = [10, 30, 50, 70, 90]
    lvl_counts = merged[merged['comment_num'].isin(levels_preview)].groupby('comment_num').size()
    print(f"Exact comment_num level counts (all dims, row-wise): {lvl_counts.to_dict()}")
    return merged


def compute_correlations_comments_level(merged: pd.DataFrame) -> pd.DataFrame:
    dims = list(get_dimension_questions().keys())
    rows: List[Dict[str, object]] = []
    for dim in dims:
        sub = merged[(merged['dimension'] == dim)].dropna(subset=['comment_num'])
        # rating corr
        sr = sub.dropna(subset=['avg_rating'])
        if len(sr) >= 3 and sr['avg_rating'].nunique() > 1 and sr['comment_num'].nunique() > 1:
            try:
                pr, pp = stats.pearsonr(sr['comment_num'].values, sr['avg_rating'].values)
            except Exception:
                pr, pp = np.nan, np.nan
        else:
            pr, pp = np.nan, np.nan
        # win rate corr
        sw = sub.dropna(subset=['win_rate'])
        if len(sw) >= 3 and sw['win_rate'].nunique() > 1 and sw['comment_num'].nunique() > 1:
            try:
                pr_w, pp_w = stats.pearsonr(sw['comment_num'].values, sw['win_rate'].values)
            except Exception:
                pr_w, pp_w = np.nan, np.nan
        else:
            pr_w, pp_w = np.nan, np.nan
        rows.append({'level': 'comments', 'dimension': dim,
                     'pearson_comment_vs_avg_rating': pr, 'p_rating': pp,
                     'pearson_comment_vs_win_rate': pr_w, 'p_win': pp_w,
                     'n_rating': int(len(sr)), 'n_win': int(len(sw))})
    return pd.DataFrame(rows)


def aggregate_model_level(merged: pd.DataFrame) -> pd.DataFrame:
    # Aggregate per model and dimension
    agg = merged.groupby(['model', 'dimension']).agg({
        'avg_rating': 'mean',
        'win_rate': 'mean',
        'comment_num': 'mean',
        'summary_id': 'count',
    }).reset_index().rename(columns={'summary_id': 'n_summaries'})
    return agg


def compute_correlations_model_level(agg: pd.DataFrame) -> pd.DataFrame:
    dims = list(get_dimension_questions().keys())
    rows: List[Dict[str, object]] = []
    for dim in dims:
        sub = agg[agg['dimension'] == dim]
        # rating corr across models
        sr = sub.dropna(subset=['avg_rating', 'comment_num'])
        if len(sr) >= 3 and sr['avg_rating'].nunique() > 1 and sr['comment_num'].nunique() > 1:
            try:
                pr, pp = stats.pearsonr(sr['comment_num'].values, sr['avg_rating'].values)
            except Exception:
                pr, pp = np.nan, np.nan
        else:
            pr, pp = np.nan, np.nan
        # win rate corr across models
        sw = sub.dropna(subset=['win_rate', 'comment_num'])
        if len(sw) >= 3 and sw['win_rate'].nunique() > 1 and sw['comment_num'].nunique() > 1:
            try:
                pr_w, pp_w = stats.pearsonr(sw['comment_num'].values, sw['win_rate'].values)
            except Exception:
                pr_w, pp_w = np.nan, np.nan
        else:
            pr_w, pp_w = np.nan, np.nan
        rows.append({'level': 'model', 'dimension': dim,
                     'pearson_comment_vs_avg_rating': pr, 'p_rating': pp,
                     'pearson_comment_vs_win_rate': pr_w, 'p_win': pp_w,
                     'n_models': int(len(sub))})
    return pd.DataFrame(rows)


def plot_scatter_comments_level(merged: pd.DataFrame, output_dir: Path) -> None:
    dims = list(get_dimension_questions().keys())
    for dim in dims:
        sub = merged[(merged['dimension'] == dim)].copy()
        if sub.empty:
            continue
        # avg rating vs comment_num
        sr = sub.dropna(subset=['avg_rating', 'comment_num'])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sr['comment_num'], sr['avg_rating'], s=10, alpha=0.7)
        if len(sr) >= 3 and sr['comment_num'].nunique() > 1:
            try:
                coef = np.polyfit(sr['comment_num'], sr['avg_rating'], 1)
                xs = np.linspace(sr['comment_num'].min(), sr['comment_num'].max(), 50)
                ys = coef[0] * xs + coef[1]
                ax.plot(xs, ys, color='red', linestyle='--', linewidth=1)
            except Exception:
                pass
        ax.set_xlabel('Comment number')
        ax.set_ylabel('Average rating')
        ax.set_title(f'Comments vs Avg rating — {dim}')
        plt.tight_layout()
        fig.savefig(output_dir / f'comments_vs_avg_rating_{dim}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # win rate vs comment_num
        sw = sub.dropna(subset=['win_rate', 'comment_num'])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sw['comment_num'], sw['win_rate'], s=10, alpha=0.7)
        if len(sw) >= 3 and sw['comment_num'].nunique() > 1:
            try:
                coef = np.polyfit(sw['comment_num'], sw['win_rate'], 1)
                xs = np.linspace(sw['comment_num'].min(), sw['comment_num'].max(), 50)
                ys = coef[0] * xs + coef[1]
                ax.plot(xs, ys, color='red', linestyle='--', linewidth=1)
            except Exception:
                pass
        ax.set_xlabel('Comment number')
        ax.set_ylabel('Win rate')
        ax.set_ylim(0, 1)
        ax.set_title(f'Comments vs Win rate — {dim}')
        plt.tight_layout()
        fig.savefig(output_dir / f'comments_vs_win_rate_{dim}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_heatmap_correlations(corr_df: pd.DataFrame, title: str, output_path: Path) -> None:
    dims = list(get_dimension_questions().keys())
    # Build 2 columns: Pearson(comment, avg_rating) and Pearson(comment, win_rate)
    mat = []
    for dim in dims:
        sub = corr_df[corr_df['dimension'] == dim]
        if sub.empty:
            mat.append([np.nan, np.nan])
        else:
            mat.append([
                float(sub.iloc[0]['pearson_comment_vs_avg_rating']) if pd.notna(sub.iloc[0]['pearson_comment_vs_avg_rating']) else np.nan,
                float(sub.iloc[0]['pearson_comment_vs_win_rate']) if pd.notna(sub.iloc[0]['pearson_comment_vs_win_rate']) else np.nan,
            ])
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.6 * len(dims) + 1)))
    sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=['Pearson r (rating)', 'Pearson r (win rate)'], yticklabels=dims, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Computing per-summary metrics...')
    sample_metrics = compute_sample_level_metrics()
    print(f'Per-summary metrics: {len(sample_metrics)} rows')

    print('Merging with raw comment numbers...')
    merged = merge_with_raw_comments(sample_metrics)
    # No CSV saving per request

    print('Computing comments-level correlations...')
    corr_comments = compute_correlations_comments_level(merged)
    # No CSV saving per request

    print('Plotting comments-level scatter and heatmap...')
    plot_scatter_comments_level(merged, OUTPUT_DIR)
    plot_heatmap_correlations(corr_comments, 'Comments-level Pearson correlations', OUTPUT_DIR / 'comments_level_correlations_heatmap.pdf')

    print('Aggregating model-level metrics...')
    model_agg = aggregate_model_level(merged)
    # No CSV saving per request

    print('Computing model-level correlations...')
    corr_model = compute_correlations_model_level(model_agg)
    # No CSV saving per request

    print('Plotting model-level heatmap...')
    plot_heatmap_correlations(corr_model, 'Model-level Pearson correlations', OUTPUT_DIR / 'model_level_correlations_heatmap.pdf')

    # Fixed comment levels (treat as exact values, not ranges)
    dims = list(get_dimension_questions().keys())

    # Fixed comment levels (treat as exact values, not ranges)
    print('Computing fixed-level (exact comment_num) correlations and means...')
    levels = [10, 30, 50, 70, 90]

    # Correlation between avg_rating and win_rate within each exact comment_num level
    rows_corr_rw = []
    rows_mean_rating_lv = []
    rows_mean_win_lv = []
    for dim in dims:
        sub_all = merged[merged['dimension'] == dim]
        for lv in levels:
            sub = sub_all[sub_all['comment_num'] == lv]
            # corr between avg_rating and win_rate
            sw = sub.dropna(subset=['avg_rating', 'win_rate'])
            if len(sw) >= 3 and sw['avg_rating'].nunique() > 1 and sw['win_rate'].nunique() > 1:
                try:
                    pr_rw, _ = stats.pearsonr(sw['avg_rating'].values, sw['win_rate'].values)
                except Exception:
                    pr_rw = np.nan
            else:
                pr_rw = np.nan
            rows_corr_rw.append({'dimension': dim, 'level': lv, 'pearson_r': pr_rw})

            # mean rating (simple mean)
            mean_rating = float(sub['avg_rating'].mean()) if len(sub) > 0 else np.nan
            rows_mean_rating_lv.append({'dimension': dim, 'level': lv, 'mean_rating': mean_rating})

            # mean win rate (weighted by n_comparisons)
            sum_comps = float(sub['n_comparisons'].sum()) if 'n_comparisons' in sub.columns else 0.0
            if len(sub) > 0 and sum_comps > 0:
                mean_win = float((sub['win_rate'] * sub['n_comparisons']).sum() / sum_comps)
            else:
                mean_win = np.nan
            rows_mean_win_lv.append({'dimension': dim, 'level': lv, 'mean_win_rate': mean_win})
            # Debug per level/dim summary
            n_summaries = sub['summary_id'].nunique()
            total_rows = len(sub)
            wins_weighted = float((sub['win_rate'] * sub['n_comparisons']).sum()) if sum_comps > 0 else np.nan
            print(f"[DEBUG] level={lv}, dim={dim}: n_rows={total_rows}, n_summaries={n_summaries}, sum_comparisons={sum_comps:.0f}, weighted_wins={wins_weighted if not np.isnan(wins_weighted) else 'nan'}, mean_win={mean_win if not np.isnan(mean_win) else 'nan'}, mean_rating={mean_rating if not np.isnan(mean_rating) else 'nan'}")

    corr_rw_df = pd.DataFrame(rows_corr_rw)
    mean_rating_lv_df = pd.DataFrame(rows_mean_rating_lv)
    mean_win_lv_df = pd.DataFrame(rows_mean_win_lv)

    # Save fixed-level CSVs
    corr_rw_csv = OUTPUT_DIR / 'fixed_levels_corr_rating_vs_winrate.csv'
    mean_rating_lv_csv = OUTPUT_DIR / 'fixed_levels_mean_ratings.csv'
    mean_win_lv_csv = OUTPUT_DIR / 'fixed_levels_mean_winrates.csv'
    # No CSV saving per request

    # Heatmap: y = dimensions, x = levels (corr between avg_rating and win_rate)
    def heatmap_fixed_levels(df_long: pd.DataFrame, title: str, out_name: str):
        pivot = df_long.pivot_table(index='dimension', columns='level', values='pearson_r', aggfunc='mean')
        pivot = pivot.reindex(index=dims, columns=levels)
        mat = pivot.values
        fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(dims) + 1)))
        sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    xticklabels=pivot.columns.astype(int).tolist(), yticklabels=pivot.index.tolist(), ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / out_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

    heatmap_fixed_levels(corr_rw_df, 'Pearson corr (avg rating vs win rate) by exact comment_num', 'fixed_levels_corr_rating_vs_winrate_heatmap.pdf')

    # Grouped bars for fixed levels
    def grouped_bar_fixed(df_long: pd.DataFrame, value_col: str, ylabel: str, title: str, out_name: str, ylim=None):
        pivot = df_long.pivot_table(index='level', columns='dimension', values=value_col, aggfunc='mean')
        pivot = pivot.reindex(index=levels, columns=dims)
        x = np.arange(len(levels))
        width = 0.18
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, dim in enumerate(dims):
            yvals = pivot[dim].values
            bars = ax.bar(x + (i - 1.5) * width, yvals, width=width, label=dim)
            # annotate value on each bar
            for bx, by in zip(x + (i - 1.5) * width, yvals):
                if not np.isnan(by):
                    ax.text(bx, by + 0.01 if value_col == 'mean_win_rate' else by + 0.02, f"{by:.2f}", ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in levels])
        ax.set_xlabel('Exact comment number')
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.legend(ncol=2, title='Dimension')
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / out_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

    grouped_bar_fixed(mean_rating_lv_df, 'mean_rating', 'Average rating', 'Average rating by exact comment number', 'fixed_levels_mean_ratings_bars.pdf', None)
    grouped_bar_fixed(mean_win_lv_df, 'mean_win_rate', 'Average win rate', 'Average win rate by exact comment number', 'fixed_levels_mean_winrates_bars.pdf', (0, 1))

    # Counts of summaries per exact comment level (unique summaries)
    uniq = merged[['summary_id', 'comment_num']].drop_duplicates('summary_id')
    counts = uniq[uniq['comment_num'].isin(levels)].groupby('comment_num').size().reindex(levels, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([str(x) for x in levels], counts.values)
    ax.set_xlabel('Exact comment number')
    ax.set_ylabel('Number of summaries')
    ax.set_title('Counts of summaries by exact comment number')
    # annotate counts on bars
    for rect, val in zip(bars, counts.values):
        ax.text(rect.get_x() + rect.get_width() / 2.0, val + max(1, val * 0.01), str(int(val)), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fixed_levels_summary_counts_bars.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Overall (across dimensions) weighted mean win rate by level
    sw_all = merged[merged['comment_num'].isin(levels)].dropna(subset=['win_rate', 'n_comparisons'])
    overall_win = sw_all.groupby('comment_num').apply(
        lambda g: float((g['win_rate'] * g['n_comparisons']).sum() / g['n_comparisons'].sum()) if g['n_comparisons'].sum() > 0 else np.nan
    ).reindex(levels)
    fig, ax = plt.subplots(figsize=(8, 4))
    ow = overall_win.values
    bars = ax.bar([str(x) for x in levels], ow)
    ax.set_xlabel('Exact comment number')
    ax.set_ylabel('Average win rate (weighted)')
    ax.set_ylim(0, 1)
    ax.set_title('Overall average win rate by exact comment number')
    # annotate values
    for rect, by in zip(bars, ow):
        if not np.isnan(by):
            ax.text(rect.get_x() + rect.get_width() / 2.0, by + 0.01, f"{by:.2f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fixed_levels_overall_mean_winrates_bars.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Per-model averages by exact comment number (collapsing dimensions)
    print('Computing per-model averages by exact comment number (overall across dimensions)...')
    mfilter = merged[merged['comment_num'].isin(levels)].copy()
    mfilter['n_comparisons'] = mfilter['n_comparisons'].fillna(0)

    # Weighted win rate across dimensions per model & level
    def _weighted_mean(g):
        w = g['n_comparisons'].sum()
        return float((g['win_rate'] * g['n_comparisons']).sum() / w) if w > 0 else np.nan

    model_win = mfilter.dropna(subset=['win_rate']).groupby(['model', 'comment_num']).apply(_weighted_mean).reset_index(name='win_rate')
    # Average rating across dimensions per model & level (simple mean)
    model_rating = mfilter.dropna(subset=['avg_rating']).groupby(['model', 'comment_num'])['avg_rating'].mean().reset_index()
    model_avgs = pd.merge(model_win, model_rating, on=['model', 'comment_num'], how='outer')

    # Debug: sample
    print('[DEBUG] model-level overall averages sample:')
    print(model_avgs.head(12))

    # Plot per-model overview (two y-axes): win rate (0-1) and avg rating (1-5)
    for model in sorted(model_avgs['model'].dropna().unique()):
        sub = model_avgs[model_avgs['model'] == model].sort_values('comment_num')
        if sub.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()
        ax1.plot(sub['comment_num'], sub['win_rate'], marker='o', color='tab:blue', label='Win rate')
        ax2.plot(sub['comment_num'], sub['avg_rating'], marker='s', color='tab:orange', label='Avg rating')
        ax1.set_ylim(0, 1)
        ax2.set_ylim(1, 5)
        ax1.set_xlabel('Exact comment number')
        ax1.set_ylabel('Win rate')
        ax2.set_ylabel('Average rating')
        ax1.set_title(f'{model} — averages by comment number')
        ax1.set_xticks(levels)
        # Legend combining both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f'fixed_levels_model_overview_{model}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Overall averages across models per level
    # Weighted win rate across models by total comparisons; average rating simple mean across models
    # Prepare weights per model-level group
    weight_by_group = mfilter.groupby(['model', 'comment_num'])['n_comparisons'].sum().reset_index(name='total_comparisons')
    model_win_w = pd.merge(model_win, weight_by_group, on=['model', 'comment_num'], how='left')
    overall_win = model_win_w.groupby('comment_num').apply(lambda g: float((g['win_rate'] * g['total_comparisons']).sum() / g['total_comparisons'].sum()) if g['total_comparisons'].sum() > 0 else np.nan).reset_index(name='win_rate')
    overall_rating = model_rating.groupby('comment_num')['avg_rating'].mean().reset_index()
    overall = pd.merge(overall_win, overall_rating, on='comment_num', how='outer').sort_values('comment_num')

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(overall['comment_num'], overall['win_rate'], marker='o', color='tab:blue', label='Win rate (overall)')
    ax2.plot(overall['comment_num'], overall['avg_rating'], marker='s', color='tab:orange', label='Avg rating (overall)')
    ax1.set_ylim(0, 1)
    ax2.set_ylim(1, 5)
    ax1.set_xlabel('Exact comment number')
    ax1.set_ylabel('Win rate')
    ax2.set_ylabel('Average rating')
    ax1.set_title('Overall averages by comment number')
    ax1.set_xticks(levels)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fixed_levels_overall_overview.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('Done.')


if __name__ == '__main__':
    main()


