#!/usr/bin/env python3
"""
Model-level analysis of correlation between rating and comparison annotations.

This script mirrors the summary-level analysis but computes correlations per model.
It joins human annotation outputs with processed triplet metadata to attribute
each rating and comparison to specific models, then calculates point-biserial
and Spearman correlations by model and dimension.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# Plot style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


PROJECT_ROOT = Path('/ibex/project/c2328/LLMs-Scalable-Deliberation')

# Data sources
ANNOTATED_CSV = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full/annotated_instances.csv'
TRIPLET_CSV = PROJECT_ROOT / 'annotation/summary-rating/data_files/processed/sum_humanstudy_triplet_full_ring.csv'

# Outputs
OUTPUT_DIR = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation/model_level'


def get_dimension_questions() -> Dict[str, Dict[str, str]]:
    """Return mapping of dimension -> {rating question, comparison question}."""
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


def _rating_value_from_row(row: pd.Series, rating_q: str) -> float:
    """Extract 1-5 rating from annotated row for a dimension rating question.

    The CSV has five columns ending with :::scale_1..5. Return the selected scale
    index if present; otherwise NaN.
    """
    for i in range(1, 6):
        col = f"{rating_q}:::scale_{i}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            return float(i)
    return np.nan


def _comparison_choice_a_from_row(row: pd.Series, comp_q: str) -> float:
    """Return 1 if Summary A was chosen, 0 if Summary B was chosen, else NaN."""
    col_a = f"{comp_q}:::scale_1"
    col_b = f"{comp_q}:::scale_2"
    val_a = row.get(col_a, np.nan)
    val_b = row.get(col_b, np.nan)
    if pd.notna(val_a) and str(val_a).strip() != "":
        return 1.0
    if pd.notna(val_b) and str(val_b).strip() != "":
        return 0.0
    return np.nan


def _extract_triplet_base(instance_id: str) -> str:
    if not isinstance(instance_id, str):
        return None
    # triplet_123_rating -> triplet_123 ; triplet_123_comparison -> triplet_123
    parts = instance_id.rsplit('_', 1)
    return parts[0] if len(parts) == 2 else instance_id


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load annotated instances and triplet metadata."""
    ann = pd.read_csv(ANNOTATED_CSV)
    trip = pd.read_csv(TRIPLET_CSV)
    return ann, trip


def build_joined_frame(ann: pd.DataFrame, trip: pd.DataFrame) -> pd.DataFrame:
    """Join rating and comparison annotation rows per user and triplet base.

    Returns a wide DataFrame where each row corresponds to (user, triplet_base)
    with columns for rating (per dimension), chosen_A (per dimension), and model
    attribution for rating (which should equal model_a of the comparison).
    """
    dims = get_dimension_questions()

    # Split annotation rows
    ann = ann.copy()
    ann['triplet_base'] = ann['instance_id'].apply(_extract_triplet_base)

    rating_rows = ann[ann['instance_id'].str.contains('_rating', na=False)].copy()
    comp_rows = ann[ann['instance_id'].str.contains('_comparison', na=False)].copy()

    # Attach model info to rating rows via triplet id join
    trip_rating = trip[trip['type'] == 'rating'][['id', 'model']].rename(columns={'model': 'rating_model'})
    rating_rows = rating_rows.merge(trip_rating, left_on='instance_id', right_on='id', how='left')
    rating_rows.drop(columns=['id'], inplace=True)

    # Attach A/B models to comparison rows
    trip_comp = trip[trip['type'] == 'comparison'][['id', 'model_a', 'model_b']]
    comp_rows = comp_rows.merge(trip_comp, left_on='instance_id', right_on='id', how='left')
    comp_rows.drop(columns=['id'], inplace=True)

    # Compute rating values per dimension
    for dim, qs in dims.items():
        rating_rows[f'rating_{dim}'] = rating_rows.apply(lambda r: _rating_value_from_row(r, qs['rating']), axis=1)

    # Compute chosen_A per dimension
    for dim, qs in dims.items():
        comp_rows[f'chosenA_{dim}'] = comp_rows.apply(lambda r: _comparison_choice_a_from_row(r, qs['comparison']), axis=1)

    # Select necessary columns for merging
    rating_sel_cols = ['user', 'triplet_base', 'instance_id', 'rating_model'] + [f'rating_{d}' for d in dims]
    comp_sel_cols = ['user', 'triplet_base', 'instance_id', 'model_a', 'model_b'] + [f'chosenA_{d}' for d in dims]

    rating_sel = rating_rows[rating_sel_cols].rename(columns={'instance_id': 'rating_instance_id'})
    comp_sel = comp_rows[comp_sel_cols].rename(columns={'instance_id': 'comp_instance_id'})

    # Join per (user, triplet_base)
    joined = rating_sel.merge(comp_sel, on=['user', 'triplet_base'], how='inner')

    # Consistency check: rating_model should equal model_a for the joined triplet
    joined['model_match'] = (joined['rating_model'] == joined['model_a'])

    return joined


def compute_model_correlations(joined: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute per-model correlations between rating and comparison outcomes by dimension.

    Returns nested dict: correlations[model][dimension] -> metrics dict
    """
    dims = get_dimension_questions()
    correlations: Dict[str, Dict[str, Dict[str, float]]] = {}

    models = sorted(joined['rating_model'].dropna().unique())

    for model in models:
        correlations[model] = {}
        model_df = joined[joined['rating_model'] == model]

        for dim in dims.keys():
            ratings = model_df[f'rating_{dim}'].astype(float)
            chosen_a = model_df[f'chosenA_{dim}'].astype(float)

            mask = ratings.notna() & chosen_a.notna()
            x = ratings[mask].values
            y = chosen_a[mask].values

            if len(x) >= 3 and np.unique(y).size > 1:
                try:
                    rpb, p_rpb = stats.pointbiserialr(x, y)
                except Exception:
                    rpb, p_rpb = np.nan, np.nan
                try:
                    rho, p_s = stats.spearmanr(x, y)
                except Exception:
                    rho, p_s = np.nan, np.nan
            else:
                rpb, p_rpb, rho, p_s = np.nan, np.nan, np.nan, np.nan

            correlations[model][dim] = {
                'point_biserial_corr': float(rpb) if pd.notna(rpb) else np.nan,
                'point_biserial_p': float(p_rpb) if pd.notna(p_rpb) else np.nan,
                'spearman_corr': float(rho) if pd.notna(rho) else np.nan,
                'spearman_p': float(p_s) if pd.notna(p_s) else np.nan,
                'n_pairs': int(len(x)),
                'rating_mean': float(np.nanmean(x)) if len(x) else np.nan,
                'rating_std': float(np.nanstd(x)) if len(x) else np.nan,
                'chosenA_mean': float(np.nanmean(y)) if len(y) else np.nan,
                'chosenA_std': float(np.nanstd(y)) if len(y) else np.nan,
            }

    return correlations


def correlations_to_dataframe(corr: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for model, dim_map in corr.items():
        for dim, metrics in dim_map.items():
            row = {'model': model, 'dimension': dim}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmaps(corr_df: pd.DataFrame, output_dir: Path) -> None:
    dims = list(get_dimension_questions().keys())
    models = sorted(corr_df['model'].unique())

    # Prepare matrices (models x dims)
    mat_pb = np.array([
        [corr_df.query("model == @m and dimension == @d")['point_biserial_corr'].values[0]
         if not corr_df.query("model == @m and dimension == @d").empty else np.nan
         for d in dims]
        for m in models
    ])
    mat_sp = np.array([
        [corr_df.query("model == @m and dimension == @d")['spearman_corr'].values[0]
         if not corr_df.query("model == @m and dimension == @d").empty else np.nan
         for d in dims]
        for m in models
    ])

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.6 * len(models) + 2)))
    sns.heatmap(mat_pb, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=dims, yticklabels=models, ax=axes[0])
    axes[0].set_title('Point-biserial (model x dimension)')

    sns.heatmap(mat_sp, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=dims, yticklabels=models, ax=axes[1])
    axes[1].set_title('Spearman (model x dimension)')

    plt.tight_layout()
    fig.savefig(output_dir / 'rating_comparison_model_correlations_heatmaps.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_report(corr_df: pd.DataFrame, output_dir: Path) -> None:
    lines: List[str] = []
    lines.append('=' * 80)
    lines.append('MODEL-LEVEL RATING VS COMPARISON CORRELATION ANALYSIS')
    lines.append('=' * 80)

    for model in sorted(corr_df['model'].unique()):
        lines.append(f"\nMODEL: {model}")
        lines.append('-' * 40)
        sub = corr_df[corr_df['model'] == model]
        for _, row in sub.iterrows():
            dim = row['dimension']
            lines.append(f"  Dimension: {dim}")
            lines.append(f"    n_pairs: {int(row['n_pairs'])}")
            lines.append(f"    rating mean/std: {row['rating_mean']:.3f} / {row['rating_std']:.3f}")
            lines.append(f"    chosenA mean/std: {row['chosenA_mean']:.3f} / {row['chosenA_std']:.3f}")
            lines.append(f"    point-biserial: {row['point_biserial_corr']:.3f} (p={row['point_biserial_p']:.3e})")
            lines.append(f"    spearman:       {row['spearman_corr']:.3f} (p={row['spearman_p']:.3e})")

    (output_dir / 'rating_comparison_model_correlation_report.txt').write_text('\n'.join(lines), encoding='utf-8')


def compute_winrate_by_rating(joined: pd.DataFrame) -> pd.DataFrame:
    """Compute win rates (A chosen) by rating value 1-5, per model and dimension.

    Only use rows where the rated model matches comparison model_a.
    """
    dims = list(get_dimension_questions().keys())
    df = joined.copy()
    df = df[df['model_match'] == True]

    rows: List[Dict[str, float]] = []
    for model in sorted(df['rating_model'].dropna().unique()):
        sub_m = df[df['rating_model'] == model]
        for dim in dims:
            ratings = sub_m[f'rating_{dim}']
            choices = sub_m[f'chosenA_{dim}']
            mask = ratings.notna() & choices.notna()
            if mask.sum() == 0:
                continue
            r_vals = ratings[mask].astype(int)
            y = choices[mask].astype(float)
            for r in [1, 2, 3, 4, 5]:
                m2 = (r_vals == r)
                n = int(m2.sum())
                if n == 0:
                    win_rate = np.nan
                    wins = 0
                else:
                    wins = int(y[m2].sum())
                    win_rate = float(wins / n)
                rows.append({
                    'model': model,
                    'dimension': dim,
                    'rating_value': r,
                    'n_pairs': n,
                    'n_wins': wins,
                    'win_rate': win_rate,
                })
    return pd.DataFrame(rows)


def compute_monotonicity(winrate_df: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    """Assess monotonicity and association between rating and choice per model/dimension."""
    results: List[Dict[str, float]] = []
    dims = list(get_dimension_questions().keys())

    for model in sorted(winrate_df['model'].unique()):
        for dim in dims:
            wsub = winrate_df[(winrate_df['model'] == model) & (winrate_df['dimension'] == dim)]
            if wsub.empty:
                continue
            wsub = wsub.sort_values('rating_value')
            rates = wsub['win_rate'].values

            # Non-decreasing check across available (non-nan) bins
            valid_idx = ~np.isnan(rates)
            rates_v = rates[valid_idx]
            violations = 0
            for i in range(1, len(rates_v)):
                if rates_v[i] + 1e-12 < rates_v[i - 1]:
                    violations += 1

            # Spearman using individual-level data (rating vs chosenA)
            jsub = joined[(joined['rating_model'] == model) & (joined['model_match'] == True)]
            x = jsub[f'rating_{dim}'].astype(float)
            y = jsub[f'chosenA_{dim}'].astype(float)
            m = x.notna() & y.notna()
            if m.sum() >= 3 and np.unique(y[m]).size > 1:
                rho, pval = stats.spearmanr(x[m], y[m])
            else:
                rho, pval = np.nan, np.nan

            results.append({
                'model': model,
                'dimension': dim,
                'monotonic_non_decreasing': bool(violations == 0),
                'num_adjacent_violations': int(violations),
                'spearman_rating_vs_choice': float(rho) if pd.notna(rho) else np.nan,
                'spearman_p': float(pval) if pd.notna(pval) else np.nan,
                'total_pairs_used': int(m.sum()),
            })

    return pd.DataFrame(results)


def plot_winrate_by_rating_per_model(winrate_df: pd.DataFrame, output_dir: Path) -> None:
    dims = list(get_dimension_questions().keys())
    models = sorted(winrate_df['model'].unique())
    for model in models:
        sub = winrate_df[winrate_df['model'] == model]
        if sub.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        axes = axes.flatten()
        for idx, dim in enumerate(dims):
            dsub = sub[sub['dimension'] == dim].sort_values('rating_value')
            ax = axes[idx]
            ax.plot(dsub['rating_value'], dsub['win_rate'], marker='o')
            ax.set_title(f'{dim}')
            ax.set_xlabel('Rating (1-5)')
            ax.set_ylabel('Win rate (A chosen)')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_ylim(0, 1)
            # Annotate counts
            for x, y, n in zip(dsub['rating_value'], dsub['win_rate'], dsub['n_pairs']):
                if not np.isnan(y):
                    ax.text(x, y + 0.03, f'n={int(n)}', ha='center', va='bottom', fontsize=8)
        plt.suptitle(f'Win rate by rating — {model}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_dir / f'rating_comparison_model_winrate_by_rating_{model}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

def compute_model_winrate_and_avg_rating(joined: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model win rate and average rating per dimension.

    Returns DataFrame with columns: model, dimension, win_rate, n_comp, avg_rating, n_rating.
    """
    dims = list(get_dimension_questions().keys())

    # Ratings: only where the rating belongs to model_a for the same triplet
    ratings_records: List[Dict[str, float]] = []
    rated = joined[joined['model_match'] == True]
    for dim in dims:
        r = rated[["rating_model", f"rating_{dim}"]].dropna()
        g = r.groupby("rating_model")[f"rating_{dim}"]
        for model, ser in g:
            ratings_records.append({
                'model': model,
                'dimension': dim,
                'avg_rating': float(ser.mean()),
                'n_rating': int(ser.count()),
            })
    ratings_df = pd.DataFrame(ratings_records)

    # Win rates: derive wins for model_a and model_b from chosenA
    comp_records: List[Dict[str, float]] = []
    for dim in dims:
        ca = joined[f'chosenA_{dim}']
        valid = ca.notna() & joined['model_a'].notna() & joined['model_b'].notna()
        sub = joined[valid].copy()
        a_df = sub[['model_a', f'chosenA_{dim}']].rename(columns={'model_a': 'model', f'chosenA_{dim}': 'win'})
        b_df = sub[['model_b', f'chosenA_{dim}']].rename(columns={'model_b': 'model'})
        b_df['win'] = 1.0 - b_df[f'chosenA_{dim}']
        b_df = b_df[['model', 'win']]
        all_df = pd.concat([a_df[['model', 'win']], b_df], ignore_index=True)
        g = all_df.groupby('model')['win']
        for model, ser in g:
            comp_records.append({
                'model': model,
                'dimension': dim,
                'win_rate': float(ser.mean()),
                'n_comp': int(ser.count()),
            })
    win_df = pd.DataFrame(comp_records)

    return pd.merge(win_df, ratings_df, on=['model', 'dimension'], how='outer')

def corr_winrate_vs_avg_rating(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation between per-model win rate and average rating.

    Produces rows per dimension and an 'overall' aggregation across dimensions.
    """
    results: List[Dict[str, float]] = []
    dims = list(get_dimension_questions().keys())

    # Per-dimension correlations
    for dim in dims:
        sub = metrics_df[metrics_df['dimension'] == dim].dropna(subset=['win_rate', 'avg_rating'])
        models = sub['model'].unique()
        if len(models) >= 3:
            x = sub['avg_rating'].values
            y = sub['win_rate'].values
            try:
                pearson_r, pearson_p = stats.pearsonr(x, y)
            except Exception:
                pearson_r, pearson_p = np.nan, np.nan
            try:
                spearman_r, spearman_p = stats.spearmanr(x, y)
            except Exception:
                spearman_r, spearman_p = np.nan, np.nan
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = np.nan

        results.append({
            'dimension': dim,
            'num_models': int(len(models)),
            'pearson_r': float(pearson_r) if pd.notna(pearson_r) else np.nan,
            'pearson_p': float(pearson_p) if pd.notna(pearson_p) else np.nan,
            'spearman_r': float(spearman_r) if pd.notna(spearman_r) else np.nan,
            'spearman_p': float(spearman_p) if pd.notna(spearman_p) else np.nan,
        })

    # Overall aggregation per model across dimensions
    overall = metrics_df.dropna(subset=['win_rate', 'avg_rating']).groupby('model').agg({
        'win_rate': 'mean',
        'avg_rating': 'mean',
    }).reset_index()
    if len(overall) >= 3:
        try:
            pearson_r, pearson_p = stats.pearsonr(overall['avg_rating'].values, overall['win_rate'].values)
        except Exception:
            pearson_r, pearson_p = np.nan, np.nan
        try:
            spearman_r, spearman_p = stats.spearmanr(overall['avg_rating'].values, overall['win_rate'].values)
        except Exception:
            spearman_r, spearman_p = np.nan, np.nan
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan

    results.append({
        'dimension': 'overall',
        'num_models': int(len(overall)),
        'pearson_r': float(pearson_r) if pd.notna(pearson_r) else np.nan,
        'pearson_p': float(pearson_p) if pd.notna(pearson_p) else np.nan,
        'spearman_r': float(spearman_r) if pd.notna(spearman_r) else np.nan,
        'spearman_p': float(spearman_p) if pd.notna(spearman_p) else np.nan,
    })

    return pd.DataFrame(results)

def plot_corr_scatter(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plots: per-model win rate vs avg rating for each dimension and overall."""
    dims = list(get_dimension_questions().keys()) + ['overall']

    overall = metrics_df.dropna(subset=['win_rate', 'avg_rating']).groupby('model').agg({
        'win_rate': 'mean',
        'avg_rating': 'mean',
    }).reset_index()

    for dim in dims:
        if dim == 'overall':
            sub = overall.copy()
            title_dim = 'overall'
        else:
            sub = metrics_df[metrics_df['dimension'] == dim].dropna(subset=['win_rate', 'avg_rating']).copy()
            title_dim = dim
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sub['avg_rating'], sub['win_rate'])
        if len(sub) >= 2 and sub['avg_rating'].nunique() > 1:
            try:
                coef = np.polyfit(sub['avg_rating'], sub['win_rate'], 1)
                xs = np.linspace(sub['avg_rating'].min(), sub['avg_rating'].max(), 50)
                ys = coef[0] * xs + coef[1]
                ax.plot(xs, ys, color='red', linestyle='--', linewidth=1)
            except Exception:
                pass
        # Correlation annotations (Pearson and Spearman)
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan
        if len(sub) >= 3 and sub['avg_rating'].nunique() > 1 and sub['win_rate'].nunique() > 1:
            try:
                pearson_r, pearson_p = stats.pearsonr(sub['avg_rating'].values, sub['win_rate'].values)
            except Exception:
                pearson_r, pearson_p = np.nan, np.nan
            try:
                spearman_r, spearman_p = stats.spearmanr(sub['avg_rating'].values, sub['win_rate'].values)
            except Exception:
                spearman_r, spearman_p = np.nan, np.nan
        annot = f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e})\nSpearman r={spearman_r:.3f} (p={spearman_p:.2e})"
        ax.text(0.03, 0.97, annot, ha='left', va='top', transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        for _, r in sub.iterrows():
            ax.text(r['avg_rating'], r['win_rate'] + 0.01, r['model'], fontsize=7, ha='center')
        ax.set_xlabel('Average rating')
        ax.set_ylabel('Win rate')
        ax.set_title(f'Win rate vs avg rating — {title_dim}')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(output_dir / f'rating_comparison_model_corr_winrate_vs_rating_{title_dim}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading data...')
    ann, trip = load_data()
    print(f"Annotated rows: {len(ann)} ; Triplet rows: {len(trip)}")

    print('Building joined frame (user x triplet)...')
    joined = build_joined_frame(ann, trip)
    print(f"Joined rows: {len(joined)} ; Model match rate: {joined['model_match'].mean():.3f}")

    print('Computing correlations by model and dimension...')
    corr = compute_model_correlations(joined)
    corr_df = correlations_to_dataframe(corr)

    csv_path = OUTPUT_DIR / 'rating_comparison_model_correlations.csv'
    corr_df.to_csv(csv_path, index=False)
    print(f'Saved correlations CSV to: {csv_path}')

    print('Plotting heatmaps...')
    plot_heatmaps(corr_df, OUTPUT_DIR)
    print(f"Saved heatmaps to: {OUTPUT_DIR}")

    print('Saving text report...')
    save_report(corr_df, OUTPUT_DIR)
    print('Computing win-rate by rating and monotonicity...')
    winrate_df = compute_winrate_by_rating(joined)
    winrate_csv = OUTPUT_DIR / 'rating_comparison_model_winrate_by_rating.csv'
    winrate_df.to_csv(winrate_csv, index=False)
    print(f'Saved win-rate by rating CSV to: {winrate_csv}')

    mono_df = compute_monotonicity(winrate_df, joined)
    mono_csv = OUTPUT_DIR / 'rating_comparison_model_monotonicity.csv'
    mono_df.to_csv(mono_csv, index=False)
    print(f'Saved monotonicity CSV to: {mono_csv}')

    print('Plotting win-rate by rating per model...')
    plot_winrate_by_rating_per_model(winrate_df, OUTPUT_DIR)
    print('Computing per-model win rate and avg rating metrics...')
    metrics_df = compute_model_winrate_and_avg_rating(joined)
    metrics_csv = OUTPUT_DIR / 'rating_comparison_model_winrate_and_avg_rating.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    print(f'Saved model metrics CSV to: {metrics_csv}')

    print('Computing correlation between win rate and avg rating...')
    corr_wa_df = corr_winrate_vs_avg_rating(metrics_df)
    corr_wa_csv = OUTPUT_DIR / 'rating_comparison_model_corr_winrate_vs_avg_rating.csv'
    corr_wa_df.to_csv(corr_wa_csv, index=False)
    print(f'Saved correlation CSV to: {corr_wa_csv}')

    print('Plotting scatter correlations...')
    plot_corr_scatter(metrics_df, OUTPUT_DIR)
    print('Done.')


if __name__ == '__main__':
    main()


