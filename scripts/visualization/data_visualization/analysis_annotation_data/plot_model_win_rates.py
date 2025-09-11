#!/usr/bin/env python3
"""
Plot per-model average win rate across four dimensions from human annotations.

Reads annotated_instances.csv and triplet metadata, computes for each model and
dimension the probability of winning in pairwise comparisons, and produces a
grouped bar chart plus a CSV of the underlying data.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


PROJECT_ROOT = Path('/ibex/project/c2328/LLMs-Scalable-Deliberation')
ANNOTATED_CSV = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full/annotated_instances.csv'
TRIPLET_CSV = PROJECT_ROOT / 'annotation/summary-rating/data_files/processed/sum_humanstudy_triplet_full_ring.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation'


def get_dimension_questions() -> Dict[str, Dict[str, str]]:
    return {
        'perspective': {
            'comparison': "Which summary is more representative of your perspective?",
        },
        'informativeness': {
            'comparison': "Which summary is more informative?",
        },
        'neutrality': {
            'comparison': "Which summary presents a more neutral and balanced view of the issue?",
        },
        'policy': {
            'comparison': "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?",
        },
    }


def _extract_triplet_base(instance_id: str) -> str:
    if not isinstance(instance_id, str):
        return None
    parts = instance_id.rsplit('_', 1)
    return parts[0] if len(parts) == 2 else instance_id


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


def compute_model_win_rates() -> pd.DataFrame:
    dims = get_dimension_questions()

    ann = pd.read_csv(ANNOTATED_CSV)
    trip = pd.read_csv(TRIPLET_CSV)

    ann['triplet_base'] = ann['instance_id'].apply(_extract_triplet_base)

    comp_rows = ann[ann['instance_id'].str.contains('_comparison', na=False)].copy()
    trip_comp = trip[trip['type'] == 'comparison'][['id', 'model_a', 'model_b']]
    comp_rows = comp_rows.merge(trip_comp, left_on='instance_id', right_on='id', how='left')
    comp_rows.drop(columns=['id'], inplace=True)

    # For each dimension, compute model wins by expanding A and B sides
    records: List[Dict[str, float]] = []
    for dim, qs in dims.items():
        comp_rows[f'chosenA_{dim}'] = comp_rows.apply(lambda r: _comparison_choice_a_from_row(r, qs['comparison']), axis=1)
        sub = comp_rows.dropna(subset=[f'chosenA_{dim}', 'model_a', 'model_b'])
        # A side wins if chosenA==1
        a_df = sub[['model_a', f'chosenA_{dim}']].rename(columns={'model_a': 'model', f'chosenA_{dim}': 'win'})
        # B side wins if chosenA==0
        b_df = sub[['model_b', f'chosenA_{dim}']].rename(columns={'model_b': 'model'})
        b_df['win'] = 1.0 - b_df[f'chosenA_{dim}']
        b_df = b_df[['model', 'win']]
        all_df = pd.concat([a_df[['model', 'win']], b_df], ignore_index=True)
        stat = all_df.groupby('model')['win'].agg(['mean', 'count']).reset_index()
        stat['dimension'] = dim
        stat.rename(columns={'mean': 'win_rate', 'count': 'n_comparisons'}, inplace=True)
        records.append(stat)

    out = pd.concat(records, ignore_index=True)
    return out


def plot_grouped_bar(win_df: pd.DataFrame, output_dir: Path) -> None:
    dims = list(get_dimension_questions().keys())
    models = sorted(win_df['model'].unique())
    # pivot to (model x dimension)
    pivot = win_df.pivot_table(index='model', columns='dimension', values='win_rate', aggfunc='mean')
    pivot = pivot.reindex(index=models, columns=dims)

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(models) + 4), 6))
    width = 0.18
    x = np.arange(len(models))
    for i, dim in enumerate(dims):
        y = pivot[dim].values
        ax.bar(x + (i - 1.5) * width, y, width=width, label=dim)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Win rate')
    ax.set_ylim(0, 1)
    ax.set_title('Per-model average win rate across dimensions')
    ax.legend(title='Dimension', ncol=2)
    plt.tight_layout()
    fig.savefig(output_dir / 'model_win_rates.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Computing model win rates...')
    win_df = compute_model_win_rates()
    csv_path = OUTPUT_DIR / 'model_win_rates.csv'
    win_df.to_csv(csv_path, index=False)
    print(f'Saved CSV to: {csv_path}')
    print('Plotting grouped bar chart...')
    plot_grouped_bar(win_df, OUTPUT_DIR)
    print(f'Saved plot to: {OUTPUT_DIR / "model_win_rates.pdf"}')


if __name__ == '__main__':
    main()


