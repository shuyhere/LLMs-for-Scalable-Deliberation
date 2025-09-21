#!/usr/bin/env python3
"""
Plot per-model average win rate across four dimensions from human annotations.

Reads JSONL annotation files and triplet metadata, computes for each model and
dimension the probability of winning in pairwise comparisons, and produces a
grouped bar chart plus a CSV of the underlying data.
"""

from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


PROJECT_ROOT = Path('/ibex/project/c2328/LLMs-Scalable-Deliberation')
ANNOTATED_DIR = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full_augment'
TRIPLET_CSV = PROJECT_ROOT / 'annotation/summary-rating/data_files/processed/sum_humanstudy_triplet_full_ring_augmented.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation_winrate'


def get_dimension_questions() -> Dict[str, Dict[str, str]]:
    return {
        'perspective': {
            'comparison': "Which summary is more representative of your perspective? ",  # Note: trailing space
        },
        'informativeness': {
            'comparison': "Which summary is more informative? ",  # Note: trailing space
        },
        'neutrality': {
            'comparison': "Which summary presents a more neutral and balanced view of the issue? ",  # Note: trailing space
        },
        'policy': {
            'comparison': "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?",
        },
    }


def get_dimension_display_names() -> Dict[str, str]:
    """Get display names for dimensions."""
    return {
        'perspective': 'Representiveness',
        'informativeness': 'Informativeness', 
        'neutrality': 'Neutrality',
        'policy': 'Policy Approval'
    }


def get_model_display_names() -> Dict[str, str]:
    """Get display names for models."""
    return {
        'web-rev-claude-opus-4-20250514': 'Claude-4-Opus',
        'qwen3-32b': 'Qwen3-32B',
        'gemini-2.5-pro': 'Gemini-2.5-Pro',
        'gpt-5': 'GPT-5',
        'deepseek-chat': 'DeepSeek-V3.1 (chat)'
    }


def _extract_triplet_base(instance_id: str) -> str:
    if not isinstance(instance_id, str):
        return None
    parts = instance_id.rsplit('_', 1)
    return parts[0] if len(parts) == 2 else instance_id


def load_annotated_data() -> pd.DataFrame:
    """Load annotation data from JSONL files in user directories."""
    all_data = []
    user_dirs = [d for d in ANNOTATED_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(user_dirs)} user directories")
    
    for user_dir in user_dirs:
        jsonl_file = user_dir / 'annotated_instances.jsonl'
        if jsonl_file.exists():
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        all_data.append(data)
    
    print(f"Loaded {len(all_data)} annotation records")
    return pd.DataFrame(all_data)


def _comparison_choice_a_from_row(row: pd.Series, comp_q: str) -> float:
    """Extract comparison choice from nested label_annotations structure."""
    label_annotations = row.get('label_annotations', {})
    if comp_q in label_annotations:
        question_data = label_annotations[comp_q]
        if isinstance(question_data, dict):
            values = list(question_data.values())
            if values:
                value_str = str(values[0])
                try:
                    value = float(value_str)
                    # Map 1-5 scale to 0-1: 1,2 -> 1.0 (A wins), 4,5 -> 0.0 (B wins), 3 -> 0.5 (neutral, to be filtered)
                    if value == 1 or value == 2:
                        return 1.0
                    elif value == 3:
                        return 0.5  # Neutral, will be filtered out
                    elif value == 4 or value == 5:
                        return 0.0
                    else:
                        return np.nan
                except (ValueError, TypeError):
                    return np.nan
        else:
            # Fallback for direct value
            value_str = str(question_data)
            try:
                value = float(value_str)
                if value == 1 or value == 2:
                    return 1.0
                elif value == 3:
                    return 0.5
                elif value == 4 or value == 5:
                    return 0.0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
    return np.nan


def compute_model_win_rates() -> pd.DataFrame:
    dims = get_dimension_questions()

    # Load annotation data from JSONL files
    ann = load_annotated_data()
    trip = pd.read_csv(TRIPLET_CSV)

    ann['triplet_base'] = ann['id'].apply(_extract_triplet_base)

    comp_rows = ann[ann['id'].str.contains('_comparison', na=False)].copy()
    trip_comp = trip[trip['type'] == 'comparison'][['id', 'model_a', 'model_b']]
    comp_rows = comp_rows.merge(trip_comp, left_on='id', right_on='id', how='left')
    comp_rows.drop(columns=['id'], inplace=True)

    # For each dimension, compute model wins by expanding A and B sides
    records: List[Dict[str, float]] = []
    for dim, qs in dims.items():
        comp_rows[f'chosenA_{dim}'] = comp_rows.apply(lambda r: _comparison_choice_a_from_row(r, qs['comparison']), axis=1)
        
        # Filter out neutral choices (0.5) and NaN values
        sub = comp_rows.dropna(subset=[f'chosenA_{dim}', 'model_a', 'model_b'])
        sub = sub[sub[f'chosenA_{dim}'] != 0.5]  # Remove "Both are about the same"
        
        if sub.empty:
            print(f"[WARNING] No valid comparison data found for dimension: {dim}")
            continue
            
        print(f"[DEBUG] {dim}: {len(sub)} valid comparisons")
        
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

    if not records:
        print("[ERROR] No valid comparison data found for any dimension!")
        return pd.DataFrame(columns=['model', 'win_rate', 'n_comparisons', 'dimension'])
        
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


def plot_win_rate_heatmap(win_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot heatmap of model win rates across all dimensions."""
    dims = list(get_dimension_questions().keys())
    models = sorted(win_df['model'].unique())
    
    # Create pivot table: models x dimensions
    pivot = win_df.pivot_table(index='model', columns='dimension', values='win_rate', aggfunc='mean')
    pivot = pivot.reindex(index=models, columns=dims)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(8, max(6, 0.4 * len(models))))
    
    # Use a colormap that goes from low (blue) to high (red) win rates
    # Adjust range to better show data distribution
    data_min = pivot.values.min()
    data_max = pivot.values.max()
    data_range = data_max - data_min
    
    # Set range to be slightly wider than actual data range for better contrast
    vmin = max(0, data_min - 0.05)
    vmax = min(1, data_max + 0.05)
    center = 0.5
    
    sns.heatmap(pivot.values, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (red=high win rate)
                vmin=vmin, 
                vmax=vmax,
                center=center,
                xticklabels=pivot.columns,
                yticklabels=pivot.index,
                ax=ax,
                cbar_kws={'label': 'Win Rate'})
    
    ax.set_title('Model Win Rates Across All Dimensions', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'model_win_rates_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved heatmap to: {output_dir / 'model_win_rates_heatmap.pdf'}")


def compute_model_vs_model_win_rates(win_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute model vs model win rate matrices for each dimension."""
    model_display_names = get_model_display_names()
    dimension_display_names = get_dimension_display_names()
    
    matrices = {}
    
    for dim in win_df['dimension'].unique():
        dim_data = win_df[win_df['dimension'] == dim]
        
        # Get unique models for this dimension
        models = sorted(dim_data['model'].unique())
        
        # Create model vs model matrix
        matrix = pd.DataFrame(index=models, columns=models, dtype=float)
        
        # Fill diagonal with 0.5 (self vs self)
        for model in models:
            matrix.loc[model, model] = 0.5
        
        # For each model, calculate win rate against all other models
        for model in models:
            model_data = dim_data[dim_data['model'] == model]
            if not model_data.empty:
                # Win rate is already calculated as average win rate for this model
                win_rate = model_data['win_rate'].iloc[0]
                
                # Fill this model's row with its win rate against all models
                for other_model in models:
                    if other_model != model:
                        matrix.loc[model, other_model] = win_rate
        
        # Apply display names
        matrix.index = [model_display_names.get(m, m) for m in matrix.index]
        matrix.columns = [model_display_names.get(m, m) for m in matrix.columns]
        
        matrices[dimension_display_names[dim]] = matrix
    
    return matrices


def plot_model_vs_model_matrices(win_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 2x2 subplot of model vs model win rate matrices for each dimension."""
    matrices = compute_model_vs_model_win_rates(win_df)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    dimensions = ['Representiveness', 'Informativeness', 'Neutrality', 'Policy Approval']
    
    for i, dim_name in enumerate(dimensions):
        if dim_name in matrices:
            matrix = matrices[dim_name]
            
            # Create heatmap with adjusted range for better contrast
            data_min = matrix.values.min()
            data_max = matrix.values.max()
            
            # Set range to be slightly wider than actual data range for better contrast
            vmin = max(0, data_min - 0.05)
            vmax = min(1, data_max + 0.05)
            center = 0.5
            
            sns.heatmap(matrix.values,
                       annot=True,
                       fmt='.3f',
                       cmap='RdYlBu_r',
                       vmin=vmin,
                       vmax=vmax,
                       center=center,
                       xticklabels=matrix.columns,
                       yticklabels=matrix.index,
                       ax=axes[i],
                       cbar_kws={'label': 'Win Rate'})
            
            axes[i].set_title(f'{dim_name}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Model (Column)', fontsize=12)
            axes[i].set_ylabel('Model (Row)', fontsize=12)
            
            # Rotate x-axis labels
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].tick_params(axis='y', rotation=0)
        else:
            axes[i].text(0.5, 0.5, f'No data for {dim_name}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{dim_name} (No Data)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'model_vs_model_win_rates.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved model vs model matrices to: {output_dir / 'model_vs_model_win_rates.pdf'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Computing model win rates...')
    win_df = compute_model_win_rates()
    
    if win_df.empty:
        print("[ERROR] No data to plot!")
        return
    
    csv_path = OUTPUT_DIR / 'model_win_rates.csv'
    win_df.to_csv(csv_path, index=False)
    print(f'Saved CSV to: {csv_path}')
    
    print('Plotting win rate heatmap...')
    plot_win_rate_heatmap(win_df, OUTPUT_DIR)
    print(f'Saved heatmap to: {OUTPUT_DIR / "model_win_rates_heatmap.pdf"}')
    
    print('Plotting model vs model matrices...')
    plot_model_vs_model_matrices(win_df, OUTPUT_DIR)
    print(f'Saved model vs model matrices to: {OUTPUT_DIR / "model_vs_model_win_rates.pdf"}')
    
    # Print summary statistics
    print('\n' + '='*50)
    print('SUMMARY STATISTICS')
    print('='*50)
    dimension_display_names = get_dimension_display_names()
    for dim in win_df['dimension'].unique():
        dim_data = win_df[win_df['dimension'] == dim]
        display_name = dimension_display_names.get(dim, dim)
        print(f'\n{display_name.upper()}:')
        print(f'  Models: {len(dim_data)}')
        print(f'  Win rate range: {dim_data["win_rate"].min():.3f} - {dim_data["win_rate"].max():.3f}')
        print(f'  Average win rate: {dim_data["win_rate"].mean():.3f}')
        print(f'  Total comparisons: {dim_data["n_comparisons"].sum()}')


if __name__ == '__main__':
    main()


