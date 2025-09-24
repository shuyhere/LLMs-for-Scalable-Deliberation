#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Global quiet flag (set in main)
QUIET = True

DIMENSIONS = [
    "perspective_representation",
    "informativeness",
    "neutrality_balance",
    "policy_approval",
]


def _normalize_id_variants(identifier: str) -> set:
    """Return a set of tolerant ID variants for matching.
    Handles: _A/_B suffixes, comparison<->rating, triplet prefixes, hyphen/underscore.
    """
    if not identifier:
        return set()
    ids = set()
    s = str(identifier)
    ids.add(s)
    # Hyphen/underscore variants
    ids.add(s.replace('-', '_'))
    ids.add(s.replace('_', '-'))
    # Strip _A/_B suffix
    if s.endswith('_A') or s.endswith('_B'):
        ids.add(s[:-2])
    # rating/comparison swaps
    if '_comparison' in s:
        ids.add(s.replace('_comparison', '_rating'))
    if '_rating' in s:
        ids.add(s.replace('_rating', '_comparison'))
    # Triplet base variants
    if 'triplet_' in s:
        # extract number and build rating ids
        try:
            num = s.split('triplet_')[1].split('_')[0]
            ids.add(f'triplet_{num}_rating')
            ids.add(f'triplet_{num}_rating_A')
            ids.add(f'triplet_{num}_rating_B')
            ids.add(f'triplet_{num}_comparison')
        except Exception:
            pass
    # Also produce variants on previously added forms
    expanded = set(ids)
    for x in list(expanded):
        if x.endswith('_A') or x.endswith('_B'):
            ids.add(x[:-2])
        if '_comparison' in x:
            ids.add(x.replace('_comparison', '_rating'))
        if '_rating' in x:
            ids.add(x.replace('_rating', '_comparison'))
        ids.add(x.replace('-', '_'))
        ids.add(x.replace('_', '-'))
    return ids

def _normalize_key_lookup(src: Dict, target_key: str) -> Optional[float]:
    """Robustly fetch a dimension value with aliasing and case-insensitive keys.
    Returns the raw value (not coerced), or None if not found.
    """
    if not isinstance(src, dict):
        return None
    # Exact
    if target_key in src:
        return src[target_key]
    # Case-insensitive direct match
    lower_map = {str(k).lower(): k for k in src.keys()}
    if target_key.lower() in lower_map:
        return src[lower_map[target_key.lower()]]
    # Aliases for common typos/variants
    aliases = {
        'neutrality_balance': [
            'neutralitybalance', 'neutral_balance', 'neutrality', 'neutral',
            'neutrality_and_balance', 'neutrality-balance', 'neutrality/balance',
            'neutraliy_balance', 'neutriality_balance', 'nurtrainlate_balance',
            'neutrality_balanced', 'balance_neutrality'
        ],
        'perspective_representation': ['perspective', 'representation', 'perspective_repr', 'perspective-representation'],
        'informativeness': ['informative', 'information', 'info', 'informativity'],
        'policy_approval': ['policy', 'approval', 'policyapproval', 'policy-approval']
    }
    for alias in aliases.get(target_key, []):
        if alias in src:
            return src[alias]
        if alias.lower() in lower_map:
            return src[lower_map[alias.lower()]]
    return None


def _coerce_numeric(value: Optional[object]) -> Optional[float]:
    """Convert a value to float if possible, else None."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        # Try string digits like '3 '
        try:
            s = str(value).strip()
            return float(s)
        except Exception:
            return None


def read_test_ids(test_jsonl: Path) -> set:
    """Read test set IDs and extract base triplet IDs for matching."""
    ids = set()
    base_ids = set()  # For matching with results that may not have _A/_B suffix
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Prefer 'id', fallback to 'annotation_id'
            candidate = None
            if "id" in obj:
                candidate = str(obj["id"])
            elif "annotation_id" in obj:
                candidate = str(obj["annotation_id"])
            if candidate:
                variants = _normalize_id_variants(candidate)
                ids.update(variants)
                base_ids.update(variants)
    
    # Return both full IDs and base IDs for flexible matching
    return ids | base_ids


def find_json_results(results_dir: Path) -> List[Path]:
    out: List[Path] = []
    for root, _, files in os.walk(results_dir):
        for fn in files:
            if fn.endswith(".json"):
                out.append(Path(root) / fn)
    return out


def extract_rating_pairs(json_path: Path, test_ids: set) -> pd.DataFrame:
    """Extract rating task results for test set items."""
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    
    model = data.get("experiment_metadata", {}).get("model", Path(json_path).stem)
    rows: List[Dict] = []
    
    matched_count = 0
    total_count = 0
    
    for item in data.get("rating_results", []):
        total_count += 1
        ann_id = str(item.get("annotation_id") or item.get("id") or "")
        if not ann_id:
            continue
        
        # Check tolerant variants against test_ids
        matched = False
        for v in _normalize_id_variants(ann_id):
            if v in test_ids:
                matched = True
                break
        
        if not matched:
            continue
            
        matched_count += 1
        human_raw = item.get("human_ratings", {})
        llm_raw = (item.get("llm_result", {}) or {}).get("ratings", {})
        # Normalize keys and coerce values to numeric where possible
        human = {}
        llm = {}
        for k in DIMENSIONS:
            human[k] = _coerce_numeric(_normalize_key_lookup(human_raw, k))
            llm[k] = _coerce_numeric(_normalize_key_lookup(llm_raw, k))
        
        # Ensure all dims present
        # Keep rows where at least one dimension pair is available
        if not any(human[k] is not None and llm[k] is not None for k in DIMENSIONS):
            continue
            
        row = {"model": model, "annotation_id": ann_id, "task": "rating"}
        for k in DIMENSIONS:
            row[f"human_{k}"] = human[k]
            row[f"llm_{k}"] = llm[k]
        rows.append(row)
    
    if matched_count > 0 and not QUIET:
        print(f"  {json_path.name}: Matched {matched_count}/{total_count} rating items")
    
    return pd.DataFrame(rows)


def extract_comparison_pairs(json_path: Path, test_ids: set) -> pd.DataFrame:
    """Extract comparison task results for test set items."""
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    
    model = data.get("experiment_metadata", {}).get("model", Path(json_path).stem)
    rows: List[Dict] = []
    
    matched_count = 0
    total_count = 0
    
    for item in data.get("comparison_results", []):
        total_count += 1
        ann_id = str(item.get("annotation_id") or item.get("id") or "")
        if not ann_id:
            continue
        
        # Check tolerant variants against test_ids
        matched = False
        for v in _normalize_id_variants(ann_id):
            if v in test_ids:
                matched = True
                break
        
        if not matched:
            continue
            
        matched_count += 1
        human_cmp_raw = item.get("human_comparisons", {})
        llm_cmp_raw = (item.get("llm_result", {}) or {}).get("comparisons", {})
        # Normalize keys and coerce
        human_cmp = {}
        llm_cmp = {}
        for k in DIMENSIONS:
            human_cmp[k] = _coerce_numeric(_normalize_key_lookup(human_cmp_raw, k))
            llm_cmp[k] = _coerce_numeric(_normalize_key_lookup(llm_cmp_raw, k))
        
        # Ensure all dims present
        if not any(human_cmp[k] is not None and llm_cmp[k] is not None for k in DIMENSIONS):
            continue
            
        row = {"model": model, "annotation_id": ann_id, "task": "comparison"}
        for k in DIMENSIONS:
            row[f"human_{k}"] = human_cmp[k]
            row[f"llm_{k}"] = llm_cmp[k]
        rows.append(row)
    
    if matched_count > 0 and not QUIET:
        print(f"  {json_path.name}: Matched {matched_count}/{total_count} comparison items")
    
    return pd.DataFrame(rows)


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations for both rating and comparison tasks."""
    if df.empty:
        return pd.DataFrame()
    
    records = []
    
    # Group by model and task
    for (model, task), g in df.groupby(["model", "task"]):
        for dim in DIMENSIONS:
            h = pd.to_numeric(g[f"human_{dim}"], errors="coerce")
            l = pd.to_numeric(g[f"llm_{dim}"], errors="coerce")
            mask = h.notna() & l.notna()
            
            if mask.sum() >= 2:
                try:
                    s = spearmanr(h[mask], l[mask]).correlation
                    p = pearsonr(h[mask], l[mask]).statistic
                    mae = np.mean(np.abs(h[mask] - l[mask]))
                except:
                    s = p = mae = np.nan
            else:
                s = p = mae = np.nan

            # Coerce NaN correlations to 0.0 as requested
            try:
                if pd.isna(s):
                    s = 0.0
            except Exception:
                pass
            try:
                if pd.isna(p):
                    p = 0.0
            except Exception:
                pass
            
            records.append({
                "model": model,
                "task": task,
                "dimension": dim,
                "spearman": s,
                "pearson": p,
                "mae": mae,
                "n": int(mask.sum())
            })
    
    return pd.DataFrame(records)


def plot_heatmap(df: pd.DataFrame, metric: str, task: str, outdir: Path, title_suffix: str = "") -> None:
    """Plot heatmap for a specific metric and task type."""
    if df.empty:
        return
        
    # Filter for the specific task
    task_df = df[df['task'] == task]
    if task_df.empty:
        return
    
    # Create pivot table
    pivot_data = task_df.pivot_table(
        index='model',
        columns='dimension',
        values=metric,
        aggfunc='mean'
    )
    
    # Reorder columns to match DIMENSIONS order
    pivot_data = pivot_data.reindex(columns=DIMENSIONS)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Choose colormap based on metric
    if metric == 'mae':
        cmap = 'RdYlBu'  # Lower is better for MAE
        center = None
    else:
        cmap = 'RdYlBu_r'  # Higher is better for correlations
        center = 0
    
    # Create heatmap
    sns.heatmap(
        pivot_data, 
        annot=True, 
        annot_kws={'fontsize': 14}, 
        cmap=cmap, 
        center=center,
        ax=ax, 
        cbar_kws={'label': ''}, 
        fmt='.3f',
        vmin=-1 if metric in ['spearman', 'pearson'] else None,
        vmax=1 if metric in ['spearman', 'pearson'] else None
    )
    
    # Adjust colorbar font size
    try:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
    except Exception:
        pass
    
    # Set title and labels
    metric_display = {
        'pearson': 'Pearson r',
        'spearman': 'Spearman r',
        'mae': 'MAE'
    }.get(metric, metric)
    
    title = f'{task.title()} Task - {metric_display}'
    if title_suffix:
        title += f' {title_suffix}'
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Model', fontsize=14, fontweight='bold')
    # Set custom x tick labels and rotation
    display_labels = ['Representiveness', 'Informativeness', 'Neutrality', 'Policy Approval']
    try:
        ax.set_xticklabels(display_labels)
    except Exception:
        pass
    ax.tick_params(axis='x', rotation=30, labelsize=12)
    ax.tick_params(axis='y', labelrotation=0, labelsize=14)
    
    # Bold x-axis tick labels (dimension names)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight('bold')
    
    plt.tight_layout()
    
    # Save figure
    filename = f"test_{task}_{metric}_heatmap.pdf"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {filename}")


def create_combined_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    """Create combined heatmap showing both rating and comparison results."""
    if df.empty:
        return
    
    rating_df = df[df['task'] == 'rating']
    comparison_df = df[df['task'] == 'comparison']
    
    # Set style
    sns.set(style="whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Rating task - Pearson and Spearman
    for idx, metric in enumerate(['pearson', 'spearman']):
        ax = axes[0, idx]
        
        if not rating_df.empty:
            pivot_data = rating_df.pivot_table(
                index='model',
                columns='dimension',
                values=metric,
                aggfunc='mean'
            ).reindex(columns=DIMENSIONS)
            
            sns.heatmap(
                pivot_data, 
                annot=True, 
                annot_kws={'fontsize': 14}, 
                cmap='RdYlBu_r', 
                center=0,
                ax=ax, 
                cbar_kws={'label': ''}, 
                fmt='.3f',
                vmin=-1, vmax=1
            )
            
            try:
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=14)
            except:
                pass
            
            metric_display = 'Pearson r' if metric == 'pearson' else 'Spearman r'
            ax.set_title(f'Rating Task - {metric_display} (Test Set Only)', fontsize=16, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Model' if idx == 0 else '', fontsize=14, fontweight='bold')
            # Custom x tick labels and rotation
            try:
                ax.set_xticklabels(['Representiveness', 'Informativeness', 'Neutrality', 'Policy Approval'])
            except Exception:
                pass
            ax.tick_params(axis='x', rotation=30, labelsize=12)
            ax.tick_params(axis='y', labelrotation=0, labelsize=14)
            
            for lbl in ax.get_xticklabels():
                lbl.set_fontweight('bold')
        else:
            ax.set_title(f'Rating Task - No Data', fontsize=16)
            ax.axis('off')
    
    # Comparison task - Pearson and Spearman
    for idx, metric in enumerate(['pearson', 'spearman']):
        ax = axes[1, idx]
        
        if not comparison_df.empty:
            pivot_data = comparison_df.pivot_table(
                index='model',
                columns='dimension',
                values=metric,
                aggfunc='mean'
            ).reindex(columns=DIMENSIONS)
            
            sns.heatmap(
                pivot_data, 
                annot=True, 
                annot_kws={'fontsize': 14}, 
                cmap='RdYlBu_r', 
                center=0,
                ax=ax, 
                cbar_kws={'label': ''}, 
                fmt='.3f',
                vmin=-1, vmax=1
            )
            
            try:
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=14)
            except:
                pass
            
            metric_display = 'Pearson r' if metric == 'pearson' else 'Spearman r'
            ax.set_title(f'Comparison Task - {metric_display} (Test Set Only)', fontsize=16, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Model' if idx == 0 else '', fontsize=14, fontweight='bold')
            try:
                ax.set_xticklabels(['Representiveness', 'Informativeness', 'Neutrality', 'Policy Approval'])
            except Exception:
                pass
            ax.tick_params(axis='x', rotation=30, labelsize=12)
            ax.tick_params(axis='y', labelrotation=0, labelsize=14)
            
            for lbl in ax.get_xticklabels():
                lbl.set_fontweight('bold')
        else:
            ax.set_title(f'Comparison Task - No Data', fontsize=16)
            ax.axis('off')
    
    plt.suptitle('Model-Human Correlation on Test Set', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "test_combined_correlation_heatmaps.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: test_combined_correlation_heatmaps.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter JSON results by test set and recompute correlations with heatmap visualization.")
    parser.add_argument("--results-dir", type=str, default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/eval_llm_human_correlation")
    parser.add_argument("--test-jsonl", type=str, default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings/test.jsonl")
    parser.add_argument("--out-dir", type=str, default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/eval_llm_human_correlation/plots_test")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed logs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    test_jsonl = Path(args.test_jsonl)
    outdir = Path(args.out_dir)
    global QUIET
    QUIET = True if args.quiet else True

    if not QUIET:
        print(f"Reading test IDs from: {test_jsonl}")
    test_ids = read_test_ids(test_jsonl)
    if not test_ids:
        if not QUIET:
            print(f"No test IDs found in {test_jsonl}")
        return
    if not QUIET:
        print(f"Found {len(test_ids)} test IDs (including base IDs for matching)")

    json_files = find_json_results(results_dir)
    if not json_files:
        if not QUIET:
            print(f"No JSON result files found under {results_dir}")
        return
    if not QUIET:
        print(f"Found {len(json_files)} JSON result files")

    # Extract both rating and comparison pairs
    if not QUIET:
        print("\nExtracting matching data...")
    all_frames: List[pd.DataFrame] = []
    
    for jf in json_files:
        # Extract rating pairs
        rating_df = extract_rating_pairs(jf, test_ids)
        if not rating_df.empty:
            all_frames.append(rating_df)
        
        # Extract comparison pairs
        comparison_df = extract_comparison_pairs(jf, test_ids)
        if not comparison_df.empty:
            all_frames.append(comparison_df)
    
    if not all_frames:
        print("\n❌ No matching records between results and test set.")
        return

    # Combine all data
    all_pairs = pd.concat(all_frames, ignore_index=True)
    if not QUIET:
        print(f"\n✅ Total matched pairs across all models: {len(all_pairs)}")
    
    # Show summary by task
    task_counts = all_pairs.groupby(['task', 'model']).size().reset_index(name='count')
    if not QUIET:
        print("\nData summary:")
    for task in ['rating', 'comparison']:
        task_data = task_counts[task_counts['task'] == task]
        if not task_data.empty:
            if not QUIET:
                print(f"\n{task.title()} task:")
            for _, row in task_data.iterrows():
                if not QUIET:
                    print(f"  {row['model']}: {row['count']} items")
    
    # Compute correlations
    if not QUIET:
        print("\nComputing correlations...")
    corr_results = compute_correlations(all_pairs)
    
    # Save results to CSV
    outdir.mkdir(parents=True, exist_ok=True)
    corr_results.to_csv(outdir / "test_correlations_detailed.csv", index=False)
    
    # Create summary statistics
    summary = corr_results.groupby(['model', 'task']).agg({
        'spearman': 'mean',
        'pearson': 'mean',
        'mae': 'mean',
        'n': 'sum'
    }).round(4)
    summary.to_csv(outdir / "test_correlations_summary.csv")
    
    # Print correlation results
    if not QUIET:
        print("\n📊 Correlation Results (Test Set Only):")
        print("="*60)
    
    for task in ['rating', 'comparison']:
        task_data = corr_results[corr_results['task'] == task]
        if task_data.empty:
            continue
            
        if not QUIET:
            print(f"\n{task.upper()} TASK:")
        
        # Group by model for overall stats
        for model in task_data['model'].unique():
            model_data = task_data[task_data['model'] == model]
            avg_spearman = model_data['spearman'].mean()
            avg_pearson = model_data['pearson'].mean()
            avg_mae = model_data['mae'].mean()
            total_n = model_data['n'].sum()
            
            if not QUIET:
                print(f"\n  {model}:")
                print(f"    Avg Spearman: {avg_spearman:.4f}")
                print(f"    Avg Pearson:  {avg_pearson:.4f}")
                print(f"    Avg MAE:      {avg_mae:.4f}")
                print(f"    N samples:    {total_n}")
    
    # Create visualizations
    if not QUIET:
        print("\n📊 Creating visualizations...")
    
    # Individual heatmaps for each metric and task
    for task in ['rating', 'comparison']:
        for metric in ['pearson', 'spearman']:
            plot_heatmap(corr_results, metric, task, outdir, title_suffix="(Test Set Only)")
    
    # Combined heatmap
    create_combined_heatmap(corr_results, outdir)
    
    if not QUIET:
        print(f"\n📁 All results saved to {outdir}")
        print("Files created:")
        print("  - test_correlations_detailed.csv")
        print("  - test_correlations_summary.csv")
        print("  - test_combined_correlation_heatmaps.pdf")
        print("  - Individual heatmaps for each task and metric")


if __name__ == "__main__":
    main()