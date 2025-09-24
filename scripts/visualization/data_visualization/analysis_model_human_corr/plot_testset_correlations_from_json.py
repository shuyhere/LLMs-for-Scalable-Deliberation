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

DIMENSIONS = [
    "perspective_representation",
    "informativeness",
    "neutrality_balance",
    "policy_approval",
]


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
            if "id" in obj:
                full_id = str(obj["id"])
                ids.add(full_id)
                # Extract base ID without _A/_B suffix for matching
                if "_rating_" in full_id:
                    base_id = full_id.rsplit("_", 1)[0]  # Remove _A or _B
                    base_ids.add(base_id)
                else:
                    base_ids.add(full_id)
            elif "annotation_id" in obj:
                ids.add(str(obj["annotation_id"]))
    
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
        
        # Check if this ID matches any test ID (including base IDs)
        matched = False
        if ann_id in test_ids:
            matched = True
        elif "_rating" in ann_id:
            # Try with _A and _B suffixes if base ID
            if f"{ann_id}_A" in test_ids or f"{ann_id}_B" in test_ids:
                matched = True
        
        if not matched:
            continue
            
        matched_count += 1
        human = item.get("human_ratings", {})
        llm = (item.get("llm_result", {}) or {}).get("ratings", {})
        
        # Ensure all dims present
        if not all(k in human for k in DIMENSIONS):
            continue
        if not all(k in llm for k in DIMENSIONS):
            continue
            
        row = {"model": model, "annotation_id": ann_id, "task": "rating"}
        for k in DIMENSIONS:
            row[f"human_{k}"] = human[k]
            row[f"llm_{k}"] = llm[k]
        rows.append(row)
    
    if matched_count > 0:
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
        
        # Check if this ID matches any test ID
        matched = False
        # Try direct match
        if ann_id in test_ids:
            matched = True
        # Try extracting base ID from comparison ID
        elif "_comparison" in ann_id:
            base_id = ann_id.replace("_comparison", "_rating")
            if base_id in test_ids or f"{base_id}_A" in test_ids or f"{base_id}_B" in test_ids:
                matched = True
        # Try extracting triplet ID
        elif "triplet_" in ann_id:
            triplet_num = ann_id.split("triplet_")[1].split("_")[0]
            possible_ids = [
                f"triplet_{triplet_num}_rating",
                f"triplet_{triplet_num}_rating_A",
                f"triplet_{triplet_num}_rating_B"
            ]
            if any(pid in test_ids for pid in possible_ids):
                matched = True
        
        if not matched:
            continue
            
        matched_count += 1
        human_cmp = item.get("human_comparisons", {})
        llm_cmp = (item.get("llm_result", {}) or {}).get("comparisons", {})
        
        # Ensure all dims present
        if not all(k in human_cmp for k in DIMENSIONS):
            continue
        if not all(k in llm_cmp for k in DIMENSIONS):
            continue
            
        row = {"model": model, "annotation_id": ann_id, "task": "comparison"}
        for k in DIMENSIONS:
            row[f"human_{k}"] = human_cmp[k]
            row[f"llm_{k}"] = llm_cmp[k]
        rows.append(row)
    
    if matched_count > 0:
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
    ax.tick_params(axis='x', rotation=0, labelsize=12)
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
            ax.tick_params(axis='x', rotation=0, labelsize=12)
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
            ax.tick_params(axis='x', rotation=0, labelsize=12)
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
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    test_jsonl = Path(args.test_jsonl)
    outdir = Path(args.out_dir)

    print(f"Reading test IDs from: {test_jsonl}")
    test_ids = read_test_ids(test_jsonl)
    if not test_ids:
        print(f"No test IDs found in {test_jsonl}")
        return
    print(f"Found {len(test_ids)} test IDs (including base IDs for matching)")

    json_files = find_json_results(results_dir)
    if not json_files:
        print(f"No JSON result files found under {results_dir}")
        return
    print(f"Found {len(json_files)} JSON result files")

    # Extract both rating and comparison pairs
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
    print(f"\n✅ Total matched pairs across all models: {len(all_pairs)}")
    
    # Show summary by task
    task_counts = all_pairs.groupby(['task', 'model']).size().reset_index(name='count')
    print("\nData summary:")
    for task in ['rating', 'comparison']:
        task_data = task_counts[task_counts['task'] == task]
        if not task_data.empty:
            print(f"\n{task.title()} task:")
            for _, row in task_data.iterrows():
                print(f"  {row['model']}: {row['count']} items")
    
    # Compute correlations
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
    print("\n📊 Correlation Results (Test Set Only):")
    print("="*60)
    
    for task in ['rating', 'comparison']:
        task_data = corr_results[corr_results['task'] == task]
        if task_data.empty:
            continue
            
        print(f"\n{task.upper()} TASK:")
        
        # Group by model for overall stats
        for model in task_data['model'].unique():
            model_data = task_data[task_data['model'] == model]
            avg_spearman = model_data['spearman'].mean()
            avg_pearson = model_data['pearson'].mean()
            avg_mae = model_data['mae'].mean()
            total_n = model_data['n'].sum()
            
            print(f"\n  {model}:")
            print(f"    Avg Spearman: {avg_spearman:.4f}")
            print(f"    Avg Pearson:  {avg_pearson:.4f}")
            print(f"    Avg MAE:      {avg_mae:.4f}")
            print(f"    N samples:    {total_n}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Individual heatmaps for each metric and task
    for task in ['rating', 'comparison']:
        for metric in ['pearson', 'spearman']:
            plot_heatmap(corr_results, metric, task, outdir, title_suffix="(Test Set Only)")
    
    # Combined heatmap
    create_combined_heatmap(corr_results, outdir)
    
    print(f"\n📁 All results saved to {outdir}")
    print("Files created:")
    print("  - test_correlations_detailed.csv")
    print("  - test_correlations_summary.csv")
    print("  - test_combined_correlation_heatmaps.pdf")
    print("  - Individual heatmaps for each task and metric")


if __name__ == "__main__":
    main()