#!/usr/bin/env python3
"""
Analyze summarization evaluation results and generate per-topic visualizations.

Outputs per topic:
- CSV of mean scores by summarization model and evaluator model
- Heatmap of mean scores (evaluator x summarizer)
- Distribution plots (histograms + KDE) of per-comment scores for each (summarizer, evaluator)

Usage:
  python scripts/visualization/analyze_summary_evaluations.py \
    --results-dir /ibex/project/c2328/LLMs-Scalable-Deliberation/results/summary \
    --out-dir /ibex/project/c2328/LLMs-Scalable-Deliberation/scripts/visualization/outputs
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Mapping for nicer display names in plots
DISPLAY_NAME_MAP = {
    "web-rev-claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
}


def display_model_name(name: str) -> str:
    return DISPLAY_NAME_MAP.get(name, name)


@dataclass
class ScoreRecord:
    topic: str
    summary_model: str
    evaluator_model: str
    score: float


def find_summary_models(results_dir: Path) -> List[Path]:
    summary_model_dirs: List[Path] = []
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        # Skip non-model directories like logs
        if child.name.lower() in {"logs"}:
            continue
        # Expect topic subdirectories or batch report inside
        summary_model_dirs.append(child)
    return sorted(summary_model_dirs, key=lambda p: p.name)


def find_topics(summary_model_dir: Path) -> List[str]:
    topics: List[str] = []
    for child in summary_model_dir.iterdir():
        if child.is_dir():
            topics.append(child.name)
    return sorted(topics)


def load_eval_file(eval_path: Path) -> Optional[Dict]:
    try:
        with eval_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def extract_scores(topic: str, summary_model: str, data: Dict) -> List[ScoreRecord]:
    records: List[ScoreRecord] = []
    evaluations = data.get("evaluations", {})
    for evaluator_model, evaluator_payload in evaluations.items():
        evaluation_data = evaluator_payload.get("evaluation_data", {})
        results = evaluation_data.get("evaluation_results", [])
        for item in results:
            # Prefer "score"; fall back to "extracted_score"
            score_val = item.get("score", item.get("extracted_score"))
            if score_val is None:
                continue
            try:
                score_float = float(score_val)
            except Exception:
                # Last resort: try to parse number from string (e.g., "\\boxed{4}")
                score_float = _best_effort_extract_number(str(score_val))
                if score_float is None:
                    continue
            records.append(
                ScoreRecord(
                    topic=topic,
                    summary_model=summary_model,
                    evaluator_model=evaluator_model,
                    score=score_float,
                )
            )
    return records


def _best_effort_extract_number(text: str) -> Optional[float]:
    import re

    matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def aggregate_means(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["topic", "summary_model", "evaluator_model"], as_index=False)["score"].mean()
    )
    grouped.rename(columns={"score": "mean_score"}, inplace=True)
    return grouped


def plot_heatmap(means_df: pd.DataFrame, topic: str, out_dir: Path) -> None:
    # Apply display name mapping for readability
    display_df = means_df.copy()
    display_df["evaluator_display"] = display_df["evaluator_model"].map(display_model_name)
    display_df["summary_display"] = display_df["summary_model"].map(display_model_name)

    pivot = display_df.pivot(index="evaluator_display", columns="summary_display", values="mean_score")
    # Scale figure size more generously to avoid compression
    num_cols = max(1, len(pivot.columns))
    num_rows = max(1, len(pivot.index))
    fig_width = 2.0 + 1.6 * num_cols
    fig_height = 1.5 + 1.2 * num_rows
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "Mean score"}, ax=ax)
    ax.set_title(f"Mean evaluation scores by summarizer and evaluator\nTopic: {topic}")
    ax.set_xlabel("Summarization model")
    ax.set_ylabel("Evaluation model")
    # Use tight_layout with margins reserved for title
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    out_path = out_dir / f"{topic}__mean_scores_heatmap.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(raw_df: pd.DataFrame, topic: str, out_dir: Path) -> None:
    # Create mapped display columns to shorten long model names in titles
    df = raw_df.copy()
    df["evaluator_display"] = df["evaluator_model"].map(display_model_name)
    df["summary_display"] = df["summary_model"].map(display_model_name)
    
    # Get unique evaluators and summarizers
    evaluators = sorted(df["evaluator_display"].unique())
    summarizers = sorted(df["summary_display"].unique())
    
    # Create subplot grid
    fig, axes = plt.subplots(
        len(evaluators), len(summarizers), 
        figsize=(2.2 * len(summarizers), 1.8 * len(evaluators)),
        sharex=True, sharey=True
    )
    
    # Handle single row/column case
    if len(evaluators) == 1:
        axes = axes.reshape(1, -1)
    if len(summarizers) == 1:
        axes = axes.reshape(-1, 1)
    
    # Score values and bin edges
    scores = [1, 2, 3, 4, 5]
    bin_edges = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    
    for i, evaluator in enumerate(evaluators):
        for j, summarizer in enumerate(summarizers):
            ax = axes[i, j]
            
            # Filter data for this combination
            mask = (df["evaluator_display"] == evaluator) & (df["summary_display"] == summarizer)
            subset = df[mask]
            
            if len(subset) > 0:
                # Count scores and convert to density
                counts, _ = np.histogram(subset["score"], bins=bin_edges)
                densities = counts / len(subset)  # Normalize to sum to 1
                
                # Create bars with equal width and gaps
                bar_width = 0.8
                x_pos = np.array(scores)
                
                bars = ax.bar(x_pos, densities, width=bar_width, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Color bars based on score (optional)
                colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4ecdc4', '#45b7d1']
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                # Set x-axis
                ax.set_xlim(0.5, 5.5)
                ax.set_xticks(scores)
                ax.set_xticklabels([str(s) for s in scores], fontsize=8)
                
                # Set y-axis to uniform range 0.0-0.7
                ax.set_ylim(0, 0.7)
                ax.set_ylabel("Probability", fontsize=8)
                
                # Add value labels on bars
                for x, density in zip(x_pos, densities):
                    if density > 0:
                        ax.text(x, density + 0.01, f'{density:.2f}', 
                               ha='center', va='bottom', fontsize=7)
                
                # Grid
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_axisbelow(True)
            else:
                # No data case
                ax.text(3, 0.35, 'No data', ha='center', va='center', 
                       transform=ax.transData, fontsize=9, alpha=0.5)
                ax.set_xlim(0.5, 5.5)
                ax.set_ylim(0, 0.7)
            
            # Set title for first row with smaller font
            if i == 0:
                ax.set_title(f"{summarizer}", fontsize=9, fontweight='bold', pad=8)
            
            # Set y-label for first column with smaller font
            if j == 0:
                ax.set_ylabel(f"{evaluator}\nProbability", fontsize=9, fontweight='bold')
    
    # Set common x-label with smaller font
    fig.text(0.5, 0.02, 'Score', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add axis group labels
    fig.text(0.5, 0.92, 'Summarizers (columns)', ha='center', va='center', fontsize=9, style='italic', alpha=0.7)
    fig.text(0.02, 0.5, 'Evaluators (rows)', ha='center', va='center', fontsize=9, style='italic', alpha=0.7, rotation=90)
    
    # Adjust layout with more spacing
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.85, left=0.12, right=0.95, hspace=0.4, wspace=0.3)
    
    # Add main title with smaller font
    fig.suptitle(f"Score distributions by evaluator and summarizer\nTopic: {topic}", 
                 fontsize=11, fontweight='bold', y=0.98)
    
    out_path = out_dir / f"{topic}__score_distributions.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize summarization evaluation results per topic")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "results" / "summary"),
        help="Path to results/summary directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Directory to write visualizations and CSVs",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    ensure_out_dir(out_root)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    model_dirs = find_summary_models(results_dir)
    if not model_dirs:
        print(f"No summarization model directories found in: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine topics present across models (intersection to ensure completeness)
    topics_per_model = {m.name: set(find_topics(m)) for m in model_dirs}
    if not topics_per_model:
        print("No topics found.", file=sys.stderr)
        sys.exit(1)

    common_topics = set.intersection(*topics_per_model.values()) if len(topics_per_model) > 1 else next(iter(topics_per_model.values()))
    if not common_topics:
        # Fallback: use union if intersection is empty
        common_topics = set().union(*topics_per_model.values())

    all_records: List[ScoreRecord] = []

    for model_dir in model_dirs:
        summary_model = model_dir.name
        for topic in sorted(common_topics):
            topic_dir = model_dir / topic
            # Expect file like eva_summary_{topic}.json
            eval_filename = f"eva_summary_{topic}.json"
            eval_path = topic_dir / eval_filename
            data = load_eval_file(eval_path)
            if data is None:
                # Try alternative naming: sometimes topics may replace spaces with hyphens already handled above
                # If missing, skip gracefully
                continue
            records = extract_scores(topic=topic, summary_model=summary_model, data=data)
            all_records.extend(records)

    if not all_records:
        print("No evaluation scores found.", file=sys.stderr)
        sys.exit(1)

    # Build DataFrame of raw scores
    raw_df = pd.DataFrame([r.__dict__ for r in all_records])

    # Generate per-topic outputs
    for topic in sorted(raw_df["topic"].unique()):
        topic_df = raw_df[raw_df["topic"] == topic]
        topic_out = out_root / topic
        ensure_out_dir(topic_out)

        means_df = aggregate_means(topic_df)
        means_csv = topic_out / f"{topic}__mean_scores.csv"
        means_df.to_csv(means_csv, index=False)

        if not means_df.empty:
            plot_heatmap(means_df, topic=topic, out_dir=topic_out)

        if not topic_df.empty:
            plot_distributions(topic_df, topic=topic, out_dir=topic_out)

    # Also write a combined CSV across all topics
    combined_out = out_root / "all_topics_raw_scores.csv"
    raw_df.to_csv(combined_out, index=False)

    combined_means = aggregate_means(raw_df)
    combined_means.to_csv(out_root / "all_topics_mean_scores.csv", index=False)

    print(f"Wrote outputs to: {out_root}")


if __name__ == "__main__":
    main()


