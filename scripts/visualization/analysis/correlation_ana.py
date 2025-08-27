#!/usr/bin/env python3
"""
Correlation analysis for summarization evaluation scores across different models.

This script analyzes the correlation between different evaluation models' scoring patterns
for each topic, helping to understand how consistent different models are in their evaluations.

Outputs per topic:
- Correlation matrix heatmap between evaluator models
- Correlation matrix heatmap between summarizer models  
- Statistical correlation analysis (Pearson, Spearman)
- Model agreement analysis
- CSV with correlation coefficients

Usage:
  python scripts/visualization/analysis/correlation_ana.py \
    --results-dir /ibex/project/c2328/LLMs-Scalable-Deliberation/results/summary \
    --out-dir /ibex/project/c2328/LLMs-Scalable-Deliberation/results/summary/visualization
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr


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
    comment_index: int
    score: float


def load_eval_file(eval_path: Path) -> Optional[Dict]:
    """Load evaluation JSON file."""
    try:
        with eval_path.open("r", encoding="utf-8") as Path:
            return json.load(Path)
    except FileNotFoundError:
        return None


def extract_scores_with_comments(topic: str, summary_model: str, data: Dict) -> List[ScoreRecord]:
    """Extract scores with comment indices for correlation analysis."""
    records: List[ScoreRecord] = []
    evaluations = data.get("evaluations", {})
    
    for evaluator_model, evaluator_payload in evaluations.items():
        evaluation_data = evaluator_payload.get("evaluation_data", {})
        results = evaluation_data.get("evaluation_results", [])
        
        for item in results:
            score_val = item.get("score", item.get("extracted_score"))
            comment_idx = item.get("comment_index")
            
            if score_val is None or comment_idx is None:
                continue
                
            try:
                score_float = float(score_val)
            except Exception:
                # Try to parse number from string
                score_float = _best_effort_extract_number(str(score_val))
                if score_float is None:
                    continue
                    
            records.append(
                ScoreRecord(
                    topic=topic,
                    summary_model=summary_model,
                    evaluator_model=evaluator_model,
                    comment_index=comment_idx,
                    score=score_float,
                )
            )
    return records


def _best_effort_extract_number(text: str) -> Optional[float]:
    """Extract number from text using regex."""
    import re
    matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def create_score_matrix(records: List[ScoreRecord], group_by: str = "evaluator") -> Tuple[pd.DataFrame, List[str]]:
    """Create a score matrix where rows are comments and columns are models."""
    df = pd.DataFrame([r.__dict__ for r in records])
    
    if group_by == "evaluator":
        # For evaluator correlation: we need to handle multiple evaluators scoring same comments
        # Group by comment_index and evaluator_model, then pivot
        # First, ensure we have unique (comment_index, evaluator_model) pairs
        df_clean = df.drop_duplicates(subset=["comment_index", "evaluator_model"])
        pivot_df = df_clean.pivot(index="comment_index", columns="evaluator_model", values="score")
        model_names = sorted(df["evaluator_model"].unique())
    else:
        # For summarizer correlation: each comment is evaluated by all evaluators
        # We can aggregate scores across evaluators (e.g., mean) for each comment
        # Group by comment_index and summary_model, calculate mean score across evaluators
        df_agg = df.groupby(["comment_index", "summary_model"])["score"].mean().reset_index()
        pivot_df = df_agg.pivot(index="comment_index", columns="summary_model", values="score")
        model_names = sorted(df["summary_model"].unique())
    
    return pivot_df, model_names


def calculate_correlations(score_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Pearson and Spearman correlations."""
    # Remove rows with any NaN values
    clean_matrix = score_matrix.dropna()
    
    if len(clean_matrix) < 2:
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate correlations
    pearson_corr = clean_matrix.corr(method='pearson')
    spearman_corr = clean_matrix.corr(method='spearman')
    
    return pearson_corr, spearman_corr


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str, out_path: Path, 
                           group_type: str = "evaluator") -> None:
    """Plot correlation matrix heatmap."""
    if corr_matrix.empty:
        return
        
    # Apply display name mapping for readability
    display_corr = corr_matrix.copy()
    display_corr.index = display_corr.index.map(display_model_name)
    display_corr.columns = display_corr.columns.map(display_model_name)
        
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap with diverging color scheme
    mask = np.triu(np.ones_like(display_corr, dtype=bool))  # Mask upper triangle
    sns.heatmap(display_corr, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r", 
                center=0, square=True, linewidths=0.5, cbar_kws={"label": "Correlation"},
                vmin=-1, vmax=1, ax=ax)
    
    ax.set_title(f"{title}\n{group_type.title()} Models", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"{group_type.title()} Model", fontsize=12)
    ax.set_ylabel(f"{group_type.title()} Model", fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def analyze_model_agreement(corr_matrix: pd.DataFrame, group_type: str) -> Dict:
    """Analyze model agreement and consistency."""
    if corr_matrix.empty:
        return {}
    
    # Calculate statistics
    n_models = len(corr_matrix)
    correlations = []
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            corr_val = corr_matrix.iloc[i, j]
            if not pd.isna(corr_val):
                correlations.append(corr_val)
    
    if not correlations:
        return {}
    
    correlations = np.array(correlations)
    
    analysis = {
        "n_models": n_models,
        "n_pairs": len(correlations),
        "mean_correlation": float(np.mean(correlations)),
        "std_correlation": float(np.std(correlations)),
        "min_correlation": float(np.min(correlations)),
        "max_correlation": float(np.max(correlations)),
        "high_agreement_pairs": int(np.sum(correlations > 0.7)),
        "moderate_agreement_pairs": int(np.sum((correlations > 0.4) & (correlations <= 0.7))),
        "low_agreement_pairs": int(np.sum(correlations <= 0.4)),
        "group_type": group_type
    }
    
    return analysis


def run_statistical_tests(score_matrix: pd.DataFrame) -> Dict:
    """Run statistical tests for model agreement."""
    if score_matrix.empty or len(score_matrix.columns) < 2:
        return {}
    
    # Remove rows with any NaN values
    clean_matrix = score_matrix.dropna()
    
    if len(clean_matrix) < 10:  # Need sufficient data for tests
        return {}
    
    results = {}
    models = list(clean_matrix.columns)
    
    # Pairwise statistical tests
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            scores1 = clean_matrix[model1].values
            scores2 = clean_matrix[model2].values
            
            # Pearson correlation test
            pearson_r, pearson_p = pearsonr(scores1, scores2)
            
            # Spearman correlation test
            spearman_r, spearman_p = spearmanr(scores1, scores2)
            
            # Paired t-test
            t_stat, t_p = stats.ttest_rel(scores1, scores2)
            
            # Wilcoxon signed-rank test
            try:
                w_stat, w_p = stats.wilcoxon(scores1, scores2)
            except:
                w_stat, w_p = np.nan, np.nan
            
            pair_name = f"{model1}_vs_{model2}"
            results[pair_name] = {
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "t_statistic": t_stat,
                "t_p_value": t_p,
                "wilcoxon_statistic": w_stat,
                "wilcoxon_p_value": w_p,
                "n_samples": len(scores1)
            }
    
    return results


def find_summary_models(results_dir: Path) -> List[Path]:
    """Find summarization model directories."""
    summary_model_dirs = []
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.lower() in {"logs", "visualization"}:
            continue
        summary_model_dirs.append(child)
    return sorted(summary_model_dirs, key=lambda p: p.name)


def find_topics(summary_model_dir: Path) -> List[str]:
    """Find topic directories."""
    topics = []
    for child in summary_model_dir.iterdir():
        if child.is_dir():
            topics.append(child.name)
    return sorted(topics)


def ensure_out_dir(path: Path) -> None:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Correlation analysis for evaluation scores across models")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "results" / "summary"),
        help="Path to results/summary directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write correlation analysis outputs (default: {results-dir}/visualization/analysis/correlation_analysis)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    
    # Set default out-dir
    if args.out_dir is None:
        out_root = results_dir / "visualization" / "analysis" / "correlation_analysis"
    else:
        out_root = Path(args.out_dir).resolve()
    
    ensure_out_dir(out_root)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Find models and topics
    model_dirs = find_summary_models(results_dir)
    if not model_dirs:
        print(f"No summarization model directories found in: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Get common topics
    topics_per_model = {m.name: set(find_topics(m)) for m in model_dirs}
    common_topics = set.intersection(*topics_per_model.values()) if len(topics_per_model) > 1 else next(iter(topics_per_model.values()))

    print(f"Analyzing correlations for {len(common_topics)} topics...")
    
    # Store overall results
    all_correlations = []
    all_agreements = []
    all_statistical_tests = []

    # Process each topic
    for topic in sorted(common_topics):
        print(f"\nProcessing topic: {topic}")
        topic_out = out_root / topic
        ensure_out_dir(topic_out)
        
        # Collect all scores for this topic
        topic_records = []
        
        for model_dir in model_dirs:
            summary_model = model_dir.name
            topic_dir = model_dir / topic
            eval_filename = f"eva_summary_{topic}.json"
            eval_path = topic_dir / eval_filename
            
            data = load_eval_file(eval_path)
            if data is None:
                continue
                
            records = extract_scores_with_comments(topic=topic, summary_model=summary_model, data=data)
            topic_records.extend(records)
        
        if not topic_records:
            print(f"  No evaluation data found for topic: {topic}")
            continue
        
        print(f"  Found {len(topic_records)} score records")
        
        # Create score matrices
        evaluator_matrix, evaluator_models = create_score_matrix(topic_records, group_by="evaluator")
        summarizer_matrix, summarizer_models = create_score_matrix(topic_records, group_by="summarizer")
        
        # Calculate correlations
        evaluator_pearson, evaluator_spearman = calculate_correlations(evaluator_matrix)
        summarizer_pearson, summarizer_spearman = calculate_correlations(summarizer_matrix)
        
        # Plot correlation heatmaps
        if not evaluator_pearson.empty:
            plot_correlation_heatmap(
                evaluator_pearson, 
                f"Pearson Correlation - {topic}", 
                topic_out / f"{topic}__evaluator_pearson_correlation.png",
                "evaluator"
            )
            
            plot_correlation_heatmap(
                evaluator_spearman,
                f"Spearman Correlation - {topic}",
                topic_out / f"{topic}__evaluator_spearman_correlation.png", 
                "evaluator"
            )
        
        if not summarizer_pearson.empty:
            plot_correlation_heatmap(
                summarizer_pearson,
                f"Pearson Correlation - {topic}",
                topic_out / f"{topic}__summarizer_pearson_correlation.png",
                "summarizer"
            )
            
            plot_correlation_heatmap(
                summarizer_spearman,
                f"Spearman Correlation - {topic}",
                topic_out / f"{topic}__summarizer_spearman_correlation.png",
                "summarizer"
            )
        
        # Analyze model agreement
        evaluator_agreement = analyze_model_agreement(evaluator_pearson, "evaluator")
        summarizer_agreement = analyze_model_agreement(summarizer_pearson, "summarizer")
        
        if evaluator_agreement:
            evaluator_agreement["topic"] = topic
            all_agreements.append(evaluator_agreement)
            
        if summarizer_agreement:
            summarizer_agreement["topic"] = topic
            all_agreements.append(summarizer_agreement)
        
        # Run statistical tests
        evaluator_tests = run_statistical_tests(evaluator_matrix)
        summarizer_tests = run_statistical_tests(summarizer_matrix)
        
        # Save correlation matrices to CSV
        if not evaluator_pearson.empty:
            evaluator_pearson.to_csv(topic_out / f"{topic}__evaluator_pearson_correlation.csv")
            evaluator_spearman.to_csv(topic_out / f"{topic}__evaluator_spearman_correlation.csv")
            
        if not summarizer_pearson.empty:
            summarizer_pearson.to_csv(topic_out / f"{topic}__summarizer_pearson_correlation.csv")
            summarizer_spearman.to_csv(topic_out / f"{topic}__summarizer_spearman_correlation.csv")
        
        # Store statistical test results
        for test_name, test_results in evaluator_tests.items():
            test_results["topic"] = topic
            test_results["group_type"] = "evaluator"
            all_statistical_tests.append(test_results)
            
        for test_name, test_results in summarizer_tests.items():
            test_results["topic"] = topic
            test_results["group_type"] = "summarizer"
            all_statistical_tests.append(test_results)
        
        print(f"  Completed correlation analysis for {topic}")
    
    # Save overall results
    if all_agreements:
        agreements_df = pd.DataFrame(all_agreements)
        agreements_df.to_csv(out_root / "all_topics_model_agreement.csv", index=False)
        print(f"\nSaved model agreement analysis to: {out_root / 'all_topics_model_agreement.csv'}")
    
    if all_statistical_tests:
        tests_df = pd.DataFrame(all_statistical_tests)
        tests_df.to_csv(out_root / "all_topics_statistical_tests.csv", index=False)
        print(f"Saved statistical test results to: {out_root / 'all_topics_statistical_tests.csv'}")
    
    print(f"\nCorrelation analysis completed!")
    print(f"Outputs saved to: {out_root}")


if __name__ == "__main__":
    main()
