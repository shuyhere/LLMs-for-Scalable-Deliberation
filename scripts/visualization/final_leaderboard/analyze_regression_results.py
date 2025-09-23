#!/usr/bin/env python3
"""
Comprehensive analysis script for regression evaluation results.
Generates visualizations and CSV files for model performance analysis.

Analyzes by:
1. Number of comments
2. Topic
3. Model type
4. Performance dimensions (perspective_representation, informativeness, neutrality_balance, policy_approval)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

TARGET_KEYS = [
    "perspective_representation",
    "informativeness",
    "neutrality_balance", 
    "policy_approval"
]

# Display names for better readability
DISPLAY_NAMES = {
    "perspective_representation": "Perspective",
    "informativeness": "Informativeness",
    "neutrality_balance": "Neutrality",
    "policy_approval": "Policy Approval"
}

MODEL_DISPLAY_NAMES = {
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5 Mini",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "deepseek-chat": "DeepSeek Chat",
    "deepseek-reasoner": "DeepSeek Reasoner",
    "grok-4-latest": "Grok 4",
    "qwen3-0.6b": "Qwen3 0.6B",
    "qwen3-1.7b": "Qwen3 1.7B",
    "qwen3-4b": "Qwen3 4B",
    "qwen3-8b": "Qwen3 8B",
    "qwen3-14b": "Qwen3 14B",
    "qwen3-30b-a3b": "Qwen3 30B",
    "qwen3-32b": "Qwen3 32B",
    "qwen3-235b-a22b": "Qwen3 235B",
    "web-rev-claude-opus-4-20250514": "Claude Opus",
    "TA": "Teaching Assistant"
}


def load_evaluation_results(base_dir: Path) -> Dict[str, Any]:
    """Load all evaluation results from the directory structure."""
    results = defaultdict(lambda: defaultdict(dict))
    
    # Iterate through model directories
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Iterate through comment number directories
        for comment_dir in model_dir.iterdir():
            if not comment_dir.is_dir():
                continue
            
            try:
                comment_num = int(comment_dir.name)
            except ValueError:
                continue
            
            # Load evaluation results
            results_file = comment_dir / "evaluation_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    results[model_name][comment_num] = data
    
    return dict(results)


def extract_metrics(results: Dict[str, Any]) -> pd.DataFrame:
    """Extract metrics into a structured DataFrame."""
    rows = []
    
    for model_name, comment_data in results.items():
        for comment_num, eval_data in comment_data.items():
            if 'results' not in eval_data:
                continue
            
            for topic_result in eval_data['results']:
                topic = topic_result['topic']
                
                # Check if we have predictions/ground_truth format
                has_predictions = False
                predictions = []
                ground_truths = []
                
                for summary in topic_result.get('summaries', []):
                    if 'predictions' in summary and 'ground_truth' in summary:
                        predictions.append(summary['predictions'])
                        ground_truths.append(summary['ground_truth'])
                        has_predictions = True
                    elif 'evaluation_stats' in summary:
                        # Handle evaluation_stats format - extract mean values
                        # Note: We don't have ground truth here, so we'll skip error calculation
                        eval_stats = summary['evaluation_stats']
                        pred_values = {k: eval_stats.get(k, {}).get('mean', 0.0) for k in TARGET_KEYS}
                        predictions.append(pred_values)
                
                if predictions:
                    if has_predictions and ground_truths:
                        # We have both predictions and ground truth
                        avg_predictions = {k: np.mean([p[k] for p in predictions]) 
                                         for k in TARGET_KEYS}
                        avg_ground_truth = {k: np.mean([g[k] for g in ground_truths]) 
                                          for k in TARGET_KEYS}
                        
                        # Calculate errors
                        errors = {k: abs(avg_predictions[k] - avg_ground_truth[k]) 
                                for k in TARGET_KEYS}
                    else:
                        # We only have evaluation_stats (predictions), no ground truth
                        avg_predictions = {k: np.mean([p[k] for p in predictions]) 
                                         for k in TARGET_KEYS}
                        # Use a placeholder for ground truth and errors
                        avg_ground_truth = {k: np.nan for k in TARGET_KEYS}
                        errors = {k: np.nan for k in TARGET_KEYS}
                    
                    row = {
                        'model': model_name,
                        'comment_num': comment_num,
                        'topic': topic,
                        'topic_type': 'Binary' if 'Binary' in topic else 'OpenQA'
                    }
                    
                    # Add predictions, ground truth, and errors
                    for key in TARGET_KEYS:
                        row[f'pred_{key}'] = avg_predictions[key]
                        row[f'true_{key}'] = avg_ground_truth[key]
                        row[f'error_{key}'] = errors[key]
                    
                    rows.append(row)
    
    return pd.DataFrame(rows)


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between predictions and ground truth."""
    correlations = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        for comment_num in model_df['comment_num'].unique():
            subset = model_df[model_df['comment_num'] == comment_num]
            
            if len(subset) > 1:
                row = {'model': model, 'comment_num': comment_num}
                
                for key in TARGET_KEYS:
                    pred_col = f'pred_{key}'
                    true_col = f'true_{key}'
                    
                    if pred_col in subset.columns and true_col in subset.columns:
                        corr = subset[pred_col].corr(subset[true_col], method='spearman')
                        row[f'corr_{key}'] = corr
                
                correlations.append(row)
    
    return pd.DataFrame(correlations)


def plot_performance_by_comments(df: pd.DataFrame, output_dir: Path):
    """Create visualizations for performance by number of comments."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Group by model and comment_num, calculate mean error
    grouped = df.groupby(['model', 'comment_num']).agg({
        f'error_{key}': 'mean' for key in TARGET_KEYS
    }).reset_index()
    
    for idx, key in enumerate(TARGET_KEYS):
        ax = axes[idx]
        
        # Plot for each model
        for model in sorted(df['model'].unique()):
            model_data = grouped[grouped['model'] == model]
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            
            ax.plot(model_data['comment_num'], 
                   model_data[f'error_{key}'],
                   marker='o', label=display_name, linewidth=2)
        
        ax.set_xlabel('Number of Comments', fontsize=12)
        ax.set_ylabel('Mean Absolute Error', fontsize=12)
        ax.set_title(f'{DISPLAY_NAMES[key]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle('Model Performance by Number of Comments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_comments.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data to CSV
    grouped.to_csv(output_dir / 'performance_by_comments.csv', index=False)
    print(f"Saved: performance_by_comments.png and .csv")


def plot_performance_by_topic(df: pd.DataFrame, output_dir: Path):
    """Create visualizations for performance by topic."""
    # Calculate mean error by topic and model
    topic_perf = df.groupby(['topic', 'model']).agg({
        f'error_{key}': 'mean' for key in TARGET_KEYS
    }).reset_index()
    
    # Create heatmap for each dimension
    for key in TARGET_KEYS:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Pivot for heatmap
        pivot_data = topic_perf.pivot(index='topic', columns='model', values=f'error_{key}')
        
        # Rename columns for display
        pivot_data.columns = [MODEL_DISPLAY_NAMES.get(col, col) for col in pivot_data.columns]
        
        # Sort topics by type (Binary vs OpenQA)
        sorted_topics = sorted(pivot_data.index, 
                              key=lambda x: (0 if 'Binary' in x else 1, x))
        pivot_data = pivot_data.reindex(sorted_topics)
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   cbar_kws={'label': 'Mean Absolute Error'}, ax=ax)
        
        ax.set_title(f'{DISPLAY_NAMES[key]} - Error by Topic and Model', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Topic', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{key}_by_topic.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save aggregated data to CSV
    topic_perf.to_csv(output_dir / 'performance_by_topic.csv', index=False)
    print(f"Saved: heatmap_*_by_topic.png and performance_by_topic.csv")


def plot_topic_type_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare performance between Binary and OpenQA topics."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    # Group by topic type and model
    topic_type_perf = df.groupby(['topic_type', 'model']).agg({
        f'error_{key}': 'mean' for key in TARGET_KEYS
    }).reset_index()
    
    for idx, key in enumerate(TARGET_KEYS):
        ax = axes[idx]
        
        # Create grouped bar plot
        x_pos = np.arange(len(df['model'].unique()))
        width = 0.35
        
        models = sorted(df['model'].unique())
        
        binary_data = []
        openqa_data = []
        
        for model in models:
            binary_val = topic_type_perf[(topic_type_perf['model'] == model) & 
                                        (topic_type_perf['topic_type'] == 'Binary')]
            openqa_val = topic_type_perf[(topic_type_perf['model'] == model) & 
                                        (topic_type_perf['topic_type'] == 'OpenQA')]
            
            binary_data.append(binary_val[f'error_{key}'].values[0] if not binary_val.empty else 0)
            openqa_data.append(openqa_val[f'error_{key}'].values[0] if not openqa_val.empty else 0)
        
        ax.bar(x_pos - width/2, binary_data, width, label='Binary', alpha=0.8)
        ax.bar(x_pos + width/2, openqa_data, width, label='OpenQA', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Mean Absolute Error', fontsize=10)
        ax.set_title(DISPLAY_NAMES[key], fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in models], 
                          rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Binary vs OpenQA Topic Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    topic_type_perf.to_csv(output_dir / 'topic_type_comparison.csv', index=False)
    print(f"Saved: topic_type_comparison.png and .csv")


def create_leaderboard(df: pd.DataFrame, output_dir: Path):
    """Create overall leaderboard ranking models."""
    # Calculate overall performance metrics
    leaderboard = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        row = {'model': MODEL_DISPLAY_NAMES.get(model, model)}
        
        # Calculate mean error and std for each dimension
        for key in TARGET_KEYS:
            row[f'{DISPLAY_NAMES[key]}_MAE'] = model_df[f'error_{key}'].mean()
            row[f'{DISPLAY_NAMES[key]}_STD'] = model_df[f'error_{key}'].std()
        
        # Calculate overall score (average of all dimensions)
        row['Overall_MAE'] = np.mean([row[f'{DISPLAY_NAMES[key]}_MAE'] for key in TARGET_KEYS])
        
        leaderboard.append(row)
    
    # Convert to DataFrame and sort by overall score
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df = leaderboard_df.sort_values('Overall_MAE')
    leaderboard_df['Rank'] = range(1, len(leaderboard_df) + 1)
    
    # Reorder columns
    cols = ['Rank', 'model', 'Overall_MAE'] + [col for col in leaderboard_df.columns 
                                                if col not in ['Rank', 'model', 'Overall_MAE']]
    leaderboard_df = leaderboard_df[cols]
    
    # Save leaderboard
    leaderboard_df.to_csv(output_dir / 'leaderboard.csv', index=False)
    print(f"Saved: leaderboard.csv")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = leaderboard_df['model'].values[::-1]  # Reverse for better display
    overall_scores = leaderboard_df['Overall_MAE'].values[::-1]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
    bars = ax.barh(models, overall_scores, color=colors)
    
    # Add value labels
    for bar, score in zip(bars, overall_scores):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
               f'{score:.4f}', va='center', fontsize=10)
    
    ax.set_xlabel('Overall Mean Absolute Error (Lower is Better)', fontsize=12)
    ax.set_title('Model Leaderboard - Overall Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'leaderboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: leaderboard.png")
    
    return leaderboard_df


def plot_correlation_analysis(df: pd.DataFrame, corr_df: pd.DataFrame, output_dir: Path):
    """Plot correlation analysis between predictions and ground truth."""
    if corr_df.empty:
        print("No correlation data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, key in enumerate(TARGET_KEYS):
        ax = axes[idx]
        
        # Plot correlation vs comment number for each model
        for model in sorted(corr_df['model'].unique()):
            model_data = corr_df[corr_df['model'] == model]
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            
            if f'corr_{key}' in model_data.columns:
                ax.plot(model_data['comment_num'], 
                       model_data[f'corr_{key}'],
                       marker='o', label=display_name, linewidth=2)
        
        ax.set_xlabel('Number of Comments', fontsize=12)
        ax.set_ylabel('Spearman Correlation', fontsize=12)
        ax.set_title(f'{DISPLAY_NAMES[key]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Moderate Correlation')
        ax.set_ylim([-0.1, 1.1])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle('Prediction-Truth Correlation by Number of Comments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save correlation data
    corr_df.to_csv(output_dir / 'correlation_analysis.csv', index=False)
    print(f"Saved: correlation_analysis.png and .csv")


def create_summary_statistics(df: pd.DataFrame, output_dir: Path):
    """Create summary statistics table."""
    summary = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        stats = {'Model': MODEL_DISPLAY_NAMES.get(model, model)}
        stats['N_Samples'] = len(model_df)
        stats['Comment_Nums'] = model_df['comment_num'].nunique()
        stats['Topics'] = model_df['topic'].nunique()
        
        # Calculate statistics for each dimension
        for key in TARGET_KEYS:
            error_col = f'error_{key}'
            stats[f'{DISPLAY_NAMES[key]}_Mean'] = model_df[error_col].mean()
            stats[f'{DISPLAY_NAMES[key]}_Median'] = model_df[error_col].median()
            stats[f'{DISPLAY_NAMES[key]}_Q25'] = model_df[error_col].quantile(0.25)
            stats[f'{DISPLAY_NAMES[key]}_Q75'] = model_df[error_col].quantile(0.75)
        
        summary.append(stats)
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values(f'{DISPLAY_NAMES[TARGET_KEYS[0]]}_Mean')
    
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"Saved: summary_statistics.csv")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Analyze regression evaluation results')
    parser.add_argument('--input_dir', type=str, 
                       default='/ibex/project/c2328/LLMs-Scalable-Deliberation/results/regression_evaluation',
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str,
                       default='/ibex/project/c2328/LLMs-Scalable-Deliberation/scripts/visualization/final_leaderboard/results',
                       help='Directory for output files')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading evaluation results from: {input_dir}")
    
    # Load all results
    results = load_evaluation_results(input_dir)
    print(f"Loaded results for {len(results)} models")
    
    # Extract metrics into DataFrame
    df = extract_metrics(results)
    print(f"Extracted {len(df)} data points")
    
    if df.empty:
        print("No data found to analyze!")
        return
    
    # Save raw data
    df.to_csv(output_dir / 'raw_metrics.csv', index=False)
    print(f"Saved: raw_metrics.csv")
    
    # Generate visualizations and analyses
    print("\nGenerating visualizations...")
    
    # 1. Performance by number of comments
    plot_performance_by_comments(df, output_dir)
    
    # 2. Performance by topic (heatmaps)
    plot_performance_by_topic(df, output_dir)
    
    # 3. Binary vs OpenQA comparison
    plot_topic_type_comparison(df, output_dir)
    
    # 4. Create leaderboard
    leaderboard = create_leaderboard(df, output_dir)
    print("\n=== TOP 5 MODELS ===")
    print(leaderboard[['Rank', 'model', 'Overall_MAE']].head())
    
    # 5. Correlation analysis
    corr_df = calculate_correlations(df)
    plot_correlation_analysis(df, corr_df, output_dir)
    
    # 6. Summary statistics
    summary = create_summary_statistics(df, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  - raw_metrics.csv: All extracted metrics")
    print(f"  - leaderboard.csv/png: Overall model rankings")
    print(f"  - performance_by_comments.csv/png: Performance vs comment number")
    print(f"  - performance_by_topic.csv: Performance by topic")
    print(f"  - heatmap_*_by_topic.png: Heatmaps for each dimension")
    print(f"  - topic_type_comparison.csv/png: Binary vs OpenQA comparison")
    print(f"  - correlation_analysis.csv/png: Prediction-truth correlations")
    print(f"  - summary_statistics.csv: Detailed statistics")


if __name__ == "__main__":
    main()