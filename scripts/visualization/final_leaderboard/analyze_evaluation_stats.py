#!/usr/bin/env python3
"""
Analysis script for regression evaluation results using evaluation_stats.
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
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
    "deepseek-chat": "DeepSeek Chat",
    "deepseek-reasoner": "DeepSeek Reasoner",
    "grok-2-latest": "Grok 2",
    "qwen3-0.5b": "Qwen3 0.5B",
    "qwen3-1.5b": "Qwen3 1.5B",
    "qwen3-3b": "Qwen3 3B",
    "qwen3-7b": "Qwen3 7B",
    "qwen3-14b": "Qwen3 14B",
    "qwen3-30b-a14b": "Qwen3 30B",
    "qwen3-32b": "Qwen3 32B",
    "qwen3-235b-a22b": "Qwen3 235B",
    "web-gpt-claude-sonnet-3-20241022": "Claude Sonnet",
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
    """Extract metrics from evaluation_stats into a structured DataFrame."""
    rows = []
    
    for model_name, comment_data in results.items():
        for comment_num, eval_data in comment_data.items():
            if 'results' not in eval_data:
                continue
            
            for topic_result in eval_data['results']:
                topic = topic_result['topic']
                
                # Collect mean values from evaluation_stats
                mean_values = {key: [] for key in TARGET_KEYS}
                std_values = {key: [] for key in TARGET_KEYS}
                
                for summary in topic_result.get('summaries', []):
                    if 'evaluation_stats' in summary:
                        eval_stats = summary['evaluation_stats']
                        for key in TARGET_KEYS:
                            if key in eval_stats:
                                mean_values[key].append(eval_stats[key].get('mean', 0.0))
                                std_values[key].append(eval_stats[key].get('std', 0.0))
                
                if mean_values[TARGET_KEYS[0]]:  # Check if we have data
                    row = {
                        'model': model_name,
                        'comment_num': comment_num,
                        'topic': topic,
                        'topic_type': 'Binary' if 'Binary' in topic else 'OpenQA'
                    }
                    
                    # Average across summaries for each dimension
                    for key in TARGET_KEYS:
                        if mean_values[key]:
                            row[f'mean_{key}'] = np.mean(mean_values[key])
                            row[f'std_{key}'] = np.mean(std_values[key])
                    
                    rows.append(row)
    
    return pd.DataFrame(rows)


def plot_performance_by_comments(df: pd.DataFrame, output_dir: Path):
    """Create visualizations for performance by number of comments with error bars."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Group by model and comment_num, calculate mean, min, max values
    grouped = df.groupby(['model', 'comment_num']).agg({
        f'mean_{key}': ['mean', 'min', 'max'] for key in TARGET_KEYS
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['model', 'comment_num'] + [
        f'{col[0]}_{col[1]}' if col[1] else col[0] 
        for col in grouped.columns[2:]
    ]
    
    for idx, key in enumerate(TARGET_KEYS):
        ax = axes[idx]
        
        # Plot for each model with error bars
        for model in sorted(df['model'].unique()):
            model_data = grouped[grouped['model'] == model]
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            
            mean_col = f'mean_{key}_mean'
            min_col = f'mean_{key}_min'
            max_col = f'mean_{key}_max'
            
            # Calculate error bars (distance from mean to min/max)
            yerr_lower = model_data[mean_col] - model_data[min_col]
            yerr_upper = model_data[max_col] - model_data[mean_col]
            yerr = [yerr_lower.values, yerr_upper.values]
            
            ax.errorbar(model_data['comment_num'], 
                       model_data[mean_col],
                       yerr=yerr,
                       marker='o', label=display_name, linewidth=2,
                       capsize=3, capthick=1, alpha=0.8)
        
        ax.set_xlabel('Number of Comments', fontsize=12)
        ax.set_ylabel('Mean Score', fontsize=12)
        ax.set_title(f'{DISPLAY_NAMES[key]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_ylim([0, 1])
    
    plt.suptitle('Model Performance by Number of Comments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_comments.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data to CSV
    grouped.to_csv(output_dir / 'performance_by_comments.csv', index=False)
    print(f"Saved: performance_by_comments.pdf and .csv")


def plot_performance_by_topic(df: pd.DataFrame, output_dir: Path):
    """Create visualizations for performance by topic."""
    # Calculate mean scores by topic and model
    topic_perf = df.groupby(['topic', 'model']).agg({
        f'mean_{key}': 'mean' for key in TARGET_KEYS
    }).reset_index()
    
    # Create heatmap for each dimension
    for key in TARGET_KEYS:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Pivot for heatmap
        pivot_data = topic_perf.pivot(index='topic', columns='model', values=f'mean_{key}')
        
        # Rename columns for display
        pivot_data.columns = [MODEL_DISPLAY_NAMES.get(col, col) for col in pivot_data.columns]
        
        # Sort topics by type (Binary vs OpenQA)
        sorted_topics = sorted(pivot_data.index, 
                              key=lambda x: (0 if 'Binary' in x else 1, x))
        pivot_data = pivot_data.reindex(sorted_topics)
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Mean Score'}, ax=ax, vmin=0, vmax=1)
        
        ax.set_title(f'{DISPLAY_NAMES[key]} - Scores by Topic and Model', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Topic', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{key}_by_topic.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save aggregated data to CSV
    topic_perf.to_csv(output_dir / 'performance_by_topic.csv', index=False)
    print(f"Saved: heatmap_*_by_topic.pdf and performance_by_topic.csv")


def plot_topic_type_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare performance between Binary and OpenQA topics with error bars."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    # Group by topic type and model, calculate mean, std, min, max
    topic_type_perf = df.groupby(['topic_type', 'model']).agg({
        f'mean_{key}': ['mean', 'std', 'min', 'max'] for key in TARGET_KEYS
    }).reset_index()
    
    # Flatten column names
    topic_type_perf.columns = ['topic_type', 'model'] + [
        f'{col[0]}_{col[1]}' if col[1] else col[0] 
        for col in topic_type_perf.columns[2:]
    ]
    
    for idx, key in enumerate(TARGET_KEYS):
        ax = axes[idx]
        
        # Create grouped bar plot with error bars
        models = sorted(df['model'].unique())
        x_pos = np.arange(len(models))
        width = 0.35
        
        binary_data = []
        binary_err = []
        openqa_data = []
        openqa_err = []
        
        for model in models:
            binary_val = topic_type_perf[(topic_type_perf['model'] == model) & 
                                        (topic_type_perf['topic_type'] == 'Binary')]
            openqa_val = topic_type_perf[(topic_type_perf['model'] == model) & 
                                        (topic_type_perf['topic_type'] == 'OpenQA')]
            
            if not binary_val.empty:
                mean_val = binary_val[f'mean_{key}_mean'].values[0]
                min_val = binary_val[f'mean_{key}_min'].values[0]
                max_val = binary_val[f'mean_{key}_max'].values[0]
                binary_data.append(mean_val)
                binary_err.append([mean_val - min_val, max_val - mean_val])
            else:
                binary_data.append(0)
                binary_err.append([0, 0])
            
            if not openqa_val.empty:
                mean_val = openqa_val[f'mean_{key}_mean'].values[0]
                min_val = openqa_val[f'mean_{key}_min'].values[0]
                max_val = openqa_val[f'mean_{key}_max'].values[0]
                openqa_data.append(mean_val)
                openqa_err.append([mean_val - min_val, max_val - mean_val])
            else:
                openqa_data.append(0)
                openqa_err.append([0, 0])
        
        # Transpose error arrays for matplotlib
        binary_err = np.array(binary_err).T
        openqa_err = np.array(openqa_err).T
        
        ax.bar(x_pos - width/2, binary_data, width, yerr=binary_err, 
               label='Binary', alpha=0.8, capsize=2)
        ax.bar(x_pos + width/2, openqa_data, width, yerr=openqa_err, 
               label='OpenQA', alpha=0.8, capsize=2)
        
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Mean Score', fontsize=10)
        ax.set_title(DISPLAY_NAMES[key], fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in models], 
                          rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.suptitle('Binary vs OpenQA Topic Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_type_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    topic_type_perf.to_csv(output_dir / 'topic_type_comparison.csv', index=False)
    print(f"Saved: topic_type_comparison.pdf and .csv")


def create_leaderboard(df: pd.DataFrame, output_dir: Path):
    """Create overall leaderboard ranking models with min/max ranges."""
    # Calculate overall performance metrics
    leaderboard = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        row = {'model': MODEL_DISPLAY_NAMES.get(model, model)}
        
        # Calculate mean score, std, min, max for each dimension
        for key in TARGET_KEYS:
            row[f'{DISPLAY_NAMES[key]}_Mean'] = model_df[f'mean_{key}'].mean()
            row[f'{DISPLAY_NAMES[key]}_STD'] = model_df[f'mean_{key}'].std()
            row[f'{DISPLAY_NAMES[key]}_Min'] = model_df[f'mean_{key}'].min()
            row[f'{DISPLAY_NAMES[key]}_Max'] = model_df[f'mean_{key}'].max()
        
        # Calculate overall score (average of all dimensions)
        row['Overall_Score'] = np.mean([row[f'{DISPLAY_NAMES[key]}_Mean'] for key in TARGET_KEYS])
        row['Overall_Min'] = np.mean([row[f'{DISPLAY_NAMES[key]}_Min'] for key in TARGET_KEYS])
        row['Overall_Max'] = np.mean([row[f'{DISPLAY_NAMES[key]}_Max'] for key in TARGET_KEYS])
        
        leaderboard.append(row)
    
    # Convert to DataFrame and sort by overall score (descending for scores)
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df = leaderboard_df.sort_values('Overall_Score', ascending=False)
    leaderboard_df['Rank'] = range(1, len(leaderboard_df) + 1)
    
    # Reorder columns
    cols = ['Rank', 'model', 'Overall_Score'] + [col for col in leaderboard_df.columns 
                                                if col not in ['Rank', 'model', 'Overall_Score']]
    leaderboard_df = leaderboard_df[cols]
    
    # Save leaderboard
    leaderboard_df.to_csv(output_dir / 'leaderboard.csv', index=False)
    print(f"Saved: leaderboard.csv")
    
    # Create visualization with error bars
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = leaderboard_df['model'].values
    overall_scores = leaderboard_df['Overall_Score'].values
    overall_mins = leaderboard_df['Overall_Min'].values
    overall_maxs = leaderboard_df['Overall_Max'].values
    
    # Calculate error bars
    xerr_lower = overall_scores - overall_mins
    xerr_upper = overall_maxs - overall_scores
    xerr = np.array([xerr_lower, xerr_upper])
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models)))
    bars = ax.barh(models, overall_scores, xerr=xerr, color=colors,
                   capsize=3, error_kw={'linewidth': 1.5, 'ecolor': 'gray'})
    
    # Add value labels with range
    for i, (bar, score, min_val, max_val) in enumerate(zip(bars, overall_scores, overall_mins, overall_maxs)):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{score:.4f}\n[{min_val:.4f}-{max_val:.4f}]', 
               va='center', fontsize=9)
    
    ax.set_xlabel('Overall Mean Score (Higher is Better)', fontsize=12)
    ax.set_title('Model Leaderboard - Overall Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'leaderboard.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: leaderboard.pdf")
    
    return leaderboard_df


def plot_variance_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze variance (std) across models and dimensions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Group by model, calculate mean std across all topics/comments
    std_data = df.groupby('model').agg({
        f'std_{key}': 'mean' for key in TARGET_KEYS
    }).reset_index()
    
    for idx, key in enumerate(TARGET_KEYS):
        ax = axes[idx]
        
        models = std_data['model'].values
        std_values = std_data[f'std_{key}'].values
        
        # Sort by std values
        sorted_idx = np.argsort(std_values)
        models = models[sorted_idx]
        std_values = std_values[sorted_idx]
        
        display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
        bars = ax.barh(display_names, std_values, color=colors)
        
        # Add value labels
        for bar, val in zip(bars, std_values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', va='center', fontsize=9)
        
        ax.set_xlabel('Mean Standard Deviation', fontsize=11)
        ax.set_title(f'{DISPLAY_NAMES[key]} - Variance Analysis', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Model Prediction Variance (Lower is More Consistent)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'variance_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save variance data
    std_data.to_csv(output_dir / 'variance_analysis.csv', index=False)
    print(f"Saved: variance_analysis.pdf and .csv")


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
            mean_col = f'mean_{key}'
            stats[f'{DISPLAY_NAMES[key]}_Mean'] = model_df[mean_col].mean()
            stats[f'{DISPLAY_NAMES[key]}_Median'] = model_df[mean_col].median()
            stats[f'{DISPLAY_NAMES[key]}_Q25'] = model_df[mean_col].quantile(0.25)
            stats[f'{DISPLAY_NAMES[key]}_Q75'] = model_df[mean_col].quantile(0.75)
        
        summary.append(stats)
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values(f'{DISPLAY_NAMES[TARGET_KEYS[0]]}_Mean', ascending=False)
    
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
    
    if results:
        # Print model names and comment numbers
        for model, comment_data in results.items():
            comment_nums = sorted(comment_data.keys())
            print(f"  - {model}: {len(comment_nums)} comment numbers {comment_nums}")
    
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
    print(leaderboard[['Rank', 'model', 'Overall_Score']].head())
    
    # 5. Variance analysis
    plot_variance_analysis(df, output_dir)
    
    # 6. Summary statistics
    summary = create_summary_statistics(df, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  - raw_metrics.csv: All extracted metrics")
    print(f"  - leaderboard.csv/pdf: Overall model rankings with error bars")
    print(f"  - performance_by_comments.csv/pdf: Performance vs comment number with error bars")
    print(f"  - performance_by_topic.csv: Performance by topic")
    print(f"  - heatmap_*_by_topic.pdf: Heatmaps for each dimension")
    print(f"  - topic_type_comparison.csv/pdf: Binary vs OpenQA comparison with error bars")
    print(f"  - variance_analysis.csv/pdf: Model prediction variance")
    print(f"  - summary_statistics.csv: Detailed statistics")


if __name__ == "__main__":
    main()