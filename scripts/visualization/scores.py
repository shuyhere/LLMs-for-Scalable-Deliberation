#!/usr/bin/env python3
"""
Score Distribution Visualization Script

This script analyzes and visualizes the distribution of evaluation scores from multiple datasets
and evaluation models. It reads eva_summary_*.json files and creates various visualizations
to understand the performance patterns across different models and datasets.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ScoreAnalyzer:
    """Analyze and visualize evaluation scores from multiple datasets and models."""
    
    def __init__(self, base_dir: str = "results/summary/gpt-4o-mini"):
        """
        Initialize the analyzer.
        
        Args:
            base_dir: Base directory containing evaluation results
        """
        self.base_dir = Path(base_dir)
        self.datasets = ["protest", "gun_use", "operation", "bowling-green"]
        self.score_data = []
        self.evaluation_models = set()
        
    def load_evaluation_data(self) -> pd.DataFrame:
        """
        Load evaluation data from all datasets.
        
        Returns:
            DataFrame containing all score data
        """
        print("Loading evaluation data...")
        
        for dataset in self.datasets:
            dataset_dir = self.base_dir / dataset
            eval_file = dataset_dir / f"eva_summary_{dataset}.json"
            
            if not eval_file.exists():
                print(f"Warning: Evaluation file not found for {dataset}")
                continue
                
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract evaluation data
                evaluations = data.get("evaluations", {})
                
                for model_name, eval_info in evaluations.items():
                    self.evaluation_models.add(model_name)
                    
                    # Extract scores from evaluation results
                    eval_data = eval_info.get("evaluation_data", {})
                    evaluation_results = eval_data.get("evaluation_results", [])
                    
                    for result in evaluation_results:
                        if isinstance(result, dict) and "score" in result:
                            score = result["score"]
                            comment_text = result.get("comment", "")[:100] + "..."  # Truncate long comments
                            
                            self.score_data.append({
                                "dataset": dataset,
                                "evaluation_model": model_name,
                                "score": score,
                                "comment": comment_text,
                                "comment_index": result.get("comment_index", 0)
                            })
                
                print(f"  Loaded {dataset}: {len(evaluations)} evaluation models")
                
            except Exception as e:
                print(f"Error loading {dataset}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(self.score_data)
        
        if df.empty:
            print("No evaluation data found!")
            return df
            
        print(f"\nTotal scores loaded: {len(df)}")
        print(f"Datasets: {df['dataset'].nunique()}")
        print(f"Evaluation models: {df['evaluation_model'].nunique()}")
        print(f"Score range: {df['score'].min():.1f} - {df['score'].max():.1f}")
        
        return df
    
    def create_score_distribution_plot(self, df: pd.DataFrame, save_path: str = "score_distribution.png"):
        """
        Create a comprehensive score distribution visualization.
        
        Args:
            df: DataFrame containing score data
            save_path: Path to save the plot
        """
        if df.empty:
            print("No data to plot!")
            return
            
        # Create figure with subplots - one for each evaluation model
        num_models = len(self.evaluation_models)
        if num_models == 0:
            print("No evaluation models found!")
            return
            
        # Determine subplot layout
        if num_models <= 4:
            rows, cols = 2, 2
        elif num_models <= 6:
            rows, cols = 2, 3
        elif num_models <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 3
            
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        fig.suptitle('Score Distribution by Evaluation Model', fontsize=16, fontweight='bold')
        
        # Flatten axes if it's a 2D array
        if num_models > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Create one subplot for each evaluation model
        for i, model in enumerate(sorted(self.evaluation_models)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            model_df = df[df['evaluation_model'] == model]
            
            if not model_df.empty:
                # Plot histogram for this model
                ax.hist(model_df['score'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], alpha=0.7, 
                       edgecolor='black', color=f'C{i}')
                ax.set_title(f'{model} Score Distribution')
                ax.set_xlabel('Score')
                ax.set_ylabel('Frequency')
                ax.set_xlim(0.5, 5.5)
                ax.set_xticks([1, 2, 3, 4, 5])
                
                # Add mean line
                mean_score = model_df['score'].mean()
                ax.axvline(mean_score, color='red', linestyle='--', 
                          label=f'Mean: {mean_score:.2f}')
                
                # Add statistics text
                stats_text = f'Count: {len(model_df)}\nStd: {model_df["score"].std():.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.8))
                
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'No data for {model}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model} Score Distribution')
        
        # Hide unused subplots
        for i in range(num_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Score distribution plot saved to: {save_path}")
        plt.show()
    
    def create_heatmap_plot(self, df: pd.DataFrame, save_path: str = "score_heatmap.png"):
        """
        Create a heatmap showing average scores by dataset and evaluation model.
        
        Args:
            df: DataFrame containing score data
            save_path: Path to save the plot
        """
        if df.empty:
            print("No data to plot!")
            return
            
        # Create pivot table for heatmap
        heatmap_data = df.pivot_table(
            values='score', 
            index='dataset', 
            columns='evaluation_model', 
            aggfunc=['mean', 'std', 'count']
        )
        
        # Plot mean scores heatmap
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Score Analysis Heatmaps', fontsize=16, fontweight='bold')
        
        # 1. Mean scores
        mean_scores = heatmap_data['mean']
        sns.heatmap(mean_scores, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=3.0, ax=axes[0], cbar_kws={'label': 'Mean Score'})
        axes[0].set_title('Mean Scores by Dataset and Model')
        axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
        
        # 2. Standard deviation
        std_scores = heatmap_data['std']
        sns.heatmap(std_scores, annot=True, fmt='.2f', cmap='Blues', 
                   ax=axes[1], cbar_kws={'label': 'Standard Deviation'})
        axes[1].set_title('Score Standard Deviation by Dataset and Model')
        axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
        
        # 3. Sample counts
        count_scores = heatmap_data['count']
        sns.heatmap(count_scores, annot=True, fmt='.0f', cmap='Purples', 
                   ax=axes[2], cbar_kws={'label': 'Number of Evaluations'})
        axes[2].set_title('Number of Evaluations by Dataset and Model')
        axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Score heatmap plot saved to: {save_path}")
        plt.show()
    

    
    def create_dataset_distribution_plot(self, df: pd.DataFrame, save_path: str = "dataset_distribution.png"):
        """
        Create a score distribution visualization by dataset.
        
        Args:
            df: DataFrame containing score data
            save_path: Path to save the plot
        """
        if df.empty:
            print("No data to plot!")
            return
            
        # Create figure with subplots - one for each dataset
        num_datasets = len(self.datasets)
        
        # Determine subplot layout
        if num_datasets <= 4:
            rows, cols = 2, 2
        elif num_datasets <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
            
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        fig.suptitle('Score Distribution by Dataset', fontsize=16, fontweight='bold')
        
        # Flatten axes if it's a 2D array
        if num_datasets > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Create one subplot for each dataset
        for i, dataset in enumerate(self.datasets):
            if i >= len(axes):
                break
                
            ax = axes[i]
            dataset_df = df[df['dataset'] == dataset]
            
            if not dataset_df.empty:
                # Plot histogram for this dataset
                ax.hist(dataset_df['score'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], alpha=0.7, 
                       edgecolor='black', color=f'C{i}')
                ax.set_title(f'{dataset.title()} Dataset Score Distribution')
                ax.set_xlabel('Score')
                ax.set_ylabel('Frequency')
                ax.set_xlim(0.5, 5.5)
                ax.set_xticks([1, 2, 3, 4, 5])
                
                # Add mean line
                mean_score = dataset_df['score'].mean()
                ax.axvline(mean_score, color='red', linestyle='--', 
                          label=f'Mean: {mean_score:.2f}')
                
                # Add statistics text
                stats_text = f'Count: {len(dataset_df)}\nStd: {dataset_df["score"].std():.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='lightblue', alpha=0.8))
                
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'No data for {dataset}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{dataset.title()} Dataset Score Distribution')
        
        # Hide unused subplots
        for i in range(num_datasets, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset distribution plot saved to: {save_path}")
        plt.show()


    
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics.
        
        Args:
            df: DataFrame containing score data
            
        Returns:
            DataFrame with summary statistics
        """
        if df.empty:
            return pd.DataFrame()
            
        # Group by dataset and evaluation model
        stats = df.groupby(['dataset', 'evaluation_model'])['score'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
        
        # Add overall statistics
        overall_stats = df.groupby('dataset')['score'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
        overall_stats['evaluation_model'] = 'Overall'
        overall_stats = overall_stats.reset_index().set_index(['dataset', 'evaluation_model'])
        
        # Combine individual and overall stats
        combined_stats = pd.concat([stats, overall_stats])
        
        return combined_stats
    
    def save_analysis_results(self, df: pd.DataFrame, output_dir: str = "analysis_results"):
        """
        Save all analysis results to files.
        
        Args:
            df: DataFrame containing score data
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw data
        df.to_csv(f"{output_dir}/raw_scores.csv", index=False)
        print(f"Raw scores saved to: {output_dir}/raw_scores.csv")
        
        # Save summary statistics
        stats_df = self.generate_summary_statistics(df)
        stats_df.to_csv(f"{output_dir}/summary_statistics.csv")
        print(f"Summary statistics saved to: {output_dir}/summary_statistics.csv")
        
        # Save detailed statistics by dataset
        for dataset in self.datasets:
            dataset_df = df[df['dataset'] == dataset]
            if not dataset_df.empty:
                dataset_stats = dataset_df.groupby('evaluation_model')['score'].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(3)
                dataset_stats.to_csv(f"{output_dir}/{dataset}_statistics.csv")
                print(f"{dataset} statistics saved to: {output_dir}/{dataset}_statistics.csv")
    
    def run_complete_analysis(self, output_dir: str = "analysis_results"):
        """
        Run the complete analysis pipeline.
        
        Args:
            output_dir: Directory to save all results
        """
        print("=" * 60)
        print("EVALUATION SCORE ANALYSIS")
        print("=" * 60)
        
        # Load data
        df = self.load_evaluation_data()
        
        if df.empty:
            print("No data to analyze!")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.create_score_distribution_plot(df, f"{output_dir}/score_distribution.png")
        self.create_heatmap_plot(df, f"{output_dir}/score_heatmap.png")
        self.create_dataset_distribution_plot(df, f"{output_dir}/dataset_distribution.png")
        
        # Generate statistics
        print("\nGenerating statistics...")
        self.save_analysis_results(df, output_dir)
        
        # Display summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {output_dir}/")
        print(f"Total scores analyzed: {len(df)}")
        print(f"Datasets processed: {df['dataset'].nunique()}")
        print(f"Evaluation models: {df['evaluation_model'].nunique()}")
        
        # Show top-level statistics
        stats_df = self.generate_summary_statistics(df)
        print(f"\nTop-level statistics:")
        print(stats_df.head(10))


def main():
    """Main function to run the score analysis."""
    
    # Initialize analyzer
    analyzer = ScoreAnalyzer("results/summary/gpt-4o-mini")
    
    # Run complete analysis
    analyzer.run_complete_analysis("score_analysis_results")


if __name__ == "__main__":
    main()
