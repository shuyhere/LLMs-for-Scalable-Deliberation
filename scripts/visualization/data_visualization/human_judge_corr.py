#!/usr/bin/env python3
"""
Script to process completed human-LLM correlation JSON files.
Filters out checkpoint files and analyzes correlation data.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class HumanJudgeCorrelationProcessor:
    """Process and analyze human-LLM correlation data from JSON files."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the processor with the data directory.
        
        Args:
            data_dir: Path to directory containing correlation JSON files
        """
        self.data_dir = Path(data_dir)
        self.completed_files = []
        self.correlation_data = []
        
        # Model name mapping for cleaner display
        self.model_name_mapping = {
            'web-rev-claude-sonnet-4-20250514': 'Claude-Sonnet-4',
            'web-rev-claude-opus-4-20250514': 'Claude-Opus-4',
            'web-rev-claude-3-7-sonnet-20250219': 'Claude-Sonnet-3.7',
            'qwen3-235b-a22b': 'Qwen3-235B',
            'qwen3-32b': 'Qwen3-32B',
            'qwen3-30b-a3b': 'Qwen3-30B',
            'qwen3-14b': 'Qwen3-14B',
            'qwen3-8b': 'Qwen3-8B',
            'qwen3-4b': 'Qwen3-4B',
            'qwen3-1.7b': 'Qwen3-1.7B',
            'qwen3-0.6b': 'Qwen3-0.6B',
            'gpt-5-mini': 'GPT-5-Mini',
            'gpt-5-nano': 'GPT-5-Nano',
            'gpt-4o-mini': 'GPT-4o-Mini',
            'gemini-2.5-flash-lite': 'Gemini-2.5-Flash',
            'deepseek-chat': 'DeepSeek-Chat'
        }
        
        # Dimension name mapping for cleaner display
        self.dimension_mapping = {
            'perspective_representation': 'Perspective',
            'informativeness': 'Informativeness',
            'neutrality_balance': 'Neutrality',
            'policy_approval': 'Policy'
        }
        
    def find_completed_files(self) -> List[str]:
        """
        Find all completed correlation JSON files (excluding checkpoint files).
        
        Returns:
            List of completed JSON file paths
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        # Find all JSON files that don't start with 'checkpoint_'
        json_files = list(self.data_dir.glob("*.json"))
        completed_files = [f for f in json_files if not f.name.startswith("checkpoint_")]
        
        self.completed_files = [str(f) for f in completed_files]
        print(f"Found {len(self.completed_files)} completed correlation files:")
        for file in self.completed_files:
            print(f"  - {Path(file).name}")
            
        return self.completed_files
    
    def load_correlation_data(self) -> List[Dict]:
        """
        Load correlation data from all completed JSON files.
        
        Returns:
            List of correlation data dictionaries
        """
        if not self.completed_files:
            self.find_completed_files()
            
        correlation_data = []
        
        for file_path in self.completed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract model name from filename
                model_name = Path(file_path).stem.replace("human_llm_correlation_", "")
                
                # Extract experiment metadata
                metadata = data.get("experiment_metadata", {})
                
                # Extract correlation results
                correlations = data.get("correlations", {})
                
                if correlations:
                    # Process each correlation metric
                    for metric_name, metric_data in correlations.items():
                        correlation_data.append({
                            'model': model_name,
                            'metric': metric_name,
                            'pearson_r': metric_data.get('pearson_r'),
                            'pearson_p': metric_data.get('pearson_p'),
                            'spearman_r': metric_data.get('spearman_r'),
                            'spearman_p': metric_data.get('spearman_p'),
                            'cohen_kappa': metric_data.get('cohen_kappa'),
                            'mae': metric_data.get('mae'),
                            'n_samples': metric_data.get('n_samples'),
                            'temperature': metadata.get('temperature'),
                            'n_rating_samples': metadata.get('n_rating_samples'),
                            'n_comparison_samples': metadata.get('n_comparison_samples')
                        })
                
                print(f"Loaded data for model: {model_name}")
                
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        self.correlation_data = correlation_data
        print(f"Total correlation metrics loaded: {len(correlation_data)}")
        return correlation_data
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a summary DataFrame from the correlation data.
        
        Returns:
            DataFrame with correlation metrics
        """
        if not self.correlation_data:
            self.load_correlation_data()
            
        df = pd.DataFrame(self.correlation_data)
        
        # Separate rating and comparison metrics
        df['task_type'] = df['metric'].str.split('_').str[0]
        df['dimension'] = df['metric'].str.split('_', n=1).str[1]
        
        # Apply model name mapping
        df['model_display'] = df['model'].map(self.model_name_mapping).fillna(df['model'])
        
        # Apply dimension name mapping
        df['dimension_display'] = df['dimension'].map(self.dimension_mapping).fillna(df['dimension'])
        
        return df
    
    def plot_correlation_summary(self, save_path: Optional[str] = None):
        """
        Create summary plots of correlation metrics for both rating and comparison tasks.
        
        Args:
            save_path: Optional path to save the plot
        """
        df = self.create_summary_dataframe()
        
        # Create subplots: 2 rows (rating/comparison) x 4 columns (metrics)
        fig, axes = plt.subplots(2, 4, figsize=(30, 12))
        fig.suptitle('Human-LLM Correlation Analysis Summary', fontsize=20, fontweight='bold')
        
        # Define metrics and their display names
        metrics = [
            ('pearson_r', 'Pearson r'),
            ('spearman_r', 'Spearman r'),
            ('cohen_kappa', "Cohen's κ"),
            ('mae', 'MAE')
        ]
        
        # Plot for rating task (top row)
        rating_df = df[df['task_type'] == 'rating']
        if not rating_df.empty:
            for i, (metric, metric_display) in enumerate(metrics):
                pivot_data = rating_df.pivot_table(
                    values=metric, 
                    index='model_display', 
                    columns='dimension_display', 
                    aggfunc='mean'
                )
                
                # Use different colormap for MAE (lower is better)
                cmap = 'RdYlBu_r' if metric != 'mae' else 'RdYlBu'
                center = 0 if metric != 'mae' else None
                
                sns.heatmap(pivot_data, annot=True, cmap=cmap, center=center,
                           ax=axes[0,i], cbar_kws={'label': metric_display}, fmt='.3f')
                axes[0,i].set_title(f'Rating - {metric_display}', fontweight='bold', fontsize=14)
                axes[0,i].set_xlabel('Dimension', fontweight='bold', fontsize=12)
                axes[0,i].set_ylabel('Model', fontweight='bold', fontsize=12)
                axes[0,i].tick_params(axis='x', rotation=0, labelsize=10)
                axes[0,i].tick_params(axis='y', rotation=0, labelsize=10)
        
        # Plot for comparison task (bottom row)
        comparison_df = df[df['task_type'] == 'comparison']
        if not comparison_df.empty:
            for i, (metric, metric_display) in enumerate(metrics):
                pivot_data = comparison_df.pivot_table(
                    values=metric, 
                    index='model_display', 
                    columns='dimension_display', 
                    aggfunc='mean'
                )
                
                # Use different colormap for MAE (lower is better)
                cmap = 'RdYlBu_r' if metric != 'mae' else 'RdYlBu'
                center = 0 if metric != 'mae' else None
                
                sns.heatmap(pivot_data, annot=True, cmap=cmap, center=center,
                           ax=axes[1,i], cbar_kws={'label': metric_display}, fmt='.3f')
                axes[1,i].set_title(f'Comparison - {metric_display}', fontweight='bold', fontsize=14)
                axes[1,i].set_xlabel('Dimension', fontweight='bold', fontsize=12)
                axes[1,i].set_ylabel('Model', fontweight='bold', fontsize=12)
                axes[1,i].tick_params(axis='x', rotation=0, labelsize=10)
                axes[1,i].tick_params(axis='y', rotation=0, labelsize=10)
        
        # Add row labels
        fig.text(0.02, 0.75, 'Rating Task', rotation=90, fontsize=16, fontweight='bold', va='center')
        fig.text(0.02, 0.25, 'Comparison Task', rotation=90, fontsize=16, fontweight='bold', va='center')
        
        # Adjust layout with more spacing
        plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=1.5)
        
        if save_path:
            # Ensure PDF format
            if not save_path.endswith('.pdf'):
                save_path = save_path.replace('.png', '.pdf')
            plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_individual_heatmaps(self, save_dir: Optional[str] = None):
        """
        Create individual heatmap plots for each metric and task type.
        
        Args:
            save_dir: Optional directory to save the plots
        """
        df = self.create_summary_dataframe()
        
        # Define metrics and their display names
        metrics = [
            ('pearson_r', 'Pearson r'),
            ('spearman_r', 'Spearman r'),
            ('cohen_kappa', "Cohen's κ"),
            ('mae', 'MAE')
        ]
        
        # Define task types
        task_types = ['rating', 'comparison']
        
        for task_type in task_types:
            task_df = df[df['task_type'] == task_type]
            if task_df.empty:
                continue
                
            for metric, metric_display in metrics:
                # Create individual heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                
                pivot_data = task_df.pivot_table(
                    values=metric, 
                    index='model_display', 
                    columns='dimension_display', 
                    aggfunc='mean'
                )
                
                # Use different colormap for MAE (lower is better)
                cmap = 'RdYlBu_r' if metric != 'mae' else 'RdYlBu'
                center = 0 if metric != 'mae' else None
                
                sns.heatmap(pivot_data, annot=True, cmap=cmap, center=center,
                           ax=ax, cbar_kws={'label': metric_display}, fmt='.3f')
                
                ax.set_title(f'{task_type.title()} Task - {metric_display}', 
                           fontsize=16, fontweight='bold')
                ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
                ax.set_ylabel('Model', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=0)
                ax.tick_params(axis='y', rotation=0)
                
                plt.tight_layout()
                
                if save_dir:
                    save_path = Path(save_dir) / f"{task_type}_{metric}_heatmap.pdf"
                    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
                    print(f"Individual heatmap saved to: {save_path}")
                
                plt.show()
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report of the correlation analysis.
        
        Returns:
            Summary report string
        """
        df = self.create_summary_dataframe()
        
        report = []
        report.append("=" * 60)
        report.append("HUMAN-LLM CORRELATION ANALYSIS SUMMARY")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        models = df['model'].unique()
        report.append(f"Number of models analyzed: {len(models)}")
        report.append(f"Models: {', '.join(models)}")
        report.append("")
        
        # Rating task summary
        rating_df = df[df['task_type'] == 'rating']
        if not rating_df.empty:
            report.append("RATING TASK CORRELATIONS:")
            report.append("-" * 30)
            
            for dimension in rating_df['dimension'].unique():
                dim_data = rating_df[rating_df['dimension'] == dimension]
                report.append(f"\n{dimension.replace('_', ' ').title()}:")
                
                # Best performing models
                best_pearson = dim_data.loc[dim_data['pearson_r'].idxmax()]
                best_spearman = dim_data.loc[dim_data['spearman_r'].idxmax()]
                best_kappa = dim_data.loc[dim_data['cohen_kappa'].idxmax()]
                lowest_mae = dim_data.loc[dim_data['mae'].idxmin()]
                
                report.append(f"  Best Pearson r: {best_pearson['model']} ({best_pearson['pearson_r']:.3f})")
                report.append(f"  Best Spearman r: {best_spearman['model']} ({best_spearman['spearman_r']:.3f})")
                report.append(f"  Best Cohen's κ: {best_kappa['model']} ({best_kappa['cohen_kappa']:.3f})")
                report.append(f"  Lowest MAE: {lowest_mae['model']} ({lowest_mae['mae']:.3f})")
                
                # Average across all models
                avg_pearson = dim_data['pearson_r'].mean()
                avg_spearman = dim_data['spearman_r'].mean()
                avg_kappa = dim_data['cohen_kappa'].mean()
                avg_mae = dim_data['mae'].mean()
                
                report.append(f"  Average Pearson r: {avg_pearson:.3f}")
                report.append(f"  Average Spearman r: {avg_spearman:.3f}")
                report.append(f"  Average Cohen's κ: {avg_kappa:.3f}")
                report.append(f"  Average MAE: {avg_mae:.3f}")
        
        # Comparison task summary
        comparison_df = df[df['task_type'] == 'comparison']
        if not comparison_df.empty:
            report.append("\n\nCOMPARISON TASK CORRELATIONS:")
            report.append("-" * 30)
            
            for dimension in comparison_df['dimension'].unique():
                dim_data = comparison_df[comparison_df['dimension'] == dimension]
                report.append(f"\n{dimension.replace('_', ' ').title()}:")
                
                # Best performing models
                best_pearson = dim_data.loc[dim_data['pearson_r'].idxmax()]
                best_spearman = dim_data.loc[dim_data['spearman_r'].idxmax()]
                best_kappa = dim_data.loc[dim_data['cohen_kappa'].idxmax()]
                lowest_mae = dim_data.loc[dim_data['mae'].idxmin()]
                
                report.append(f"  Best Pearson r: {best_pearson['model']} ({best_pearson['pearson_r']:.3f})")
                report.append(f"  Best Spearman r: {best_spearman['model']} ({best_spearman['spearman_r']:.3f})")
                report.append(f"  Best Cohen's κ: {best_kappa['model']} ({best_kappa['cohen_kappa']:.3f})")
                report.append(f"  Lowest MAE: {lowest_mae['model']} ({lowest_mae['mae']:.3f})")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_detailed_results(self, output_path: str):
        """
        Save detailed results to CSV files.
        
        Args:
            output_path: Directory to save the results
        """
        df = self.create_summary_dataframe()
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        full_path = output_dir / "correlation_results_full.csv"
        df.to_csv(full_path, index=False)
        print(f"Full results saved to: {full_path}")
        
        # Save rating task results
        rating_df = df[df['task_type'] == 'rating']
        if not rating_df.empty:
            rating_path = output_dir / "correlation_results_rating.csv"
            rating_df.to_csv(rating_path, index=False)
            print(f"Rating results saved to: {rating_path}")
        
        # Save comparison task results
        comparison_df = df[df['task_type'] == 'comparison']
        if not comparison_df.empty:
            comparison_path = output_dir / "correlation_results_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"Comparison results saved to: {comparison_path}")


def main():
    """Main function to run the correlation analysis."""
    parser = argparse.ArgumentParser(description="Process human-LLM correlation JSON files")
    parser.add_argument("--data_dir", 
                       default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/eval_llm_human_correlation",
                       help="Directory containing correlation JSON files")
    parser.add_argument("--output_dir", 
                       default="./results/correlation_analysis_output",
                       help="Directory to save analysis results")
    parser.add_argument("--plot", action="store_true", 
                       help="Generate heatmap plots")
    parser.add_argument("--individual", action="store_true", 
                       help="Generate individual heatmaps for each metric")
    parser.add_argument("--report", action="store_true", 
                       help="Generate summary report")
    parser.add_argument("--save_csv", action="store_true", 
                       help="Save results to CSV files")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = HumanJudgeCorrelationProcessor(args.data_dir)
    
    # Find and load completed files
    print("Finding completed correlation files...")
    processor.find_completed_files()
    
    print("\nLoading correlation data...")
    processor.load_correlation_data()
    
    # Generate summary report
    if args.report:
        print("\nGenerating summary report...")
        report = processor.generate_summary_report()
        print(report)
        
        # Save report to file
        report_path = Path(args.output_dir) / "correlation_summary_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
    
    # Generate plots
    if args.plot:
        print("\nGenerating summary heatmap...")
        plot_dir = Path(args.output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary heatmap (both rating and comparison tasks)
        summary_plot_path = plot_dir / "correlation_summary.pdf"
        processor.plot_correlation_summary(save_path=str(summary_plot_path))
    
    # Generate individual heatmaps
    if args.individual:
        print("\nGenerating individual heatmaps...")
        plot_dir = Path(args.output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        processor.plot_individual_heatmaps(save_dir=str(plot_dir))
    
    # Save CSV results
    if args.save_csv:
        print("\nSaving detailed results to CSV...")
        processor.save_detailed_results(args.output_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
