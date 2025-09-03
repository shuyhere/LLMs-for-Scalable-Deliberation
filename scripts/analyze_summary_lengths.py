#!/usr/bin/env python3
"""
Script to analyze summary lengths for different sample sizes in the batch summarization results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_summary_lengths(json_file_path):
    """
    Analyze summary lengths for different sample sizes.
    
    Args:
        json_file_path: Path to the JSON file containing batch summarization results
    """
    
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract sample sizes and summary lengths
    sample_sizes = []
    topic_lengths = []
    main_lengths = []
    custom_lengths = []
    
    # Get the results by sample size
    results_by_sample_size = data.get("results_by_sample_size", {})
    
    for sample_size_str, result in results_by_sample_size.items():
        sample_size = int(sample_size_str)
        sample_sizes.append(sample_size)
        
        # Get summary lengths from statistics
        stats = result.get("statistics", {})
        topic_lengths.append(stats.get("topic_summary_length", 0))
        main_lengths.append(stats.get("main_summary_length", 0))
        custom_lengths.append(stats.get("custom_summary_length", 0))
    
    # Sort by sample size
    sorted_indices = np.argsort(sample_sizes)
    sample_sizes = [sample_sizes[i] for i in sorted_indices]
    topic_lengths = [topic_lengths[i] for i in sorted_indices]
    main_lengths = [main_lengths[i] for i in sorted_indices]
    custom_lengths = [custom_lengths[i] for i in sorted_indices]
    
    # Print analysis results
    print("=== SUMMARY LENGTH ANALYSIS ===")
    print(f"Dataset: {data.get('metadata', {}).get('dataset', 'Unknown')}")
    print(f"Model: {data.get('metadata', {}).get('model_used', 'Unknown')}")
    print(f"Total sample sizes processed: {len(sample_sizes)}")
    print()
    
    print("Sample Size | Topic | Main Points | Custom Analysis | Total")
    print("-" * 65)
    
    for i, size in enumerate(sample_sizes):
        total = topic_lengths[i] + main_lengths[i] + custom_lengths[i]
        print(f"{size:11d} | {topic_lengths[i]:5d} | {main_lengths[i]:10d} | {custom_lengths[i]:14d} | {total:5d}")
    
    print()
    
    # Calculate statistics
    print("=== STATISTICS ===")
    print(f"Topic Summary Lengths:")
    print(f"  Min: {min(topic_lengths)} characters")
    print(f"  Max: {max(topic_lengths)} characters")
    print(f"  Mean: {np.mean(topic_lengths):.1f} characters")
    print(f"  Std: {np.std(topic_lengths):.1f} characters")
    print()
    
    print(f"Main Points Summary Lengths:")
    print(f"  Min: {min(main_lengths)} characters")
    print(f"  Max: {max(main_lengths)} characters")
    print(f"  Mean: {np.mean(main_lengths):.1f} characters")
    print(f"  Std: {np.std(main_lengths):.1f} characters")
    print()
    
    print(f"Custom Analysis Summary Lengths:")
    print(f"  Min: {min(custom_lengths)} characters")
    print(f"  Max: {max(custom_lengths)} characters")
    print(f"  Mean: {np.mean(custom_lengths):.1f} characters")
    print(f"  Std: {np.std(custom_lengths):.1f} characters")
    print()
    
    # Calculate total lengths
    total_lengths = [t + m + c for t, m, c in zip(topic_lengths, main_lengths, custom_lengths)]
    print(f"Total Summary Lengths:")
    print(f"  Min: {min(total_lengths)} characters")
    print(f"  Max: {max(total_lengths)} characters")
    print(f"  Mean: {np.mean(total_lengths):.1f} characters")
    print(f"  Std: {np.std(total_lengths):.1f} characters")
    print()
    
    # Calculate length per comment
    print("=== LENGTH PER COMMENT ANALYSIS ===")
    for i, size in enumerate(sample_sizes):
        if size > 0:
            topic_per_comment = topic_lengths[i] / size
            main_per_comment = main_lengths[i] / size
            custom_per_comment = custom_lengths[i] / size
            total_per_comment = total_lengths[i] / size
            
            print(f"Sample size {size:3d}: Topic={topic_per_comment:6.1f}, Main={main_per_comment:6.1f}, Custom={custom_per_comment:6.1f}, Total={total_per_comment:6.1f} chars/comment")
    
    print()
    
    # Find patterns
    print("=== PATTERN ANALYSIS ===")
    
    # Check if lengths increase with sample size
    topic_correlation = np.corrcoef(sample_sizes, topic_lengths)[0, 1]
    main_correlation = np.corrcoef(sample_sizes, main_lengths)[0, 1]
    custom_correlation = np.corrcoef(sample_sizes, custom_lengths)[0, 1]
    total_correlation = np.corrcoef(sample_sizes, total_lengths)[0, 1]
    
    print(f"Correlation with sample size:")
    print(f"  Topic Summary: {topic_correlation:.3f}")
    print(f"  Main Points: {main_correlation:.3f}")
    print(f"  Custom Analysis: {custom_correlation:.3f}")
    print(f"  Total: {total_correlation:.3f}")
    
    # Check for outliers
    print("\nOutlier detection (using 2 standard deviations):")
    
    for summary_type, lengths in [("Topic", topic_lengths), ("Main Points", main_lengths), ("Custom Analysis", custom_lengths)]:
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        outliers = []
        
        for i, length in enumerate(lengths):
            if abs(length - mean_len) > 2 * std_len:
                outliers.append((sample_sizes[i], length))
        
        if outliers:
            print(f"  {summary_type}: {len(outliers)} outliers found")
            for size, length in outliers:
                print(f"    Sample size {size}: {length} chars (expected: {mean_len:.1f} ± {2*std_len:.1f})")
        else:
            print(f"  {summary_type}: No outliers detected")
    
    return {
        'sample_sizes': sample_sizes,
        'topic_lengths': topic_lengths,
        'main_lengths': main_lengths,
        'custom_lengths': custom_lengths,
        'total_lengths': total_lengths
    }

def create_main_points_plots(data, output_dir="results/analysis"):
    """
    Create specialized plots for Main Points analysis.
    
    Args:
        data: Dictionary containing the analysis data
        output_dir: Directory to save the plots
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Main Points Summary Analysis', fontsize=16, fontweight='bold')
    
    sample_sizes = data['sample_sizes']
    main_lengths = data['main_lengths']
    
    # Calculate Main Points length per comment
    main_per_comment = [m/s if s > 0 else 0 for m, s in zip(main_lengths, sample_sizes)]
    
    # Plot 1: Main Points Summary Length vs Sample Size
    ax1.plot(sample_sizes, main_lengths, 'o-', color='#2E86AB', linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Main Points Summary Length (characters)', fontsize=12, fontweight='bold')
    ax1.set_title('Main Points Summary Length by Sample Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # Add trend line
    z = np.polyfit(sample_sizes, main_lengths, 1)
    p = np.poly1d(z)
    ax1.plot(sample_sizes, p(sample_sizes), "--", color='#A23B72', alpha=0.8, linewidth=2, label=f'Trend line (slope: {z[0]:.1f})')
    ax1.legend(fontsize=10)
    
    # Add data points labels for key points
    for i, (size, length) in enumerate(zip(sample_sizes, main_lengths)):
        if i % 3 == 0 or i == len(sample_sizes) - 1:  # Label every 3rd point and the last one
            ax1.annotate(f'{length}', (size, length), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Main Points Length per Comment vs Sample Size
    ax2.plot(sample_sizes, main_per_comment, 's-', color='#F18F01', linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax2.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Main Points Length per Comment (characters)', fontsize=12, fontweight='bold')
    ax2.set_title('Main Points Length per Comment by Sample Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f9fa')
    
    # Add trend line for per-comment analysis
    z_per_comment = np.polyfit(sample_sizes, main_per_comment, 1)
    p_per_comment = np.poly1d(z_per_comment)
    ax2.plot(sample_sizes, p_per_comment(sample_sizes), "--", color='#C73E1D', alpha=0.8, linewidth=2, label=f'Trend line (slope: {z_per_comment[0]:.3f})')
    ax2.legend(fontsize=10)
    
    # Add data points labels for key points
    for i, (size, per_comment) in enumerate(zip(sample_sizes, main_per_comment)):
        if i % 3 == 0 or i == len(sample_sizes) - 1:  # Label every 3rd point and the last one
            ax2.annotate(f'{per_comment:.1f}', (size, per_comment), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"{output_dir}/main_points_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Main Points analysis plots saved to: {plot_path}")
    
    # Also save as PDF for high quality
    pdf_path = f"{output_dir}/main_points_analysis.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Main Points analysis plots also saved as PDF: {pdf_path}")
    
    plt.show()
    
    # Create additional detailed analysis plot
    create_detailed_main_points_plot(data, output_dir)

def create_detailed_main_points_plot(data, output_dir):
    """
    Create a detailed analysis plot with additional insights.
    
    Args:
        data: Dictionary containing the analysis data
        output_dir: Directory to save the plots
    """
    
    sample_sizes = data['sample_sizes']
    main_lengths = data['main_lengths']
    main_per_comment = [m/s if s > 0 else 0 for m, s in zip(main_lengths, sample_sizes)]
    
    # Create detailed plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Detailed Main Points Analysis', fontsize=18, fontweight='bold')
    
    # Plot 1: Raw lengths with confidence intervals
    ax1.plot(sample_sizes, main_lengths, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    ax1.fill_between(sample_sizes, 
                     [m - np.std(main_lengths) for m in main_lengths], 
                     [m + np.std(main_lengths) for m in main_lengths], 
                     alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Length (characters)')
    ax1.set_title('Main Points Length with Standard Deviation')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Per-comment analysis with moving average
    ax2.plot(sample_sizes, main_per_comment, 's-', color='#F18F01', linewidth=2, markersize=6)
    
    # Calculate moving average (window size 3)
    if len(main_per_comment) >= 3:
        moving_avg = []
        for i in range(len(main_per_comment)):
            start = max(0, i-1)
            end = min(len(main_per_comment), i+2)
            moving_avg.append(np.mean(main_per_comment[start:end]))
        ax2.plot(sample_sizes, moving_avg, '--', color='#C73E1D', linewidth=2, label='Moving Average (window=3)')
        ax2.legend()
    
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Length per Comment (characters)')
    ax2.set_title('Main Points Length per Comment with Moving Average')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Growth rate analysis
    if len(main_lengths) > 1:
        growth_rates = []
        for i in range(1, len(main_lengths)):
            if main_lengths[i-1] > 0:
                growth_rate = (main_lengths[i] - main_lengths[i-1]) / main_lengths[i-1] * 100
                growth_rates.append(growth_rate)
            else:
                growth_rates.append(0)
        
        ax3.plot(sample_sizes[1:], growth_rates, '^-', color='#A23B72', linewidth=2, markersize=6)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Sample Size')
        ax3.set_ylabel('Growth Rate (%)')
        ax3.set_title('Main Points Length Growth Rate')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency analysis (length vs sample size ratio)
    efficiency = [m/s if s > 0 else 0 for m, s in zip(main_lengths, sample_sizes)]
    ax4.plot(sample_sizes, efficiency, 'D-', color='#6B5B95', linewidth=2, markersize=6)
    ax4.set_xlabel('Sample Size')
    ax4.set_ylabel('Efficiency (length/sample_size)')
    ax4.set_title('Main Points Generation Efficiency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save detailed plot
    detailed_plot_path = f"{output_dir}/main_points_detailed_analysis.png"
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Detailed Main Points analysis saved to: {detailed_plot_path}")
    
    plt.show()

def create_visualization(data, output_dir="results/analysis"):
    """
    Create visualizations of the summary length analysis.
    
    Args:
        data: Dictionary containing the analysis data
        output_dir: Directory to save the plots
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Summary Length Analysis by Sample Size', fontsize=16, fontweight='bold')
    
    sample_sizes = data['sample_sizes']
    topic_lengths = data['topic_lengths']
    main_lengths = data['main_lengths']
    custom_lengths = data['custom_lengths']
    total_lengths = data['total_lengths']
    
    # Plot 1: Individual summary types
    axes[0, 0].plot(sample_sizes, topic_lengths, 'o-', label='Topic Summary', linewidth=2, markersize=6)
    axes[0, 0].plot(sample_sizes, main_lengths, 's-', label='Main Points', linewidth=2, markersize=6)
    axes[0, 0].plot(sample_sizes, custom_lengths, '^-', label='Custom Analysis', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Sample Size')
    axes[0, 0].set_ylabel('Length (characters)')
    axes[0, 0].set_title('Summary Lengths by Type')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Total length
    axes[0, 1].plot(sample_sizes, total_lengths, 'o-', color='red', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Sample Size')
    axes[0, 1].set_ylabel('Total Length (characters)')
    axes[0, 1].set_title('Total Summary Length')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Length per comment
    topic_per_comment = [t/s if s > 0 else 0 for t, s in zip(topic_lengths, sample_sizes)]
    main_per_comment = [m/s if s > 0 else 0 for m, s in zip(main_lengths, sample_sizes)]
    custom_per_comment = [c/s if s > 0 else 0 for c, s in zip(custom_lengths, sample_sizes)]
    total_per_comment = [tot/s if s > 0 else 0 for tot, s in zip(total_lengths, sample_sizes)]
    
    axes[1, 0].plot(sample_sizes, topic_per_comment, 'o-', label='Topic Summary', linewidth=2, markersize=6)
    axes[1, 0].plot(sample_sizes, main_per_comment, 's-', label='Main Points', linewidth=2, markersize=6)
    axes[1, 0].plot(sample_sizes, custom_per_comment, '^-', label='Custom Analysis', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Sample Size')
    axes[1, 0].set_ylabel('Length per Comment (characters)')
    axes[1, 0].set_title('Summary Length per Comment')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Length distribution
    axes[1, 1].hist(topic_lengths, alpha=0.7, label='Topic Summary', bins=10)
    axes[1, 1].hist(main_lengths, alpha=0.7, label='Main Points', bins=10)
    axes[1, 1].hist(custom_lengths, alpha=0.7, label='Custom Analysis', bins=10)
    axes[1, 1].set_xlabel('Length (characters)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Length Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"{output_dir}/summary_length_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_path}")
    
    plt.show()

def main():
    """Main function."""
    
    # Path to the JSON file
    json_file = "results/summary/gpt5nano/green_batch_summaries.json"
    
    if not Path(json_file).exists():
        print(f"Error: File not found: {json_file}")
        return
    
    # Analyze the data
    data = analyze_summary_lengths(json_file)
    
    # Create specialized Main Points plots
    try:
        print("\n" + "="*60)
        print("Creating Main Points Analysis Plots...")
        print("="*60)
        create_main_points_plots(data)
    except ImportError:
        print("Matplotlib not available. Skipping Main Points visualization.")
    except Exception as e:
        print(f"Error creating Main Points visualization: {e}")
    
    # Create general visualizations
    try:
        print("\n" + "="*60)
        print("Creating General Summary Analysis Plots...")
        print("="*60)
        create_visualization(data)
    except ImportError:
        print("Matplotlib not available. Skipping general visualization.")
    except Exception as e:
        print(f"Error creating general visualization: {e}")

if __name__ == "__main__":
    main()
