#!/usr/bin/env python3
"""
Visualize annotation data distributions
- Binary questions (scale_1, scale_2): 4 pairwise comparison questions
- 5-scale questions (scale_1 to scale_5): 4 rating questions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set font and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Unified color palette for all plots
UNIFIED_COLORS = plt.cm.Set3(np.linspace(0, 1, 5))

def _short_option_label(option: str) -> str:
    """Shorten option titles to fit axis display."""
    if option is None:
        return ""
    text = str(option)
    low = text.lower()
    # Common pattern mappings
    patterns = [
        ("not represented", "Not"),
        ("slightly represented", "Slightly"),
        ("partially represented", "Partially"),
        ("mostly represented", "Mostly"),
        ("fully represented", "Fully"),
        ("not informative", "Not"),
        ("slightly informative", "Slightly"),
        ("moderately informative", "Moderately"),
        ("very informative", "Very"),
        ("extremely informative", "Extremely"),
        ("very non-neutral", "Very non"),
        ("somewhat non-neutral", "Somewhat non"),
        ("moderately neutral", "Moderately"),
        ("fairly neutral", "Fairly"),
        ("completely neutral", "Completely"),
        ("strongly disapprove", "Str disapprove"),
        ("strongly approve", "Str approve"),
    ]
    for pat, rep in patterns:
        if pat in low:
            return rep
    # Generic cropping: for em dash/dash descriptions, take the first part
    for sep in [" — ", " - ", "–", "—"]:
        if sep in text:
            head = text.split(sep)[0].strip()
            if len(head) >= 1:
                text = head
                break
    # Final length limit
    return (text[:16] + "…") if len(text) > 17 else text

def _scale_order_patterns(question: str):
    """Given question text, return the corresponding ordered option patterns (from low to high)."""
    q = (question or "").lower()
    if "perspective represented" in q or "to what extent" in q:
        return [
            "not represented",
            "slightly represented",
            "partially represented",
            "mostly represented",
            "fully represented",
        ]
    if "how informative" in q:
        return [
            "not informative",
            "slightly informative",
            "moderately informative",
            "very informative",
            "extremely informative",
        ]
    if "neutral and balanced" in q:
        return [
            "very non-neutral",
            "somewhat non-neutral",
            "moderately neutral",
            "fairly neutral",
            "completely neutral",
        ]
    if "would you approve" in q:
        return [
            "strongly disapprove",
            "disapprove",
            "neutral",
            "approve",
            "strongly approve",
        ]
    return None

def _scale_rank(question: str, option_text: str) -> int:
    patterns = _scale_order_patterns(question)
    if not patterns:
        return 999
    low = (option_text or "").lower()
    for idx, token in enumerate(patterns):
        if token in low:
            return idx
    return 999

def _comparison_rank(option_text: str) -> int:
    """Fixed order for comparison questions: A much → A slightly → Both → B slightly → B much."""
    low = (option_text or "").lower()
    if "a is much" in low:
        return 0
    if "a is slightly" in low:
        return 1
    if "both are about the same" in low or "both about the same" in low:
        return 2
    if "b is slightly" in low:
        return 3
    if "b is much" in low:
        return 4
    return 999

def _short_comparison_label(option: str) -> str:
    """Shorten comparison option labels for x-axis display."""
    if option is None:
        return ""
    text = str(option).lower()
    
    # Common comparison patterns
    if "a is much" in text:
        return "A Much Better"
    if "a is slightly" in text:
        return "A Slightly Better"
    if "both are about the same" in text or "both about the same" in text:
        return "About the Same"
    if "b is slightly" in text:
        return "B Slightly Better"
    if "b is much" in text:
        return "B Much Better"
    
    # Fallback: truncate long text
    return (str(option)[:15] + "...") if len(str(option)) > 15 else str(option)

def load_data():
    """Load annotation data (prefer full_augment, fallback to full if not exists)"""
    path_augment = 'annotation/summary-rating/annotation_output/full_ablation/annotated_instances.csv'
    path_full = 'annotation/summary-rating/annotation_output/full/annotated_instances.csv'
    try:
        df = pd.read_csv(path_augment)
        print(f"Loaded {len(df)} annotation records (full_augment)")
        return df
    except FileNotFoundError:
        df = pd.read_csv(path_full)
        print(f"Loaded {len(df)} annotation records (full)")
        return df

def get_binary_columns(df):
    """Get columns corresponding to comparison questions (new format: question:::option). Returns {question: [cols...]}"""
    questions = [
        "Which summary is more representative of your perspective?",
        "Which summary is more informative?", 
        "Which summary presents a more neutral and balanced view of the issue?",
        "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
    ]
    binary_cols = {}
    for q in questions:
        cols = [c for c in df.columns if str(c).startswith(q)]
        binary_cols[q] = cols
    return binary_cols

def get_5scale_columns(df):
    """Get columns corresponding to 5-scale questions (new format: question:::option). Returns {question: [cols...]}"""
    questions = [
        "To what extent is your perspective represented in this response?",
        "How informative is this summary?",
        "Do you think this summary presents a neutral and balanced view of the issue?",
        "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?"
    ]
    scale_cols = {}
    for q in questions:
        cols = [c for c in df.columns if str(c).startswith(q)]
        scale_cols[q] = cols
    return scale_cols

def plot_binary_distributions(df, binary_cols):
    """Plot distributions for comparison questions (new 5-option format): count each option by "question:::option" columns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Define dimension titles for comparison questions
    dimension_titles = [
        "Representiveness",
        "Informativeness", 
        "Neutrality",
        "Policy Approval"
    ]
    
    for idx, (question, cols) in enumerate(binary_cols.items()):
        ax = axes[idx]
        
        # Count non-null values for each option
        option_counts = {}
        for col in cols:
            if col in df.columns:
                option = str(col).split(':::', 1)[-1].strip()
                option_counts[option] = int(df[col].notna().sum())
        
        if option_counts:
            # Fixed order rather than sorting by count
            items = list(option_counts.items())
            items_sorted = sorted(items, key=lambda t: (_comparison_rank(t[0]), t[0]))
            ordered_options = [opt for opt, _ in items_sorted]
            ordered_values = [option_counts[opt] for opt in ordered_options]
            
            # Plot bar chart
            bars = ax.bar(range(len(ordered_values)), ordered_values, 
                         color=UNIFIED_COLORS[:len(ordered_values)], alpha=0.7, edgecolor='black')
            
            ax.set_xticks(range(len(ordered_options)))
            short_labels = [_short_comparison_label(x) for x in ordered_options]
            ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=14, fontweight='bold')
            ax.set_ylabel('Count', fontsize=16, fontweight='bold')
            
            # Use standardized dimension title
            ax.set_title(f'{dimension_titles[idx]}', fontsize=14, pad=10, fontweight='bold')
            
            # Display values on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
            
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f'No data: {dimension_titles[idx]}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('annotation/summary-rating/notebooks/binary_distributions.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_5scale_distributions(df, scale_cols):
    """Plot distributions for 5-scale questions: count each option by "question:::option" columns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Define dimension titles for 5-scale questions
    dimension_titles = [
        "Representiveness",
        "Informativeness",
        "Neutrality", 
        "Policy Approval"
    ]
    
    for idx, (question, cols) in enumerate(scale_cols.items()):
        ax = axes[idx]
        
        # Count non-null values for each option
        option_counts = {}
        for col in cols:
            if col in df.columns:
                option = str(col).split(':::', 1)[-1].strip()
                option_counts[option] = int(df[col].notna().sum())
        
        if option_counts:
            # Sort by scale order defined for question type (unmatched items go to end)
            items = list(option_counts.items())
            items_sorted = sorted(items, key=lambda t: (_scale_rank(question, t[0]), t[0]))
            ordered_options = [opt for opt, _ in items_sorted]
            ordered_values = [option_counts[opt] for opt in ordered_options]
            
            # Plot bar chart
            bars = ax.bar(range(len(ordered_values)), ordered_values,
                         color=UNIFIED_COLORS[:len(ordered_values)], alpha=0.7, edgecolor='black')
            
            ax.set_xticks(range(len(ordered_options)))
            short_labels = [ _short_option_label(x) for x in ordered_options ]
            ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=14, fontweight='bold')
            ax.set_ylabel('Count', fontsize=16, fontweight='bold')
            
            # Use standardized dimension title
            ax.set_title(f'{dimension_titles[idx]}', fontsize=14, pad=10, fontweight='bold')
            
            # Display values on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
            
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f'No data: {dimension_titles[idx]}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('annotation/summary-rating/notebooks/5scale_distributions.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()

def print_data_summary(df, binary_cols, scale_cols):
    """Print data summary"""
    print("="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"Total records: {len(df)}")
    
    print(f"\nBINARY QUESTIONS (4 questions):")
    for question, cols in binary_cols.items():
        total_responses = 0
        for col in cols:
            if col in df.columns:
                total_responses += len(df[col].dropna())
        print(f"  {question[:50]}... : {total_responses} responses")
    
    print(f"\n5-SCALE QUESTIONS (4 questions):")
    for question, cols in scale_cols.items():
        total_responses = 0
        for col in cols:
            if col in df.columns:
                total_responses += len(df[col].dropna())
        print(f"  {question[:50]}... : {total_responses} responses")

def main():
    """Main function"""
    print("Loading and visualizing annotation data...")
    
    # Load data
    df = load_data()
    
    # Get question column names (dynamically match columns based on current data)
    binary_cols = get_binary_columns(df)
    scale_cols = get_5scale_columns(df)
    
    # Print data summary
    print_data_summary(df, binary_cols, scale_cols)
    
    print(f"\nGenerating visualizations...")
    
    # Plot binary question distributions
    print("Plotting binary distributions...")
    plot_binary_distributions(df, binary_cols)
    
    # Plot 5-scale question distributions  
    print("Plotting 5-scale distributions...")
    plot_5scale_distributions(df, scale_cols)
    
    print("Visualization complete!")
    print("Saved plots:")
    print("  - binary_distributions.pdf")
    print("  - 5scale_distributions.pdf")

if __name__ == "__main__":
    main()