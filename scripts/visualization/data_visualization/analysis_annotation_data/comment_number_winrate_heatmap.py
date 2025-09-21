#!/usr/bin/env python3
"""
Create comment number vs comment number win rate heatmaps from comparison data.

This script:
1. Reads comparison data from full_augment annotations
2. Extracts summary IDs and comment numbers directly from assigned_user_data.json
3. Calculates win rates for each comment number vs comment number combination
4. Creates heatmaps for each dimension showing A win rates
"""

from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path('/ibex/project/c2328/LLMs-Scalable-Deliberation')
ANNOTATED_DIR = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full_augment'
OUTPUT_DIR = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation/comments_winrate_heatmap'


def get_dimension_questions() -> Dict[str, str]:
    """Get comparison questions for each dimension."""
    return {
        'perspective': "Which summary is more representative of your perspective? ",
        'informativeness': "Which summary is more informative? ",
        'neutrality': "Which summary presents a more neutral and balanced view of the issue? ",
        'policy': "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
    }


def get_dimension_display_names() -> Dict[str, str]:
    """Get display names for dimensions."""
    return {
        'perspective': 'Representiveness',
        'informativeness': 'Informativeness', 
        'neutrality': 'Neutrality',
        'policy': 'Policy Approval'
    }


def load_annotation_data() -> pd.DataFrame:
    """Load annotation data from JSONL files and assigned_user_data.json files."""
    all_data = []
    user_dirs = [d for d in ANNOTATED_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(user_dirs)} user directories")
    
    for user_dir in user_dirs:
        # Load annotated_instances.jsonl
        jsonl_file = user_dir / 'annotated_instances.jsonl'
        if jsonl_file.exists():
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        all_data.append(data)
        
        # Load assigned_user_data.json for metadata
        assigned_file = user_dir / 'assigned_user_data.json'
        if assigned_file.exists():
            with open(assigned_file, 'r', encoding='utf-8') as f:
                assigned_data = json.load(f)
                # Store assigned data for later use
                for item_id, item_data in assigned_data.items():
                    if item_data.get('type') == 'comparison':
                        # Add metadata to the corresponding annotation record
                        for ann_data in all_data:
                            if ann_data.get('id') == item_id:
                                ann_data.update(item_data)
                                break
    
    print(f"Loaded {len(all_data)} annotation records")
    return pd.DataFrame(all_data)


def extract_comparison_choice(row: pd.Series, question: str) -> float:
    """Extract comparison choice from nested label_annotations structure."""
    label_annotations = row.get('label_annotations', {})
    if question in label_annotations:
        question_data = label_annotations[question]
        if isinstance(question_data, dict):
            values = list(question_data.values())
            if values:
                value_str = str(values[0])
                try:
                    value = float(value_str)
                    # Map 1-5 scale: 1,2 -> 1.0 (A wins), 4,5 -> 0.0 (B wins), 3 -> 0.5 (neutral, to be filtered)
                    if value == 1 or value == 2:
                        return 1.0
                    elif value == 3:
                        return 0.5  # Neutral, will be filtered out
                    elif value == 4 or value == 5:
                        return 0.0
                    else:
                        return np.nan
                except (ValueError, TypeError):
                    return np.nan
        else:
            # Fallback for direct value
            value_str = str(question_data)
            try:
                value = float(value_str)
                if value == 1 or value == 2:
                    return 1.0
                elif value == 3:
                    return 0.5
                elif value == 4 or value == 5:
                    return 0.0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
    return np.nan


def process_comparison_data() -> pd.DataFrame:
    """Process comparison data and create comment number vs comment number win rate data."""
    dims = get_dimension_questions()
    
    # Load data
    ann = load_annotation_data()
    
    # Filter for comparison data only
    comp_rows = ann[ann['id'].str.contains('_comparison', na=False)].copy()
    print(f"Found {len(comp_rows)} comparison records")
    
    # Extract comment numbers directly from the data
    comp_rows['comment_num_a'] = pd.to_numeric(comp_rows['num_samples_group_a'], errors='coerce')
    comp_rows['comment_num_b'] = pd.to_numeric(comp_rows['num_samples_group_b'], errors='coerce')
    
    print(f"Comment num A - Valid: {comp_rows['comment_num_a'].notna().sum()}, NaN: {comp_rows['comment_num_a'].isna().sum()}")
    print(f"Comment num B - Valid: {comp_rows['comment_num_b'].notna().sum()}, NaN: {comp_rows['comment_num_b'].isna().sum()}")
    
    # Process each dimension
    all_records = []
    for dim, question in dims.items():
        print(f"\nProcessing dimension: {dim}")
        
        # Extract comparison choices
        comp_rows[f'choice_{dim}'] = comp_rows.apply(lambda r: extract_comparison_choice(r, question), axis=1)
        
        # Debug: check choice distribution
        choice_counts = comp_rows[f'choice_{dim}'].value_counts(dropna=False)
        print(f"  Choice distribution: {dict(choice_counts)}")
        
        # Filter out neutral choices and NaN values
        dim_data = comp_rows.dropna(subset=[f'choice_{dim}', 'comment_num_a', 'comment_num_b'])
        print(f"  After removing NaN: {len(dim_data)}")
        
        dim_data = dim_data[dim_data[f'choice_{dim}'] != 0.5]  # Remove "Both are about the same"
        print(f"  After removing neutral (0.5): {len(dim_data)}")
        
        if not dim_data.empty:
            # A wins if choice == 1, B wins if choice == 0
            dim_data['a_wins'] = (dim_data[f'choice_{dim}'] == 1.0).astype(int)
            dim_data['dimension'] = dim
            
            # Keep only necessary columns
            records = dim_data[['comment_num_a', 'comment_num_b', 'a_wins', 'dimension']].copy()
            all_records.append(records)
            
            print(f"  Records with valid data: {len(records)}")
        else:
            print(f"  No valid data for {dim}")
    
    if all_records:
        result_df = pd.concat(all_records, ignore_index=True)
        print(f"\nTotal records created: {len(result_df)}")
        return result_df
    else:
        print("No valid data found!")
        return pd.DataFrame()


def create_winrate_heatmaps(comp_df: pd.DataFrame, output_dir: Path) -> None:
    """Create comment number vs comment number win rate heatmaps for each dimension."""
    dims = list(get_dimension_questions().keys())
    dimension_display_names = get_dimension_display_names()
    
    # Create 2x2 subplot for the four dimensions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, dim in enumerate(dims):
        dim_data = comp_df[comp_df['dimension'] == dim]
        print(f"\nCreating heatmap for {dim}: {len(dim_data)} records")
        
        if not dim_data.empty:
            # Group by comment number combinations and calculate win rates
            heatmap_data = dim_data.groupby(['comment_num_a', 'comment_num_b'])['a_wins'].agg(['mean', 'count']).reset_index()
            heatmap_data = heatmap_data[heatmap_data['count'] >= 3]  # Filter for sufficient data
            
            print(f"  Valid combinations: {len(heatmap_data)}")
            
            if not heatmap_data.empty:
                # Create pivot table
                pivot = heatmap_data.pivot_table(index='comment_num_a', columns='comment_num_b', 
                                               values='mean', fill_value=np.nan)
                
                # Sort by comment numbers
                pivot = pivot.sort_index(axis=0).sort_index(axis=1)
                
                # Create heatmap using seaborn
                sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                           vmin=0, vmax=1, center=0.5, ax=axes[i], 
                           cbar_kws={'label': 'A Win Rate'})
                
                axes[i].set_title(f'{dimension_display_names[dim]} - A Win Rate Heatmap', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Comment Number B')
                axes[i].set_ylabel('Comment Number A')
                
                # No additional text annotations needed
                
                print(f"  Heatmap created with {len(pivot)} x {len(pivot.columns)} cells")
            else:
                axes[i].text(0.5, 0.5, 'No sufficient data', ha='center', va='center', 
                           transform=axes[i].transAxes, fontsize=12)
                axes[i].set_title(f'{dimension_display_names[dim]} - A Win Rate Heatmap', fontsize=14, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(f'{dimension_display_names[dim]} - A Win Rate Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'comment_number_vs_comment_number_winrate_heatmaps.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved heatmaps to: {output_dir / 'comment_number_vs_comment_number_winrate_heatmaps.pdf'}")


def create_individual_heatmaps(comp_df: pd.DataFrame, output_dir: Path) -> None:
    """Create individual heatmaps for each dimension."""
    dims = list(get_dimension_questions().keys())
    dimension_display_names = get_dimension_display_names()
    
    for dim in dims:
        dim_data = comp_df[comp_df['dimension'] == dim]
        print(f"\nCreating individual heatmap for {dim}: {len(dim_data)} records")
        
        if not dim_data.empty:
            # Group by comment number combinations and calculate win rates
            heatmap_data = dim_data.groupby(['comment_num_a', 'comment_num_b'])['a_wins'].agg(['mean', 'count']).reset_index()
            heatmap_data = heatmap_data[heatmap_data['count'] >= 3]  # Filter for sufficient data
            
            if not heatmap_data.empty:
                # Create pivot table
                pivot = heatmap_data.pivot_table(index='comment_num_a', columns='comment_num_b', 
                                               values='mean', fill_value=np.nan)
                
                # Sort by comment numbers
                pivot = pivot.sort_index(axis=0).sort_index(axis=1)
                
                # Create individual heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                
                sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                           vmin=0, vmax=1, center=0.5, ax=ax, 
                           cbar_kws={'label': 'A Win Rate'})
                
                ax.set_title(f'{dimension_display_names[dim]} - A Win Rate Heatmap', fontsize=16, fontweight='bold')
                ax.set_xlabel('Comment Number B', fontsize=14)
                ax.set_ylabel('Comment Number A', fontsize=14)
                
                # No additional text annotations needed
                
                plt.tight_layout()
                filename = f'comment_number_winrate_heatmap_{dim}.pdf'
                fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"  Saved individual heatmap: {filename}")
            else:
                print(f"  No sufficient data for {dim}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Processing comparison data...")
    comp_df = process_comparison_data()
    
    if comp_df.empty:
        print("[ERROR] No data found!")
        return
    
    print(f"\nData summary:")
    print(f"  Total records: {len(comp_df)}")
    print(f"  Dimensions: {sorted(comp_df['dimension'].unique())}")
    print(f"  Comment number A range: {comp_df['comment_num_a'].min()} - {comp_df['comment_num_a'].max()}")
    print(f"  Comment number B range: {comp_df['comment_num_b'].min()} - {comp_df['comment_num_b'].max()}")
    
    # Save data
    comp_df.to_csv(OUTPUT_DIR / 'comment_number_comparison_data.csv', index=False)
    print(f"Saved data to: {OUTPUT_DIR / 'comment_number_comparison_data.csv'}")
    
    # Create combined heatmaps
    print("\nCreating combined heatmaps...")
    create_winrate_heatmaps(comp_df, OUTPUT_DIR)
    
    # Create individual heatmaps
    print("\nCreating individual heatmaps...")
    create_individual_heatmaps(comp_df, OUTPUT_DIR)
    
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()