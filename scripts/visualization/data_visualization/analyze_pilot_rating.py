#!/usr/bin/env python3
"""
专门分析pilot_rating标注数据的脚本

这个脚本专门分析pilot_rating任务中的评分分布情况，包括：
1. 各个问题的得分分布
2. 按topic、model、comment_num等维度的分析
3. 用户评分模式分析

Usage: python analyze_pilot_rating.py
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


# Abbreviation maps
TOPIC_ABBR_MAP = {
    'Binary-Health-Care-Policy': 'HealthCare',
    'Binary-Online-Identity-Policies': 'OnlineID',
    'Binary-Refugee-Policies': 'Refugee',
    'Binary-Tariff-Policy': 'Tariff',
    'Binary-Vaccination-Policy': 'Vaccination',
    'Openqa-AI-changes-human-life': 'AI-Life',
    'Openqa-Tipping-System': 'Tipping',
    'Openqa-Trump-cutting-funding': 'TrumpFunding',
    'Openqa-Updates-of-electronic-products': 'ElecProducts',
    'Openqa-Influencers-as-a-job': 'Influencers',
}

QUESTION_ABBR_MAP = {
    "To what extent is your perspective represented in this response?": 'Representative',
    "How informative is this summary?": 'Informative',
    "Do you think this summary presents a neutral and balanced view of the issue?": 'Balanced',
    "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?": 'PolicyUse',
}

def abbreviate_topic(topic: str) -> str:
    return TOPIC_ABBR_MAP.get(topic, topic[:12] + ('…' if len(topic) > 12 else ''))

def abbreviate_question(question: str) -> str:
    return QUESTION_ABBR_MAP.get(question, f"Q:{question[:10]}…" if len(question) > 10 else question)

def write_label_legend(output_dir: str, topics: list, questions: list) -> None:
    legend_path = Path(output_dir) / 'pilot_rating_labels_legend.txt'
    with open(legend_path, 'w', encoding='utf-8') as f:
        f.write('Topic Abbreviations:\n')
        for t in sorted(set(topics)):
            f.write(f"  {abbreviate_topic(t)} => {t}\n")
        f.write('\nQuestion Abbreviations:\n')
        for q in questions:
            f.write(f"  {abbreviate_question(q)} => {q}\n")


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data


def load_csv_file(file_path: str) -> pd.DataFrame:
    """Load and parse a CSV file."""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def load_raw_data(raw_data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load raw data files and create ID mapping."""
    raw_data = {}
    
    # Load simple and detail versions
    simple_file = f"{raw_data_dir}/summaries_V0903_for_humanstudy_simple.csv"
    detail_file = f"{raw_data_dir}/summaries_V0903_for_humanstudy_detail.csv"
    
    if os.path.exists(simple_file):
        raw_data['simple'] = load_csv_file(simple_file)
        print(f"Loaded simple data: {len(raw_data['simple'])} records")
    
    if os.path.exists(detail_file):
        raw_data['detail'] = load_csv_file(detail_file)
        print(f"Loaded detail data: {len(raw_data['detail'])} records")
    
    return raw_data


def map_annotation_to_raw_data(annotations: List[Dict[str, Any]], raw_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Map annotation IDs to raw data and add metadata."""
    enriched_annotations = []
    
    # Use detail data if available, otherwise simple data
    data_source = raw_data.get('detail', raw_data.get('simple', pd.DataFrame()))
    
    if data_source.empty:
        print("Warning: No raw data available for mapping")
        return annotations
    
    # Create ID mapping
    id_to_metadata = {}
    for _, row in data_source.iterrows():
        base_id = row['id']
        # Map both summary and question IDs
        id_to_metadata[f"{base_id}_summary"] = {
            'topic': row.get('topic', ''),
            'model': row.get('model', ''),
            'comment_num': row.get('comment_num', 0),
            'dataset_name': row.get('dataset_name', ''),
            'question': row.get('question', ''),
            'summary': row.get('summary', ''),
            'comments': row.get('comments', ''),
            'num_samples_group': row.get('num_samples_group', 0),
            'sample_id': row.get('sample_id', 0)
        }
        id_to_metadata[f"{base_id}_question"] = {
            'topic': row.get('topic', ''),
            'model': row.get('model', ''),
            'comment_num': row.get('comment_num', 0),
            'dataset_name': row.get('dataset_name', ''),
            'question': row.get('question', ''),
            'summary': row.get('summary', ''),
            'comments': row.get('comments', ''),
            'num_samples_group': row.get('num_samples_group', 0),
            'sample_id': row.get('sample_id', 0)
        }
    
    # Enrich annotations with raw data metadata
    for ann in annotations:
        enriched_ann = ann.copy()
        ann_id = ann.get('id', '')
        
        if ann_id in id_to_metadata:
            enriched_ann['raw_metadata'] = id_to_metadata[ann_id]
        else:
            # Try to find partial match
            base_id = ann_id.replace('_summary', '').replace('_question', '')
            if base_id in id_to_metadata:
                enriched_ann['raw_metadata'] = id_to_metadata[base_id]
            else:
                enriched_ann['raw_metadata'] = {}
        
        enriched_annotations.append(enriched_ann)
    
    return enriched_annotations


def extract_pilot_rating_data(annotation_dir: str) -> List[Dict[str, Any]]:
    """Extract all pilot_rating annotation data."""
    all_annotations = []
    annotation_path = Path(annotation_dir)
    
    if not annotation_path.exists():
        print(f"Annotation directory not found: {annotation_dir}")
        return all_annotations
    
    # Process user directories
    user_dirs = [d for d in annotation_path.iterdir() if d.is_dir() and d.name != 'archived_users']
    
    for user_dir in user_dirs:
        user_id = user_dir.name
        
        # Load annotated instances
        annotated_file = user_dir / 'annotated_instances.jsonl'
        if annotated_file.exists():
            annotations = load_jsonl_file(str(annotated_file))
            for ann in annotations:
                ann['user_id'] = user_id
                ann['task_type'] = 'rating'
                all_annotations.append(ann)
    
    return all_annotations


def analyze_rating_scores(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze rating scores from annotations."""
    analysis = {
        'rating_distributions': {},
        'detailed_ratings': [],
        'user_statistics': defaultdict(list),
        'topic_statistics': defaultdict(list),
        'model_statistics': defaultdict(list),
        'comment_num_statistics': defaultdict(list)
    }
    
    # Define rating questions
    rating_questions = [
        "To what extent is your perspective represented in this response?",
        "How informative is this summary?",
        "Do you think this summary presents a neutral and balanced view of the issue?",
        "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?"
    ]
    
    # Initialize rating distributions
    for question in rating_questions:
        analysis['rating_distributions'][question] = []
    
    for ann in annotations:
        if 'label_annotations' in ann:
            user_id = ann.get('user_id', 'unknown')
            ann_id = ann.get('id', 'unknown')
            
            # Extract raw metadata
            raw_metadata = ann.get('raw_metadata', {})
            topic = raw_metadata.get('topic', 'Unknown')
            model = raw_metadata.get('model', 'Unknown')
            comment_num = raw_metadata.get('comment_num', 0)
            
            # Extract ratings for each question
            rating_record = {
                'annotation_id': ann_id,
                'user_id': user_id,
                'topic': topic,
                'model': model,
                'comment_num': comment_num,
                'ratings': {}
            }
            
            for question in rating_questions:
                if question in ann['label_annotations']:
                    rating_data = ann['label_annotations'][question]
                    # Look for scale_1, scale_2, scale_3, etc.
                    rating = None
                    for key, value in rating_data.items():
                        if key.startswith('scale_'):
                            rating = int(value)
                            break
                    
                    if rating is not None:
                        analysis['rating_distributions'][question].append(rating)
                        rating_record['ratings'][question] = rating
                        
                        # Add to dimension statistics
                        analysis['user_statistics'][user_id].append(rating)
                        analysis['topic_statistics'][topic].append(rating)
                        analysis['model_statistics'][model].append(rating)
                        analysis['comment_num_statistics'][comment_num].append(rating)
            
            analysis['detailed_ratings'].append(rating_record)
    
    return analysis


def create_rating_distribution_plots(analysis: Dict[str, Any], output_dir: str):
    """Create plots for rating distributions."""
    rating_questions = [
        "To what extent is your perspective represented in this response?",
        "How informative is this summary?",
        "Do you think this summary presents a neutral and balanced view of the issue?",
        "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?"
    ]
    
    # Create main distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Compute a unified y-axis max across all questions and ensure x-axis is 1-5
    global_max_count = 0
    for question in rating_questions:
        ratings = analysis['rating_distributions'].get(question, [])
        if ratings:
            # Count occurrences for ratings 1..5 (include zeros to unify bars)
            counts = Counter(ratings)
            heights = [counts.get(v, 0) for v in range(1, 6)]
            global_max_count = max(global_max_count, max(heights))

    for i, question in enumerate(rating_questions):
        if question in analysis['rating_distributions']:
            ratings = analysis['rating_distributions'][question]
            if ratings:
                # Create histogram
                ax = axes[i]
                counts = Counter(ratings)
                # Enforce bars for 1..5 to keep x-axis consistent
                values = list(range(1, 6))
                heights = [counts.get(v, 0) for v in values]
                
                bars = ax.bar(values, heights, alpha=0.7, color=sns.color_palette("husl", len(values)))
                ax.set_title(f'{question}\n(n={len(ratings)})', fontsize=10, wrap=True)
                ax.set_xlabel('Rating (1-5)')
                ax.set_ylabel('Count')
                # Unified x-axis 1..5 for all subplots
                ax.set_xlim(0.5, 5.5)
                ax.set_xticks(range(1, 6))
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, height in zip(bars, heights):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height}', ha='center', va='bottom')
                
                # Add statistics
                mean_rating = np.mean(ratings)
                std_rating = np.std(ratings)
                ax.text(0.02, 0.98, f'Mean: {mean_rating:.2f}\nStd: {std_rating:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                # Unified y-axis across subplots
                ymax = max(global_max_count, max(heights) if heights else 0)
                ax.set_ylim(0, ymax + max(1, int(0.1 * ymax)))
                # Set integer y-ticks for readability
                ax.set_yticks(range(0, ymax + 1))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pilot_rating_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_dimension_analysis_plots(analysis: Dict[str, Any], output_dir: str):
    """Create plots analyzing ratings by different dimensions."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Average ratings by topic
    if analysis['topic_statistics']:
        topics = list(analysis['topic_statistics'].keys())
        avg_ratings = [np.mean(analysis['topic_statistics'][topic]) for topic in topics]
        
        bars = axes[0, 0].bar(range(len(topics)), avg_ratings, alpha=0.7, color=sns.color_palette("husl", len(topics)))
        axes[0, 0].set_title('Average Ratings by Topic')
        axes[0, 0].set_xlabel('Topic')
        axes[0, 0].set_ylabel('Average Rating')
        axes[0, 0].set_xticks(range(len(topics)))
        axes[0, 0].set_xticklabels([abbreviate_topic(t) for t in topics], rotation=45, ha='right')
        axes[0, 0].set_ylim(1, 5)
        
        for i, (bar, avg) in enumerate(zip(bars, avg_ratings)):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., avg + 0.05,
                           f'{avg:.2f}', ha='center', va='bottom')
    
    # Plot 2: Average ratings by model
    if analysis['model_statistics']:
        models = list(analysis['model_statistics'].keys())
        avg_ratings = [np.mean(analysis['model_statistics'][model]) for model in models]
        
        bars = axes[0, 1].bar(range(len(models)), avg_ratings, alpha=0.7, color=sns.color_palette("husl", len(models)))
        axes[0, 1].set_title('Average Ratings by Model')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Average Rating')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels([m.replace('-', '\n') for m in models], rotation=45, ha='right')
        axes[0, 1].set_ylim(1, 5)
        
        for i, (bar, avg) in enumerate(zip(bars, avg_ratings)):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., avg + 0.05,
                           f'{avg:.2f}', ha='center', va='bottom')
    
    # Plot 3: Average ratings by comment number
    if analysis['comment_num_statistics']:
        comment_nums = sorted(analysis['comment_num_statistics'].keys())
        avg_ratings = [np.mean(analysis['comment_num_statistics'][cn]) for cn in comment_nums]
        
        bars = axes[0, 2].bar(comment_nums, avg_ratings, alpha=0.7, color=sns.color_palette("husl", len(comment_nums)))
        axes[0, 2].set_title('Average Ratings by Comment Number')
        axes[0, 2].set_xlabel('Comment Number')
        axes[0, 2].set_ylabel('Average Rating')
        axes[0, 2].set_ylim(1, 5)
        
        for i, (bar, avg) in enumerate(zip(bars, avg_ratings)):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., avg + 0.05,
                           f'{avg:.2f}', ha='center', va='bottom')
    
    # Plot 4: Rating distribution heatmap by topic
    if analysis['detailed_ratings']:
        # Create a DataFrame for easier analysis
        df_data = []
        for record in analysis['detailed_ratings']:
            for question, rating in record['ratings'].items():
                df_data.append({
                    'topic': record['topic'],
                    'question': question,
                    'rating': rating
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # Create pivot table for heatmap
            pivot_data = df.groupby(['topic', 'question'])['rating'].mean().unstack(fill_value=0)
            
            if not pivot_data.empty:
                im = axes[1, 0].imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
                axes[1, 0].set_title('Average Ratings: Topic vs Question')
                axes[1, 0].set_xlabel('Question')
                axes[1, 0].set_ylabel('Topic')
                axes[1, 0].set_xticks(range(len(pivot_data.columns)))
                axes[1, 0].set_yticks(range(len(pivot_data.index)))
                axes[1, 0].set_xticklabels([abbreviate_question(q) for q in pivot_data.columns], rotation=45, ha='right')
                axes[1, 0].set_yticklabels([abbreviate_topic(t) for t in pivot_data.index])
                
                # Add text annotations
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        text = axes[1, 0].text(j, i, f'{pivot_data.iloc[i, j]:.2f}', 
                                             ha="center", va="center", color="black")
                
                plt.colorbar(im, ax=axes[1, 0])

                # Write legend file
                write_label_legend(output_dir, list(pivot_data.index), list(pivot_data.columns))
    
    # Plot 5: Inter-annotator distance by Topic x Comment Number (average across 4 questions)
    # We compute, for each base item (topic, comment_num, base_id), the mean pairwise
    # absolute difference across the four questions between annotators, then average per (topic, comment_num)
    try:
        # Prepare structures from detailed_ratings
        # Build per-item ratings: key -> (topic, comment_num, base_id)
        per_item: Dict[tuple, Dict[str, Dict[str, int]]] = defaultdict(dict)
        # also track questions order
        for rec in analysis.get('detailed_ratings', []):
            ann_id = rec.get('annotation_id', '')
            base_id = ann_id.replace('_summary', '').replace('_question', '')
            topic = rec.get('topic', 'Unknown')
            comment_num = rec.get('comment_num', 0)
            user_id = rec.get('user_id', 'unknown')
            ratings = rec.get('ratings', {})  # question -> int
            key = (topic, comment_num, base_id)
            per_item[key][user_id] = ratings

        # Compute per (topic, comment_num) distances, while collecting missing-data reasons
        tc_to_distances: Dict[tuple, List[float]] = defaultdict(list)
        tc_total_items: Dict[tuple, int] = defaultdict(int)
        tc_used_items: Dict[tuple, int] = defaultdict(int)
        tc_skipped_insufficient_annot: Dict[tuple, int] = defaultdict(int)
        tc_skipped_no_overlap: Dict[tuple, int] = defaultdict(int)

        rating_questions = [
            "To what extent is your perspective represented in this response?",
            "How informative is this summary?",
            "Do you think this summary presents a neutral and balanced view of the issue?",
            "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?"
        ]

        for (topic, comment_num, base_id), user_map in per_item.items():
            users = list(user_map.keys())
            tc_key = (topic, comment_num)
            tc_total_items[tc_key] += 1
            if len(users) < 2:
                tc_skipped_insufficient_annot[tc_key] += 1
                continue
            pair_diffs: List[float] = []
            # pairwise over annotators
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    ri = user_map[users[i]]
                    rj = user_map[users[j]]
                    # compute diffs over the 4 rating questions (skip missing)
                    diffs_q: List[float] = []
                    for q in rating_questions:
                        if q in ri and q in rj:
                            try:
                                diffs_q.append(abs(int(ri[q]) - int(rj[q])))
                            except Exception:
                                pass
                    if not diffs_q:
                        continue
                    pair_diffs.append(float(np.mean(diffs_q)))
            if pair_diffs:
                tc_to_distances[(topic, comment_num)].append(float(np.mean(pair_diffs)))
                tc_used_items[tc_key] += 1
            else:
                # had >=2 annotators but no overlapping answered questions across pairs
                tc_skipped_no_overlap[tc_key] += 1

        # Aggregate to matrix: include all known topics and comment_nums, not only those with distances
        # Topics: from overall analysis keys (ensures up to 10 topics appear even if some have no distances)
        topics = sorted(list(analysis.get('topic_statistics', {}).keys()))
        # Comment numbers: from overall stats; fallback to observed in distances
        comment_nums = sorted(list(analysis.get('comment_num_statistics', {}).keys()))
        if not comment_nums:
            comment_nums = sorted({cn for (_t, cn) in [(k[0], k[1]) for k in tc_to_distances.keys()]})
        if topics and comment_nums:
            heatmap = np.zeros((len(topics), len(comment_nums)))
            heatmap[:] = np.nan
            topic_index = {t: i for i, t in enumerate(topics)}
            cn_index = {cn: j for j, cn in enumerate(comment_nums)}
            for (t, cn), vals in tc_to_distances.items():
                i = topic_index[t]
                j = cn_index[cn]
                heatmap[i, j] = float(np.mean(vals))

            # Replace NaNs with zeros for display (optional); keep colormap readable
            masked = np.ma.masked_invalid(heatmap)
            im = axes[1, 1].imshow(masked, cmap='Purples', aspect='auto', vmin=0, vmax=4)
            axes[1, 1].set_title('Inter-annotator Distance (avg over questions)')
            axes[1, 1].set_xlabel('Comment Number')
            axes[1, 1].set_ylabel('Topic')
            axes[1, 1].set_xticks(range(len(comment_nums)))
            axes[1, 1].set_yticks(range(len(topics)))
            axes[1, 1].set_xticklabels(comment_nums)
            axes[1, 1].set_yticklabels([abbreviate_topic(t) for t in topics])
            # annotate
            for i in range(len(topics)):
                for j in range(len(comment_nums)):
                    val = heatmap[i, j]
                    if not np.isnan(val):
                        axes[1, 1].text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
            plt.colorbar(im, ax=axes[1, 1])

            # Write missing-data report
            report_path = Path(output_dir) / 'pilot_rating_missing_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('Missing Data Report for Inter-annotator Distance (by Topic x Comment Number)\n')
                f.write('='*70 + '\n')
                f.write('Counts are per setting; reasons are mutually non-exclusive at item level.\n\n')
                f.write('Columns: topic_abbr, comment_num, total_items, used_items, skipped_insufficient_annot, skipped_no_overlap\n')
                for t in topics:
                    for cn in comment_nums:
                        key = (t, cn)
                        total = tc_total_items.get(key, 0)
                        used = tc_used_items.get(key, 0)
                        s_ins = tc_skipped_insufficient_annot.get(key, 0)
                        s_no = tc_skipped_no_overlap.get(key, 0)
                        f.write(f"{abbreviate_topic(t)}, {cn}, {total}, {used}, {s_ins}, {s_no}\n")
        else:
            axes[1, 1].axis('off')
    except Exception as e:
        # Fallback: hide plot if anything unexpected occurs
        axes[1, 1].axis('off')
    
    # Plot 6: Rating correlation matrix
    if analysis['detailed_ratings']:
        # Create correlation matrix for different questions
        rating_questions = [
            "To what extent is your perspective represented in this response?",
            "How informative is this summary?",
            "Do you think this summary presents a neutral and balanced view of the issue?",
            "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?"
        ]
        
        # Create DataFrame with all ratings
        rating_data = []
        for record in analysis['detailed_ratings']:
            row = {'annotation_id': record['annotation_id']}
            for question in rating_questions:
                if question in record['ratings']:
                    row[question] = record['ratings'][question]
            rating_data.append(row)
        
        if rating_data:
            df_ratings = pd.DataFrame(rating_data)
            numeric_cols = [col for col in df_ratings.columns if col != 'annotation_id']
            
            if len(numeric_cols) > 1:
                corr_matrix = df_ratings[numeric_cols].corr()
                
                im = axes[1, 2].imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axes[1, 2].set_title('Rating Correlation Matrix')
                axes[1, 2].set_xlabel('Question')
                axes[1, 2].set_ylabel('Question')
                axes[1, 2].set_xticks(range(len(corr_matrix.columns)))
                axes[1, 2].set_yticks(range(len(corr_matrix.index)))
                axes[1, 2].set_xticklabels([abbreviate_question(col) for col in corr_matrix.columns], rotation=45, ha='right')
                axes[1, 2].set_yticklabels([abbreviate_question(idx) for idx in corr_matrix.index])
                
                # Add text annotations
                for i in range(len(corr_matrix.index)):
                    for j in range(len(corr_matrix.columns)):
                        text = axes[1, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                             ha="center", va="center", color="black")
                
                plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pilot_rating_dimension_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_rating_summary_report(analysis: Dict[str, Any], output_dir: str):
    """Generate a detailed summary report of rating analysis."""
    report_path = f'{output_dir}/pilot_rating_summary_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PILOT RATING ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        total_annotations = len(analysis['detailed_ratings'])
        f.write(f"Total annotations analyzed: {total_annotations}\n")
        f.write(f"Total users: {len(analysis['user_statistics'])}\n")
        f.write(f"Total topics: {len(analysis['topic_statistics'])}\n")
        f.write(f"Total models: {len(analysis['model_statistics'])}\n")
        f.write(f"Comment number range: {min(analysis['comment_num_statistics'].keys()) if analysis['comment_num_statistics'] else 0} - {max(analysis['comment_num_statistics'].keys()) if analysis['comment_num_statistics'] else 0}\n\n")
        
        # Rating distributions for each question
        rating_questions = [
            "To what extent is your perspective represented in this response?",
            "How informative is this summary?",
            "Do you think this summary presents a neutral and balanced view of the issue?",
            "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?"
        ]
        
        f.write("RATING DISTRIBUTIONS BY QUESTION\n")
        f.write("-" * 40 + "\n")
        
        for question in rating_questions:
            if question in analysis['rating_distributions']:
                ratings = analysis['rating_distributions'][question]
                if ratings:
                    f.write(f"\n{question}:\n")
                    f.write(f"  Total responses: {len(ratings)}\n")
                    f.write(f"  Mean: {np.mean(ratings):.3f}\n")
                    f.write(f"  Median: {np.median(ratings):.3f}\n")
                    f.write(f"  Std Dev: {np.std(ratings):.3f}\n")
                    f.write(f"  Min: {min(ratings)}\n")
                    f.write(f"  Max: {max(ratings)}\n")
                    
                    # Distribution
                    counts = Counter(ratings)
                    f.write(f"  Distribution:\n")
                    for rating in sorted(counts.keys()):
                        percentage = (counts[rating] / len(ratings)) * 100
                        f.write(f"    {rating}: {counts[rating]} ({percentage:.1f}%)\n")
        
        # Topic analysis
        f.write(f"\n\nTOPIC ANALYSIS\n")
        f.write("-" * 20 + "\n")
        for topic, ratings in analysis['topic_statistics'].items():
            if ratings:
                f.write(f"{topic}:\n")
                f.write(f"  Count: {len(ratings)}\n")
                f.write(f"  Mean: {np.mean(ratings):.3f}\n")
                f.write(f"  Std Dev: {np.std(ratings):.3f}\n")
        
        # Model analysis
        f.write(f"\n\nMODEL ANALYSIS\n")
        f.write("-" * 20 + "\n")
        for model, ratings in analysis['model_statistics'].items():
            if ratings:
                f.write(f"{model}:\n")
                f.write(f"  Count: {len(ratings)}\n")
                f.write(f"  Mean: {np.mean(ratings):.3f}\n")
                f.write(f"  Std Dev: {np.std(ratings):.3f}\n")
        
        # Comment number analysis
        f.write(f"\n\nCOMMENT NUMBER ANALYSIS\n")
        f.write("-" * 30 + "\n")
        for comment_num, ratings in sorted(analysis['comment_num_statistics'].items()):
            if ratings:
                f.write(f"Comment Number {comment_num}:\n")
                f.write(f"  Count: {len(ratings)}\n")
                f.write(f"  Mean: {np.mean(ratings):.3f}\n")
                f.write(f"  Std Dev: {np.std(ratings):.3f}\n")
        
        # User analysis
        f.write(f"\n\nUSER ANALYSIS\n")
        f.write("-" * 15 + "\n")
        for user_id, ratings in analysis['user_statistics'].items():
            if ratings:
                f.write(f"User {user_id}:\n")
                f.write(f"  Count: {len(ratings)}\n")
                f.write(f"  Mean: {np.mean(ratings):.3f}\n")
                f.write(f"  Std Dev: {np.std(ratings):.3f}\n")
                f.write(f"  Range: {min(ratings)} - {max(ratings)}\n")
    
    print(f"Rating summary report saved to: {report_path}")


def main():
    # Configuration
    annotation_dir = "/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/pilot_rating"
    raw_data_dir = "/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/data_files/raw"
    output_dir = "/ibex/project/c2328/LLMs-Scalable-Deliberation/results/pilot_rating_analysis"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PILOT RATING ANALYSIS")
    print("=" * 60)
    print(f"Annotation directory: {annotation_dir}")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load raw data for ID mapping
    print("\n📁 Loading raw data files...")
    raw_data = load_raw_data(raw_data_dir)
    
    # Extract pilot rating data
    print("\n📊 Extracting pilot rating annotations...")
    annotations = extract_pilot_rating_data(annotation_dir)
    print(f"Found {len(annotations)} annotations")
    
    if not annotations:
        print("No annotations found!")
        return
    
    # Map annotations to raw data
    if raw_data:
        print("Mapping annotations to raw data...")
        enriched_annotations = map_annotation_to_raw_data(annotations, raw_data)
    else:
        enriched_annotations = annotations
    
    # Analyze rating scores
    print("Analyzing rating scores...")
    analysis = analyze_rating_scores(enriched_annotations)
    
    # Create visualizations
    print("Creating visualizations...")
    create_rating_distribution_plots(analysis, str(output_path))
    create_dimension_analysis_plots(analysis, str(output_path))
    
    # Generate summary report
    print("Generating summary report...")
    generate_rating_summary_report(analysis, str(output_path))
    
    print(f"\n✅ Pilot rating analysis completed successfully!")
    print(f"📁 Output files saved to: {output_dir}")
    print(f"📊 Generated files:")
    print(f"  - pilot_rating_distributions.png")
    print(f"  - pilot_rating_dimension_analysis.png")
    print(f"  - pilot_rating_summary_report.txt")


if __name__ == "__main__":
    main()
