#!/usr/bin/env python3
"""
Process raw summary data for human study annotation.
Converts raw CSV data into three formats: rating, pairwise comparison, and triplet formats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import markdown
from markdown.extensions import codehilite
import re
from itertools import combinations
import argparse


def load_and_shuffle_data(input_path):
    """Load and shuffle the raw data."""
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows with models: {df.model.unique()}")
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    print("Data shuffled")
    
    return df


def process_rating_data(df):
    """Process dataframe into rating format (question + summary pairs)."""
    print("Processing data for rating format...")
    
    processed_data = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        raw_id = row['id']
        question = row['question']
        summary = row['summary']
        
        # Add question entry
        question_entry = {
            "id": f"{raw_id}_question",
            "raw_id": raw_id,
            "question": question,
            "text": '[Question]' + question.replace("\n", "<br>").replace(" Please answer briefly in 2–3 sentences.", "").replace("Please answer briefly in 1–2 sentences.", ""),
            "model": row['model'],
            "summary_length": row.get('summary_length', None)
        }
        processed_data.append(question_entry)
        
        # Add summary entry (convert markdown to HTML)
        summary_html = markdown.markdown(summary, extensions=['extra', 'codehilite'])
        summary_entry = {
            "id": f"{raw_id}_summary",
            "raw_id": raw_id,
            "question": question,
            "text": "<h4>Below is a summary of people's opinions on the issue.</h4><hr>" + summary_html,
            "model": row['model'],
            "summary_length": row.get('summary_length', None)
        }
        processed_data.append(summary_entry)
    
    processed_df = pd.DataFrame(processed_data)
    print(f"Created {len(processed_df)} rating entries ({len(processed_df)//2} pairs)")
    
    return processed_df


def balanced_pair_priority(pair_indices, summary_counts, min_comparisons):
    """Priority function for balanced pair creation."""
    i, j = pair_indices
    count_i, count_j = summary_counts[i], summary_counts[j]
    
    both_below_min = (count_i < min_comparisons) and (count_j < min_comparisons)
    one_below_min = (count_i < min_comparisons) or (count_j < min_comparisons)
    
    max_count = max(count_i, count_j)
    min_count = min(count_i, count_j)
    sum_count = count_i + count_j
    
    if both_below_min:
        return (0, sum_count, max_count - min_count)  # Highest priority
    elif one_below_min:
        return (1, sum_count, max_count - min_count)  # Medium priority
    else:
        return (2, sum_count, max_count - min_count)  # Lowest priority


def create_balanced_pairs(summaries, min_comparisons_per_summary=6):
    """Create balanced pairwise comparisons for a group of summaries."""
    pairs_created = []
    summary_counts = {i: 0 for i in range(len(summaries))}
    
    all_pairs = list(combinations(range(len(summaries)), 2))
    max_comparisons = min_comparisons_per_summary + 3
    
    while any(count < min_comparisons_per_summary for count in summary_counts.values()) and all_pairs:
        # Filter valid pairs
        valid_pairs = [
            pair for pair in all_pairs 
            if summary_counts[pair[0]] < max_comparisons 
            and summary_counts[pair[1]] < max_comparisons
        ]
        
        if not valid_pairs:
            valid_pairs = [
                pair for pair in all_pairs
                if (summary_counts[pair[0]] < min_comparisons_per_summary 
                    or summary_counts[pair[1]] < min_comparisons_per_summary)
            ]
        
        if not valid_pairs:
            break
            
        # Sort by priority and select best pair
        valid_pairs.sort(key=lambda p: balanced_pair_priority(p, summary_counts, min_comparisons_per_summary))
        pair_indices = valid_pairs[0]
        all_pairs.remove(pair_indices)
        i, j = pair_indices
        
        pairs_created.append(pair_indices)
        summary_counts[i] += 1
        summary_counts[j] += 1
    
    return pairs_created, summary_counts


def reduce_html_headers(html_text):
    """Reduce HTML header levels (h1->h4, h2->h5, h3->h6)."""
    html_text = re.sub(r'<h1>', '<h4>', html_text)
    html_text = re.sub(r'</h1>', '</h4>', html_text)
    html_text = re.sub(r'<h2>', '<h5>', html_text)
    html_text = re.sub(r'</h2>', '</h5>', html_text)
    html_text = re.sub(r'<h3>', '<h6>', html_text)
    html_text = re.sub(r'</h3>', '</h6>', html_text)
    return html_text


def process_pair_data(df, min_comparisons_per_summary=6):
    """Process dataframe into pairwise comparison format."""
    print("Processing data for pairwise comparison format...")
    
    processed_pair_data = []
    question_sample_groups = df.groupby(['question', 'num_samples_group'])
    
    print(f"Total groups to process: {len(question_sample_groups)}")
    
    for (question, num_samples_group), group in question_sample_groups:
        print(f"Processing group: question='{question[:50]}...', num_samples_group={num_samples_group}, size={len(group)}")
        
        if len(group) != 15:
            print(f"  WARNING: Expected 15 summaries, got {len(group)}")
            print(f"  Models in group: {group['model'].value_counts().to_dict()}")
        
        summaries = list(group.iterrows())
        pairs_created, summary_counts = create_balanced_pairs(summaries, min_comparisons_per_summary)
        
        # Create comparison entries for each pair
        for pair_idx, (i, j) in enumerate(pairs_created):
            _, row_a = summaries[i]
            _, row_b = summaries[j]
            
            # Add question entry
            question_entry = {
                "id": f"{row_a['id']}_{row_b['id']}_pair_{pair_idx}_question",
                "raw_id": f"{row_a['id']}_{row_b['id']}_pair_{pair_idx}",
                "question": question,
                "text": '<h3>[Question]</h3>' + '<h4>' + question.replace("\n", "<br>").replace(" Please answer briefly in 2–3 sentences.", "").replace("Please answer briefly in 1–2 sentences.", "") + '</h4>',
                "model": "question",
                "num_samples_group": num_samples_group,
                "summary_length": None
            }
            processed_pair_data.append(question_entry)
            
            # Convert summaries to HTML and reduce header levels
            summary_a_html = markdown.markdown(row_a['summary'], extensions=['extra', 'codehilite'])
            summary_b_html = markdown.markdown(row_b['summary'], extensions=['extra', 'codehilite'])
            
            summary_a_html = reduce_html_headers(summary_a_html)
            summary_b_html = reduce_html_headers(summary_b_html)
            
            # Create HTML layout for pairwise comparison
            comparison_html = f"""
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1; border: 1px solid #ccc; padding: 15px; border-radius: 5px;">
                    <h4 style="margin-top: 0; color: #2c5aa0;">Summary A</h4>
                    <div style="max-height: 800px; overflow-y: auto; padding-right: 10px;">
                        {summary_a_html}
                    </div>
                </div>
                <div style="flex: 1; border: 1px solid #ccc; padding: 15px; border-radius: 5px;">
                    <h4 style="margin-top: 0; color: #2c5aa0;">Summary B</h4>
                    <div style="max-height: 800px; overflow-y: auto; padding-right: 10px;">
                        {summary_b_html}
                    </div>
                </div>
            </div>
            """
            
            pair_entry = {
                "id": f"{row_a['id']}_{row_b['id']}_pair_{pair_idx}",
                "raw_id": f"{row_a['id']}_{row_b['id']}",
                "question": question,
                "text": "<h4>Two summaries of opinions are shown below. Read carefully and answer according to your prior opinion. Both are scrollable.</h4><hr>" + comparison_html,
                "model_a": row_a['model'],
                "model_b": row_b['model'],
                "num_samples_group": num_samples_group,
                "summary_a_id": row_a['id'],
                "summary_b_id": row_b['id'],
                "summary_a_text": row_a['summary'],
                "summary_b_text": row_b['summary'],
                "summary_length_a": row_a.get('summary_length', None),
                "summary_length_b": row_b.get('summary_length', None)
            }
            processed_pair_data.append(pair_entry)
        
        print(f"  Created {len(pairs_created)} pairs for this group")
        print(f"  Each summary comparison count: {dict(summary_counts)}")
    
    processed_pair_df = pd.DataFrame(processed_pair_data)
    
    print(f"\nFinal statistics:")
    print(f"Total processed_pair_data entries: {len(processed_pair_data)}")
    print(f"Total question entries: {len([x for x in processed_pair_data if x.get('model') == 'question'])}")
    print(f"Total pair entries: {len([x for x in processed_pair_data if x.get('model_a') is not None])}")
    
    return processed_pair_df


def extract_rating_pairs(rating_df):
    """Extract rating pairs from rating dataframe."""
    rating_pairs = []
    for i in range(0, len(rating_df), 2):
        if i + 1 < len(rating_df):
            question_row = rating_df.iloc[i]
            summary_row = rating_df.iloc[i + 1]
            
            if (question_row['raw_id'] == summary_row['raw_id'] and 
                '_question' in question_row['id'] and 
                '_summary' in summary_row['id']):
                rating_pairs.append({
                    'raw_id': question_row['raw_id'],
                    'question': question_row['question'],
                    'question_text': question_row['text'],
                    'summary_text': summary_row['text'],
                    'model': question_row['model'],
                    'summary_length': question_row.get('summary_length')
                })
    
    return pd.DataFrame(rating_pairs)


def extract_pair_comparisons(pair_df):
    """Extract pair comparisons from pair dataframe."""
    pair_comparisons = []
    for i in range(0, len(pair_df), 2):
        if i + 1 < len(pair_df):
            question_row = pair_df.iloc[i]
            comparison_row = pair_df.iloc[i + 1]
            
            question_base_id = question_row['raw_id'].rsplit('_pair_', 1)[0] if '_pair_' in question_row['raw_id'] else question_row['raw_id']
            comparison_base_id = comparison_row['raw_id']
            
            if (question_base_id == comparison_base_id and
                question_row['model'] == 'question'):
                pair_comparisons.append({
                    'raw_id': comparison_row['raw_id'], 
                    'question': question_row['question'],
                    'question_text': question_row['text'],
                    'comparison_text': comparison_row['text'],
                    'model_a': comparison_row.get('model_a'),
                    'model_b': comparison_row.get('model_b'),
                    'summary_a_id': comparison_row.get('summary_a_id'),
                    'summary_b_id': comparison_row.get('summary_b_id'),
                    'summary_a_text': comparison_row.get('summary_a_text'),
                    'summary_b_text': comparison_row.get('summary_b_text'),
                    'num_samples_group': comparison_row.get('num_samples_group')
                })
    
    return pd.DataFrame(pair_comparisons)


def create_triplet_data(rating_pairs_df, pair_comparisons_df):
    """Create triplet data by joining rating and pair data."""
    print("Creating triplets by matching summary_a_id...")
    print(f"Total pair comparisons: {len(pair_comparisons_df)}")
    
    # Create lookup dictionary for ratings
    ratings_by_id = {}
    for _, rating_pair in rating_pairs_df.iterrows():
        raw_id = rating_pair['raw_id']
        ratings_by_id[raw_id] = rating_pair
    
    print(f"Created lookup for {len(ratings_by_id)} rating summaries")
    
    # Match pairs with ratings
    joined_data = []
    matched_count = 0
    unmatched_count = 0
    rating_usage_count = {}
    
    for idx, comparison_pair in pair_comparisons_df.iterrows():
        summary_a_id = comparison_pair['summary_a_id']
        pair_raw_id = comparison_pair['raw_id']
        question = comparison_pair['question']
        
        if summary_a_id in ratings_by_id:
            matched_count += 1
            rating_pair = ratings_by_id[summary_a_id]
            
            rating_usage_count[summary_a_id] = rating_usage_count.get(summary_a_id, 0) + 1
            
            clean_question = question.replace(" Please answer briefly in 2–3 sentences.", "").replace("Please answer briefly in 1–2 sentences.", "")
            triplet_id = f"triplet_{idx}"
            
            # Question row
            question_row = {
                'id': f"{triplet_id}_question",
                'raw_id': pair_raw_id,
                'question': question,
                'text': f'<h3>[Question]</h3><h4>{clean_question}</h4>',
                'type': 'question',
                'model': 'question',
                'num_samples_group': comparison_pair.get('num_samples_group'),
                'summary_length': None,
                'model_a': None,
                'model_b': None,
                'summary_a_id': None,
                'summary_b_id': None,
                'summary_a_text': None,
                'summary_b_text': None,
                'summary_length_a': None,
                'summary_length_b': None
            }
            joined_data.append(question_row)
            
            # Rating row
            rating_row = {
                'id': f"{triplet_id}_rating",
                'raw_id': pair_raw_id,
                'question': question,
                'text': rating_pair['summary_text'],
                'type': 'rating',
                'model': rating_pair['model'],
                'num_samples_group': comparison_pair.get('num_samples_group'),
                'summary_length': rating_pair.get('summary_length'),
                'model_a': None,
                'model_b': None,
                'summary_a_id': None,
                'summary_b_id': None,
                'summary_a_text': None,
                'summary_b_text': None,
                'summary_length_a': None,
                'summary_length_b': None
            }
            joined_data.append(rating_row)
            
            # Comparison row
            comparison_row = {
                'id': f"{triplet_id}_comparison",
                'raw_id': pair_raw_id,
                'question': question,
                'text': comparison_pair['comparison_text'],
                'type': 'comparison',
                'model': 'comparison',
                'num_samples_group': comparison_pair.get('num_samples_group'),
                'summary_length': None,
                'model_a': comparison_pair.get('model_a'),
                'model_b': comparison_pair.get('model_b'),
                'summary_a_id': comparison_pair.get('summary_a_id'),
                'summary_b_id': comparison_pair.get('summary_b_id'),
                'summary_a_text': comparison_pair.get('summary_a_text'),
                'summary_b_text': comparison_pair.get('summary_b_text'),
                'summary_length_a': comparison_pair.get('summary_length_a'),
                'summary_length_b': comparison_pair.get('summary_length_b')
            }
            joined_data.append(comparison_row)
        else:
            unmatched_count += 1
    
    print(f"Matched {matched_count} pair comparisons with ratings")
    print(f"Unmatched {unmatched_count} pair comparisons")
    print(f"Total triplets created: {len(joined_data) // 3}")
    
    if rating_usage_count:
        usage_values = list(rating_usage_count.values())
        print(f"Rating usage stats: min={min(usage_values)}, max={max(usage_values)}, avg={sum(usage_values)/len(usage_values):.1f}")
    
    return pd.DataFrame(joined_data)


def analyze_data_structure(df):
    """Analyze and print data structure information."""
    print("Data structure analysis:")
    print(f"Total rows: {len(df)}")
    print(f"Unique questions: {df['question'].nunique()}")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Unique num_samples_group: {df['num_samples_group'].nunique()}")
    
    group_sizes = df.groupby(['question', 'num_samples_group']).size()
    print(f"\nGroup sizes (question, num_samples_group):")
    print(group_sizes.describe())


def main():
    parser = argparse.ArgumentParser(description='Process raw summary data for human study annotation')
    parser.add_argument('--input', '-i', 
                       default='../data_files/raw/summaries_V0903_for_humanstudy_detail.csv',
                       help='Input CSV file path')
    parser.add_argument('--output-dir', '-o',
                       default='../data_files/processed/',
                       help='Output directory for processed files')
    parser.add_argument('--min-comparisons', '-m',
                       type=int, default=6,
                       help='Minimum comparisons per summary in pairwise data')
    parser.add_argument('--skip-rating', action='store_true',
                       help='Skip rating data processing')
    parser.add_argument('--skip-pair', action='store_true',
                       help='Skip pairwise data processing')
    parser.add_argument('--skip-triplet', action='store_true',
                       help='Skip triplet data processing')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    df = load_and_shuffle_data(args.input)
    analyze_data_structure(df)
    
    # Process rating data
    if not args.skip_rating:
        print("\n" + "="*50)
        print("PROCESSING RATING DATA")
        print("="*50)
        processed_rating_df = process_rating_data(df)
        rating_output_path = os.path.join(args.output_dir, 'sum_humanstudy_rating_full.csv')
        processed_rating_df.to_csv(rating_output_path, index=False)
        print(f"Saved rating data to: {rating_output_path}")
    
    # Process pairwise data
    if not args.skip_pair:
        print("\n" + "="*50)
        print("PROCESSING PAIRWISE DATA")
        print("="*50)
        processed_pair_df = process_pair_data(df, args.min_comparisons)
        pair_output_path = os.path.join(args.output_dir, 'sum_humanstudy_pair_full.csv')
        processed_pair_df.to_csv(pair_output_path, index=False)
        print(f"Saved pairwise data to: {pair_output_path}")
    
    # Process triplet data
    if not args.skip_triplet:
        print("\n" + "="*50)
        print("PROCESSING TRIPLET DATA")
        print("="*50)
        
        # Load processed data if not already created
        if args.skip_rating:
            rating_output_path = os.path.join(args.output_dir, 'sum_humanstudy_rating_full.csv')
            processed_rating_df = pd.read_csv(rating_output_path)
        
        if args.skip_pair:
            pair_output_path = os.path.join(args.output_dir, 'sum_humanstudy_pair_full.csv')
            processed_pair_df = pd.read_csv(pair_output_path)
        
        # Extract and join data
        rating_pairs_df = extract_rating_pairs(processed_rating_df)
        pair_comparisons_df = extract_pair_comparisons(processed_pair_df)
        
        print(f"Extracted {len(rating_pairs_df)} rating pairs")
        print(f"Extracted {len(pair_comparisons_df)} comparison pairs")
        
        # Create triplets
        triplet_df = create_triplet_data(rating_pairs_df, pair_comparisons_df)
        
        # Verify triplet structure
        triplet_count = len(triplet_df) // 3
        remainder = len(triplet_df) % 3
        if remainder == 0:
            print(f"✅ Perfect triplet structure: {triplet_count} complete triplets")
        else:
            print(f"⚠️ Incomplete triplets: {triplet_count} complete + {remainder} remaining entries")
        
        triplet_output_path = os.path.join(args.output_dir, 'sum_humanstudy_triplet_full.csv')
        triplet_df.to_csv(triplet_output_path, index=False)
        print(f"Saved triplet data to: {triplet_output_path}")
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
