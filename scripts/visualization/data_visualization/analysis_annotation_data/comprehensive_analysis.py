#!/usr/bin/env python3
"""
Comprehensive analysis of annotation data
Focuses on rating statistics by topic, model, and comment numbers
Enhanced with detailed statistics and clear win rate reporting
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Define unified color palette with Pastel1 style
UNIFIED_COLORS = {
    'primary': plt.cm.Pastel1.colors[:5],  # Use first 5 colors from Pastel1
    'heatmap': 'YlOrRd',  # Yellow-Orange-Red for softer appearance
    'win_rate': 'RdBu'  # Red-Blue for win rates (softer than _r version)
}

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Model name mapping to unify display with Human Judge Corr
MODEL_NAME_MAPPING = {
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
    'gpt-5': 'GPT-5',
    'gpt-5-mini': 'GPT-5-Mini',
    'gpt-5-nano': 'GPT-5-Nano',
    'gpt-4o-mini': 'GPT-4o-Mini',
    'gemini-2.5-pro': 'Gemini-2.5-Pro',
    'gemini-2.5-flash': 'Gemini-2.5-Flash',
    'gemini-2.5-flash-lite': 'Gemini-2.5-Flash-Lite',
    'grok-4-latest': 'Grok-4-Latest',
    'deepseek-chat': 'DeepSeek-Chat',
    'deepseek-reasoner': 'DeepSeek-Reasoner',
}

def get_model_display_name(name: str) -> str:
    return MODEL_NAME_MAPPING.get(name, name)

# Topic name mapping
TOPIC_NAME_MAP = {
    'Binary-Health-Care-Policy': 'Health Care',
    'Binary-Online-Identity-Policies': 'Online Identity',
    'Binary-Refugee-Policies': 'Refugee Policy',
    'Binary-Tariff-Policy': 'Tariff Policy',
    'Binary-Vaccination-Policy': 'Vaccination Policy',
    'Openqa-AI-changes-human-life': 'AI Changes Life',
    'Openqa-Tipping-System': 'Tipping System',
    'Openqa-Trump-cutting-funding': 'Academic Funding',
    'Openqa-Updates-of-electronic-products': 'Electronic Products',
    'Openqa-Influencers-as-a-job': 'Influencer',
}

# Desired topic display order
TOPIC_DISPLAY_ORDER = [
    'Tipping System',
    'AI Changes Life',
    'Academic Funding',
    'Influencer',
    'Electronic Products',
    'Tariff Policy',
    'Health Care',
    'Vaccination Policy',
    'Refugee Policy',
    'Online Identity',
]

def get_topic_display_name(topic: str) -> str:
    """Get display name for topic"""
    return TOPIC_NAME_MAP.get(topic, topic)

def load_raw_data():
    """Load raw summary data with metadata"""
    raw_data_path = PROJECT_ROOT / 'annotation/summary-rating/data_files/raw/summaries_V0903_for_humanstudy_simple.csv'
    if raw_data_path.exists():
        df = pd.read_csv(raw_data_path)
        print(f"Loaded {len(df)} raw summary records")
        return df
    else:
        print("Raw data file not found")
        return None

def load_annotation_data(base_path):
    """Load all annotation data with detailed counting"""
    base_path = Path(base_path)
    all_annotations = {
        'comparisons': [],
        'ratings': [],
        'questions': []
    }
    
    # Detailed counting
    valid_count = {'comparison': 0, 'rating': 0, 'question': 0, 'total': 0}
    invalid_count = {'comparison': 0, 'rating': 0, 'question': 0, 'total': 0}
    
    for user_dir in base_path.iterdir():
        if user_dir.is_dir():
            jsonl_file = user_dir / "annotated_instances.jsonl"
            assign_file = user_dir / "assigned_user_data.json"
            
            # Load assigned data for metadata
            assigned_data = {}
            if assign_file.exists():
                with open(assign_file, 'r') as f:
                    assigned_data = json.load(f)
            
            if jsonl_file.exists():
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            data['user_id'] = user_dir.name
                            
                            # Add metadata from assigned data
                            if data['id'] in assigned_data:
                                data['metadata'] = assigned_data[data['id']]
                            
                            # Determine type
                            if 'comparison' in data['id']:
                                ann_type = 'comparison'
                            elif 'rating' in data['id']:
                                ann_type = 'rating'
                            elif 'question' in data['id']:
                                ann_type = 'question'
                            else:
                                ann_type = None
                            
                            # Check if data is valid
                            if 'label_annotations' in data and data['label_annotations']:
                                if ann_type:
                                    valid_count[ann_type] += 1
                                    valid_count['total'] += 1
                                    
                                    if ann_type == 'comparison':
                                        all_annotations['comparisons'].append(data)
                                    elif ann_type == 'rating':
                                        all_annotations['ratings'].append(data)
                                    elif ann_type == 'question':
                                        all_annotations['questions'].append(data)
                            else:
                                if ann_type:
                                    invalid_count[ann_type] += 1
                                invalid_count['total'] += 1
                        except json.JSONDecodeError:
                            invalid_count['total'] += 1
                            continue
    
    return all_annotations, valid_count, invalid_count

def extract_rating_questions():
    """Define the 4 rating questions"""
    return [
        "To what extent is your perspective represented in this response?",
        "How informative is this summary?",
        "Do you think this summary presents a neutral and balanced view of the issue?",
        "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?"
    ]

def analyze_overall_ratings(ratings):
    """Calculate overall average ratings for 4 questions with detailed stats"""
    questions = extract_rating_questions()
    overall_stats = {}
    
    for question in questions:
        all_ratings = []
        
        for rating in ratings:
            if 'label_annotations' in rating and question in rating['label_annotations']:
                scales = rating['label_annotations'][question]
                for scale, value in scales.items():
                    if value and str(value).isdigit():
                        all_ratings.append(int(value))
        
        if all_ratings:
            overall_stats[question] = {
                'mean': np.mean(all_ratings),
                'std': np.std(all_ratings),
                'median': np.median(all_ratings),
                'count': len(all_ratings),
                'min': min(all_ratings),
                'max': max(all_ratings),
                'distribution': dict(pd.Series(all_ratings).value_counts().sort_index())
            }
    
    return overall_stats

def analyze_ratings_by_topic(ratings, raw_df):
    """Analyze ratings by topic with detailed breakdowns"""
    questions = extract_rating_questions()
    topic_stats = defaultdict(lambda: defaultdict(list))
    topic_counts = defaultdict(int)
    
    for rating in ratings:
        # Get topic from metadata
        topic = None
        if 'metadata' in rating and 'question' in rating['metadata']:
            question_text = rating['metadata']['question']
            # Match with raw data to get topic
            if raw_df is not None:
                matched = raw_df[raw_df['question'] == question_text]
                if not matched.empty:
                    topic = matched.iloc[0]['topic']
        
        if topic:
            topic_counts[topic] += 1
            for question in questions:
                if 'label_annotations' in rating and question in rating['label_annotations']:
                    scales = rating['label_annotations'][question]
                    for scale, value in scales.items():
                        if value and str(value).isdigit():
                            topic_stats[topic][question].append(int(value))
    
    # Calculate statistics
    topic_summary = {}
    for topic, questions_data in topic_stats.items():
        topic_summary[topic] = {
            'total_annotations': topic_counts[topic],
            'questions': {}
        }
        for question, ratings_list in questions_data.items():
            if ratings_list:
                topic_summary[topic]['questions'][question] = {
                    'mean': np.mean(ratings_list),
                    'std': np.std(ratings_list),
                    'median': np.median(ratings_list),
                    'count': len(ratings_list),
                    'distribution': dict(pd.Series(ratings_list).value_counts().sort_index())
                }
    
    return topic_summary

def analyze_ratings_by_model(ratings, raw_df):
    """Analyze ratings by model with counts"""
    questions = extract_rating_questions()
    model_stats = defaultdict(lambda: defaultdict(list))
    model_counts = defaultdict(int)
    
    for rating in ratings:
        # Get model from metadata
        model = None
        if 'metadata' in rating and 'model' in rating['metadata']:
            model = rating['metadata']['model']
        
        if model and model != 'question' and model != 'comparison':
            model_counts[model] += 1
            for question in questions:
                if 'label_annotations' in rating and question in rating['label_annotations']:
                    scales = rating['label_annotations'][question]
                    for scale, value in scales.items():
                        if value and str(value).isdigit():
                            model_stats[model][question].append(int(value))
    
    # Calculate statistics
    model_summary = {}
    for model, questions_data in model_stats.items():
        model_summary[model] = {
            'total_annotations': model_counts[model],
            'questions': {}
        }
        for question, ratings_list in questions_data.items():
            if ratings_list:
                model_summary[model]['questions'][question] = {
                    'mean': np.mean(ratings_list),
                    'std': np.std(ratings_list),
                    'median': np.median(ratings_list),
                    'count': len(ratings_list),
                    'distribution': dict(pd.Series(ratings_list).value_counts().sort_index())
                }
    
    return model_summary

def analyze_ratings_by_comment_num(ratings, raw_df):
    """Analyze ratings by exact comment numbers (10, 30, 50, 70, 90)"""
    questions = extract_rating_questions()
    comment_stats = defaultdict(lambda: defaultdict(list))
    comment_counts = defaultdict(int)
    
    # Create ID to comment_num mapping
    id_to_comment = {}
    if raw_df is not None:
        for _, row in raw_df.iterrows():
            id_to_comment[row['id']] = row['comment_num']
    
    for rating in ratings:
        # Get comment number from metadata
        comment_num = None
        if 'metadata' in rating and raw_df is not None:
            # Try to get comment_num from raw_id
            if 'raw_id' in rating['metadata']:
                raw_id = rating['metadata']['raw_id']
                
                # Try direct match
                if raw_id in id_to_comment:
                    comment_num = id_to_comment[raw_id]
                else:
                    # Try splitting by underscore
                    parts = raw_id.split('_')
                    for part in parts:
                        if part in id_to_comment:
                            comment_num = id_to_comment[part]
                            break
        
        if comment_num is not None:
            # Use exact comment number as key
            comment_key = str(comment_num)
            comment_counts[comment_key] += 1
            
            for question in questions:
                if 'label_annotations' in rating and question in rating['label_annotations']:
                    scales = rating['label_annotations'][question]
                    for scale, value in scales.items():
                        if value and str(value).isdigit():
                            comment_stats[comment_key][question].append(int(value))
    
    # Calculate statistics
    comment_summary = {}
    for comment_num, questions_data in comment_stats.items():
        comment_summary[comment_num] = {
            'total_annotations': comment_counts[comment_num],
            'questions': {}
        }
        for question, ratings_list in questions_data.items():
            if ratings_list:
                comment_summary[comment_num]['questions'][question] = {
                    'mean': np.mean(ratings_list),
                    'std': np.std(ratings_list),
                    'median': np.median(ratings_list),
                    'count': len(ratings_list),
                    'distribution': dict(pd.Series(ratings_list).value_counts().sort_index())
                }
    
    return comment_summary

def analyze_model_win_rates(comparisons):
    """Analyze model vs model win rates with clear naming"""
    comparison_questions = [
        "Which summary is more representative of your perspective?",
        "Which summary is more informative?",
        "Which summary presents a more neutral and balanced view of the issue?",
        "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
    ]
    
    model_pairs = defaultdict(lambda: defaultdict(lambda: {'A_wins': 0, 'B_wins': 0, 'ties': 0, 'total': 0}))
    
    for comp in comparisons:
        if 'metadata' not in comp:
            continue
        
        model_a = comp['metadata'].get('model_a', 'unknown')
        model_b = comp['metadata'].get('model_b', 'unknown')
        
        if model_a == 'unknown' or model_b == 'unknown':
            continue
        
        for question in comparison_questions:
            if 'label_annotations' in comp and question in comp['label_annotations']:
                scales = comp['label_annotations'][question]
                model_pairs[(model_a, model_b)][question]['total'] += 1
                
                # Determine winner
                winner_found = False
                if 'scale_1' in scales and scales['scale_1'] == '1':
                    model_pairs[(model_a, model_b)][question]['A_wins'] += 1
                    winner_found = True
                elif 'scale_2' in scales and scales['scale_2'] == '2':
                    model_pairs[(model_a, model_b)][question]['B_wins'] += 1
                    winner_found = True
                else:
                    # Check for any response
                    for scale, value in scales.items():
                        if value == '1':
                            model_pairs[(model_a, model_b)][question]['A_wins'] += 1
                            winner_found = True
                            break
                        elif value == '2':
                            model_pairs[(model_a, model_b)][question]['B_wins'] += 1
                            winner_found = True
                            break
                
                if not winner_found:
                    model_pairs[(model_a, model_b)][question]['ties'] += 1
    
    # Calculate win rates with clear naming
    win_rates = {}
    for (model_a, model_b), questions_data in model_pairs.items():
        pair_key = f"{model_a} vs {model_b}"
        win_rates[pair_key] = {
            'model_a': model_a,
            'model_b': model_b,
            'questions': {}
        }
        
        for question, results in questions_data.items():
            if results['total'] > 0:
                win_rates[pair_key]['questions'][question] = {
                    f'{model_a}_wins': results['A_wins'],
                    f'{model_b}_wins': results['B_wins'],
                    'ties': results['ties'],
                    'total_comparisons': results['total'],
                    f'{model_a}_win_rate_%': (results['A_wins'] / results['total']) * 100 if results['total'] > 0 else 0,
                    f'{model_b}_win_rate_%': (results['B_wins'] / results['total']) * 100 if results['total'] > 0 else 0,
                    'tie_rate_%': (results['ties'] / results['total']) * 100 if results['total'] > 0 else 0
                }
    
    return win_rates

def create_rating_visualization(overall_stats, output_path):
    """Create visualization for overall ratings"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    questions_short = [
        "Representiveness",
        "Informativeness",
        "Neutrality",
        "Policy Approval"
    ]
    
    colors = UNIFIED_COLORS['primary']
    
    for idx, (question, short_name) in enumerate(zip(overall_stats.keys(), questions_short)):
        ax = axes[idx]
        
        stats = overall_stats[question]
        dist = stats['distribution']
        
        # Ensure all scales 1-5 are represented
        scales = list(range(1, 6))
        values = [dist.get(s, 0) for s in scales]
        
        bars = ax.bar(scales, values, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Rating Scale', fontsize=18)
        ax.set_ylabel('Frequency', fontsize=18)
        ax.set_title(f'{short_name}', fontsize=18, fontweight='bold')
        ax.set_xticks(scales)
        ax.tick_params(axis='both', labelsize=14)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                pct = (val / stats["count"]) * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                       f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Overall Rating Distributions', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'overall_ratings.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_topic_heatmap(topic_summary, output_path):
    """Create heatmap for ratings by topic"""
    if not topic_summary:
        return
    
    questions_short = ["Representiveness", "Informativeness", "Neutrality", "Policy Approval"]
    
    # Prepare data for heatmap
    topics = list(topic_summary.keys())
    # Sort topics by desired display order
    topics.sort(key=lambda t: TOPIC_DISPLAY_ORDER.index(get_topic_display_name(t)) if get_topic_display_name(t) in TOPIC_DISPLAY_ORDER else len(TOPIC_DISPLAY_ORDER))
    questions = list(extract_rating_questions())
    
    matrix = []
    annotations = []
    for topic in topics:
        row = []
        for question in questions:
            if question in topic_summary[topic]['questions']:
                row.append(topic_summary[topic]['questions'][question]['mean'])
            else:
                row.append(np.nan)
        matrix.append(row)
        display_name = get_topic_display_name(topic)
        annotations.append(f"{display_name}")
    
    if matrix:
        plt.figure(figsize=(10, max(6, len(topics) * 0.4)))
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                   vmin=1, vmax=5, center=3,
                   xticklabels=questions_short, 
                   yticklabels=annotations,
                   annot_kws={'fontsize': 14},
                   cbar_kws={'label': 'Average Rating'})
        
        plt.title('Average Ratings by Topic', fontsize=16, fontweight='bold')
        plt.xlabel('Dimensions', fontsize=16)
        plt.ylabel('Topics', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Increase colorbar fonts
        cbar = plt.gcf().axes[-1]
        cbar.tick_params(labelsize=14)
        cbar.set_ylabel('Average Rating', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'ratings_by_topic.pdf', dpi=300, bbox_inches='tight')
        plt.close()

def create_model_comparison_chart(model_summary, output_path):
    """Create comparison chart for model ratings"""
    if not model_summary:
        return
    
    questions_short = ["Representiveness", "Informativeness", "Neutrality", "Policy Approval"]
    questions = extract_rating_questions()
    
    # Calculate overall average for sorting
    model_overall_avg = {}
    for model, data in model_summary.items():
        all_means = []
        for question in questions:
            if question in data['questions']:
                all_means.append(data['questions'][question]['mean'])
        if all_means:
            model_overall_avg[model] = np.mean(all_means)
    
    # Sort models by overall average
    sorted_models = sorted(model_overall_avg.items(), key=lambda x: x[1], reverse=True)[:10]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Use unified colors, cycling if we have more models than colors
    base_colors = UNIFIED_COLORS['primary']
    colors = [base_colors[i % len(base_colors)] for i in range(len(sorted_models))]
    
    for idx, (question, short_name) in enumerate(zip(questions, questions_short)):
        ax = axes[idx]
        
        model_means = []
        model_stds = []
        model_names = []
        model_counts = []
        
        for model, _ in sorted_models:
            if question in model_summary[model]['questions']:
                stats = model_summary[model]['questions'][question]
                model_means.append(stats['mean'])
                model_stds.append(stats['std'])
                model_names.append(f"{get_model_display_name(model)[:20]}")
                model_counts.append(stats['count'])
        
        if model_means:
            x_pos = np.arange(len(model_names))
            bars = ax.bar(x_pos, model_means, 
                         color=colors[:len(model_names)], alpha=0.7, 
                         edgecolor='black')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Average Rating', fontsize=18)
            ax.set_title(short_name, fontsize=18, fontweight='bold')
            ax.set_ylim([1, 5])
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='both', labelsize=14)
            
            # Add value labels
            for bar, mean in zip(bars, model_means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Average Ratings by Model', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'ratings_by_model.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_comment_num_chart(comment_summary, output_path):
    """Create chart for ratings by comment number"""
    if not comment_summary:
        return
    
    questions = extract_rating_questions()
    questions_short = ["Representiveness", "Informativeness", "Neutrality", "Policy Approval"]
    
    # Expected comment numbers
    expected_nums = ['10', '30', '50', '70', '90']
    
    plt.figure(figsize=(12, 8))
    
    for q_idx, (question, short_name) in enumerate(zip(questions, questions_short)):
        means = []
        nums_with_data = []
        
        for num in expected_nums:
            if num in comment_summary and question in comment_summary[num]['questions']:
                means.append(comment_summary[num]['questions'][question]['mean'])
                nums_with_data.append(int(num))
        
        if means:
            color = UNIFIED_COLORS['primary'][q_idx % len(UNIFIED_COLORS['primary'])]
            plt.plot(nums_with_data, means, marker='o', label=short_name, linewidth=2, markersize=8, color=color)
    
    plt.xlabel('Number of Comments', fontsize=18)
    plt.ylabel('Average Rating', fontsize=18)
    plt.title('Average Ratings by Number of Comments', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([1, 5])
    plt.xticks([10, 30, 50, 70, 90], fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path / 'ratings_by_comment_number.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_win_rate_matrix(win_rates, output_path):
    """Create win rate matrix visualization with clear labels"""
    if not win_rates:
        return
    
    # Focus on the policy maker preference question
    main_question = "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
    
    # Extract unique models
    all_models = set()
    for pair_data in win_rates.values():
        all_models.add(pair_data['model_a'])
        all_models.add(pair_data['model_b'])
    
    all_models = sorted(list(all_models))
    
    # Create matrix
    matrix = np.zeros((len(all_models), len(all_models)))
    count_matrix = np.zeros((len(all_models), len(all_models)))
    
    for i, model_a in enumerate(all_models):
        for j, model_b in enumerate(all_models):
            if i == j:
                matrix[i][j] = 50  # Diagonal
            else:
                pair_key = f"{model_a} vs {model_b}"
                reverse_key = f"{model_b} vs {model_a}"
                
                if pair_key in win_rates and main_question in win_rates[pair_key]['questions']:
                    data = win_rates[pair_key]['questions'][main_question]
                    matrix[i][j] = data[f'{model_a}_win_rate_%']
                    count_matrix[i][j] = data['total_comparisons']
                elif reverse_key in win_rates and main_question in win_rates[reverse_key]['questions']:
                    data = win_rates[reverse_key]['questions'][main_question]
                    matrix[i][j] = 100 - data[f'{model_b}_win_rate_%']
                    count_matrix[i][j] = data['total_comparisons']
    
    # Plot
    plt.figure(figsize=(14, 12))
    
    # Create annotations with win rate and count
    annot_matrix = []
    for i in range(len(all_models)):
        row = []
        for j in range(len(all_models)):
            if i == j:
                row.append("---")
            elif count_matrix[i][j] > 0:
                row.append(f"{matrix[i][j]:.1f}%")
            else:
                row.append("N/A")
        annot_matrix.append(row)
    
    sns.heatmap(matrix, annot=annot_matrix, fmt='', cmap='RdBu_r',
               center=50, vmin=0, vmax=100,
               xticklabels=[get_model_display_name(m)[:20] for m in all_models],
               yticklabels=[get_model_display_name(m)[:20] for m in all_models],
               cbar_kws={'label': 'Win Rate (%) - A vs B'},
               annot_kws={'fontsize': 14})
    
    # Increase tick label font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Increase colorbar font sizes
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=14)
    cbar.set_ylabel('Win Rate (%) - A vs B', fontsize=16)
    
    plt.title('Model Win Rates - Policy Maker Preference\n(A vs B: Row Model A wins against Column Model B)', 
             fontsize=20, fontweight='bold')
    plt.xlabel('Model B (Column)', fontsize=18)
    plt.ylabel('Model A (Row)', fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path / 'model_win_rates.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(valid_count, invalid_count, overall_stats, 
                                 topic_summary, model_summary, comment_summary, win_rates):
    """Generate comprehensive analysis report with detailed statistics"""
    report = []
    
    report.append("=" * 100)
    report.append("COMPREHENSIVE ANNOTATION ANALYSIS REPORT")
    report.append("=" * 100)
    
    # Data validity with breakdown
    report.append(f"\n📊 DATA VALIDITY STATISTICS")
    report.append(f"  {'Type':<15} {'Valid':<10} {'Invalid':<10} {'Validity Rate':<15}")
    report.append(f"  {'-'*50}")
    
    for ann_type in ['comparison', 'rating', 'question']:
        valid = valid_count.get(ann_type, 0)
        invalid = invalid_count.get(ann_type, 0)
        total = valid + invalid
        rate = (valid / total * 100) if total > 0 else 0
        report.append(f"  {ann_type:<15} {valid:<10} {invalid:<10} {rate:<.1f}%")
    
    report.append(f"  {'-'*50}")
    report.append(f"  {'TOTAL':<15} {valid_count['total']:<10} {invalid_count['total']:<10} "
                 f"{valid_count['total']/(valid_count['total']+invalid_count['total'])*100 if (valid_count['total']+invalid_count['total']) > 0 else 0:.1f}%")
    
    # Overall ratings with full distribution
    report.append(f"\n📈 OVERALL AVERAGE RATINGS (4 QUESTIONS)")
    report.append(f"  {'='*80}")
    
    for i, question in enumerate(overall_stats, 1):
        stats = overall_stats[question]
        report.append(f"\n  Question {i}: {question[:70]}...")
        report.append(f"    Valid Responses: {stats['count']}")
        report.append(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        report.append(f"    Median: {stats['median']:.1f}")
        report.append(f"    Range: {stats['min']}-{stats['max']}")
        
        report.append(f"    Distribution:")
        total = stats['count']
        for rating in range(1, 6):
            count = stats['distribution'].get(rating, 0)
            pct = (count / total * 100) if total > 0 else 0
            bar = '█' * int(pct / 2)
            report.append(f"      Rating {rating}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Topic analysis with counts
    if topic_summary:
        report.append(f"\n🏷️ RATINGS BY TOPIC")
        report.append(f"  {'='*80}")
        
        # Sort topics by total annotations
        sorted_topics = sorted(topic_summary.items(), 
                              key=lambda x: x[1]['total_annotations'], 
                              reverse=True)
        
        for topic, data in sorted_topics:
            display_name = get_topic_display_name(topic)
            report.append(f"\n  Topic: {display_name} ({topic})")
            report.append(f"    Total Annotations: {data['total_annotations']}")
            
            # Calculate overall average across questions
            all_means = []
            for question in data['questions']:
                all_means.append(data['questions'][question]['mean'])
            
            if all_means:
                report.append(f"    Overall Average: {np.mean(all_means):.3f}")
                
                # Show each question's stats
                for q_idx, question in enumerate(extract_rating_questions(), 1):
                    if question in data['questions']:
                        q_stats = data['questions'][question]
                        report.append(f"    Q{q_idx}: Mean={q_stats['mean']:.2f}, "
                                    f"Std={q_stats['std']:.2f}, N={q_stats['count']}")
    
    # Model analysis with counts
    if model_summary:
        report.append(f"\n🤖 RATINGS BY MODEL")
        report.append(f"  {'='*80}")
        
        # Calculate overall average for sorting
        model_avg = {}
        for model, data in model_summary.items():
            all_means = []
            for question in data['questions']:
                all_means.append(data['questions'][question]['mean'])
            if all_means:
                model_avg[model] = (np.mean(all_means), data['total_annotations'])
        
        # Sort by average rating
        sorted_models = sorted(model_avg.items(), key=lambda x: x[1][0], reverse=True)
        
        report.append(f"\n  {'Model':<30} {'Avg Rating':<12} {'Total Ann.':<12}")
        report.append(f"  {'-'*54}")
        
        for model, (avg, count) in sorted_models[:15]:  # Top 15 models
            display = get_model_display_name(model)
            report.append(f"  {display[:30]:<30} {avg:<12.3f} {count:<12}")
    
    # Comment number analysis with exact numbers
    if comment_summary:
        report.append(f"\n💬 RATINGS BY NUMBER OF COMMENTS")
        report.append(f"  {'='*80}")
        
        # Sort by comment number
        sorted_comments = sorted(comment_summary.items(), key=lambda x: int(x[0]))
        
        report.append(f"\n  {'Comments':<12} {'Annotations':<15} {'Avg Rating':<15}")
        report.append(f"  {'-'*42}")
        
        for comment_num, data in sorted_comments:
            all_means = []
            for question in data['questions']:
                all_means.append(data['questions'][question]['mean'])
            
            if all_means:
                avg = np.mean(all_means)
                report.append(f"  {comment_num:<12} {data['total_annotations']:<15} {avg:<15.3f}")
                
                # Add question breakdown
                for q_idx, question in enumerate(extract_rating_questions(), 1):
                    if question in data['questions']:
                        q_stats = data['questions'][question]
                        report.append(f"    Q{q_idx}: Mean={q_stats['mean']:.2f}, N={q_stats['count']}")
    
    # Win rates with clear model identification
    if win_rates:
        report.append(f"\n🏆 MODEL VS MODEL WIN RATES")
        report.append(f"  {'='*80}")
        
        # Focus on policy maker preference
        policy_question = "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
        
        report.append(f"\n  Policy Maker Preference Win Rates:")
        report.append(f"  {'-'*60}")
        
        # Sort by total comparisons
        sorted_pairs = []
        for pair, data in win_rates.items():
            if policy_question in data['questions']:
                q_data = data['questions'][policy_question]
                sorted_pairs.append((pair, data, q_data['total_comparisons']))
        
        sorted_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for pair, data, total in sorted_pairs[:20]:  # Top 20 pairs
            model_a = data['model_a']
            model_b = data['model_b']
            
            if policy_question in data['questions']:
                q_data = data['questions'][policy_question]
                
                report.append(f"\n  {model_a} vs {model_b}")
                report.append(f"    Total Comparisons: {q_data['total_comparisons']}")
                report.append(f"    {model_a} wins: {q_data[f'{model_a}_wins']} ({q_data[f'{model_a}_win_rate_%']:.1f}%)")
                report.append(f"    {model_b} wins: {q_data[f'{model_b}_wins']} ({q_data[f'{model_b}_win_rate_%']:.1f}%)")
                if q_data['ties'] > 0:
                    report.append(f"    Ties: {q_data['ties']} ({q_data['tie_rate_%']:.1f}%)")
        
        # Overall model performance
        report.append(f"\n  Overall Model Performance (Policy Maker Preference):")
        report.append(f"  {'-'*60}")
        
        model_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})
        
        for pair, data in win_rates.items():
            if policy_question in data['questions']:
                q_data = data['questions'][policy_question]
                model_a = data['model_a']
                model_b = data['model_b']
                
                model_performance[model_a]['wins'] += q_data[f'{model_a}_wins']
                model_performance[model_a]['losses'] += q_data[f'{model_b}_wins']
                model_performance[model_a]['total'] += q_data['total_comparisons']
                
                model_performance[model_b]['wins'] += q_data[f'{model_b}_wins']
                model_performance[model_b]['losses'] += q_data[f'{model_a}_wins']
                model_performance[model_b]['total'] += q_data['total_comparisons']
        
        # Calculate win rates
        model_win_rates = []
        for model, stats in model_performance.items():
            if stats['total'] > 0:
                win_rate = (stats['wins'] / stats['total']) * 100
                model_win_rates.append((model, win_rate, stats['wins'], stats['losses'], stats['total']))
        
        model_win_rates.sort(key=lambda x: x[1], reverse=True)
        
        report.append(f"\n  {'Model':<30} {'Win Rate':<12} {'Wins':<10} {'Losses':<10} {'Total':<10}")
        report.append(f"  {'-'*72}")
        
        for model, win_rate, wins, losses, total in model_win_rates[:15]:  # Top 15
            report.append(f"  {model[:30]:<30} {win_rate:<12.1f}% {wins:<10} {losses:<10} {total:<10}")
    
    return "\n".join(report)

def main():
    """Main analysis function"""
    # Paths
    annotation_path = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full'
    output_path = PROJECT_ROOT / 'results/dataset_visulization/analysis_annotation'
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw data...")
    raw_df = load_raw_data()
    
    print("Loading annotation data...")
    annotations, valid_count, invalid_count = load_annotation_data(annotation_path)
    
    print(f"\nData Summary:")
    print(f"  Valid - Comparisons: {valid_count['comparison']}, Ratings: {valid_count['rating']}, Questions: {valid_count['question']}")
    print(f"  Invalid - Total: {invalid_count['total']}")
    
    # Extract different types
    ratings = annotations['ratings']
    comparisons = annotations['comparisons']
    questions = annotations['questions']
    
    # Analyze overall ratings
    print("\nAnalyzing overall ratings...")
    overall_stats = analyze_overall_ratings(ratings)
    
    # Analyze by dimensions
    print("Analyzing ratings by topic...")
    topic_summary = analyze_ratings_by_topic(ratings, raw_df) if raw_df is not None else {}
    
    print("Analyzing ratings by model...")
    model_summary = analyze_ratings_by_model(ratings, raw_df) if raw_df is not None else {}
    
    print("Analyzing ratings by comment number...")
    comment_summary = analyze_ratings_by_comment_num(ratings, raw_df) if raw_df is not None else {}
    
    print("Analyzing model win rates...")
    win_rates = analyze_model_win_rates(comparisons)
    
    # Generate report
    report = generate_comprehensive_report(valid_count, invalid_count, overall_stats,
                                          topic_summary, model_summary, comment_summary, win_rates)
    
    print("\n" + report[:2000] + "...")  # Print first part of report
    
    # Save report
    report_file = output_path / 'comprehensive_analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_rating_visualization(overall_stats, output_path)
    create_topic_heatmap(topic_summary, output_path)
    create_model_comparison_chart(model_summary, output_path)
    create_comment_num_chart(comment_summary, output_path)
    create_win_rate_matrix(win_rates, output_path)
    
    print(f"Visualizations saved to: {output_path}")
    
    # Skip CSV outputs per requirement
    print("\nAnalysis complete!")
    print(f"Visualizations and report saved to: {output_path}")

if __name__ == "__main__":
    main()