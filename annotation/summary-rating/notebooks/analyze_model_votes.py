#!/usr/bin/env python3
"""
分析每个模型在A-B比较中获得的票数
通过join annotated_instances.csv 和 sum_humanstudy_triplet_full_ring.csv 获取模型信息
"""

import pandas as pd
import numpy as np

def load_annotation_data():
    """加载标注结果数据"""
    annotation_path = '/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full_augment/annotated_instances.csv'
    df = pd.read_csv(annotation_path)
    print(f"Loaded annotation data: {len(df)} records")
    return df

def load_triplet_data():
    """加载原始三元组数据"""
    triplet_path = '/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/data_files/processed/sum_humanstudy_triplet_full_ring_augmented.csv'
    df = pd.read_csv(triplet_path)
    print(f"Loaded triplet data: {len(df)} records")
    return df

def get_comparison_columns():
    """获取A-B比较问题的列名"""
    comparison_questions = [
        "Which summary is more representative of your perspective?",
        "Which summary is more informative?", 
        "Which summary presents a more neutral and balanced view of the issue?",
        "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
    ]
    
    comparison_cols = {}
    for q in comparison_questions:
        scale_1 = f"{q}:::scale_1"
        scale_2 = f"{q}:::scale_2" 
        comparison_cols[q] = [scale_1, scale_2]
    
    return comparison_cols

def join_annotation_with_triplet(annotation_df, triplet_df):
    """将标注数据与三元组数据进行join"""
    print("Joining annotation data with triplet data...")
    
    # 筛选comparison类型的标注数据
    comparison_annotations = annotation_df[
        annotation_df['instance_id'].str.contains('comparison', na=False)
    ].copy()
    
    print(f"Found {len(comparison_annotations)} comparison annotations")
    
    # 筛选comparison类型的三元组数据
    comparison_triplets = triplet_df[
        triplet_df['type'] == 'comparison'
    ].copy()
    
    print(f"Found {len(comparison_triplets)} comparison triplets")
    
    # 进行join - 通过instance_id匹配
    joined_df = comparison_annotations.merge(
        comparison_triplets[['id', 'model_a', 'model_b', 'summary_a_id', 'summary_b_id']], 
        left_on='instance_id', 
        right_on='id', 
        how='inner'
    )
    
    print(f"Successfully joined {len(joined_df)} records")
    return joined_df

def analyze_model_votes(joined_df, comparison_cols):
    """分析每个模型获得的票数"""
    print("\nAnalyzing model votes...")
    
    # 初始化模型票数统计
    model_votes = {}
    question_stats = {}
    
    for question, cols in comparison_cols.items():
        print(f"\nAnalyzing: {question}")
        question_stats[question] = {
            'total_votes': 0,
            'model_votes': {},
            'vote_distribution': {}
        }
        
        scale_1_col = cols[0]  # Summary A
        scale_2_col = cols[1]  # Summary B
        
        # 统计每个问题的投票
        total_votes = 0
        
        # 检查scale_1列 (Summary A获胜)
        if scale_1_col in joined_df.columns:
            scale_1_votes = joined_df[joined_df[scale_1_col].notna()]
            for _, row in scale_1_votes.iterrows():
                model_a = row['model_a']
                vote_value = row[scale_1_col]
                
                if model_a not in model_votes:
                    model_votes[model_a] = 0
                if model_a not in question_stats[question]['model_votes']:
                    question_stats[question]['model_votes'][model_a] = 0
                
                # 假设scale_1表示选择Summary A
                model_votes[model_a] += 1
                question_stats[question]['model_votes'][model_a] += 1
                total_votes += 1
                
                # 记录投票分布
                vote_key = f"A_{vote_value}"
                if vote_key not in question_stats[question]['vote_distribution']:
                    question_stats[question]['vote_distribution'][vote_key] = 0
                question_stats[question]['vote_distribution'][vote_key] += 1
        
        # 检查scale_2列 (Summary B获胜)
        if scale_2_col in joined_df.columns:
            scale_2_votes = joined_df[joined_df[scale_2_col].notna()]
            for _, row in scale_2_votes.iterrows():
                model_b = row['model_b']
                vote_value = row[scale_2_col]
                
                if model_b not in model_votes:
                    model_votes[model_b] = 0
                if model_b not in question_stats[question]['model_votes']:
                    question_stats[question]['model_votes'][model_b] = 0
                
                # 假设scale_2表示选择Summary B
                model_votes[model_b] += 1
                question_stats[question]['model_votes'][model_b] += 1
                total_votes += 1
                
                # 记录投票分布
                vote_key = f"B_{vote_value}"
                if vote_key not in question_stats[question]['vote_distribution']:
                    question_stats[question]['vote_distribution'][vote_key] = 0
                question_stats[question]['vote_distribution'][vote_key] += 1
        
        question_stats[question]['total_votes'] = total_votes
        print(f"  Total votes: {total_votes}")
        
        # 显示该问题的模型排名
        if question_stats[question]['model_votes']:
            sorted_models = sorted(
                question_stats[question]['model_votes'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            print(f"  Model ranking:")
            for rank, (model, votes) in enumerate(sorted_models, 1):
                percentage = (votes / total_votes * 100) if total_votes > 0 else 0
                print(f"    {rank}. {model}: {votes} votes ({percentage:.1f}%)")
    
    return model_votes, question_stats

def print_overall_results(model_votes, question_stats):
    """打印整体结果"""
    print("\n" + "="*80)
    print("OVERALL MODEL PERFORMANCE (All A-B Comparisons)")
    print("="*80)
    
    if model_votes:
        # 总体排名
        sorted_models = sorted(model_votes.items(), key=lambda x: x[1], reverse=True)
        total_votes = sum(model_votes.values())
        
        print(f"Total votes across all questions: {total_votes}")
        print(f"\nOverall Model Ranking:")
        print("-" * 50)
        
        for rank, (model, votes) in enumerate(sorted_models, 1):
            percentage = (votes / total_votes * 100) if total_votes > 0 else 0
            print(f"{rank:2d}. {model:30s}: {votes:4d} votes ({percentage:5.1f}%)")
        
        # 按问题分解的详细结果
        print(f"\n" + "="*80)
        print("DETAILED RESULTS BY QUESTION")
        print("="*80)
        
        for question, stats in question_stats.items():
            print(f"\n{question}")
            print("-" * len(question))
            print(f"Total votes: {stats['total_votes']}")
            
            if stats['model_votes']:
                sorted_models = sorted(stats['model_votes'].items(), key=lambda x: x[1], reverse=True)
                for rank, (model, votes) in enumerate(sorted_models, 1):
                    percentage = (votes / stats['total_votes'] * 100) if stats['total_votes'] > 0 else 0
                    print(f"  {rank}. {model}: {votes} votes ({percentage:.1f}%)")
            
            # 显示投票分布
            if stats['vote_distribution']:
                print(f"  Vote distribution: {stats['vote_distribution']}")
    else:
        print("No voting data found!")

def save_results_to_csv(model_votes, question_stats):
    """保存结果到CSV文件"""
    print(f"\nSaving results to CSV files...")
    
    # 保存总体结果
    if model_votes:
        overall_df = pd.DataFrame([
            {'model': model, 'total_votes': votes, 'percentage': votes/sum(model_votes.values())*100}
            for model, votes in model_votes.items()
        ]).sort_values('total_votes', ascending=False)
        
        overall_df.to_csv('/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/notebooks/model_votes_overall.csv', index=False)
        print("  - model_votes_overall.csv")
    
    # 保存按问题分解的结果
    detailed_results = []
    for question, stats in question_stats.items():
        for model, votes in stats['model_votes'].items():
            detailed_results.append({
                'question': question,
                'model': model,
                'votes': votes,
                'percentage': votes/stats['total_votes']*100 if stats['total_votes'] > 0 else 0
            })
    
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv('/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/notebooks/model_votes_by_question.csv', index=False)
        print("  - model_votes_by_question.csv")

def main():
    """主函数"""
    print("Analyzing model votes in A-B comparisons...")
    print("="*60)
    
    try:
        # 加载数据
        annotation_df = load_annotation_data()
        triplet_df = load_triplet_data()
        
        # 获取比较问题列名
        comparison_cols = get_comparison_columns()
        print(f"Found {len(comparison_cols)} comparison questions")
        
        # Join数据
        joined_df = join_annotation_with_triplet(annotation_df, triplet_df)
        
        if len(joined_df) == 0:
            print("ERROR: No matching records found after join!")
            return
        
        # 分析模型票数
        model_votes, question_stats = analyze_model_votes(joined_df, comparison_cols)
        
        # 打印结果
        print_overall_results(model_votes, question_stats)
        
        # 保存结果
        save_results_to_csv(model_votes, question_stats)
        
        print(f"\nAnalysis complete!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
