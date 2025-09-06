#!/usr/bin/env python3
"""
可视化模型投票结果
绘制总体排名和按问题分解的柱状图
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

def load_vote_data():
    """加载投票结果数据"""
    try:
        overall_df = pd.read_csv('/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/notebooks/model_votes_overall.csv')
        detailed_df = pd.read_csv('/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/notebooks/model_votes_by_question.csv')
        print(f"Loaded overall data: {len(overall_df)} models")
        print(f"Loaded detailed data: {len(detailed_df)} records")
        return overall_df, detailed_df
    except FileNotFoundError as e:
        print(f"Error: Could not find vote data files. Please run analyze_model_votes.py first.")
        print(f"Missing file: {e}")
        return None, None

def plot_overall_votes(overall_df):
    """绘制总体投票结果柱状图"""
    if overall_df is None or len(overall_df) == 0:
        print("No overall data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 按票数排序
    overall_df_sorted = overall_df.sort_values('total_votes', ascending=True)
    
    # 颜色方案
    colors = plt.cm.Set3(np.linspace(0, 1, len(overall_df_sorted)))
    
    # 左图：总票数
    bars1 = ax1.barh(range(len(overall_df_sorted)), overall_df_sorted['total_votes'], 
                     color=colors, alpha=0.8, edgecolor='black')
    ax1.set_yticks(range(len(overall_df_sorted)))
    ax1.set_yticklabels(overall_df_sorted['model'], fontsize=11)
    ax1.set_xlabel('Total Votes', fontsize=12)
    ax1.set_title('Model Performance - Total Votes in A-B Comparisons', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 在柱子上显示数值
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 右图：百分比
    bars2 = ax2.barh(range(len(overall_df_sorted)), overall_df_sorted['percentage'], 
                     color=colors, alpha=0.8, edgecolor='black')
    ax2.set_yticks(range(len(overall_df_sorted)))
    ax2.set_yticklabels(overall_df_sorted['model'], fontsize=11)
    ax2.set_xlabel('Percentage (%)', fontsize=12)
    ax2.set_title('Model Performance - Win Rate Percentage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 在柱子上显示百分比
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/notebooks/model_votes_overall.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_votes_by_question(detailed_df):
    """绘制按问题分解的投票结果"""
    if detailed_df is None or len(detailed_df) == 0:
        print("No detailed data to plot")
        return
    
    # 获取所有问题
    questions = detailed_df['question'].unique()
    
    # 简化问题标题
    question_mapping = {
        "Which summary is more representative of your perspective?": "Perspective\nRepresentative",
        "Which summary is more informative?": "More\nInformative", 
        "Which summary presents a more neutral and balanced view of the issue?": "Neutral &\nBalanced",
        "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?": "Policy Maker\nPreference"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(detailed_df['model'].unique())))
    model_color_map = dict(zip(detailed_df['model'].unique(), colors))
    
    for idx, question in enumerate(questions):
        if idx >= 4:  # 只显示前4个问题
            break
            
        ax = axes[idx]
        question_data = detailed_df[detailed_df['question'] == question].copy()
        
        if len(question_data) == 0:
            ax.set_title(f'No data: {question_mapping.get(question, question[:20])}')
            continue
        
        # 按票数排序
        question_data_sorted = question_data.sort_values('votes', ascending=True)
        
        # 获取对应的颜色
        bar_colors = [model_color_map[model] for model in question_data_sorted['model']]
        
        # 绘制柱状图
        bars = ax.barh(range(len(question_data_sorted)), question_data_sorted['votes'],
                      color=bar_colors, alpha=0.8, edgecolor='black')
        
        ax.set_yticks(range(len(question_data_sorted)))
        ax.set_yticklabels(question_data_sorted['model'], fontsize=10)
        ax.set_xlabel('Votes', fontsize=11)
        
        # 使用简化的标题
        short_title = question_mapping.get(question, question[:30] + "...")
        ax.set_title(short_title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 在柱子上显示数值和百分比
        for i, bar in enumerate(bars):
            width = bar.get_width()
            percentage = question_data_sorted.iloc[i]['percentage']
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                   f'{int(width)} ({percentage:.1f}%)', 
                   ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/notebooks/model_votes_by_question.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison_matrix(detailed_df):
    """绘制模型比较矩阵热力图"""
    if detailed_df is None or len(detailed_df) == 0:
        print("No detailed data for comparison matrix")
        return
    
    # 创建数据透视表
    pivot_df = detailed_df.pivot(index='model', columns='question', values='votes')
    
    # 简化列名
    question_mapping = {
        "Which summary is more representative of your perspective?": "Perspective",
        "Which summary is more informative?": "Informative", 
        "Which summary presents a more neutral and balanced view of the issue?": "Neutral",
        "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?": "Policy"
    }
    
    pivot_df.columns = [question_mapping.get(col, col[:15]) for col in pivot_df.columns]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制热力图
    sns.heatmap(pivot_df, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Votes'}, ax=ax,
                linewidths=0.5, linecolor='white')
    
    ax.set_title('Model Performance Matrix - Votes by Question Type', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Question Type', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/notebooks/model_comparison_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_stats(overall_df, detailed_df):
    """打印统计摘要"""
    print("="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    
    if overall_df is not None:
        print(f"Total models analyzed: {len(overall_df)}")
        print(f"Total votes: {overall_df['total_votes'].sum()}")
        print(f"Average votes per model: {overall_df['total_votes'].mean():.1f}")
        
        winner = overall_df.loc[overall_df['total_votes'].idxmax()]
        print(f"Best performing model: {winner['model']} ({winner['total_votes']} votes, {winner['percentage']:.1f}%)")
    
    if detailed_df is not None:
        print(f"Questions analyzed: {detailed_df['question'].nunique()}")
        print(f"Total question-model combinations: {len(detailed_df)}")

def main():
    """主函数"""
    print("Visualizing model voting results...")
    print("="*50)
    
    # 加载数据
    overall_df, detailed_df = load_vote_data()
    
    if overall_df is None or detailed_df is None:
        print("Cannot proceed without vote data. Please run analyze_model_votes.py first.")
        return
    
    # 打印统计摘要
    print_summary_stats(overall_df, detailed_df)
    
    print(f"\nGenerating visualizations...")
    
    # 绘制总体结果
    print("1. Plotting overall model performance...")
    plot_overall_votes(overall_df)
    
    # 绘制按问题分解的结果
    print("2. Plotting results by question...")
    plot_votes_by_question(detailed_df)
    
    # 绘制比较矩阵
    print("3. Plotting model comparison matrix...")
    plot_model_comparison_matrix(detailed_df)
    
    print("\nVisualization complete!")
    print("Generated plots:")
    print("  - model_votes_overall.png")
    print("  - model_votes_by_question.png") 
    print("  - model_comparison_matrix.png")

if __name__ == "__main__":
    main()
