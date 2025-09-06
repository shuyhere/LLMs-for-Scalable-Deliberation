#!/usr/bin/env python3
"""
可视化标注数据分布
- Binary questions (scale_1, scale_2): 4个配对比较问题
- 5-scale questions (scale_1 到 scale_5): 4个评分问题
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

def load_data():
    """加载标注数据"""
    annotation_path = '/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full/annotated_instances.csv'
    df = pd.read_csv(annotation_path)
    print(f"Loaded {len(df)} annotation records")
    return df

def get_binary_columns():
    """获取binary问题的列名"""
    binary_questions = [
        "Which summary is more representative of your perspective?",
        "Which summary is more informative?", 
        "Which summary presents a more neutral and balanced view of the issue?",
        "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
    ]
    
    binary_cols = {}
    for q in binary_questions:
        scale_1 = f"{q}:::scale_1"
        scale_2 = f"{q}:::scale_2" 
        binary_cols[q] = [scale_1, scale_2]
    
    return binary_cols

def get_5scale_columns():
    """获取5-scale问题的列名"""
    scale_questions = [
        "To what extent is your perspective represented in this response?",
        "How informative is this summary?",
        "Do you think this summary presents a neutral and balanced view of the issue?",
        "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?"
    ]
    
    scale_cols = {}
    for q in scale_questions:
        cols = []
        for i in range(1, 6):  # scale_1 到 scale_5
            cols.append(f"{q}:::scale_{i}")
        scale_cols[q] = cols
    
    return scale_cols

def plot_binary_distributions(df, binary_cols):
    """绘制binary问题的分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e']
    
    for idx, (question, cols) in enumerate(binary_cols.items()):
        ax = axes[idx]
        
        # 合并两列数据
        all_responses = []
        labels = []
        
        for col in cols:
            if col in df.columns:
                responses = df[col].dropna()
                all_responses.extend(responses.tolist())
                labels.extend([col.split(':::')[1]] * len(responses))
        
        if all_responses:
            # 统计每个选项的数量
            response_counts = pd.Series(all_responses).value_counts()
            
            # 绘制柱状图
            bars = ax.bar(range(len(response_counts)), response_counts.values, 
                         color=colors[:len(response_counts)], alpha=0.7, edgecolor='black')
            
            ax.set_xticks(range(len(response_counts)))
            ax.set_xticklabels(response_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Count')
            
            # 简化标题
            short_title = question.replace("Which summary ", "").replace("?", "")[:40] + "..."
            ax.set_title(f'Binary: {short_title}', fontsize=11, pad=10)
            
            # 在柱子上显示数值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
            
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f'No data: {question[:40]}...', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/notebooks/binary_distributions.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_5scale_distributions(df, scale_cols):
    """绘制5-scale问题的分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    
    for idx, (question, cols) in enumerate(scale_cols.items()):
        ax = axes[idx]
        
        # 合并所有scale的数据
        all_responses = []
        
        for col in cols:
            if col in df.columns:
                responses = df[col].dropna()
                all_responses.extend(responses.tolist())
        
        if all_responses:
            # 统计每个评分的数量
            response_counts = pd.Series(all_responses).value_counts().sort_index()
            
            # 绘制柱状图
            bars = ax.bar(range(len(response_counts)), response_counts.values,
                         color=colors[:len(response_counts)], alpha=0.7, edgecolor='black')
            
            ax.set_xticks(range(len(response_counts)))
            ax.set_xticklabels(response_counts.index)
            ax.set_ylabel('Count')
            ax.set_xlabel('Scale Rating')
            
            # 简化标题
            if "To what extent" in question:
                short_title = "Perspective Representation"
            elif "How informative" in question:
                short_title = "Informativeness"
            elif "neutral and balanced" in question:
                short_title = "Neutrality & Balance"
            elif "approve" in question:
                short_title = "Policy Maker Approval"
            else:
                short_title = question[:30] + "..."
                
            ax.set_title(f'5-Scale: {short_title}', fontsize=11, pad=10)
            
            # 在柱子上显示数值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
            
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f'No data: {question[:30]}...', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('./annotation/summary-rating/notebooks/5scale_distributions.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def print_data_summary(df, binary_cols, scale_cols):
    """打印数据摘要"""
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
    """主函数"""
    print("Loading and visualizing annotation data...")
    
    # 加载数据
    df = load_data()
    
    # 获取问题列名
    binary_cols = get_binary_columns()
    scale_cols = get_5scale_columns()
    
    # 打印数据摘要
    print_data_summary(df, binary_cols, scale_cols)
    
    print(f"\nGenerating visualizations...")
    
    # 绘制binary问题分布
    print("Plotting binary distributions...")
    plot_binary_distributions(df, binary_cols)
    
    # 绘制5-scale问题分布  
    print("Plotting 5-scale distributions...")
    plot_5scale_distributions(df, scale_cols)
    
    print("Visualization complete!")
    print("Saved plots:")
    print("  - binary_distributions.png")
    print("  - 5scale_distributions.png")

if __name__ == "__main__":
    main()
