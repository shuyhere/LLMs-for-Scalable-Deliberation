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

def _short_option_label(option: str) -> str:
    """将选项标题缩短以适配坐标轴显示。"""
    if option is None:
        return ""
    text = str(option)
    low = text.lower()
    # 常见模式映射
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
    # 通用裁剪：遇到 em dash/短横说明的，取前半部分
    for sep in [" — ", " - ", "–", "—"]:
        if sep in text:
            head = text.split(sep)[0].strip()
            if len(head) >= 1:
                text = head
                break
    # 最终长度限制
    return (text[:16] + "…") if len(text) > 17 else text

def _scale_order_patterns(question: str):
    """给定问题文本，返回该题对应的有序选项模式（从低到高）。"""
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
    """比较题固定顺序：A much → A slightly → Both → B slightly → B much。"""
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

def load_data():
    """加载标注数据（优先 full_augment，若不存在则回退到 full）"""
    path_augment = '/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full_augment/annotated_instances.csv'
    path_full = '/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full/annotated_instances.csv'
    try:
        df = pd.read_csv(path_augment)
        print(f"Loaded {len(df)} annotation records (full_augment)")
        return df
    except FileNotFoundError:
        df = pd.read_csv(path_full)
        print(f"Loaded {len(df)} annotation records (full)")
        return df

def get_binary_columns(df):
    """获取比较问题对应的列（新版：问题:::选项）。返回 {question: [cols...]}"""
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
    """获取5级量表问题对应的列（新版：问题:::选项）。返回 {question: [cols...]}"""
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
    """绘制比较题（新版5选项）的分布图：按“问题:::选项”列统计每个选项的计数。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (question, cols) in enumerate(binary_cols.items()):
        ax = axes[idx]
        
        # 统计每个选项的非空计数
        option_counts = {}
        for col in cols:
            if col in df.columns:
                option = str(col).split(':::', 1)[-1].strip()
                option_counts[option] = int(df[col].notna().sum())
        
        if option_counts:
            # 固定顺序而非按数量排序
            items = list(option_counts.items())
            items_sorted = sorted(items, key=lambda t: (_comparison_rank(t[0]), t[0]))
            ordered_options = [opt for opt, _ in items_sorted]
            ordered_values = [option_counts[opt] for opt in ordered_options]
            
            # 绘制柱状图
            bars = ax.bar(range(len(ordered_values)), ordered_values, 
                         color=plt.cm.Set2(np.linspace(0, 1, len(ordered_values))), alpha=0.7, edgecolor='black')
            
            ax.set_xticks(range(len(ordered_options)))
            ax.set_xticklabels(ordered_options, rotation=45, ha='right')
            ax.set_ylabel('Count')
            
            # 简化标题
            short_title = question.replace("Which summary ", "").replace("?", "")[:40] + "..."
            ax.set_title(f'Comparison: {short_title}', fontsize=11, pad=10)
            
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
    """绘制5级量表题的分布图：按“问题:::选项”列统计各选项计数。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (question, cols) in enumerate(scale_cols.items()):
        ax = axes[idx]
        
        # 统计每个选项的非空计数
        option_counts = {}
        for col in cols:
            if col in df.columns:
                option = str(col).split(':::', 1)[-1].strip()
                option_counts[option] = int(df[col].notna().sum())
        
        if option_counts:
            # 按题型定义的刻度顺序排序（找不到匹配则排在末尾）
            items = list(option_counts.items())
            items_sorted = sorted(items, key=lambda t: (_scale_rank(question, t[0]), t[0]))
            ordered_options = [opt for opt, _ in items_sorted]
            ordered_values = [option_counts[opt] for opt in ordered_options]
            
            # 绘制柱状图
            bars = ax.bar(range(len(ordered_values)), ordered_values,
                         color=plt.cm.viridis(np.linspace(0, 1, len(ordered_values))), alpha=0.7, edgecolor='black')
            
            ax.set_xticks(range(len(ordered_options)))
            short_labels = [ _short_option_label(x) for x in ordered_options ]
            ax.set_xticklabels(short_labels, rotation=0)
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
    plt.savefig('/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/notebooks/5scale_distributions.png', 
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
    
    # 获取问题列名（按当前数据动态匹配列）
    binary_cols = get_binary_columns(df)
    scale_cols = get_5scale_columns(df)
    
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
