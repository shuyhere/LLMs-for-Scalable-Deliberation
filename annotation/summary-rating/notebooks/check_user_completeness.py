#!/usr/bin/env python3
"""
检查每个用户是否完成了三个问题类型的标注
数据结构：每个用户有3行数据（question, rating, comparison），但不一定连续
"""

import pandas as pd
import numpy as np
import os

def check_user_directories():
    """检查用户目录结构，找出只有assigned_user_data.json的用户"""
    annotation_dir = './annotation/summary-rating/annotation_output/full_augment'
    
    assigned_only_users = []
    
    # 遍历所有子目录
    for item in os.listdir(annotation_dir):
        item_path = os.path.join(annotation_dir, item)
        
        # 跳过非目录和特殊目录
        if not os.path.isdir(item_path) or item in ['archived_users']:
            continue
        
        # 检查目录中的文件
        try:
            files = os.listdir(item_path)
            
            # 如果只有assigned_user_data.json，说明用户被分配但未开始
            if len(files) == 1 and 'assigned_user_data.json' in files:
                assigned_only_users.append(item)
            elif len(files) < 3:
                # 如果文件数少于3个，也可能是不完整
                print(f"⚠️  用户 {item}: 只有 {len(files)} 个文件: {files}")
                
        except PermissionError:
            print(f"⚠️  无法访问目录: {item}")
            continue
    
    return assigned_only_users

def check_user_completeness():
    """检查用户完成情况，返回不完整用户列表"""
    print("检查用户标注完成情况...")
    print("="*50)
    
    # 1. 检查用户目录结构
    print("1. 检查用户目录结构...")
    assigned_only_users = check_user_directories()
    
    if assigned_only_users:
        print(f"找到 {len(assigned_only_users)} 个只被分配但未开始的用户:")
        for user in assigned_only_users[:10]:  # 只显示前10个
            print(f"  - {user}")
        if len(assigned_only_users) > 10:
            print(f"  ... 还有 {len(assigned_only_users)-10} 个")
    
    # 2. 加载标注数据并检查完成情况
    print(f"\n2. 检查CSV中的用户完成情况...")
    annotation_path = './annotation/summary-rating/annotation_output/full_augment/annotated_instances.csv'
    df = pd.read_csv(annotation_path)
    
    print(f"总标注记录数: {len(df)}")
    print(f"CSV中用户数: {df['user'].nunique()}")
    
    # 按用户分组检查CSV中的数据完整性
    csv_incomplete_users = []
    complete_users = []
    
    for user_id, user_data in df.groupby('user'):
        user_issues = []
        
        # 检查是否有三种类型的数据
        instance_types = user_data['instance_id'].tolist()
        has_question = any('question' in str(inst_id) for inst_id in instance_types)
        has_rating = any('rating' in str(inst_id) for inst_id in instance_types)
        has_comparison = any('comparison' in str(inst_id) for inst_id in instance_types)
        
        if not (has_question and has_rating and has_comparison):
            missing_types = []
            if not has_question:
                missing_types.append('question')
            if not has_rating:
                missing_types.append('rating')
            if not has_comparison:
                missing_types.append('comparison')
            user_issues.append(f"缺少类型: {', '.join(missing_types)}")
        
        # 检查每种类型的数据完整性
        for idx, row in user_data.iterrows():
            instance_id = str(row['instance_id'])
            
            if 'question' in instance_id:
                # triplet_n_question: answer:::text_box 列应该有值
                text_answer = row.get('answer:::text_box', '')
                if pd.isna(text_answer) or str(text_answer).strip() == '':
                    user_issues.append(f"question类型无文本回答")
                    
            elif 'rating' in instance_id:
                # 新版：列名为“问题本体 + ::: + 具体选项”，不再使用 scale_1..5
                # 动态匹配该行在每个问题下是否至少选择了一个选项
                questions = [
                    'To what extent is your perspective represented in this response?',
                    'How informative is this summary?',
                    'Do you think this summary presents a neutral and balanced view of the issue?',
                    'Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?'
                ]
                
                missing_questions = []
                for question in questions:
                    # 该问题对应的所有列：以问题开头（后接可有空格）
                    question_cols = [c for c in df.columns if str(c).startswith(question)]
                    has_answer = any(not pd.isna(row.get(col, '')) and str(row.get(col, '')).strip() != ''
                                     for col in question_cols)
                    if not has_answer:
                        missing_questions.append(question[:50] + "...")
                
                if missing_questions:
                    user_issues.append(f"rating类型缺少回答: {len(missing_questions)}个问题")
                    
            elif 'comparison' in instance_id:
                # 新版：比较问题也改为“问题本体 + ::: + 具体选项”（五选项）
                questions = [
                    'Which summary is more representative of your perspective?',
                    'Which summary is more informative?', 
                    'Which summary presents a more neutral and balanced view of the issue?',
                    'Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?'
                ]
                
                missing_questions = []
                for question in questions:
                    question_cols = [c for c in df.columns if str(c).startswith(question)]
                    has_answer = any(not pd.isna(row.get(col, '')) and str(row.get(col, '')).strip() != ''
                                     for col in question_cols)
                    if not has_answer:
                        missing_questions.append(question[:50] + "...")
                
                if missing_questions:
                    user_issues.append(f"comparison类型缺少回答: {len(missing_questions)}个问题")
        
        # 判断用户是否完整
        if user_issues:
            csv_incomplete_users.append(user_id)
            if len(csv_incomplete_users) <= 10:  # 只显示前10个不完整用户的详细信息
                print(f"❌ 用户 {user_id}: {'; '.join(user_issues)}")
        else:
            complete_users.append(user_id)
            if len(complete_users) <= 5:  # 只显示前5个完整用户
                print(f"✅ 用户 {user_id}: 完整")
    
    # 3. 合并所有不完整用户
    all_incomplete_users = list(set(assigned_only_users + csv_incomplete_users))
    
    csv_users = len(df['user'].unique())
    total_assigned_users = csv_users + len(assigned_only_users)
    complete_count = len(complete_users)
    incomplete_count = len(all_incomplete_users)
    
    print(f"\n总结:")
    print(f"- 被分配任务的总用户数: {total_assigned_users}")
    print(f"- CSV中的用户数: {csv_users}")
    print(f"- 只被分配未开始的用户数: {len(assigned_only_users)}")
    print(f"- CSV中不完整的用户数: {len(csv_incomplete_users)}")
    print(f"- 完整用户数: {complete_count} ({complete_count/total_assigned_users*100:.1f}%)")
    print(f"- 总不完整用户数: {incomplete_count} ({incomplete_count/total_assigned_users*100:.1f}%)")
    
    if all_incomplete_users:
        print(f"\n所有不完整用户列表 (前20个):")
        for i, user in enumerate(all_incomplete_users[:20]):
            user_type = "未开始" if user in assigned_only_users else "CSV中不完整"
            print(f"  {i+1:2d}. {user} ({user_type})")
        if len(all_incomplete_users) > 20:
            print(f"  ... 还有 {len(all_incomplete_users)-20} 个用户")
    
    return all_incomplete_users

def save_incomplete_users(incomplete_users):
    """保存不完整用户列表到文件"""
    if not incomplete_users:
        print("没有不完整用户需要保存")
        return None
        
    user_file_path = './annotation/summary-rating/notebooks/incomplete_users.txt'
    
    with open(user_file_path, 'w') as f:
        for user in incomplete_users:
            f.write(str(user) + '\n')
    
    print(f"\n不完整用户列表已保存到: {user_file_path}")
    print(f"共 {len(incomplete_users)} 个用户需要移除")
    
    return user_file_path

def main():
    """主函数"""
    print("检查用户标注完成情况")
    print("="*60)
    
    try:
        # 检查用户完成情况
        incomplete_users = check_user_completeness()
        
        # 保存不完整用户列表
        user_file_path = save_incomplete_users(incomplete_users)
        
        if user_file_path:
            print(f"\n下一步可以运行:")
            print(f"python3 clean_incomplete_users.py")
            print(f"来移除这些不完整的用户")
        else:
            print(f"\n✅ 所有用户都已完成标注!")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()