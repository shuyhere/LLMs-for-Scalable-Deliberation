#!/usr/bin/env python3
"""
清理脚本：根据 incomplete_users.txt 移除不完整的用户
1. 读取 incomplete_users.txt 文件
2. 从 annotated_instances.csv 中移除这些用户的数据
3. 归档用户目录到 archived_users
4. 更新 task_assignment.json
"""

import pandas as pd
import os
import json
import shutil

def load_incomplete_users():
    """从文件中加载不完整用户列表"""
    user_file_path = '/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/notebooks/incomplete_users.txt'
    
    if not os.path.exists(user_file_path):
        print(f"错误: 找不到用户列表文件 {user_file_path}")
        print("请先运行 check_user_completeness.py 生成用户列表")
        return []
    
    users = []
    with open(user_file_path, 'r') as f:
        for line in f:
            user = line.strip()
            if user:
                users.append(user)
    
    print(f"从 {user_file_path} 加载了 {len(users)} 个不完整用户")
    return users

def remove_users_from_data(users):
    """从数据中移除指定用户"""
    print(f"\n开始移除 {len(users)} 个不完整用户...")
    
    if not users:
        print("没有需要移除的用户")
        return False
        
    print(f"准备移除的用户 (前10个):")
    for i, user in enumerate(users[:10]):
        print(f"  {i+1}. {user}")
    if len(users) > 10:
        print(f"  ... 还有 {len(users)-10} 个用户")
    
    user_set = set(users)  # 转换为集合以提高查找效率
    
    # 设置路径
    task_assignment_path = '/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full/task_assignment.json'
    annotation_data_dir = '/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full'
    annotation_data_path = os.path.join(annotation_data_dir, "annotated_instances.csv")
    
    # 用户确认
    print(f"\n即将执行以下操作:")
    print(f"1. 从 annotated_instances.csv 中移除用户数据")
    print(f"2. 将用户目录移动到 archived_users")
    print(f"3. 更新 task_assignment.json")
    
    response = input("\n确认执行? (y/N): ").strip().lower()
    if response != 'y':
        print("操作已取消")
        return False
    
    try:
        # 1. 从CSV文件中移除用户标注
        print("1. 从标注文件中移除用户数据...")
        if os.path.exists(annotation_data_path):
            annotated_df = pd.read_csv(annotation_data_path)
            original_count = len(annotated_df)
            new_annotated_df = annotated_df[~annotated_df['user'].isin(users)]
            removed_count = original_count - len(new_annotated_df)
            print(f"   移除了 {removed_count} 条标注记录")
            
            new_annotation_path = annotation_data_path + '_new'
            new_annotated_df.to_csv(new_annotation_path, index=False)
            print(f"   新文件保存为: {new_annotation_path}")
        else:
            print("   annotated_instances.csv 不存在，跳过此步骤")
        
        # 2. 移动用户目录到archived_users
        print("2. 归档用户目录...")
        bad_user_dir = os.path.join(annotation_data_dir, "archived_users")
        if not os.path.exists(bad_user_dir):
            os.makedirs(bad_user_dir)
        
        archived_count = 0
        for user in users:
            user_dir = os.path.join(annotation_data_dir, user)
            if os.path.exists(user_dir):
                shutil.move(user_dir, os.path.join(bad_user_dir, user))
                archived_count += 1
                if archived_count <= 10:  # 只显示前10个
                    print(f"   归档用户目录: {user}")
        
        print(f"   共归档了 {archived_count} 个用户目录到 {bad_user_dir}")
        
        # 3. 从任务分配中移除用户
        if os.path.exists(task_assignment_path):
            print("3. 从任务分配中移除用户...")
            
            with open(task_assignment_path, "r") as f:
                task_assignment = json.load(f)
            
            removed_assignments = 0
            for inst_id in task_assignment.get('assigned', {}):
                if isinstance(task_assignment['assigned'][inst_id], list):
                    new_list = []
                    for u in task_assignment['assigned'][inst_id]:
                        if u in user_set:
                            removed_assignments += 1
                            if 'unassigned' not in task_assignment:
                                task_assignment['unassigned'] = {}
                            if inst_id not in task_assignment['unassigned']:
                                task_assignment['unassigned'][inst_id] = 0
                            task_assignment['unassigned'][inst_id] += 1
                        else:
                            new_list.append(u)
                    task_assignment['assigned'][inst_id] = new_list
            
            new_task_assignment_path = task_assignment_path + '_new'
            with open(new_task_assignment_path, "w") as f:
                json.dump(task_assignment, f, indent=2)
            
            print(f"   移除了 {removed_assignments} 个任务分配")
            print(f"   新任务分配文件保存为: {new_task_assignment_path}")
            unassigned_count = sum(task_assignment.get('unassigned', {}).values())
            print(f"   未分配实例总数: {unassigned_count}")
        else:
            print("3. 任务分配文件不存在，跳过此步骤")
        
        print("✅ 移除操作成功完成！")
        print(f"\n后续操作建议:")
        print(f"1. 检查生成的新文件是否正确")
        print(f"2. 如果确认无误，可以替换原文件:")
        print(f"   mv {annotation_data_path}_new {annotation_data_path}")
        print(f"   mv {task_assignment_path}_new {task_assignment_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 移除操作失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("清理不完整用户数据")
    print("="*60)
    
    try:
        # Step 1: 加载不完整用户列表
        incomplete_users = load_incomplete_users()
        
        if not incomplete_users:
            print("没有找到需要清理的用户")
            return
        
        # Step 2: 移除不完整用户
        success = remove_users_from_data(incomplete_users)
        
        if success:
            print(f"\n✅ 成功完成用户清理操作!")
        else:
            print(f"\n❌ 用户清理操作失败!")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
