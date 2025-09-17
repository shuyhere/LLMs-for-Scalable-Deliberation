#!/usr/bin/env python3
"""
修复任务分配脚本：将空的assigned任务移动到unassigned
处理 task_assignment.json 中assigned为空列表[]的任务
"""

import json
import os

def fix_empty_assignments():
    """修复空的assigned任务"""
    print("修复空的assigned任务...")
    print("="*50)
    
    # 设置路径
    task_assignment_path = '/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full_augment/task_assignment.json'
    
    if not os.path.exists(task_assignment_path):
        print(f"错误: 任务分配文件不存在: {task_assignment_path}")
        return False
    
    try:
        # 读取任务分配文件
        print("1. 读取任务分配文件...")
        with open(task_assignment_path, "r") as f:
            task_assignment = json.load(f)
        
        # 统计信息
        total_assigned = len(task_assignment.get('assigned', {}))
        original_unassigned = len(task_assignment.get('unassigned', {}))
        
        print(f"   原始assigned任务数: {total_assigned}")
        print(f"   原始unassigned任务数: {original_unassigned}")
        
        # 找出空的assigned任务
        print("2. 查找空的assigned任务...")
        empty_inst_ids = []
        
        for inst_id, assigned_users in task_assignment.get('assigned', {}).items():
            if isinstance(assigned_users, list) and len(assigned_users) == 0:
                empty_inst_ids.append(inst_id)
        
        if not empty_inst_ids:
            print("   没有找到空的assigned任务")
            return True
        
        print(f"   找到 {len(empty_inst_ids)} 个空的assigned任务")
        
        # 显示前10个空任务
        print("   空的assigned任务 (前10个):")
        for i, inst_id in enumerate(empty_inst_ids[:10]):
            print(f"     {i+1}. {inst_id}")
        if len(empty_inst_ids) > 10:
            print(f"     ... 还有 {len(empty_inst_ids)-10} 个")
        
        # 用户确认
        response = input(f"\n确认将这 {len(empty_inst_ids)} 个空任务移到unassigned? (y/N): ").strip().lower()
        if response != 'y':
            print("操作已取消")
            return False
        
        # 移动空的assigned任务到unassigned
        print("3. 移动空任务到unassigned...")
        
        if 'unassigned' not in task_assignment:
            task_assignment['unassigned'] = {}
        
        moved_count = 0
        for inst_id in empty_inst_ids:
            # 将任务添加到unassigned
            task_assignment['unassigned'][inst_id] = 1  # 表示有1个未分配的实例
            
            # 从assigned中删除
            del task_assignment['assigned'][inst_id]
            moved_count += 1
            
            if moved_count <= 10:  # 只显示前10个
                print(f"   移动任务: {inst_id}")
        
        print(f"   共移动了 {moved_count} 个任务")
        
        # 保存新的任务分配文件
        print("4. 保存修复后的文件...")
        new_task_assignment_path = task_assignment_path + '_fixed'
        
        with open(new_task_assignment_path, "w") as f:
            json.dump(task_assignment, f, indent=2)
        
        # 统计修复后的信息
        final_assigned = len(task_assignment.get('assigned', {}))
        final_unassigned = len(task_assignment.get('unassigned', {}))
        total_unassigned_count = sum(task_assignment.get('unassigned', {}).values())
        
        print(f"   修复后assigned任务数: {final_assigned}")
        print(f"   修复后unassigned任务数: {final_unassigned}")
        print(f"   未分配实例总数: {total_unassigned_count}")
        print(f"   新文件保存为: {new_task_assignment_path}")
        
        print("✅ 修复操作成功完成！")
        print(f"\n后续操作建议:")
        print(f"1. 检查生成的新文件是否正确")
        print(f"2. 如果确认无误，可以替换原文件:")
        print(f"   mv {new_task_assignment_path} {task_assignment_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复操作失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("修复空的assigned任务")
    print("="*60)
    
    try:
        success = fix_empty_assignments()
        
        if success:
            print(f"\n✅ 成功完成任务修复!")
        else:
            print(f"\n❌ 任务修复失败!")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
