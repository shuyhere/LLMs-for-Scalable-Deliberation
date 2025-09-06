#!/usr/bin/env python3
"""
检查每个用户是否完成了三个问题类型的标注：question、rating、comparison
"""

import pandas as pd

# 加载标注数据
annotation_path = '/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full/annotated_instances.csv'
df = pd.read_csv(annotation_path)

print("检查用户标注完成情况...")
print("="*50)

# 提取实例类型
df['instance_type'] = df['instance_id'].apply(
    lambda x: 'comparison' if 'comparison' in x 
    else 'rating' if 'rating' in x 
    else 'question' if 'question' in x
    else 'unknown'
)

print(f"总标注记录数: {len(df)}")
print(f"总用户数: {df['user'].nunique()}")

# 按用户和实例类型统计
user_type_counts = df.groupby(['user', 'instance_type']).size().unstack(fill_value=0)

print(f"\n实例类型分布:")
for instance_type in ['question', 'rating', 'comparison', 'unknown']:
    if instance_type in user_type_counts.columns:
        count = user_type_counts[instance_type].sum()
        print(f"  {instance_type}: {count} 个标注")

print(f"\n用户完成情况检查:")

# 检查每个用户是否完成了所有三种类型
complete_users = []
incomplete_users = []

for user in user_type_counts.index:
    user_counts = user_type_counts.loc[user]
    
    has_question = user_counts.get('question', 0) > 0
    has_rating = user_counts.get('rating', 0) > 0  
    has_comparison = user_counts.get('comparison', 0) > 0
    
    if has_question and has_rating and has_comparison:
        complete_users.append(user)
        total_annotations = user_counts.sum()
        print(f"✅ {user}: 完整 (question:{user_counts.get('question', 0)}, rating:{user_counts.get('rating', 0)}, comparison:{user_counts.get('comparison', 0)}) 总计:{total_annotations}")
    else:
        incomplete_users.append(user)
        missing = []
        if not has_question: missing.append('question')
        if not has_rating: missing.append('rating')
        if not has_comparison: missing.append('comparison')
        total_annotations = user_counts.sum()
        print(f"❌ {user}: 不完整 - 缺少:{missing} (question:{user_counts.get('question', 0)}, rating:{user_counts.get('rating', 0)}, comparison:{user_counts.get('comparison', 0)}) 总计:{total_annotations}")

print(f"\n总结:")
print(f"- 完整用户数: {len(complete_users)} / {len(user_type_counts)} ({len(complete_users)/len(user_type_counts)*100:.1f}%)")
print(f"- 不完整用户数: {len(incomplete_users)} / {len(user_type_counts)} ({len(incomplete_users)/len(user_type_counts)*100:.1f}%)")

# 检查是否每个用户恰好回答3个问题
print(f"\n详细检查每个用户的标注数量:")
user_total_counts = df['user'].value_counts()

exactly_3_users = []
not_3_users = []

for user, count in user_total_counts.items():
    if count == 3:
        exactly_3_users.append(user)
    else:
        not_3_users.append((user, count))

print(f"- 恰好3个标注的用户: {len(exactly_3_users)} 个")
print(f"- 非3个标注的用户: {len(not_3_users)} 个")

if not_3_users:
    print(f"\n非3个标注的用户详情:")
    for user, count in not_3_users:
        print(f"  {user}: {count} 个标注")

if len(exactly_3_users) == len(user_total_counts):
    print(f"\n✅ 所有用户都恰好完成了3个标注！")
else:
    print(f"\n⚠️ 有 {len(not_3_users)} 个用户的标注数量不是3个")
