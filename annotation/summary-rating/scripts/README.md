# Process Raw Data Scripts

将notebook `process_raw_data.ipynb` 转换为Python脚本。

## 脚本版本

### process_raw_data.py (平衡算法)
使用平衡算法，每个摘要至少比较6次，生成约45个配对/组。

### process_raw_data_ring.py (环形算法)
使用环形算法，每个摘要作为A恰好出现3次，生成精确45个配对/组。

## 使用方法

```bash
# 平衡算法 - 处理所有数据格式
./process_raw_data.py

# 环形算法 - 处理所有数据格式  
./process_raw_data_ring.py

# 指定输入文件和输出目录
./process_raw_data_ring.py -i ../data_files/raw/summaries_V0903_for_humanstudy_detail.csv -o ../data_files/processed/

# 只处理配对数据 (环形算法)
./process_raw_data_ring.py --skip-rating --skip-triplet

# 查看帮助
./process_raw_data_ring.py --help
```

## 输出文件

### 平衡算法输出:
- `sum_humanstudy_rating_full.csv` - 评分格式数据
- `sum_humanstudy_pair_full.csv` - 配对比较格式数据  
- `sum_humanstudy_triplet_full.csv` - 三元组格式数据

### 环形算法输出:
- `sum_humanstudy_rating_full_ring.csv` - 评分格式数据
- `sum_humanstudy_pair_full_ring.csv` - 配对比较格式数据
- `sum_humanstudy_triplet_full_ring.csv` - 三元组格式数据

## 算法对比

### 平衡算法 (process_raw_data.py)
- 每个摘要作为A或B至少出现6次
- 使用贪心算法平衡分配
- 总配对数约为45个/组，可能有小幅波动

### 环形算法 (process_raw_data_ring.py)  
- 每个摘要作为A恰好出现3次
- 使用环形结构：Qi → Q(i+1), Q(i+2), Q(i+3)
- 总配对数精确为45个/组 (15×3=45)
- 更对称，无反向重复配对
