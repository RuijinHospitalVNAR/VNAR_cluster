# 命名规则说明 - AF3精细聚类

## 概述

为了更好的文件管理和结果追踪，`run_fine_clustering.sh` 脚本现在使用基于聚类方法、日期和时间的动态命名规则。

## 命名格式

### 输出目录命名
```
af3_fine_clustering_{聚类方法}_{日期}_{时间}
```

**格式说明：**
- `af3_fine_clustering`: 固定前缀
- `{聚类方法}`: 聚类算法名称（hdbscan, spectral）
- `{日期}`: YYYYMMDD 格式（如：20250827）
- `{时间}`: HHMMSS 格式（如：143022）

**示例：**
- `af3_fine_clustering_hdbscan_20250827_143022`
- `af3_fine_clustering_spectral_20250827_143022`

### 日志文件命名
```
logs_{日期}/fine_clustering_run_{时间}.log
```

**格式说明：**
- `logs_{日期}`: 日志目录，按日期分组
- `fine_clustering_run_{时间}.log`: 日志文件名，包含运行时间

**示例：**
- `logs_20250827/fine_clustering_run_143022.log`

## 配置参数

### 聚类方法配置
```bash
# 在脚本中设置
CLUSTERING_METHOD="hdbscan"  # 可选: 'hdbscan', 'spectral'
```

### 时间格式
```bash
# 自动获取当前时间
CURRENT_DATE=$(date +%Y%m%d)  # 20250827
CURRENT_TIME=$(date +%H%M%S)  # 143022
```

## 优势

### 📁 **文件组织**
- **按日期分组**: 日志文件按日期自动分组到不同目录
- **按方法区分**: 不同聚类方法的结果分开存储
- **时间戳追踪**: 精确到秒的时间戳，避免文件覆盖

### 🔍 **结果追踪**
- **运行历史**: 可以追踪每次运行的详细信息
- **方法对比**: 便于比较不同聚类方法的结果
- **时间分析**: 可以分析不同时间点的聚类效果

### 🛠️ **管理便利**
- **自动命名**: 无需手动指定输出目录名称
- **唯一性**: 时间戳确保每次运行都有唯一的输出目录
- **可读性**: 命名格式清晰，便于理解

## 使用示例

### 运行脚本
```bash
# 使用默认配置（hdbscan）
bash run_fine_clustering.sh

# 修改聚类方法（需要编辑脚本）
# CLUSTERING_METHOD="spectral"
```

### 输出结构
```
当前目录/
├── af3_fine_clustering_hdbscan_20250827_143022/
│   ├── fine_clustering_results.pkl
│   ├── fine_clustering_results.csv
│   ├── fine_clusters/
│   └── visualizations/
├── af3_fine_clustering_spectral_20250827_143025/
│   ├── fine_clustering_results.pkl
│   ├── fine_clustering_results.csv
│   ├── fine_clusters/
│   └── visualizations/
└── logs_20250827/
    ├── fine_clustering_run_143022.log
    └── fine_clustering_run_143025.log
```

## 测试脚本

使用 `test_naming.sh` 脚本可以测试命名格式：

```bash
bash test_naming.sh
```

该脚本会：
1. 显示当前日期和时间
2. 为不同聚类方法生成示例目录名
3. 创建测试目录和文件
4. 显示命名格式示例

## 注意事项

1. **时间精度**: 时间戳精确到秒，确保每次运行都有唯一标识
2. **目录创建**: 脚本会自动创建所需的目录结构
3. **日志记录**: 每次运行都会在日志中记录聚类方法和时间信息
4. **向后兼容**: 如果未设置聚类方法，会使用默认值 "hdbscan"

## 自定义配置

如需修改命名格式，可以编辑脚本中的相关变量：

```bash
# 修改日期格式
CURRENT_DATE=$(date +%Y-%m-%d)  # 2025-08-27

# 修改时间格式
CURRENT_TIME=$(date +%H:%M:%S)  # 14:30:22

# 修改输出目录前缀
OUTPUT_DIR="my_fine_clustering_${CLUSTERING_METHOD}_${CURRENT_DATE}_${CURRENT_TIME}"
```
