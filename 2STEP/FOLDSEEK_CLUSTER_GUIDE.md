# Foldseek + US-align精细聚类指南

## 概述

本指南介绍如何使用Foldseek + US-align进行蛋白质结构精细聚类。该方案采用三步骤的聚类策略，结合了Foldseek的快速搜索能力和US-align的精确计算。

## 聚类方案

采用三步骤的精细聚类方案：

1. **Foldseek快速近邻搜索**: 使用Foldseek为每个结构找到前K个近邻（K=50-100）
2. **US-align精确RMSD计算**: 仅在近邻对上使用US-align计算精确的RMSD距离
3. **稀疏图聚类**: 基于稀疏距离矩阵使用HDBSCAN或谱聚类进行聚类

## 主要优势

1. **高效性**: 只计算近邻对的RMSD，避免全距离矩阵计算
2. **精确性**: 使用US-align计算精确的RMSD，而非近似距离
3. **可扩展性**: 稀疏图方法可以处理大规模数据集
4. **原生格式支持**: Foldseek直接支持CIF和PDB格式，无需格式转换

## 使用方法

### 1. 基本使用

```bash
# 运行精细聚类脚本
bash run_fine_clustering.sh
```

### 2. 配置参数

在`run_fine_clustering.sh`中可以修改以下参数：

```bash
# 精细聚类配置
CLUSTERING_METHOD="hdbscan"  # 聚类方法: 'hdbscan' 或 'spectral'
K_NEIGHBORS="50"            # Foldseek近邻搜索的K值
```

### 3. 聚类方法选项

- `hdbscan`: 使用HDBSCAN算法（推荐，自动确定聚类数）
- `spectral`: 使用光谱聚类（需要指定聚类数）

### 4. 主要参数说明

- **k_neighbors**: 每个结构的前K个近邻，值越大计算量越大但结果更准确
- **clustering_method**: 聚类算法选择
- **max_clusters_to_refine**: 要精细聚类的最大粗聚类数量
- **max_cluster_size_for_refine**: 单个粗聚类的最大大小限制

## 工作流程

1. **准备结构文件**: 从粗聚类结果中提取需要精细聚类的结构（支持CIF和PDB格式）
2. **Foldseek近邻搜索**: 使用`foldseek search`为每个结构找到K个最近邻
3. **US-align精确计算**: 对近邻对使用US-align计算精确RMSD
4. **稀疏图聚类**: 使用HDBSCAN或谱聚类在稀疏距离矩阵上进行聚类
5. **结果输出**: 输出精细聚类标签和可视化结果

## 优势

1. **更高效**: 只计算近邻对的RMSD，避免全距离矩阵计算
2. **更精确**: 使用US-align计算精确的RMSD，而非近似距离
3. **更可扩展**: 稀疏图方法可以处理大规模数据集
4. **更稳定**: 结合了Foldseek的快速搜索和US-align的精确计算

## 故障排除

### 常见问题

1. **"No structures found"错误**
   - 确保结构文件格式正确（PDB或CIF格式）
   - 检查文件是否包含有效的结构数据
   - 验证文件路径是否正确

2. **Foldseek搜索失败**
   - 检查Foldseek版本和参数兼容性
   - 确保输入目录包含有效的结构文件
   - 验证文件权限和磁盘空间

3. **US-align计算失败**
   - 检查US-align是否正确安装
   - 验证结构文件格式是否被US-align支持
   - 调整超时参数

4. **聚类结果不理想**
   - 调整`k_neighbors`参数
   - 尝试不同的聚类方法（HDBSCAN vs 谱聚类）
   - 检查输入结构的多样性

## 示例输出

```
2025-08-26 17:30:15,123 [INFO] - Step 1: Foldseek neighbor search (k=50)
2025-08-26 17:30:15,456 [INFO] - Found 245 neighbor pairs via Foldseek
2025-08-26 17:30:16,789 [INFO] - Step 2: Computing US-align RMSD for neighbor pairs
2025-08-26 17:30:20,012 [INFO] - Successfully computed RMSD for 245 pairs
2025-08-26 17:30:20,345 [INFO] - Step 3: Sparse graph clustering using hdbscan
2025-08-26 17:30:21,678 [INFO] - Fine clustering complete: 8 clusters, 3 noise points
```

## 注意事项

1. 确保有足够的磁盘空间用于临时文件
2. 对于大型数据集，可能需要调整超时设置
3. 建议在运行前备份重要数据
4. US-align计算可能比较耗时，请耐心等待
