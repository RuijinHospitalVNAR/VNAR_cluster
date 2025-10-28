# Foldseek + US-align 集成指南

## 概述

本指南介绍如何在 `2T_AF3_Cluster_v1.py` 中集成真正的 Foldseek 和 US-align 工具，以实现更准确和高效的蛋白质结构聚类。

## 工具介绍

### Foldseek
- **功能**：快速蛋白质结构相似性搜索
- **优势**：比传统方法快1000-10000倍
- **搜索类型**：
  - `3di`：3D相互作用（推荐）
  - `ca`：Cα原子距离
  - `tm`：TM-score

### US-align
- **功能**：精确的蛋白质结构对齐和RMSD计算
- **优势**：比传统方法更准确
- **输出**：TM-score和RMSD

## 安装指南

### 1. 安装 Foldseek

#### Linux/macOS
```bash
# 方法1：使用conda（推荐）
conda install -c conda-forge foldseek

# 方法2：从源码编译
git clone https://github.com/steineggerlab/foldseek.git
cd foldseek
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. ..
make -j $(nproc)
make install
export PATH=$PATH:$(pwd)/bin
```

#### Windows
```bash
# 使用WSL2或conda
conda install -c conda-forge foldseek
```

### 2. 安装 US-align

#### Linux/macOS
```bash
# 下载预编译版本
wget https://zhanggroup.org/US-align/US-align
chmod +x US-align
sudo mv US-align /usr/local/bin/

# 或者从源码编译
git clone https://github.com/pylelab/US-align.git
cd US-align
make
sudo mv US-align /usr/local/bin/
```

#### Windows
```bash
# 使用WSL2或conda
conda install -c conda-forge us-align
```

### 3. 验证安装

```bash
# 验证 Foldseek
foldseek version

# 验证 US-align
US-align -h
```

## 配置使用

### 1. 基本配置

```python
CLUSTERING_CONFIG = {
    'coarse_method': 'hdbscan',
    'coarse_params': {'min_cluster_size': 'auto'},
    'fine_params': {
        'use_foldseek_usalign': True,      # 启用 Foldseek + US-align
        'k_neighbors': 50,                 # 每个结构找50个近邻
        'clustering_method': 'hdbscan',    # 聚类方法
        'max_clusters_to_refine': 3,       # 处理前3个最大簇
        
        # Foldseek 参数
        'foldseek_path': 'foldseek',       # Foldseek路径
        'search_type': '3di',              # 搜索类型
        'sensitivity': 4,                  # 敏感度
        'max_evalue': 1e-3,               # 最大E值
        
        # US-align 参数
        'usalign_path': 'US-align',        # US-align路径
        'use_usalign': True                # 使用US-align
    }
}
```

### 2. 高级配置

```python
# 针对不同数据集的优化配置
fine_params = {
    # 小数据集（<100结构）
    'small_dataset': {
        'use_foldseek_usalign': True,
        'k_neighbors': 30,
        'search_type': '3di',
        'sensitivity': 5,
        'clustering_method': 'hdbscan'
    },
    
    # 中等数据集（100-500结构）
    'medium_dataset': {
        'use_foldseek_usalign': True,
        'k_neighbors': 50,
        'search_type': '3di',
        'sensitivity': 4,
        'clustering_method': 'hdbscan'
    },
    
    # 大数据集（>500结构）
    'large_dataset': {
        'use_foldseek_usalign': True,
        'k_neighbors': 100,
        'search_type': '3di',
        'sensitivity': 3,
        'clustering_method': 'spectral'
    }
}
```

## 工作流程

### 1. 快速近邻搜索（Foldseek）
```
输入：结构文件列表
↓
Foldseek 3D相互作用搜索
↓
输出：每个结构的前K个近邻
```

### 2. 精确距离计算（US-align）
```
输入：近邻对列表
↓
US-align 精确对齐和RMSD计算
↓
输出：稀疏距离矩阵
```

### 3. 稀疏图聚类
```
输入：稀疏距离矩阵
↓
HDBSCAN/谱聚类
↓
输出：精细聚类标签
```

## 性能对比

### 计算复杂度
| 方法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|------------|------------|----------|
| 传统iRMSD | O(n²) | O(n²) | 小数据集 |
| 快速近邻 | O(n×k) | O(n×k) | 中等数据集 |
| Foldseek+US-align | O(n×log(n)) | O(n×k) | 大数据集 |

### 实际性能提升
- **500个结构**：
  - 传统方法：124,750个iRMSD计算
  - Foldseek+US-align：25,000个US-align计算
  - **性能提升**：80%计算量减少

- **1000个结构**：
  - 传统方法：499,500个iRMSD计算
  - Foldseek+US-align：50,000个US-align计算
  - **性能提升**：90%计算量减少

## 故障排除

### 1. Foldseek 问题

#### 问题：找不到 foldseek 命令
```bash
# 解决方案
which foldseek
# 如果找不到，添加到PATH
export PATH=$PATH:/path/to/foldseek/bin
```

#### 问题：Foldseek 搜索失败
```python
# 检查参数
fine_params = {
    'foldseek_path': '/full/path/to/foldseek',
    'search_type': '3di',
    'sensitivity': 4,
    'max_evalue': 1e-3
}
```

### 2. US-align 问题

#### 问题：US-align 超时
```python
# 增加超时时间或减少并行数
fine_params = {
    'usalign_path': '/full/path/to/US-align',
    'n_jobs': 4  # 减少并行数
}
```

#### 问题：US-align 输出解析错误
```python
# 检查输出格式
# US-align 输出：TM-score1 TM-score2 RMSD
```

### 3. 内存问题

#### 问题：内存不足
```python
# 减少批处理大小
fine_params = {
    'k_neighbors': 30,  # 减少邻居数
    'max_cluster_size_for_refine': 100  # 减少簇大小限制
}
```

## 最佳实践

### 1. 参数调优

#### 根据数据集大小调整K值
```python
def get_optimal_k(cluster_size):
    if cluster_size < 50:
        return 20
    elif cluster_size < 200:
        return 50
    else:
        return 100
```

#### 根据结构复杂度调整敏感度
```python
def get_optimal_sensitivity(avg_residues):
    if avg_residues < 100:
        return 5  # 高敏感度
    elif avg_residues < 300:
        return 4  # 中等敏感度
    else:
        return 3  # 低敏感度
```

### 2. 混合策略

```python
def adaptive_clustering_strategy(cluster_size, avg_residues):
    if cluster_size < 50:
        return {
            'use_foldseek_usalign': False,
            'use_fast_neighbor_clustering': True
        }
    elif cluster_size < 200:
        return {
            'use_foldseek_usalign': True,
            'k_neighbors': 50,
            'search_type': '3di'
        }
    else:
        return {
            'use_foldseek_usalign': True,
            'k_neighbors': 100,
            'search_type': 'ca'  # 使用更快的搜索类型
        }
```

### 3. 结果验证

```python
# 比较不同方法的结果
def compare_clustering_methods():
    methods = {
        'traditional': {'use_foldseek_usalign': False},
        'fast_neighbor': {'use_fast_neighbor_clustering': True},
        'foldseek_usalign': {'use_foldseek_usalign': True}
    }
    
    results = {}
    for name, config in methods.items():
        # 运行聚类
        results[name] = run_clustering(config)
    
    return results
```

## 总结

集成 Foldseek + US-align 可以显著提高大规模蛋白质结构聚类的效率和准确性：

1. **性能提升**：80-90%的计算量减少
2. **准确性提升**：使用专业的结构对齐工具
3. **可扩展性**：支持大规模数据集
4. **灵活性**：支持多种搜索和聚类策略

建议根据数据集大小和计算资源选择合适的配置参数。
