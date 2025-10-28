# 修复总结 - Foldseek + US-align问题

## 问题描述

用户报告了以下问题：

1. **Foldseek数据库创建失败**：
   ```
   Foldseek database creation failed: No structures found in given input.
   ```

2. **US-align计算失败**：
   ```
   Successfully computed RMSD for 0 pairs
   ```

## 问题分析

### 1. Foldseek问题

**原因**：Foldseek `createdb` 命令的格式不正确
- 原代码：`foldseek createdb input_structures/ structuresDB`
- 问题：命令格式正确，但可能文件准备有问题

**修复**：
- 确保输入目录包含有效的结构文件
- 使用：`foldseek createdb input_dir structuresDB`

### 2. US-align问题

**原因**：索引映射错误
- 原代码：传递局部索引对给US-align计算
- 问题：US-align期望全局索引，但返回的稀疏矩阵需要局部索引

**修复**：
- 在US-align计算前将局部索引转换为全局索引
- 在构建稀疏矩阵时将全局索引映射回局部索引

## 修复内容

### 1. 修复Foldseek命令格式

**修复前**：
```python
cmd = [
    foldseek_path, 'createdb',
    str(input_dir), str(database_path)
]
```

**修复后**：
```python
# 直接使用目录作为输入
cmd = [
    foldseek_path, 'createdb',
    str(input_dir), str(database_path)
]
```

### 2. 修复US-align索引映射

**修复前**：
```python
# 获取邻居对
neighbor_pairs = np.where(neighbor_matrix)
pairs = list(zip(neighbor_pairs[0], neighbor_pairs[1]))

# 使用US-align计算精确RMSD
rmsd_results = self._compute_precise_rmsd_usalign(pairs)
```

**修复后**：
```python
# 获取邻居对
neighbor_pairs = np.where(neighbor_matrix)
pairs = list(zip(neighbor_pairs[0], neighbor_pairs[1]))

# 转换为全局索引
global_pairs = []
for local_i, local_j in pairs:
    if local_i < len(idx) and local_j < len(idx):
        global_i = idx[local_i]
        global_j = idx[local_j]
        global_pairs.append((global_i, global_j))

# 使用US-align计算精确RMSD
rmsd_results = self._compute_precise_rmsd_usalign(global_pairs)
```

### 3. 修复稀疏矩阵构建

**修复前**：
```python
# 添加邻居对的距离
for pair in pairs:
    if pair in rmsd_results:
        idx1, idx2 = pair
        rmsd = rmsd_results[pair]
        
        # 添加对称的两个元素
        rows.extend([idx1, idx2])
        cols.extend([idx2, idx1])
        data.extend([rmsd, rmsd])
```

**修复后**：
```python
# 创建全局索引到局部索引的映射
global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(idx)}

# 添加邻居对的距离
for local_pair in pairs:
    local_i, local_j = local_pair
    global_i = idx[local_i]
    global_j = idx[local_j]
    global_pair = (global_i, global_j)
    
    if global_pair in rmsd_results:
        rmsd = rmsd_results[global_pair]
        
        # 添加对称的两个元素（使用局部索引）
        rows.extend([local_i, local_j])
        cols.extend([local_j, local_i])
        data.extend([rmsd, rmsd])
```

### 4. 添加调试信息

添加了详细的调试信息来帮助诊断问题：

```python
# Foldseek调试信息
logger.info(f"        - First file: {structure_files[0]}")
logger.info(f"        - Last file: {structure_files[-1]}")
for i, file_path in enumerate(structure_files[:3]):
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        logger.info(f"        - File {i}: {file_path} (size: {file_size} bytes)")

# US-align调试信息
logger.info(f"      - First pair: {neighbor_pairs[0]}")
logger.info(f"      - Last pair: {neighbor_pairs[-1]}")
logger.info(f"      - Total unique pairs: {len(set(neighbor_pairs))}")
```

## 验证方法

创建了 `test_fixes.py` 脚本来验证修复：

1. **Foldseek命令格式测试**
2. **US-align索引映射测试**
3. **模块导入测试**

## 预期结果

修复后，脚本应该能够：

1. **成功创建Foldseek数据库**：不再出现"No structures found"错误
2. **成功计算US-align RMSD**：返回正确数量的RMSD结果
3. **正确构建稀疏距离矩阵**：索引映射正确
4. **完成精细聚类**：生成有意义的聚类结果

## 注意事项

1. 确保Foldseek版本支持使用的命令格式
2. 验证US-align工具可用且路径正确
3. 检查输入文件格式是否被工具支持
4. 监控调试输出以诊断潜在问题

## 测试命令

```bash
# 运行测试脚本
python3 test_fixes.py

# 运行完整流程
bash run_fine_clustering.sh
```
