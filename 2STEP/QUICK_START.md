# AF3精细聚类快速开始指南

## 简化配置方案

现在所有配置都已硬编码在shell脚本中，使用更加简单！

## 快速开始

### 1. 修改配置路径

编辑 `run_fine_clustering.sh` 文件，修改以下配置：

```bash
# 外部工具路径配置（请根据实际情况修改）
FOLDSEEK_CMD="/mnt/share/public/foldseek/bin/foldseek"
USALIGN_CMD="/mnt/share/public/USalign"

# 默认配置（请根据实际情况修改）
COARSE_RESULTS_FILE="your_coarse_results.pkl"
PDB_DIR="./your_structures"
COARSE_CLUSTERS_DIR="your_coarse_clusters"
OUTPUT_DIR="your_fine_results"
N_JOBS="4"
```

### 2. 运行脚本

```bash
# 给脚本执行权限
chmod +x run_fine_clustering.sh

# 运行精细聚类
./run_fine_clustering.sh

# 仅检查配置（不运行）
./run_fine_clustering.sh -d

# 详细输出模式
./run_fine_clustering.sh -v
```

## 优势

✅ **极简配置**：所有配置集中在一个脚本中  
✅ **零配置文件**：无需额外的配置文件  
✅ **减少错误**：避免配置文件路径错误  
✅ **快速部署**：只需修改几个路径即可使用  
✅ **易于维护**：所有设置一目了然  

## 注意事项

- 所有配置路径都在脚本开头，修改非常方便
- 脚本会自动检测工具是否在指定路径存在
- 如果工具不在指定路径，会尝试从PATH中查找
- 如果都找不到，会使用备用方法继续运行
- 使用 `-d` 参数可以检查所有配置是否正确
- 日志文件会自动保存在按日期命名的文件夹中（如 `logs_0826/`）

## 故障排除

### biopython导入问题

如果遇到biopython导入错误，请运行测试脚本：

```bash
python3 test_biopython.py
```

如果测试失败，请确保biopython已正确安装：

```bash
pip install biopython
# 或者
conda install biopython
```
