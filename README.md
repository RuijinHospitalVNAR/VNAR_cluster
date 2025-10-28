# 预测结构聚类与界面分析工具集

本仓库用于对100+以上的大量蛋白复合物结构进行聚类（cluster），以比较不同复合物之间的相互作用界面差异，并提供可复用的脚本与管线。

## 功能概览
- **批量聚类**：支持KMeans、DBSCAN、HDBSCAN、谱聚类等多种方法
- **降维与可视化**：PCA、t-SNE、UMAP；统一使用自然配色
- **结构解析**：支持PDB/mmCIF，依赖BioPython
- **两阶段细化**：提供coarse→fine两阶段聚类脚本
- **示例与测试**：含`protein_clustering/`模块化实现与`2STEP/`示例管线

## 环境安装
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 快速开始
- 单脚本快速聚类（示例）：
```bash
python optimized_protein_clustering_v14.3.py \
  --input_dir path/to/structures \
  --ext .cif \
  --out_dir results
```
- 模块化管线：
```bash
python -m protein_clustering.pipeline \
  --config path/to/config.json
```
- 两阶段细化：参考 `2STEP/README_FINE_CLUSTERING.md` 与 `2STEP/run_fine_clustering.*`

## 接口（相互作用界面）聚类统一入口
使用 contact map 进行复合物界面聚类：
```bash
python cluster_interfaces.py \
  --cif_dir path/to/cifs \
  --antibody_chain A \
  --antigen_chains B C \
  --cutoff 5.0 \
  --method hdbscan \
  --out_dir results_interfaces \
  --save_plots --save_numpy
```
或使用配置文件：
```bash
python cluster_interfaces.py --cif_dir $(jq -r .cif_dir config_interface_clustering.json)
```
配置模板见 `config_interface_clustering.json`。

输出内容：`labels.txt`、`metrics.json`、`files.txt`、`tsne.png`（以及可选的 `X.npy`）。

## 目录结构
- `interface_clustering/`：统一的contact map生成、聚类与可视化
- `protein_clustering/`：模块化的分析、评估、可视化
- `2STEP/`：两阶段聚类与免疫原性优化示例管线
- `optimized_protein_clustering_v14.*.py`：单文件版本的迭代实现
- `Cluster_pipeline_v1.2.py` 等：历史版本与对比脚本

## 数据与输出
- 输入：PDB 或 mmCIF 结构目录
- 输出：聚类标签、评估指标、可视化图（PCA/t-SNE/UMAP）

## 颜色偏好
绘图默认使用自然配色：`['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']`

## 许可证
MIT
