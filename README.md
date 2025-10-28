# 预测结构聚类与界面分析工具集（两段式方案）

本仓库面向对100+以上的大量蛋白复合物结构进行聚类（cluster），以比较不同复合物之间的相互作用界面差异。现统一采用“两段式聚类”：先进行粗聚类（KMeans），再在各粗聚类内部进行细聚类（HDBSCAN）。

## 功能概览
- **两段式聚类**：Coarse(KMeans) → Fine(HDBSCAN)
- **降维与可视化**：t-SNE 可视化聚类结果（自然配色）
- **结构解析**：支持 PDB/mmCIF，依赖 BioPython

## 环境安装
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 使用方法（两段式统一入口）
```bash
python two_stage_interfaces.py \
  --cif_dir path/to/cifs \
  --antibody_chain A \
  --antigen_chains B C \
  --cutoff 5.0 \
  --k_coarse 10 \
  --min_cluster_size 10 \
  --out_dir results_two_stage \
  --save_plots --save_numpy
```
或使用配置模板 `config_two_stage_interface.json`。

输出：`labels_coarse.txt`、`labels_fine.txt`、`metrics.json`、`files.txt`、`tsne_coarse.png`、`tsne_fine.png`（以及可选的 `X.npy`）。

## 最小可复现实例
- Windows PowerShell：
```powershell
./examples/run_example.ps1
```
- Linux/macOS：
```bash
bash examples/run_example.sh
```
- Python：
```bash
python examples/run_minimal_example.py
```
脚本会下载少量公开 mmCIF 到 `examples/cifs/`，并运行两段式聚类生成结果 `examples/results/`。

## 目录结构
- `interface_clustering/`：contact map 生成、聚类与可视化
- `two_stage_interfaces.py`：两段式聚类统一入口
- `examples/`：最小示例与一键运行脚本

## 颜色偏好
绘图默认使用自然配色：`['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']`

## 许可证
MIT
