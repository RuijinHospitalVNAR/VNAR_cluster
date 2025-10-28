#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
蛋白质复合物聚类分析脚本
功能：
1. 从 PDB/mmCIF 文件中提取抗体-抗原的 Cα 坐标
2. 计算接触图 (contact map) 和几何特征
3. 聚类 (支持 HDBSCAN, KMeans, DBSCAN)
4. 可视化聚类结果 (标签分布 / 簇大小 / t-SNE / 特征重要性)
"""

import os
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
import logging

from Bio.PDB import PDBParser, MMCIFParser  # Biopython 解析蛋白结构
import hdbscan  # 高级聚类算法 HDBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from matplotlib import patches

# ------------------------
# 日志配置
# ------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ProteinClusterAnalyzer:
    """
    蛋白质聚类分析器类
    - 输入：结构文件目录
    - 输出：聚类结果 + 可视化
    """

    def __init__(self, cif_dir, antibody_chain='A', antigen_chains=['B', 'C'],
                 dist_cutoff=5.0, n_jobs=-1):
        """
        初始化分析器

        参数：
        - cif_dir: 结构文件目录
        - antibody_chain: 抗体链ID
        - antigen_chains: 抗原链ID列表
        - dist_cutoff: 接触判断的距离阈值（Å）
        - n_jobs: 并行处理线程数（-1=自动检测CPU数）
        """
        self.cif_dir = Path(cif_dir)
        self.antibody_chain = antibody_chain
        self.antigen_chains = antigen_chains
        self.dist_cutoff = dist_cutoff
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        # HDBSCAN 参数（可调）
        self.min_cluster_size = 10
        self.min_samples = 5
        self.cluster_selection_epsilon = 0.1

        # 数据存储
        self.contact_maps = []     # 每个结构的接触图
        self.feature_vectors = []  # 每个结构的特征向量
        self.file_names = []       # 文件名
        self.cluster_labels = None # 聚类标签
        self.scaler = StandardScaler()  # 标准化器
        # 统一色卡（Nature 风格）
        self.palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # --------------------
    # 自适应参数估计
    # --------------------
    def _auto_hdbscan_params(self, n_samples: int, n_features: int):
        """
        基于样本数与维度的简单启发式估计 HDBSCAN 参数。
        - min_cluster_size: 约 n/20，在 [5, 100] 之间截断
        - min_samples: 约 log2(n)，不超过 min_cluster_size，且 >=3
        """
        min_cluster_size = max(5, min(100, max(1, n_samples // 20)))
        min_samples = max(3, min(min_cluster_size, int(np.log2(max(n_samples, 2))) + 1))
        return min_cluster_size, min_samples

    def _auto_dbscan_params(self, X: np.ndarray):
        """
        通过子采样 + k-距离中位数估计 eps；k = log2(n) + 1，min_samples = k。
        若估计失败，回退到 (eps=0.5, min_samples=5)。
        """
        try:
            n_samples = len(X)
            if n_samples < 10:
                return 0.5, 5
            k = max(3, int(np.log2(n_samples)) + 1)
            sample_size = min(1000, n_samples)
            rng = np.random.default_rng(42)
            idx = rng.choice(n_samples, size=sample_size, replace=False)
            S = X[idx]
            # 距离矩阵
            diffs = S[:, None, :] - S[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            # 排除自距离
            np.fill_diagonal(dists, np.inf)
            # 第 k 邻近距离的中位数作为 eps 估计
            kth = np.partition(dists, kth=k-1, axis=1)[:, k-1]
            eps = float(np.median(kth))
            eps = max(1e-6, eps)
            return eps, k
        except Exception as e:
            logger.warning(f"Auto DBSCAN parameter estimation failed: {e}")
            return 0.5, 5

    # ==========================================================
    # (1) 提取 Cα 坐标和残基信息
    # ==========================================================
    def extract_ca_coords_with_residues(self, structure_file, chain_ids):
        """
        提取指定链的 Cα 原子坐标和残基信息
        返回：(坐标数组, 残基信息列表, 链长度字典)
        """
        try:
            # 判断文件格式：PDB 或 mmCIF
            ext = str(structure_file).split('.')[-1].lower()
            parser = PDBParser(QUIET=True) if ext == "pdb" else MMCIFParser(QUIET=True)
            structure = parser.get_structure('model', structure_file)

            ca_coords, residue_infos = [], []
            chain_info, found_chains = {}, set()

            # 遍历所有链
            for model in structure:
                for chain in model:
                    if chain.id in chain_ids:
                        found_chains.add(chain.id)
                        chain_coords, chain_residues = [], []
                        for res in chain:
                            if 'CA' in res:  # 只取 Cα
                                coord = res['CA'].get_coord()
                                resname = res.get_resname().strip()
                                resid = res.id[1]
                                chain_coords.append(coord)
                                chain_residues.append((resname, resid, chain.id))
                        if chain_coords:
                            ca_coords.extend(chain_coords)
                            residue_infos.extend(chain_residues)
                            chain_info[chain.id] = len(chain_coords)

            # 检查是否缺少链
            missing = set(chain_ids) - found_chains
            if missing:
                logger.warning(f"Chains {missing} not found in {structure_file}")

            return np.array(ca_coords), residue_infos, chain_info
        except Exception as e:
            logger.warning(f"Error processing {structure_file}: {e}")
            return np.array([]), [], {}

    # ==========================================================
    # (2) 计算接触图
    # ==========================================================
    def compute_contact_map_with_residues(self, coords_ab, coords_ag, residues_ab, residues_ag):
        """
        输入抗体/抗原坐标，计算接触图
        返回：(接触图, 接触到的抗体残基, 接触到的抗原残基)
        """
        if len(coords_ab) == 0 or len(coords_ag) == 0:
            return np.array([]), [], []

        # 计算两两距离
        dists = np.linalg.norm(coords_ab[:, None, :] - coords_ag[None, :, :], axis=-1)
        contact_map = (dists < self.dist_cutoff).astype(np.float32)

        rows, cols = np.where(contact_map > 0)
        contact_ab_residues = [residues_ab[i][0] for i in rows]
        contact_ag_residues = [residues_ag[i][0] for i in cols]

        return contact_map, contact_ab_residues, contact_ag_residues

    # ==========================================================
    # (3) 提取几何特征
    # ==========================================================
    def extract_interaction_features(self, contact_map, contact_ab_residues, contact_ag_residues):
        """
        从接触图提取几何统计特征
        返回：固定长度的特征向量
        """
        if contact_map.size == 0:
            return np.zeros(15)

        features = [
            np.sum(contact_map), np.mean(contact_map), np.std(contact_map)  # 总数/均值/方差
        ]
        row_sums, col_sums = np.sum(contact_map, axis=1), np.sum(contact_map, axis=0)
        features.extend([
            np.mean(row_sums) if len(row_sums) else 0,
            np.std(row_sums) if len(row_sums) else 0,
            np.max(row_sums) if len(row_sums) else 0,
            np.mean(col_sums) if len(col_sums) else 0,
            np.std(col_sums) if len(col_sums) else 0,
            np.max(col_sums) if len(col_sums) else 0,
        ])

        if np.sum(contact_map) > 0:
            rows, cols = np.where(contact_map > 0)
            features.extend([
                np.std(rows) if len(rows) else 0,
                np.std(cols) if len(cols) else 0,
                len(np.unique(rows)), len(np.unique(cols))
            ])
        else:
            features.extend([0, 0, 0, 0])

        if contact_map.shape[0] > 1 and contact_map.shape[1] > 1:
            row_cont = np.sum(contact_map[:-1, :] * contact_map[1:, :])
            col_cont = np.sum(contact_map[:, :-1] * contact_map[:, 1:])
            features.extend([row_cont, col_cont])
        else:
            features.extend([0, 0])

        return np.array(features, dtype=np.float32)

    # ==========================================================
    # (4) 文件处理与缓存
    # ==========================================================
    def process_single_file(self, structure_file):
        """
        处理单个结构文件
        返回：(接触图flatten, 特征向量, 文件名)
        """
        try:
            ab_coords, ab_residues, _ = self.extract_ca_coords_with_residues(structure_file, [self.antibody_chain])
            ag_coords, ag_residues, _ = self.extract_ca_coords_with_residues(structure_file, self.antigen_chains)

            if len(ab_coords) == 0 or len(ag_coords) == 0:
                return None, None, None

            contact_map, ab_res, ag_res = self.compute_contact_map_with_residues(
                ab_coords, ag_coords, ab_residues, ag_residues)

            if contact_map.size == 0:
                return None, None, None

            features = self.extract_interaction_features(contact_map, ab_res, ag_res)
            return contact_map.flatten(), features, Path(structure_file).name
        except Exception as e:
            logger.error(f"Error processing {structure_file}: {e}")
            return None, None, None

    def load_and_process_data(self, cache_file="protein_data_cache.pkl"):
        """
        加载数据（支持缓存）
        - 若缓存文件存在，直接读取
        - 否则重新处理结构文件并存储缓存
        """
        if Path(cache_file).exists():
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            # 兼容两种缓存格式：
            # 1) 旧版: (contact_maps, feature_vectors, file_names)
            # 2) 新版: { 'contact_maps': ..., 'feature_vectors': ..., 'file_names': ... }
            if isinstance(cached, dict):
                self.contact_maps = cached.get("contact_maps", [])
                self.feature_vectors = cached.get("feature_vectors", [])
                self.file_names = cached.get("file_names", [])
            elif isinstance(cached, (list, tuple)) and len(cached) == 3:
                self.contact_maps, self.feature_vectors, self.file_names = cached
            else:
                logger.error(f"Unrecognized cache format in {cache_file}. Ignoring cache.")
                self.contact_maps, self.feature_vectors, self.file_names = [], [], []
                return False
            logger.info(f"Loaded cached data from {cache_file} (files: {len(self.file_names)})")
            if not self.file_names:
                logger.error("Cache is empty. Please remove cache and re-run processing.")
                return False
            return True

        # 校验输入目录
        if not self.cif_dir.exists() or not self.cif_dir.is_dir():
            try:
                abs_path = self.cif_dir.resolve()
            except Exception:
                abs_path = str(self.cif_dir)
            logger.error(f"Structure directory not found or not a directory: {abs_path}")
            return False

        # 仅处理结构文件
        structure_files = list(self.cif_dir.glob("*.cif")) + list(self.cif_dir.glob("*.pdb"))
        try:
            abs_path = self.cif_dir.resolve()
        except Exception:
            abs_path = str(self.cif_dir)
        logger.info(f"Found {len(structure_files)} structure files in {abs_path}")

        if len(structure_files) == 0:
            logger.error("No CIF or PDB files found. Please check the input directory.")
            return False

        # 并行处理结构文件
        with Pool(processes=self.n_jobs) as pool:
            results = list(pool.imap_unordered(self.process_single_file, structure_files))

        # 收集有效结果
        for cmap, feat, fname in results:
            if feat is not None:
                self.contact_maps.append(cmap)
                self.feature_vectors.append(feat)
                self.file_names.append(fname)

        if len(self.file_names) == 0:
            logger.error("No valid results obtained from provided structure files.")
            return False

        with open(cache_file, "wb") as f:
            pickle.dump((self.contact_maps, self.feature_vectors, self.file_names), f)
        logger.info(f"Processed and cached {len(self.file_names)} files")
        return True

    # ==========================================================
    # (5) 特征预处理
    # ==========================================================
    def prepare_features(self, feature_type='combined', use_pca=True, n_components=20):
        """
        标准化 + 可选PCA降维
        """
        if not self.feature_vectors:
            logger.error("No engineered features available. Did data loading/processing fail?")
            return np.array([])

        X = np.array(self.feature_vectors)
        if X.size == 0:
            logger.error("Feature vector list is empty after conversion.")
            return np.array([])
        X_scaled = self.scaler.fit_transform(X)
        if use_pca and X.shape[1] > n_components:
            pca = PCA(n_components=n_components, random_state=42)
            X_scaled = pca.fit_transform(X_scaled)
        return X_scaled

    # ==========================================================
    # (6) 聚类方法 (可切换)
    # ==========================================================
    def perform_clustering(self, X, method='hdbscan', **kwargs):
        """
        执行聚类
        支持方法：hdbscan / kmeans / dbscan
        """
        n_samples = len(X)
        n_features = X.shape[1] if X.ndim == 2 else 1

        if method == 'hdbscan':
            # 自适应参数（可被 kwargs 覆盖）
            auto_min_cluster_size, auto_min_samples = self._auto_hdbscan_params(n_samples, n_features)
            min_cluster_size = kwargs.get('min_cluster_size', auto_min_cluster_size)
            min_samples = kwargs.get('min_samples', auto_min_samples)
            cluster_selection_epsilon = kwargs.get('cluster_selection_epsilon', self.cluster_selection_epsilon)
            logger.info(f"HDBSCAN params -> min_cluster_size={min_cluster_size}, min_samples={min_samples}, eps={cluster_selection_epsilon}")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                core_dist_n_jobs=self.n_jobs,
                **{k: v for k, v in kwargs.items() if k not in ['min_cluster_size','min_samples','cluster_selection_epsilon']}
            )
            self.cluster_labels = clusterer.fit_predict(X)

        elif method == 'kmeans':
            n_clusters = kwargs.get("n_clusters")
            if n_clusters is None:
                # 自动选择簇数（silhouette 最大）
                max_k = min(20, max(2, n_samples // 5))
                best_k, best_score = None, -1.0
                for k in range(2, max_k + 1):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(X)
                    # 需要至少 2 个簇且每簇至少有 2 个点
                    if len(set(labels)) > 1 and min(np.bincount(labels)) >= 2:
                        try:
                            score = silhouette_score(X, labels)
                            if score > best_score:
                                best_k, best_score = k, score
                        except Exception:
                            pass
                n_clusters = best_k if best_k is not None else 5
                logger.info(f"Auto-selected KMeans n_clusters={n_clusters} (best silhouette={best_score:.4f})")
            else:
                logger.info(f"KMeans n_clusters={n_clusters} (provided)")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = clusterer.fit_predict(X)

        elif method == 'dbscan':
            eps = kwargs.get("eps")
            min_samples = kwargs.get("min_samples")
            if eps is None or min_samples is None:
                auto_eps, auto_min_samples = self._auto_dbscan_params(X)
                eps = auto_eps if eps is None else eps
                min_samples = auto_min_samples if min_samples is None else min_samples
                logger.info(f"Auto DBSCAN params -> eps={eps:.6f}, min_samples={min_samples}")
            else:
                logger.info(f"DBSCAN params -> eps={eps}, min_samples={min_samples} (provided)")
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.n_jobs)
            self.cluster_labels = clusterer.fit_predict(X)
        else:
            raise ValueError(f"Unsupported clustering method {method}")
        return self.cluster_labels

    # ==========================================================
    # (7) 聚类评估
    # ==========================================================
    def evaluate_clustering(self, X):
        """
        使用轮廓系数 (silhouette) 评估聚类质量
        """
        if self.cluster_labels is None or len(set(self.cluster_labels)) <= 1:
            return {"silhouette": None}
        score = silhouette_score(X, self.cluster_labels)
        return {"silhouette": score}

    # ==========================================================
    # (8) 保存结果
    # ==========================================================
    def save_results(self, filename, X):
        results = {
            "file_names": self.file_names,
            "labels": self.cluster_labels,
            "features": X
        }
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {filename}")

        # 同步保存CSV摘要（文件名与聚类标签）
        try:
            csv_path = Path(filename).with_suffix('.csv')
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(["file_name", "label"])
                for fn, lb in zip(self.file_names, self.cluster_labels):
                    writer.writerow([fn, lb])
            logger.info(f"CSV summary saved to {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save CSV summary: {e}")

    # ==========================================================
    # (9) 获取簇代表
    # ==========================================================
    def get_cluster_representatives(self, X):
        """
        获取每个簇的代表性结构（最靠近簇中心的样本）
        """
        reps = {}
        labels = np.unique(self.cluster_labels)
        for label in labels:
            if label == -1:  # -1 代表噪声
                continue
            indices = np.where(self.cluster_labels == label)[0]
            cluster_feats = X[indices]
            center = cluster_feats.mean(axis=0)
            dists = np.linalg.norm(cluster_feats - center, axis=1)
            sorted_idx = indices[np.argsort(dists)]
            reps[label] = {
                "size": len(indices),
                "files": [self.file_names[i] for i in sorted_idx[:5]],
                "distances": [dists[j] for j in np.argsort(dists)[:5]]
            }
        return reps

    # ==========================================================
    # (10) 可视化
    # ==========================================================
    def visualize_results(self, X, save_path=None, show_plot=False):
        """
        生成 4 个子图：
        1. 聚类标签分布
        2. 簇大小分布
        3. t-SNE 可视化
        4. 特征重要性
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering results found.")

        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Protein Complex Clustering Results', fontsize=16)

        unique_labels = np.unique(self.cluster_labels)
        # 颜色映射：非噪声按色卡，噪声(-1)为黑色
        non_noise_labels = [l for l in unique_labels if l != -1]
        label_to_color = {l: self.palette[i % len(self.palette)] for i, l in enumerate(non_noise_labels)}
        if -1 in unique_labels:
            label_to_color[-1] = '#000000'

        # 1. 聚类标签分布
        ax1 = axes[0, 0]
        for label in unique_labels:
            color = label_to_color[label]
            mask = self.cluster_labels == label
            indices = np.where(mask)[0]
            ax1.scatter(indices, [label] * len(indices),
                        c=[color], alpha=0.7, s=30,
                        label=f'Cluster {label}' if label != -1 else 'Noise')
        ax1.set_title('Cluster Assignment')

        # 2. 聚类大小分布（手风琴图，横轴显示 1..K）
        ax2 = axes[0, 1]
        cluster_ids = [l for l in unique_labels if l != -1]
        cluster_sizes = [int(np.sum(self.cluster_labels == l)) for l in cluster_ids]
        if cluster_sizes:
            x_ticks = list(range(1, len(cluster_ids) + 1))
            # 手风琴：每个簇绘制一个矩形条，条宽相同，高度为样本数，条之间留薄间隔
            total_w = len(cluster_ids)
            bar_w = 0.8
            for i, (cid, sz) in enumerate(zip(cluster_ids, cluster_sizes), start=1):
                rect = patches.Rectangle((i - bar_w/2, 0), bar_w, sz,
                                         facecolor=label_to_color.get(int(cid), self.palette[int(cid) % len(self.palette)]),
                                         edgecolor='none', alpha=0.9)
                ax2.add_patch(rect)
            ax2.set_xlim(0.5, len(cluster_ids) + 0.5)
            ax2.set_ylim(0, max(cluster_sizes) * 1.05)
            ax2.set_xticks(x_ticks)
            ax2.set_xticklabels([str(i) for i in x_ticks])
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Size')
            ax2.set_title('Cluster Size (Accordion)')

        # 3. t-SNE 可视化
        ax3 = axes[1, 0]
        if len(X) > 1:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(X)-1)))
            X_tsne = tsne.fit_transform(X)
            for label in unique_labels:
                color = label_to_color[label]
                mask = self.cluster_labels == label
                ax3.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[color], alpha=0.7, s=30)
            ax3.set_title('t-SNE Visualization')

        # 4. 特征重要性（方差排名）
        ax4 = axes[1, 1]
        feature_matrix = np.array(self.feature_vectors)
        feature_variance = np.var(feature_matrix, axis=0)
        sorted_indices = np.argsort(feature_variance)[::-1][:15]
        # 定义特征名称（15个工程特征）
        default_feature_names = [
            'Total_Contacts', 'Contact_Density', 'Contact_Variability',
            'Ab_Mean_Contacts', 'Ab_Std_Contacts', 'Ab_Max_Contacts',
            'Ag_Mean_Contacts', 'Ag_Std_Contacts', 'Ag_Max_Contacts',
            'Ab_Region_Dispersion', 'Ag_Region_Dispersion',
            'Ab_Residue_Count', 'Ag_Residue_Count',
            'Ab_Continuity', 'Ag_Continuity'
        ]
        if feature_matrix.shape[1] == len(default_feature_names):
            feature_names = default_feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(feature_matrix.shape[1])]
        ax4.barh(range(len(sorted_indices)), feature_variance[sorted_indices], color="skyblue")
        ax4.set_yticks(range(len(sorted_indices)))
        ax4.set_yticklabels([feature_names[i] for i in sorted_indices])
        ax4.set_title('Top 15 Feature Importance')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        # 导出单图 SVG 与数据 CSV
        if save_path:
            base_dir = Path(save_path).with_suffix("").parent
            base_name = Path(save_path).stem.replace("_clustering", "")
            # 目录
            fig_dir = base_dir / "figures"
            data_dir = base_dir / "data"
            fig_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            # 1) Cluster Assignment 单图 + CSV
            fig1, ax = plt.subplots(figsize=(8, 6))
            for label in unique_labels:
                color = label_to_color[label]
                mask = self.cluster_labels == label
                indices = np.where(mask)[0]
                ax.scatter(indices, [label] * len(indices), c=[color], alpha=0.7, s=30)
            ax.set_title('Cluster Assignment')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Cluster Label')
            fig1.savefig(fig_dir / f"{base_name}_cluster_assignment.svg", format='svg', bbox_inches='tight')
            plt.close(fig1)
            # CSV
            assignment_csv = data_dir / f"{base_name}_cluster_assignment.csv"
            with open(assignment_csv, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(["index", "label"])
                for i, lb in enumerate(self.cluster_labels):
                    writer.writerow([i, int(lb)])

            # 2) Cluster Size 单图 + CSV
            fig2, ax = plt.subplots(figsize=(8, 6))
            bar_colors = [label_to_color.get(int(cid), self.palette[int(cid) % len(self.palette)]) for cid in cluster_ids]
            ax.bar([str(cid) for cid in cluster_ids], cluster_sizes, color=bar_colors)
            ax.set_title('Cluster Size Distribution')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Size')
            fig2.savefig(fig_dir / f"{base_name}_cluster_sizes.svg", format='svg', bbox_inches='tight')
            plt.close(fig2)
            size_csv = data_dir / f"{base_name}_cluster_sizes.csv"
            with open(size_csv, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(["cluster_id", "size"])
                for cid, sz in zip(cluster_ids, cluster_sizes):
                    writer.writerow([int(cid), int(sz)])

            # 3) t-SNE 单图 + CSV
            if len(X) > 1:
                fig3, ax = plt.subplots(figsize=(8, 6))
                for label in unique_labels:
                    color = label_to_color[label]
                    mask = self.cluster_labels == label
                    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.7, s=30, label=str(label), c=color)
                ax.set_title('t-SNE Visualization')
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.legend()
                fig3.savefig(fig_dir / f"{base_name}_tsne.svg", format='svg', bbox_inches='tight')
                plt.close(fig3)
                tsne_csv = data_dir / f"{base_name}_tsne.csv"
                with open(tsne_csv, 'w', newline='') as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["index", "tsne1", "tsne2", "label"])
                    for i in range(len(X)):
                        writer.writerow([i, float(X_tsne[i,0]), float(X_tsne[i,1]), int(self.cluster_labels[i])])

            # 4) Feature Importance 单图 + CSV
            fig4, ax = plt.subplots(figsize=(8, 6))
            ax.barh(range(len(sorted_indices)), feature_variance[sorted_indices], color=self.palette[0])
            ax.set_title('Top 15 Feature Importance')
            ax.set_xlabel('Variance')
            ax.set_yticks(range(len(sorted_indices)))
            ax.set_yticklabels([feature_names[i] for i in sorted_indices])
            fig4.savefig(fig_dir / f"{base_name}_feature_importance.svg", format='svg', bbox_inches='tight')
            plt.close(fig4)
            feat_csv = data_dir / f"{base_name}_feature_importance.csv"
            with open(feat_csv, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(["feature", "variance"])
                for i in sorted_indices:
                    writer.writerow([feature_names[i], float(feature_variance[i])])

    def plot_cluster_radial_tree(self, X, save_path=None, show_plot=False):
        """
        基于聚类质心的层次聚类，绘制径向树图（参考图片风格）。
        - 对每个聚类（排除 -1）计算质心
        - 对质心做层次聚类（ward）
        - 将树在极坐标下径向展开，优化角度分布和簇标签
        同时导出 SVG 和链接矩阵 CSV。
        """
        labels = np.array(self.cluster_labels)
        valid_mask = labels != -1
        if not np.any(valid_mask):
            logger.warning("No valid clusters to plot radial tree.")
            return
        
        # 计算每个聚类质心
        unique_labels = np.unique(labels[valid_mask])
        centroids = []
        cluster_sizes = []
        for lab in unique_labels:
            cluster_mask = labels == lab
            centroids.append(X[cluster_mask].mean(axis=0))
            cluster_sizes.append(np.sum(cluster_mask))
        centroids = np.vstack(centroids)

        # 层次聚类
        Z = linkage(centroids, method='ward')
        
        # 生成dendrogram获取叶节点顺序
        fig_temp, ax_temp = plt.subplots(figsize=(1, 1))
        dendro = dendrogram(Z, orientation='right', no_labels=True, ax=ax_temp)
        plt.close(fig_temp)
        
        leaves = dendro['leaves']
        n_clusters = len(leaves)
        
        # 优化角度分布：从90度开始，逆时针排列，确保簇间角度均匀
        start_angle = np.pi/2  # 90度开始
        angles = np.linspace(start_angle, start_angle - 2*np.pi, n_clusters, endpoint=False)
        
        # 创建角度到标签的映射
        angle_to_label = {angles[i]: unique_labels[leaves[i]] for i in range(n_clusters)}
        
        # 设置随机种子确保节点分布一致
        np.random.seed(42)
        
        # 绘制径向树
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # 设置极坐标网格
        ax.set_rticks([])  # 隐藏半径刻度
        ax.set_thetagrids([])  # 隐藏角度刻度
        ax.grid(False)
        
        # 绘制连接线（从中心到叶节点）
        center_radius = 0.1
        leaf_radius = 0.8
        
        for i, angle in enumerate(angles):
            # 连接线
            ax.plot([0, angle], [center_radius, leaf_radius], 
                   color='#666666', linewidth=1.5, alpha=0.7)
            
            # 根据簇大小计算节点数量（1-10个节点）
            label = angle_to_label[angle]
            cluster_size = cluster_sizes[leaves[i]]
            color = self.palette[label % len(self.palette)]
            
            # 计算节点数量：根据簇大小比例，最少1个，最多10个
            max_cluster_size = max(cluster_sizes)
            if max_cluster_size > 0:
                node_count = max(1, min(10, int(round(cluster_size / max_cluster_size * 10))))
            else:
                node_count = 1
            
            # 统一节点大小
            node_size = 80
            
            # 在角度周围分布多个节点
            if node_count == 1:
                # 单个节点在中心
                ax.scatter([angle], [leaf_radius], c=[color], s=node_size, 
                          edgecolors='white', linewidth=2, zorder=5, alpha=0.9)
            else:
                # 多个节点在角度周围分布
                angle_spread = 0.2  # 角度分布范围（弧度）
                radius_spread = 0.03  # 半径分布范围
                
                # 计算节点分布
                if node_count <= 3:
                    # 少量节点：线性分布
                    for j in range(node_count):
                        offset = (j - (node_count-1)/2) * angle_spread / max(1, node_count-1)
                        node_angle = angle + offset
                        node_radius = leaf_radius + (np.random.random() - 0.5) * radius_spread * 0.5
                        
                        ax.scatter([node_angle], [node_radius], c=[color], s=node_size, 
                                  edgecolors='white', linewidth=2, zorder=5, alpha=0.9)
                else:
                    # 多节点：网格分布
                    grid_size = int(np.ceil(np.sqrt(node_count)))
                    for j in range(node_count):
                        row = j // grid_size
                        col = j % grid_size
                        
                        # 在网格中计算相对位置
                        rel_x = (col - (grid_size-1)/2) / max(1, grid_size-1)
                        rel_y = (row - (grid_size-1)/2) / max(1, grid_size-1)
                        
                        # 转换为极坐标偏移
                        node_angle = angle + rel_x * angle_spread
                        node_radius = leaf_radius + rel_y * radius_spread
                        
                        ax.scatter([node_angle], [node_radius], c=[color], s=node_size, 
                                  edgecolors='white', linewidth=2, zorder=5, alpha=0.9)
            
            # 添加簇标签
            label_text = f'Cluster {label} ({cluster_size})'
            # 调整文本位置避免重叠
            text_angle = np.degrees(angle)
            if text_angle > 90 and text_angle < 270:
                text_angle += 180
                ha = 'right'
            else:
                ha = 'left'
            
            ax.text(angle, leaf_radius + 0.15, label_text, 
                   ha=ha, va='center', fontsize=10, fontweight='bold',
                   color=color, transform=ax.transData)
        
        # 中心节点
        ax.scatter([0], [0], c='#333333', s=200, zorder=6, alpha=0.8)
        
        # 设置显示范围
        ax.set_rlim(0, 1.2)
        
        if save_path:
            fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
            logger.info(f"Radial tree saved to {save_path}")
            
            # 导出链接矩阵和簇信息CSV
            csv_path = Path(save_path).with_suffix('.csv')
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(["cluster_id", "size", "angle_degrees", "x_coord", "y_coord"])
                for i, angle in enumerate(angles):
                    label = angle_to_label[angle]
                    x_coord = leaf_radius * np.cos(angle)
                    y_coord = leaf_radius * np.sin(angle)
                    writer.writerow([int(label), int(cluster_sizes[leaves[i]]), 
                                   float(np.degrees(angle)), float(x_coord), float(y_coord)])
            
            # 导出层次聚类链接矩阵
            linkage_csv = Path(save_path).with_suffix('_linkage.csv')
            with open(linkage_csv, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(["idx1", "idx2", "distance", "sample_count"])
                for r in Z:
                    writer.writerow([int(r[0]), int(r[1]), float(r[2]), int(r[3])])
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


# ==========================================================
# (11) 主函数入口
# ==========================================================
def main():
    CIF_DIR = "./your_structure_dir"
    analyzer = ProteinClusterAnalyzer(CIF_DIR, antibody_chain='A', antigen_chains=['B','C'], dist_cutoff=5.0)
    logger.info("Starting analysis pipeline ...")
    try:
        ok = analyzer.load_and_process_data()
        if not ok:
            logger.error("Data loading/processing produced no results. Please check CIF_DIR and input files.")
            return

        X = analyzer.prepare_features()
        if X.size == 0:
            logger.error("Feature matrix is empty. Aborting pipeline.")
            return

        # 👉 选择聚类方法：hdbscan / kmeans / dbscan
        # 在此统一控制三种算法是否使用自适应参数或手动参数
        cluster_method = "kmeans"  # 可选: "hdbscan" | "kmeans" | "dbscan"

        # 统一的参数控制开关：auto=True 走自适应；auto=False 使用下方给定参数
        cluster_params = {
            "kmeans": {
                "auto": True,       # True: 自动选择簇数；False: 使用 n_clusters
                "n_clusters": 4
            },
            "hdbscan": {
                "auto": True,       # True: 自适应 min_cluster_size/min_samples；False: 使用下方设定
                "min_cluster_size": 10,
                "min_samples": 5,
                "cluster_selection_epsilon": 0.1
            },
            "dbscan": {
                "auto": True,       # True: 自动估计 eps/min_samples；False: 使用下方设定
                "eps": 0.5,
                "min_samples": 5
            }
        }

        # 构造传入 perform_clustering 的 kwargs
        kwargs = {}
        if cluster_method == "kmeans":
            if not cluster_params["kmeans"]["auto"]:
                kwargs["n_clusters"] = cluster_params["kmeans"]["n_clusters"]
                logger.info(f"KMeans manual params -> n_clusters={kwargs['n_clusters']}")
            else:
                logger.info("KMeans auto mode -> auto-select n_clusters by silhouette")
        elif cluster_method == "hdbscan":
            if not cluster_params["hdbscan"]["auto"]:
                kwargs["min_cluster_size"] = cluster_params["hdbscan"]["min_cluster_size"]
                kwargs["min_samples"] = cluster_params["hdbscan"]["min_samples"]
                kwargs["cluster_selection_epsilon"] = cluster_params["hdbscan"]["cluster_selection_epsilon"]
                logger.info(
                    f"HDBSCAN manual params -> min_cluster_size={kwargs['min_cluster_size']}, "
                    f"min_samples={kwargs['min_samples']}, eps={kwargs['cluster_selection_epsilon']}"
                )
            else:
                logger.info("HDBSCAN auto mode -> estimate min_cluster_size/min_samples")
        elif cluster_method == "dbscan":
            if not cluster_params["dbscan"]["auto"]:
                kwargs["eps"] = cluster_params["dbscan"]["eps"]
                kwargs["min_samples"] = cluster_params["dbscan"]["min_samples"]
                logger.info(f"DBSCAN manual params -> eps={kwargs['eps']}, min_samples={kwargs['min_samples']}")
            else:
                logger.info("DBSCAN auto mode -> estimate eps/min_samples")

        analyzer.perform_clustering(X, method=cluster_method, **kwargs)

        # 构建输出目录与文件命名：输入文件夹名 + cluster方法 + 日期
        input_base = Path(CIF_DIR).resolve().name
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{input_base}_{cluster_method}_{date_str}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 评估 + 保存 + 输出代表
        metrics = analyzer.evaluate_clustering(X)
        logger.info(f"Clustering metrics: {metrics}")
        results_pkl = output_dir / f"{input_base}_{cluster_method}_{date_str}_results.pkl"
        analyzer.save_results(results_pkl, X)
        reps = analyzer.get_cluster_representatives(X)
        logger.info(f"Representatives: {reps}")

        # 可视化
        viz_path = output_dir / f"{input_base}_{cluster_method}_{date_str}_clustering.png"
        analyzer.visualize_results(X, save_path=viz_path, show_plot=False)

        # 径向树图（第二张图风格）
        radial_svg = output_dir / f"{input_base}_{cluster_method}_{date_str}_radial_tree.svg"
        analyzer.plot_cluster_radial_tree(X, save_path=radial_svg, show_plot=False)
        logger.info("All tasks finished. Exiting.")
    except Exception:
        logger.exception("Analysis failed with an unexpected error")


if __name__ == "__main__":
    main()
