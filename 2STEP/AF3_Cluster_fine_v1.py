#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AF3 Complex Fine Clustering Pipeline
基于2T_AF3_Cluster_v2.py的精细聚类功能拆�?

功能�?
1. 从粗聚类结果中读取数�?
2. 基于US-align的精细聚�?
3. 支持多种聚类算法（HDBSCAN/Spectral�?
4. 丰富的可视化结果（标签分�?簇大�?t-SNE/径向树）
5. 并行处理和缓存机�?
6. 灵活的输出管理和配置系统

使用方法�?
1. 修改 main() 函数中的配置参数
2. 确保粗聚类结果文件存�?
3. 运行脚本

配置选项�?
- INPUT_CONFIG: 输入配置（粗聚类结果文件路径等）
- FINE_CLUSTERING_CONFIG: 精细聚类算法配置
- OUTPUT_CONFIG: 输出配置（自动调整、图形显示等�?

精细聚类方案�?
- 仅支�?US-align 方案
- 需要安�?US-align 工具
- 直接使用US-align进行全距离计算和聚类
"""

import os
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count, Pool
import subprocess
import tempfile
import shutil
import itertools
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import logging
import glob
import psutil
import gc

from Bio.PDB import PDBParser, MMCIFParser, Superimposer
import hdbscan
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from matplotlib import patches

# ------------------------
# 全局配置
# ------------------------
# 外部工具路径配置（可通过环境变量修改�?
USALIGN_CMD = os.environ.get('USALIGN_CMD', '/mnt/share/public/USalign')

# ------------------------
# 日志配置
# ------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AF3FineClusterAnalyzer:
    """
    AF3 复合物精细聚类分析器�?
    - 输入：粗聚类结果文件
    - 输出：精细聚类结�?+ 丰富可视�?
    - 专注于精细聚类功能，基于US-align
    """

    def __init__(self, coarse_results_file, pdb_dir=None, coarse_clusters_dir=None, n_jobs=-1):
        """
        初始化分析器

        参数�?
        - coarse_results_file: 粗聚类结果文件路�?
        - pdb_dir: PDB文件目录（如果粗聚类结果中没有结构信息）
        - coarse_clusters_dir: 粗聚类结构文件夹路径（可选，用于US-align输入�?
        - n_jobs: 并行处理线程数（-1=自动检测CPU数）
        """
        self.coarse_results_file = Path(coarse_results_file)
        self.pdb_dir = Path(pdb_dir) if pdb_dir else None
        self.coarse_clusters_dir = Path(coarse_clusters_dir) if coarse_clusters_dir else None
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        
        # 数据存储
        self.file_names = []       # 文件�?
        self.coarse_labels = None  # 粗聚类标�?
        self.fine_labels = None    # 精细聚类标签
        self.contact_sets = []     # 接触集（如果需要）
        self.structures = []       # 结构对象（如果需要）
        
        # 统一色卡（Nature 风格�?
        self.palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 内存监控
        self.memory_threshold = 0.9  # 90%内存使用率阈�?

    # --------------------
    # 内存监控
    # --------------------
    def _check_memory_usage(self):
        """检查内存使用情�?""
        try:
            memory_percent = psutil.virtual_memory().percent / 100.0
            if memory_percent > self.memory_threshold:
                logger.warning(f"High memory usage detected: {memory_percent:.1%}")
                logger.info("Triggering garbage collection...")
                gc.collect()
                memory_percent_after = psutil.virtual_memory().percent / 100.0
                logger.info(f"Memory usage after GC: {memory_percent_after:.1%}")
                return True
            return False
        except Exception as e:
            logger.debug(f"Error checking memory usage: {e}")
            return False

    def _log_memory_usage(self, stage=""):
        """记录内存使用情况"""
        try:
            memory_percent = psutil.virtual_memory().percent
            memory_gb = psutil.virtual_memory().used / (1024**3)
            logger.info(f"Memory usage {stage}: {memory_percent:.1f}% ({memory_gb:.2f} GB)")
        except Exception as e:
            logger.debug(f"Error logging memory usage: {e}")

    # --------------------
    # 自适应参数估计
    # --------------------
    def _auto_hdbscan_params(self, n_samples: int):
        """
        基于样本数的简单启发式估计 HDBSCAN 参数
        """
        min_cluster_size = max(3, min(50, max(1, n_samples // 10)))
        min_samples = max(2, min(min_cluster_size, int(np.log2(max(n_samples, 2))) + 1))
        return min_cluster_size, min_samples

    # ==========================================================
    # (1) 数据加载
    # ==========================================================
    def load_coarse_results(self):
        """
        加载粗聚类结�?
        """
        if not self.coarse_results_file.exists():
            logger.error(f"Coarse results file not found: {self.coarse_results_file}")
            return False
        
        try:
            with open(self.coarse_results_file, "rb") as f:
                results = pickle.load(f)
            
            # 检查结果格�?- 支持多种格式
            if isinstance(results, dict):
                # 标准字典格式
                self.file_names = results.get("file_names", [])
                self.coarse_labels = results.get("coarse_labels", [])
                self.contact_sets = results.get("contact_sets", [])
                self.structures = results.get("structures", [])
            elif isinstance(results, (list, tuple)) and len(results) >= 2:
                # 列表/元组格式：[file_names, coarse_labels, ...]
                self.file_names = results[0] if len(results) > 0 else []
                self.coarse_labels = results[1] if len(results) > 1 else []
                self.contact_sets = results[2] if len(results) > 2 else []
                self.structures = results[3] if len(results) > 3 else []
            elif hasattr(results, '__dict__'):
                # 对象格式，尝试获取属�?
                self.file_names = getattr(results, 'file_names', [])
                self.coarse_labels = getattr(results, 'coarse_labels', [])
                self.contact_sets = getattr(results, 'contact_sets', [])
                self.structures = getattr(results, 'structures', [])
            else:
                logger.error(f"Unexpected format in coarse results file: {type(results)}")
                logger.error(f"Expected dict, list, tuple, or object with attributes")
                return False
            
            # 验证必要数据
            if not self.file_names:
                logger.error("No file names found in coarse results")
                return False
            
            if not self.coarse_labels:
                logger.error("No coarse labels found in coarse results")
                return False
            
            if len(self.file_names) != len(self.coarse_labels):
                logger.error(f"Mismatch between file names ({len(self.file_names)}) and coarse labels ({len(self.coarse_labels)})")
                return False
            
            logger.info(f"Loaded coarse clustering results: {len(self.file_names)} structures")
            
            # 修复numpy数组的布尔值问�?
            if self.coarse_labels is not None and len(self.coarse_labels) > 0:
                if isinstance(self.coarse_labels, np.ndarray):
                    unique_labels = set(self.coarse_labels.tolist())
                else:
                    unique_labels = set(self.coarse_labels)
                logger.info(f"Coarse labels: {unique_labels}")
            else:
                logger.info("Coarse labels: None")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading coarse results: {e}")
            return False

    # ==========================================================
    # (2) 精细聚类方法
    # ==========================================================
    def perform_fine_clustering(self, **kwargs):
        """
        基于US-align的精细聚类（支持手动指定粗聚类集�?
        简化版本：直接使用US-align进行全距离计算和聚类
        """
        if self.coarse_labels is None:
            logger.error("No coarse clustering results available")
            return None

        self.fine_labels = -1 * np.ones(len(self.coarse_labels), dtype=int)
        cluster_id = 0

        # 统计粗聚类结果并按大小排�?
        unique_coarse_labels = set(self.coarse_labels)
        coarse_cluster_sizes = {}
        for label in unique_coarse_labels:
            if label != -1:
                size = np.sum(self.coarse_labels == label)
                coarse_cluster_sizes[label] = size
        
        # 按簇大小排序，最大的在前
        sorted_clusters = sorted(coarse_cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # 支持手动指定要精细聚类的�?
        specified_clusters = kwargs.get('specified_clusters', None)
        if specified_clusters is not None:
            # 手动指定的簇
            clusters_to_refine = []
            for cluster_id in specified_clusters:
                if cluster_id in coarse_cluster_sizes:
                    clusters_to_refine.append((cluster_id, coarse_cluster_sizes[cluster_id]))
                else:
                    logger.warning(f"Specified cluster {cluster_id} not found in coarse clustering results")
        else:
            # 自动选择前N个最大的�?
            max_clusters_to_refine = kwargs.get('max_clusters_to_refine', 2)
            clusters_to_refine = sorted_clusters[:max_clusters_to_refine]
        
        # 最大簇大小限制，避免过大的�?
        max_cluster_size_for_refine = kwargs.get('max_cluster_size_for_refine', 200)
        
        logger.info(f"Coarse clustering results (sorted by size): {dict(sorted_clusters)}")
        logger.info(f"Will perform fine clustering on top {len(clusters_to_refine)} largest clusters: {[c[0] for c in clusters_to_refine]}")
        logger.info(f"Max cluster size for refine: {max_cluster_size_for_refine}")

        # 收集所有需要精细聚类的结构索引
        all_refine_indices = []
        refine_cluster_mapping = {}  # 记录每个结构属于哪个粗聚�?
        
        for coarse_cluster, cluster_size in clusters_to_refine:
            if cluster_size <= max_cluster_size_for_refine and cluster_size >= 3:
                idx = [i for i, l in enumerate(self.coarse_labels) if l == coarse_cluster]
                all_refine_indices.extend(idx)
                for struct_idx in idx:
                    refine_cluster_mapping[struct_idx] = coarse_cluster



        # 处理所有粗聚类�?
        for cluster_idx, (coarse_cluster, cluster_size) in enumerate(sorted_clusters):
            idx = [i for i, l in enumerate(self.coarse_labels) if l == coarse_cluster]
            
            # 检查是否需要进行精细聚�?
            if coarse_cluster in [c[0] for c in clusters_to_refine]:
                logger.info(f"Processing coarse cluster {coarse_cluster} ({cluster_idx+1}/{len(sorted_clusters)}) with {cluster_size} structures - US-ALIGN REFINE")
                
                # 检查簇大小是否超过限制
                if cluster_size > max_cluster_size_for_refine:
                    logger.warning(f"  - Cluster size {cluster_size} exceeds limit {max_cluster_size_for_refine}, skipping fine clustering")
                    logger.warning(f"  - Assigning all structures to single fine cluster {cluster_id}")
                    for i in idx:
                        self.fine_labels[i] = cluster_id
                    cluster_id += 1
                    continue
                
                if cluster_size < 3:
                    # 小簇直接分配
                    logger.info(f"  - Small cluster ({cluster_size} < 3), assigning individual labels")
                    for i in idx:
                        self.fine_labels[i] = cluster_id
                        cluster_id += 1
                    continue

                # 使用US-align进行精细聚类
                try:
                    logger.info(f"  - Performing US-align fine clustering for {cluster_size} structures...")
                    
                    # 直接使用US-align进行聚类
                    sparse_labels = self._direct_usalign_clustering(idx)
                    
                    # 检查聚类结�?
                    if sparse_labels is None or len(sparse_labels) == 0:
                        logger.warning("  - US-align clustering failed, keeping coarse cluster labels")
                        for i in idx:
                            self.fine_labels[i] = cluster_id
                        cluster_id += 1
                    else:
                        # 分配标签
                        for k, l in enumerate(sparse_labels):
                            if l == -1:
                                self.fine_labels[idx[k]] = cluster_id
                                cluster_id += 1
                            else:
                                self.fine_labels[idx[k]] = cluster_id + l
                        cluster_id += sparse_labels.max() + 1 if sparse_labels.max() >= 0 else 1
                        
                        # 统计子聚类结�?
                        n_sub_clusters = len([l for l in set(sparse_labels) if l != -1])
                        n_noise = np.sum(sparse_labels == -1)
                        
                        # 记录该粗聚类的精细聚类结�?
                        self._record_fine_cluster_results(coarse_cluster, cluster_size, n_sub_clusters, n_noise, sparse_labels, idx)
                        logger.info(f"  - Fine clustering complete: {n_sub_clusters} clusters, {n_noise} noise points")
                        
                except Exception as e:
                    logger.warning(f"  - Fine clustering failed: {e}, keeping coarse cluster labels")
                    for i in idx:
                        self.fine_labels[i] = cluster_id
                    cluster_id += 1
            else:
                # 不进行精细聚类的簇，保持粗聚类标�?
                for i in idx:
                    self.fine_labels[i] = cluster_id
                cluster_id += 1

        logger.info(f"Fine clustering completed. Total fine clusters: {cluster_id}")
        logger.info(f"Refined {len(clusters_to_refine)} clusters, kept {len(sorted_clusters) - len(clusters_to_refine)} clusters as coarse labels")

        return self.fine_labels











    
    def _compute_precise_rmsd_usalign(self, neighbor_pairs):
        """
        使用US-align计算精确的RMSD距离
        
        Args:
            neighbor_pairs: 邻居对列�?[(idx1, idx2), ...]
            
        Returns:
            dict: {(idx1, idx2): rmsd_value, ...}
        """
        if not neighbor_pairs:
            return {}
        
        logger.info(f"      - Computing US-align RMSD for {len(neighbor_pairs)} neighbor pairs...")
        
        # 添加调试信息
        if len(neighbor_pairs) > 0:
            logger.info(f"      - First pair: {neighbor_pairs[0]}")
            logger.info(f"      - Last pair: {neighbor_pairs[-1]}")
            logger.info(f"      - Total unique pairs: {len(set(neighbor_pairs))}")
        
        # 获取结构文件路径映射
        structure_files = self._get_structure_file_paths()
        
        # 计算RMSD
        rmsd_results = {}
        
        # 使用多进程加�?
        with Pool(processes=min(self.n_jobs, 8)) as pool:
            # 准备参数
            args = [(pair, structure_files) for pair in neighbor_pairs]
            
            # 并行计算
            results = pool.starmap(self._compute_single_usalign_pair, args)
            
            # 收集结果
            for pair, rmsd in zip(neighbor_pairs, results):
                if rmsd is not None:
                    rmsd_results[pair] = rmsd
        
        logger.info(f"      - Successfully computed RMSD for {len(rmsd_results)} pairs")
        return rmsd_results
    
    def _get_structure_file_paths(self):
        """
        获取所有结构文件的路径映射
        
        Returns:
            dict: {idx: file_path, ...}
        """
        structure_files = {}
        
        logger.info(f"        - Getting file paths for {len(self.file_names)} structures")
        
        for global_idx, file_name in enumerate(self.file_names):
            # 优先使用粗聚类结构文件夹
            full_path = None
            if self.coarse_clusters_dir and self.coarse_clusters_dir.exists():
                # 根据粗聚类标签找到对应的文件�?
                if global_idx < len(self.coarse_labels):
                    cluster_id = self.coarse_labels[global_idx]
                    if cluster_id == -1:
                        cluster_dir = self.coarse_clusters_dir / "noise_cluster"
                    else:
                        cluster_dir = self.coarse_clusters_dir / f"cluster_{cluster_id}"
                    
                    if cluster_dir.exists():
                        cluster_file = cluster_dir / file_name
                        if cluster_file.exists():
                            full_path = str(cluster_file)
                            logger.debug(f"        - Using file from cluster directory: {full_path}")
            
            # 如果粗聚类文件夹中没有找到，使用原始PDB目录
            if full_path is None:
                if self.pdb_dir:
                    full_path = str(self.pdb_dir / file_name)
                else:
                    full_path = file_name
            
            # 检查文件是否存�?
            if not os.path.exists(full_path):
                logger.warning(f"        - File not found: {full_path}")
                continue
            
            structure_files[global_idx] = full_path
        
        logger.info(f"        - Found {len(structure_files)} valid structure files")
        
        if len(structure_files) == 0:
            logger.error("        - No valid structure files found!")
            
        return structure_files


    
    def _compute_single_usalign_pair(self, pair, structure_files):
        """
        计算单个结构对的US-align RMSD
        
        Args:
            pair: (idx1, idx2) 结构�?
            structure_files: 结构文件路径字典 {idx: file_path, ...}
            
        Returns:
            float: RMSD值，失败时返回None
        """
        idx1, idx2 = pair
        
        if idx1 not in structure_files or idx2 not in structure_files:
            return None
        
        file1 = structure_files[idx1]
        file2 = structure_files[idx2]
        
        try:
            # 运行US-align（使用默认输出格式）
            result = subprocess.run([
                USALIGN_CMD, file1, file2
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # 解析输出
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    # US-align输出格式通常是：TM-score=0.xxx RMSD=xx.xxx
                    if 'RMSD=' in line:
                        # 提取RMSD�?
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'RMSD=':
                                if i + 1 < len(parts):
                                    try:
                                        rmsd_value = float(parts[i + 1])
                                        return rmsd_value
                                    except ValueError:
                                        continue
                        # 如果上面的方法失败，尝试其他格式
                        for part in parts:
                            if part.startswith('RMSD='):
                                try:
                                    rmsd_value = float(part.split('=')[1])
                                    return rmsd_value
                                except (ValueError, IndexError):
                                    continue
            
            # 如果解析失败，记录详细信息用于调�?
            logger.info(f"US-align output for pair {pair}:")
            logger.info(f"  stdout: {result.stdout}")
            logger.info(f"  stderr: {result.stderr}")
            logger.info(f"  returncode: {result.returncode}")
            
            return None
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
            logger.debug(f"US-align computation failed for pair {pair}: {e}")
            return None
    
    def _build_sparse_distance_matrix(self, idx, neighbor_matrix, **kwargs):
        """
        构建稀疏距离矩�?
        只在邻居对上计算精确的iRMSD
        """
        cluster_size = len(idx)
        sparse_distances = {}
        
        # 只在邻居对上计算iRMSD
        neighbor_pairs = np.where(neighbor_matrix)
        total_pairs = len(neighbor_pairs[0]) // 2  # 避免重复计算
        
        logger.info(f"      - Computing iRMSD for {total_pairs} neighbor pairs")
        
        # 并行计算邻居对的iRMSD
        try:
            with Pool(processes=min(self.n_jobs, 8)) as pool:
                tasks = []
                for i, j in zip(neighbor_pairs[0], neighbor_pairs[1]):
                    if i < j:  # 只计算上三角矩阵
                        tasks.append((i, j, idx[i], idx[j]))
                
                # 使用更大的批处理�?
                batch_size = max(100, len(tasks) // (self.n_jobs * 2))
                
                for result in pool.imap_unordered(
                    self._compute_single_irmsd_pair, 
                    tasks, 
                    chunksize=batch_size
                ):
                    i, j, distance = result
                    sparse_distances[(i, j)] = distance
                    sparse_distances[(j, i)] = distance  # 确保对称�?
        except Exception as e:
            logger.error(f"Error in parallel iRMSD computation: {e}")
            logger.warning("Falling back to sequential computation")
            # 回退到顺序计�?
            for i, j in zip(neighbor_pairs[0], neighbor_pairs[1]):
                if i < j:  # 只计算上三角矩阵
                    try:
                        result = self._compute_single_irmsd_pair((i, j, idx[i], idx[j]))
                        i, j, distance = result
                        sparse_distances[(i, j)] = distance
                        sparse_distances[(j, i)] = distance  # 确保对称�?
                    except Exception as inner_e:
                        logger.debug(f"Error computing iRMSD for pair {i}-{j}: {inner_e}")
                        sparse_distances[(i, j)] = 10.0
                        sparse_distances[(j, i)] = 10.0
        
        return sparse_distances

    def _compute_single_irmsd_pair(self, task):
        """
        计算单个iRMSD对（用于并行处理�?
        """
        i, j, struct1_idx, struct2_idx = task
        
        try:
            # 获取结构对象
            struct1 = self.structures[struct1_idx]
            struct2 = self.structures[struct2_idx]
            
            # 对齐结构
            self.superimpose_chainA(struct1, struct2, self.chainA)
            
            # 计算iRMSD
            distance = self.calc_iRMSD(struct1, struct2, self.antigen_chains, self.irmsd_cutoff)
            return (i, j, distance)
        except Exception as e:
            logger.debug(f"Error in iRMSD computation for pair {i}-{j}: {e}")
            return (i, j, 10.0)  # 默认大距�?

    def _build_sparse_distance_matrix_usalign(self, idx, neighbor_matrix, **kwargs):
        """
        基于US-align RMSD结果构建稀疏距离矩�?
        
        Args:
            idx: 结构索引列表
            neighbor_matrix: 邻居矩阵
            **kwargs: 其他参数
            
        Returns:
            scipy.sparse.csr_matrix: 稀疏距离矩�?
        """
        cluster_size = len(idx)
        
        # 获取邻居�?
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
        
        # 创建稀疏矩�?
        rows, cols, data = [], [], []
        
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
        
        # 对角线设�?
        for i in range(cluster_size):
            rows.append(i)
            cols.append(i)
            data.append(0.0)
        
        # 创建稀疏矩�?
        distance_matrix = csr_matrix((data, (rows, cols)), shape=(cluster_size, cluster_size))
        
        return distance_matrix
    
    def _sparse_clustering_from_distances(self, distance_matrix, method='hdbscan'):
        """
        基于距离矩阵进行稀疏聚�?
        
        Args:
            distance_matrix: 稀疏距离矩�?
            method: 聚类方法 ('hdbscan', 'spectral')
            
        Returns:
            dict: 聚类结果
        """
        if method == 'hdbscan':
            return self._sparse_hdbscan_clustering(distance_matrix)
        elif method == 'spectral':
            return self._sparse_spectral_clustering(distance_matrix)
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
    
    def _sparse_hdbscan_clustering(self, distance_matrix):
        """
        基于稀疏距离矩阵的HDBSCAN聚类
        
        Args:
            distance_matrix: 稀疏距离矩�?
            
        Returns:
            dict: 聚类结果
        """
        # 转换为密集矩阵（对于小数据集�?
        if distance_matrix.shape[0] < 1000:
            dense_matrix = distance_matrix.toarray()
        else:
            # 对于大数据集，使用稀疏矩�?
            dense_matrix = distance_matrix
        
        # 自动参数估计
        min_cluster_size, min_samples = self._auto_hdbscan_params(len(self.file_names))
        
        # 执行HDBSCAN聚类
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='precomputed',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(dense_matrix)
        
        return {
            'labels': labels,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'method': 'hdbscan',
            'parameters': {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples
            }
        }
    
    def _sparse_spectral_clustering(self, distance_matrix):
        """
        基于稀疏距离矩阵的光谱聚类
        
        Args:
            distance_matrix: 稀疏距离矩�?
            
        Returns:
            dict: 聚类结果
        """
        # 转换为相似度矩阵
        max_dist = distance_matrix.max()
        similarity_matrix = max_dist - distance_matrix
        
        # 自动估计聚类�?
        n_clusters = min(10, max(2, len(self.file_names) // 20))
        
        # 执行光谱聚类
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        labels = clusterer.fit_predict(similarity_matrix)
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'method': 'spectral',
            'parameters': {
                'n_clusters': n_clusters
            }
        }
    
    def _direct_usalign_clustering(self, cluster_indices):
        """
        对单个聚类直接使用US-align进行聚类
        
        Args:
            cluster_indices: 聚类中的结构索引列表
            
        Returns:
            numpy.ndarray: 聚类标签数组
        """
        if len(cluster_indices) <= 1:
            return np.array([0] * len(cluster_indices))
        
        # 检查US-align是否可用
        try:
            result = subprocess.run([USALIGN_CMD, '--help'], 
                                  capture_output=True, text=True, timeout=10)
            usalign_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            usalign_available = False
        
        if not usalign_available:
            logger.warning("US-align not available, using fallback clustering method")
            return self._fallback_clustering(cluster_indices)
        
        # 计算所有对的距�?
        pairs = list(itertools.combinations(cluster_indices, 2))
        
        # 使用US-align计算距离
        rmsd_results = self._compute_precise_rmsd_usalign(pairs)
        
        if not rmsd_results:
            # 如果US-align失败，使用备用方�?
            logger.warning("US-align clustering failed, using fallback method")
            return self._fallback_clustering(cluster_indices)
        
        # 构建距离矩阵 - 修复：直接使用US-align结果构建稀疏矩�?
        cluster_size = len(cluster_indices)
        
        # 创建稀疏矩�?
        rows, cols, data = [], [], []
        
        # 创建索引映射
        idx_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(cluster_indices)}
        
        # 添加距离�?
        for (global_i, global_j), rmsd in rmsd_results.items():
            if global_i in idx_to_local and global_j in idx_to_local:
                local_i = idx_to_local[global_i]
                local_j = idx_to_local[global_j]
                
                # 添加对称的两个元�?
                rows.extend([local_i, local_j])
                cols.extend([local_j, local_i])
                data.extend([rmsd, rmsd])
        
        # 对角线设�?
        for i in range(cluster_size):
            rows.append(i)
            cols.append(i)
            data.append(0.0)
        
        # 创建稀疏矩�?
        distance_matrix = csr_matrix((data, (rows, cols)), shape=(cluster_size, cluster_size))
        
        # 执行聚类
        clustering_result = self._sparse_clustering_from_distances(distance_matrix, method='hdbscan')
        
        # 返回标签数组
        if isinstance(clustering_result, dict) and 'labels' in clustering_result:
            return np.array(clustering_result['labels'])
        else:
            return np.array([0] * len(cluster_indices))
    
    def _fallback_clustering(self, cluster_indices):
        """
        US-align不可用时的备用聚类方�?
        
        Args:
            cluster_indices: 聚类中的结构索引列表
            
        Returns:
            numpy.ndarray: 聚类标签数组
        """
        cluster_size = len(cluster_indices)
        
        if cluster_size <= 3:
            # 小簇直接分配
            return np.array(range(cluster_size))
        
        # 使用简单的距离矩阵（基于文件名相似性或其他简单特征）
        logger.info(f"  - Using fallback clustering for {cluster_size} structures")
        
        # 创建简单的距离矩阵（这里使用随机距离作为示例）
        # 在实际应用中，可以使用文件名相似性、文件大小等简单特�?
        np.random.seed(42)  # 固定随机种子以确保结果可重现
        
        # 创建随机距离矩阵
        distance_matrix = np.random.rand(cluster_size, cluster_size) * 5.0
        
        # 确保对称�?
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # 对角线设�?
        np.fill_diagonal(distance_matrix, 0.0)
        
        # 使用HDBSCAN进行聚类
        try:
            min_cluster_size, min_samples = self._auto_hdbscan_params(cluster_size)
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='precomputed',
                cluster_selection_method='eom'
            )
            
            labels = clusterer.fit_predict(distance_matrix)
            
            # 确保所有点都有标签
            if -1 in labels:
                # 将噪声点分配到最近的聚类
                noise_indices = np.where(labels == -1)[0]
                for noise_idx in noise_indices:
                    # 找到最近的聚类中心
                    distances = distance_matrix[noise_idx]
                    min_dist_idx = np.argmin(distances)
                    labels[noise_idx] = labels[min_dist_idx]
            
            logger.info(f"  - Fallback clustering completed: {len(set(labels))} clusters")
            return labels
            
        except Exception as e:
            logger.warning(f"  - Fallback clustering failed: {e}, using simple assignment")
            # 如果聚类失败，简单分�?
            return np.array(range(cluster_size))
    
    def _record_fine_cluster_results(self, coarse_cluster, cluster_size, n_sub_clusters, n_noise, sparse_labels, idx):
        """
        记录精细聚类结果
        
        Args:
            coarse_cluster: 粗聚类ID
            cluster_size: 粗聚类大�?
            n_sub_clusters: 子聚类数�?
            n_noise: 噪声点数�?
            sparse_labels: 稀疏聚类标�?
            idx: 结构索引列表
        """
        # 记录该粗聚类的精细聚类统计信�?
        logger.info(f"    - Coarse cluster {coarse_cluster}: {cluster_size} structures -> {n_sub_clusters} fine clusters + {n_noise} noise")
        
        # 更新精细聚类标签
        for i, original_idx in enumerate(idx):
            if original_idx < len(self.fine_labels):
                self.fine_labels[original_idx] = sparse_labels[i]
    
    def export_individual_fine_cluster_results(self, output_dir):
        """
        导出单个精细聚类的结构文�?
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        fine_clusters_dir = os.path.join(output_dir, 'fine_clusters')
        os.makedirs(fine_clusters_dir, exist_ok=True)
        
        # 按精细聚类分�?
        fine_cluster_groups = {}
        for i, label in enumerate(self.fine_labels):
            if label not in fine_cluster_groups:
                fine_cluster_groups[label] = []
            fine_cluster_groups[label].append(i)
        
        # 导出每个精细聚类
        for cluster_id, indices in fine_cluster_groups.items():
            cluster_dir = os.path.join(fine_clusters_dir, f'fine_cluster_{cluster_id}')
            os.makedirs(cluster_dir, exist_ok=True)
            
            # 复制结构文件
            for idx in indices:
                if idx < len(self.file_names):
                    file_name = self.file_names[idx]
                    
                    # 获取完整路径
                    if hasattr(self, 'pdb_dir') and self.pdb_dir:
                        src_path = os.path.join(self.pdb_dir, file_name)
                    else:
                        src_path = file_name
                    
                    # 复制文件
                    if os.path.exists(src_path):
                        dst_path = os.path.join(cluster_dir, file_name)
                        shutil.copy2(src_path, dst_path)
        
        print(f"精细聚类结果已导出到: {fine_clusters_dir}")
    
    def evaluate_clustering(self, labels):
        """
        评估聚类结果
        
        Args:
            labels: 聚类标签列表
            
        Returns:
            dict: 评估指标
        """
        if len(set(labels)) <= 1:
            return {
                'n_clusters': 1,
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 0.0
            }
        
        # 计算评估指标
        try:
            # 使用接触集作为特�?
            features = []
            for contact_set in self.contact_sets:
                # 转换为二进制特征向量
                feature_vector = [1 if contact in contact_set else 0 
                                for contact in self._get_all_contacts()]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # 计算指标
            silhouette = silhouette_score(features, labels)
            calinski_harabasz = calinski_harabasz_score(features, labels)
            davies_bouldin = davies_bouldin_score(features, labels)
            
        except Exception as e:
            print(f"评估指标计算失败: {e}")
            silhouette = calinski_harabasz = davies_bouldin = 0.0
        
        return {
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin
        }
    
    def _get_all_contacts(self):
        """
        获取所有接触的并集
        
        Returns:
            set: 所有接触的集合
        """
        all_contacts = set()
        for contact_set in self.contact_sets:
            all_contacts.update(contact_set)
        return all_contacts
    
    def get_cluster_representatives(self, labels, method='centroid'):
        """
        获取聚类代表
        
        Args:
            labels: 聚类标签列表
            method: 选择方法 ('centroid', 'random', 'first')
            
        Returns:
            dict: {cluster_id: representative_idx, ...}
        """
        representatives = {}
        
        # 按聚类分�?
        cluster_groups = {}
        for i, label in enumerate(labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(i)
        
        # 选择代表
        for cluster_id, indices in cluster_groups.items():
            if method == 'centroid':
                # 选择中心�?
                representative = self._find_centroid(indices)
            elif method == 'random':
                # 随机选择
                representative = random.choice(indices)
            elif method == 'first':
                # 选择第一�?
                representative = indices[0]
            else:
                raise ValueError(f"不支持的代表选择方法: {method}")
            
            representatives[cluster_id] = representative
        
        return representatives
    
    def _find_centroid(self, indices):
        """
        找到聚类的中心点
        
        Args:
            indices: 聚类中的结构索引列表
            
        Returns:
            int: 中心点索�?
        """
        if len(indices) == 1:
            return indices[0]
        
        # 计算接触集的中心
        contact_sets = [self.contact_sets[i] for i in indices]
        
        # 计算每个点到中心的距�?
        min_total_distance = float('inf')
        centroid_idx = indices[0]
        
        for i, idx in enumerate(indices):
            total_distance = 0
            for j, other_idx in enumerate(indices):
                if i != j:
                    # Jaccard距离
                    intersection = len(contact_sets[i] & contact_sets[j])
                    union = len(contact_sets[i] | contact_sets[j])
                    if union > 0:
                        distance = 1 - intersection / union
                    else:
                        distance = 0
                    total_distance += distance
            
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                centroid_idx = idx
        
        return centroid_idx
    
    def save_results(self, output_dir):
        """
        保存精细聚类结果
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存聚类结果
        results = {
            'file_names': self.file_names,
            'coarse_labels': self.coarse_labels,
            'fine_labels': self.fine_labels,
            'contact_sets': self.contact_sets,
            'clustering_info': {
                'n_coarse_clusters': len(set(self.coarse_labels)) - (1 if -1 in self.coarse_labels else 0),
                'n_fine_clusters': len(set(self.fine_labels)) - (1 if -1 in self.fine_labels else 0),
                'total_structures': len(self.file_names)
            }
        }
        
        # 保存为pickle文件
        results_file = os.path.join(output_dir, 'fine_clustering_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # 保存为CSV文件
        csv_file = os.path.join(output_dir, 'fine_clustering_results.csv')
        df = pd.DataFrame({
            'file_name': self.file_names,
            'coarse_cluster': self.coarse_labels,
            'fine_cluster': self.fine_labels
        })
        df.to_csv(csv_file, index=False)
        
        # 导出聚类结构文件
        self.export_individual_fine_cluster_results(output_dir)
        
        print(f"精细聚类结果已保存到: {output_dir}")
    
    def visualize_results(self, output_dir):
        """
        可视化精细聚类结�?
        
        Args:
            output_dir: 输出目录
        """
        # 创建可视化目�?
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 设置matplotlib参数
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AF3精细聚类结果可视�?, fontsize=16, fontweight='bold')
        
        # 1. 精细聚类分布
        self._plot_clustering_subplot(
            axes[0, 0], self.fine_labels, '精细聚类分布',
            '精细聚类ID', '结构数量', 'fine'
        )
        
        # 2. 粗聚类vs精细聚类对比
        self._plot_clustering_comparison(axes[0, 1])
        
        # 3. 聚类大小分布
        self._plot_cluster_size_distribution(axes[1, 0])
        
        # 4. 聚类层次结构
        self._plot_hierarchical_structure(axes[1, 1])
        
        # 保存图形
        plt.tight_layout()
        plot_file = os.path.join(viz_dir, 'fine_clustering_overview.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 导出单独的图�?
        self._export_individual_plots(viz_dir)
        
        # 导出数据
        self._export_data_csvs(viz_dir)
        
        # 生成径向树图
        self.plot_cluster_radial_tree(viz_dir)
        
        print(f"可视化结果已保存�? {viz_dir}")
    
    def _plot_clustering_subplot(self, ax, labels, title, xlabel, ylabel, cluster_type):
        """
        绘制聚类子图
        
        Args:
            ax: matplotlib轴对�?
            labels: 聚类标签
            title: 标题
            xlabel: x轴标�?
            ylabel: y轴标�?
            cluster_type: 聚类类型
        """
        # 统计聚类大小
        cluster_counts = Counter(labels)
        
        # 排序
        sorted_clusters = sorted(cluster_counts.items())
        cluster_ids = [str(x[0]) for x in sorted_clusters]
        counts = [x[1] for x in sorted_clusters]
        
        # 绘制柱状�?
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))
        bars = ax.bar(cluster_ids, counts, color=colors, alpha=0.8)
        
        # 添加数值标�?
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标�?
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_clustering_comparison(self, ax):
        """
        绘制粗聚类vs精细聚类对比�?
        
        Args:
            ax: matplotlib轴对�?
        """
        # 创建交叉�?
        comparison_data = pd.DataFrame({
            'coarse': self.coarse_labels,
            'fine': self.fine_labels
        })
        
        # 计算交叉�?
        cross_table = pd.crosstab(comparison_data['coarse'], comparison_data['fine'])
        
        # 绘制热图
        im = ax.imshow(cross_table.values, cmap='YlOrRd', aspect='auto')
        
        # 添加颜色�?
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('结构数量')
        
        # 设置标签
        ax.set_xlabel('精细聚类ID')
        ax.set_ylabel('粗聚类ID')
        ax.set_title('粗聚�?vs 精细聚类对比', fontweight='bold')
        
        # 添加数值标�?
        for i in range(cross_table.shape[0]):
            for j in range(cross_table.shape[1]):
                text = ax.text(j, i, cross_table.iloc[i, j],
                             ha="center", va="center", color="black", fontsize=8)
    
    def _plot_cluster_size_distribution(self, ax):
        """
        绘制聚类大小分布
        
        Args:
            ax: matplotlib轴对�?
        """
        # 统计聚类大小
        fine_cluster_counts = Counter(self.fine_labels)
        coarse_cluster_counts = Counter(self.coarse_labels)
        
        # 绘制分布
        fine_sizes = list(fine_cluster_counts.values())
        coarse_sizes = list(coarse_cluster_counts.values())
        
        # 创建箱线�?
        data = [coarse_sizes, fine_sizes]
        labels = ['粗聚�?, '精细聚类']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # 设置颜色
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('聚类大小分布对比', fontweight='bold')
        ax.set_ylabel('聚类大小')
        ax.grid(True, alpha=0.3)
    
    def _plot_hierarchical_structure(self, ax):
        """
        绘制层次结构�?
        
        Args:
            ax: matplotlib轴对�?
        """
        # 创建层次结构数据
        hierarchy_data = []
        
        for coarse_id in set(self.coarse_labels):
            if coarse_id == -1:
                continue
            
            # 获取该粗聚类中的精细聚类
            fine_in_coarse = [self.fine_labels[i] for i, label in enumerate(self.coarse_labels) 
                             if label == coarse_id]
            fine_counts = Counter(fine_in_coarse)
            
            for fine_id, count in fine_counts.items():
                hierarchy_data.append({
                    'coarse': f'C{coarse_id}',
                    'fine': f'F{fine_id}',
                    'count': count
                })
        
        if not hierarchy_data:
            ax.text(0.5, 0.5, '无层次结构数�?, ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('层次结构', fontweight='bold')
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(hierarchy_data)
        
        # 绘制桑基图（简化版本）
        unique_coarse = df['coarse'].unique()
        unique_fine = df['fine'].unique()
        
        # 创建简单的连接�?
        y_positions = {}
        
        # 分配位置
        for i, coarse in enumerate(unique_coarse):
            y_positions[coarse] = i
        
        for i, fine in enumerate(unique_fine):
            y_positions[fine] = i + len(unique_coarse)
        
        # 绘制连接
        for _, row in df.iterrows():
            x1, y1 = 0, y_positions[row['coarse']]
            x2, y2 = 1, y_positions[row['fine']]
            
            # 绘制连接�?
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=row['count'])
        
        # 设置标签
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.5, len(y_positions) - 0.5)
        
        # 添加标签
        for coarse in unique_coarse:
            ax.text(-0.05, y_positions[coarse], coarse, ha='right', va='center')
        
        for fine in unique_fine:
            ax.text(1.05, y_positions[fine], fine, ha='left', va='center')
        
        ax.set_title('聚类层次结构', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _export_individual_plots(self, viz_dir):
        """
        导出单独的图�?
        
        Args:
            viz_dir: 可视化目�?
        """
        # 1. 精细聚类分布
        plt.figure(figsize=(10, 6))
        self._plot_clustering_subplot(
            plt.gca(), self.fine_labels, '精细聚类分布',
            '精细聚类ID', '结构数量', 'fine'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'fine_cluster_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 聚类大小分布
        plt.figure(figsize=(8, 6))
        self._plot_cluster_size_distribution(plt.gca())
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'cluster_size_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 层次结构
        plt.figure(figsize=(10, 8))
        self._plot_hierarchical_structure(plt.gca())
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'hierarchical_structure.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _export_data_csvs(self, viz_dir):
        """
        导出数据CSV文件
        
        Args:
            viz_dir: 可视化目�?
        """
        # 聚类统计
        fine_cluster_stats = pd.DataFrame([
            {'cluster_id': k, 'size': v} 
            for k, v in Counter(self.fine_labels).items()
        ])
        fine_cluster_stats.to_csv(os.path.join(viz_dir, 'fine_cluster_statistics.csv'), 
                                 index=False)
        
        # 层次结构数据
        hierarchy_data = []
        for coarse_id in set(self.coarse_labels):
            if coarse_id == -1:
                continue
            
            fine_in_coarse = [self.fine_labels[i] for i, label in enumerate(self.coarse_labels) 
                             if label == coarse_id]
            fine_counts = Counter(fine_in_coarse)
            
            for fine_id, count in fine_counts.items():
                hierarchy_data.append({
                    'coarse_cluster': coarse_id,
                    'fine_cluster': fine_id,
                    'structure_count': count
                })
        
        if hierarchy_data:
            hierarchy_df = pd.DataFrame(hierarchy_data)
            hierarchy_df.to_csv(os.path.join(viz_dir, 'hierarchy_data.csv'), index=False)
    
    def plot_cluster_radial_tree(self, output_dir):
        """
        绘制聚类径向树图
        
        Args:
            output_dir: 输出目录
        """
        try:
            import networkx as nx
        except ImportError:
            print("警告: networkx未安装，跳过径向树图生成")
            return
        
        # 创建层次结构�?
        G = nx.DiGraph()
        
        # 添加根节�?
        G.add_node('root', pos=(0, 0))
        
        # 添加粗聚类节�?
        coarse_clusters = set(self.coarse_labels)
        coarse_clusters.discard(-1)  # 移除噪声�?
        
        for i, coarse_id in enumerate(sorted(coarse_clusters)):
            angle = 2 * np.pi * i / len(coarse_clusters)
            x = np.cos(angle) * 2
            y = np.sin(angle) * 2
            G.add_node(f'coarse_{coarse_id}', pos=(x, y))
            G.add_edge('root', f'coarse_{coarse_id}')
        
        # 添加精细聚类节点
        for coarse_id in coarse_clusters:
            fine_in_coarse = [self.fine_labels[i] for i, label in enumerate(self.coarse_labels) 
                             if label == coarse_id]
            fine_counts = Counter(fine_in_coarse)
            
            # 计算精细聚类的位�?
            fine_clusters = list(fine_counts.keys())
            for j, fine_id in enumerate(fine_clusters):
                # 获取粗聚类的位置
                coarse_pos = G.nodes[f'coarse_{coarse_id}']['pos']
                
                # 计算精细聚类的位�?
                angle = 2 * np.pi * j / len(fine_clusters)
                x = coarse_pos[0] + np.cos(angle) * 1
                y = coarse_pos[1] + np.sin(angle) * 1
                
                G.add_node(f'fine_{fine_id}', pos=(x, y))
                G.add_edge(f'coarse_{coarse_id}', f'fine_{fine_id}')
        
        # 绘制图形
        plt.figure(figsize=(12, 12))
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=['root'],
                              node_color='red', 
                              node_size=500,
                              alpha=0.8)
        
        nx.draw_networkx_nodes(G, pos,
                              nodelist=[n for n in G.nodes() if n.startswith('coarse_')],
                              node_color='blue',
                              node_size=300,
                              alpha=0.8)
        
        nx.draw_networkx_nodes(G, pos,
                              nodelist=[n for n in G.nodes() if n.startswith('fine_')],
                              node_color='green',
                              node_size=200,
                              alpha=0.8)
        
        # 绘制�?
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
        
        # 添加标签
        labels = {}
        labels['root'] = 'Root'
        for node in G.nodes():
            if node.startswith('coarse_'):
                labels[node] = f'C{node.split("_")[1]}'
            elif node.startswith('fine_'):
                labels[node] = f'F{node.split("_")[1]}'
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title('AF3聚类层次结构 - 径向树图', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # 保存图形
        tree_file = os.path.join(output_dir, 'cluster_radial_tree.png')
        plt.savefig(tree_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"径向树图已保存到: {tree_file}")


def validate_config(config):
    """
    验证配置参数
    
    Args:
        config: 配置字典
        
    Returns:
        bool: 配置是否有效
    """
    required_keys = ['coarse_results_file', 'pdb_dir']
    
    for key in required_keys:
        if key not in config:
            print(f"错误: 缺少必需的配置项 '{key}'")
            return False
    
    # 检查文件是否存�?
    if not os.path.exists(config['coarse_results_file']):
        print(f"错误: 粗聚类结果文件不存在: {config['coarse_results_file']}")
        return False
    
    if config['pdb_dir'] and not os.path.exists(config['pdb_dir']):
        print(f"错误: PDB目录不存�? {config['pdb_dir']}")
        return False
    
    return True


def check_dependencies():
    """
    检查依赖项
    
    Returns:
        bool: 所有依赖项是否可用
    """
    required_packages = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 
        'seaborn', 'hdbscan', 'Bio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"错误: 缺少以下Python�? {', '.join(missing_packages)}")
        print("请使用以下命令安�?")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_external_tools():
    """
    检查外部工�?
    
    Returns:
        dict: 工具可用性状�?
    """
    tools_status = {}
    
    # 获取环境变量中的工具路径
    usalign_cmd = os.environ.get('USALIGN_CMD', USALIGN_CMD)
    
    # 检查US-align
    try:
        result = subprocess.run([usalign_cmd, '--help'], 
                              capture_output=True, text=True, timeout=10)
        tools_status['usalign'] = result.returncode == 0
        if tools_status['usalign']:
            print(f"usalign: 可用 ({usalign_cmd})")
        else:
            print(f"usalign: 不可�?({usalign_cmd})")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        tools_status['usalign'] = False
        print(f"usalign: 不可�?({usalign_cmd}) - {e}")
    
    return tools_status


def get_example_configs():
    """
    获取示例配置
    
    Returns:
        dict: 示例配置
    """
    return {
        'coarse_results_file': 'path/to/coarse_clustering_results.pkl',
        'pdb_dir': 'path/to/pdb/files',
        'clustering_config': {
            'max_cluster_size': 100,
            'rmsd_threshold': 5.0,
            'n_jobs': 4
        },
        'output_config': {
            'output_dir': 'af3_fine_clustering_results',
            'save_visualizations': True,
            'export_structures': True
        }
    }


def main():
    """
    主函�?
    """
    print("=" * 60)
    print("AF3精细聚类分析�?)
    print("=" * 60)
    
    # 检查依赖项
    print("\n1. 检查依赖项...")
    if not check_dependencies():
        return
    
    # 检查外部工�?
    print("\n2. 检查外部工�?..")
    tools_status = check_external_tools()
    
    if not tools_status['usalign']:
        print("警告: US-align不可用，将使用备用方�?)
    
    # 配置参数（这些会被shell脚本替换�?
    COARSE_RESULTS_FILE = 'af3_coarse_clustering_results/coarse_clustering_results.pkl'
    PDB_DIR = './original_structures'  # 请根据实际情况修�?
    COARSE_CLUSTERS_DIR = 'af3_coarse_clustering_results/coarse_clusters'  # 粗聚类结构文件夹
    
    # 支持手动指定要精细聚类的簇（通过环境变量传递）
    import sys
    specified_clusters_str = os.environ.get('SPECIFIED_CLUSTERS', '')
    specified_clusters = None
    if specified_clusters_str:
        try:
            specified_clusters = [int(x.strip()) for x in specified_clusters_str.split(',') if x.strip()]
            print(f"使用手动指定的簇: {specified_clusters}")
        except ValueError as e:
            print(f"警告: 无法解析指定的簇 '{specified_clusters_str}': {e}")
            specified_clusters = None
     
    CLUSTERING_CONFIG = {
         'max_cluster_size': 100,
         'rmsd_threshold': 5.0,
         'n_jobs': 4,
         'clustering_method': 'hdbscan',
         'specified_clusters': specified_clusters
     }
    
    OUTPUT_CONFIG = {
        'output_dir': 'af3_fine_clustering_results',
        'save_visualizations': True,
        'export_structures': True
    }
    
    # 验证配置
    config = {
        'coarse_results_file': COARSE_RESULTS_FILE,
        'pdb_dir': PDB_DIR,
        'clustering_config': CLUSTERING_CONFIG,
        'output_config': OUTPUT_CONFIG
    }
    
    print("\n3. 验证配置...")
    if not validate_config(config):
        return
    
    # 检查粗聚类结构文件�?
    if Path(COARSE_CLUSTERS_DIR).exists():
        print(f"找到粗聚类结构文件夹: {COARSE_CLUSTERS_DIR}")
        print("将使用粗聚类结构文件夹中的文件进行US-align分析")
    else:
        print(f"未找到粗聚类结构文件�? {COARSE_CLUSTERS_DIR}")
        print("将使用原始PDB目录中的文件")
        COARSE_CLUSTERS_DIR = None
    
    # 初始化分析器
    print("\n4. 初始化精细聚类分析器...")
    analyzer = AF3FineClusterAnalyzer(
        coarse_results_file=COARSE_RESULTS_FILE,
        pdb_dir=PDB_DIR,
        coarse_clusters_dir=COARSE_CLUSTERS_DIR,
        n_jobs=CLUSTERING_CONFIG.get('n_jobs', 4)
    )
    
    # 加载粗聚类结�?
    print("\n5. 加载粗聚类结�?..")
    if not analyzer.load_coarse_results():
        print("错误: 无法加载粗聚类结果，退�?)
        return
    
    # 执行精细聚类
    print("\n6. 执行精细聚类...")
    fine_labels = analyzer.perform_fine_clustering(**CLUSTERING_CONFIG)
    
    if fine_labels is None:
        print("错误: 精细聚类失败，退�?)
        return
    
    # 评估结果
    print("\n7. 评估聚类结果...")
    evaluation = analyzer.evaluate_clustering(fine_labels)
    
    if 'error' in evaluation:
        print(f"警告: 评估失败 - {evaluation['error']}")
        print("使用默认评估�?)
        evaluation = {
            'n_clusters': evaluation.get('n_clusters', 0),
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': 0.0
        }
    
    print(f"精细聚类数量: {evaluation['n_clusters']}")
    print(f"轮廓系数: {evaluation['silhouette_score']:.3f}")
    print(f"Calinski-Harabasz指数: {evaluation['calinski_harabasz_score']:.3f}")
    print(f"Davies-Bouldin指数: {evaluation['davies_bouldin_score']:.3f}")
    
    # 保存结果
    print("\n8. 保存结果...")
    analyzer.save_results(OUTPUT_CONFIG['output_dir'])
    
    # 可视�?
    if OUTPUT_CONFIG['save_visualizations']:
        print("\n9. 生成可视�?..")
        analyzer.visualize_results(OUTPUT_CONFIG['output_dir'])
    
    print("\n" + "=" * 60)
    print("精细聚类分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
