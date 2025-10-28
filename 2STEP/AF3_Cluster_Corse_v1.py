#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AF3 Complex Coarse Clustering Pipeline
基于2T_AF3_Cluster_v2.py的粗聚类功能拆分

功能：
1. 从 AF3 预测的 CIF/PDB 文件中提取蛋白质复合物接触信息
2. 基于Jaccard距离的粗聚类
3. 支持多种聚类算法和自适应参数（HDBSCAN/KMeans/DBSCAN）
4. 丰富的可视化结果（标签分布/簇大小/t-SNE/径向树）
5. 并行处理和缓存机制
6. 灵活的输出管理和配置系统

使用方法：
1. 修改 main() 函数中的配置参数
2. 选择聚类算法和参数
3. 运行脚本

配置选项：
- CHAIN_CONFIG: 链配置（抗体/抗原链、距离阈值等）
- CLUSTERING_CONFIG: 聚类算法配置（HDBSCAN/KMeans/DBSCAN）
- OUTPUT_CONFIG: 输出配置（自动调整、图形显示等）

聚类算法选择：
- HDBSCAN: 适合大数据集，自动参数估计
- KMeans: 适合已知簇数量的情况
- DBSCAN: 适合密度不均匀的数据

接触检测选项：
- 'interface': 真正的界面原子（基于残基级别识别界面，再选择Cα+侧链，推荐用于蛋白质互作分析）
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
import glob
import psutil
import gc

from Bio.PDB import PDBParser, MMCIFParser, Superimposer
import hdbscan
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from matplotlib import patches

# ------------------------
# 日志配置
# ------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AF3CoarseClusterAnalyzer:
    """
    AF3 复合物粗聚类分析器类
    - 输入：AF3预测的PDB文件目录
    - 输出：粗聚类结果 + 丰富可视化
    - 专注于粗聚类功能，不包含精细聚类
    """

    def __init__(self, pdb_dir, chainA='A', antigen_chains=['B', 'C'], 
                 contact_cutoff=5.0, irmsd_cutoff=5.0, n_jobs=-1,
                 residue_ranges=None, contact_mode='jaccard', contact_atom_type='interface'):
        """
        初始化分析器

        参数：
        - pdb_dir: PDB文件目录
        - chainA: 抗体/受体链ID（通常是单个链）
        - antigen_chains: 抗原链ID列表（可能包含多个链，如['B', 'C', 'D']）
        - contact_cutoff: 接触判断的距离阈值（Å）
        - irmsd_cutoff: iRMSD计算的距离阈值（Å）
        - n_jobs: 并行处理线程数（-1=自动检测CPU数）
        - residue_ranges: 残基范围字典，格式：{'A': [(1,50), (70,100)], 'B': None}
        - contact_mode: 接触分析模式 ('jaccard')
        - contact_atom_type: 接触检测原子类型 ('interface'，基于残基级别的界面原子识别)
        """
        self.pdb_dir = Path(pdb_dir)
        self.chainA = chainA
        self.antigen_chains = antigen_chains if isinstance(antigen_chains, list) else [antigen_chains]
        self.contact_cutoff = contact_cutoff
        self.irmsd_cutoff = irmsd_cutoff
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.residue_ranges = residue_ranges
        self.contact_mode = contact_mode
        self.contact_atom_type = contact_atom_type.lower()
        self.interface_cutoff = getattr(self, 'interface_cutoff', 8.0)
        self.interface_method = getattr(self, 'interface_method', 'residue').lower()

        # HDBSCAN 参数（可调）
        self.min_cluster_size = 10
        self.min_samples = 5
        self.cluster_selection_epsilon = 0.1

        # 数据存储
        self.contact_sets = []     # 每个结构的接触集
        self.binary_features = []  # 二进制接触特征
        self.structures = []       # 结构对象
        self.file_names = []       # 文件名
        self.coarse_labels = None  # 粗聚类标签
        self.scaler = StandardScaler()  # 标准化器
        # 统一色卡（Nature 风格）
        self.palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 内存监控
        self.memory_threshold = 0.9  # 90%内存使用率阈值

    # --------------------
    # 内存监控
    # --------------------
    def _check_memory_usage(self):
        """检查内存使用情况"""
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
        min_cluster_size = max(5, min(100, max(1, n_samples // 20)))
        min_samples = max(3, min(min_cluster_size, int(np.log2(max(n_samples, 2))) + 1))
        return min_cluster_size, min_samples

    def _auto_dbscan_params(self, X: np.ndarray):
        """
        通过子采样 + k-距离中位数估计 eps
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
            diffs = S[:, None, :] - S[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, np.inf)
            kth = np.partition(dists, kth=k-1, axis=1)[:, k-1]
            eps = float(np.median(kth))
            eps = max(1e-6, eps)
            return eps, k
        except Exception as e:
            logger.warning(f"Auto DBSCAN parameter estimation failed: {e}")
            return 0.5, 5

    def parse_chain_and_ranges(self, chain_pair):
        """
        解析链对和残基范围（来自sites_cluster.py）
        
        输入格式示例：
        - "A:100-200;B:50-150"
        - "A:20-100,150-200;B:10-35,200-250"
        - "A:all;B:100-200"
        
        返回：
            dict: 残基范围字典
        """
        chain_specs = chain_pair.split(';')
        if len(chain_specs) != 2:
            raise ValueError("Chain pair must specify exactly two chains")
        
        chains_ranges = {}
        for spec in chain_specs:
            parts = spec.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid chain specification: {spec}")
            
            chain_id = parts[0]
            range_str = parts[1].lower()  # Convert to lowercase for 'all' comparison
            
            if range_str == 'all':
                chains_ranges[chain_id] = None  # None indicates all residues
            else:
                ranges = []
                for range_part in range_str.split(','):
                    try:
                        start, end = map(int, range_part.split('-'))
                        if start > end:
                            raise ValueError(f"Invalid range: {start}-{end}, start must be less than end")
                        ranges.append((start, end))
                    except ValueError as e:
                        raise ValueError(f"Invalid range format in {range_part}. Expected format: start-end") from e
                chains_ranges[chain_id] = ranges
        
        return chains_ranges

    def set_residue_ranges(self, chain_pair):
        """
        设置残基范围
        
        参数：
        - chain_pair: 链对字符串，如 "A:1-50,70-100;B:all"
        """
        self.residue_ranges = self.parse_chain_and_ranges(chain_pair)
        logger.info(f"Set residue ranges: {self.residue_ranges}")

    def extract_chain_residues_by_ranges(self, structure, chain_id, ranges):
        """
        根据指定范围提取链残基的CA原子坐标（来自sites_cluster.py）
        
        参数：
        - structure: 结构对象
        - chain_id: 链ID
        - ranges: 残基范围列表或None（表示所有残基）
        
        返回：
            List[np.ndarray]: CA原子坐标列表
        """
        try:
            chain = structure[0][chain_id]
            coords = []
            
            if ranges is None:  # if 'all'
                return [residue["CA"].coord for residue in chain if "CA" in residue]
            
            for start, end in ranges:
                for residue in chain:
                    residue_id = residue.id[1]
                    if (start <= residue_id <= end) and ("CA" in residue):
                        coords.append(residue["CA"].coord)
            
            return coords
        except KeyError:
            logger.warning(f"Chain {chain_id} not found in structure")
            return []
        except Exception as e:
            logger.warning(f"Error extracting residues from chain {chain_id}: {e}")
            return []

    # ==========================================================
    # (1) 结构加载和接触提取
    # ==========================================================
    def load_structure(self, structure_file):
        """加载结构文件（支持PDB和CIF格式）"""
        try:
            file_path = Path(structure_file)
            ext = file_path.suffix.lower()
            
            # 根据文件扩展名选择解析器
            if ext == ".pdb":
                parser = PDBParser(QUIET=True)
            elif ext == ".cif":
                parser = MMCIFParser(QUIET=True)
            else:
                logger.warning(f"Unsupported file format: {ext}")
                return None
            
            # 加载结构
            structure = parser.get_structure(file_path.stem, file_path)
            
            # 验证结构完整性
            if len(structure) == 0:
                logger.warning(f"Empty structure in {file_path}")
                return None
                
            # 检查是否有链
            if len(structure[0]) == 0:
                logger.warning(f"No chains found in {file_path}")
                return None
                
            return structure
            
        except Exception as e:
            logger.warning(f"Error loading {structure_file}: {e}")
            return None

    def get_chain_coords(self, structure, chain_id, atom_type="CA"):
        """获取指定链的坐标和残基信息"""
        coords, residues = [], []
        try:
            for res in structure[0][chain_id]:
                if atom_type in [a.get_id() for a in res]:
                    coords.append(res[atom_type].get_coord())
                    residues.append(res.get_id()[1])  # residue number
        except KeyError:
            logger.warning(f"Chain {chain_id} not found in structure")
        return np.array(coords), residues

    def get_contacts_interface_atoms(self, structure, chainA="A", antigen_chains=None, cutoff=5.0, interface_cutoff=8.0):
        """
        基于残基级别的界面原子接触检测方法
        
        参数：
        - structure: 结构对象
        - chainA: 抗体链ID
        - antigen_chains: 抗原链ID列表
        - cutoff: 接触距离阈值（Å）
        - interface_cutoff: 界面原子定义距离阈值（Å）
        """
        if antigen_chains is None:
            antigen_chains = self.antigen_chains
            
        try:
            # 第一步：识别界面残基
            interface_residuesA = set()
            interface_residues_antigen = set()
            
            # 获取所有残基对
            for resA in structure[0][chainA]:
                for chain_id in antigen_chains:
                    for resB in structure[0][chain_id]:
                        # 计算残基间最小距离
                        min_distance = float('inf')
                        for atomA in resA:
                            if atomA.get_id()[0] == 'H':
                                continue
                            for atomB in resB:
                                if atomB.get_id()[0] == 'H':
                                    continue
                                dist = atomA - atomB
                                if dist < min_distance:
                                    min_distance = dist
                        
                        # 如果残基间最小距离小于界面阈值，认为是界面残基
                        if min_distance <= interface_cutoff:
                            interface_residuesA.add(resA)
                            interface_residues_antigen.add(resB)
            
            # 第二步：从界面残基中选择Cα和侧链原子
            atomsA = []
            for res in interface_residuesA:
                # 包括Cα原子
                if "CA" in res:
                    atomsA.append(res["CA"])
                # 包括侧链原子
                sidechain_atoms = [a for a in res if a.get_id() not in ['N', 'CA', 'C', 'O'] and a.get_id()[0] != 'H']
                atomsA.extend(sidechain_atoms)
            
            selected_antigen_atoms = []
            for res in interface_residues_antigen:
                # 包括Cα原子
                if "CA" in res:
                    selected_antigen_atoms.append(res["CA"])
                # 包括侧链原子
                sidechain_atoms = [a for a in res if a.get_id() not in ['N', 'CA', 'C', 'O'] and a.get_id()[0] != 'H']
                selected_antigen_atoms.extend(sidechain_atoms)
            
            # 第三步：计算接触
            contacts = set()
            for a in atomsA:
                for b in selected_antigen_atoms:
                    if (a - b) <= cutoff:
                        antigen_chain_id = b.get_parent().get_parent().id
                        contacts.add((a.get_parent().get_id()[1], f"{antigen_chain_id}:{b.get_parent().get_id()[1]}"))
            
            return contacts
            
        except Exception as e:
            logger.warning(f"Error in residue-based interface detection: {e}")
            return set()

    def process_single_file(self, structure_file):
        """处理单个结构文件"""
        try:
            structure = self.load_structure(structure_file)
            if structure is None:
                return None, None, None
            
            # 使用界面原子接触检测方法
            interface_cutoff = getattr(self, 'interface_cutoff', 8.0)
            contacts = self.get_contacts_interface_atoms(structure, self.chainA, self.antigen_chains, self.contact_cutoff, interface_cutoff)
            
            if len(contacts) == 0:
                logger.warning(f"No contacts found in {structure_file}")
                return None, None, None
            
            return contacts, structure, Path(structure_file).name
        except Exception as e:
            logger.error(f"Error processing {structure_file}: {e}")
            return None, None, None

    def load_and_process_data(self, cache_file="af3_coarse_data_cache.pkl"):
        """
        加载数据（支持缓存）
        """
        if Path(cache_file).exists():
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                self.contact_sets = cached.get("contact_sets", [])
                self.structures = cached.get("structures", [])
                self.file_names = cached.get("file_names", [])
            elif isinstance(cached, (list, tuple)) and len(cached) == 4:
                self.contact_sets, self.binary_features, self.structures, self.file_names = cached
            elif isinstance(cached, (list, tuple)) and len(cached) == 3:
                self.contact_sets, self.structures, self.file_names = cached
                self.binary_features = [None] * len(self.contact_sets)  # 为兼容性添加空的binary_features
            else:
                logger.error(f"Unrecognized cache format in {cache_file}")
                return False
            logger.info(f"Loaded cached data from {cache_file} (files: {len(self.file_names)})")
            return True

        # 校验输入目录
        if not self.pdb_dir.exists() or not self.pdb_dir.is_dir():
            logger.error(f"PDB directory not found: {self.pdb_dir}")
            return False

        # 查找结构文件（优先CIF，因为AF3输出CIF格式）
        cif_files = list(self.pdb_dir.glob("*.cif"))
        pdb_files = list(self.pdb_dir.glob("*.pdb"))
        structure_files = cif_files + pdb_files  # CIF优先
        
        logger.info(f"Found {len(cif_files)} CIF files and {len(pdb_files)} PDB files in {self.pdb_dir}")
        logger.info(f"Total structure files: {len(structure_files)}")

        if len(structure_files) == 0:
            logger.error("No CIF or PDB files found")
            return False

        # 并行处理文件
        with Pool(processes=self.n_jobs) as pool:
            results = list(pool.imap_unordered(self.process_single_file, structure_files))

        # 收集有效结果
        for contacts, structure, fname in results:
            if contacts is not None:
                self.contact_sets.append(contacts)
                self.structures.append(structure)
                self.file_names.append(fname)

        if len(self.file_names) == 0:
            logger.error("No valid results obtained")
            return False

        # 为兼容性添加空的binary_features
        self.binary_features = [None] * len(self.contact_sets)
        
        # 保存缓存
        with open(cache_file, "wb") as f:
            pickle.dump((self.contact_sets, self.binary_features, self.structures, self.file_names), f)
        logger.info(f"Processed and cached {len(self.file_names)} files")
        return True

    # ==========================================================
    # (2) 距离计算
    # ==========================================================
    def jaccard_distance_matrix(self, contact_sets):
        """计算Jaccard距离矩阵"""
        N = len(contact_sets)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                inter = len(contact_sets[i] & contact_sets[j])
                union = len(contact_sets[i] | contact_sets[j])
                d = 1.0 - (inter/union if union > 0 else 0.0)
                D[i, j] = D[j, i] = d
        return D

    def normalized_hamming_distance(self, arr1, arr2):
        """
        计算归一化汉明距离（来自sites_cluster.py）
        
        参数：
        - arr1, arr2: 二进制数组
        
        返回：
            float: 归一化汉明距离
        """
        if arr1.shape != arr2.shape:
            raise ValueError("Arrays must have the same shape")
        
        # 创建掩码，其中arr1或arr2不为0
        mask = (arr1 != 0) | (arr2 != 0)
        
        # 应用掩码到两个数组
        filtered_arr1 = arr1[mask]
        filtered_arr2 = arr2[mask]
        
        # 计算汉明距离
        if len(filtered_arr1) == 0:
            return 1.0  # 如果没有元素可比较，返回1.0
        
        hamming_dist = np.sum(filtered_arr1 != filtered_arr2)
        normalized_dist = hamming_dist / len(filtered_arr1)
        
        return normalized_dist

    def cosine_distance(self, vec_a, vec_b):
        """
        计算余弦距离（来自sites_cluster.py）
        
        参数：
        - vec_a, vec_b: 向量
        
        返回：
            float: 余弦距离
        """
        from scipy.spatial.distance import cosine
        vec_a = np.asarray(vec_a)
        vec_b = np.asarray(vec_b)
        return cosine(vec_a, vec_b)

    def binary_distance_matrix(self, binary_features):
        """计算二进制特征的距离矩阵"""
        N = len(binary_features)
        D = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i+1, N):
                try:
                    d = self.normalized_hamming_distance(
                        np.array(binary_features[i]), 
                        np.array(binary_features[j])
                    )
                    D[i, j] = D[j, i] = d
                except Exception as e:
                    logger.warning(f"Error computing distance between {i} and {j}: {e}")
                    D[i, j] = D[j, i] = 1.0  # 默认最大距离
        return D

    def cosine_distance_matrix(self, binary_features):
        """计算余弦距离矩阵"""
        N = len(binary_features)
        D = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i+1, N):
                try:
                    d = self.cosine_distance(binary_features[i], binary_features[j])
                    D[i, j] = D[j, i] = d
                except Exception as e:
                    logger.warning(f"Error computing cosine distance between {i} and {j}: {e}")
                    D[i, j] = D[j, i] = 2.0  # 余弦距离的最大值
        return D

    # ==========================================================
    # (3) 聚类方法
    # ==========================================================
    def perform_coarse_clustering(self, method='hdbscan', distance_metric=None, **kwargs):
        """
        基于接触集的粗聚类
        
        参数：
        - method: 聚类方法 ('hdbscan', 'kmeans', 'dbscan')
        - distance_metric: 距离度量 ('jaccard', 'binary', 'cosine')
        """
        # 确定距离度量
        if distance_metric is None:
            distance_metric = self.contact_mode
            
        # 根据距离度量选择特征和计算距离矩阵
        if distance_metric == 'binary' and self.binary_features:
            if not self.binary_features or not any(self.binary_features):
                logger.error("No binary features available")
                return None
            logger.info("Using binary contact features for clustering")
            D = self.binary_distance_matrix(self.binary_features)
            n_samples = len(self.binary_features)
        elif distance_metric == 'cosine' and self.binary_features:
            if not self.binary_features or not any(self.binary_features):
                logger.error("No binary features available")
                return None
            logger.info("Using cosine distance with binary features for clustering")
            D = self.cosine_distance_matrix(self.binary_features)
            n_samples = len(self.binary_features)
        else:
            # 默认使用Jaccard距离
            if not self.contact_sets:
                logger.error("No contact sets available")
                return None
            logger.info("Using Jaccard distance with contact sets for clustering")
            D = self.jaccard_distance_matrix(self.contact_sets)
            n_samples = len(self.contact_sets)

        if method == 'hdbscan':
            # 自适应参数
            auto_min_cluster_size, auto_min_samples = self._auto_hdbscan_params(n_samples)
            min_cluster_size = kwargs.get('min_cluster_size', auto_min_cluster_size)
            min_samples = kwargs.get('min_samples', auto_min_samples)
            
            logger.info(f"Coarse HDBSCAN params -> min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='precomputed',
                core_dist_n_jobs=self.n_jobs
            )
            self.coarse_labels = clusterer.fit_predict(D)

        elif method == 'kmeans':
            n_clusters = kwargs.get("n_clusters", 5)
            logger.info(f"Coarse KMeans n_clusters={n_clusters}")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.coarse_labels = clusterer.fit_predict(D)

        elif method == 'dbscan':
            eps = kwargs.get("eps", 0.5)
            min_samples = kwargs.get("min_samples", 5)
            logger.info(f"Coarse DBSCAN params -> eps={eps}, min_samples={min_samples}")
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            self.coarse_labels = clusterer.fit_predict(D)

        else:
            raise ValueError(f"Unsupported clustering method {method}")

        # 验证聚类结果
        self._validate_clustering_results(method, n_samples)
        
        return self.coarse_labels

    def _validate_clustering_results(self, method, n_samples):
        """
        验证聚类结果的有效性
        
        参数：
        - method: 聚类方法
        - n_samples: 样本数量
        """
        if self.coarse_labels is None:
            logger.error("Clustering failed: coarse_labels is None")
            return False
        
        unique_labels = np.unique(self.coarse_labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        noise_count = np.sum(self.coarse_labels == -1)
        noise_ratio = noise_count / len(self.coarse_labels)
        
        logger.info(f"Clustering validation:")
        logger.info(f"  - Total samples: {n_samples}")
        logger.info(f"  - Number of clusters: {n_clusters}")
        logger.info(f"  - Noise samples: {noise_count} ({noise_ratio:.2%})")
        logger.info(f"  - Unique labels: {unique_labels}")
        
        # 检查是否有有效的聚类结果
        if n_clusters == 0:
            logger.error("Clustering failed: No valid clusters found!")
            logger.error("All samples were classified as noise.")
            logger.error("Possible solutions:")
            logger.error("1. Adjust clustering parameters (e.g., reduce min_cluster_size for HDBSCAN)")
            logger.error("2. Try different clustering method")
            logger.error("3. Check if input data is suitable for clustering")
            return False
        
        # 检查噪声比例
        if noise_ratio > 0.8:
            logger.warning(f"High noise ratio detected: {noise_ratio:.2%}")
            logger.warning("Consider adjusting clustering parameters")
        
        # 检查聚类大小分布
        cluster_sizes = []
        for label in unique_labels:
            if label != -1:
                size = np.sum(self.coarse_labels == label)
                cluster_sizes.append(size)
        
        if cluster_sizes:
            min_size = min(cluster_sizes)
            max_size = max(cluster_sizes)
            logger.info(f"  - Cluster size range: {min_size} - {max_size}")
            
            if min_size < 2:
                logger.warning("Some clusters have very small sizes (< 2 samples)")
        
        return True

    # ==========================================================
    # (4) 聚类评估
    # ==========================================================
    def evaluate_clustering(self):
        """评估聚类质量"""
        if self.coarse_labels is None:
            return {"silhouette": None, "n_clusters": 0, "noise_ratio": 1.0}

        # 计算Jaccard距离用于评估
        D = self.jaccard_distance_matrix(self.contact_sets)
        
        # 轮廓系数
        try:
            score = silhouette_score(D, self.coarse_labels, metric='precomputed')
        except:
            score = None

        # 统计信息
        unique_labels = set(self.coarse_labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        noise_count = np.sum(self.coarse_labels == -1)
        noise_ratio = noise_count / len(self.coarse_labels)

        return {
            "silhouette": score,
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "total_samples": len(self.coarse_labels)
        }

    # ==========================================================
    # (5) 获取簇代表
    # ==========================================================
    def get_cluster_representatives(self):
        """获取每个簇的代表性结构（medoid）"""
        if self.coarse_labels is None:
            return {}

        D = self.jaccard_distance_matrix(self.contact_sets)
        reps = {}
        
        for label in set(self.coarse_labels):
            if label == -1:
                continue
            indices = np.where(self.coarse_labels == label)[0]
            if len(indices) == 0:
                continue
                
            # 计算medoid
            subD = D[np.ix_(indices, indices)]
            mean_dist = subD.mean(axis=1)
            medoid_idx = indices[np.argmin(mean_dist)]
            
            reps[label] = {
                "size": len(indices),
                "representative": self.file_names[medoid_idx],
                "members": [self.file_names[i] for i in indices],
                "mean_distance": float(mean_dist.min())
            }

        return reps

    # ==========================================================
    # (6) 保存结果
    # ==========================================================
    def save_results(self, filename):
        """
        保存粗聚类结果
        
        参数：
        - filename: 基础文件名
        """
        if self.coarse_labels is None:
            logger.error("No clustering results to save")
            return

        # 保存粗聚类结果
        coarse_results = {
            "file_names": self.file_names,
            "coarse_labels": self.coarse_labels,
            "contact_sets": self.contact_sets,
            "clustering_type": "coarse"
        }
        
        with open(filename, "wb") as f:
            pickle.dump(coarse_results, f)
        logger.info(f"Coarse clustering results saved to {filename}")

        # 保存粗聚类CSV摘要
        try:
            csv_path = Path(filename).with_suffix('.csv')
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(["file_name", "coarse_label", "contact_size"])
                for fn, cl, cs in zip(self.file_names, self.coarse_labels, self.contact_sets):
                    writer.writerow([fn, int(cl), len(cs)])
            logger.info(f"Coarse clustering CSV saved to {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save coarse clustering CSV: {e}")

        # 导出聚类结构文件
        try:
            output_dir = Path(filename).parent
            self.export_cluster_structures(output_dir)
        except Exception as e:
            logger.warning(f"Failed to export cluster structures: {e}")

    def export_cluster_structures(self, output_dir):
        """
        将聚类结果按簇分别导出到不同文件夹
        
        参数：
        - output_dir: 输出目录
        """
        if self.coarse_labels is None:
            logger.error("No clustering results to export")
            return

        # 创建聚类结构目录
        clusters_dir = output_dir / "coarse_clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)
        
        # 按聚类分组
        cluster_groups = {}
        for i, label in enumerate(self.coarse_labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(i)
        
        logger.info(f"Exporting {len(cluster_groups)} clusters to separate directories...")
        
        # 统计实际复制的文件数量
        total_files_copied = 0
        cluster_copy_stats = {}
        
        # 导出每个聚类
        for cluster_id, indices in cluster_groups.items():
            # 创建聚类目录
            if cluster_id == -1:
                cluster_dir = clusters_dir / "noise_cluster"
            else:
                cluster_dir = clusters_dir / f"cluster_{cluster_id}"
            
            cluster_dir.mkdir(exist_ok=True)
            
            # 复制结构文件
            files_copied = 0
            for idx in indices:
                if idx < len(self.file_names):
                    file_name = self.file_names[idx]
                    src_path = self.pdb_dir / file_name
                    
                    # 复制文件
                    if src_path.exists():
                        dst_path = cluster_dir / file_name
                        try:
                            import shutil
                            shutil.copy2(src_path, dst_path)
                            files_copied += 1
                            total_files_copied += 1
                        except Exception as e:
                            logger.warning(f"Failed to copy {file_name} to cluster {cluster_id}: {e}")
                    else:
                        logger.warning(f"Source file not found: {src_path}")
                else:
                    logger.warning(f"Index {idx} out of range for file_names (length: {len(self.file_names)})")
            
            cluster_copy_stats[cluster_id] = files_copied
            logger.info(f"Cluster {cluster_id}: {files_copied}/{len(indices)} structures exported to {cluster_dir}")
        
        # 检查是否有空文件夹
        empty_clusters = [cid for cid, count in cluster_copy_stats.items() if count == 0]
        if empty_clusters:
            logger.warning(f"Empty clusters detected: {empty_clusters}")
            logger.warning("This may indicate clustering issues or file access problems")
        
        # 检查总复制文件数
        if total_files_copied == 0:
            logger.error("No files were copied to any cluster directory!")
            logger.error("Please check:")
            logger.error("1. Source PDB directory exists and is accessible")
            logger.error("2. File names in file_names list are correct")
            logger.error("3. Clustering results are valid")
        else:
            logger.info(f"Total files copied: {total_files_copied}")
        
        # 创建聚类信息文件
        cluster_info_file = clusters_dir / "cluster_info.txt"
        with open(cluster_info_file, 'w') as f:
            f.write("Coarse Clustering Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total files processed: {len(self.file_names)}\n")
            f.write(f"Total files copied: {total_files_copied}\n")
            f.write(f"Total clusters: {len(cluster_groups)}\n")
            f.write(f"Empty clusters: {len(empty_clusters)}\n\n")
            
            for cluster_id, indices in cluster_groups.items():
                files_copied = cluster_copy_stats.get(cluster_id, 0)
                f.write(f"Cluster {cluster_id}:\n")
                f.write(f"  Size: {len(indices)} structures\n")
                f.write(f"  Files copied: {files_copied}\n")
                f.write(f"  Directory: cluster_{cluster_id}\n")
                f.write("  Files:\n")
                for idx in indices:
                    if idx < len(self.file_names):
                        f.write(f"    - {self.file_names[idx]}\n")
                f.write("\n")
        
        logger.info(f"Cluster structures exported to: {clusters_dir}")
        logger.info(f"Cluster information saved to: {cluster_info_file}")
        
        return clusters_dir

    # ==========================================================
    # (7) 可视化
    # ==========================================================
    def visualize_results(self, save_path=None, show_plot=False):
        """
        生成丰富的可视化结果
        
        参数：
        - save_path: 保存路径
        - show_plot: 是否显示图形
        """
        if self.coarse_labels is None:
            raise ValueError("No coarse clustering results found")

        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')

        # 生成粗聚类结果
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('AF3 Coarse Clustering Results', fontsize=16)
        
        self._plot_clustering_subplot(axes[0, 0], axes[0, 1], self.coarse_labels, "Coarse")
        self._plot_clustering_subplot(axes[1, 0], axes[1, 1], self.coarse_labels, "Coarse", is_second=True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        # 导出单图和数据
        if save_path:
            self._export_individual_plots(save_path)

    def _plot_clustering_subplot(self, ax1, ax2, labels, cluster_type, is_second=False):
        """绘制聚类子图"""
        if labels is None:
            return
            
        # 颜色映射
        unique_labels = np.unique(labels)
        non_noise_labels = [l for l in unique_labels if l != -1]
        label_to_color = {l: self.palette[i % len(self.palette)] for i, l in enumerate(non_noise_labels)}
        if -1 in unique_labels:
            label_to_color[-1] = '#000000'

        if not is_second:
            # 1. 聚类标签分布
            for label in unique_labels:
                color = label_to_color[label]
                mask = labels == label
                indices = np.where(mask)[0]
                ax1.scatter(indices, [label] * len(indices),
                           c=[color], alpha=0.7, s=30,
                           label=f'Cluster {label}' if label != -1 else 'Noise')
            ax1.set_title(f'{cluster_type} Cluster Assignment')
            ax1.legend()

            # 2. 簇大小分布
            cluster_ids = [l for l in unique_labels if l != -1]
            cluster_sizes = [int(np.sum(labels == l)) for l in cluster_ids]
            if cluster_sizes:
                bar_colors = [label_to_color.get(int(cid), self.palette[int(cid) % len(self.palette)]) for cid in cluster_ids]
                ax2.bar(range(len(cluster_sizes)), cluster_sizes, color=bar_colors)
                ax2.set_title(f'{cluster_type} Cluster Size Distribution')
                ax2.set_xlabel('Cluster ID')
                ax2.set_ylabel('Size')
        else:
            # 3. t-SNE 可视化（基于Jaccard距离）
            if len(self.contact_sets) > 1:
                D = self.jaccard_distance_matrix(self.contact_sets)
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(D)-1)), metric='precomputed', init='random')
                X_tsne = tsne.fit_transform(D)
                for label in unique_labels:
                    color = label_to_color[label]
                    mask = labels == label
                    ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[color], alpha=0.7, s=30)
                ax1.set_title(f'{cluster_type} t-SNE Visualization (Jaccard Distance)')

            # 4. 接触集大小分布
            contact_sizes = [len(cs) for cs in self.contact_sets]
            ax2.hist(contact_sizes, bins=20, alpha=0.7, color=self.palette[0])
            ax2.set_title(f'{cluster_type} Contact Set Size Distribution')
            ax2.set_xlabel('Number of Contacts')
            ax2.set_ylabel('Frequency')

    def _export_individual_plots(self, save_path, X_tsne=None):
        """导出单个图表和数据"""
        base_dir = Path(save_path).with_suffix("").parent
        base_name = Path(save_path).stem.replace("_clustering", "")
        
        fig_dir = base_dir / "figures"
        data_dir = base_dir / "data"
        fig_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        labels = self.coarse_labels
        if labels is None:
            logger.warning("No labels available for visualization")
            return
            
        unique_labels = np.unique(labels)
        non_noise_labels = [l for l in unique_labels if l != -1]
        label_to_color = {l: self.palette[i % len(self.palette)] for i, l in enumerate(non_noise_labels)}
        if -1 in unique_labels:
            label_to_color[-1] = '#000000'

        # 1. Cluster Assignment
        fig1, ax = plt.subplots(figsize=(8, 6))
        for label in unique_labels:
            color = label_to_color[label]
            mask = labels == label
            indices = np.where(mask)[0]
            ax.scatter(indices, [label] * len(indices), c=[color], alpha=0.7, s=30)
        ax.set_title('AF3 Coarse Cluster Assignment')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Cluster Label')
        fig1.savefig(fig_dir / f"{base_name}_cluster_assignment.svg", format='svg', bbox_inches='tight')
        plt.close(fig1)

        # 2. Cluster Sizes
        cluster_ids = [l for l in unique_labels if l != -1]
        cluster_sizes = [int(np.sum(labels == l)) for l in cluster_ids]
        fig2, ax = plt.subplots(figsize=(8, 6))
        bar_colors = [label_to_color.get(int(cid), self.palette[int(cid) % len(self.palette)]) for cid in cluster_ids]
        ax.bar([str(cid) for cid in cluster_ids], cluster_sizes, color=bar_colors)
        ax.set_title('AF3 Coarse Cluster Size Distribution')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Size')
        fig2.savefig(fig_dir / f"{base_name}_cluster_sizes.svg", format='svg', bbox_inches='tight')
        plt.close(fig2)

        # 3. t-SNE
        if X_tsne is not None:
            fig3, ax = plt.subplots(figsize=(8, 6))
            for label in unique_labels:
                color = label_to_color[label]
                mask = labels == label
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.7, s=30, label=str(label), c=color)
            ax.set_title('AF3 Coarse t-SNE Visualization')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.legend()
            fig3.savefig(fig_dir / f"{base_name}_tsne.svg", format='svg', bbox_inches='tight')
            plt.close(fig3)

        # 4. Contact Size Distribution
        contact_sizes = [len(cs) for cs in self.contact_sets]
        fig4, ax = plt.subplots(figsize=(8, 6))
        ax.hist(contact_sizes, bins=20, alpha=0.7, color=self.palette[0])
        ax.set_title('AF3 Coarse Contact Set Size Distribution')
        ax.set_xlabel('Number of Contacts')
        ax.set_ylabel('Frequency')
        fig4.savefig(fig_dir / f"{base_name}_contact_distribution.svg", format='svg', bbox_inches='tight')
        plt.close(fig4)

        # 导出数据CSV
        self._export_data_csvs(data_dir, base_name, X_tsne)

    def _export_data_csvs(self, data_dir, base_name, X_tsne=None):
        """导出数据CSV文件"""
        labels = self.coarse_labels
        if labels is None:
            logger.warning("No labels available for data export")
            return
            
        # 聚类结果
        with open(data_dir / f"{base_name}_clustering_results.csv", 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(["file_name", "coarse_label", "contact_size"])
            for fn, cl, cs in zip(self.file_names, labels, self.contact_sets):
                writer.writerow([fn, int(cl), len(cs)])

        # 簇大小
        unique_labels = np.unique(labels)
        cluster_ids = [l for l in unique_labels if l != -1]
        cluster_sizes = [int(np.sum(labels == l)) for l in cluster_ids]
        with open(data_dir / f"{base_name}_cluster_sizes.csv", 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(["cluster_id", "size"])
            for cid, sz in zip(cluster_ids, cluster_sizes):
                writer.writerow([int(cid), int(sz)])

        # t-SNE数据
        if X_tsne is not None:
            with open(data_dir / f"{base_name}_tsne.csv", 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(["index", "tsne1", "tsne2", "label"])
                for i in range(len(X_tsne)):
                    writer.writerow([i, float(X_tsne[i,0]), float(X_tsne[i,1]), int(labels[i])])

        # 接触集统计
        contact_sizes = [len(cs) for cs in self.contact_sets]
        with open(data_dir / f"{base_name}_contact_stats.csv", 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(["file_name", "contact_size"])
            for fn, cs in zip(self.file_names, contact_sizes):
                writer.writerow([fn, cs])

    def plot_cluster_radial_tree(self, save_path=None, show_plot=False):
        """绘制径向树图"""
        if self.coarse_labels is None:
            logger.warning("No clustering results to plot radial tree")
            return

        # 计算簇质心（基于接触集特征）
        labels = np.array(self.coarse_labels)
        valid_mask = labels != -1
        if not np.any(valid_mask):
            logger.warning("No valid clusters to plot radial tree")
            return

        # 将接触集转换为特征向量
        unique_labels = np.unique(labels[valid_mask])
        centroids = []
        cluster_sizes = []
        
        for lab in unique_labels:
            cluster_mask = labels == lab
            cluster_contact_sets = [self.contact_sets[i] for i in np.where(cluster_mask)[0]]
            
            # 计算接触集特征（接触数量、残基分布等）
            contact_sizes = [len(cs) for cs in cluster_contact_sets]
            if contact_sizes:
                centroid_features = [
                    np.mean(contact_sizes),
                    np.std(contact_sizes),
                    len(set().union(*cluster_contact_sets))  # 总残基数
                ]
            else:
                centroid_features = [0, 0, 0]
            centroids.append(centroid_features)
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
        
        # 角度分布
        start_angle = np.pi/2
        angles = np.linspace(start_angle, start_angle - 2*np.pi, n_clusters, endpoint=False)
        angle_to_label = {angles[i]: unique_labels[leaves[i]] for i in range(n_clusters)}
        
        # 设置随机种子
        np.random.seed(42)
        
        # 绘制径向树
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        ax.set_rticks([])
        ax.set_thetagrids([])
        ax.grid(False)
        
        center_radius = 0.1
        leaf_radius = 0.8
        
        for i, angle in enumerate(angles):
            # 连接线
            ax.plot([0, angle], [center_radius, leaf_radius], 
                   color='#666666', linewidth=1.5, alpha=0.7)
            
            # 节点
            label = angle_to_label[angle]
            cluster_size = cluster_sizes[leaves[i]]
            color = self.palette[label % len(self.palette)]
            
            # 根据簇大小计算节点数量
            max_cluster_size = max(cluster_sizes)
            if max_cluster_size > 0:
                node_count = max(1, min(10, int(round(cluster_size / max_cluster_size * 10))))
            else:
                node_count = 1
            
            node_size = 80
            
            if node_count == 1:
                ax.scatter([angle], [leaf_radius], c=[color], s=node_size, 
                          edgecolors='white', linewidth=2, zorder=5, alpha=0.9)
            else:
                angle_spread = 0.2
                radius_spread = 0.03
                
                if node_count <= 3:
                    for j in range(node_count):
                        offset = (j - (node_count-1)/2) * angle_spread / max(1, node_count-1)
                        node_angle = angle + offset
                        node_radius = leaf_radius + (np.random.random() - 0.5) * radius_spread * 0.5
                        ax.scatter([node_angle], [node_radius], c=[color], s=node_size, 
                                  edgecolors='white', linewidth=2, zorder=5, alpha=0.9)
                else:
                    grid_size = int(np.ceil(np.sqrt(node_count)))
                    for j in range(node_count):
                        row = j // grid_size
                        col = j % grid_size
                        rel_x = (col - (grid_size-1)/2) / max(1, grid_size-1)
                        rel_y = (row - (grid_size-1)/2) / max(1, grid_size-1)
                        node_angle = angle + rel_x * angle_spread
                        node_radius = leaf_radius + rel_y * radius_spread
                        ax.scatter([node_angle], [node_radius], c=[color], s=node_size, 
                                  edgecolors='white', linewidth=2, zorder=5, alpha=0.9)
            
            # 标签
            label_text = f'Cluster {label} ({cluster_size})'
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
        ax.set_rlim(0, 1.2)
        
        if save_path:
            fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
            logger.info(f"Radial tree saved to {save_path}")
            
            # 导出数据
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
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    # ==========================================================
    # 辅助方法
    # ==========================================================
    def get_available_chains(self, structure_file):
        """获取结构文件中的可用链信息"""
        try:
            structure = self.load_structure(structure_file)
            if structure is None:
                return {}
            
            chain_info = {}
            for chain in structure[0]:
                chain_id = chain.id
                residue_count = len(list(chain))
                ca_count = len([r for r in chain if "CA" in r])
                
                # 获取链的额外信息
                chain_info[chain_id] = {
                    "residue_count": residue_count,
                    "ca_count": ca_count,
                    "chain_type": "protein" if ca_count > 0 else "other"
                }
                
                # 如果是CIF文件，尝试获取更多信息
                if str(structure_file).lower().endswith('.cif'):
                    try:
                        # 获取链的实体信息（如果可用）
                        if hasattr(chain, 'entity_id'):
                            chain_info[chain_id]["entity_id"] = chain.entity_id
                    except:
                        pass
                        
            return chain_info
        except Exception as e:
            logger.warning(f"Error getting chain info for {structure_file}: {e}")
            return {}

    def analyze_structure_composition(self):
        """分析结构文件的链组成（基于前5个文件的样本）"""
        if not self.file_names:
            logger.warning("No structures loaded")
            return {}
        
        chain_summary = {}
        file_types = {"cif": 0, "pdb": 0}
        
        # 只分析前5个文件作为样本，以提高性能
        sample_files = self.file_names[:5]
        logger.info(f"Analyzing chain composition from {len(sample_files)} sample files out of {len(self.file_names)} total files")
        
        for fname in sample_files:
            file_path = self.pdb_dir / fname
            file_ext = file_path.suffix.lower()
            if file_ext == ".cif":
                file_types["cif"] += 1
            elif file_ext == ".pdb":
                file_types["pdb"] += 1
                
            chain_info = self.get_available_chains(file_path)
            for chain_id, info in chain_info.items():
                if chain_id not in chain_summary:
                    chain_summary[chain_id] = {
                        "count": 0,
                        "total_residues": 0,
                        "total_ca": 0,
                        "chain_type": info.get("chain_type", "unknown")
                    }
                chain_summary[chain_id]["count"] += 1
                chain_summary[chain_id]["total_residues"] += info["residue_count"]
                chain_summary[chain_id]["total_ca"] += info["ca_count"]
        
        # 计算平均值
        for chain_id in chain_summary:
            count = chain_summary[chain_id]["count"]
            chain_summary[chain_id]["avg_residues"] = chain_summary[chain_id]["total_residues"] / count
            chain_summary[chain_id]["avg_ca"] = chain_summary[chain_id]["total_ca"] / count
        
        # 记录文件类型统计
        logger.info(f"File type distribution (sample of 5 files): {file_types}")
        logger.info(f"Total files processed: {len(self.file_names)}")
        
        return chain_summary

    def suggest_chain_configuration(self, chain_summary):
        """基于链分析结果建议链配置"""
        if not chain_summary:
            return None, []
        
        available_chains = list(chain_summary.keys())
        
        # 按CA原子数量排序（假设CA原子多的链是蛋白质链）
        protein_chains = [(chain_id, info["avg_ca"]) for chain_id, info in chain_summary.items() 
                         if info["avg_ca"] > 10]  # 至少10个CA原子
        protein_chains.sort(key=lambda x: x[1], reverse=True)
        
        if len(protein_chains) >= 2:
            # 假设CA原子最多的链是抗体/受体
            suggested_chainA = protein_chains[0][0]
            suggested_antigen_chains = [chain_id for chain_id, _ in protein_chains[1:]]
        elif len(available_chains) >= 2:
            # 如果没有明显的蛋白质链，使用字母顺序
            suggested_chainA = available_chains[0]
            suggested_antigen_chains = available_chains[1:]
        else:
            # 只有一个链的情况
            suggested_chainA = available_chains[0] if available_chains else 'A'
            suggested_antigen_chains = []
        
        return suggested_chainA, suggested_antigen_chains


# ==========================================================
# 主函数入口
# ==========================================================
def validate_config(chain_config, clustering_config, output_config):
    """验证配置参数的有效性"""
    logger.info("Validating configuration...")
    
    # 验证链配置
    if not isinstance(chain_config['chainA'], str):
        raise ValueError("chainA must be a string")
    if not isinstance(chain_config['antigen_chains'], list):
        raise ValueError("antigen_chains must be a list")
    if chain_config['contact_cutoff'] <= 0:
        raise ValueError("contact_cutoff must be positive")
    if chain_config['irmsd_cutoff'] <= 0:
        raise ValueError("irmsd_cutoff must be positive")
    
    # 验证接触检测配置
    valid_contact_atom_types = ['interface']
    contact_atom_type = chain_config.get('contact_atom_type', 'interface')
    if contact_atom_type not in valid_contact_atom_types:
        raise ValueError(f"contact_atom_type must be 'interface'")
    
    valid_contact_modes = ['jaccard']
    contact_mode = chain_config.get('contact_mode', 'jaccard')
    if contact_mode not in valid_contact_modes:
        raise ValueError(f"contact_mode must be 'jaccard'")
    
    # 验证聚类配置
    valid_methods = ['hdbscan', 'kmeans', 'dbscan']
    if clustering_config['coarse_method'] not in valid_methods:
        raise ValueError(f"coarse_method must be one of {valid_methods}")
    
    # 验证输出配置
    if not isinstance(output_config['auto_adjust_chains'], bool):
        raise ValueError("auto_adjust_chains must be a boolean")
    if not isinstance(output_config['show_plots'], bool):
        raise ValueError("show_plots must be a boolean")
    if not isinstance(output_config['save_individual_plots'], bool):
        raise ValueError("save_individual_plots must be a boolean")
    if not isinstance(output_config['save_radial_tree'], bool):
        raise ValueError("save_radial_tree must be a boolean")
    
    logger.info("Configuration validation passed!")

def check_dependencies():
    """检查必要的依赖是否已安装"""
    missing_deps = []
    
    try:
        import hdbscan
    except ImportError:
        missing_deps.append("hdbscan")
    
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
    
    try:
        from Bio.PDB import PDBParser, MMCIFParser, Superimposer
    except ImportError:
        missing_deps.append("biopython")
    
    try:
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.manifold import TSNE
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist, squareform
    except ImportError:
        missing_deps.append("scipy")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {missing_deps}")
        logger.error("Please install missing packages using pip:")
        for dep in missing_deps:
            logger.error(f"  pip install {dep}")
        return False
    
    logger.info("All dependencies are available")
    return True

def get_example_configs():
    """获取示例配置"""
    configs = {
        'hdbscan_auto': {
            'coarse_method': 'hdbscan',
            'coarse_params': {
                'min_cluster_size': 'auto',
                'min_samples': 'auto'
            }
        },
        'hdbscan_manual': {
            'coarse_method': 'hdbscan',
            'coarse_params': {
                'min_cluster_size': 15,
                'min_samples': 8
            }
        },
        'kmeans': {
            'coarse_method': 'kmeans',
            'coarse_params': {
                'n_clusters': 5
            }
        },
        'dbscan': {
            'coarse_method': 'dbscan',
            'coarse_params': {
                'eps': 0.3,
                'min_samples': 5
            }
        }
    }
    return configs

def main():
    # ==========================================================
    # 用户可配置参数
    # ==========================================================
    
    # 基本路径和链配置
    PDB_DIR = "./2d4_pi3ka_renamed"
    CHAIN_CONFIG = {
        'chainA': 'A',                    # 抗体/受体链
        'antigen_chains': ['B', 'C'],     # 抗原链（可能多个）
        'contact_cutoff': 5.0,            # 接触距离阈值（Å）
        'irmsd_cutoff': 5.0,              # iRMSD阈值（Å）
        'residue_ranges': None,           # 残基范围，如 "A:1-50,70-100;B:all"
        'contact_mode': 'jaccard',        # 接触分析模式 ('jaccard')
        'contact_atom_type': 'interface',  # 接触检测原子类型: 'interface'(基于残基级别的界面原子识别)
        'interface_cutoff': 8.0,           # 界面原子识别距离阈值（Å）
        'interface_method': 'residue'     # 界面原子识别方法: 'residue'(残基级别，推荐)
    }
    
    # 聚类算法配置 - 选择以下之一：
    # 1. HDBSCAN自动参数（推荐用于大数据集）
    # 2. HDBSCAN手动参数（适合有经验的用户）
    # 3. KMeans（适合已知簇数量的情况）
    # 4. DBSCAN（适合密度不均匀的数据）
    
    # 示例：使用HDBSCAN自动参数
    CLUSTERING_CONFIG = {
        'coarse_method': 'hdbscan',       # 粗聚类方法: 'hdbscan', 'kmeans', 'dbscan'
        'coarse_params': {
            # HDBSCAN参数（当coarse_method='hdbscan'时使用）
            'min_cluster_size': 'auto',   # 'auto'或具体数值
            'min_samples': 'auto',        # 'auto'或具体数值
            
            # KMeans参数（当coarse_method='kmeans'时使用）
            'n_clusters': 5,              # 簇的数量
            
            # DBSCAN参数（当coarse_method='dbscan'时使用）
            'eps': 0.5,                   # 邻域半径
            'min_samples': 5              # 最小样本数
        }
    }
    
    # 其他配置示例（取消注释使用）：
    # 
    # # 使用KMeans
    # CLUSTERING_CONFIG = {
    #     'coarse_method': 'kmeans',
    #     'coarse_params': {'n_clusters': 8}
    # }
    # 
    # # 使用DBSCAN
    # CLUSTERING_CONFIG = {
    #     'coarse_method': 'dbscan',
    #     'coarse_params': {'eps': 0.3, 'min_samples': 5}
    # }
    # 
    # # 使用HDBSCAN手动参数
    # CLUSTERING_CONFIG = {
    #     'coarse_method': 'hdbscan',
    #     'coarse_params': {'min_cluster_size': 20, 'min_samples': 10}
    # }
    
    # 输出配置
    OUTPUT_CONFIG = {
        'auto_adjust_chains': True,       # 是否自动调整链配置
        'show_plots': False,              # 是否显示图形
        'save_individual_plots': True,    # 是否保存单独的图表
        'save_radial_tree': True          # 是否保存径向树图
    }
    
    # ==========================================================
    # 依赖检查
    # ==========================================================
    if not check_dependencies():
        logger.error("Dependency check failed. Please install missing packages.")
        return
    
    # ==========================================================
    # 配置验证
    # ==========================================================
    try:
        validate_config(CHAIN_CONFIG, CLUSTERING_CONFIG, OUTPUT_CONFIG)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # ==========================================================
    # 创建分析器
    # ==========================================================
    
    # 创建分析器
    analyzer = AF3CoarseClusterAnalyzer(
        PDB_DIR, 
        chainA=CHAIN_CONFIG['chainA'],
        antigen_chains=CHAIN_CONFIG['antigen_chains'],
        contact_cutoff=CHAIN_CONFIG['contact_cutoff'],
        irmsd_cutoff=CHAIN_CONFIG['irmsd_cutoff'],
        residue_ranges=CHAIN_CONFIG.get('residue_ranges'),
        contact_mode=CHAIN_CONFIG.get('contact_mode', 'jaccard'),
        contact_atom_type=CHAIN_CONFIG.get('contact_atom_type', 'interface')
    )
    
    # 设置界面原子参数
    if CHAIN_CONFIG.get('interface_cutoff'):
        analyzer.interface_cutoff = CHAIN_CONFIG['interface_cutoff']
    if CHAIN_CONFIG.get('interface_method'):
        analyzer.interface_method = CHAIN_CONFIG['interface_method']
    
    # 如果指定了残基范围，设置它
    if CHAIN_CONFIG.get('residue_ranges'):
        analyzer.set_residue_ranges(CHAIN_CONFIG['residue_ranges'])
    
    logger.info("Starting AF3 coarse clustering pipeline ...")
    logger.info(f"Target chains: antibody={analyzer.chainA}, antigens={analyzer.antigen_chains}")
    logger.info(f"Contact detection: {analyzer.contact_atom_type.upper()} atoms")
    logger.info(f"Clustering method: {CLUSTERING_CONFIG['coarse_method']}")
    logger.info(f"Clustering params: {CLUSTERING_CONFIG['coarse_params']}")
    
    try:
        # 加载数据
        ok = analyzer.load_and_process_data()
        if not ok:
            logger.error("Data loading failed")
            return
        
        # 检查是否有足够的数据进行聚类
        if len(analyzer.file_names) < 2:
            logger.error("Insufficient data for clustering. Need at least 2 structures.")
            return
        
        logger.info(f"Successfully loaded {len(analyzer.file_names)} structures")
        
        # 记录内存使用情况
        analyzer._log_memory_usage("after data loading")

        # 分析结构组成
        logger.info("Analyzing structure composition...")
        chain_summary = analyzer.analyze_structure_composition()
        logger.info(f"Available chains: {chain_summary}")
        
        # 保存链组成信息供后续使用
        analyzer.chain_summary = chain_summary
        
        # 验证指定的链是否存在
        if analyzer.chainA not in chain_summary:
            logger.warning(f"Chain {analyzer.chainA} not found in structures")
        else:
            logger.info(f"Chain {analyzer.chainA} found: {chain_summary[analyzer.chainA]}")
        
        missing_antigen_chains = [c for c in analyzer.antigen_chains if c not in chain_summary]
        if missing_antigen_chains:
            logger.warning(f"Antigen chains {missing_antigen_chains} not found in structures")
        else:
            for chain_id in analyzer.antigen_chains:
                if chain_id in chain_summary:
                    logger.info(f"Antigen chain {chain_id} found: {chain_summary[chain_id]}")
        
        # 提供链建议
        available_chains = list(chain_summary.keys())
        logger.info(f"Available chains: {available_chains}")
        
        # 如果指定的链不存在，提供建议
        if analyzer.chainA not in chain_summary or missing_antigen_chains:
            logger.info("Chain configuration suggestions:")
            logger.info(f"  - Current: chainA='{analyzer.chainA}', antigen_chains={analyzer.antigen_chains}")
            
            # 使用智能建议
            suggested_chainA, suggested_antigen_chains = analyzer.suggest_chain_configuration(chain_summary)
            if suggested_chainA:
                logger.info(f"  - Suggested: chainA='{suggested_chainA}', antigen_chains={suggested_antigen_chains}")
                
                # 根据配置决定是否自动调整
                if OUTPUT_CONFIG['auto_adjust_chains'] and (analyzer.chainA not in chain_summary or missing_antigen_chains):
                    logger.info("  - Auto-adjusting chain configuration...")
                    analyzer.chainA = suggested_chainA
                    analyzer.antigen_chains = suggested_antigen_chains
                    logger.info(f"  - Updated to: chainA='{analyzer.chainA}', antigen_chains={analyzer.antigen_chains}")
                elif not OUTPUT_CONFIG['auto_adjust_chains']:
                    logger.warning("  - Auto-adjustment disabled. Please manually adjust chain configuration.")
                    logger.warning("  - Set OUTPUT_CONFIG['auto_adjust_chains'] = True to enable auto-adjustment.")
        
        # 粗聚类
        logger.info("Performing coarse clustering...")
        coarse_method = CLUSTERING_CONFIG['coarse_method']
        coarse_params = CLUSTERING_CONFIG['coarse_params'].copy()
        
        # 处理自动参数
        if coarse_method == 'hdbscan':
            if coarse_params.get('min_cluster_size') == 'auto':
                del coarse_params['min_cluster_size']
            if coarse_params.get('min_samples') == 'auto':
                del coarse_params['min_samples']
        
        logger.info(f"Coarse clustering with {coarse_method} and params: {coarse_params}")
        analyzer.perform_coarse_clustering(method=coarse_method, distance_metric=CHAIN_CONFIG.get('contact_mode'), **coarse_params)
        
        # 记录内存使用情况
        analyzer._log_memory_usage("after coarse clustering")

        # 构建输出目录
        input_base = Path(PDB_DIR).resolve().name
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{input_base}_af3_coarse_cluster_{date_str}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 评估结果
        metrics = analyzer.evaluate_clustering()
        logger.info(f"Clustering metrics: {metrics}")

        # 获取代表结构
        reps = analyzer.get_cluster_representatives()
        logger.info(f"Cluster representatives: {reps}")

        # 保存结果
        base_results_pkl = output_dir / f"{input_base}_af3_coarse_cluster_{date_str}_results.pkl"
        analyzer.save_results(base_results_pkl)
        
        # 提示用户关于聚类结构文件夹的使用
        clusters_dir = output_dir / "coarse_clusters"
        if clusters_dir.exists():
            logger.info("=" * 60)
            logger.info("聚类结构文件夹已创建，可用于精细聚类:")
            logger.info(f"聚类结构文件夹路径: {clusters_dir}")
            logger.info("文件夹结构:")
            
            total_files_in_clusters = 0
            empty_clusters = []
            
            for cluster_dir in clusters_dir.iterdir():
                if cluster_dir.is_dir():
                    # 更准确的文件计数
                    pdb_files = list(cluster_dir.glob("*.pdb"))
                    cif_files = list(cluster_dir.glob("*.cif"))
                    file_count = len(pdb_files) + len(cif_files)
                    total_files_in_clusters += file_count
                    
                    if file_count == 0:
                        empty_clusters.append(cluster_dir.name)
                        logger.warning(f"  {cluster_dir.name}: {file_count} 个结构文件 (空文件夹!)")
                    else:
                        logger.info(f"  {cluster_dir.name}: {file_count} 个结构文件")
            
            # 检查总体情况
            if total_files_in_clusters == 0:
                logger.error("=" * 60)
                logger.error("警告: 所有聚类文件夹都是空的!")
                logger.error("这可能是由于以下原因:")
                logger.error("1. 聚类失败 - 所有样本被标记为噪声")
                logger.error("2. 源文件路径不正确")
                logger.error("3. 文件权限问题")
                logger.error("4. 聚类参数设置不当")
                logger.error("=" * 60)
            elif empty_clusters:
                logger.warning(f"发现 {len(empty_clusters)} 个空聚类文件夹: {empty_clusters}")
                logger.warning("这可能是正常的（如果某些聚类确实没有样本）")
            else:
                logger.info(f"总计: {total_files_in_clusters} 个结构文件分布在 {len(list(clusters_dir.iterdir()))} 个聚类文件夹中")
            
            logger.info("=" * 60)
            logger.info("在精细聚类脚本中，可以设置:")
            logger.info(f"COARSE_CLUSTERS_DIR = '{clusters_dir}'")
            logger.info("这样精细聚类将优先使用这些已分组的结构文件")
            logger.info("=" * 60)
        
        # 粗聚类可视化
        if OUTPUT_CONFIG['save_individual_plots']:
            try:
                coarse_viz_path = output_dir / f"{input_base}_af3_coarse_cluster_{date_str}_clustering.png"
                logger.info(f"Generating coarse clustering visualization...")
                analyzer.visualize_results(save_path=coarse_viz_path, show_plot=OUTPUT_CONFIG['show_plots'])
                logger.info(f"Coarse clustering visualization saved to {coarse_viz_path}")
                
                # 导出单独的图表和数据
                logger.info(f"Exporting individual plots and data...")
                analyzer._export_individual_plots(coarse_viz_path)
                logger.info(f"Individual plots and data exported successfully")
                
            except Exception as e:
                logger.error(f"Failed to generate coarse clustering visualization: {e}")
                logger.exception("Visualization error details:")
        
        # 径向树图（基于粗聚类结果）
        if OUTPUT_CONFIG['save_radial_tree'] and analyzer.coarse_labels is not None:
            try:
                radial_svg = output_dir / f"{input_base}_af3_coarse_cluster_{date_str}_radial_tree.svg"
                logger.info(f"Generating coarse clustering radial tree...")
                analyzer.plot_cluster_radial_tree(save_path=radial_svg, show_plot=OUTPUT_CONFIG['show_plots'])
                logger.info(f"Coarse clustering radial tree saved to {radial_svg}")
            except Exception as e:
                logger.error(f"Failed to generate coarse clustering radial tree: {e}")
                logger.exception("Radial tree error details:")
        
        logger.info("All AF3 coarse clustering tasks finished. Exiting.")
        
    except Exception:
        logger.exception("AF3 coarse clustering failed with an unexpected error")


if __name__ == "__main__":
    main()
