#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced AF3 Complex Clustering Pipeline
基于optimized_protein_clustering_v14.3.py的优秀特性优化

功能：
1. 从 AF3 预测的 CIF/PDB 文件中提取蛋白质复合物接触信息
2. 两阶段聚类：Jaccard距离粗聚类 + Foldseek+US-align精细聚类
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

精细聚类方案：
- 仅支持 Foldseek + US-align 方案，放弃传统RMSD对比方案
- 需要安装 Foldseek 和 US-align 工具
- Foldseek参数：sensitivity(敏感度), max_evalue(E值), coverage(覆盖度), search_type(搜索类型)

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


class AF3ClusterAnalyzer:
    """
    AF3 复合物聚类分析器类
    - 输入：AF3预测的PDB文件目录
    - 输出：两阶段聚类结果 + 丰富可视化
    - 精细聚类：仅使用 Foldseek + US-align 方案
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
        self.fine_labels = None    # 精细聚类标签
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

    def load_and_process_data(self, cache_file="af3_data_cache.pkl"):
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

    def superimpose_chainA(self, ref_struct, mob_struct, chainA="A"):
        """将mob_struct的chainA对齐到ref_struct"""
        ref_atoms, mob_atoms = [], []
        try:
            # 限制最大残基数量以避免过长的计算
            max_residues = 1000
            ref_chain = list(ref_struct[0][chainA])[:max_residues]
            mob_chain = list(mob_struct[0][chainA])[:max_residues]
            
            for r1, r2 in zip(ref_chain, mob_chain):
                if "CA" in r1 and "CA" in r2:
                    ref_atoms.append(r1["CA"])
                    mob_atoms.append(r2["CA"])
                    
            if len(ref_atoms) >= 3:
                sup = Superimposer()
                sup.set_atoms(ref_atoms, mob_atoms)
                sup.apply(mob_struct.get_atoms())
            else:
                logger.debug(f"Insufficient CA atoms for superimposition: {len(ref_atoms)}")
        except Exception as e:
            logger.debug(f"Superimposition failed: {e}")
            # 不抛出异常，继续执行

    def calc_iRMSD(self, ref_struct, mob_struct, antigen_chains=None, cutoff=8.0):
        """计算iRMSD（基于Cα原子，考虑多个抗原链）"""
        if antigen_chains is None:
            antigen_chains = self.antigen_chains
            
        try:
            # 获取界面残基
            contacts_ref = self.get_contacts(ref_struct, self.chainA, antigen_chains, cutoff)
            contacts_mob = self.get_contacts(mob_struct, self.chainA, antigen_chains, cutoff)
            
            # 提取抗原残基信息（格式：chain_id:res_id）
            iface_ref = {j for (i, j) in contacts_ref}
            iface_mob = {j for (i, j) in contacts_mob}
            common_iface = iface_ref & iface_mob
            
            if len(common_iface) == 0:
                return 10.0
            
            # 按链分组收集原子
            ref_atoms, mob_atoms = [], []
            
            for res_info in common_iface:
                try:
                    chain_id, res_id = res_info.split(':', 1)
                    res_id = int(res_id)
                    
                    # 从参考结构获取原子
                    if chain_id in ref_struct[0] and res_id in ref_struct[0][chain_id]:
                        if "CA" in ref_struct[0][chain_id][res_id]:
                            ref_atoms.append(ref_struct[0][chain_id][res_id]["CA"])
                    
                    # 从移动结构获取原子
                    if chain_id in mob_struct[0] and res_id in mob_struct[0][chain_id]:
                        if "CA" in mob_struct[0][chain_id][res_id]:
                            mob_atoms.append(mob_struct[0][chain_id][res_id]["CA"])
                            
                except (ValueError, KeyError) as e:
                    logger.debug(f"Error processing residue {res_info}: {e}")
                    continue
            
            if len(ref_atoms) < 3 or len(mob_atoms) < 3:
                return 10.0
            
            # 确保原子数量匹配
            min_atoms = min(len(ref_atoms), len(mob_atoms))
            ref_atoms = ref_atoms[:min_atoms]
            mob_atoms = mob_atoms[:min_atoms]
            
            # 计算RMSD
            diffs = [a.get_coord() - b.get_coord() for a, b in zip(ref_atoms, mob_atoms)]
            return np.sqrt(np.mean([np.dot(d, d) for d in diffs]))
            
        except Exception as e:
            logger.warning(f"iRMSD calculation failed: {e}")
            return 10.0

    # ==========================================================
    # (3) 聚类方法
    # ==========================================================
    def perform_coarse_clustering(self, method='hdbscan', distance_metric=None, **kwargs):
        """
        第一阶段：基于接触集的粗聚类
        
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

        return self.coarse_labels

    def perform_fine_clustering(self, **kwargs):
        """
        第二阶段：基于Foldseek + US-align的精细聚类（只对最大的几个簇进行）
        仅保留Foldseek + US-align方案，放弃传统RMSD对比方案
        """
        if self.coarse_labels is None:
            logger.error("No coarse clustering results available")
            return None

        self.fine_labels = -1 * np.ones(len(self.coarse_labels), dtype=int)
        cluster_id = 0

        # 统计粗聚类结果并按大小排序
        unique_coarse_labels = set(self.coarse_labels)
        coarse_cluster_sizes = {}
        for label in unique_coarse_labels:
            if label != -1:
                size = np.sum(self.coarse_labels == label)
                coarse_cluster_sizes[label] = size
        
        # 按簇大小排序，最大的在前
        sorted_clusters = sorted(coarse_cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # 获取要精细聚类的簇数量（仅前2个最大的簇）
        max_clusters_to_refine = kwargs.get('max_clusters_to_refine', 2)
        clusters_to_refine = sorted_clusters[:max_clusters_to_refine]
        
        # 最大簇大小限制，避免过大的簇
        max_cluster_size_for_refine = kwargs.get('max_cluster_size_for_refine', 200)
        
        logger.info(f"Coarse clustering results (sorted by size): {dict(sorted_clusters)}")
        logger.info(f"Will perform fine clustering on top {len(clusters_to_refine)} largest clusters: {[c[0] for c in clusters_to_refine]}")
        logger.info(f"Max cluster size for refine: {max_cluster_size_for_refine}")

        # 处理所有粗聚类簇
        for cluster_idx, (coarse_cluster, cluster_size) in enumerate(sorted_clusters):
            idx = [i for i, l in enumerate(self.coarse_labels) if l == coarse_cluster]
            
            # 检查是否需要进行精细聚类
            if coarse_cluster in [c[0] for c in clusters_to_refine]:
                logger.info(f"Processing coarse cluster {coarse_cluster} ({cluster_idx+1}/{len(sorted_clusters)}) with {cluster_size} structures - FOLDSEEK REFINE")
                
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

                # 使用改进的精细聚类方法
                try:
                    logger.info(f"  - Performing improved fine clustering for {cluster_size} structures...")
                    
                    # 获取参数
                    k_neighbors = kwargs.get('k_neighbors', 50)
                    clustering_method = kwargs.get('clustering_method', 'hdbscan')
                    use_usalign = kwargs.get('use_usalign', True)
                    
                    # 从kwargs中移除已提取的参数，避免重复传递
                    kwargs_clean = kwargs.copy()
                    kwargs_clean.pop('k_neighbors', None)
                    kwargs_clean.pop('clustering_method', None)
                    kwargs_clean.pop('use_usalign', None)
                    
                    # 尝试Foldseek + US-align聚类
                    sparse_labels = self._foldseek_usalign_clustering(idx, k_neighbors, clustering_method, use_usalign, **kwargs_clean)
                    
                    # 如果Foldseek失败，直接使用US-align进行全距离计算
                    if sparse_labels is None or np.all(sparse_labels == -1):
                        logger.warning("  - Foldseek failed, proceeding with direct US-align clustering")
                        sparse_labels = self._direct_usalign_clustering(idx, clustering_method, **kwargs_clean)
                    
                    # 检查聚类结果
                    if sparse_labels is None or len(sparse_labels) == 0:
                        logger.warning("  - All clustering methods failed, keeping coarse cluster labels")
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
                        
                        # 统计子聚类结果
                        n_sub_clusters = len([l for l in set(sparse_labels) if l != -1])
                        n_noise = np.sum(sparse_labels == -1)
                        
                        # 记录该粗聚类的精细聚类结果
                        self._record_fine_cluster_results(coarse_cluster, cluster_size, n_sub_clusters, n_noise, sparse_labels, idx)
                        logger.info(f"  - Fine clustering complete: {n_sub_clusters} clusters, {n_noise} noise points")
                        
                except Exception as e:
                    logger.warning(f"  - Fine clustering failed: {e}, keeping coarse cluster labels")
                    for i in idx:
                        self.fine_labels[i] = cluster_id
                    cluster_id += 1

        logger.info(f"Fine clustering completed. Total fine clusters: {cluster_id}")
        logger.info(f"Refined {len(clusters_to_refine)} clusters, kept {len(sorted_clusters) - len(clusters_to_refine)} clusters as coarse labels")

        return self.fine_labels



    def _fast_neighbor_clustering(self, idx, k_neighbors, clustering_method='hdbscan', **kwargs):
        """
        基于快速近邻搜索的聚类
        
        参数：
        - idx: 结构索引列表
        - k_neighbors: 每个结构的前K个近邻
        - clustering_method: 聚类方法 ('hdbscan', 'spectral')
        """
        cluster_size = len(idx)
        
        # 步骤1: 快速近邻搜索
        logger.info(f"    - Step 1: Fast neighbor search (k={k_neighbors})")
        neighbor_matrix = self._fast_neighbor_search(idx, k_neighbors)
        
        # 步骤2: 构建稀疏距离矩阵
        logger.info(f"    - Step 2: Building sparse distance matrix")
        sparse_distance_matrix = self._build_sparse_distance_matrix(idx, neighbor_matrix, **kwargs)
        
        # 步骤3: 稀疏图聚类
        logger.info(f"    - Step 3: Sparse graph clustering using {clustering_method}")
        if clustering_method == 'hdbscan':
            labels = self._sparse_hdbscan_clustering(sparse_distance_matrix, **kwargs)
        elif clustering_method == 'spectral':
            labels = self._sparse_spectral_clustering(sparse_distance_matrix, **kwargs)
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")
        
        return labels

    def _fast_neighbor_search(self, idx, k_neighbors):
        """
        快速近邻搜索（模拟 Foldseek/US-align 的功能）
        
        这里我们使用基于接触集的快速相似性搜索作为替代
        在实际应用中，可以集成真正的 Foldseek/US-align
        """
        cluster_size = len(idx)
        neighbor_matrix = np.zeros((cluster_size, cluster_size), dtype=bool)
        
        # 计算接触集的Jaccard相似性作为快速相似性度量
        contact_sets = [self.contact_sets[i] for i in idx]
        
        # 为每个结构找到前k个最相似的邻居
        for i in range(cluster_size):
            similarities = []
            for j in range(cluster_size):
                if i != j:
                    # 计算Jaccard相似性
                    inter = len(contact_sets[i] & contact_sets[j])
                    union = len(contact_sets[i] | contact_sets[j])
                    similarity = inter / union if union > 0 else 0
                    similarities.append((j, similarity))
            
            # 选择前k个最相似的邻居
            similarities.sort(key=lambda x: x[1], reverse=True)
            for j, sim in similarities[:k_neighbors]:
                neighbor_matrix[i, j] = True
                neighbor_matrix[j, i] = True  # 确保对称性
        
        logger.info(f"      - Found {np.sum(neighbor_matrix) // 2} neighbor pairs")
        return neighbor_matrix

    def _build_sparse_distance_matrix(self, idx, neighbor_matrix, **kwargs):
        """
        构建稀疏距离矩阵
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
                
                # 使用更大的批处理块
                batch_size = max(100, len(tasks) // (self.n_jobs * 2))
                
                for result in pool.imap_unordered(
                    self._compute_single_irmsd_pair, 
                    tasks, 
                    chunksize=batch_size
                ):
                    i, j, distance = result
                    sparse_distances[(i, j)] = distance
                    sparse_distances[(j, i)] = distance  # 确保对称性
        except Exception as e:
            logger.error(f"Error in parallel iRMSD computation: {e}")
            logger.warning("Falling back to sequential computation")
            # 回退到顺序计算
            for i, j in zip(neighbor_pairs[0], neighbor_pairs[1]):
                if i < j:  # 只计算上三角矩阵
                    try:
                        result = self._compute_single_irmsd_pair((i, j, idx[i], idx[j]))
                        i, j, distance = result
                        sparse_distances[(i, j)] = distance
                        sparse_distances[(j, i)] = distance  # 确保对称性
                    except Exception as inner_e:
                        logger.debug(f"Error computing iRMSD for pair {i}-{j}: {inner_e}")
                        sparse_distances[(i, j)] = 10.0
                        sparse_distances[(j, i)] = 10.0
        
        return sparse_distances

    def _compute_single_irmsd_pair(self, task):
        """
        计算单个iRMSD对（用于并行处理）
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
            return (i, j, 10.0)  # 默认大距离

    def _sparse_hdbscan_clustering(self, sparse_distances, **kwargs):
        """
        在稀疏距离矩阵上使用HDBSCAN聚类
        """
        import hdbscan
        
        # 获取当前簇的大小（从稀疏距离的键中推断）
        all_indices = set()
        for (i, j) in sparse_distances.keys():
            all_indices.add(i)
            all_indices.add(j)
        cluster_size = max(all_indices) + 1 if all_indices else 0
        
        if cluster_size == 0:
            return np.array([])
        
        # HDBSCAN参数 - 自动调整
        auto_min_cluster_size, auto_min_samples = self._auto_hdbscan_params(cluster_size)
        min_cluster_size = kwargs.get('fine_min_cluster_size', auto_min_cluster_size)
        min_samples = kwargs.get('min_samples', auto_min_samples)
        
        logger.info(f"      - Fine HDBSCAN params -> min_cluster_size={min_cluster_size}, min_samples={min_samples} (auto-adjusted for cluster_size={cluster_size})")
        
        # 使用HDBSCAN进行聚类
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='precomputed',
            core_dist_n_jobs=self.n_jobs
        )
        
        # 构建完整的距离矩阵
        distance_matrix = np.full((cluster_size, cluster_size), 10.0)  # 默认大距离
        np.fill_diagonal(distance_matrix, 0.0)  # 对角线设为0
        
        # 填充已知的距离
        for (i, j), distance in sparse_distances.items():
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # 确保对称性
        
        # 聚类
        labels = clusterer.fit_predict(distance_matrix)
        
        return labels

    def _sparse_spectral_clustering(self, sparse_distances, **kwargs):
        """
        在稀疏距离矩阵上使用谱聚类
        """
        from sklearn.cluster import SpectralClustering
        
        # 获取当前簇的大小（从稀疏距离的键中推断）
        all_indices = set()
        for (i, j) in sparse_distances.keys():
            all_indices.add(i)
            all_indices.add(j)
        cluster_size = max(all_indices) + 1 if all_indices else 0
        
        if cluster_size == 0:
            return np.array([])
        
        # 谱聚类参数 - 自动调整
        auto_n_clusters = max(2, min(10, cluster_size // 5))  # 基于簇大小自动调整
        n_clusters = kwargs.get('n_clusters', auto_n_clusters)
        
        logger.info(f"      - Fine Spectral clustering params -> n_clusters={n_clusters} (auto-adjusted for cluster_size={cluster_size})")
        
        # 使用谱聚类
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        # 构建完整的距离矩阵
        distance_matrix = np.full((cluster_size, cluster_size), 10.0)  # 默认大距离
        np.fill_diagonal(distance_matrix, 0.0)  # 对角线设为0
        
        # 填充已知的距离
        for (i, j), distance in sparse_distances.items():
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # 确保对称性
        
        # 聚类
        labels = clusterer.fit_predict(distance_matrix)
        
        return labels



    # ==========================================================
    # (4) 聚类评估
    # ==========================================================
    def evaluate_clustering(self):
        """评估聚类质量"""
        if self.fine_labels is None:
            return {"silhouette": None, "n_clusters": 0, "noise_ratio": 1.0}

        # 计算Jaccard距离用于评估
        D = self.jaccard_distance_matrix(self.contact_sets)
        
        # 轮廓系数
        try:
            score = silhouette_score(D, self.fine_labels, metric='precomputed')
        except:
            score = None

        # 统计信息
        unique_labels = set(self.fine_labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        noise_count = np.sum(self.fine_labels == -1)
        noise_ratio = noise_count / len(self.fine_labels)

        return {
            "silhouette": score,
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "total_samples": len(self.fine_labels)
        }

    # ==========================================================
    # (5) 获取簇代表
    # ==========================================================
    def get_cluster_representatives(self):
        """获取每个簇的代表性结构（medoid）"""
        if self.fine_labels is None:
            return {}

        D = self.jaccard_distance_matrix(self.contact_sets)
        reps = {}
        
        for label in set(self.fine_labels):
            if label == -1:
                continue
            indices = np.where(self.fine_labels == label)[0]
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
    def save_results(self, filename, save_type="both"):
        """
        保存聚类结果
        
        参数：
        - filename: 基础文件名
        - save_type: 保存类型 ("coarse", "fine", "both")
        """
        if self.coarse_labels is None:
            logger.error("No clustering results to save")
            return

        if save_type in ["coarse", "both"]:
            # 保存粗聚类结果
            coarse_results = {
                "file_names": self.file_names,
                "coarse_labels": self.coarse_labels,
                "contact_sets": self.contact_sets,
                "clustering_type": "coarse"
            }
            
            coarse_filename = str(filename).replace("_results", "_coarse_results")
            with open(coarse_filename, "wb") as f:
                pickle.dump(coarse_results, f)
            logger.info(f"Coarse clustering results saved to {coarse_filename}")

            # 保存粗聚类CSV摘要
            try:
                coarse_csv_path = Path(coarse_filename).with_suffix('.csv')
                with open(coarse_csv_path, 'w', newline='') as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["file_name", "coarse_label", "contact_size"])
                    for fn, cl, cs in zip(self.file_names, self.coarse_labels, self.contact_sets):
                        writer.writerow([fn, int(cl), len(cs)])
                logger.info(f"Coarse clustering CSV saved to {coarse_csv_path}")
            except Exception as e:
                logger.warning(f"Failed to save coarse clustering CSV: {e}")

        if save_type in ["fine", "both"] and self.fine_labels is not None:
            # 保存精细聚类结果
            fine_results = {
                "file_names": self.file_names,
                "coarse_labels": self.coarse_labels,
                "fine_labels": self.fine_labels,
                "contact_sets": self.contact_sets,
                "clustering_type": "fine"
            }
            
            fine_filename = str(filename).replace("_results", "_fine_results")
            with open(fine_filename, "wb") as f:
                pickle.dump(fine_results, f)
            logger.info(f"Fine clustering results saved to {fine_filename}")

            # 保存精细聚类CSV摘要
            try:
                fine_csv_path = Path(fine_filename).with_suffix('.csv')
                with open(fine_csv_path, 'w', newline='') as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["file_name", "coarse_label", "fine_label", "contact_size"])
                    for fn, cl, fl, cs in zip(self.file_names, self.coarse_labels, self.fine_labels, self.contact_sets):
                        writer.writerow([fn, int(cl), int(fl), len(cs)])
                logger.info(f"Fine clustering CSV saved to {fine_csv_path}")
            except Exception as e:
                logger.warning(f"Failed to save fine clustering CSV: {e}")

        # 保存综合结果（包含所有信息）
        if save_type == "both":
            combined_results = {
                "file_names": self.file_names,
                "coarse_labels": self.coarse_labels,
                "fine_labels": self.fine_labels,
                "contact_sets": self.contact_sets,
                "clustering_type": "combined"
            }
            
            with open(filename, "wb") as f:
                pickle.dump(combined_results, f)
            logger.info(f"Combined results saved to {filename}")

            # 保存综合CSV摘要
            try:
                csv_path = Path(filename).with_suffix('.csv')
                with open(csv_path, 'w', newline='') as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["file_name", "coarse_label", "fine_label", "contact_size"])
                    for fn, cl, fl, cs in zip(self.file_names, self.coarse_labels, self.fine_labels, self.contact_sets):
                        writer.writerow([fn, int(cl), int(fl), len(cs)])
                logger.info(f"Combined CSV summary saved to {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to save combined CSV summary: {e}")

    # ==========================================================
    # (7) 可视化
    # ==========================================================
    def visualize_results(self, save_path=None, show_plot=False, clustering_type="fine"):
        """
        生成丰富的可视化结果
        
        参数：
        - save_path: 保存路径
        - show_plot: 是否显示图形
        - clustering_type: 聚类类型 ("coarse", "fine", "both")
        """
        if clustering_type == "coarse" and self.coarse_labels is None:
            raise ValueError("No coarse clustering results found")
        elif clustering_type == "fine" and self.fine_labels is None:
            raise ValueError("No fine clustering results found")
        elif clustering_type == "both" and (self.coarse_labels is None or self.fine_labels is None):
            raise ValueError("Both coarse and fine clustering results required")

        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')

        if clustering_type == "both":
            # 生成两个子图：粗聚类和精细聚类
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AF3 Complex Clustering Results (Coarse + Fine)', fontsize=16)
            
            # 粗聚类结果
            self._plot_clustering_subplot(axes[0, 0], axes[0, 1], self.coarse_labels, "Coarse")
            
            # 精细聚类结果
            self._plot_clustering_subplot(axes[1, 0], axes[1, 1], self.fine_labels, "Fine")
            
        else:
            # 生成单个聚类结果
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            cluster_type = "Coarse" if clustering_type == "coarse" else "Fine"
            fig.suptitle(f'AF3 Complex Clustering Results ({cluster_type})', fontsize=16)
            
            labels = self.coarse_labels if clustering_type == "coarse" else self.fine_labels
            self._plot_clustering_subplot(axes[0, 0], axes[0, 1], labels, cluster_type)
            self._plot_clustering_subplot(axes[1, 0], axes[1, 1], labels, cluster_type, is_second=True)

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
            self._export_individual_plots(save_path, clustering_type=clustering_type)

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

    def _export_individual_plots(self, save_path, X_tsne=None, clustering_type="fine"):
        """导出单个图表和数据"""
        base_dir = Path(save_path).with_suffix("").parent
        base_name = Path(save_path).stem.replace("_clustering", "")
        
        fig_dir = base_dir / "figures"
        data_dir = base_dir / "data"
        fig_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        # 根据聚类类型选择标签
        if clustering_type == "coarse":
            labels = self.coarse_labels
        elif clustering_type == "fine":
            labels = self.fine_labels
        elif clustering_type == "both":
            labels = self.fine_labels if self.fine_labels is not None else self.coarse_labels
        else:
            labels = self.fine_labels if self.fine_labels is not None else self.coarse_labels
            
        if labels is None:
            logger.warning(f"No labels available for clustering_type: {clustering_type}")
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
        ax.set_title('AF3 Cluster Assignment')
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
        ax.set_title('AF3 Cluster Size Distribution')
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
            ax.set_title('AF3 t-SNE Visualization')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.legend()
            fig3.savefig(fig_dir / f"{base_name}_tsne.svg", format='svg', bbox_inches='tight')
            plt.close(fig3)

        # 4. Contact Size Distribution
        contact_sizes = [len(cs) for cs in self.contact_sets]
        fig4, ax = plt.subplots(figsize=(8, 6))
        ax.hist(contact_sizes, bins=20, alpha=0.7, color=self.palette[0])
        ax.set_title('AF3 Contact Set Size Distribution')
        ax.set_xlabel('Number of Contacts')
        ax.set_ylabel('Frequency')
        fig4.savefig(fig_dir / f"{base_name}_contact_distribution.svg", format='svg', bbox_inches='tight')
        plt.close(fig4)

        # 导出数据CSV
        self._export_data_csvs(data_dir, base_name, X_tsne, clustering_type)
        
        # 导出链组成信息
        if hasattr(self, 'chain_summary'):
            chain_csv = data_dir / f"{base_name}_chain_composition.csv"
            with open(chain_csv, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(["chain_id", "count", "avg_residues", "avg_ca"])
                for chain_id, info in self.chain_summary.items():
                    writer.writerow([chain_id, info["count"], 
                                   round(info["avg_residues"], 2), 
                                   round(info["avg_ca"], 2)])

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

    def _export_data_csvs(self, data_dir, base_name, X_tsne=None, clustering_type="fine"):
        """导出数据CSV文件"""
        # 根据聚类类型选择标签
        if clustering_type == "coarse":
            labels = self.coarse_labels
        elif clustering_type == "fine":
            labels = self.fine_labels
        elif clustering_type == "both":
            labels = self.fine_labels if self.fine_labels is not None else self.coarse_labels
        else:
            labels = self.fine_labels if self.fine_labels is not None else self.coarse_labels
            
        if labels is None:
            logger.warning(f"No labels available for clustering_type: {clustering_type}")
            return
            
        # 聚类结果
        with open(data_dir / f"{base_name}_clustering_results.csv", 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(["file_name", "coarse_label", "fine_label", "contact_size"])
            for fn, cl, fl, cs in zip(self.file_names, self.coarse_labels, self.fine_labels, self.contact_sets):
                writer.writerow([fn, int(cl), int(fl), len(cs)])

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

    def plot_cluster_radial_tree(self, save_path=None, show_plot=False, use_coarse_labels=False):
        """绘制径向树图"""
        # 选择使用粗聚类还是精细聚类标签
        if use_coarse_labels:
            labels = self.coarse_labels
            if labels is None:
                logger.warning("No coarse clustering results to plot radial tree")
                return
        else:
            labels = self.fine_labels
            if labels is None:
                logger.warning("No clustering results to plot radial tree")
                return

        # 计算簇质心（基于接触集特征）
        labels = np.array(labels)
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

    def _fast_neighbor_search_foldseek(self, idx, k_neighbors, **kwargs):
        """
        使用真正的 Foldseek 进行快速近邻搜索
        
        参数：
        - idx: 结构索引列表
        - k_neighbors: 每个结构的前K个近邻
        """
        import subprocess
        import tempfile
        import os
        from pathlib import Path
        
        cluster_size = len(idx)
        neighbor_matrix = np.zeros((cluster_size, cluster_size), dtype=bool)
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 步骤1: 准备结构文件
            logger.info(f"      - Step 1: Preparing structure files for Foldseek")
            structure_files = self._prepare_structure_files_for_foldseek(idx, temp_path)
            
            # 步骤2: 创建 Foldseek 数据库
            logger.info(f"      - Step 2: Creating Foldseek database")
            database_path = temp_path / "structures_db"
            self._create_foldseek_database(structure_files, database_path, **kwargs)
            
            # 步骤3: 运行 Foldseek 搜索
            logger.info(f"      - Step 3: Running Foldseek search (k={k_neighbors})")
            foldseek_results = self._run_foldseek_search(
                database_path, database_path, temp_path, k_neighbors, **kwargs
            )
            
            # 步骤4: 解析结果并构建邻居矩阵
            logger.info(f"      - Step 4: Parsing Foldseek results")
            neighbor_matrix = self._parse_foldseek_results(foldseek_results, cluster_size, k_neighbors)
        
        logger.info(f"      - Found {np.sum(neighbor_matrix) // 2} neighbor pairs via Foldseek")
        return neighbor_matrix

    def _prepare_structure_files_for_foldseek(self, idx, temp_path):
        """
        为 Foldseek 准备结构文件
        """
        structure_files = []
        
        logger.info(f"        - Preparing {len(idx)} structure files for Foldseek")
        
        for i, struct_idx in enumerate(idx):
            # 获取原始文件路径
            original_file = self.pdb_dir / self.file_names[struct_idx]
            
            # 创建临时文件（确保格式正确）
            temp_file = temp_path / f"struct_{i:04d}.pdb"
            
            # 检查原始文件是否存在
            if not original_file.exists():
                logger.warning(f"        - Original file not found: {original_file}")
                continue
            
            try:
                # 如果是CIF文件，转换为PDB格式
                if original_file.suffix.lower() == '.cif':
                    self._convert_cif_to_pdb(original_file, temp_file)
                else:
                    # 复制PDB文件
                    import shutil
                    shutil.copy2(original_file, temp_file)
                
                # 验证转换后的文件是否存在且非空
                if temp_file.exists() and temp_file.stat().st_size > 0:
                    # 简单验证文件格式（检查是否包含ATOM或HETATM记录）
                    try:
                        with open(temp_file, 'r') as f:
                            first_lines = [f.readline().strip() for _ in range(10)]
                        if any(line.startswith(('ATOM', 'HETATM')) for line in first_lines):
                            # 使用绝对路径
                            abs_path = str(temp_file.resolve())
                            structure_files.append(abs_path)
                            if i < 3:  # 记录前几个文件用于调试
                                logger.debug(f"        - Prepared file {i}: {abs_path} (size: {temp_file.stat().st_size} bytes)")
                        else:
                            logger.warning(f"        - File does not contain valid PDB records: {temp_file}")
                    except Exception as read_error:
                        logger.warning(f"        - Error reading file {temp_file}: {read_error}")
                else:
                    logger.warning(f"        - Failed to create valid file: {temp_file}")
                    
            except Exception as e:
                logger.warning(f"        - Error preparing file {i} ({original_file}): {e}")
                continue
        
        logger.info(f"        - Successfully prepared {len(structure_files)} structure files")
        
        if len(structure_files) == 0:
            logger.error("        - No valid structure files prepared!")
            
        return structure_files

    def _convert_cif_to_pdb(self, cif_file, pdb_file):
        """
        将CIF文件转换为PDB格式（Foldseek需要PDB格式）
        """
        try:
            from Bio.PDB import MMCIFParser, PDBIO
            
            # 检查输入文件
            if not cif_file.exists():
                raise Exception(f"CIF file does not exist: {cif_file}")
            
            # 解析CIF文件
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('temp', cif_file)
            
            # 检查结构是否为空
            if len(list(structure.get_chains())) == 0:
                raise Exception("No chains found in structure")
            
            # 保存为PDB格式
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb_file))
            
            # 验证输出文件
            if not pdb_file.exists() or pdb_file.stat().st_size == 0:
                raise Exception("Failed to create valid PDB file")
            
        except Exception as e:
            logger.warning(f"Failed to convert {cif_file} to PDB: {e}")
            # 如果转换失败，尝试直接复制
            try:
                import shutil
                shutil.copy2(cif_file, pdb_file)
                logger.info(f"        - Copied CIF file directly: {cif_file} -> {pdb_file}")
            except Exception as copy_error:
                logger.error(f"        - Failed to copy file: {copy_error}")
                raise

    def _create_foldseek_database(self, structure_files, database_path, **kwargs):
        """
        创建 Foldseek 数据库
        """
        import subprocess
        
        foldseek_path = kwargs.get('foldseek_path', 'foldseek')
        
        # 验证输入文件
        if not structure_files:
            raise Exception("No structure files provided for database creation")
        
        logger.info(f"        - Creating database from {len(structure_files)} structure files")
        
        # 创建结构文件列表
        list_file = database_path.parent / "structures_list.txt"
        with open(list_file, 'w') as f:
            for i, struct_file in enumerate(structure_files):
                # 验证文件是否存在且可读
                import os
                if os.path.exists(struct_file) and os.access(struct_file, os.R_OK):
                    f.write(f"{struct_file}\n")
                    if i < 3:  # 记录前几个文件用于调试
                        logger.debug(f"        - Added to list: {struct_file}")
                else:
                    logger.warning(f"        - File not accessible: {struct_file}")
        
        # 验证列表文件
        if not list_file.exists():
            raise Exception(f"Failed to create list file: {list_file}")
        
        list_file_size = list_file.stat().st_size
        logger.info(f"        - List file created: {list_file} (size: {list_file_size} bytes)")
        
        if list_file_size == 0:
            raise Exception("List file is empty")
        
        # 构建创建数据库的命令
        cmd = [
            foldseek_path, 'createdb',
            str(list_file), str(database_path)
        ]
        
        try:
            logger.info(f"        - Creating database with command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # 记录详细的输出信息
            if result.stdout:
                logger.debug(f"        - Foldseek stdout: {result.stdout.strip()}")
            if result.stderr:
                logger.debug(f"        - Foldseek stderr: {result.stderr.strip()}")
            
            if result.returncode != 0:
                logger.warning(f"Foldseek database creation failed: {result.stderr}")
                raise Exception(f"Database creation failed: {result.stderr}")
            
            # 验证数据库文件是否创建成功
            if not database_path.exists():
                raise Exception(f"Database file not created: {database_path}")
            
            logger.info(f"        - Database created successfully at {database_path}")
            
        except subprocess.TimeoutExpired:
            logger.error("Foldseek database creation timed out")
            raise
        except FileNotFoundError:
            logger.error(f"Foldseek not found at: {foldseek_path}")
            raise
        except Exception as e:
            logger.error(f"Foldseek database creation error: {e}")
            raise

    def _run_foldseek_search(self, query_db, target_db, temp_path, k_neighbors, **kwargs):
        """
        运行 Foldseek 搜索
        """
        import subprocess
        import os
        
        # Foldseek 参数
        foldseek_path = kwargs.get('foldseek_path', 'foldseek')  # Foldseek可执行文件路径
        search_type = kwargs.get('search_type', '3di+aa')  # 搜索类型：3di, tm, 3di+aa(默认)
        sensitivity = kwargs.get('sensitivity', 9.5)  # 敏感度：1-9，默认9.5
        max_evalue = kwargs.get('max_evalue', 0.001)  # 最大E值，默认0.001
        coverage = kwargs.get('coverage', 0.0)  # 覆盖度阈值，默认0.0
        
        # 根据search_type确定alignment_type值
        if search_type == '3di':
            alignment_type_value = 0  # 3Di Gotoh-Smith-Waterman
        elif search_type == 'tm':
            alignment_type_value = 1  # TMalign
        elif search_type == '3di+aa':
            alignment_type_value = 2  # 3Di+AA (默认)
        else:
            alignment_type_value = 2  # 默认使用3Di+AA
        
        # 参数说明：
        # -s: 敏感度，较低值更快，较高值更敏感 (fast: 7.5, default: 9.5)
        # -e: E值阈值，增加可报告更远的结构 (default: 0.001)
        # -c: 覆盖度阈值，更高覆盖度=更全局对齐 (default: 0.0)
        # --alignment-type: 0=3Di Gotoh-Smith-Waterman, 1=TMalign, 2=3Di+AA(默认)
        
        # 输出文件
        result_file = temp_path / "foldseek_results.m8"
        
        # 构建 Foldseek 命令（简化参数以避免版本兼容性问题）
        cmd = [
            foldseek_path, 'search',
            str(query_db), str(target_db),
            str(result_file), str(temp_path / "tmp"),
            '--max-seqs', str(k_neighbors),
            '-s', str(sensitivity),  # 敏感度
            '-e', str(max_evalue),   # E值阈值
            '-c', str(coverage),     # 覆盖度阈值
            '--alignment-type', str(alignment_type_value),  # 对齐类型
            '--threads', str(self.n_jobs)
        ]
        
        try:
            logger.info(f"        - Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5分钟超时
            )
            
            if result.returncode != 0:
                logger.warning(f"Foldseek search failed: {result.stderr}")
                return []
            
            # 读取结果
            with open(result_file, 'r') as f:
                results = f.readlines()
            
            # 添加调试信息
            if results:
                logger.info(f"        - Foldseek returned {len(results)} results")
                logger.info(f"        - First result line: {results[0].strip()}")
            else:
                logger.warning("        - Foldseek returned no results")
            
            return results
            
        except subprocess.TimeoutExpired:
            logger.error("Foldseek search timed out")
            return []
        except FileNotFoundError:
            logger.error(f"Foldseek not found at: {foldseek_path}")
            logger.error("Please install Foldseek or specify correct path")
            return []
        except Exception as e:
            logger.error(f"Foldseek search error: {e}")
            return []

    def _parse_foldseek_results(self, foldseek_results, cluster_size, k_neighbors):
        """
        解析 Foldseek 结果并构建邻居矩阵
        """
        neighbor_matrix = np.zeros((cluster_size, cluster_size), dtype=bool)
        
        # 为每个查询结构记录前k个邻居
        query_neighbors = {}
        
        logger.info(f"        - Parsing {len(foldseek_results)} Foldseek result lines")
        
        for line in foldseek_results:
            if line.strip():
                parts = line.strip().split('\t')
                # 尝试不同的输出格式
                if len(parts) >= 7:
                    # 格式：query target evalue qstart qend tstart tend
                    query_idx = int(parts[0])
                    target_idx = int(parts[1])
                    evalue = float(parts[2])
                elif len(parts) >= 3:
                    # 格式：query target evalue
                    query_idx = int(parts[0])
                    target_idx = int(parts[1])
                    evalue = float(parts[2])
                else:
                    continue
                
                # 跳过自匹配
                if query_idx == target_idx:
                    continue
                
                # 记录邻居
                if query_idx not in query_neighbors:
                    query_neighbors[query_idx] = []
                
                # 只保留前k个邻居
                if len(query_neighbors[query_idx]) < k_neighbors:
                    query_neighbors[query_idx].append((target_idx, evalue))
        
        # 构建邻居矩阵
        for query_idx, neighbors in query_neighbors.items():
            for target_idx, evalue in neighbors:
                neighbor_matrix[query_idx, target_idx] = True
                neighbor_matrix[target_idx, query_idx] = True  # 确保对称性
        
        return neighbor_matrix

    def _compute_precise_rmsd_usalign(self, struct1_file, struct2_file, **kwargs):
        """
        使用 US-align 计算精确的 RMSD
        
        参数：
        - struct1_file: 结构文件1路径
        - struct2_file: 结构文件2路径
        """
        import subprocess
        import tempfile
        import gc
        
        usalign_path = kwargs.get('usalign_path', '/mnt/share/public/USalign')  # US-align可执行文件路径
        
        try:
            # 构建 US-align 命令
            cmd = [
                usalign_path,
                str(struct1_file),
                str(struct2_file),
                '-outfmt', '1'  # 详细输出格式，包含RMSD
            ]
            
            # 运行 US-align（增加内存管理和错误处理）
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=120  # 2分钟超时
                )
            except MemoryError:
                logger.warning("Memory error during US-align execution, forcing garbage collection")
                gc.collect()
                return 10.0  # 返回默认大距离
            
            # 调试：记录前几个US-align调用的输出
            if hasattr(self, '_usalign_debug_count'):
                self._usalign_debug_count += 1
            else:
                self._usalign_debug_count = 1
            
            if self._usalign_debug_count <= 3:  # 只记录前3次调用的输出
                logger.debug(f"US-align command: {' '.join(cmd)}")
                logger.debug(f"US-align stdout: {result.stdout.strip()}")
                logger.debug(f"US-align stderr: {result.stderr.strip()}")
            
            if result.returncode != 0:
                logger.warning(f"US-align failed: {result.stderr}")
                return 10.0  # 默认大距离
            
            # 解析输出
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) >= 1:
                # 尝试多种US-align输出格式
                for line in output_lines:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            # 尝试直接解析RMSD（第3个字段）
                            rmsd = float(parts[2])
                            return rmsd
                        except (ValueError, IndexError):
                            continue
                    
                    # 如果上面的方法失败，尝试查找包含RMSD的行
                    if 'RMSD' in line or 'rmsd' in line:
                        try:
                            # 查找数字
                            import re
                            numbers = re.findall(r'\d+\.?\d*', line)
                            if numbers:
                                rmsd = float(numbers[-1])  # 取最后一个数字作为RMSD
                                return rmsd
                        except (ValueError, IndexError):
                            continue
                    
                    # 尝试查找包含"TM1"的行（可能是TM-score格式）
                    if 'TM1' in line:
                        try:
                            import re
                            # 查找TM1后面的数字
                            tm1_match = re.search(r'TM1\s*=\s*(\d+\.?\d*)', line)
                            if tm1_match:
                                tm_score = float(tm1_match.group(1))
                                # 将TM-score转换为RMSD（近似转换）
                                if tm_score > 0.5:
                                    rmsd = 2.0 * (1.0 - tm_score)  # 近似转换
                                else:
                                    rmsd = 10.0
                                return rmsd
                        except (ValueError, IndexError):
                            continue
                    
                    # 尝试查找任何数字作为RMSD
                    try:
                        import re
                        numbers = re.findall(r'\d+\.?\d*', line)
                        if numbers:
                            # 尝试解析最后一个数字
                            potential_rmsd = float(numbers[-1])
                            if 0.1 <= potential_rmsd <= 20.0:  # 合理的RMSD范围
                                return potential_rmsd
                    except (ValueError, IndexError):
                        continue
            
            # 如果所有解析方法都失败，记录原始输出用于调试
            logger.debug(f"US-align output parsing failed. Raw output: {result.stdout.strip()}")
            return 10.0  # 默认大距离
            
        except subprocess.TimeoutExpired:
            logger.warning("US-align timed out")
            return 10.0
        except FileNotFoundError:
            logger.error(f"US-align not found at: {usalign_path}")
            return 10.0
        except MemoryError:
            logger.warning("Memory error in US-align computation")
            gc.collect()
            return 10.0
        except Exception as e:
            logger.warning(f"US-align error: {e}")
            return 10.0

    def _build_sparse_distance_matrix_usalign(self, idx, neighbor_matrix, **kwargs):
        """
        使用 US-align 构建稀疏距离矩阵
        """
        import tempfile
        from pathlib import Path
        
        cluster_size = len(idx)
        sparse_distances = {}
        
        # 获取邻居对
        neighbor_pairs = np.where(neighbor_matrix)
        total_pairs = len(neighbor_pairs[0]) // 2
        
        logger.info(f"      - Computing US-align RMSD for {total_pairs} neighbor pairs")
        
        # 创建临时目录
        import gc
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 准备结构文件
            structure_files = self._prepare_structure_files_for_foldseek(idx, temp_path)
            
            # 并行计算邻居对的US-align RMSD（减少并行度以防止内存不足）
            max_processes = min(4, self.n_jobs)  # 限制最大进程数
            with Pool(processes=max_processes) as pool:
                tasks = []
                usalign_path = kwargs.get('usalign_path', '/mnt/share/public/USalign')
                for i, j in zip(neighbor_pairs[0], neighbor_pairs[1]):
                    if i < j:  # 只计算上三角矩阵
                        struct1_file = structure_files[i]
                        struct2_file = structure_files[j]
                        tasks.append((i, j, struct1_file, struct2_file, usalign_path))
                
                # 使用较小的批处理大小
                batch_size = max(10, len(tasks) // (max_processes * 4))
                
                # 添加内存监控
                completed_pairs = 0
                for result in pool.imap_unordered(
                    self._compute_single_usalign_pair, 
                    tasks, 
                    chunksize=batch_size
                ):
                    i, j, distance = result
                    sparse_distances[(i, j)] = distance
                    sparse_distances[(j, i)] = distance  # 确保对称性
                    
                    # 每完成一定数量的计算后检查内存
                    completed_pairs += 1
                    if completed_pairs % 50 == 0:
                        self._log_memory_usage(f"after {completed_pairs} US-align pairs")
                        gc.collect()  # 强制垃圾回收
        
        return sparse_distances

    def _compute_single_usalign_pair(self, task):
        """
        计算单个US-align对（用于并行处理）
        """
        if len(task) == 4:
            i, j, struct1_file, struct2_file = task
            usalign_path = '/mnt/share/public/USalign'  # 默认路径
        else:
            i, j, struct1_file, struct2_file, usalign_path = task
        
        try:
            distance = self._compute_precise_rmsd_usalign(struct1_file, struct2_file, usalign_path=usalign_path)
            return (i, j, distance)
        except Exception as e:
            logger.debug(f"Error in US-align computation for pair {i}-{j}: {e}")
            return (i, j, 10.0)  # 默认大距离



    def _foldseek_usalign_clustering(self, idx, k_neighbors, clustering_method, use_usalign, **kwargs):
        """
        使用Foldseek + US-align进行聚类
        
        参数：
        - idx: 结构索引列表
        - k_neighbors: 每个结构的前K个近邻
        - clustering_method: 聚类方法 ('hdbscan', 'spectral')
        - use_usalign: 是否使用US-align计算精确RMSD
        """
        if not idx:
            logger.warning("Empty index list provided to _foldseek_usalign_clustering")
            return np.array([])
            
        cluster_size = len(idx)
        
        # 验证参数
        if k_neighbors <= 0:
            logger.warning(f"Invalid k_neighbors={k_neighbors}, setting to 1")
            k_neighbors = 1
        elif k_neighbors >= cluster_size:
            logger.warning(f"k_neighbors={k_neighbors} >= cluster_size={cluster_size}, setting to {cluster_size-1}")
            k_neighbors = max(1, cluster_size - 1)
        
        try:
            # 步骤1: Foldseek快速近邻搜索
            logger.info(f"    - Step 1: Foldseek neighbor search (k={k_neighbors})")
            neighbor_matrix = self._fast_neighbor_search_foldseek(idx, k_neighbors, **kwargs)
            
            # 检查邻居矩阵是否为空
            if neighbor_matrix is None or neighbor_matrix.size == 0:
                logger.warning("Neighbor matrix is empty, returning individual labels")
                return np.array([-1] * cluster_size)
            
            # 步骤2: 构建稀疏距离矩阵
            if use_usalign:
                logger.info(f"    - Step 2: Computing US-align RMSD for neighbor pairs")
                sparse_distances = self._build_sparse_distance_matrix_usalign(idx, neighbor_matrix, **kwargs)
            else:
                logger.info(f"    - Step 2: Computing iRMSD for neighbor pairs")
                sparse_distances = self._build_sparse_distance_matrix(idx, neighbor_matrix, **kwargs)
            
            # 检查稀疏距离矩阵是否为空
            if not sparse_distances:
                logger.warning("Sparse distance matrix is empty, returning individual labels")
                return np.array([-1] * cluster_size)
            
            # 步骤3: 稀疏图聚类
            logger.info(f"    - Step 3: Sparse graph clustering using {clustering_method}")
            if clustering_method == 'hdbscan':
                labels = self._sparse_hdbscan_clustering(sparse_distances, **kwargs)
            elif clustering_method == 'spectral':
                labels = self._sparse_spectral_clustering(sparse_distances, **kwargs)
            else:
                raise ValueError(f"Unsupported clustering method: {clustering_method}")
            
            # 验证返回的标签
            if labels is None or len(labels) == 0:
                logger.warning("Clustering returned empty labels, returning individual labels")
                return np.array([-1] * cluster_size)
            
            return labels
            
        except Exception as e:
            logger.error(f"Error in _foldseek_usalign_clustering: {e}")
            logger.warning("Foldseek failed, proceeding with direct US-align clustering")
            # 尝试直接US-align聚类作为备选方案
            try:
                return self._direct_usalign_clustering(idx, clustering_method, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Direct US-align clustering also failed: {fallback_error}")
                logger.warning("Returning individual labels due to error")
                return np.array([-1] * cluster_size)

    def _sparse_clustering_from_distances(self, sparse_distances, clustering_method, **kwargs):
        """
        从稀疏距离字典进行聚类
        """
        if clustering_method == 'hdbscan':
            return self._sparse_hdbscan_clustering(sparse_distances, **kwargs)
        elif clustering_method == 'spectral':
            return self._sparse_spectral_clustering(sparse_distances, **kwargs)
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")

    def _direct_usalign_clustering(self, idx, clustering_method, **kwargs):
        """
        直接使用US-align计算所有结构对的距离并进行聚类
        
        参数：
        - idx: 结构索引列表
        - clustering_method: 聚类方法 ('hdbscan', 'spectral')
        """
        cluster_size = len(idx)
        
        if cluster_size < 2:
            logger.warning("Cluster size < 2, cannot perform clustering")
            return np.array([0] * cluster_size)
        
        try:
            logger.info(f"    - Computing all pairwise US-align distances for {cluster_size} structures")
            
            # 创建临时目录
            import tempfile
            import gc
            from pathlib import Path
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # 准备结构文件
                structure_files = self._prepare_structure_files_for_foldseek(idx, temp_path)
                
                # 计算所有结构对的距离
                distance_matrix = np.zeros((cluster_size, cluster_size))
                
                # 并行计算所有对的距离
                from multiprocessing import Pool
                # 并行计算所有对的US-align RMSD（减少并行度以防止内存不足）
                max_processes = min(2, self.n_jobs)  # 进一步限制进程数
                with Pool(processes=max_processes) as pool:
                    tasks = []
                    usalign_path = kwargs.get('usalign_path', '/mnt/share/public/USalign')
                    for i in range(cluster_size):
                        for j in range(i+1, cluster_size):  # 只计算上三角矩阵
                            struct1_file = structure_files[i]
                            struct2_file = structure_files[j]
                            tasks.append((i, j, struct1_file, struct2_file, usalign_path))
                    
                    # 使用较小的批处理大小
                    batch_size = max(5, len(tasks) // (max_processes * 8))
                    
                    # 添加内存监控
                    completed_pairs = 0
                    for result in pool.imap_unordered(
                        self._compute_single_usalign_pair, 
                        tasks, 
                        chunksize=batch_size
                    ):
                        i, j, distance = result
                        distance_matrix[i, j] = distance
                        distance_matrix[j, i] = distance  # 确保对称性
                        
                        # 每完成一定数量的计算后检查内存
                        completed_pairs += 1
                        if completed_pairs % 20 == 0:  # 更频繁的内存检查
                            self._log_memory_usage(f"after {completed_pairs} direct US-align pairs")
                            gc.collect()  # 强制垃圾回收
            
            # 执行聚类
            logger.info(f"    - Performing {clustering_method} clustering on full distance matrix")
            
            if clustering_method == 'hdbscan':
                # HDBSCAN参数 - 自动调整
                auto_min_cluster_size, auto_min_samples = self._auto_hdbscan_params(cluster_size)
                min_cluster_size = kwargs.get('fine_min_cluster_size', auto_min_cluster_size)
                min_samples = kwargs.get('min_samples', auto_min_samples)
                
                logger.info(f"      - HDBSCAN params -> min_cluster_size={min_cluster_size}, min_samples={min_samples}")
                
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='precomputed',
                    core_dist_n_jobs=self.n_jobs
                )
                
            elif clustering_method == 'spectral':
                # 谱聚类参数 - 自动调整
                auto_n_clusters = max(2, min(10, cluster_size // 5))
                n_clusters = kwargs.get('n_clusters', auto_n_clusters)
                
                logger.info(f"      - Spectral clustering params -> n_clusters={n_clusters}")
                
                from sklearn.cluster import SpectralClustering
                clusterer = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported clustering method: {clustering_method}")
            
            # 执行聚类
            labels = clusterer.fit_predict(distance_matrix)
            
            # 验证结果
            if labels is None or len(labels) == 0:
                logger.warning("Clustering returned empty labels")
                return None
            
            # 统计结果
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            logger.info(f"      - Clustering complete: {n_clusters} clusters, {n_noise} noise points")
            
            return labels
            
        except Exception as e:
            logger.error(f"Error in _direct_usalign_clustering: {e}")
            return None

    def _record_fine_cluster_results(self, coarse_cluster, cluster_size, n_sub_clusters, n_noise, labels, idx):
        """
        记录单个粗聚类的精细聚类结果
        """
        if not hasattr(self, 'fine_cluster_results'):
            self.fine_cluster_results = {}
        
        self.fine_cluster_results[coarse_cluster] = {
            'coarse_cluster': coarse_cluster,
            'cluster_size': cluster_size,
            'n_sub_clusters': n_sub_clusters,
            'n_noise': n_noise,
            'labels': labels.copy(),
            'indices': idx.copy(),
            'file_names': [self.file_names[i] for i in idx]
        }

    def export_individual_fine_cluster_results(self, save_path):
        """
        导出每个粗聚类的精细聚类结果
        """
        if not hasattr(self, 'fine_cluster_results') or not self.fine_cluster_results:
            logger.warning("No fine cluster results to export")
            return
        
        logger.info("Exporting individual fine cluster results...")
        
        for coarse_cluster, results in self.fine_cluster_results.items():
            # 创建该粗聚类的结果目录
            cluster_dir = save_path / f"coarse_cluster_{coarse_cluster}_fine_results"
            cluster_dir.mkdir(exist_ok=True)
            
            # 导出CSV文件
            csv_file = cluster_dir / f"coarse_cluster_{coarse_cluster}_fine_clustering.csv"
            with open(csv_file, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(['File_Name', 'Coarse_Cluster', 'Fine_Cluster', 'Is_Noise'])
                
                for i, (file_name, label) in enumerate(zip(results['file_names'], results['labels'])):
                    is_noise = 'Yes' if label == -1 else 'No'
                    writer.writerow([file_name, coarse_cluster, label, is_noise])
            
            # 导出统计信息
            stats_file = cluster_dir / f"coarse_cluster_{coarse_cluster}_fine_stats.txt"
            with open(stats_file, 'w') as f:
                f.write(f"Coarse Cluster {coarse_cluster} Fine Clustering Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"Original cluster size: {results['cluster_size']}\n")
                f.write(f"Number of sub-clusters: {results['n_sub_clusters']}\n")
                f.write(f"Number of noise points: {results['n_noise']}\n")
                f.write(f"Noise ratio: {results['n_noise']/results['cluster_size']:.3f}\n")
                f.write("\nSub-cluster sizes:\n")
                
                # 统计每个子簇的大小
                unique_labels = set(results['labels'])
                if -1 in unique_labels:
                    unique_labels.remove(-1)
                
                for label in sorted(unique_labels):
                    size = np.sum(np.array(results['labels']) == label)
                    f.write(f"  Sub-cluster {label}: {size} structures\n")
                
                if -1 in results['labels']:
                    f.write(f"  Noise points: {results['n_noise']} structures\n")
            
            logger.info(f"  - Exported fine clustering results for coarse cluster {coarse_cluster} to {cluster_dir}")


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
    
    # 验证精细聚类配置
    fine_params = clustering_config.get('fine_params', {})
    if 'k_neighbors' in fine_params and fine_params['k_neighbors'] <= 0:
        raise ValueError("k_neighbors must be positive")
    if 'max_clusters_to_refine' in fine_params and fine_params['max_clusters_to_refine'] < 0:
        raise ValueError("max_clusters_to_refine must be non-negative")
    if 'max_cluster_size_for_refine' in fine_params and fine_params['max_cluster_size_for_refine'] <= 0:
        raise ValueError("max_cluster_size_for_refine must be positive")
    
    # 验证聚类方法
    clustering_method = fine_params.get('clustering_method', 'hdbscan')
    valid_fine_methods = ['hdbscan', 'spectral']
    if clustering_method not in valid_fine_methods:
        raise ValueError(f"fine_params.clustering_method must be one of {valid_fine_methods}")
    
    # 验证输出配置
    if not isinstance(output_config['auto_adjust_chains'], bool):
        raise ValueError("auto_adjust_chains must be a boolean")
    if not isinstance(output_config['show_plots'], bool):
        raise ValueError("show_plots must be a boolean")
    if not isinstance(output_config['save_individual_plots'], bool):
        raise ValueError("save_individual_plots must be a boolean")
    if not isinstance(output_config['save_radial_tree'], bool):
        raise ValueError("save_radial_tree must be a boolean")
    if not isinstance(output_config['fast_mode'], bool):
        raise ValueError("fast_mode must be a boolean")
    
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

def check_external_tools(fine_params):
    """检查外部工具（Foldseek, US-align）是否可用"""
    import subprocess
    import shutil
    
    tools_status = {}
    
    # 检查Foldseek
    foldseek_path = fine_params.get('foldseek_path', 'foldseek')
    try:
        if shutil.which(foldseek_path):
            tools_status['foldseek'] = True
            logger.info(f"Foldseek found at: {shutil.which(foldseek_path)}")
        else:
            tools_status['foldseek'] = False
            logger.warning(f"Foldseek not found at: {foldseek_path}")
    except Exception as e:
        tools_status['foldseek'] = False
        logger.warning(f"Error checking Foldseek: {e}")
    
    # 检查US-align
    usalign_path = fine_params.get('usalign_path', './USalign')
    try:
        if Path(usalign_path).exists() and os.access(usalign_path, os.X_OK):
            tools_status['usalign'] = True
            logger.info(f"US-align found at: {usalign_path}")
        else:
            tools_status['usalign'] = False
            logger.warning(f"US-align not found or not executable at: {usalign_path}")
    except Exception as e:
        tools_status['usalign'] = False
        logger.warning(f"Error checking US-align: {e}")
    
    return tools_status

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
        },
        
        # 精细聚类参数
        'fine_params': {
            'fine_min_cluster_size': 3,   # 精细聚类最小簇大小
            'max_clusters_to_refine': 2,   # 只对前N个最大的簇进行精细聚类（设为0跳过所有精细聚类）
            'max_cluster_size_for_refine': 200, # 精细聚类时最大簇大小限制（超过此大小的簇跳过精细聚类）
            'k_neighbors': 50,             # 每个结构的前K个近邻
            'clustering_method': 'hdbscan', # 稀疏图聚类方法 ('hdbscan', 'spectral')
            'min_samples': 5,              # HDBSCAN的min_samples参数
            'use_usalign': True,           # 是否使用US-align计算精确RMSD
            
            # Foldseek 参数
            'foldseek_path': '/mnt/share/public/foldseek/bin/foldseek',   # Foldseek可执行文件路径
            'search_type': '3di+aa',      # 搜索类型：3di(3Di), tm(TMalign), 3di+aa(3Di+AA,默认)
            'sensitivity': 9.5,            # 敏感度：1-9，默认9.5，较低值更快，较高值更敏感
            'max_evalue': 0.001,          # 最大E值，默认0.001，增加可报告更远的结构
            'coverage': 0.0,              # 覆盖度阈值，默认0.0，更高覆盖度=更全局对齐
            
            # US-align 参数
            'usalign_path': '/mnt/share/public/USalign'     # US-align可执行文件路径
        }
    }
    
    # 其他配置示例（取消注释使用）：
    # 
    # # 使用KMeans，只对前2个最大簇进行精细聚类
    # CLUSTERING_CONFIG = {
    #     'coarse_method': 'kmeans',
    #     'coarse_params': {'n_clusters': 8},
    #     'fine_params': {'fine_min_cluster_size': 3, 'max_clusters_to_refine': 2}
    # }
    # 
    # # 使用DBSCAN，只对前1个最大簇进行精细聚类
    # CLUSTERING_CONFIG = {
    #     'coarse_method': 'dbscan',
    #     'coarse_params': {'eps': 0.3, 'min_samples': 5},
    #     'fine_params': {'fine_min_cluster_size': 3, 'max_clusters_to_refine': 1}
    # }
    # 
    # # 使用HDBSCAN手动参数，跳过所有精细聚类
    # CLUSTERING_CONFIG = {
    #     'coarse_method': 'hdbscan',
    #     'coarse_params': {'min_cluster_size': 20, 'min_samples': 10},
    #     'fine_params': {'fine_min_cluster_size': 3, 'max_clusters_to_refine': 0}
    # }
    # 
    # 注意：精细聚类仅支持 Foldseek + US-align 方案，需要安装相应工具
    
    # 输出配置
    OUTPUT_CONFIG = {
        'auto_adjust_chains': True,       # 是否自动调整链配置
        'show_plots': False,              # 是否显示图形
        'save_individual_plots': True,    # 是否保存单独的图表
        'save_radial_tree': True,         # 是否保存径向树图
        'fast_mode': False                # 快速模式：True=跳过精细聚类，False=进行精细聚类
    }
    
    # ==========================================================
    # 依赖检查
    # ==========================================================
    if not check_dependencies():
        logger.error("Dependency check failed. Please install missing packages.")
        return
    
    # ==========================================================
    # 外部工具检查
    # ==========================================================
    tools_status = check_external_tools(CLUSTERING_CONFIG['fine_params'])
    
    # 如果外部工具不可用，给出警告和建议
    if not tools_status.get('foldseek', False):
        logger.warning("Foldseek not found. Fine clustering may fail.")
        logger.info("To install Foldseek:")
        logger.info("  1. Download from: https://github.com/steineggerlab/foldseek")
        logger.info("  2. Or use conda: conda install -c conda-forge foldseek")
    
    if not tools_status.get('usalign', False):
        logger.warning("US-align not found. Fine clustering may fail.")
        logger.info("To install US-align:")
        logger.info("  1. Download from: https://zhanggroup.org/US-align/")
        logger.info("  2. Compile: g++ -O3 -ffast-math -lm -o USalign USalign.cpp")
        logger.info("  3. Place USalign executable in the same directory as this script")
    
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
    analyzer = AF3ClusterAnalyzer(
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
    
    logger.info("Starting AF3 clustering pipeline ...")
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
        
        # 第一阶段：粗聚类
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

        # 第二阶段：精细聚类
        logger.info("Performing fine clustering...")
        fine_params = CLUSTERING_CONFIG['fine_params']
        logger.info(f"Fine clustering params: {fine_params}")
        
        # 添加粗聚类统计信息
        unique_coarse = set(analyzer.coarse_labels)
        coarse_stats = {label: np.sum(analyzer.coarse_labels == label) for label in unique_coarse}
        logger.info(f"Coarse clustering results: {coarse_stats}")
        
        # 检查是否有大簇可能导致长时间计算
        large_clusters = {label: size for label, size in coarse_stats.items() if size > 50}
        if large_clusters:
            logger.warning(f"Large clusters detected: {large_clusters}")
            logger.warning("Large clusters may take significant time for Foldseek + US-align computation")
        
        # 显示精细聚类策略
        max_refine = fine_params.get('max_clusters_to_refine', 3)
        if max_refine > 0:
            logger.info(f"Fine clustering strategy: will refine top {max_refine} largest clusters using Foldseek + US-align")
        else:
            logger.info("Fine clustering strategy: skip all fine clustering (coarse labels only)")
        
        # 检查是否跳过精细聚类
        if OUTPUT_CONFIG.get('fast_mode', False):
            logger.info("Fast mode enabled: skipping fine clustering, using coarse labels only")
            analyzer.fine_labels = analyzer.coarse_labels.copy()
        else:
            # 使用Foldseek + US-align进行精细聚类
            logger.info("Using Foldseek + US-align for fine clustering")
            analyzer.perform_fine_clustering(**fine_params)

        # 构建输出目录
        input_base = Path(PDB_DIR).resolve().name
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{input_base}_af3_cluster_{date_str}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出每个粗聚类的精细聚类结果（如果进行了精细聚类）
        if not OUTPUT_CONFIG.get('fast_mode', False):
            analyzer.export_individual_fine_cluster_results(output_dir)

        # 评估结果
        metrics = analyzer.evaluate_clustering()
        logger.info(f"Clustering metrics: {metrics}")

        # 获取代表结构
        reps = analyzer.get_cluster_representatives()
        logger.info(f"Cluster representatives: {reps}")

        # 保存结果
        base_results_pkl = output_dir / f"{input_base}_af3_cluster_{date_str}_results.pkl"
        
        # 根据是否进行精细聚类决定保存类型
        if OUTPUT_CONFIG.get('fast_mode', False):
            # 快速模式：只保存粗聚类结果
            analyzer.save_results(base_results_pkl, save_type="coarse")
            
            # 粗聚类可视化
            if OUTPUT_CONFIG['save_individual_plots']:
                try:
                    coarse_viz_path = output_dir / f"{input_base}_af3_cluster_{date_str}_coarse_clustering.png"
                    logger.info(f"Generating coarse clustering visualization...")
                    analyzer.visualize_results(save_path=coarse_viz_path, show_plot=OUTPUT_CONFIG['show_plots'], clustering_type="coarse")
                    logger.info(f"Coarse clustering visualization saved to {coarse_viz_path}")
                    
                    # 导出单独的图表和数据
                    logger.info(f"Exporting individual plots and data...")
                    analyzer._export_individual_plots(coarse_viz_path, clustering_type="coarse")
                    logger.info(f"Individual plots and data exported successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to generate coarse clustering visualization: {e}")
                    logger.exception("Visualization error details:")
            
            # 径向树图（基于粗聚类结果）
            if OUTPUT_CONFIG['save_radial_tree'] and analyzer.coarse_labels is not None:
                try:
                    radial_svg = output_dir / f"{input_base}_af3_cluster_{date_str}_coarse_radial_tree.svg"
                    logger.info(f"Generating coarse clustering radial tree...")
                    analyzer.plot_cluster_radial_tree(save_path=radial_svg, show_plot=OUTPUT_CONFIG['show_plots'], use_coarse_labels=True)
                    logger.info(f"Coarse clustering radial tree saved to {radial_svg}")
                except Exception as e:
                    logger.error(f"Failed to generate coarse clustering radial tree: {e}")
                    logger.exception("Radial tree error details:")
        else:
            # 完整模式：保存粗聚类和精细聚类结果
            analyzer.save_results(base_results_pkl, save_type="both")
            
            # 综合可视化（包含粗聚类和精细聚类）
            if OUTPUT_CONFIG['save_individual_plots']:
                try:
                    combined_viz_path = output_dir / f"{input_base}_af3_cluster_{date_str}_combined_clustering.png"
                    logger.info(f"Generating combined clustering visualization...")
                    analyzer.visualize_results(save_path=combined_viz_path, show_plot=OUTPUT_CONFIG['show_plots'], clustering_type="both")
                    logger.info(f"Combined clustering visualization saved to {combined_viz_path}")
                    
                    # 分别保存粗聚类和精细聚类的可视化
                    coarse_viz_path = output_dir / f"{input_base}_af3_cluster_{date_str}_coarse_clustering.png"
                    analyzer.visualize_results(save_path=coarse_viz_path, show_plot=OUTPUT_CONFIG['show_plots'], clustering_type="coarse")
                    logger.info(f"Coarse clustering visualization saved to {coarse_viz_path}")
                    
                    fine_viz_path = output_dir / f"{input_base}_af3_cluster_{date_str}_fine_clustering.png"
                    analyzer.visualize_results(save_path=fine_viz_path, show_plot=OUTPUT_CONFIG['show_plots'], clustering_type="fine")
                    logger.info(f"Fine clustering visualization saved to {fine_viz_path}")
                    
                    # 导出单独的图表和数据
                    logger.info(f"Exporting individual plots and data...")
                    analyzer._export_individual_plots(combined_viz_path, clustering_type="both")
                    logger.info(f"Individual plots and data exported successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to generate clustering visualizations: {e}")
                    logger.exception("Visualization error details:")

        # 径向树图（基于精细聚类结果，如果有的话）
        if OUTPUT_CONFIG['save_radial_tree'] and analyzer.fine_labels is not None:
            try:
                radial_svg = output_dir / f"{input_base}_af3_cluster_{date_str}_radial_tree.svg"
                logger.info(f"Generating fine clustering radial tree...")
                analyzer.plot_cluster_radial_tree(save_path=radial_svg, show_plot=OUTPUT_CONFIG['show_plots'])
                logger.info(f"Radial tree saved to {radial_svg}")
            except Exception as e:
                logger.error(f"Failed to generate radial tree: {e}")
                logger.exception("Radial tree error details:")
        
        logger.info("All AF3 clustering tasks finished. Exiting.")
        
    except Exception:
        logger.exception("AF3 clustering failed with an unexpected error")


if __name__ == "__main__":
    main()
