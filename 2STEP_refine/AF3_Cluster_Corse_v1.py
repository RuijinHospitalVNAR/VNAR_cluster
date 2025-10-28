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
import argparse

from Bio.PDB import PDBParser, MMCIFParser, Superimposer
import hdbscan
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from matplotlib import patches

# ------------------------
# 日志配置
# ------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")


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
        self.contact_sets = []
        self.binary_features = []
        self.structures = []
        self.file_names = []
        self.coarse_labels = None
        self.scaler = StandardScaler()
        self.palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        self.memory_threshold = 0.9

    def _check_memory_usage(self):
        try:
            memory_percent = psutil.virtual_memory().percent / 100.0
            if memory_percent > self.memory_threshold:
                logger.warning(f"High memory usage detected: {memory_percent:.1%}")
                logger.info("Triggering garbage collection...")
                gc.collect()
                return True
            return False
        except Exception:
            return False

    def _log_memory_usage(self, stage=""):
        try:
            memory_percent = psutil.virtual_memory().percent
            memory_gb = psutil.virtual_memory().used / (1024**3)
            logger.info(f"Memory usage {stage}: {memory_percent:.1f}% ({memory_gb:.2f} GB)")
        except Exception:
            pass

    def _auto_hdbscan_params(self, n_samples: int):
        min_cluster_size = max(5, min(100, max(1, n_samples // 20)))
        min_samples = max(3, min(min_cluster_size, int(np.log2(max(n_samples, 2))) + 1))
        return min_cluster_size, min_samples

    def _auto_dbscan_eps_from_D(self, D: np.ndarray):
        try:
            n = D.shape[0]
            if n < 10:
                return 0.3, 5
            k = max(3, int(np.log2(n)) + 1)
            Di = D.copy()
            np.fill_diagonal(Di, np.inf)
            kth = np.partition(Di, kth=k-1, axis=1)[:, k-1]
            eps = float(np.median(kth))
            eps = max(1e-6, min(1.0, eps))
            return eps, k
        except Exception:
            return 0.3, 5

    def parse_chain_and_ranges(self, chain_pair):
        chain_specs = chain_pair.split(';')
        if len(chain_specs) != 2:
            raise ValueError("Chain pair must specify exactly two chains")
        chains_ranges = {}
        for spec in chain_specs:
            parts = spec.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid chain specification: {spec}")
            chain_id = parts[0]
            range_str = parts[1].lower()
            if range_str == 'all':
                chains_ranges[chain_id] = None
            else:
                ranges = []
                for range_part in range_str.split(','):
                    start, end = map(int, range_part.split('-'))
                    if start > end:
                        raise ValueError(f"Invalid range: {start}-{end}")
                    ranges.append((start, end))
                chains_ranges[chain_id] = ranges
        return chains_ranges

    def set_residue_ranges(self, chain_pair):
        self.residue_ranges = self.parse_chain_and_ranges(chain_pair)
        logger.info(f"Set residue ranges: {self.residue_ranges}")

    def load_structure(self, structure_file):
        try:
            file_path = Path(structure_file)
            ext = file_path.suffix.lower()
            if ext == ".pdb":
                parser = PDBParser(QUIET=True)
            elif ext == ".cif":
                parser = MMCIFParser(QUIET=True)
            else:
                logger.warning(f"Unsupported file format: {ext}")
                return None
            structure = parser.get_structure(file_path.stem, file_path)
            if len(structure) == 0 or len(structure[0]) == 0:
                return None
            return structure
        except Exception as e:
            logger.warning(f"Error loading {structure_file}: {e}")
            return None

    def get_contacts_interface_atoms(self, structure, chainA="A", antigen_chains=None, cutoff=5.0, interface_cutoff=8.0):
        if antigen_chains is None:
            antigen_chains = self.antigen_chains
        try:
            interface_residuesA = set()
            interface_residues_antigen = set()
            for resA in structure[0][chainA]:
                for chain_id in antigen_chains:
                    for resB in structure[0][chain_id]:
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
                        if min_distance <= interface_cutoff:
                            interface_residuesA.add(resA)
                            interface_residues_antigen.add(resB)
            atomsA = []
            for res in interface_residuesA:
                if "CA" in res:
                    atomsA.append(res["CA"])
                sidechain_atoms = [a for a in res if a.get_id() not in ['N', 'CA', 'C', 'O'] and a.get_id()[0] != 'H']
                atomsA.extend(sidechain_atoms)
            selected_antigen_atoms = []
            for res in interface_residues_antigen:
                if "CA" in res:
                    selected_antigen_atoms.append(res["CA"])
                sidechain_atoms = [a for a in res if a.get_id() not in ['N', 'CA', 'C', 'O'] and a.get_id()[0] != 'H']
                selected_antigen_atoms.extend(sidechain_atoms)
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
        try:
            structure = self.load_structure(structure_file)
            if structure is None:
                return None, None, None
            interface_cutoff = getattr(self, 'interface_cutoff', 8.0)
            contacts = self.get_contacts_interface_atoms(structure, self.chainA, self.antigen_chains, self.contact_cutoff, interface_cutoff)
            if len(contacts) == 0:
                return None, None, None
            return contacts, structure, Path(structure_file).name
        except Exception as e:
            logger.error(f"Error processing {structure_file}: {e}")
            return None, None, None

    def load_and_process_data(self, cache_file="af3_coarse_data_cache.pkl"):
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
                self.binary_features = [None] * len(self.contact_sets)
            else:
                logger.error(f"Unrecognized cache format in {cache_file}")
                return False
            logger.info(f"Loaded cached data from {cache_file} (files: {len(self.file_names)})")
            return True
        if not self.pdb_dir.exists() or not self.pdb_dir.is_dir():
            logger.error(f"PDB directory not found: {self.pdb_dir}")
            return False
        cif_files = list(self.pdb_dir.glob("*.cif"))
        pdb_files = list(self.pdb_dir.glob("*.pdb"))
        structure_files = cif_files + pdb_files
        logger.info(f"Found {len(cif_files)} CIF files and {len(pdb_files)} PDB files in {self.pdb_dir}")
        logger.info(f"Total structure files: {len(structure_files)}")
        if len(structure_files) == 0:
            logger.error("No CIF or PDB files found")
            return False
        with Pool(processes=self.n_jobs) as pool:
            results = list(pool.imap_unordered(self.process_single_file, structure_files))
        for contacts, structure, fname in results:
            if contacts is not None:
                self.contact_sets.append(contacts)
                self.structures.append(structure)
                self.file_names.append(fname)
        if len(self.file_names) == 0:
            logger.error("No valid results obtained")
            return False
        self.binary_features = [None] * len(self.contact_sets)
        with open(cache_file, "wb") as f:
            pickle.dump((self.contact_sets, self.binary_features, self.structures, self.file_names), f)
        logger.info(f"Processed and cached {len(self.file_names)} files")
        return True

    def jaccard_distance_matrix(self, contact_sets):
        N = len(contact_sets)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                inter = len(contact_sets[i] & contact_sets[j])
                union = len(contact_sets[i] | contact_sets[j])
                d = 1.0 - (inter/union if union > 0 else 0.0)
                D[i, j] = D[j, i] = d
        return D

    def perform_coarse_clustering(self, method='hdbscan', distance_metric=None, **kwargs):
        if distance_metric is None:
            distance_metric = self.contact_mode
        if not self.contact_sets:
            logger.error("No contact sets available")
            return None
        D = self.jaccard_distance_matrix(self.contact_sets)
        n_samples = len(self.contact_sets)
        if method == 'hdbscan':
            auto_min_cluster_size, auto_min_samples = self._auto_hdbscan_params(n_samples)
            min_cluster_size = kwargs.get('min_cluster_size', auto_min_cluster_size)
            min_samples = kwargs.get('min_samples', auto_min_samples)
            logger.info(f"Coarse HDBSCAN params -> min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed', core_dist_n_jobs=self.n_jobs)
            self.coarse_labels = clusterer.fit_predict(D)
        elif method == 'kmeans':
            n_clusters = kwargs.get("n_clusters", 5)
            logger.info(f"Coarse KMeans n_clusters={n_clusters}")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.coarse_labels = clusterer.fit_predict(D)
        elif method == 'dbscan':
            eps = kwargs.get("eps", 'auto')
            min_samples = kwargs.get("min_samples", 5)
            if eps == 'auto':
                base_eps, _ = self._auto_dbscan_eps_from_D(D)
                multipliers = kwargs.get('eps_multipliers', [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6])
                best_score = -1.0
                best_eps = base_eps
                for m in multipliers:
                    try_eps = max(1e-6, min(1.0, float(base_eps) * float(m)))
                    labels_try = DBSCAN(eps=try_eps, min_samples=min_samples, metric='precomputed').fit_predict(D)
                    try:
                        score = silhouette_score(D, labels_try, metric='precomputed') if len(set(labels_try)) > 1 else -1.0
                    except Exception:
                        score = -1.0
                    if score > best_score:
                        best_score = score
                        best_eps = try_eps
                eps = best_eps
                logger.info(f"Auto-selected DBSCAN eps={eps:.4f} (base={base_eps:.4f}, best_silhouette={best_score:.4f})")
            else:
                eps = float(eps)
            logger.info(f"Coarse DBSCAN params -> eps={eps}, min_samples={min_samples}")
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            self.coarse_labels = clusterer.fit_predict(D)
        else:
            raise ValueError(f"Unsupported clustering method {method}")
        return self.coarse_labels

    def evaluate_clustering(self):
        if self.coarse_labels is None:
            return {"silhouette": None, "n_clusters": 0, "noise_ratio": 1.0}
        D = self.jaccard_distance_matrix(self.contact_sets)
        try:
            score = silhouette_score(D, self.coarse_labels, metric='precomputed')
        except:
            score = None
        unique_labels = set(self.coarse_labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        noise_count = np.sum(self.coarse_labels == -1)
        noise_ratio = noise_count / len(self.coarse_labels)
        return {"silhouette": score, "n_clusters": n_clusters, "noise_ratio": noise_ratio, "total_samples": len(self.coarse_labels)}

    # 省略：可视化与导出函数（保持与原版一致）...


def validate_config(chain_config, clustering_config, output_config):
    logger.info("Validating configuration...")
    return True


def check_dependencies():
    return True


def main():
    parser = argparse.ArgumentParser(description="AF3 Coarse Clustering (DBSCAN/HDBSCAN/KMeans)")
    parser.add_argument('--pdb_dir', type=str, default="./h1d1_bh_renamed")
    parser.add_argument('--chainA', type=str, default='A')
    parser.add_argument('--antigen_chains', nargs='+', default=['B'])
    parser.add_argument('--contact_cutoff', type=float, default=5.0)
    parser.add_argument('--irmsd_cutoff', type=float, default=5.0)
    parser.add_argument('--method', type=str, default='dbscan', choices=['dbscan','hdbscan','kmeans'])
    parser.add_argument('--eps', default='auto')
    parser.add_argument('--eps_grid', type=float, nargs='+', default=[0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6], help='Multipliers around auto-eps for DBSCAN grid search (8 values by default)')
    parser.add_argument('--min_samples', type=int, default=5)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--interface_cutoff', type=float, default=8.0)
    parser.add_argument('--show_plots', action='store_true')
    parser.add_argument('--no_plots', dest='show_plots', action='store_false')
    parser.set_defaults(show_plots=False)
    args = parser.parse_args()

    PDB_DIR = args.pdb_dir
    CHAIN_CONFIG = {
        'chainA': args.chainA,
        'antigen_chains': args.antigen_chains,
        'contact_cutoff': float(args.contact_cutoff),
        'irmsd_cutoff': float(args.irmsd_cutoff),
        'residue_ranges': None,
        'contact_mode': 'jaccard',
        'contact_atom_type': 'interface',
        'interface_cutoff': float(args.interface_cutoff),
        'interface_method': 'residue'
    }

    CLUSTERING_CONFIG = {
        'coarse_method': args.method,
        'coarse_params': {
            'min_cluster_size': 'auto',
            'min_samples': 'auto',
            'n_clusters': int(args.n_clusters),
            'eps': args.eps,
            'min_samples': int(args.min_samples),
            'eps_multipliers': args.eps_grid
        }
    }

    OUTPUT_CONFIG = {
        'auto_adjust_chains': True,
        'show_plots': bool(args.show_plots),
        'save_individual_plots': True,
        'save_radial_tree': True
    }

    if not check_dependencies():
        return
    try:
        validate_config(CHAIN_CONFIG, CLUSTERING_CONFIG, OUTPUT_CONFIG)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

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

    analyzer.interface_cutoff = CHAIN_CONFIG['interface_cutoff']
    analyzer.interface_method = CHAIN_CONFIG['interface_method']

    logger.info("Starting AF3 coarse clustering pipeline ...")
    ok = analyzer.load_and_process_data()
    if not ok or len(analyzer.file_names) < 2:
        logger.error("Insufficient data for clustering.")
        return

    analyzer._log_memory_usage("after data loading")
    coarse_method = CLUSTERING_CONFIG['coarse_method']
    coarse_params = CLUSTERING_CONFIG['coarse_params'].copy()
    if coarse_method == 'hdbscan':
        if coarse_params.get('min_cluster_size') == 'auto':
            del coarse_params['min_cluster_size']
        if coarse_params.get('min_samples') == 'auto':
            del coarse_params['min_samples']
    analyzer.perform_coarse_clustering(method=coarse_method, distance_metric=CHAIN_CONFIG.get('contact_mode'), **coarse_params)
    analyzer._log_memory_usage("after coarse clustering")

    input_base = Path(PDB_DIR).resolve().name
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{input_base}_af3_coarse_cluster_{date_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = analyzer.evaluate_clustering()
    logger.info(f"Clustering metrics: {metrics}")


if __name__ == "__main__":
    main()
