import os
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from tqdm import tqdm
import hdbscan
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import pickle
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
from pathlib import Path
import gc

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 优化警告处理
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=ConvergenceWarning)


class ProteinClusterAnalyzer:
    def __init__(self, cif_dir, antibody_chain='A', antigen_chains=['B', 'C'], 
                 dist_cutoff=5.0, n_jobs=-1):
        """
        初始化蛋白质聚类分析器
        
        Parameters:
        -----------
        cif_dir : str
            CIF/PDB文件目录路径
        antibody_chain : str
            抗体链ID
        antigen_chains : list
            抗原链ID列表
        dist_cutoff : float
            距离截断值
        n_jobs : int
            并行处理数量，-1表示使用所有CPU
        """
        self.cif_dir = Path(cif_dir)
        self.antibody_chain = antibody_chain
        self.antigen_chains = antigen_chains
        self.dist_cutoff = dist_cutoff
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        
        # HDBSCAN参数
        self.min_cluster_size = 10
        self.min_samples = 5
        self.cluster_selection_epsilon = 0.1
        
        # 数据存储
        self.contact_maps = []
        self.feature_vectors = []
        self.file_names = []
        self.cluster_labels = None
        self.scaler = StandardScaler()
        
        # 氨基酸性质字典
        self._setup_amino_acid_properties()
        
    def _setup_amino_acid_properties(self):
        """设置氨基酸性质字典"""
        # 20种标准氨基酸的性质
        self.aa_properties = {
            # 疏水性氨基酸
            'ALA': {'hydrophobic': 1, 'hydrophilic': 0, 'positive': 0, 'negative': 0, 'polar': 0, 'aromatic': 0, 'sulfur': 0, 'size': 1},
            'VAL': {'hydrophobic': 1, 'hydrophilic': 0, 'positive': 0, 'negative': 0, 'polar': 0, 'aromatic': 0, 'sulfur': 0, 'size': 2},
            'LEU': {'hydrophobic': 1, 'hydrophilic': 0, 'positive': 0, 'negative': 0, 'polar': 0, 'aromatic': 0, 'sulfur': 0, 'size': 3},
            'ILE': {'hydrophobic': 1, 'hydrophilic': 0, 'positive': 0, 'negative': 0, 'polar': 0, 'aromatic': 0, 'sulfur': 0, 'size': 3},
            'PRO': {'hydrophobic': 1, 'hydrophilic': 0, 'positive': 0, 'negative': 0, 'polar': 0, 'aromatic': 0, 'sulfur': 0, 'size': 2},
            'MET': {'hydrophobic': 1, 'hydrophilic': 0, 'positive': 0, 'negative': 0, 'polar': 0, 'aromatic': 0, 'sulfur': 1, 'size': 3},
            'PHE': {'hydrophobic': 1, 'hydrophilic': 0, 'positive': 0, 'negative': 0, 'polar': 0, 'aromatic': 1, 'sulfur': 0, 'size': 4},
            'TRP': {'hydrophobic': 1, 'hydrophilic': 0, 'positive': 0, 'negative': 0, 'polar': 0, 'aromatic': 1, 'sulfur': 0, 'size': 5},
            
            # 极性不带电氨基酸
            'SER': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 0, 'negative': 0, 'polar': 1, 'aromatic': 0, 'sulfur': 0, 'size': 1},
            'THR': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 0, 'negative': 0, 'polar': 1, 'aromatic': 0, 'sulfur': 0, 'size': 2},
            'CYS': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 0, 'negative': 0, 'polar': 1, 'aromatic': 0, 'sulfur': 1, 'size': 1},
            'TYR': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 0, 'negative': 0, 'polar': 1, 'aromatic': 1, 'sulfur': 0, 'size': 4},
            'ASN': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 0, 'negative': 0, 'polar': 1, 'aromatic': 0, 'sulfur': 0, 'size': 2},
            'GLN': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 0, 'negative': 0, 'polar': 1, 'aromatic': 0, 'sulfur': 0, 'size': 3},
            
            # 带正电氨基酸
            'LYS': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 1, 'negative': 0, 'polar': 1, 'aromatic': 0, 'sulfur': 0, 'size': 4},
            'ARG': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 1, 'negative': 0, 'polar': 1, 'aromatic': 0, 'sulfur': 0, 'size': 4},
            'HIS': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 1, 'negative': 0, 'polar': 1, 'aromatic': 1, 'sulfur': 0, 'size': 3},
            
            # 带负电氨基酸
            'ASP': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 0, 'negative': 1, 'polar': 1, 'aromatic': 0, 'sulfur': 0, 'size': 2},
            'GLU': {'hydrophobic': 0, 'hydrophilic': 1, 'positive': 0, 'negative': 1, 'polar': 1, 'aromatic': 0, 'sulfur': 0, 'size': 3},
            
            # 特殊氨基酸
            'GLY': {'hydrophobic': 0, 'hydrophilic': 0, 'positive': 0, 'negative': 0, 'polar': 0, 'aromatic': 0, 'sulfur': 0, 'size': 0},
        }
        
        # 氨基酸疏水性指数 (Kyte-Doolittle scale)
        self.hydrophobicity_scale = {
            'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
            'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
            'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
            'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
        }
        
    def extract_ca_coords_with_residues(self, structure_file, chain_ids):
        """提取指定链的Cα原子坐标和残基信息，返回(resname, resid, chain)"""
        try:
            ext = str(structure_file).split('.')[-1].lower()
            if ext == "pdb":
                parser = PDBParser(QUIET=True)
            else:
                parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('model', structure_file)

            ca_coords = []
            residue_infos = []
            chain_info = {}
            found_chains = set()

            for model in structure:
                for chain in model:
                    if chain.id in chain_ids:
                        found_chains.add(chain.id)
                        chain_coords = []
                        chain_residues = []
                        for res in chain:
                            if 'CA' in res:
                                coord = res['CA'].get_coord()
                                resname = res.get_resname().strip()  # 去除空格
                                resid = res.id[1]
                                chain_id = chain.id
                                chain_coords.append(coord)
                                chain_residues.append((resname, resid, chain_id))
                        if chain_coords:
                            ca_coords.extend(chain_coords)
                            residue_infos.extend(chain_residues)
                            chain_info[chain.id] = len(chain_coords)

            missing = set(chain_ids) - found_chains
            if missing:
                logger.warning(f"Chains {missing} not found in {structure_file}")

            return np.array(ca_coords), residue_infos, chain_info
        except Exception as e:
            logger.warning(f"Error processing {structure_file}: {e}")
            return np.array([]), [], {}

    def compute_contact_map_with_residues(self, coords_ab, coords_ag, residues_ab, residues_ag):
        """计算contact map并返回接触残基信息"""
        if len(coords_ab) == 0 or len(coords_ag) == 0:
            return np.array([]), [], []
        
        # 计算距离矩阵
        dists = np.linalg.norm(coords_ab[:, None, :] - coords_ag[None, :, :], axis=-1)
        contact_map = (dists < self.dist_cutoff).astype(np.float32)
        
        # 获取接触残基
        contact_rows, contact_cols = np.where(contact_map > 0)
        contact_ab_residues = [residues_ab[i][0] for i in contact_rows]  # 只取残基名称
        contact_ag_residues = [residues_ag[i][0] for i in contact_cols]
        
        return contact_map, contact_ab_residues, contact_ag_residues

    def extract_interaction_features(self, contact_map, contact_ab_residues, contact_ag_residues):
        """从contact map和氨基酸信息提取综合特征"""
        if contact_map.size == 0:
            return np.zeros(39)  # 返回固定长度的零向量
        
        features = []
        
        # 1. 基本几何特征
        features.extend([
            np.sum(contact_map),  # 总接触数
            np.mean(contact_map),  # 接触密度
            np.std(contact_map),   # 接触变异性
        ])
        
        # 2. 行列统计（抗体和抗原残基的接触模式）
        row_sums = np.sum(contact_map, axis=1)
        col_sums = np.sum(contact_map, axis=0)
        
        features.extend([
            np.mean(row_sums) if len(row_sums) > 0 else 0, 
            np.std(row_sums) if len(row_sums) > 0 else 0, 
            np.max(row_sums) if len(row_sums) > 0 else 0,
            np.mean(col_sums) if len(col_sums) > 0 else 0, 
            np.std(col_sums) if len(col_sums) > 0 else 0, 
            np.max(col_sums) if len(col_sums) > 0 else 0,
        ])
        
        # 3. 接触区域的几何特征
        if np.sum(contact_map) > 0:
            contact_rows, contact_cols = np.where(contact_map > 0)
            features.extend([
                np.std(contact_rows) if len(contact_rows) > 0 else 0,  # 抗体接触区域分散度
                np.std(contact_cols) if len(contact_cols) > 0 else 0,  # 抗原接触区域分散度
                len(np.unique(contact_rows)),  # 参与接触的抗体残基数
                len(np.unique(contact_cols)),  # 参与接触的抗原残基数
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 4. 接触模式特征（局部连续性）
        if contact_map.shape[0] > 1 and contact_map.shape[1] > 1:
            row_continuity = np.sum(contact_map[:-1, :] * contact_map[1:, :])
            col_continuity = np.sum(contact_map[:, :-1] * contact_map[:, 1:])
            features.extend([row_continuity, col_continuity])
        else:
            features.extend([0, 0])
        
        # 5. 氨基酸性质特征
        aa_features = self.extract_amino_acid_features(contact_ab_residues, contact_ag_residues)
        features.extend(aa_features)
        
        return np.array(features, dtype=np.float32)
    
    def extract_amino_acid_features(self, contact_ab_residues, contact_ag_residues):
        """提取接触位点的氨基酸性质特征"""
        if not contact_ab_residues or not contact_ag_residues:
            return np.zeros(24)  # 返回零向量
        
        features = []
        
        # 1. 抗体接触残基的性质统计
        ab_props = self._get_residue_properties(contact_ab_residues)
        ag_props = self._get_residue_properties(contact_ag_residues)
        
        # 2. 基本性质比例（抗体侧）
        features.extend([
            ab_props['hydrophobic_ratio'],
            ab_props['hydrophilic_ratio'], 
            ab_props['positive_ratio'],
            ab_props['negative_ratio'],
            ab_props['polar_ratio'],
            ab_props['aromatic_ratio'],
        ])
        
        # 3. 基本性质比例（抗原侧）
        features.extend([
            ag_props['hydrophobic_ratio'],
            ag_props['hydrophilic_ratio'],
            ag_props['positive_ratio'], 
            ag_props['negative_ratio'],
            ag_props['polar_ratio'],
            ag_props['aromatic_ratio'],
        ])
        
        # 4. 相互作用特征
        features.extend([
            self._calculate_charge_complementarity(contact_ab_residues, contact_ag_residues),
            self._calculate_hydrophobic_interaction(contact_ab_residues, contact_ag_residues),
            self._calculate_polar_interaction(contact_ab_residues, contact_ag_residues),
            ab_props['avg_hydrophobicity'],
            ag_props['avg_hydrophobicity'],
            ab_props['avg_size'],
            ag_props['avg_size'],
        ])
        
        # 5. 多样性特征
        features.extend([
            ab_props['diversity'],
            ag_props['diversity'],
            self._calculate_interface_complementarity(ab_props, ag_props),
            len(set(contact_ab_residues)),  # 抗体接触残基种类数
            len(set(contact_ag_residues)),  # 抗原接触残基种类数
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _get_residue_properties(self, residues):
        """计算残基列表的性质统计"""
        if not residues:
            return {k: 0.0 for k in ['hydrophobic_ratio', 'hydrophilic_ratio', 'positive_ratio', 
                                   'negative_ratio', 'polar_ratio', 'aromatic_ratio', 
                                   'avg_hydrophobicity', 'avg_size', 'diversity']}
        
        total = len(residues)
        props = {
            'hydrophobic': 0, 'hydrophilic': 0, 'positive': 0, 
            'negative': 0, 'polar': 0, 'aromatic': 0
        }
        hydrophobicity_sum = 0
        size_sum = 0
        
        for res in residues:
            if res in self.aa_properties:
                aa_prop = self.aa_properties[res]
                for key in props:
                    props[key] += aa_prop[key]
                hydrophobicity_sum += self.hydrophobicity_scale.get(res, 0)
                size_sum += aa_prop['size']
        
        # 计算比例
        result = {}
        for key in props:
            result[f'{key}_ratio'] = props[key] / total if total > 0 else 0
        
        result['avg_hydrophobicity'] = hydrophobicity_sum / total if total > 0 else 0
        result['avg_size'] = size_sum / total if total > 0 else 0
        
        # 计算氨基酸多样性（Shannon熵）
        unique_residues = list(set(residues))
        if len(unique_residues) > 1:
            diversity = 0
            for res in unique_residues:
                p = residues.count(res) / total
                if p > 0:
                    diversity -= p * np.log2(p)
            result['diversity'] = diversity
        else:
            result['diversity'] = 0
        
        return result
    
    def _calculate_charge_complementarity(self, ab_residues, ag_residues):
        """计算电荷互补性"""
        ab_positive = sum(1 for res in ab_residues if res in self.aa_properties and self.aa_properties[res]['positive'])
        ab_negative = sum(1 for res in ab_residues if res in self.aa_properties and self.aa_properties[res]['negative'])
        ag_positive = sum(1 for res in ag_residues if res in self.aa_properties and self.aa_properties[res]['positive'])
        ag_negative = sum(1 for res in ag_residues if res in self.aa_properties and self.aa_properties[res]['negative'])
        
        # 电荷互补性：正电荷与负电荷的匹配程度
        total_residues = len(ab_residues) + len(ag_residues)
        complementarity = (ab_positive * ag_negative + ab_negative * ag_positive) / max(total_residues, 1)
        return complementarity
    
    def _calculate_hydrophobic_interaction(self, ab_residues, ag_residues):
        """计算疏水相互作用强度"""
        ab_hydrophobic = sum(1 for res in ab_residues if res in self.aa_properties and self.aa_properties[res]['hydrophobic'])
        ag_hydrophobic = sum(1 for res in ag_residues if res in self.aa_properties and self.aa_properties[res]['hydrophobic'])
        
        # 疏水相互作用：双方疏水残基的乘积
        total_interactions = len(ab_residues) * len(ag_residues)
        interaction = (ab_hydrophobic * ag_hydrophobic) / max(total_interactions, 1)
        return interaction
    
    def _calculate_polar_interaction(self, ab_residues, ag_residues):
        """计算极性相互作用强度"""
        ab_polar = sum(1 for res in ab_residues if res in self.aa_properties and self.aa_properties[res]['polar'])
        ag_polar = sum(1 for res in ag_residues if res in self.aa_properties and self.aa_properties[res]['polar'])
        
        # 极性相互作用
        total_interactions = len(ab_residues) * len(ag_residues)
        interaction = (ab_polar * ag_polar) / max(total_interactions, 1)
        return interaction
    
    def _calculate_interface_complementarity(self, ab_props, ag_props):
        """计算界面互补性"""
        # 计算性质的互补程度（负相关表示互补）
        complementarity = 0
        prop_pairs = [
            ('hydrophobic_ratio', 'hydrophilic_ratio'),
            ('positive_ratio', 'negative_ratio'),
        ]
        
        for ab_prop, ag_prop in prop_pairs:
            complementarity += ab_props[ab_prop] * ag_props[ag_prop]
        
        return complementarity / len(prop_pairs)

    def process_single_file(self, structure_file):
        """处理单个结构文件"""
        try:
            # 提取坐标和残基信息
            ab_coords, ab_residues, ab_info = self.extract_ca_coords_with_residues(structure_file, [self.antibody_chain])
            ag_coords, ag_residues, ag_info = self.extract_ca_coords_with_residues(structure_file, self.antigen_chains)
            
            if len(ab_coords) == 0 or len(ag_coords) == 0:
                logger.warning(f"No valid coordinates found in {structure_file}")
                return None, None, None
            
            # 计算contact map和接触残基
            contact_map, contact_ab_residues, contact_ag_residues = self.compute_contact_map_with_residues(
                ab_coords, ag_coords, ab_residues, ag_residues)
            
            if contact_map.size == 0:
                return None, None, None
            
            # 提取综合特征（包括氨基酸性质）
            features = self.extract_interaction_features(contact_map, contact_ab_residues, contact_ag_residues)
            
            return contact_map.flatten(), features, structure_file.name
            
        except Exception as e:
            logger.error(f"Error processing {structure_file}: {e}")
            return None, None, None

    def load_and_process_data(self, cache_file=None):
        """加载和处理所有结构文件"""
        # 检查缓存
        if cache_file and Path(cache_file).exists():
            logger.info(f"Loading cached data from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.contact_maps = cached_data['contact_maps']
                    self.feature_vectors = cached_data['feature_vectors']
                    self.file_names = cached_data['file_names']
                    return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Processing from scratch.")
        
        # 查找CIF和PDB文件
        structure_files = list(self.cif_dir.glob('*.cif')) + list(self.cif_dir.glob('*.pdb'))
        logger.info(f"Found {len(structure_files)} structure files")
        
        if len(structure_files) == 0:
            raise ValueError("No CIF or PDB files found in the specified directory")
        
        # 并行处理文件
        logger.info("Processing structure files in parallel...")
        with Pool(processes=self.n_jobs) as pool:
            results = list(tqdm(
                pool.map(self.process_single_file, structure_files),
                total=len(structure_files),
                desc="Processing structure files"
            ))
        
        # 过滤有效结果
        valid_results = [(cm, fv, fn) for cm, fv, fn in results if cm is not None]
        
        if len(valid_results) == 0:
            raise ValueError("No valid results obtained from structure files")
        
        logger.info(f"Successfully processed {len(valid_results)}/{len(structure_files)} files")
        
        self.contact_maps, self.feature_vectors, self.file_names = zip(*valid_results)
        self.contact_maps = list(self.contact_maps)
        self.feature_vectors = list(self.feature_vectors)
        self.file_names = list(self.file_names)
        
        # 缓存结果
        if cache_file:
            try:
                logger.info(f"Caching results to {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'contact_maps': self.contact_maps,
                        'feature_vectors': self.feature_vectors,
                        'file_names': self.file_names
                    }, f)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def prepare_features(self, feature_type='combined', use_pca=True, n_components=0.95):
        """准备聚类特征"""
        logger.info(f"Preparing features with type: {feature_type}")
        
        if feature_type == 'contact_map':
            # 使用原始contact map
            X = np.array(self.contact_maps)
        elif feature_type == 'engineered':
            # 使用工程化特征
            X = np.array(self.feature_vectors)
        elif feature_type == 'combined':
            # 结合两种特征
            contact_features = np.array(self.contact_maps)
            engineered_features = np.array(self.feature_vectors)
            
            # 标准化工程化特征
            if engineered_features.shape[1] > 0:
                engineered_features = self.scaler.fit_transform(engineered_features)
            
            # 降维contact map特征（如果维度太高）
            if contact_features.shape[1] > 1000:
                pca_contact = PCA(n_components=min(500, contact_features.shape[0] - 1))
                contact_features = pca_contact.fit_transform(contact_features)
                logger.info(f"Contact map features reduced to {contact_features.shape[1]} dimensions")
            
            X = np.hstack([contact_features, engineered_features])
        else:
            raise ValueError("feature_type must be 'contact_map', 'engineered', or 'combined'")
        
        # 处理NaN和inf值
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 标准化
        if feature_type != 'engineered':  # engineered已经标准化过了
            X = self.scaler.fit_transform(X)
        
        # PCA降维
        if use_pca and X.shape[1] > 50:
            pca = PCA(n_components=n_components)
            X = pca.fit_transform(X)
            logger.info(f"PCA reduced features to {X.shape[1]} dimensions "
                       f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")
        
        return X

    def perform_clustering(self, X, method='hdbscan'):
        """执行聚类分析"""
        logger.info(f"Performing {method} clustering...")
        
        if method == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(5, min(self.min_cluster_size, len(X) // 20)),
                min_samples=max(3, min(self.min_samples, len(X) // 50)),
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric='euclidean',
                n_jobs=self.n_jobs
            )
            self.cluster_labels = clusterer.fit_predict(X)
            
        elif method == 'kmeans':
            # 使用肘部法则确定最佳k值
            max_k = min(20, len(X) // 5)
            if max_k < 2:
                max_k = 2
            
            inertias = []
            silhouette_scores = []
            
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X, labels))
            
            # 选择最佳k值（基于silhouette score）
            best_k = np.argmax(silhouette_scores) + 2
            logger.info(f"Selected k={best_k} based on silhouette score")
            
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(X)
        
        else:
            raise ValueError("method must be 'hdbscan' or 'kmeans'")
        
        return self.cluster_labels

    def evaluate_clustering(self, X):
        """评估聚类质量"""
        if self.cluster_labels is None:
            raise ValueError("No clustering results found. Run perform_clustering first.")
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        logger.info(f"Clustering results:")
        logger.info(f"  Total samples: {len(X)}")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Noise points: {n_noise}")
        
        # 计算聚类质量指标
        if n_clusters > 1:
            valid_mask = self.cluster_labels != -1
            if np.sum(valid_mask) > 1:
                valid_labels = self.cluster_labels[valid_mask]
                valid_X = X[valid_mask]
                
                try:
                    silhouette_avg = silhouette_score(valid_X, valid_labels)
                    calinski_harabasz = calinski_harabasz_score(valid_X, valid_labels)
                    davies_bouldin = davies_bouldin_score(valid_X, valid_labels)
                    
                    logger.info(f"  Silhouette Score: {silhouette_avg:.3f}")
                    logger.info(f"  Calinski-Harabasz Score: {calinski_harabasz:.3f}")
                    logger.info(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
                    
                    return {
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette_score': silhouette_avg,
                        'calinski_harabasz_score': calinski_harabasz,
                        'davies_bouldin_score': davies_bouldin
                    }
                except Exception as e:
                    logger.warning(f"Error calculating clustering metrics: {e}")
        
        return {'n_clusters': n_clusters, 'n_noise': n_noise}

    def visualize_results(self, X, save_path=None):
        """可视化聚类结果"""
        if self.cluster_labels is None:
            raise ValueError("No clustering results found.")
        
        # 使用更安全的matplotlib样式
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Protein Complex Clustering Results', fontsize=16)
        
        unique_labels = np.unique(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        # 1. 聚类标签分布
        ax1 = axes[0, 0]
        for label, color in zip(unique_labels, colors):
            mask = self.cluster_labels == label
            indices = np.where(mask)[0]
            ax1.scatter(indices, [label] * len(indices), 
                       c=[color], alpha=0.7, s=30,
                       label=f'Cluster {label}' if label != -1 else 'Noise')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Cluster Label')
        ax1.set_title('Cluster Assignment')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. 聚类大小分布
        ax2 = axes[0, 1]
        cluster_sizes = []
        cluster_ids = []
        for label in unique_labels:
            if label != -1:
                size = np.sum(self.cluster_labels == label)
                cluster_sizes.append(size)
                cluster_ids.append(label)
        
        if cluster_sizes:
            bars = ax2.bar(range(len(cluster_sizes)), cluster_sizes, 
                          color=colors[unique_labels != -1])
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Cluster Size')
            ax2.set_title('Cluster Size Distribution')
            ax2.set_xticks(range(len(cluster_ids)))
            ax2.set_xticklabels(cluster_ids)
            
            # 添加数值标签
            for bar, size in zip(bars, cluster_sizes):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(size), ha='center', va='bottom')
        
        # 3. t-SNE可视化
        ax3 = axes[1, 0]
        if len(X) > 1:
            try:
                perplexity = min(30, max(5, len(X) - 1))
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                X_tsne = tsne.fit_transform(X)
                
                for label, color in zip(unique_labels, colors):
                    mask = self.cluster_labels == label
                    ax3.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                               c=[color], alpha=0.7, s=30,
                               label=f'Cluster {label}' if label != -1 else 'Noise')
                ax3.set_xlabel('t-SNE 1')
                ax3.set_ylabel('t-SNE 2')
                ax3.set_title('t-SNE Visualization')
                ax3.legend()
            except Exception as e:
                ax3.text(0.5, 0.5, f't-SNE failed: {str(e)}', 
                        ha='center', va='center', transform=ax3.transAxes)
        
        # 4. 特征重要性（包括氨基酸性质特征）
        ax4 = axes[1, 1]
        if hasattr(self, 'feature_vectors') and len(self.feature_vectors) > 0:
            feature_matrix = np.array(self.feature_vectors)
            
            # 定义特征名称（包括新的氨基酸性质特征）
            geometric_features = [
                'Total_Contacts', 'Contact_Density', 'Contact_Variability',
                'Ab_Mean_Contacts', 'Ab_Std_Contacts', 'Ab_Max_Contacts', 
                'Ag_Mean_Contacts', 'Ag_Std_Contacts', 'Ag_Max_Contacts',
                'Ab_Region_Dispersion', 'Ag_Region_Dispersion', 
                'Ab_Residue_Count', 'Ag_Residue_Count',
                'Ab_Continuity', 'Ag_Continuity'
            ]
            
            aa_features = [
                'Ab_Hydrophobic_Ratio', 'Ab_Hydrophilic_Ratio', 'Ab_Positive_Ratio',
                'Ab_Negative_Ratio', 'Ab_Polar_Ratio', 'Ab_Aromatic_Ratio',
                'Ag_Hydrophobic_Ratio', 'Ag_Hydrophilic_Ratio', 'Ag_Positive_Ratio', 
                'Ag_Negative_Ratio', 'Ag_Polar_Ratio', 'Ag_Aromatic_Ratio',
                'Charge_Complementarity', 'Hydrophobic_Interaction', 'Polar_Interaction',
                'Ab_Avg_Hydrophobicity', 'Ag_Avg_Hydrophobicity', 'Ab_Avg_Size', 'Ag_Avg_Size',
                'Ab_Diversity', 'Ag_Diversity', 'Interface_Complementarity',
                'Ab_Residue_Types', 'Ag_Residue_Types'
            ]
            
            feature_names = geometric_features + aa_features
            
            # 确保特征名称数量与实际特征数量匹配
            if len(feature_names) != feature_matrix.shape[1]:
                feature_names = [f'Feature_{i}' for i in range(feature_matrix.shape[1])]
            
            # 计算每个特征的方差
            feature_variance = np.var(feature_matrix, axis=0)
            sorted_indices = np.argsort(feature_variance)[::-1][:15]  # 前15个最重要的特征
            
            bars = ax4.barh(range(len(sorted_indices)), 
                           feature_variance[sorted_indices])
            ax4.set_yticks(range(len(sorted_indices)))
            ax4.set_yticklabels([feature_names[i] for i in sorted_indices])
            ax4.set_xlabel('Variance')
            ax4.set_title('Top 15 Feature Importance')
            
            # 为氨基酸性质特征添加不同颜色
            for i, bar in enumerate(bars):
                feature_name = feature_names[sorted_indices[i]]
                if any(aa_keyword in feature_name for aa_keyword in 
                      ['Hydrophobic', 'Hydrophilic', 'Positive', 'Negative', 'Polar', 'Aromatic', 
                       'Charge', 'Diversity', 'Complementarity']):
                    bar.set_color('orange')  # 氨基酸性质特征用橙色
                else:
                    bar.set_color('skyblue')  # 几何特征用蓝色
                    
        else:
            ax4.text(0.5, 0.5, 'No engineered features available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()

    def get_cluster_representatives(self, X, n_representatives=3):
        """获取每个聚类的代表性样本"""
        if self.cluster_labels is None:
            raise ValueError("No clustering results found.")
        
        representatives = {}
        unique_labels = np.unique(self.cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue
                
            cluster_mask = self.cluster_labels == label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_X = X[cluster_mask]
            
            if len(cluster_X) == 0:
                continue
            
            # 计算聚类中心
            cluster_center = np.mean(cluster_X, axis=0)
            
            # 找到距离中心最近的样本
            distances = np.linalg.norm(cluster_X - cluster_center, axis=1)
            closest_indices = np.argsort(distances)[:min(n_representatives, len(distances))]
            
            representative_indices = cluster_indices[closest_indices]
            representative_files = [self.file_names[i] for i in representative_indices]
            
            representatives[label] = {
                'indices': representative_indices,
                'files': representative_files,
                'distances': distances[closest_indices],
                'size': len(cluster_indices)
            }
        
        return representatives

    def save_results(self, output_file, X=None):
        """保存聚类结果"""
        results = {
            'file_names': self.file_names,
            'cluster_labels': self.cluster_labels,
            'parameters': {
                'antibody_chain': self.antibody_chain,
                'antigen_chains': self.antigen_chains,
                'dist_cutoff': self.dist_cutoff,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon
            }
        }
        
        if X is not None:
            representatives = self.get_cluster_representatives(X)
            results['representatives'] = representatives
        
        # 保存为pickle格式
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        # 同时保存为CSV格式便于查看
        csv_file = str(output_file).replace('.pkl', '.csv')
        df = pd.DataFrame({
            'file_name': self.file_names,
            'cluster_label': self.cluster_labels
        })
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {output_file} and {csv_file}")
        
    def analyze_amino_acid_patterns(self, save_path=None):
        """分析不同聚类的氨基酸性质模式"""
        if self.cluster_labels is None:
            raise ValueError("No clustering results found.")
        
        if not hasattr(self, 'feature_vectors') or len(self.feature_vectors) == 0:
            logger.warning("No amino acid features available for analysis")
            return
        
        # 提取氨基酸特征（假设是特征向量的后24个特征）
        feature_matrix = np.array(self.feature_vectors)
        if feature_matrix.shape[1] < 24:
            logger.warning("Insufficient features for amino acid analysis")
            return
        
        aa_features = feature_matrix[:, -24:]  # 最后24个特征是氨基酸性质特征
        
        # 氨基酸特征名称
        aa_feature_names = [
            'Ab_Hydrophobic', 'Ab_Hydrophilic', 'Ab_Positive', 'Ab_Negative', 'Ab_Polar', 'Ab_Aromatic',
            'Ag_Hydrophobic', 'Ag_Hydrophilic', 'Ag_Positive', 'Ag_Negative', 'Ag_Polar', 'Ag_Aromatic',
            'Charge_Complementarity', 'Hydrophobic_Interaction', 'Polar_Interaction',
            'Ab_Avg_Hydrophobicity', 'Ag_Avg_Hydrophobicity', 'Ab_Avg_Size', 'Ag_Avg_Size',
            'Ab_Diversity', 'Ag_Diversity', 'Interface_Complementarity', 'Ab_Residue_Types', 'Ag_Residue_Types'
        ]
        
        unique_labels = np.unique(self.cluster_labels)
        valid_labels = [label for label in unique_labels if label != -1]
        
        if len(valid_labels) == 0:
            logger.warning("No valid clusters found for analysis")
            return
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Amino Acid Property Analysis by Cluster', fontsize=16)
        
        # 1. 热图：每个聚类的氨基酸性质平均值
        ax1 = axes[0, 0]
        cluster_means = []
        cluster_labels_clean = []
        
        for label in valid_labels:
            mask = self.cluster_labels == label
            if np.sum(mask) > 0:
                cluster_mean = np.mean(aa_features[mask], axis=0)
                cluster_means.append(cluster_mean)
                cluster_labels_clean.append(f'Cluster {label}')
        
        if cluster_means:
            cluster_matrix = np.array(cluster_means)
            im = ax1.imshow(cluster_matrix, cmap='RdYlBu_r', aspect='auto')
            ax1.set_xticks(range(len(aa_feature_names)))
            ax1.set_xticklabels(aa_feature_names, rotation=45, ha='right')
            ax1.set_yticks(range(len(cluster_labels_clean)))
            ax1.set_yticklabels(cluster_labels_clean)
            ax1.set_title('Cluster Average AA Properties')
            plt.colorbar(im, ax=ax1)
        
        # 2. 关键氨基酸性质的箱线图
        ax2 = axes[0, 1]
        key_features = ['Charge_Complementarity', 'Hydrophobic_Interaction', 'Polar_Interaction', 'Interface_Complementarity']
        key_indices = [i for i, name in enumerate(aa_feature_names) if any(key in name for key in key_features)]
        
        if key_indices and len(valid_labels) > 0:
            try:
                data_for_box = []
                labels_for_box = []
                
                for idx in key_indices[:4]:  # 限制显示数量
                    for label in valid_labels[:3]:  # 限制聚类数量
                        mask = self.cluster_labels == label
                        if np.sum(mask) > 0:
                            data_for_box.append(aa_features[mask, idx])
                            labels_for_box.append(f'{aa_feature_names[idx]}_C{label}')
                
                if data_for_box:
                    ax2.boxplot(data_for_box, labels=labels_for_box)
                    ax2.set_xticklabels(labels_for_box, rotation=45, ha='right')
                    ax2.set_title('Key Interaction Properties')
                    ax2.set_ylabel('Feature Value')
            except Exception as e:
                ax2.text(0.5, 0.5, f'Boxplot failed: {str(e)}', 
                        ha='center', va='center', transform=ax2.transAxes)
        
        # 3. 聚类间的氨基酸性质差异
        ax3 = axes[1, 0]
        if len(valid_labels) > 1:
            try:
                # 计算聚类间的欧几里得距离
                distances = []
                pair_labels = []
                
                for i in range(len(valid_labels)):
                    for j in range(i+1, len(valid_labels)):
                        label1, label2 = valid_labels[i], valid_labels[j]
                        mask1 = self.cluster_labels == label1
                        mask2 = self.cluster_labels == label2
                        
                        if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                            mean1 = np.mean(aa_features[mask1], axis=0)
                            mean2 = np.mean(aa_features[mask2], axis=0)
                            dist = np.linalg.norm(mean1 - mean2)
                            distances.append(dist)
                            pair_labels.append(f'C{label1}-C{label2}')
                
                if distances:
                    bars = ax3.bar(range(len(distances)), distances)
                    ax3.set_xticks(range(len(pair_labels)))
                    ax3.set_xticklabels(pair_labels, rotation=45, ha='right')
                    ax3.set_title('AA Property Distance Between Clusters')
                    ax3.set_ylabel('Euclidean Distance')
                    
                    # 颜色编码：距离越大颜色越深
                    max_dist = max(distances) if distances else 1
                    for bar, dist in zip(bars, distances):
                        bar.set_color(plt.cm.Reds(dist / max_dist))
            except Exception as e:
                ax3.text(0.5, 0.5, f'Distance calculation failed: {str(e)}', 
                        ha='center', va='center', transform=ax3.transAxes)
        
        # 4. 主要氨基酸性质的PCA
        ax4 = axes[1, 1]
        try:
            pca = PCA(n_components=2)
            aa_pca = pca.fit_transform(aa_features)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(valid_labels)))
            for label, color in zip(valid_labels, colors):
                mask = self.cluster_labels == label
                if np.sum(mask) > 0:
                    ax4.scatter(aa_pca[mask, 0], aa_pca[mask, 1], 
                              c=[color], alpha=0.7, label=f'Cluster {label}')
            
            ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax4.set_title('AA Properties PCA')
            ax4.legend()
        except Exception as e:
            ax4.text(0.5, 0.5, f'PCA failed: {str(e)}', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Amino acid analysis saved to {save_path}")
        
        plt.show()
        
        # 输出统计摘要
        print("\n=== 氨基酸性质分析摘要 ===")
        for label in valid_labels:
            mask = self.cluster_labels == label
            cluster_aa_features = aa_features[mask]
            if len(cluster_aa_features) > 0:
                print(f"\nCluster {label} ({np.sum(mask)} samples):")
                print(f"  平均电荷互补性: {np.mean(cluster_aa_features[:, 12]):.3f}")
                print(f"  平均疏水相互作用: {np.mean(cluster_aa_features[:, 13]):.3f}")
                print(f"  平均极性相互作用: {np.mean(cluster_aa_features[:, 14]):.3f}")
                print(f"  界面互补性: {np.mean(cluster_aa_features[:, 21]):.3f}")
                
                # 主要氨基酸类型
                ab_hydrophobic = np.mean(cluster_aa_features[:, 0])
                ab_positive = np.mean(cluster_aa_features[:, 2])
                ab_negative = np.mean(cluster_aa_features[:, 3])
                print(f"  抗体接触区域: 疏水性 {ab_hydrophobic:.3f}, 正电性 {ab_positive:.3f}, 负电性 {ab_negative:.3f}")
        
        return {
            'cluster_aa_means': cluster_means,
            'cluster_labels': cluster_labels_clean,
            'feature_names': aa_feature_names
        }


def main():
    """主函数"""
    # 配置参数
    CIF_DIR = "./your_structure_dir"  # 修改为你的结构文件夹路径
    ANTIBODY_CHAIN = 'A'
    ANTIGEN_CHAINS = ['B', 'C']
    DIST_CUTOFF = 5.0
    
    # 创建分析器
    analyzer = ProteinClusterAnalyzer(
        cif_dir=CIF_DIR,
        antibody_chain=ANTIBODY_CHAIN,
        antigen_chains=ANTIGEN_CHAINS,
        dist_cutoff=DIST_CUTOFF,
        n_jobs=-1
    )
    
    try:
        # 加载和处理数据
        cache_file = "protein_data_cache.pkl"
        analyzer.load_and_process_data(cache_file=cache_file)
        
        # 准备特征（包含氨基酸性质特征）
        X = analyzer.prepare_features(feature_type='combined', use_pca=True)
        
        # 执行聚类
        cluster_labels = analyzer.perform_clustering(X, method='hdbscan')
        
        # 评估聚类质量
        metrics = analyzer.evaluate_clustering(X)
        
        # 可视化结果
        analyzer.visualize_results(X, save_path='clustering_results.png')
        
        # 🆕 氨基酸性质分析
        aa_analysis = analyzer.analyze_amino_acid_patterns(save_path='amino_acid_analysis.png')
        
        # 保存结果
        analyzer.save_results('clustering_results.pkl', X)
        
        # 输出代表性样本
        representatives = analyzer.get_cluster_representatives(X)
        print("\n各聚类代表性样本：")
        for label, info in representatives.items():
            print(f"Cluster {label} ({info['size']} samples):")
            for i, (file, dist) in enumerate(zip(info['files'], info['distances'])):
                print(f"  {i+1}. {file} (distance to center: {dist:.3f})")
            print()
        
        print("\n🧬 聚类分析完成！")
        print("📊 生成的文件:")
        print("  - clustering_results.png: 聚类可视化")
        print("  - amino_acid_analysis.png: 氨基酸性质分析")
        print("  - clustering_results.pkl: 完整结果")
        print("  - clustering_results.csv: 结果表格")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()