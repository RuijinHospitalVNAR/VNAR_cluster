import os
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
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

# 忽略警告
warnings.filterwarnings('ignore')

class ProteinClusterAnalyzer:
    def __init__(self, cif_dir, antibody_chain='A', antigen_chains=['B', 'C'], 
                 dist_cutoff=5.0, n_jobs=-1):
        """
        初始化蛋白质聚类分析器
        
        Parameters:
        -----------
        cif_dir : str
            CIF文件目录路径
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
        
    def extract_ca_coords(self, cif_file, chain_ids):
        """提取指定链的所有Cα原子坐标"""
        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('model', cif_file)
            ca_coords = []
            chain_info = {}
            
            for model in structure:
                for chain in model:
                    if chain.id in chain_ids:
                        chain_coords = []
                        for res in chain:
                            if 'CA' in res:
                                coord = res['CA'].get_coord()
                                chain_coords.append(coord)
                        if chain_coords:
                            ca_coords.extend(chain_coords)
                            chain_info[chain.id] = len(chain_coords)
            
            return np.array(ca_coords), chain_info
        except Exception as e:
            logger.warning(f"Error processing {cif_file}: {e}")
            return np.array([]), {}

    def compute_contact_map(self, coords_ab, coords_ag):
        """计算contact map"""
        if len(coords_ab) == 0 or len(coords_ag) == 0:
            return np.array([])
        
        # 使用更高效的距离计算
        dists = np.linalg.norm(coords_ab[:, None, :] - coords_ag[None, :, :], axis=-1)
        contact_map = (dists < self.dist_cutoff).astype(np.float32)
        return contact_map

    def extract_interaction_features(self, contact_map):
        """从contact map提取多种特征"""
        if contact_map.size == 0:
            return np.array([])
        
        features = []
        
        # 1. 基本统计特征
        features.extend([
            np.sum(contact_map),  # 总接触数
            np.mean(contact_map),  # 接触密度
            np.std(contact_map),   # 接触变异性
        ])
        
        # 2. 行列统计（抗体和抗原残基的接触模式）
        row_sums = np.sum(contact_map, axis=1)
        col_sums = np.sum(contact_map, axis=0)
        
        features.extend([
            np.mean(row_sums), np.std(row_sums), np.max(row_sums),
            np.mean(col_sums), np.std(col_sums), np.max(col_sums),
        ])
        
        # 3. 接触区域的几何特征
        if np.sum(contact_map) > 0:
            contact_rows, contact_cols = np.where(contact_map > 0)
            features.extend([
                np.std(contact_rows),  # 抗体接触区域分散度
                np.std(contact_cols),  # 抗原接触区域分散度
                len(np.unique(contact_rows)),  # 参与接触的抗体残基数
                len(np.unique(contact_cols)),  # 参与接触的抗原残基数
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 4. 接触模式特征（局部连续性）
        if contact_map.shape[0] > 1 and contact_map.shape[1] > 1:
            # 计算相邻接触的连续性
            row_continuity = np.sum(contact_map[:-1, :] * contact_map[1:, :])
            col_continuity = np.sum(contact_map[:, :-1] * contact_map[:, 1:])
            features.extend([row_continuity, col_continuity])
        else:
            features.extend([0, 0])
        
        return np.array(features, dtype=np.float32)

    def process_single_file(self, cif_file):
        """处理单个CIF文件"""
        try:
            # 提取坐标
            ab_coords, ab_info = self.extract_ca_coords(cif_file, [self.antibody_chain])
            ag_coords, ag_info = self.extract_ca_coords(cif_file, self.antigen_chains)
            
            if len(ab_coords) == 0 or len(ag_coords) == 0:
                logger.warning(f"No valid coordinates found in {cif_file}")
                return None, None, None
            
            # 计算contact map
            contact_map = self.compute_contact_map(ab_coords, ag_coords)
            
            if contact_map.size == 0:
                return None, None, None
            
            # 提取特征
            features = self.extract_interaction_features(contact_map)
            
            return contact_map.flatten(), features, cif_file.name
            
        except Exception as e:
            logger.error(f"Error processing {cif_file}: {e}")
            return None, None, None

    def load_and_process_data(self, cache_file=None):
        """加载和处理所有CIF文件"""
        # 检查缓存
        if cache_file and Path(cache_file).exists():
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.contact_maps = cached_data['contact_maps']
                self.feature_vectors = cached_data['feature_vectors']
                self.file_names = cached_data['file_names']
                return
        
        cif_files = list(self.cif_dir.glob('*.cif'))
        logger.info(f"Found {len(cif_files)} CIF files")
        
        if len(cif_files) == 0:
            raise ValueError("No CIF files found in the specified directory")
        
        # 并行处理文件
        logger.info("Processing CIF files in parallel...")
        with Pool(processes=self.n_jobs) as pool:
            results = list(tqdm(
                pool.map(self.process_single_file, cif_files),
                total=len(cif_files),
                desc="Processing CIF files"
            ))
        
        # 过滤有效结果
        valid_results = [(cm, fv, fn) for cm, fv, fn in results if cm is not None]
        
        if len(valid_results) == 0:
            raise ValueError("No valid results obtained from CIF files")
        
        logger.info(f"Successfully processed {len(valid_results)}/{len(cif_files)} files")
        
        self.contact_maps, self.feature_vectors, self.file_names = zip(*valid_results)
        self.contact_maps = list(self.contact_maps)
        self.feature_vectors = list(self.feature_vectors)
        self.file_names = list(self.file_names)
        
        # 缓存结果
        if cache_file:
            logger.info(f"Caching results to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'contact_maps': self.contact_maps,
                    'feature_vectors': self.feature_vectors,
                    'file_names': self.file_names
                }, f)

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
        
        return {'n_clusters': n_clusters, 'n_noise': n_noise}

    def visualize_results(self, X, save_path=None):
        """可视化聚类结果"""
        if self.cluster_labels is None:
            raise ValueError("No clustering results found.")
        
        plt.style.use('seaborn-v0_8')
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
            perplexity = min(30, len(X) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                       n_jobs=self.n_jobs)
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
        
        # 4. 特征重要性（如果有工程化特征）
        ax4 = axes[1, 1]
        if hasattr(self, 'feature_vectors') and len(self.feature_vectors) > 0:
            feature_matrix = np.array(self.feature_vectors)
            feature_names = [f'Feature_{i}' for i in range(feature_matrix.shape[1])]
            
            # 计算每个特征的方差
            feature_variance = np.var(feature_matrix, axis=0)
            sorted_indices = np.argsort(feature_variance)[::-1][:10]  # 前10个最重要的特征
            
            ax4.barh(range(len(sorted_indices)), 
                    feature_variance[sorted_indices])
            ax4.set_yticks(range(len(sorted_indices)))
            ax4.set_yticklabels([feature_names[i] for i in sorted_indices])
            ax4.set_xlabel('Variance')
            ax4.set_title('Top Feature Importance (by Variance)')
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
            closest_indices = np.argsort(distances)[:n_representatives]
            
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

def main():
    """主函数"""
    # 配置参数
    CIF_DIR = "./your_cif_dir"  # 修改为你的CIF文件夹路径
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
        
        # 准备特征
        X = analyzer.prepare_features(feature_type='combined', use_pca=True)
        
        # 执行聚类
        cluster_labels = analyzer.perform_clustering(X, method='hdbscan')
        
        # 评估聚类质量
        metrics = analyzer.evaluate_clustering(X)
        
        # 可视化结果
        analyzer.visualize_results(X, save_path='clustering_results.png')
        
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
        
        print("聚类分析完成！")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()