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
        
        # 删除氨基酸性质字典设置
        # self._setup_amino_acid_properties()
        
    # 删除整个_setup_amino_acid_properties方法
    
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
        """从contact map提取几何特征"""
        if contact_map.size == 0:
            return np.zeros(15)  # 返回固定长度的零向量（15个特征）
        
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
        
        return np.array(features, dtype=np.float32)
    
    # 删除所有氨基酸性质相关的方法：
    # - extract_amino_acid_features
    # - _get_residue_properties
    # - _calculate_charge_complementarity
    # - _calculate_hydrophobic_interaction
    # - _calculate_polar_interaction
    # - _calculate_interface_complementarity

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
            
            # 提取几何特征（不包含氨基酸性质）
            features = self.extract_interaction_features(contact_map, contact_ab_residues, contact_ag_residues)
            
            return contact_map.flatten(), features, structure_file.name
            
        except Exception as e:
            logger.error(f"Error processing {structure_file}: {e}")
            return None, None, None

    # ... existing code ...

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
        
        # 4. 特征重要性（只包含几何特征）
        ax4 = axes[1, 1]
        if hasattr(self, 'feature_vectors') and len(self.feature_vectors) > 0:
            feature_matrix = np.array(self.feature_vectors)
            
            # 定义特征名称（只包含几何特征）
            geometric_features = [
                'Total_Contacts', 'Contact_Density', 'Contact_Variability',
                'Ab_Mean_Contacts', 'Ab_Std_Contacts', 'Ab_Max_Contacts', 
                'Ag_Mean_Contacts', 'Ag_Std_Contacts', 'Ag_Max_Contacts',
                'Ab_Region_Dispersion', 'Ag_Region_Dispersion', 
                'Ab_Residue_Count', 'Ag_Residue_Count',
                'Ab_Continuity', 'Ag_Continuity'
            ]
            
            # 确保特征名称数量与实际特征数量匹配
            if len(geometric_features) != feature_matrix.shape[1]:
                geometric_features = [f'Feature_{i}' for i in range(feature_matrix.shape[1])]
            
            # 计算每个特征的方差
            feature_variance = np.var(feature_matrix, axis=0)
            sorted_indices = np.argsort(feature_variance)[::-1][:15]  # 前15个最重要的特征
            
            bars = ax4.barh(range(len(sorted_indices)), 
                           feature_variance[sorted_indices])
            ax4.set_yticks(range(len(sorted_indices)))
            ax4.set_yticklabels([geometric_features[i] for i in sorted_indices])
            ax4.set_xlabel('Variance')
            ax4.set_title('Top 15 Feature Importance')
            
            # 所有特征都用蓝色（几何特征）
            for bar in bars:
                bar.set_color('skyblue')
                    
        else:
            ax4.text(0.5, 0.5, 'No engineered features available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()

    # 删除analyze_amino_acid_patterns方法

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
            
            # 准备特征（只包含几何特征和接触特征）
            X = analyzer.prepare_features(feature_type='combined', use_pca=True)
            
            # 执行聚类
            cluster_labels = analyzer.perform_clustering(X, method='hdbscan')
            
            # 评估聚类质量
            metrics = analyzer.evaluate_clustering(X)
            
            # 可视化结果
            analyzer.visualize_results(X, save_path='clustering_results.png')
            
            # 删除氨基酸性质分析
            # aa_analysis = analyzer.analyze_amino_acid_patterns(save_path='amino_acid_analysis.png')
            
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
            print("  - clustering_results.pkl: 完整结果")
            print("  - clustering_results.csv: 结果表格")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise