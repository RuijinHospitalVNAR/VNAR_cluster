import os
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import hdbscan
from sklearn.manifold import TSNE
import seaborn as sns

# 用户参数
CIF_DIR = r"./your_cif_dir"  # 修改为你的CIF文件夹路径
ANTIBODY_CHAIN = 'A'
ANTIGEN_CHAINS = ['B', 'C']  # 如果只有B链也没问题
DIST_CUTOFF = 5.0

# HDBSCAN参数
MIN_CLUSTER_SIZE = 10  # 最小聚类大小
MIN_SAMPLES = 5        # 最小样本数
CLUSTER_SELECTION_EPSILON = 0.1  # 聚类选择epsilon

# 提取指定链的所有Cα原子坐标
def extract_ca_coords(cif_file, chain_ids):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('model', cif_file)
    ca_coords = []
    for model in structure:
        for chain in model:
            if chain.id in chain_ids:
                for res in chain:
                    if 'CA' in res:
                        ca_coords.append(res['CA'].get_coord())
    return np.array(ca_coords)

# 计算contact map（抗体Cα vs 抗原Cα）
def compute_contact_map(coords_ab, coords_ag, cutoff=5.0):
    if len(coords_ab) == 0 or len(coords_ag) == 0:
        return np.zeros((len(coords_ab), len(coords_ag)), dtype=int)
    dists = np.linalg.norm(coords_ab[:, None, :] - coords_ag[None, :, :], axis=-1)
    return (dists < cutoff).astype(int)

# 主流程
cif_files = [os.path.join(CIF_DIR, f) for f in os.listdir(CIF_DIR) if f.endswith('.cif')]
contact_maps = []
map_shapes = []

print(f"共检测到{len(cif_files)}个CIF文件，正在提取contact map...")
for cif_file in tqdm(cif_files):
    ab_coords = extract_ca_coords(cif_file, [ANTIBODY_CHAIN])
    ag_coords = extract_ca_coords(cif_file, ANTIGEN_CHAINS)
    contact_map = compute_contact_map(ab_coords, ag_coords, DIST_CUTOFF)
    map_shapes.append(contact_map.shape)
    contact_maps.append(contact_map.flatten())

# 检查所有contact map长度是否一致
lengths = [len(x) for x in contact_maps]
if len(set(lengths)) != 1:
    print("Error: 不同结构的contact map长度不一致，请检查Cα原子数量是否一致！")
    exit(1)

X = np.array(contact_maps)

# HDBSCAN聚类
print("正在进行HDBSCAN聚类...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
    metric='euclidean'
)
cluster_labels = clusterer.fit_predict(X)

# 统计聚类结果
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)
print(f"聚类结果：")
print(f"  总样本数: {len(X)}")
print(f"  聚类数量: {n_clusters}")
print(f"  噪声点数量: {n_noise}")
print(f"  聚类标签分布: {np.bincount(cluster_labels + 1)}")  # +1是为了处理-1标签

# 计算聚类质量指标
if n_clusters > 1:
    # 过滤掉噪声点，只计算有标签的样本
    valid_mask = cluster_labels != -1
    if np.sum(valid_mask) > 1:
        valid_labels = cluster_labels[valid_mask]
        valid_X = X[valid_mask]
        silhouette_avg = silhouette_score(valid_X, valid_labels)
        print(f"  Silhouette Score (排除噪声点): {silhouette_avg:.3f}")

# 可视化聚类结果
plt.figure(figsize=(15, 5))

# 1. 聚类标签分布
plt.subplot(1, 3, 1)
unique_labels = np.unique(cluster_labels)
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(unique_labels, colors):
    mask = cluster_labels == label
    plt.scatter(range(len(cluster_labels))[mask], 
                [label] * np.sum(mask), 
                c=[color], label=f'Cluster {label}' if label != -1 else 'Noise')
plt.xlabel('Sample Index')
plt.ylabel('Cluster Label')
plt.title('Cluster Assignment')
plt.legend()

# 2. 聚类大小分布
plt.subplot(1, 3, 2)
cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels if label != -1]
if cluster_sizes:
    plt.bar(range(len(cluster_sizes)), cluster_sizes)
    plt.xlabel('Cluster Index')
    plt.ylabel('Cluster Size')
    plt.title('Cluster Size Distribution')
else:
    plt.text(0.5, 0.5, 'No valid clusters found', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Cluster Size Distribution')

# 3. t-SNE降维可视化
plt.subplot(1, 3, 3)
if len(X) > 1:
    # 使用t-SNE降维到2D进行可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X)
    
    for label, color in zip(unique_labels, colors):
        mask = cluster_labels == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                    c=[color], label=f'Cluster {label}' if label != -1 else 'Noise')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization')
    plt.legend()

plt.tight_layout()
plt.show()

# 保存聚类结果
results = {
    'file_names': [os.path.basename(f) for f in cif_files],
    'cluster_labels': cluster_labels,
    'n_clusters': n_clusters,
    'n_noise': n_noise,
    'contact_maps': contact_maps
}

# 输出每个聚类的代表性样本
print("\n各聚类代表性样本：")
for label in unique_labels:
    if label != -1:
        cluster_indices = np.where(cluster_labels == label)[0]
        print(f"Cluster {label} ({len(cluster_indices)} samples):")
        print(f"  样本文件: {[results['file_names'][i] for i in cluster_indices[:5]]}")  # 显示前5个
        if len(cluster_indices) > 5:
            print(f"  ... 还有 {len(cluster_indices) - 5} 个样本")

print("\n聚类分析完成！") 