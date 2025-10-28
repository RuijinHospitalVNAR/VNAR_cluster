import os
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# 用户参数
CIF_DIR = r"./your_cif_dir"  # 修改为你的CIF文件夹路径
ANTIBODY_CHAIN = 'A'
ANTIGEN_CHAINS = ['B', 'C']  # 如果只有B链也没问题
DIST_CUTOFF = 5.0

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

# KMeans聚类与K值评估
sse = []
silhouette = []
K_range = range(5, 21)
print("正在进行KMeans聚类与K值评估...")
for k in tqdm(K_range):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X)
    sse.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X, kmeans.labels_))

# 画Elbow和Silhouette曲线
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(K_range, sse, 'o-')
plt.xlabel('K')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method')
plt.subplot(1,2,2)
plt.plot(K_range, silhouette, 'o-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.show()

print("聚类分析完成！") 