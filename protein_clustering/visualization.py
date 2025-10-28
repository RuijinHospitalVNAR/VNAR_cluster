import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP


def plot_embedding(X, labels, method="umap", outfile=None, title="Clustering Visualization"):
    """
    绘制聚类结果的二维可视化 (UMAP / t-SNE)
    :param X: 特征矩阵 (n_samples, n_features)
    :param labels: 聚类标签 (array-like)
    :param method: "umap" 或 "tsne"
    :param outfile: 保存路径 (png)，默认不保存
    :param title: 图标题
    """
    if method == "umap":
        reducer = UMAP(n_components=2, random_state=42)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    else:
        raise ValueError("method must be 'umap' or 'tsne'")

    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(6, 5))
    palette = sns.color_palette("hls", len(set(labels)) - (1 if -1 in labels else 0))

    unique_labels = np.unique(labels)
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            color = "#cccccc"
            lbl = "Noise"
        else:
            color = palette[idx % len(palette)]
            lbl = f"Cluster {label}"
        plt.scatter(embedding[mask, 0], embedding[mask, 1], s=20, c=[color], label=lbl, alpha=0.7, edgecolors="none")

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300)
        plt.close()
    else:
        plt.show()
