from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

NATURE_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def plot_elbow_silhouette(k_values: Iterable[int], inertias: list[float], silhouettes: list[float]) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'o-', color=NATURE_PALETTE[0])
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.title('Elbow')

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouettes, 'o-', color=NATURE_PALETTE[1])
    plt.xlabel('K')
    plt.ylabel('Silhouette')
    plt.title('Silhouette')
    plt.tight_layout()


def plot_tsne_clusters(X: np.ndarray, labels: np.ndarray) -> None:
    if len(X) < 2:
        return
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    X2 = tsne.fit_transform(X)
    unique = np.unique(labels)
    colors = NATURE_PALETTE * ((len(unique) // len(NATURE_PALETTE)) + 1)
    for label, color in zip(unique, colors):
        mask = labels == label
        plt.scatter(X2[mask, 0], X2[mask, 1], c=color, label=str(label))
    plt.legend()
    plt.title('t-SNE')
