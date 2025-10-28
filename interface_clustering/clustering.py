from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan


def _pad_flatten_maps(contact_maps: list[np.ndarray]) -> np.ndarray:
    # Pad variable-size contact maps to a common rectangle then flatten
    max_rows = max(m.shape[0] for m in contact_maps) if contact_maps else 0
    max_cols = max(m.shape[1] for m in contact_maps) if contact_maps else 0
    X = np.zeros((len(contact_maps), max_rows * max_cols), dtype=np.uint8)
    for i, m in enumerate(contact_maps):
        r, c = m.shape
        padded = np.zeros((max_rows, max_cols), dtype=np.uint8)
        padded[:r, :c] = m
        X[i, :] = padded.reshape(-1)
    return X


def cluster_kmeans(contact_maps: list[np.ndarray], k: int, random_state: int = 0, n_init: int = 10) -> Tuple[np.ndarray, Dict[str, float]]:
    X = _pad_flatten_maps(contact_maps)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    metrics: Dict[str, float] = {}
    if k > 1 and len(X) > k:
        metrics['silhouette'] = float(silhouette_score(X, labels))
    metrics['inertia'] = float(model.inertia_)
    return labels, metrics


def cluster_hdbscan(contact_maps: list[np.ndarray], min_cluster_size: int = 10, min_samples: Optional[int] = None, cluster_selection_epsilon: float = 0.0) -> Tuple[np.ndarray, Dict[str, float]]:
    X = _pad_flatten_maps(contact_maps)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(X)
    metrics: Dict[str, float] = {}
    valid_mask = labels != -1
    if valid_mask.any() and len(np.unique(labels[valid_mask])) > 1:
        metrics['silhouette_valid'] = float(silhouette_score(X[valid_mask], labels[valid_mask]))
    return labels, metrics
