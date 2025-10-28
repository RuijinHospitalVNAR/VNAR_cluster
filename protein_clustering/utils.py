import logging
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

logger = logging.getLogger(__name__)


def compute_clustering_metrics(X, labels):
    """统一的聚类评估指标"""
    metrics = {}
    mask = labels != -1
    if np.sum(mask) < 2:
        return metrics

    X_filtered, labels_filtered = X[mask], labels[mask]
    if len(set(labels_filtered)) > 1:
        try:
            metrics['silhouette'] = silhouette_score(X_filtered, labels_filtered)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_filtered, labels_filtered)
            metrics['davies_bouldin'] = davies_bouldin_score(X_filtered, labels_filtered)
        except Exception as e:
            logger.warning(f"Metric calculation failed: {e}")

    metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
    metrics['noise_ratio'] = np.sum(labels == -1) / len(labels)
    return metrics
