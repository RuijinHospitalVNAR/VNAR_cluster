import os
import pickle
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
import hdbscan
from .utils import compute_clustering_metrics


class ProteinClusterAnalyzer:
    def __init__(self, cif_dir, antibody_chain="A", antigen_chains=None, dist_cutoff=5.0, n_jobs=-1):
        self.cif_dir = cif_dir
        self.antibody_chain = antibody_chain
        self.antigen_chains = antigen_chains or ["B", "C"]
        self.dist_cutoff = dist_cutoff
        self.n_jobs = n_jobs

        self.features = None
        self.labels = None

    def load_and_process_data(self, cache_file=None):
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.features = pickle.load(f)
            return self.features
        
        parser = MMCIFParser(QUIET=True)
        features = []
        for fname in os.listdir(self.cif_dir):
            if fname.endswith(".cif"):
                path = os.path.join(self.cif_dir, fname)
                try:
                    structure = parser.get_structure(fname, path)
                    # Dummy: 提取原子数作为简单特征
                    n_atoms = len(list(structure.get_atoms()))
                    features.append([n_atoms])
                except Exception as e:
                    print(f"Failed to parse {fname}: {e}")
        self.features = np.array(features)

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump(self.features, f)
        return self.features

    def prepare_features(self, mode="engineered", alpha=0.5, use_pca=False, n_components=10):
        X = np.array(self.features)
        if use_pca and X.shape[1] > n_components:
            X = PCA(n_components=n_components).fit_transform(X)
        return StandardScaler().fit_transform(X)

    def perform_clustering(self, X, method="hdbscan", n_clusters=8):
        if method == "hdbscan":
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, core_dist_n_jobs=self.n_jobs)
            labels = clusterer.fit_predict(X)
        elif method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(X)
        elif method == "spectral":
            clusterer = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", random_state=42)
            labels = clusterer.fit_predict(X)
        else:
            raise ValueError(f"Unknown method {method}")

        self.labels = labels
        return labels

    def evaluate(self, X, labels):
        return compute_clustering_metrics(X, labels)

    def save_results(self, outfile, X):
        data = pd.DataFrame(X)
        data["cluster"] = self.labels
        data.to_csv(str(outfile).replace(".pkl", ".csv"), index=False)
        with open(outfile, "wb") as f:
            pickle.dump({"X": X, "labels": self.labels}, f)
