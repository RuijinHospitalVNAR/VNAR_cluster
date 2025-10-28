#!/usr/bin/env python3
"""
cluster_pipeline.py
Stepwise clustering pipeline for antibody-antigen complexes focusing on contact-map first,
then adding engineered features as auxiliary input. Implements:
  1) contact-only clustering using Jaccard/Hamming similarity, visualize with UMAP/TSNE
  2) alpha sweep to combine contact + engineered (using prepare_features contact_alpha)
  3) block-wise dimensionality reduction: contact (UMAP) + engineered (PCA) then concat and cluster

Usage:
  python cluster_pipeline.py --structures ./structures_dir --cache cache.pkl --outdir results
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import umap
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False
try:
    import hdbscan
    HAVE_HDBSCAN = True
except ImportError:
    HAVE_HDBSCAN = False

from optimized_protein_clustering_v14_ab import ProteinClusterAnalyzer

def ensure_outdir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def contact_only_pipeline(analyzer, metric='jaccard', n_clusters=8, outdir='results'):
    outdir = ensure_outdir(outdir)
    contacts = np.array(analyzer.contact_maps)
    if contacts.size == 0:
        raise ValueError("No contact maps available.")
    contacts_bin = (contacts > 0.5).astype(int)
    logger.info(f"Contact matrix shape: {contacts_bin.shape}")
    D = pairwise_distances(contacts_bin, metric=metric)
    A = 1.0 - D
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = sc.fit_predict(A)
    try:
        pca = PCA(n_components=min(50, max(2, contacts_bin.shape[0]//10)))
        emb = pca.fit_transform(A)
        sil = silhouette_score(emb, labels)
    except Exception:
        sil = None
    if HAVE_UMAP:
        proj = umap.UMAP(n_components=2, random_state=42).fit_transform(contacts_bin)
        method = 'umap'
    else:
        proj = TSNE(n_components=2, random_state=42).fit_transform(contacts_bin)
        method = 'tsne'
    plt.figure(figsize=(8,6))
    palette = sns.color_palette("hsv", len(set(labels)))
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(proj[idx,0], proj[idx,1], label=f"C{lab}", s=20, alpha=0.8)
    plt.title(f"Contact-only ({metric}) projection ({method}), silhouette={sil:.3f}" if sil is not None else f"Contact-only ({metric}) projection ({method})")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(Path(outdir) / f"contact_only_{metric}_{method}.png", dpi=200)
    plt.close()
    np.save(Path(outdir) / f"contact_distance_{metric}.npy", D)
    pd.DataFrame({'file': analyzer.file_names, 'label': labels}).to_csv(Path(outdir) / f"contact_labels_{metric}.csv", index=False)
    return labels, D, proj, sil

def alpha_sweep(analyzer, alphas=[0.95,0.9,0.8,0.7,0.5,0.3,0.1], outdir='results'):
    outdir = ensure_outdir(outdir)
    results = []
    for a in alphas:
        logger.info(f"Preparing features with contact_alpha={a}")
        X = analyzer.prepare_features(feature_type='combined', use_pca=True, contact_alpha=a)
        if HAVE_HDBSCAN:
            labels = analyzer.perform_clustering(X, method='hdbscan')
        else:
            best = {'k':None, 'sil':-1, 'labels':None}
            for k in range(2, min(12, max(3, len(X)//5))):
                lbls = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
                try:
                    sil = silhouette_score(X, lbls)
                except Exception:
                    sil = -1
                if sil > best['sil']:
                    best = {'k':k, 'sil':sil, 'labels':lbls}
            labels = best['labels']
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = None
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        results.append({'alpha': a, 'n_clusters': n_clusters, 'silhouette': sil})
        pd.DataFrame({'file': analyzer.file_names, 'label': labels}).to_csv(Path(outdir) / f"labels_alpha_{a:.2f}.csv", index=False)
    pd.DataFrame(results).to_csv(Path(outdir) / "alpha_sweep_summary.csv", index=False)
    return results

def blockwise_dim_then_cluster(analyzer, contact_umap_dim=20, engineered_pca_dim=10, outdir='results'):
    outdir = ensure_outdir(outdir)
    contacts = np.array(analyzer.contact_maps)
    engineered = np.array(analyzer.feature_vectors) if analyzer.feature_vectors else np.zeros((contacts.shape[0],0))
    if contacts.size == 0:
        raise ValueError("No contact maps available")
    if HAVE_UMAP:
        contact_low = umap.UMAP(n_components=contact_umap_dim, random_state=42).fit_transform(contacts)
    else:
        contact_low = PCA(n_components=min(contact_umap_dim, max(1, contacts.shape[0]-1))).fit_transform(contacts)
    if engineered.size and engineered.ndim == 2 and engineered.shape[1] > 0:
        eng_low = PCA(n_components=min(engineered_pca_dim, max(1, engineered.shape[0]-1))).fit_transform(StandardScaler().fit_transform(engineered))
    else:
        eng_low = np.zeros((contact_low.shape[0], 0))
    X = np.hstack([contact_low, eng_low])
    if HAVE_HDBSCAN:
        labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X)
    else:
        labels = KMeans(n_clusters=min(8, max(2, len(X)//10)), random_state=42).fit_predict(X)
    proj = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(8,6))
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(proj[idx,0], proj[idx,1], s=15, alpha=0.8, label=f"C{lab}")
    plt.title("Blockwise-reduced clustering")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(Path(outdir) / "blockwise_cluster.png", dpi=200)
    plt.close()
    pd.DataFrame({'file': analyzer.file_names, 'label': labels}).to_csv(Path(outdir) / "blockwise_labels.csv", index=False)
    return labels, proj

def main():
    parser = argparse.ArgumentParser(description="Stepwise clustering pipeline (contact-first).")
    parser.add_argument('--structures', type=str, required=True, help="Directory with .pdb/.cif structure files")
    parser.add_argument('--cache', type=str, default="protein_cache.pkl", help="cache file path")
    parser.add_argument('--outdir', type=str, default="results", help="output directory")
    parser.add_argument('--metric', type=str, default='jaccard', choices=['jaccard','hamming'], help="distance metric for contact maps")
    parser.add_argument('--n_clusters', type=int, default=8, help="n clusters for spectral/KMeans baseline")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    analyzer = ProteinClusterAnalyzer(cif_dir=args.structures, antibody_chain='A', antigen_chains=['B','C'], dist_cutoff=5.0, n_jobs=-1)
    analyzer.load_and_process_data(cache_file=args.cache)

    labels_c, D, proj_c, sil_c = contact_only_pipeline(analyzer, metric=args.metric, n_clusters=args.n_clusters, outdir=outdir)
    print("Contact-only silhouette:", sil_c)
    sweep_res = alpha_sweep(analyzer, outdir=outdir)
    print("Alpha sweep summary saved to", Path(outdir) / "alpha_sweep_summary.csv")
    labels_block, proj_block = blockwise_dim_then_cluster(analyzer, outdir=outdir)
    print("Blockwise clustering saved to", Path(outdir) / "blockwise_labels.csv")

if __name__ == '__main__':
    main()
