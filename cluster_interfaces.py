import argparse
import json
from pathlib import Path
import numpy as np
from typing import List
from interface_clustering.contact_map import load_contact_maps
from interface_clustering.clustering import cluster_kmeans, cluster_hdbscan, _pad_flatten_maps
from interface_clustering.viz import plot_elbow_silhouette, plot_tsne_clusters
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Interface clustering via contact maps')
    p.add_argument('--cif_dir', type=str, required=True, help='Directory containing .cif files')
    p.add_argument('--antibody_chain', type=str, default='A')
    p.add_argument('--antigen_chains', type=str, nargs='+', default=['B'])
    p.add_argument('--cutoff', type=float, default=5.0)
    p.add_argument('--method', type=str, choices=['kmeans', 'hdbscan'], default='hdbscan')
    p.add_argument('--k', type=int, default=10, help='K for KMeans')
    p.add_argument('--min_cluster_size', type=int, default=10)
    p.add_argument('--min_samples', type=int, default=None)
    p.add_argument('--epsilon', type=float, default=0.0)
    p.add_argument('--out_dir', type=str, default='results_interfaces')
    p.add_argument('--save_plots', action='store_true')
    p.add_argument('--save_numpy', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    file_names, cmaps = load_contact_maps(args.cif_dir, args.antibody_chain, args.antigen_chains, args.cutoff)

    if args.method == 'kmeans':
        labels, metrics = cluster_kmeans(cmaps, k=args.k)
    else:
        labels, metrics = cluster_hdbscan(cmaps, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, cluster_selection_epsilon=args.epsilon)

    # Save labels and metrics
    np.savetxt(out / 'labels.txt', labels, fmt='%d')
    with open(out / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out / 'files.txt', 'w', encoding='utf-8') as f:
        for name in file_names:
            f.write(f"{name}\n")

    # Plots
    X = _pad_flatten_maps(cmaps)
    if args.save_plots:
        if args.method == 'kmeans':
            inertias = []
            silhouettes = []
            k_values: List[int] = list(range(max(2, args.k - 5), args.k + 6))
            for k in k_values:
                lab, met = cluster_kmeans(cmaps, k)
                inertias.append(met.get('inertia', np.nan))
                silhouettes.append(met.get('silhouette', np.nan))
            plot_elbow_silhouette(k_values, inertias, silhouettes)
            plt.savefig(out / 'kmeans_elbow_silhouette.png', dpi=200)
            plt.close()
        plot_tsne_clusters(X, labels)
        plt.savefig(out / 'tsne.png', dpi=200)
        plt.close()

    if args.save_numpy:
        np.save(out / 'X.npy', X)


if __name__ == '__main__':
    main()
