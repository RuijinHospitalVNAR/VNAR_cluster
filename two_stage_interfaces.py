import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from interface_clustering.contact_map import load_contact_maps
from interface_clustering.clustering import cluster_kmeans, cluster_hdbscan, _pad_flatten_maps
from interface_clustering.viz import plot_tsne_clusters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Two-stage interface clustering (coarse KMeans -> fine HDBSCAN)')
    p.add_argument('--cif_dir', type=str, required=True, help='Directory with .cif files')
    p.add_argument('--antibody_chain', type=str, default='A')
    p.add_argument('--antigen_chains', type=str, nargs='+', default=['B'])
    p.add_argument('--cutoff', type=float, default=5.0)
    # coarse
    p.add_argument('--k_coarse', type=int, default=10)
    p.add_argument('--coarse_random_state', type=int, default=0)
    # fine
    p.add_argument('--min_cluster_size', type=int, default=10)
    p.add_argument('--min_samples', type=int, default=None)
    p.add_argument('--epsilon', type=float, default=0.0)
    # io
    p.add_argument('--out_dir', type=str, default='results_two_stage')
    p.add_argument('--save_plots', action='store_true')
    p.add_argument('--save_numpy', action='store_true')
    return p.parse_args()


def run_two_stage(cmaps: List[np.ndarray], k_coarse: int, min_cluster_size: int, min_samples: int | None, epsilon: float, coarse_random_state: int = 0) -> Tuple[np.ndarray, Dict]:
    labels_coarse, metrics_coarse = cluster_kmeans(cmaps, k=k_coarse, random_state=coarse_random_state)
    X = _pad_flatten_maps(cmaps)
    labels_fine = np.full_like(labels_coarse, fill_value=-1)
    fine_metrics: Dict[int, Dict] = {}
    for c in range(k_coarse):
        idx = np.where(labels_coarse == c)[0]
        if len(idx) == 0:
            continue
        sub_maps = [cmaps[i] for i in idx]
        sub_labels, sub_metrics = cluster_hdbscan(sub_maps, min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=epsilon)
        fine_metrics[c] = sub_metrics
        # offset labels per coarse cluster to make unique final labels
        if len(sub_labels) > 0:
            # map -1 noise stays -1; others become unique using coarse cluster id
            max_label = sub_labels.max() if (sub_labels != -1).any() else -1
            for j, lab in zip(idx, sub_labels):
                if lab == -1:
                    labels_fine[j] = -1
                else:
                    labels_fine[j] = c * 1000 + lab
    metrics = {
        'coarse': metrics_coarse,
        'fine_per_coarse': fine_metrics
    }
    return labels_coarse, labels_fine, X, metrics


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files, cmaps = load_contact_maps(args.cif_dir, args.antibody_chain, args.antigen_chains, args.cutoff)

    labels_coarse, labels_fine, X, metrics = run_two_stage(
        cmaps=cmaps,
        k_coarse=args.k_coarse,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        epsilon=args.epsilon,
        coarse_random_state=args.coarse_random_state,
    )

    np.savetxt(out / 'labels_coarse.txt', labels_coarse, fmt='%d')
    np.savetxt(out / 'labels_fine.txt', labels_fine, fmt='%d')
    with open(out / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out / 'files.txt', 'w', encoding='utf-8') as f:
        for name in files:
            f.write(f"{name}\n")

    if args.save_plots:
        plot_tsne_clusters(X, labels_coarse)
        plt.title('t-SNE (coarse)')
        plt.savefig(out / 'tsne_coarse.png', dpi=200)
        plt.close()
        plot_tsne_clusters(X, labels_fine)
        plt.title('t-SNE (fine)')
        plt.savefig(out / 'tsne_fine.png', dpi=200)
        plt.close()

    if args.save_numpy:
        np.save(out / 'X.npy', X)


if __name__ == '__main__':
    main()
