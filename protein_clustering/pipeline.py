import argparse
from pathlib import Path
import json
from .analyzer import ProteinClusterAnalyzer
from .utils import compute_clustering_metrics


def run_pipeline(structures, cache, outdir, metric='jaccard', n_clusters=8):
    analyzer = ProteinClusterAnalyzer(
        cif_dir=structures,
        antibody_chain='A',
        antigen_chains=['B','C'],
        dist_cutoff=5.0,
        n_jobs=-1
    )
    analyzer.load_and_process_data(cache_file=cache)
    Path(outdir).mkdir(exist_ok=True)

    results_summary = {}

    # Step 1: contact-only
    X_contact = analyzer.prepare_features("contact_map")
    labels_contact = analyzer.perform_clustering(X_contact, method="spectral", n_clusters=n_clusters)
    metrics_contact = compute_clustering_metrics(X_contact, labels_contact)
    results_summary['contact_only'] = metrics_contact
    analyzer.save_results(Path(outdir)/"contact_only.pkl", X_contact)

    # Step 2: alpha sweep
    sweep_results = []
    for alpha in [0.95, 0.8, 0.5, 0.3, 0.1]:
        X_comb = analyzer.prepare_features("combined", use_pca=True)
        labels = analyzer.perform_clustering(X_comb, method="hdbscan")
        metrics = compute_clustering_metrics(X_comb, labels)
        sweep_results.append({"alpha": alpha, **metrics})
    results_summary['alpha_sweep'] = sweep_results

    # Step 3: blockwise
    X_block = analyzer.prepare_features("combined", use_pca=True)
    labels_block = analyzer.perform_clustering(X_block, method="hdbscan")
    metrics_block = compute_clustering_metrics(X_block, labels_block)
    results_summary['blockwise'] = metrics_block
    analyzer.save_results(Path(outdir)/"blockwise.pkl", X_block)

    # 保存 summary
    with open(Path(outdir)/"analysis_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures", required=True)
    parser.add_argument("--cache", default="protein_cache.pkl")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--metric", default="jaccard")
    parser.add_argument("--n_clusters", type=int, default=8)
    args = parser.parse_args()
    run_pipeline(args.structures, args.cache, args.outdir, args.metric, args.n_clusters)


if __name__ == "__main__":
    main()
