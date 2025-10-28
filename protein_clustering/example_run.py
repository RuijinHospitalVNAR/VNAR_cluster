"""
Example script to demonstrate usage of the protein_clustering package.
"""

from protein_clustering.analyzer import ProteinClusterAnalyzer
from protein_clustering.pipeline import run_pipeline


def demo_analyzer():
    print("=== Demo: Direct use of ProteinClusterAnalyzer ===")
    analyzer = ProteinClusterAnalyzer(
        cif_dir="./structures",  # directory containing .cif files
        antibody_chain="A",
        antigen_chains=["B", "C"],
        dist_cutoff=5.0,
        n_jobs=-1
    )
    features = analyzer.load_and_process_data(cache_file="protein_cache.pkl")
    print(f"Loaded {features.shape[0]} structures with {features.shape[1]} features.")

    X = analyzer.prepare_features(mode="engineered", use_pca=True, n_components=5)
    labels = analyzer.perform_clustering(X, method="kmeans", n_clusters=3)
    metrics = analyzer.evaluate(X, labels)
    print("Clustering metrics:", metrics)

    analyzer.save_results("demo_results.pkl", X)
    print("Results saved to demo_results.pkl and demo_results.csv")


def demo_pipeline():
    print("=== Demo: Run full pipeline ===")
    run_pipeline(
        structures="./structures",  # directory containing .cif files
        cache="protein_cache.pkl",
        outdir="results",
        metric="jaccard",
        n_clusters=5
    )
    print("Pipeline finished. Results are saved in ./results")


if __name__ == "__main__":
    demo_analyzer()
    demo_pipeline()
