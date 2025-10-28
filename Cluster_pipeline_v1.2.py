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
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import logging
from typing import Tuple, List, Dict, Optional, Union
import pickle
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle optional dependencies
try:
    import umap
    HAVE_UMAP = True
    logger.info("UMAP available")
except ImportError:
    HAVE_UMAP = False
    logger.warning("UMAP not available, falling back to t-SNE")

try:
    import hdbscan
    HAVE_HDBSCAN = True
    logger.info("HDBSCAN available")
except ImportError:
    HAVE_HDBSCAN = False
    logger.warning("HDBSCAN not available, using KMeans")

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Import your custom analyzer - handle import error gracefully
try:
    from optimized_protein_clustering_v14_ab import ProteinClusterAnalyzer
except ImportError as e:
    logger.error(f"Failed to import ProteinClusterAnalyzer: {e}")
    raise ImportError("Required module 'optimized_protein_clustering_v14_ab' not found")

@dataclass
class ClusteringResult:
    """Data class to store clustering results"""
    labels: np.ndarray
    projection: Optional[np.ndarray] = None
    distance_matrix: Optional[np.ndarray] = None
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    n_clusters: int = 0
    method: str = ""
    parameters: Optional[Dict] = None

def ensure_outdir(path: Union[str, Path]) -> Path:
    """Create output directory if it doesn't exist"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def validate_inputs(contacts: np.ndarray, file_names: List[str]) -> None:
    """Validate input data consistency"""
    if contacts.size == 0:
        raise ValueError("No contact maps available")
    if len(file_names) != contacts.shape[0]:
        raise ValueError(f"Mismatch: {len(file_names)} files but {contacts.shape[0]} contact maps")
    logger.info(f"Validated {contacts.shape[0]} structures with contact maps of shape {contacts.shape[1:]}")

def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute multiple clustering quality metrics"""
    metrics = {}
    
    # Filter out noise points for HDBSCAN (-1 labels)
    mask = labels != -1
    if np.sum(mask) < 2:
        logger.warning("Too few non-noise points for metric calculation")
        return metrics
    
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    try:
        if len(set(labels_filtered)) > 1:
            metrics['silhouette'] = silhouette_score(X_filtered, labels_filtered)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_filtered, labels_filtered)
        else:
            logger.warning("Only one cluster found, cannot compute silhouette score")
    except Exception as e:
        logger.warning(f"Error computing clustering metrics: {e}")
    
    metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
    metrics['noise_ratio'] = np.sum(labels == -1) / len(labels)
    
    return metrics

def optimal_clusters_search(X: np.ndarray, k_range: range = None) -> Tuple[int, float]:
    """Find optimal number of clusters using silhouette score"""
    if k_range is None:
        k_range = range(2, min(12, max(3, len(X)//5 + 1)))
    
    best_score = -1
    best_k = 2
    
    for k in k_range:
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            logger.warning(f"Failed to compute silhouette for k={k}: {e}")
            continue
    
    return best_k, best_score

def contact_only_pipeline(
    analyzer, 
    metric: str = 'jaccard', 
    n_clusters: int = 8, 
    outdir: Union[str, Path] = 'results'
) -> ClusteringResult:
    """
    Perform contact-only clustering pipeline
    
    Args:
        analyzer: ProteinClusterAnalyzer instance
        metric: Distance metric ('jaccard' or 'hamming')
        n_clusters: Number of clusters for spectral clustering
        outdir: Output directory
    
    Returns:
        ClusteringResult object with results
    """
    outdir = ensure_outdir(outdir)
    
    # Get and validate data
    contacts = np.array(analyzer.contact_maps)
    validate_inputs(contacts, analyzer.file_names)
    
    # Binarize contact maps
    contacts_bin = (contacts > 0.5).astype(int)
    logger.info(f"Processing {contacts_bin.shape[0]} contact maps with {contacts_bin.shape[1]} contacts each")
    
    # Compute distance matrix and affinity
    logger.info(f"Computing pairwise distances using {metric} metric")
    D = pairwise_distances(contacts_bin, metric=metric)
    A = 1.0 - D
    
    # Perform spectral clustering
    logger.info(f"Performing spectral clustering with {n_clusters} clusters")
    sc = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', 
        random_state=42,
        n_init=10
    )
    labels = sc.fit_predict(A)
    
    # Compute clustering quality metrics
    metrics = compute_clustering_metrics(contacts_bin, labels)
    
    # Dimensionality reduction for visualization
    logger.info("Computing 2D projection for visualization")
    if HAVE_UMAP:
        projector = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(contacts_bin)-1))
        proj = projector.fit_transform(contacts_bin)
        method = 'UMAP'
    else:
        # Use PCA if too many samples for t-SNE
        if len(contacts_bin) > 1000:
            pca_pre = PCA(n_components=50)
            contacts_reduced = pca_pre.fit_transform(contacts_bin)
            proj = TSNE(n_components=2, random_state=42, perplexity=min(30, len(contacts_bin)-1)).fit_transform(contacts_reduced)
        else:
            proj = TSNE(n_components=2, random_state=42, perplexity=min(30, len(contacts_bin)-1)).fit_transform(contacts_bin)
        method = 't-SNE'
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("tab10", len(set(labels)))
    
    for i, lab in enumerate(np.unique(labels)):
        idx = labels == lab
        plt.scatter(
            proj[idx, 0], proj[idx, 1], 
            label=f"Cluster {lab} (n={np.sum(idx)})", 
            s=30, alpha=0.7, c=[palette[i]]
        )
    
    title = f"Contact-only clustering ({metric} + {method})\n"
    if 'silhouette' in metrics:
        title += f"Silhouette: {metrics['silhouette']:.3f}, "
    if 'calinski_harabasz' in metrics:
        title += f"CH Index: {metrics['calinski_harabasz']:.1f}"
    
    plt.title(title)
    plt.xlabel(f"{method} Component 1")
    plt.ylabel(f"{method} Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    # Save results
    plot_path = outdir / f"contact_only_{metric}_{method.lower()}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {plot_path}")
    
    # Save data
    np.save(outdir / f"contact_distance_{metric}.npy", D)
    results_df = pd.DataFrame({
        'file': analyzer.file_names, 
        'label': labels,
        'proj_x': proj[:, 0],
        'proj_y': proj[:, 1]
    })
    results_df.to_csv(outdir / f"contact_labels_{metric}.csv", index=False)
    
    return ClusteringResult(
        labels=labels,
        projection=proj,
        distance_matrix=D,
        silhouette_score=metrics.get('silhouette'),
        calinski_harabasz_score=metrics.get('calinski_harabasz'),
        n_clusters=metrics['n_clusters'],
        method=f"spectral_{metric}",
        parameters={'metric': metric, 'n_clusters': n_clusters}
    )

def alpha_sweep(
    analyzer, 
    alphas: List[float] = None, 
    outdir: Union[str, Path] = 'results'
) -> List[Dict]:
    """
    Perform alpha sweep to find optimal contact/engineered feature balance
    
    Args:
        analyzer: ProteinClusterAnalyzer instance
        alphas: List of alpha values to test
        outdir: Output directory
    
    Returns:
        List of dictionaries with results for each alpha
    """
    if alphas is None:
        alphas = [0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]
    
    outdir = ensure_outdir(outdir)
    results = []
    
    logger.info(f"Starting alpha sweep with values: {alphas}")
    
    for a in alphas:
        logger.info(f"Processing contact_alpha={a}")
        
        try:
            # Prepare combined features
            X = analyzer.prepare_features(
                feature_type='combined', 
                use_pca=True, 
                contact_alpha=a
            )
            
            if X is None or X.size == 0:
                logger.warning(f"No features generated for alpha={a}")
                continue
            
            # Perform clustering
            if HAVE_HDBSCAN:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(2, len(X)//20),
                    min_samples=1
                )
                labels = clusterer.fit_predict(X)
                method = 'HDBSCAN'
            else:
                # Find optimal k
                k_opt, _ = optimal_clusters_search(X)
                labels = KMeans(
                    n_clusters=k_opt, 
                    random_state=42, 
                    n_init=10
                ).fit_predict(X)
                method = f'KMeans(k={k_opt})'
            
            # Compute metrics
            metrics = compute_clustering_metrics(X, labels)
            
            result = {
                'alpha': a,
                'method': method,
                'n_clusters': metrics['n_clusters'],
                'silhouette': metrics.get('silhouette'),
                'calinski_harabasz': metrics.get('calinski_harabasz'),
                'noise_ratio': metrics.get('noise_ratio', 0)
            }
            
            results.append(result)
            logger.info(f"Alpha {a}: {metrics['n_clusters']} clusters, "
                       f"silhouette={metrics.get('silhouette', 'N/A'):.3f}")
            
            # Save labels
            labels_df = pd.DataFrame({
                'file': analyzer.file_names, 
                'label': labels
            })
            labels_df.to_csv(outdir / f"labels_alpha_{a:.2f}.csv", index=False)
            
        except Exception as e:
            logger.error(f"Error processing alpha={a}: {e}")
            results.append({
                'alpha': a, 
                'method': 'FAILED', 
                'n_clusters': 0, 
                'silhouette': None,
                'calinski_harabasz': None,
                'noise_ratio': 1.0,
                'error': str(e)
            })
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = outdir / "alpha_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Alpha sweep summary saved to {summary_path}")
    
    # Create summary plot
    if results and any(r.get('silhouette') is not None for r in results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot silhouette scores
        valid_results = [r for r in results if r.get('silhouette') is not None]
        if valid_results:
            alphas_plot = [r['alpha'] for r in valid_results]
            sil_scores = [r['silhouette'] for r in valid_results]
            ax1.plot(alphas_plot, sil_scores, 'o-')
            ax1.set_xlabel('Alpha (contact weight)')
            ax1.set_ylabel('Silhouette Score')
            ax1.set_title('Silhouette Score vs Alpha')
            ax1.grid(True, alpha=0.3)
        
        # Plot number of clusters
        alphas_plot = [r['alpha'] for r in results]
        n_clusters = [r['n_clusters'] for r in results]
        ax2.plot(alphas_plot, n_clusters, 'o-', color='orange')
        ax2.set_xlabel('Alpha (contact weight)')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Number of Clusters vs Alpha')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(outdir / "alpha_sweep_plot.png", dpi=200, bbox_inches='tight')
        plt.close()
    
    return results

def blockwise_dim_then_cluster(
    analyzer, 
    contact_umap_dim: int = 20, 
    engineered_pca_dim: int = 10, 
    outdir: Union[str, Path] = 'results'
) -> ClusteringResult:
    """
    Perform block-wise dimensionality reduction then clustering
    
    Args:
        analyzer: ProteinClusterAnalyzer instance
        contact_umap_dim: Dimensions for contact map reduction
        engineered_pca_dim: Dimensions for engineered features reduction
        outdir: Output directory
    
    Returns:
        ClusteringResult object
    """
    outdir = ensure_outdir(outdir)
    
    # Get data
    contacts = np.array(analyzer.contact_maps)
    engineered = np.array(analyzer.feature_vectors) if analyzer.feature_vectors else np.zeros((contacts.shape[0], 0))
    
    validate_inputs(contacts, analyzer.file_names)
    
    logger.info(f"Contacts shape: {contacts.shape}, Engineered shape: {engineered.shape}")
    
    # Reduce contact maps
    logger.info(f"Reducing contact maps to {contact_umap_dim} dimensions")
    if HAVE_UMAP and contacts.shape[0] > contact_umap_dim:
        contact_reducer = umap.UMAP(
            n_components=contact_umap_dim, 
            random_state=42,
            n_neighbors=min(15, contacts.shape[0]-1)
        )
        contact_low = contact_reducer.fit_transform(contacts)
        contact_method = 'UMAP'
    else:
        contact_low = PCA(
            n_components=min(contact_umap_dim, contacts.shape[0]-1, contacts.shape[1])
        ).fit_transform(contacts)
        contact_method = 'PCA'
    
    # Reduce engineered features
    if engineered.size > 0 and engineered.shape[1] > 0:
        logger.info(f"Reducing engineered features to {engineered_pca_dim} dimensions")
        scaler = StandardScaler()
        engineered_scaled = scaler.fit_transform(engineered)
        eng_low = PCA(
            n_components=min(engineered_pca_dim, engineered.shape[0]-1, engineered.shape[1])
        ).fit_transform(engineered_scaled)
    else:
        eng_low = np.zeros((contact_low.shape[0], 0))
        logger.info("No engineered features available")
    
    # Combine features
    if eng_low.shape[1] > 0:
        X = np.hstack([contact_low, eng_low])
        logger.info(f"Combined feature matrix shape: {X.shape} (contact: {contact_low.shape[1]}, eng: {eng_low.shape[1]})")
    else:
        X = contact_low
        logger.info(f"Using only contact features: {X.shape}")
    
    # Perform clustering
    if HAVE_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, len(X)//15),
            min_samples=1
        )
        labels = clusterer.fit_predict(X)
        method = 'HDBSCAN'
    else:
        k_opt, _ = optimal_clusters_search(X)
        labels = KMeans(n_clusters=k_opt, random_state=42).fit_predict(X)
        method = f'KMeans(k={k_opt})'
    
    # Compute metrics
    metrics = compute_clustering_metrics(X, labels)
    
    # Create 2D projection for visualization
    proj = PCA(n_components=2).fit_transform(X)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = sns.color_palette("tab10", len(unique_labels))
    
    for i, lab in enumerate(unique_labels):
        idx = labels == lab
        label_name = "Noise" if lab == -1 else f"Cluster {lab}"
        plt.scatter(
            proj[idx, 0], proj[idx, 1], 
            s=20, alpha=0.7, 
            label=f"{label_name} (n={np.sum(idx)})",
            c=[colors[i]]
        )
    
    title = f"Blockwise Clustering ({contact_method} + PCA)\n"
    if 'silhouette' in metrics:
        title += f"Silhouette: {metrics['silhouette']:.3f}"
    
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    plot_path = outdir / "blockwise_cluster.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved blockwise clustering plot to {plot_path}")
    
    # Save results
    results_df = pd.DataFrame({
        'file': analyzer.file_names, 
        'label': labels,
        'proj_x': proj[:, 0],
        'proj_y': proj[:, 1]
    })
    results_df.to_csv(outdir / "blockwise_labels.csv", index=False)
    
    return ClusteringResult(
        labels=labels,
        projection=proj,
        silhouette_score=metrics.get('silhouette'),
        calinski_harabasz_score=metrics.get('calinski_harabasz'),
        n_clusters=metrics['n_clusters'],
        method=f"blockwise_{method}",
        parameters={
            'contact_dim': contact_umap_dim,
            'engineered_dim': engineered_pca_dim,
            'contact_method': contact_method
        }
    )

def save_analysis_summary(results_dict: Dict, outdir: Path) -> None:
    """Save a comprehensive analysis summary"""
    summary = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'methods_used': list(results_dict.keys()),
        'total_structures': None,
        'results': {}
    }
    
    for method, result in results_dict.items():
        if isinstance(result, ClusteringResult):
            summary['results'][method] = {
                'n_clusters': result.n_clusters,
                'silhouette_score': result.silhouette_score,
                'calinski_harabasz_score': result.calinski_harabasz_score,
                'method': result.method,
                'parameters': result.parameters
            }
            if summary['total_structures'] is None:
                summary['total_structures'] = len(result.labels)
        elif isinstance(result, list):  # Alpha sweep results
            best_result = max(result, key=lambda x: x.get('silhouette', -2) if x.get('silhouette') is not None else -2)
            summary['results'][method] = {
                'best_alpha': best_result.get('alpha'),
                'best_silhouette': best_result.get('silhouette'),
                'tested_alphas': [r.get('alpha') for r in result]
            }
    
    with open(outdir / 'analysis_summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Analysis summary saved to {outdir / 'analysis_summary.json'}")

def main():
    parser = argparse.ArgumentParser(
        description="Stepwise clustering pipeline for antibody-antigen complexes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--structures', type=str, required=True, 
                      help="Directory with .pdb/.cif structure files")
    parser.add_argument('--cache', type=str, default="protein_cache.pkl", 
                      help="Cache file path for processed data")
    parser.add_argument('--outdir', type=str, default="results", 
                      help="Output directory for results")
    parser.add_argument('--metric', type=str, default='jaccard', 
                      choices=['jaccard', 'hamming'], 
                      help="Distance metric for contact maps")
    parser.add_argument('--n_clusters', type=int, default=8, 
                      help="Number of clusters for spectral clustering")
    parser.add_argument('--skip_contact_only', action='store_true',
                      help="Skip contact-only clustering")
    parser.add_argument('--skip_alpha_sweep', action='store_true',
                      help="Skip alpha parameter sweep")
    parser.add_argument('--skip_blockwise', action='store_true',
                      help="Skip blockwise clustering")
    parser.add_argument('--alphas', nargs='+', type=float,
                      help="Custom alpha values for sweep (default: 0.95 0.9 0.8 0.7 0.5 0.3 0.1)")
    
    args = parser.parse_args()
    
    # Setup
    outdir = ensure_outdir(args.outdir)
    logger.info(f"Starting clustering analysis, output directory: {outdir}")
    
    # Initialize analyzer
    try:
        analyzer = ProteinClusterAnalyzer(
            cif_dir=args.structures,
            antibody_chain='A',
            antigen_chains=['B', 'C'],
            dist_cutoff=5.0,
            n_jobs=-1
        )
        analyzer.load_and_process_data(cache_file=args.cache)
        logger.info(f"Loaded {len(analyzer.file_names)} structures")
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        raise
    
    results = {}
    
    # 1. Contact-only clustering
    if not args.skip_contact_only:
        logger.info("=" * 50)
        logger.info("CONTACT-ONLY CLUSTERING")
        logger.info("=" * 50)
        try:
            contact_result = contact_only_pipeline(
                analyzer, 
                metric=args.metric, 
                n_clusters=args.n_clusters, 
                outdir=outdir
            )
            results['contact_only'] = contact_result
            logger.info(f"Contact-only clustering completed. Silhouette: {contact_result.silhouette_score:.3f}")
        except Exception as e:
            logger.error(f"Contact-only clustering failed: {e}")
    
    # 2. Alpha sweep
    if not args.skip_alpha_sweep:
        logger.info("=" * 50)
        logger.info("ALPHA PARAMETER SWEEP")
        logger.info("=" * 50)
        try:
            alphas = args.alphas if args.alphas else None
            sweep_result = alpha_sweep(analyzer, alphas=alphas, outdir=outdir)
            results['alpha_sweep'] = sweep_result
            
            best_alpha = max(sweep_result, key=lambda x: x.get('silhouette', -2) if x.get('silhouette') is not None else -2)
            logger.info(f"Alpha sweep completed. Best alpha: {best_alpha.get('alpha')}, "
                       f"Silhouette: {best_alpha.get('silhouette'):.3f}")
        except Exception as e:
            logger.error(f"Alpha sweep failed: {e}")
    
    # 3. Blockwise clustering
    if not args.skip_blockwise:
        logger.info("=" * 50)
        logger.info("BLOCKWISE CLUSTERING")
        logger.info("=" * 50)
        try:
            blockwise_result = blockwise_dim_then_cluster(analyzer, outdir=outdir)
            results['blockwise'] = blockwise_result
            logger.info(f"Blockwise clustering completed. Silhouette: {blockwise_result.silhouette_score:.3f}")
        except Exception as e:
            logger.error(f"Blockwise clustering failed: {e}")
    
    # Save comprehensive summary
    save_analysis_summary(results, outdir)
    
    logger.info("=" * 50)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Results saved to: {outdir}")
    
    # Print summary
    for method, result in results.items():
        if isinstance(result, ClusteringResult):
            print(f"{method}: {result.n_clusters} clusters, "
                  f"silhouette={result.silhouette_score:.3f if result.silhouette_score else 'N/A'}")
        elif isinstance(result, list) and result:
            best = max(result, key=lambda x: x.get('silhouette', -2) if x.get('silhouette') is not None else -2)
            print(f"{method}: best alpha={best.get('alpha')}, "
                  f"silhouette={best.get('silhouette'):.3f if best.get('silhouette') else 'N/A'}")

if __name__ == '__main__':
    main()
