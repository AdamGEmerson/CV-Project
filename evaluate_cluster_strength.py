"""
Evaluate HDBSCAN cluster strength/quality at various parameter settings.
Tests different combinations of min_cluster_size and min_samples to find optimal parameters.
"""
import sys
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan
from batch_process_all_segments import combine_hands_features
from src.cluster import load_features_from_npz, cluster_features
import time
from itertools import product


def load_all_segment_features(segments_dir, landmarks_dir, use_smoothing=True):
    """
    Load and combine features from all segments.
    
    Returns:
        Tuple of (combined_features_array, all_frame_info_list)
    """
    segment_files = sorted(segments_dir.glob("segment_*.mp4"))
    all_combined_features = []
    all_frame_info = []
    
    for segment_file in segment_files:
        segment_num = int(segment_file.stem.split('_')[1])
        
        # Determine file paths
        if use_smoothing:
            hand0_path = landmarks_dir / f"segment_{segment_num:03d}_smoothed_normalized_features_distance_matrix_hand0.npz"
            hand1_path = landmarks_dir / f"segment_{segment_num:03d}_smoothed_normalized_features_distance_matrix_hand1.npz"
            normalized_json = landmarks_dir / f"segment_{segment_num:03d}_smoothed_normalized.json"
        else:
            hand0_path = landmarks_dir / f"segment_{segment_num:03d}_normalized_features_distance_matrix_hand0.npz"
            hand1_path = landmarks_dir / f"segment_{segment_num:03d}_normalized_features_distance_matrix_hand1.npz"
            normalized_json = landmarks_dir / f"segment_{segment_num:03d}_normalized.json"
        
        if not hand0_path.exists() or not hand1_path.exists():
            continue
        
        # Combine hands features
        combined_features, common_frames = combine_hands_features(
            hand0_path, 
            hand1_path,
            normalized_json_path=str(normalized_json) if normalized_json.exists() else None
        )
        
        if combined_features is None or len(combined_features) == 0:
            continue
        
        # Add to combined dataset
        for i, frame in enumerate(common_frames):
            all_combined_features.append(combined_features[i])
            all_frame_info.append({
                'segment': segment_num,
                'frame': frame,
                'feature_idx': len(all_frame_info)
            })
    
    if len(all_combined_features) == 0:
        return None, None
    
    return np.array(all_combined_features), all_frame_info


def evaluate_clustering_quality(X_reduced, labels):
    """
    Evaluate clustering quality using HDBSCAN's DBVC validity index.
    
    Args:
        X_reduced: Reduced feature matrix (after PCA)
        labels: Cluster labels (-1 for noise)
    
    Returns:
        Dictionary with quality metrics including DBVC
    """
    n_noise = np.sum(labels == -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Calculate DBVC using HDBSCAN's validity index
    try:
        dbvc_score = hdbscan.validity.validity_index(
            X_reduced, 
            labels, 
            metric='euclidean',
            per_cluster_scores=False
        )
    except Exception as e:
        # If calculation fails, return invalid score
        dbvc_score = float('-inf')
    
    return {
        'dbvc_score': float(dbvc_score),
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'noise_ratio': float(n_noise / len(labels)) if len(labels) > 0 else 1.0
    }


def evaluate_hdbscan_parameters(
    features_array,
    pca_components_list=None,
    min_cluster_sizes=None,
    min_samples_list=None,
    random_state=42
):
    """
    Evaluate HDBSCAN cluster strength across different parameter combinations.
    
    Args:
        features_array: Feature array of shape (n_samples, n_features)
        pca_components_list: List of PCA component values to test (if None, uses default [12])
        min_cluster_sizes: List of min_cluster_size values to test (if None, uses defaults)
        min_samples_list: List of min_samples values to test (if None, uses defaults)
        random_state: Random state for PCA
    
    Returns:
        Dictionary with results for each parameter combination
    """
    if pca_components_list is None:
        pca_components_list = [12]
    if min_cluster_sizes is None:
        min_cluster_sizes = [20, 30, 40, 50, 60, 70, 80, 100]
    if min_samples_list is None:
        min_samples_list = [5, 10, 15, 20, 25, 30]
    
    print(f"Evaluating HDBSCAN cluster strength")
    print(f"Total features: {features_array.shape[0]}, Feature dimension: {features_array.shape[1]}")
    print(f"Testing {len(pca_components_list)} PCA component values: {pca_components_list}")
    print(f"Testing {len(min_cluster_sizes)} min_cluster_size values: {min_cluster_sizes}")
    print(f"Testing {len(min_samples_list)} min_samples values: {min_samples_list}")
    print(f"Total combinations: {len(pca_components_list) * len(min_cluster_sizes) * len(min_samples_list)}")
    print("="*60)
    
    results = []
    total_combinations = len(pca_components_list) * len(min_cluster_sizes) * len(min_samples_list)
    current = 0
    
    for pca_components, min_cluster_size, min_samples in product(pca_components_list, min_cluster_sizes, min_samples_list):
        current += 1
        print(f"\n[{current}/{total_combinations}] Testing PCA={pca_components}, min_cluster_size={min_cluster_size}, min_samples={min_samples}...", end=' ')
        
        try:
            # Cluster with these parameters
            cluster_result = cluster_features(
                features_array,
                pca_components=pca_components,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                random_state=random_state
            )
            
            labels = cluster_result['labels']
            X_reduced = cluster_result['X_reduced']
            
            # Evaluate quality
            quality = evaluate_clustering_quality(X_reduced, labels)
            
            result = {
                'pca_components': pca_components,
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                **quality
            }
            results.append(result)
            
            print(f"✓ {quality['n_clusters']} clusters, {quality['n_noise']} noise, "
                  f"DBVC={quality['dbvc_score']:.4f}")
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append({
                'pca_components': pca_components,
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'error': str(e),
                'n_clusters': 0,
                'n_noise': 0,
                'dbvc_score': float('-inf'),
                'noise_ratio': 1.0
            })
    
    return results


def plot_cluster_strength_results(results, output_path=None):
    """
    Plot cluster strength evaluation results.
    Creates separate heatmaps for each PCA component value.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save plot (if None, displays interactively)
    """
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    # Extract data
    pca_components_list = sorted(set(r['pca_components'] for r in valid_results))
    min_cluster_sizes = sorted(set(r['min_cluster_size'] for r in valid_results))
    min_samples_list = sorted(set(r['min_samples'] for r in valid_results))
    
    # Create a grid of subplots: one row per PCA component
    n_pca = len(pca_components_list)
    fig = plt.figure(figsize=(20, 5 * n_pca))
    
    # Find global min/max for consistent color scaling
    all_dbvc_scores = [r['dbvc_score'] for r in valid_results if np.isfinite(r['dbvc_score'])]
    global_dbvc_min = min(all_dbvc_scores) if all_dbvc_scores else 0
    global_dbvc_max = max(all_dbvc_scores) if all_dbvc_scores else 1
    
    for pca_idx, pca_components in enumerate(pca_components_list):
        # Filter results for this PCA component
        pca_results = [r for r in valid_results if r['pca_components'] == pca_components]
        
        # Create matrices for heatmaps
        dbvc_matrix = np.full((len(min_cluster_sizes), len(min_samples_list)), -np.inf)
        n_clusters_matrix = np.zeros((len(min_cluster_sizes), len(min_samples_list)))
        noise_ratio_matrix = np.zeros((len(min_cluster_sizes), len(min_samples_list)))
        
        for r in pca_results:
            i = min_cluster_sizes.index(r['min_cluster_size'])
            j = min_samples_list.index(r['min_samples'])
            dbvc_matrix[i, j] = r['dbvc_score']
            n_clusters_matrix[i, j] = r['n_clusters']
            noise_ratio_matrix[i, j] = r['noise_ratio']
        
        # Replace -inf with min finite value for visualization
        dbvc_matrix_vis = np.where(np.isfinite(dbvc_matrix), 
                                   dbvc_matrix, 
                                   global_dbvc_min)
        
        # Plot 1: DBVC Score
        ax1 = plt.subplot(n_pca, 4, pca_idx * 4 + 1)
        im1 = ax1.imshow(dbvc_matrix_vis, cmap='viridis', aspect='auto', origin='lower',
                         vmin=global_dbvc_min, vmax=global_dbvc_max)
        ax1.set_xlabel('min_samples', fontsize=10, fontweight='bold')
        ax1.set_ylabel('min_cluster_size', fontsize=10, fontweight='bold')
        ax1.set_title(f'DBVC Score (PCA={pca_components})\n(Higher is Better)', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(min_samples_list)))
        ax1.set_xticklabels(min_samples_list)
        ax1.set_yticks(range(len(min_cluster_sizes)))
        ax1.set_yticklabels(min_cluster_sizes)
        plt.colorbar(im1, ax=ax1, label='DBVC')
        
        # Add best marker
        best_idx = np.unravel_index(np.nanargmax(dbvc_matrix_vis), dbvc_matrix_vis.shape)
        ax1.plot(best_idx[1], best_idx[0], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
        
        # Plot 2: Number of Clusters
        ax2 = plt.subplot(n_pca, 4, pca_idx * 4 + 2)
        im2 = ax2.imshow(n_clusters_matrix, cmap='plasma', aspect='auto', origin='lower')
        ax2.set_xlabel('min_samples', fontsize=10, fontweight='bold')
        ax2.set_ylabel('min_cluster_size', fontsize=10, fontweight='bold')
        ax2.set_title(f'Number of Clusters (PCA={pca_components})', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(min_samples_list)))
        ax2.set_xticklabels(min_samples_list)
        ax2.set_yticks(range(len(min_cluster_sizes)))
        ax2.set_yticklabels(min_cluster_sizes)
        plt.colorbar(im2, ax=ax2)
        
        # Plot 3: Noise Ratio
        ax3 = plt.subplot(n_pca, 4, pca_idx * 4 + 3)
        im3 = ax3.imshow(noise_ratio_matrix * 100, cmap='Reds', aspect='auto', origin='lower')
        ax3.set_xlabel('min_samples', fontsize=10, fontweight='bold')
        ax3.set_ylabel('min_cluster_size', fontsize=10, fontweight='bold')
        ax3.set_title(f'Noise Ratio % (PCA={pca_components})\n(Lower is Better)', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(min_samples_list)))
        ax3.set_xticklabels(min_samples_list)
        ax3.set_yticks(range(len(min_cluster_sizes)))
        ax3.set_yticklabels(min_cluster_sizes)
        plt.colorbar(im3, ax=ax3, label='%')
        
        # Plot 4: DBVC with contours
        ax4 = plt.subplot(n_pca, 4, pca_idx * 4 + 4)
        im4 = ax4.imshow(dbvc_matrix_vis, cmap='viridis', aspect='auto', origin='lower',
                         vmin=global_dbvc_min, vmax=global_dbvc_max)
        contour = ax4.contour(dbvc_matrix_vis, levels=8, colors='white', alpha=0.4, linewidths=0.5)
        ax4.clabel(contour, inline=True, fontsize=7, fmt='%.3f')
        ax4.set_xlabel('min_samples', fontsize=10, fontweight='bold')
        ax4.set_ylabel('min_cluster_size', fontsize=10, fontweight='bold')
        ax4.set_title(f'DBVC with Contours (PCA={pca_components})', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(min_samples_list)))
        ax4.set_xticklabels(min_samples_list)
        ax4.set_yticks(range(len(min_cluster_sizes)))
        ax4.set_yticklabels(min_cluster_sizes)
        plt.colorbar(im4, ax=ax4, label='DBVC')
        ax4.plot(best_idx[1], best_idx[0], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def main():
    segments_dir = Path("data/segments")
    landmarks_dir = Path("data/landmarks")
    
    # Parameters
    use_smoothing = True
    pca_components_list = [8, 9, 10, 11, 12, 13, 14, 15]
    min_cluster_sizes = [20, 30, 40, 50, 60, 70, 80, 100]
    min_samples_list = [5, 10, 15, 20, 25, 30]
    
    # Parse command line arguments
    if '--pca' in sys.argv:
        try:
            idx = sys.argv.index('--pca')
            pca_val = int(sys.argv[idx + 1])
            pca_components_list = [pca_val]  # Single value if specified
        except (IndexError, ValueError):
            print("Warning: Invalid --pca value, using default list [8-15]")
    
    if '--no-smoothing' in sys.argv:
        use_smoothing = False
    
    print("="*60)
    print("HDBSCAN CLUSTER STRENGTH EVALUATION")
    print("="*60)
    print(f"Parameters:")
    print(f"  Smoothing: {use_smoothing}")
    print(f"  PCA components: {pca_components_list}")
    print(f"  Min cluster sizes: {min_cluster_sizes}")
    print(f"  Min samples: {min_samples_list}")
    print(f"  Total combinations: {len(pca_components_list) * len(min_cluster_sizes) * len(min_samples_list)}")
    print("="*60)
    
    # Load all features
    print("\nLoading features from all segments...")
    start_time = time.time()
    features_array, frame_info = load_all_segment_features(
        segments_dir, landmarks_dir, use_smoothing=use_smoothing
    )
    
    if features_array is None:
        print("Error: No features found. Please run batch processing first.")
        return
    
    load_time = time.time() - start_time
    print(f"Loaded {len(features_array)} feature vectors in {load_time:.1f}s")
    print(f"Feature dimension: {features_array.shape[1]}")
    
    # Evaluate parameters
    print("\nEvaluating cluster strength across parameter combinations...")
    start_time = time.time()
    results = evaluate_hdbscan_parameters(
        features_array,
        pca_components_list=pca_components_list,
        min_cluster_sizes=min_cluster_sizes,
        min_samples_list=min_samples_list,
        random_state=42
    )
    eval_time = time.time() - start_time
    print(f"\nEvaluation completed in {eval_time:.1f}s")
    
    # Find best parameters
    valid_results = [r for r in results if 'error' not in r and r['n_clusters'] > 0]
    if len(valid_results) > 0:
        best_dbvc = max(valid_results, key=lambda x: x['dbvc_score'])
        
        print("\n" + "="*60)
        print("BEST PARAMETERS:")
        print("="*60)
        print(f"\nBest DBVC Score:")
        print(f"  PCA components={best_dbvc['pca_components']}, min_cluster_size={best_dbvc['min_cluster_size']}, min_samples={best_dbvc['min_samples']}")
        print(f"  DBVC Score: {best_dbvc['dbvc_score']:.4f}")
        print(f"  Clusters: {best_dbvc['n_clusters']}, Noise: {best_dbvc['n_noise']} ({best_dbvc['noise_ratio']*100:.1f}%)")
        
        # Show best for each PCA component
        print(f"\nBest Parameters for Each PCA Component:")
        for pca_val in sorted(set(r['pca_components'] for r in valid_results)):
            pca_results = [r for r in valid_results if r['pca_components'] == pca_val]
            if pca_results:
                best_for_pca = max(pca_results, key=lambda x: x['dbvc_score'])
                print(f"  PCA={pca_val}: min_cluster_size={best_for_pca['min_cluster_size']}, "
                      f"min_samples={best_for_pca['min_samples']}, DBVC={best_for_pca['dbvc_score']:.4f}, "
                      f"Clusters={best_for_pca['n_clusters']}")
        
        # Show top 5 parameter combinations overall
        sorted_results = sorted(valid_results, key=lambda x: x['dbvc_score'], reverse=True)
        print(f"\nTop 5 Parameter Combinations by DBVC Score:")
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. PCA={r['pca_components']}, min_cluster_size={r['min_cluster_size']}, "
                  f"min_samples={r['min_samples']}: DBVC={r['dbvc_score']:.4f}, "
                  f"Clusters={r['n_clusters']}, Noise={r['n_noise']}")
    
    # Save results
    output_json = landmarks_dir / "cluster_strength_evaluation.json"
    with open(output_json, 'w') as f:
        json.dump({
            'parameters': {
                'pca_components_list': pca_components_list,
                'min_cluster_sizes': min_cluster_sizes,
                'min_samples_list': min_samples_list,
                'use_smoothing': use_smoothing,
                'n_features': len(features_array),
                'feature_dim': int(features_array.shape[1])
            },
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to: {output_json}")
    
    # Plot results
    output_plot = landmarks_dir / "cluster_strength_heatmaps.png"
    plot_cluster_strength_results(results, output_path=str(output_plot))
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

