"""
Standalone script to plot cluster strength evaluation results from JSON file.
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


def plot_dbvc_by_pca(results, output_path=None):
    """
    Create a summary plot showing how DBVC varies with PCA components.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save plot (if None, displays interactively)
    """
    valid_results = [r for r in results if 'error' not in r and np.isfinite(r['dbvc_score'])]
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    pca_components_list = sorted(set(r['pca_components'] for r in valid_results))
    
    # Calculate statistics for each PCA component
    pca_stats = {}
    for pca_val in pca_components_list:
        pca_results = [r for r in valid_results if r['pca_components'] == pca_val]
        dbvc_scores = [r['dbvc_score'] for r in pca_results]
        pca_stats[pca_val] = {
            'mean': np.mean(dbvc_scores),
            'std': np.std(dbvc_scores),
            'max': np.max(dbvc_scores),
            'min': np.min(dbvc_scores),
            'median': np.median(dbvc_scores)
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    pca_vals = sorted(pca_stats.keys())
    means = [pca_stats[pca]['mean'] for pca in pca_vals]
    stds = [pca_stats[pca]['std'] for pca in pca_vals]
    maxs = [pca_stats[pca]['max'] for pca in pca_vals]
    mins = [pca_stats[pca]['min'] for pca in pca_vals]
    medians = [pca_stats[pca]['median'] for pca in pca_vals]
    
    # Plot 1: Mean DBVC with error bars
    ax1 = axes[0, 0]
    ax1.errorbar(pca_vals, means, yerr=stds, fmt='o-', linewidth=2, markersize=8, 
                 capsize=5, capthick=2, label='Mean ± Std')
    ax1.set_xlabel('PCA Components', fontsize=12, fontweight='bold')
    ax1.set_ylabel('DBVC Score', fontsize=12, fontweight='bold')
    ax1.set_title('Mean DBVC Score by PCA Components\n(with Standard Deviation)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Max DBVC
    ax2 = axes[0, 1]
    ax2.plot(pca_vals, maxs, 'go-', linewidth=2, markersize=8, label='Maximum')
    ax2.plot(pca_vals, medians, 'bo-', linewidth=2, markersize=8, label='Median')
    ax2.plot(pca_vals, means, 'ro-', linewidth=2, markersize=8, label='Mean')
    ax2.set_xlabel('PCA Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('DBVC Score', fontsize=12, fontweight='bold')
    ax2.set_title('DBVC Statistics by PCA Components', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Range (Max - Min)
    ax3 = axes[1, 0]
    ranges = [maxs[i] - mins[i] for i in range(len(pca_vals))]
    ax3.bar(pca_vals, ranges, alpha=0.7, color='purple', edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('PCA Components', fontsize=12, fontweight='bold')
    ax3.set_ylabel('DBVC Range (Max - Min)', fontsize=12, fontweight='bold')
    ax3.set_title('DBVC Score Range by PCA Components', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Box plot style - showing distribution
    ax4 = axes[1, 1]
    box_data = []
    for pca_val in pca_vals:
        pca_results = [r for r in valid_results if r['pca_components'] == pca_val]
        dbvc_scores = [r['dbvc_score'] for r in pca_results]
        box_data.append(dbvc_scores)
    
    bp = ax4.boxplot(box_data, tick_labels=[str(p) for p in pca_vals], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax4.set_xlabel('PCA Components', fontsize=12, fontweight='bold')
    ax4.set_ylabel('DBVC Score', fontsize=12, fontweight='bold')
    ax4.set_title('DBVC Score Distribution by PCA Components', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Summary plot saved to: {output_path}")
    else:
        plt.show()


def main():
    landmarks_dir = Path("data/landmarks")
    json_path = landmarks_dir / "cluster_strength_evaluation.json"
    
    # Allow custom path
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        print("Please run evaluate_cluster_strength.py first to generate results.")
        return
    
    print(f"Loading results from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    parameters = data.get('parameters', {})
    
    print(f"Loaded {len(results)} result entries")
    print(f"Parameters tested:")
    print(f"  PCA components: {parameters.get('pca_components_list', 'N/A')}")
    print(f"  Min cluster sizes: {parameters.get('min_cluster_sizes', 'N/A')}")
    print(f"  Min samples: {parameters.get('min_samples_list', 'N/A')}")
    
    # Create main heatmap plot
    output_plot = landmarks_dir / "cluster_strength_heatmaps.png"
    print(f"\nGenerating heatmap plots...")
    plot_cluster_strength_results(results, output_path=str(output_plot))
    
    # Create summary plot
    output_summary = landmarks_dir / "cluster_strength_summary.png"
    print(f"\nGenerating summary plot...")
    plot_dbvc_by_pca(results, output_path=str(output_summary))
    
    print("\nDone!")


if __name__ == "__main__":
    main()

