"""
Analyze PCA explained variance ratio across different component values (k=1 to 20).
This helps understand how much variance is captured at each dimensionality level.
"""
import sys
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from batch_process_all_segments import combine_hands_features
from src.cluster import load_features_from_npz
import time


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


def analyze_pca_variance(features_array, k_max=20):
    """
    Analyze PCA explained variance ratio for different numbers of components.
    
    Args:
        features_array: Feature array of shape (n_samples, n_features)
        k_max: Maximum number of PCA components to analyze
    
    Returns:
        Dictionary with variance analysis results
    """
    print(f"Analyzing PCA explained variance for k=1 to {k_max}")
    print(f"Total features: {features_array.shape[0]}, Feature dimension: {features_array.shape[1]}")
    print("="*60)
    
    # Fit PCA with maximum components
    actual_k_max = min(k_max, features_array.shape[1], features_array.shape[0] - 1)
    pca_full = PCA(n_components=actual_k_max, random_state=42)
    pca_full.fit(features_array)
    
    # Get explained variance ratio for all components
    explained_variance_ratio = pca_full.explained_variance_ratio_
    explained_variance = pca_full.explained_variance_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    results = {
        'k_values': list(range(1, actual_k_max + 1)),
        'explained_variance_ratio': [float(x) for x in explained_variance_ratio],
        'explained_variance': [float(x) for x in explained_variance],
        'cumulative_variance_ratio': [float(x) for x in cumulative_variance],
        'total_variance': float(np.sum(pca_full.explained_variance_)),
        'n_samples': int(features_array.shape[0]),
        'n_features': int(features_array.shape[1])
    }
    
    # Print summary for key k values
    print(f"\nPCA Explained Variance Analysis:")
    print(f"{'k':<5} {'Individual %':<15} {'Cumulative %':<15} {'Variance':<15}")
    print("-" * 50)
    for k in range(1, min(k_max + 1, len(explained_variance_ratio) + 1)):
        individual_pct = explained_variance_ratio[k-1] * 100
        cumulative_pct = cumulative_variance[k-1] * 100
        variance = explained_variance[k-1]
        print(f"{k:<5} {individual_pct:<15.2f} {cumulative_pct:<15.2f} {variance:<15.4f}")
    
    # Find k values that capture specific variance thresholds
    thresholds = [0.50, 0.75, 0.90, 0.95, 0.99]
    print(f"\nComponents needed for variance thresholds:")
    for threshold in thresholds:
        k_needed = np.where(cumulative_variance >= threshold)[0]
        if len(k_needed) > 0:
            k_val = k_needed[0] + 1  # +1 because k is 1-indexed
            actual_variance = cumulative_variance[k_val - 1]
            print(f"  {threshold*100:.0f}% variance: k={k_val} (actual: {actual_variance*100:.2f}%)")
        else:
            print(f"  {threshold*100:.0f}% variance: k>{actual_k_max} (not achievable)")
    
    return results


def plot_variance_results(results, output_path=None):
    """
    Plot PCA variance analysis results using matplotlib.
    
    Args:
        results: Dictionary with variance analysis results
        output_path: Path to save plot (if None, displays interactively)
    """
    k_values = results['k_values']
    individual_var = results['explained_variance_ratio']
    cumulative_var = results['cumulative_variance_ratio']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Individual and cumulative explained variance ratio
    ax1_twin = ax1.twinx()
    
    # Individual variance (bars)
    bars = ax1.bar(k_values, [v * 100 for v in individual_var], 
                   alpha=0.6, color='steelblue', label='Individual %', width=0.8)
    ax1.set_xlabel('PCA Components (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Individual Explained Variance (%)', fontsize=12, color='steelblue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(k_values)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_xlim([0.5, max(k_values) + 0.5])
    
    # Cumulative variance (line)
    line = ax1_twin.plot(k_values, [v * 100 for v in cumulative_var], 
                         'ro-', linewidth=2.5, markersize=8, label='Cumulative %', markeredgecolor='darkred')
    ax1_twin.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, color='red', fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.set_ylim([0, 105])
    
    # Add threshold lines
    for threshold in [50, 75, 90, 95, 99]:
        ax1_twin.axhline(y=threshold, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    ax1.set_title('PCA Explained Variance Ratio\n(Individual bars + Cumulative line)', 
                  fontsize=14, fontweight='bold', pad=15)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Plot 2: Cumulative variance with threshold annotations
    ax2.plot(k_values, [v * 100 for v in cumulative_var], 
             'o-', linewidth=2.5, markersize=8, color='darkgreen', markeredgecolor='darkgreen', markerfacecolor='lightgreen')
    ax2.set_xlabel('PCA Components (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Explained Variance with Thresholds', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(k_values)
    ax2.set_ylim([0, 105])
    ax2.set_xlim([0.5, max(k_values) + 0.5])
    
    # Add threshold annotations
    thresholds = [50, 75, 90, 95, 99]
    threshold_colors = ['orange', 'purple', 'brown', 'pink', 'cyan']
    for threshold, color in zip(thresholds, threshold_colors):
        k_needed = None
        for i, cum_var in enumerate(cumulative_var):
            if cum_var * 100 >= threshold:
                k_needed = k_values[i]
                break
        if k_needed:
            ax2.axhline(y=threshold, color=color, linestyle='--', alpha=0.6, linewidth=1.5, label=f'{threshold}% threshold')
            ax2.axvline(x=k_needed, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
            ax2.plot(k_needed, threshold, 'o', color=color, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
            ax2.annotate(f'k={k_needed}', 
                        xy=(k_needed, threshold), 
                        xytext=(k_needed + max(k_values) * 0.05, threshold + 3),
                        fontsize=10, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
    
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()
    
    # Also create a simpler cumulative-only plot
    fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    ax3.plot(k_values, [v * 100 for v in cumulative_var], 
             'o-', linewidth=3, markersize=10, color='#2563eb', 
             markeredgecolor='#1e40af', markerfacecolor='#60a5fa', markeredgewidth=2)
    ax3.fill_between(k_values, [v * 100 for v in cumulative_var], alpha=0.2, color='#2563eb')
    ax3.set_xlabel('PCA Components (k)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cumulative Explained Variance (%)', fontsize=14, fontweight='bold')
    ax3.set_title('PCA Cumulative Explained Variance', fontsize=16, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xticks(k_values)
    ax3.set_ylim([0, 105])
    ax3.set_xlim([0.5, max(k_values) + 0.5])
    
    # Add threshold lines
    for threshold in [50, 75, 90, 95, 99]:
        ax3.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        # Find k for this threshold
        for i, cum_var in enumerate(cumulative_var):
            if cum_var * 100 >= threshold:
                k_val = k_values[i]
                ax3.axvline(x=k_val, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                break
    
    plt.tight_layout()
    
    if output_path:
        simple_plot_path = str(output_path).replace('.png', '_simple.png')
        plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Simple plot saved to: {simple_plot_path}")
    else:
        plt.show()


def main():
    segments_dir = Path("data/segments")
    landmarks_dir = Path("data/landmarks")
    
    # Parameters
    use_smoothing = True
    k_max = 20
    
    # Parse command line arguments
    if '--k-max' in sys.argv:
        try:
            idx = sys.argv.index('--k-max')
            k_max = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --k-max value, using default 20")
    
    if '--no-smoothing' in sys.argv:
        use_smoothing = False
    
    print("="*60)
    print("PCA EXPLAINED VARIANCE ANALYSIS")
    print("="*60)
    print(f"Parameters:")
    print(f"  Smoothing: {use_smoothing}")
    print(f"  Max PCA components: k={k_max}")
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
    
    # Analyze PCA variance
    print("\nFitting PCA and analyzing variance...")
    start_time = time.time()
    results = analyze_pca_variance(features_array, k_max=k_max)
    analysis_time = time.time() - start_time
    print(f"\nAnalysis completed in {analysis_time:.1f}s")
    
    # Save results
    output_json = landmarks_dir / "pca_variance_analysis.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_json}")
    
    # Plot results
    output_plot = landmarks_dir / "pca_variance_plot.png"
    plot_variance_results(results, output_path=str(output_plot))
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

