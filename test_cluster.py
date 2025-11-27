"""
Test script to cluster features from a segment.
"""
import sys
from pathlib import Path
from src.cluster import cluster_segment_features


def main():
    segment_num = 44
    
    # Find features file
    features_path = Path("data/landmarks") / f"segment_{segment_num:03d}_normalized_features_distance_matrix_hand0.npz"
    
    if not features_path.exists():
        print(f"Error: Features file not found: {features_path}")
        print("Please extract features first using test_features.py")
        return
    
    print(f"Clustering features for segment {segment_num}")
    print(f"Input: {features_path}")
    print("="*60)
    
    # Check for parameters
    pca_components = 20
    min_cluster_size = 50
    min_samples = None  # Defaults to min_cluster_size
    
    if '--pca' in sys.argv:
        try:
            idx = sys.argv.index('--pca')
            pca_components = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --pca value, using default 20")
    
    if '--min-size' in sys.argv:
        try:
            idx = sys.argv.index('--min-size')
            min_cluster_size = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --min-size value, using default 50")
    
    if '--min-samples' in sys.argv:
        try:
            idx = sys.argv.index('--min-samples')
            min_samples = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --min-samples value, using default (same as min-size)")
    
    print(f"PCA components: {pca_components}")
    print(f"Min cluster size: {min_cluster_size}")
    print(f"Min samples: {min_samples if min_samples else min_cluster_size} (default)")
    print()
    
    try:
        results = cluster_segment_features(
            features_path,
            pca_components=pca_components,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            output_path=None  # Auto-generate
        )
        
        print("="*60)
        print("SUCCESS!")
        print()
        print(f"Total frames: {results['n_frames']}")
        print(f"Number of clusters: {results['n_clusters']}")
        print(f"Noise/outliers: {results['n_noise']}")
        print(f"Feature dimension: {results['feature_dim']}")
        print(f"Reduced dimension: {results['reduced_dim']}")
        print()
        print("Cluster distribution:")
        for cluster_id, count in sorted(results['cluster_distribution'].items()):
            if cluster_id == -1:
                print(f"  Noise: {count} frames")
            else:
                print(f"  Cluster {cluster_id}: {count} frames")
        print()
        print(f"Results saved to: {results['output_path']}")
        print(f"Labels saved to: {results['labels_path']}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

