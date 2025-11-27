"""
Test script to cluster features from both hands combined.
"""
import sys
import numpy as np
from pathlib import Path
from src.cluster import cluster_features, load_features_from_npz
import json


def combine_hands_features(hand0_path, hand1_path):
    """
    Load features from both hands and combine them.
    
    Args:
        hand0_path: Path to hand0 features .npz
        hand1_path: Path to hand1 features .npz
    
    Returns:
        Combined features, frame indices, and valid frames mask
    """
    hand0_data = load_features_from_npz(hand0_path)
    hand1_data = load_features_from_npz(hand1_path)
    
    # Align frames - we need frames where BOTH hands are present
    hand0_frames = set(hand0_data['frame_indices'])
    hand1_frames = set(hand1_data['frame_indices'])
    common_frames = sorted(list(hand0_frames & hand1_frames))
    
    # Create mapping from frame to feature index
    hand0_frame_to_idx = {frame: idx for idx, frame in enumerate(hand0_data['frame_indices'])}
    hand1_frame_to_idx = {frame: idx for idx, frame in enumerate(hand1_data['frame_indices'])}
    
    # Combine features for common frames
    combined_features = []
    combined_frame_indices = []
    
    for frame in common_frames:
        h0_idx = hand0_frame_to_idx[frame]
        h1_idx = hand1_frame_to_idx[frame]
        
        # Concatenate features: [hand0_features, hand1_features]
        combined_feature = np.concatenate([
            hand0_data['features'][h0_idx],
            hand1_data['features'][h1_idx]
        ])
        
        combined_features.append(combined_feature)
        combined_frame_indices.append(frame)
    
    return np.array(combined_features), np.array(combined_frame_indices), len(common_frames)


def main():
    segment_num = 44
    use_smoothed = True  # Set to True to use smoothed features
    
    # Find feature files for both hands
    if use_smoothed:
        hand0_path = Path("data/landmarks") / f"segment_{segment_num:03d}_smoothed_normalized_features_distance_matrix_hand0.npz"
        hand1_path = Path("data/landmarks") / f"segment_{segment_num:03d}_smoothed_normalized_features_distance_matrix_hand1.npz"
    else:
        hand0_path = Path("data/landmarks") / f"segment_{segment_num:03d}_normalized_features_distance_matrix_hand0.npz"
        hand1_path = Path("data/landmarks") / f"segment_{segment_num:03d}_normalized_features_distance_matrix_hand1.npz"
    
    if not hand0_path.exists():
        print(f"Error: Hand0 features not found: {hand0_path}")
        return
    
    if not hand1_path.exists():
        print(f"Error: Hand1 features not found: {hand1_path}")
        return
    
    print(f"Clustering COMBINED features for segment {segment_num}")
    print(f"Hand0: {hand0_path.name}")
    print(f"Hand1: {hand1_path.name}")
    print("="*60)
    
    # Combine features
    print("Combining features from both hands...")
    combined_features, frame_indices, n_common_frames = combine_hands_features(hand0_path, hand1_path)
    
    print(f"Frames with both hands: {n_common_frames}")
    print(f"Combined feature dimension: {combined_features.shape[1]} (hand0: 441 + hand1: 441 = 882)")
    print()
    
    # Check for parameters
    pca_components = 20
    min_cluster_size = 50
    min_samples = None
    
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
            print("Warning: Invalid --min-samples value, using default")
    
    print(f"PCA components: {pca_components}")
    print(f"Min cluster size: {min_cluster_size}")
    print(f"Min samples: {min_samples if min_samples else min_cluster_size} (default)")
    print()
    
    try:
        # Cluster combined features
        cluster_result = cluster_features(
            combined_features,
            pca_components=pca_components,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        
        labels = cluster_result['labels']
        
        # Create output
        clustered_frames = []
        for i, (frame_idx, label) in enumerate(zip(frame_indices, labels)):
            clustered_frames.append({
                'frame': int(frame_idx),
                'cluster': int(label),
                'feature_idx': i
            })
        
        # Save results
        if use_smoothed:
            output_path = hand0_path.parent / f"segment_{segment_num:03d}_smoothed_normalized_features_distance_matrix_both_hands_clustered.json"
        else:
            output_path = hand0_path.parent / f"segment_{segment_num:03d}_normalized_features_distance_matrix_both_hands_clustered.json"
        
        results = {
            'source_files': [str(hand0_path), str(hand1_path)],
            'n_frames': len(combined_features),
            'n_clusters': cluster_result['n_clusters'],
            'n_noise': cluster_result['n_noise'],
            'pca_components': pca_components,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples if min_samples else min_cluster_size,
            'feature_dim': combined_features.shape[1],
            'reduced_dim': cluster_result['X_reduced'].shape[1],
            'clustered_frames': clustered_frames,
            'cluster_distribution': {
                int(cluster): int(np.sum(labels == cluster))
                for cluster in set(labels)
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        labels_path = output_path.with_suffix('.npy')
        np.save(labels_path, labels)
        
        print("="*60)
        print("SUCCESS!")
        print()
        print(f"Total frames (both hands): {results['n_frames']}")
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
        print(f"Results saved to: {output_path}")
        print(f"Labels saved to: {labels_path}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

