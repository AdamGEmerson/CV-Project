"""
Cluster hand pose features to identify similar gestures and movements.
"""
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
from pathlib import Path
import json


def cluster_features(X, pca_components=20, min_cluster_size=80, min_samples=20):
    """
    Cluster feature vectors using HDBSCAN with PCA dimensionality reduction.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        pca_components: Number of PCA components for dimensionality reduction
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples in neighborhood (if None, uses min_cluster_size)
    
    Returns:
        Dictionary with:
        - labels: Cluster labels (-1 for noise/outliers)
        - clusterer: Fitted HDBSCAN clusterer
        - pca: Fitted PCA transformer
        - X_reduced: Reduced feature matrix
    """
    if len(X) == 0:
        raise ValueError("Empty feature matrix")
    
    # Apply PCA for dimensionality reduction
    if X.shape[1] > pca_components:
        pca = PCA(n_components=pca_components)
        X_reduced = pca.fit_transform(X)
    else:
        # If features already have fewer dimensions, skip PCA
        pca = None
        X_reduced = X
    
    # Set min_samples if not provided
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Cluster using HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    
    labels = clusterer.fit_predict(X_reduced)
    
    return {
        'labels': labels,
        'clusterer': clusterer,
        'pca': pca,
        'X_reduced': X_reduced,
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'n_noise': int(np.sum(labels == -1))
    }


def load_features_from_npz(npz_path):
    """
    Load features from a .npz file.
    
    Args:
        npz_path: Path to .npz file
    
    Returns:
        Dictionary with features, frame_indices, and valid_frames
    """
    npz_path = Path(npz_path)
    
    data = np.load(npz_path)
    
    return {
        'features': data['features'],
        'frame_indices': data['frame_indices'],
        'valid_frames': data['valid_frames']
    }


def cluster_segment_features(npz_path, pca_components=20, min_cluster_size=50, min_samples=None, output_path=None):
    """
    Cluster features from a single segment.
    
    Args:
        npz_path: Path to features .npz file
        pca_components: Number of PCA components
        min_cluster_size: Minimum cluster size
        output_path: Path to save clustering results (if None, auto-generates)
    
    Returns:
        Dictionary with clustering results and metadata
    """
    npz_path = Path(npz_path)
    
    # Load features
    feature_data = load_features_from_npz(npz_path)
    features = feature_data['features']
    frame_indices = feature_data['frame_indices']
    
    if len(features) == 0:
        raise ValueError(f"No features found in {npz_path}")
    
    # Cluster features
    cluster_result = cluster_features(
        features,
        pca_components=pca_components,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    
    labels = cluster_result['labels']
    
    # Create output with frame information
    clustered_frames = []
    for i, (frame_idx, label) in enumerate(zip(frame_indices, labels)):
        clustered_frames.append({
            'frame': int(frame_idx),
            'cluster': int(label),
            'feature_idx': i
        })
    
    # Load metadata if available
    metadata_path = npz_path.with_suffix('.json')
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Create results dictionary
    results = {
        'source_file': str(npz_path),
        'n_frames': len(features),
        'n_clusters': cluster_result['n_clusters'],
        'n_noise': cluster_result['n_noise'],
        'pca_components': pca_components,
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples if min_samples else min_cluster_size,
        'feature_dim': features.shape[1],
        'reduced_dim': cluster_result['X_reduced'].shape[1],
        'clustered_frames': clustered_frames,
        'cluster_distribution': {
            int(cluster): int(np.sum(labels == cluster))
            for cluster in set(labels)
        },
        'metadata': metadata
    }
    
    # Save results
    if output_path is None:
        output_path = npz_path.parent / f"{npz_path.stem}_clustered.json"
    else:
        output_path = Path(output_path)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON (excluding non-serializable objects)
    save_results = results.copy()
    save_results.pop('clusterer', None)
    save_results.pop('pca', None)
    
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    # Also save labels as numpy array for easy access
    labels_path = output_path.with_suffix('.npy')
    np.save(labels_path, labels)
    
    results['output_path'] = str(output_path)
    results['labels_path'] = str(labels_path)
    
    return results


def cluster_multiple_segments(npz_paths, pca_components=20, min_cluster_size=50, min_samples=None, output_path=None):
    """
    Cluster features from multiple segments together.
    
    Args:
        npz_paths: List of paths to feature .npz files
        pca_components: Number of PCA components
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples in neighborhood (if None, uses min_cluster_size)
        output_path: Path to save results
    
    Returns:
        Dictionary with clustering results
    """
    # Load and combine all features
    all_features = []
    all_frame_info = []  # Track which segment/frame each feature came from
    
    for npz_path in npz_paths:
        npz_path = Path(npz_path)
        feature_data = load_features_from_npz(npz_path)
        
        segment_name = npz_path.stem
        for i, (feature, frame_idx) in enumerate(zip(
            feature_data['features'],
            feature_data['frame_indices']
        )):
            all_features.append(feature)
            all_frame_info.append({
                'segment': segment_name,
                'frame': int(frame_idx),
                'feature_idx': len(all_features) - 1
            })
    
    if len(all_features) == 0:
        raise ValueError("No features found in any segment")
    
    all_features = np.array(all_features)
    
    # Cluster all features together
    cluster_result = cluster_features(
        all_features,
        pca_components=pca_components,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    
    labels = cluster_result['labels']
    
    # Add cluster labels to frame info
    for i, label in enumerate(labels):
        all_frame_info[i]['cluster'] = int(label)
    
    # Create results
    results = {
        'n_segments': len(npz_paths),
        'n_frames': len(all_features),
        'n_clusters': cluster_result['n_clusters'],
        'n_noise': cluster_result['n_noise'],
        'pca_components': pca_components,
        'min_cluster_size': min_cluster_size,
        'feature_dim': all_features.shape[1],
        'reduced_dim': cluster_result['X_reduced'].shape[1],
        'clustered_frames': all_frame_info,
        'cluster_distribution': {
            int(cluster): int(np.sum(labels == cluster))
            for cluster in set(labels)
        }
    }
    
    # Save results
    if output_path is None:
        output_path = Path(npz_paths[0]).parent / "multi_segment_clustered.json"
    else:
        output_path = Path(output_path)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    save_results = results.copy()
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    # Save labels
    labels_path = output_path.with_suffix('.npy')
    np.save(labels_path, labels)
    
    results['output_path'] = str(output_path)
    results['labels_path'] = str(labels_path)
    
    return results

