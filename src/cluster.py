"""
Cluster hand pose features to identify similar gestures and movements.
"""
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
from pathlib import Path
import json


def load_frame_landmarks(source_json_path):
    """
    Load per-frame hand landmarks (all 21 points per hand) from a landmarks JSON file.
    This loads the ORIGINAL skeleton data (smoothed but not PCA-normalized).
    Returns dict mapping frame index -> list of hand landmarks (each hand is a list of 21 [x, y, z] points).
    """
    if not source_json_path:
        return {}
    source_path = Path(source_json_path)
    if not source_path.exists():
        return {}
    try:
        with open(source_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return {}
    
    lookup = {}
    for entry in data.get('landmarks', []):
        frame_idx = int(entry.get('frame', 0))
        hands = entry.get('hands', [])
        # Store full landmarks: each hand is a list of 21 [x, y, z] coordinates
        # These are the original MediaPipe coordinates (smoothed but not PCA-normalized)
        lookup[frame_idx] = hands
    return lookup


def summarize_cluster_landmarks(clustered_frames):
    """
    Aggregate all 21 landmarks per hand for each cluster across frames.
    Returns dict {cluster_id: {'hands': [ {hand_index, landmarks: [[x,y,z], ...], count}, ... ]}}
    where landmarks is a list of 21 [x, y, z] coordinates.
    """
    cluster_summaries = {}
    
    for entry in clustered_frames:
        landmarks_list = entry.get('hand_landmarks') or []
        if not landmarks_list:
            continue
        cluster_id = entry.get('cluster')
        if cluster_id is None:
            continue
        
        # Initialize cluster summary if needed
        if cluster_id not in cluster_summaries:
            cluster_summaries[cluster_id] = {}
        
        # Process each hand in this frame
        for hand_idx, hand_landmarks in enumerate(landmarks_list):
            if not hand_landmarks or len(hand_landmarks) == 0:
                continue
            
            # Initialize hand accumulator if needed
            if hand_idx not in cluster_summaries[cluster_id]:
                # Initialize accumulator for 21 landmarks
                cluster_summaries[cluster_id][hand_idx] = {
                    'landmark_sums': [[0.0, 0.0, 0.0] for _ in range(21)],
                    'count': 0
                }
            
            accum = cluster_summaries[cluster_id][hand_idx]
            
            # Sum up each landmark position
            for landmark_idx in range(min(21, len(hand_landmarks))):
                landmark = hand_landmarks[landmark_idx]
                if len(landmark) >= 3:
                    accum['landmark_sums'][landmark_idx][0] += float(landmark[0])
                    accum['landmark_sums'][landmark_idx][1] += float(landmark[1])
                    accum['landmark_sums'][landmark_idx][2] += float(landmark[2])
            
            accum['count'] += 1
    
    # Compute averages
    result = {}
    for cluster_id, hand_dict in cluster_summaries.items():
        hands_output = []
        for hand_idx in sorted(hand_dict.keys()):
            accum = hand_dict[hand_idx]
            if accum['count'] == 0:
                continue
            
            # Average each landmark
            avg_landmarks = []
            for landmark_sum in accum['landmark_sums']:
                avg_landmarks.append([
                    landmark_sum[0] / accum['count'],
                    landmark_sum[1] / accum['count'],
                    landmark_sum[2] / accum['count']
                ])
            
            hands_output.append({
                'hand_index': hand_idx,
                'landmarks': avg_landmarks,  # List of 21 [x, y, z] coordinates
                'count': accum['count']
            })
        
        if hands_output:
            result[cluster_id] = {'hands': hands_output}
    
    return result


def _extract_xyz(embedding_row):
    """
    Return the first three dimensions of an embedding vector, padding with 0.0 when needed.
    """
    if embedding_row is None:
        return 0.0, 0.0, 0.0
    x = float(embedding_row[0]) if len(embedding_row) > 0 else 0.0
    y = float(embedding_row[1]) if len(embedding_row) > 1 else 0.0
    z = float(embedding_row[2]) if len(embedding_row) > 2 else 0.0
    return x, y, z


def cluster_features(X, pca_components=11, min_cluster_size=40, min_samples=15, random_state=None):
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
        pca = PCA(n_components=pca_components, random_state=random_state)
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
    Load features from a .npz file (supports both old format and new consolidated format).
    
    Args:
        npz_path: Path to .npz file
    
    Returns:
        Dictionary with features, frame_indices, valid_frames, and metadata
    """
    npz_path = Path(npz_path)
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Check if this is a consolidated file (has combined_features)
    if 'combined_features' in data:
        # New consolidated format
        return {
            'features': data['combined_features'],
            'frame_indices': data['common_frames'],
            'valid_frames': np.ones(len(data['common_frames']), dtype=bool),
            'hand0_features': data['hand0_features'],
            'hand1_features': data['hand1_features'],
            'hand0_frame_indices': data['hand0_frame_indices'],
            'hand1_frame_indices': data['hand1_frame_indices'],
            'metadata': {
                'source_file': str(data['source_file'][0]) if 'source_file' in data else None,
                'fps': float(data['fps'][0]) if 'fps' in data and data['fps'][0] is not None else None,
                'total_frames': int(data['total_frames'][0]) if 'total_frames' in data else 0,
                'scale_method': str(data['scale_method'][0]) if 'scale_method' in data else None,
                'feature_method': str(data['feature_method'][0]) if 'feature_method' in data else None,
                'feature_dim': int(data['feature_dim'][0]) if 'feature_dim' in data else 0
            }
        }
    else:
        # Old format (individual hand files)
        result = {
            'features': data['features'],
            'frame_indices': data['frame_indices'],
            'valid_frames': data['valid_frames']
        }
        
        # Try to load metadata from NPZ if available
        if 'source_file' in data:
            result['metadata'] = {
                'source_file': str(data['source_file'][0]) if len(data['source_file']) > 0 else None,
                'fps': float(data['fps'][0]) if 'fps' in data and len(data['fps']) > 0 and data['fps'][0] is not None else None,
                'total_frames': int(data['total_frames'][0]) if 'total_frames' in data and len(data['total_frames']) > 0 else 0,
                'scale_method': str(data['scale_method'][0]) if 'scale_method' in data and len(data['scale_method']) > 0 else None,
                'feature_method': str(data['feature_method'][0]) if 'feature_method' in data and len(data['feature_method']) > 0 else None,
                'feature_dim': int(data['feature_dim'][0]) if 'feature_dim' in data and len(data['feature_dim']) > 0 else 0
            }
        else:
            # Fallback: try to load from JSON metadata file (backward compatibility)
            metadata_path = npz_path.with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    result['metadata'] = json.load(f)
            else:
                result['metadata'] = {}
        
        return result


def cluster_segment_features(npz_path, pca_components=11, min_cluster_size=40, min_samples=None, output_path=None):
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
    
    # Load per-frame original skeleton landmarks (smoothed but NOT PCA-normalized) if available
    frame_landmarks_lookup = {}
    metadata_path = npz_path.with_suffix('.json')
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        source_file = metadata.get('source_file')
        # Note: source_file points to normalized JSON, but we want smoothed JSON
        # For single-segment clustering, this may still use normalized data
        # Main batch processing uses smoothed data directly
        frame_landmarks_lookup = load_frame_landmarks(source_file)
    
    # Create output with frame information
    clustered_frames = []
    for i, (frame_idx, label) in enumerate(zip(frame_indices, labels)):
        embedding_row = cluster_result['X_reduced'][i]
        x, y, z = _extract_xyz(embedding_row)
        landmarks = frame_landmarks_lookup.get(int(frame_idx), [])
        clustered_frames.append({
            'frame': int(frame_idx),
            'cluster': int(label),
            'feature_idx': i,
            'x': x,
            'y': y,
            'z': z,
            'hand_landmarks': landmarks
        })
    
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
        'metadata': metadata,
        'cluster_landmarks': summarize_cluster_landmarks(clustered_frames)
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


def cluster_multiple_segments(npz_paths, pca_components=11, min_cluster_size=40, min_samples=None, output_path=None):
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
    
    landmarks_cache = {}
    
    for npz_path in npz_paths:
        npz_path = Path(npz_path)
        feature_data = load_features_from_npz(npz_path)
        metadata_path = npz_path.with_suffix('.json')
        source_file = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            source_file = metadata.get('source_file')
        frame_landmarks_lookup = {}
        if source_file:
            if source_file in landmarks_cache:
                frame_landmarks_lookup = landmarks_cache[source_file]
            else:
                frame_landmarks_lookup = load_frame_landmarks(source_file)
                landmarks_cache[source_file] = frame_landmarks_lookup
        
        segment_name = npz_path.stem
        for i, (feature, frame_idx) in enumerate(zip(
            feature_data['features'],
            feature_data['frame_indices']
        )):
            all_features.append(feature)
            all_frame_info.append({
                'segment': segment_name,
                'frame': int(frame_idx),
                'feature_idx': len(all_features) - 1,
                'hand_landmarks': frame_landmarks_lookup.get(int(frame_idx), [])
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
        embedding_row = cluster_result['X_reduced'][i]
        x, y, z = _extract_xyz(embedding_row)
        all_frame_info[i]['cluster'] = int(label)
        all_frame_info[i]['x'] = x
        all_frame_info[i]['y'] = y
        all_frame_info[i]['z'] = z
    
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
        },
        'cluster_landmarks': summarize_cluster_landmarks(all_frame_info)
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

