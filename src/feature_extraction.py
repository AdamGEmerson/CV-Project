"""
Extract features from normalized hand landmarks for clustering.
"""
import numpy as np
import json
from pathlib import Path


def frame_to_feature(norm_pts, method='distance_matrix'):
    """
    Convert normalized landmarks to a feature vector.
    
    Args:
        norm_pts: Normalized landmarks array of shape (21, 3)
        method: 'distance_matrix' (441-dim) or 'flatten' (63-dim)
    
    Returns:
        Feature vector (numpy array)
    """
    if method == 'distance_matrix':
        # Distance matrix: (21, 21) → flatten → 441-dim vector
        # This gives view-invariance "for free"
        D = np.linalg.norm(norm_pts[:, None, :] - norm_pts[None, :, :], axis=-1)
        return D.flatten()
    
    elif method == 'flatten':
        # Simple flatten: (21, 3) → 63-dim vector
        return norm_pts.flatten()
    
    elif method == 'distance_matrix_upper_triangle':
        # Only upper triangle (excluding diagonal) to reduce redundancy
        # (21*20)/2 = 210 dimensions
        D = np.linalg.norm(norm_pts[:, None, :] - norm_pts[None, :, :], axis=-1)
        # Get upper triangle (excluding diagonal)
        upper_triangle = D[np.triu_indices_from(D, k=1)]
        return upper_triangle
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'distance_matrix', 'flatten', or 'distance_matrix_upper_triangle'")


def extract_features_from_landmarks(landmarks_data, method='distance_matrix', hand_idx=0):
    """
    Extract features from a sequence of normalized landmarks.
    
    Args:
        landmarks_data: List of frame data, each with 'frame' and 'hands' keys
                       Hands is a list of normalized landmark arrays
        method: Feature extraction method
        hand_idx: Which hand to use (0 = first hand, 1 = second hand)
    
    Returns:
        features: Array of shape (num_frames, feature_dim)
        frame_indices: List of frame indices
        valid_frames: Boolean mask indicating which frames have valid hands
    """
    features = []
    frame_indices = []
    valid_frames = []
    
    for frame_data in landmarks_data:
        frame_idx = frame_data['frame']
        hands = frame_data['hands']
        
        # Check if requested hand exists
        if hand_idx < len(hands) and len(hands[hand_idx]) > 0:
            hand_landmarks = np.array(hands[hand_idx])
            feature = frame_to_feature(hand_landmarks, method=method)
            features.append(feature)
            frame_indices.append(frame_idx)
            valid_frames.append(True)
        else:
            valid_frames.append(False)
    
    if len(features) == 0:
        return np.array([]), [], np.array(valid_frames)
    
    features_array = np.array(features)
    return features_array, frame_indices, np.array(valid_frames)


def extract_features_from_json(json_path, method='distance_matrix', hand_idx=0, output_path=None):
    """
    Load normalized landmarks from JSON and extract features.
    
    Args:
        json_path: Path to normalized landmarks JSON file
        method: Feature extraction method
        hand_idx: Which hand to use (0 or 1)
        output_path: Path to save features (if None, auto-generates)
    
    Returns:
        Dictionary with:
        - features: Array of shape (num_frames, feature_dim)
        - frame_indices: List of frame indices
        - valid_frames: Boolean mask
        - metadata: Dictionary with feature info
    """
    json_path = Path(json_path)
    
    # Load normalized landmarks
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    fps = data.get('fps', None)
    total_frames = data.get('total_frames', len(data['landmarks']))
    scale_method = data.get('scale_method', 'palm')
    
    # Extract features
    features, frame_indices, valid_frames = extract_features_from_landmarks(
        data['landmarks'],
        method=method,
        hand_idx=hand_idx
    )
    
    # Create metadata
    metadata = {
        'source_file': str(json_path),
        'fps': fps,
        'total_frames': total_frames,
        'frames_with_features': len(features),
        'scale_method': scale_method,
        'feature_method': method,
        'hand_idx': hand_idx,
        'feature_dim': features.shape[1] if len(features) > 0 else 0,
        'frame_indices': frame_indices
    }
    
    # Save features and metadata
    if output_path is None:
        output_path = json_path.parent / f"{json_path.stem}_features_{method}.npz"
    else:
        output_path = Path(output_path)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as compressed numpy file
    np.savez_compressed(
        output_path,
        features=features,
        frame_indices=np.array(frame_indices),
        valid_frames=valid_frames
    )
    
    # Save metadata as JSON
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'features': features,
        'frame_indices': frame_indices,
        'valid_frames': valid_frames,
        'metadata': metadata,
        'output_path': str(output_path),
        'metadata_path': str(metadata_path)
    }


def extract_all_hands_features(json_path, method='distance_matrix', output_path=None):
    """
    Extract features for all hands in the sequence.
    
    Args:
        json_path: Path to normalized landmarks JSON file
        method: Feature extraction method
        output_path: Base path for output files
    
    Returns:
        Dictionary with features for each hand
    """
    json_path = Path(json_path)
    
    if output_path is None:
        output_path = json_path.parent / f"{json_path.stem}_features_{method}"
    else:
        output_path = Path(output_path)
    
    # Extract features for hand 0
    hand0_result = extract_features_from_json(
        json_path,
        method=method,
        hand_idx=0,
        output_path=f"{output_path}_hand0.npz"
    )
    
    # Extract features for hand 1 (if present)
    hand1_result = extract_features_from_json(
        json_path,
        method=method,
        hand_idx=1,
        output_path=f"{output_path}_hand1.npz"
    )
    
    return {
        'hand0': hand0_result,
        'hand1': hand1_result
    }

