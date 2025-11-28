"""
Normalize hand landmarks to a canonical orientation.
This makes hand poses view-invariant for comparison and clustering.
"""
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import json


def _compute_hand_centroid(hand_landmarks):
    """
    Compute centroid (mean position) of a hand's landmarks.
    Returns dict with x/y/z or None if landmarks invalid.
    """
    if hand_landmarks is None or len(hand_landmarks) == 0:
        return None
    pts = np.array(hand_landmarks, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    centroid = np.mean(pts, axis=0)
    return {
        'x': float(centroid[0]),
        'y': float(centroid[1]),
        'z': float(centroid[2])
    }


def normalize_landmarks(landmarks, scale_method='palm'):
    """
    Normalize a single hand's landmarks to canonical orientation.
    
    Args:
        landmarks: Array of shape (21, 3) with hand landmarks
                   Landmark indices:
                   0 = wrist
                   9 = middle finger MCP (base of middle finger)
        scale_method: 'palm' (wrist to MCP) or 'bbox' (bounding box diagonal)
    
    Returns:
        Normalized landmarks array of shape (21, 3)
    """
    pts = np.array(landmarks, dtype=np.float64)
    
    if pts.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {pts.shape}")
    
    # 1. Center: Translate so wrist (landmark 0) is at origin
    wrist = pts[0].copy()
    pts = pts - wrist
    
    # 2. Scale: Normalize by palm size
    if scale_method == 'palm':
        # Use distance from wrist to middle finger MCP (landmark 9)
        scale = np.linalg.norm(pts[9] - pts[0])
    elif scale_method == 'bbox':
        # Use bounding box diagonal
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        scale = np.linalg.norm(bbox_max - bbox_min)
    else:
        raise ValueError(f"Unknown scale_method: {scale_method}")
    
    # Avoid division by zero
    if scale < 1e-6:
        # If scale is too small, return zeros (hand might be invalid)
        return np.zeros_like(pts)
    
    pts = pts / scale
    
    # 3. Rotation normalize using PCA
    # PCA will align the hand to principal axes
    # Note: PCA normalization is used for feature extraction/clustering only.
    # For visualization, we use the original smoothed skeleton data (before normalization).
    pca = PCA(n_components=3)
    pts_normalized = pca.fit_transform(pts)
    
    return pts_normalized


def normalize_landmark_sequence(landmarks_sequence, scale_method='palm'):
    """
    Normalize a sequence of hand landmarks (e.g., from a video segment).
    
    Args:
        landmarks_sequence: List of hand landmarks, each is (21, 3) array
                           or list of (x, y, z) tuples
        scale_method: 'palm' or 'bbox'
    
    Returns:
        List of normalized landmark arrays, each shape (21, 3)
    """
    normalized_sequence = []
    
    for landmarks in landmarks_sequence:
        if isinstance(landmarks, list):
            # Convert list of tuples to numpy array
            landmarks = np.array(landmarks)
        
        normalized = normalize_landmarks(landmarks, scale_method=scale_method)
        normalized_sequence.append(normalized)
    
    return normalized_sequence


def normalize_from_json(json_path, output_path=None, scale_method='palm'):
    """
    Load landmarks from JSON file, normalize them, and save.
    
    Args:
        json_path: Path to JSON file with landmarks (from hand_tracking.py)
        output_path: Path to save normalized landmarks (if None, auto-generates)
        scale_method: 'palm' or 'bbox'
    
    Returns:
        Path to saved normalized landmarks file
    """
    json_path = Path(json_path)
    
    # Load landmarks
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    fps = data.get('fps', None)
    total_frames = data.get('total_frames', len(data['landmarks']))
    
    # Normalize each frame
    normalized_landmarks = []
    
    for frame_data in data['landmarks']:
        frame_idx = frame_data['frame']
        hands = frame_data['hands']
        
        # Normalize each hand in the frame
        normalized_hands = []
        hand_centroids = []
        for hand in hands:
            if len(hand) == 0:
                # No hand detected
                normalized_hands.append([])
                hand_centroids.append(None)
            else:
                normalized_hand = normalize_landmarks(hand, scale_method=scale_method)
                # Convert back to list of tuples for JSON serialization
                normalized_hands.append(normalized_hand.tolist())
                hand_centroids.append(_compute_hand_centroid(normalized_hand))
        
        normalized_landmarks.append({
            'frame': frame_idx,
            'hands': normalized_hands,
            'hand_centroids': hand_centroids
        })
    
    # Save normalized landmarks
    if output_path is None:
        output_path = json_path.parent / f"{json_path.stem}_normalized.json"
    else:
        output_path = Path(output_path)
    
    output_data = {
        'fps': fps,
        'total_frames': total_frames,
        'scale_method': scale_method,
        'landmarks': normalized_landmarks
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return str(output_path)


def normalize_from_numpy(npy_path, output_path=None, scale_method='palm'):
    """
    Load landmarks from numpy file, normalize them, and save.
    
    Args:
        npy_path: Path to .npy file with landmarks
        output_path: Path to save normalized landmarks
        scale_method: 'palm' or 'bbox'
    
    Returns:
        Path to saved normalized landmarks file
    """
    npy_path = Path(npy_path)
    
    # Load landmarks array: shape (num_frames, max_hands, 21, 3)
    landmarks_array = np.load(npy_path)
    
    num_frames, max_hands, num_landmarks, coords = landmarks_array.shape
    
    # Normalize each hand in each frame
    normalized_array = np.full_like(landmarks_array, np.nan)
    
    for frame_idx in range(num_frames):
        for hand_idx in range(max_hands):
            hand_landmarks = landmarks_array[frame_idx, hand_idx]
            
            # Check if hand is present (not all NaN)
            if not np.isnan(hand_landmarks).all():
                normalized_hand = normalize_landmarks(hand_landmarks, scale_method=scale_method)
                normalized_array[frame_idx, hand_idx] = normalized_hand
    
    # Save normalized array
    if output_path is None:
        output_path = npy_path.parent / f"{npy_path.stem}_normalized.npy"
    else:
        output_path = Path(output_path)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, normalized_array)
    
    # Also save metadata if it exists
    metadata_path = npy_path.parent / f"{npy_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata['scale_method'] = scale_method
        metadata['normalized'] = True
        
        output_metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(output_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return str(output_path)

