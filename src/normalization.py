"""
Normalize hand landmarks to a canonical orientation.
This makes hand poses view-invariant for comparison and clustering.
"""
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import json


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


def normalize_hands_relative(left_hand, right_hand, scale_method='palm'):
    """
    Normalize hands with left wrist as global anchor point, preserving relative orientation.
    - Left hand: normalized to origin (wrist at 0,0,0)
    - Right hand: normalized relative to left wrist position
    - Both hands use the same scale and rotation to preserve relative orientation
    
    Args:
        left_hand: Array of shape (21, 3) with left hand landmarks, or None
        right_hand: Array of shape (21, 3) with right hand landmarks, or None
        scale_method: 'palm' or 'bbox'
    
    Returns:
        Tuple of (normalized_left_hand, normalized_right_hand)
        Each is shape (21, 3) or None if hand not present
    """
    normalized_left = None
    normalized_right = None
    
    # If both hands are present, normalize together to preserve relative orientation
    if left_hand is not None and len(left_hand) > 0 and right_hand is not None and len(right_hand) > 0:
        left_pts = np.array(left_hand, dtype=np.float64)
        right_pts = np.array(right_hand, dtype=np.float64)
        
        if left_pts.shape == (21, 3) and right_pts.shape == (21, 3):
            # Get left wrist position (anchor point)
            left_wrist_pos = left_pts[0].copy()
            
            # Center left hand at origin
            left_centered = left_pts - left_wrist_pos
            
            # Translate right hand relative to left wrist
            right_relative = right_pts - left_wrist_pos
            
            # Calculate scale from left hand (to preserve relative sizes)
            if scale_method == 'palm':
                scale = np.linalg.norm(left_centered[9] - left_centered[0])  # Wrist to middle MCP
            elif scale_method == 'bbox':
                bbox_min = left_centered.min(axis=0)
                bbox_max = left_centered.max(axis=0)
                scale = np.linalg.norm(bbox_max - bbox_min)
            else:
                raise ValueError(f"Unknown scale_method: {scale_method}")
            
            if scale < 1e-6:
                # Fallback to independent normalization
                normalized_left = normalize_landmarks(left_pts, scale_method=scale_method)
                normalized_right = normalize_landmarks(right_pts - left_wrist_pos, scale_method=scale_method)
            else:
                # Apply same scale to both hands
                left_scaled = left_centered / scale
                right_scaled = right_relative / scale
                
                # Apply PCA rotation to left hand, then apply same rotation to right hand
                # This preserves relative orientation
                pca = PCA(n_components=3)
                left_normalized = pca.fit_transform(left_scaled)
                # Apply the same PCA transformation to right hand
                right_normalized = pca.transform(right_scaled)
                
                normalized_left = left_normalized
                normalized_right = right_normalized
    
    # If only left hand is present
    elif left_hand is not None and len(left_hand) > 0:
        left_pts = np.array(left_hand, dtype=np.float64)
        if left_pts.shape == (21, 3):
            normalized_left = normalize_landmarks(left_pts, scale_method=scale_method)
    
    # If only right hand is present
    elif right_hand is not None and len(right_hand) > 0:
        right_pts = np.array(right_hand, dtype=np.float64)
        if right_pts.shape == (21, 3):
            normalized_right = normalize_landmarks(right_pts, scale_method=scale_method)
    
    return normalized_left, normalized_right


def normalize_from_json(json_path, output_path=None, scale_method='palm'):
    """
    Load landmarks from JSON file, normalize them, and save.
    Uses new strategy: left wrist as anchor, right hand normalized relative to left.
    
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
        hand_labels = frame_data.get('hand_labels', [])
        
        # Identify left and right hands
        left_hand = None
        right_hand = None
        
        for hand_idx, hand in enumerate(hands):
            if len(hand) == 0:
                continue
            
            # Determine hand type from labels or position
            if hand_idx < len(hand_labels):
                hand_type = hand_labels[hand_idx].lower()
            else:
                # Fallback: classify by position (left hand typically on left side)
                wrist_x = hand[0][0] if hand else 0.5
                hand_type = 'left' if wrist_x < 0.5 else 'right'
            
            if hand_type == 'left':
                left_hand = hand
            elif hand_type == 'right':
                right_hand = hand
        
        # Normalize hands relative to each other
        normalized_left, normalized_right = normalize_hands_relative(
            left_hand, right_hand, scale_method=scale_method
        )
        
        # Build normalized hands list preserving order
        normalized_hands = []
        
        for hand_idx, hand in enumerate(hands):
            if len(hand) == 0:
                normalized_hands.append([])
            else:
                # Determine which normalized hand to use
                if hand_idx < len(hand_labels):
                    hand_type = hand_labels[hand_idx].lower()
                else:
                    wrist_x = hand[0][0] if hand else 0.5
                    hand_type = 'left' if wrist_x < 0.5 else 'right'
                
                if hand_type == 'left' and normalized_left is not None:
                    normalized_hands.append(normalized_left.tolist())
                elif hand_type == 'right' and normalized_right is not None:
                    normalized_hands.append(normalized_right.tolist())
                else:
                    # Fallback: normalize independently
                    normalized_hand = normalize_landmarks(hand, scale_method=scale_method)
                    normalized_hands.append(normalized_hand.tolist())
        
        normalized_landmarks.append({
            'frame': frame_idx,
            'hands': normalized_hands
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
        'normalization_strategy': 'left_wrist_anchor',
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

