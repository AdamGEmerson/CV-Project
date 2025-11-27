"""
Script to smooth landmarks for a specific segment using Savitzky-Golay filter.
"""
import json
from pathlib import Path
from src.hand_tracking import smooth_landmarks, save_landmarks


def load_landmarks_from_json(json_path):
    """
    Load landmarks from JSON file.
    
    Args:
        json_path: Path to JSON file containing landmarks
    
    Returns:
        Tuple of (landmarks_list, fps)
        landmarks_list: List of (frame_idx, hands_data) tuples
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    fps = data.get('fps', 30.0)
    landmarks_list = []
    
    for entry in data.get('landmarks', []):
        frame_idx = entry['frame']
        hands = entry['hands']
        landmarks_list.append((frame_idx, hands))
    
    return landmarks_list, fps


def main():
    segment_num = 44
    
    # Load original landmarks
    landmarks_path = Path("data/landmarks") / f"segment_{segment_num:03d}.json"
    
    if not landmarks_path.exists():
        print(f"Error: Landmarks file not found: {landmarks_path}")
        return
    
    print(f"Loading landmarks from: {landmarks_path}")
    landmarks, fps = load_landmarks_from_json(landmarks_path)
    
    print(f"Loaded {len(landmarks)} frames @ {fps:.2f} FPS")
    
    # Count frames with hands
    frames_with_hands = sum(1 for _, hands in landmarks if hands)
    print(f"Frames with hands: {frames_with_hands}")
    print()
    
    # Apply smoothing
    print("Applying Savitzky-Golay filter (window=11, polyorder=2)...")
    smoothed_landmarks = smooth_landmarks(landmarks, window_length=11, polyorder=2)
    
    # Save smoothed landmarks
    output_path = Path("data/landmarks") / f"segment_{segment_num:03d}_smoothed.json"
    print(f"Saving smoothed landmarks to: {output_path}")
    save_landmarks(smoothed_landmarks, fps, output_path, format='json')
    
    print()
    print("="*60)
    print("SUCCESS!")
    print(f"Smoothed landmarks saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()

