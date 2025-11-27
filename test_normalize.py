"""
Test script to normalize landmarks from a segment.
"""
import sys
from pathlib import Path
from src.normalization import normalize_from_json


def main():
    segment_num = 44
    
    # Find landmarks JSON file
    landmarks_path = Path("data/landmarks") / f"segment_{segment_num:03d}.json"
    
    if not landmarks_path.exists():
        print(f"Error: Landmarks file not found: {landmarks_path}")
        print("Please extract landmarks first using extract_landmarks.py")
        return
    
    print(f"Normalizing landmarks for segment {segment_num}")
    print(f"Input: {landmarks_path}")
    print("="*60)
    
    try:
        output_path = normalize_from_json(
            landmarks_path,
            output_path=None,  # Auto-generate output path
            scale_method='palm'
        )
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print(f"  Normalized landmarks saved to: {output_path}")
        print("="*60)
        
        # Show some stats
        import json
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        total_frames = data['total_frames']
        frames_with_hands = sum(1 for frame in data['landmarks'] 
                               if any(len(hand) > 0 for hand in frame['hands']))
        
        print(f"\nStatistics:")
        print(f"  Total frames: {total_frames}")
        print(f"  Frames with hands: {frames_with_hands}")
        print(f"  Scale method: {data['scale_method']}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

