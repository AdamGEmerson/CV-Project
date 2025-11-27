"""
Script to normalize landmarks for a specific segment.
"""
from pathlib import Path
from src.normalization import normalize_from_json


def main():
    segment_num = 44
    use_smoothed = True  # Set to True to use smoothed landmarks
    
    # Path to landmarks file
    if use_smoothed:
        landmarks_path = Path("data/landmarks") / f"segment_{segment_num:03d}_smoothed.json"
    else:
        landmarks_path = Path("data/landmarks") / f"segment_{segment_num:03d}.json"
    
    if not landmarks_path.exists():
        print(f"Error: Landmarks file not found: {landmarks_path}")
        return
    
    print(f"Normalizing landmarks from: {landmarks_path}")
    
    # Normalize landmarks
    output_path = normalize_from_json(
        str(landmarks_path),
        scale_method='palm'
    )
    
    print()
    print("="*60)
    print("SUCCESS!")
    print(f"Normalized landmarks saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()

