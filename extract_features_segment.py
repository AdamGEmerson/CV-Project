"""
Script to extract features from normalized landmarks for a specific segment.
"""
from pathlib import Path
from src.feature_extraction import extract_all_hands_features


def main():
    segment_num = 44
    use_smoothed = True  # Set to True to use smoothed normalized landmarks
    
    # Path to normalized landmarks file
    if use_smoothed:
        normalized_path = Path("data/landmarks") / f"segment_{segment_num:03d}_smoothed_normalized.json"
    else:
        normalized_path = Path("data/landmarks") / f"segment_{segment_num:03d}_normalized.json"
    
    if not normalized_path.exists():
        print(f"Error: Normalized landmarks file not found: {normalized_path}")
        return
    
    print(f"Extracting features from: {normalized_path}")
    print("Method: distance_matrix (441-dim feature vector)")
    print()
    
    # Extract features for both hands
    results = extract_all_hands_features(
        str(normalized_path),
        method='distance_matrix'
    )
    
    print()
    print("="*60)
    print("SUCCESS!")
    print()
    
    # Print results for hand 0
    hand0 = results['hand0']
    print(f"Hand 0:")
    print(f"  Features: {hand0['metadata']['frames_with_features']} frames")
    print(f"  Feature dimension: {hand0['metadata']['feature_dim']}")
    print(f"  Output: {hand0['output_path']}")
    print(f"  Metadata: {hand0['metadata_path']}")
    
    # Print results for hand 1
    hand1 = results['hand1']
    print()
    print(f"Hand 1:")
    print(f"  Features: {hand1['metadata']['frames_with_features']} frames")
    print(f"  Feature dimension: {hand1['metadata']['feature_dim']}")
    print(f"  Output: {hand1['output_path']}")
    print(f"  Metadata: {hand1['metadata_path']}")
    
    print()
    print("="*60)


if __name__ == "__main__":
    main()

