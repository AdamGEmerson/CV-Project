"""
Test script to extract features from normalized landmarks.
"""
import sys
from pathlib import Path
from src.feature_extraction import extract_features_from_json, extract_all_hands_features


def main():
    segment_num = 44
    
    # Find normalized landmarks JSON file
    normalized_path = Path("data/landmarks") / f"segment_{segment_num:03d}_normalized.json"
    
    if not normalized_path.exists():
        print(f"Error: Normalized landmarks file not found: {normalized_path}")
        print("Please normalize landmarks first using test_normalize.py")
        return
    
    print(f"Extracting features for segment {segment_num}")
    print(f"Input: {normalized_path}")
    print("="*60)
    
    # Check for method flag
    method = 'distance_matrix'
    if '--flatten' in sys.argv:
        method = 'flatten'
    elif '--upper-triangle' in sys.argv:
        method = 'distance_matrix_upper_triangle'
    
    print(f"Method: {method}")
    print()
    
    try:
        # Extract features for all hands
        result = extract_all_hands_features(
            normalized_path,
            method=method,
            output_path=None  # Auto-generate
        )
        
        print("="*60)
        print("SUCCESS!")
        print()
        
        # Hand 0 results
        hand0 = result['hand0']
        print(f"Hand 0:")
        print(f"  Features shape: {hand0['features'].shape}")
        print(f"  Frames with features: {hand0['metadata']['frames_with_features']}")
        print(f"  Feature dimension: {hand0['metadata']['feature_dim']}")
        print(f"  Saved to: {hand0['output_path']}")
        print()
        
        # Hand 1 results (if present)
        hand1 = result['hand1']
        if hand1['features'].shape[0] > 0:
            print(f"Hand 1:")
            print(f"  Features shape: {hand1['features'].shape}")
            print(f"  Frames with features: {hand1['metadata']['frames_with_features']}")
            print(f"  Feature dimension: {hand1['metadata']['feature_dim']}")
            print(f"  Saved to: {hand1['output_path']}")
        else:
            print("Hand 1: No features (hand not detected)")
        
        print()
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

