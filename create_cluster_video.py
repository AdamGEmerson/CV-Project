"""
Script to create a video with cluster labels overlaid.
"""
import sys
from pathlib import Path
from src.visualize_clusters import create_cluster_labeled_video


def main():
    segment_num = 44
    use_smoothed = True  # Set to True to use smoothed clustering results
    
    # Find files
    video_path = Path("data/segments") / f"segment_{segment_num:03d}.mp4"
    if use_smoothed:
        cluster_json = Path("data/landmarks") / f"segment_{segment_num:03d}_smoothed_normalized_features_distance_matrix_both_hands_clustered.json"
    else:
        cluster_json = Path("data/landmarks") / f"segment_{segment_num:03d}_normalized_features_distance_matrix_both_hands_clustered.json"
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    if not cluster_json.exists():
        print(f"Error: Clustering results not found: {cluster_json}")
        print("Please run clustering first using test_cluster_both_hands.py")
        return
    
    print(f"Creating cluster-labeled video for segment {segment_num}")
    print(f"Video: {video_path.name}")
    print(f"Clusters: {cluster_json.name}")
    print("="*60)
    
    try:
        output_path = create_cluster_labeled_video(
            video_path,
            cluster_json,
            output_path=None  # Auto-generate
        )
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print(f"Cluster-labeled video saved to: {output_path}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

