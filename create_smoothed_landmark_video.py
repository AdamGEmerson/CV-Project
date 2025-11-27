"""
Script to create a video from smoothed landmarks for segment 44.
"""
from pathlib import Path
from src.hand_tracking import create_landmark_video_from_json


def main():
    segment_num = 44
    
    # Paths
    video_path = Path("data/segments") / f"segment_{segment_num:03d}.mp4"
    landmarks_json_path = Path("data/landmarks") / f"segment_{segment_num:03d}_smoothed.json"
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    if not landmarks_json_path.exists():
        print(f"Error: Smoothed landmarks file not found: {landmarks_json_path}")
        print("Please run smooth_segment_landmarks.py first.")
        return
    
    print("="*60)
    print(f"Creating smoothed landmark video for segment {segment_num}")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Landmarks: {landmarks_json_path}")
    print()
    
    # Create video from smoothed landmarks
    output_path = create_landmark_video_from_json(
        str(video_path),
        str(landmarks_json_path)
    )
    
    print()
    print("="*60)
    print("SUCCESS!")
    print(f"Smoothed landmark video saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()

