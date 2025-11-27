"""
Test script to process a single segment.
"""
import sys
from pathlib import Path
from src.hand_tracking import process_segment, create_landmark_video


def main():
    segment_num = 44
    
    # Find segment file
    segment_path = Path("data/segments") / f"segment_{segment_num:03d}.mp4"
    
    if not segment_path.exists():
        print(f"Error: Segment file not found: {segment_path}")
        return
    
    print(f"Testing segment {segment_num}: {segment_path.name}")
    print("="*60)
    
    try:
        # Extract landmarks and create preview
        landmarks_path, preview_path = process_segment(
            str(segment_path),
            output_dir="data/landmarks",
            use_gpu=True,
            save_format='json',
            create_preview=True
        )
        
        print("\n" + "="*60)
        print("Landmarks extracted successfully!")
        print(f"  Landmarks saved to: {landmarks_path}")
        if preview_path:
            print(f"  Preview saved to: {preview_path}")
        
        # Create landmark video
        print("\n" + "="*60)
        print("Creating landmark video...")
        video_path = create_landmark_video(
            str(segment_path),
            output_path=None,
            use_gpu=True
        )
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print(f"  Landmark video saved to: {video_path}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

