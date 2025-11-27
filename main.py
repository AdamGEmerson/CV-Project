"""
Main script to segment videos based on hand presence.
"""
import sys
import json
from pathlib import Path
from src.shot_segmentation import segment_by_hand_presence, save_segments


def main():
    # Default video path - update this or pass as command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Use first video from videos/cv-data/ if available
        video_dir = Path("videos/cv-data")
        video_files = list(video_dir.glob("*.mp4"))
        if not video_files:
            print("Error: No video files found. Please provide a video path as argument.")
            print("Usage: python main.py <path_to_video> [--gpu|--no-gpu]")
            sys.exit(1)
        video_path = str(video_files[0])
        print(f"Using video: {video_path}")

    print(f"Processing video: {video_path}")
    print("Detecting hand presence and segmenting...")

    # Check for GPU flag
    use_gpu = '--no-gpu' not in sys.argv
    if '--gpu' in sys.argv:
        use_gpu = True

    # Segment video by hand presence
    segments, fps = segment_by_hand_presence(video_path, min_active_frames=15, use_gpu=use_gpu)

    print(f"\nFound {len(segments)} segments")
    print(f"Video FPS: {fps:.2f}")
    
    if len(segments) == 0:
        print("No segments with hand activity found.")
        return

    # Display segment information
    print("\nSegments:")
    for i, (start, end) in enumerate(segments):
        start_time = start / fps
        end_time = end / fps
        duration = (end - start) / fps
        print(f"  Segment {i:03d}: frames {start}-{end} "
              f"({start_time:.2f}s - {end_time:.2f}s, duration: {duration:.2f}s)")

    # Save segments metadata to JSON file before attempting video export
    segments_data = {
        "video_path": video_path,
        "fps": fps,
        "segments": [
            {
                "index": i,
                "start_frame": start,
                "end_frame": end,
                "start_time": start / fps,
                "end_time": end / fps,
                "duration": (end - start) / fps
            }
            for i, (start, end) in enumerate(segments)
        ]
    }
    
    segments_file = Path("data/segments_metadata.json")
    segments_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(segments_file, 'w') as f:
        json.dump(segments_data, f, indent=2)
    
    print(f"\nSaved segments metadata to: {segments_file}")
    print(f"  Total segments: {len(segments)}")
    print(f"  Video FPS: {fps:.2f}")

    # Save segments to video files
    print(f"\nExporting segments to data/segments/...")
    try:
        output_paths = save_segments(video_path, segments, fps, out_dir="data/segments")
        
        # Update metadata with output paths
        segments_data["exported_files"] = output_paths
        with open(segments_file, 'w') as f:
            json.dump(segments_data, f, indent=2)
        
        print(f"\nSuccessfully exported {len(output_paths)} segments:")
        for path in output_paths:
            print(f"  - {path}")
    except RuntimeError as e:
        print(f"\n{str(e)}")
        print(f"\nSegments metadata has been saved to: {segments_file}")
        print("You can use this file to export segments later once ffmpeg is configured.")
        sys.exit(1)


if __name__ == "__main__":
    main()

