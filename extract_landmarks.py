"""
Script to extract hand landmarks from all video segments.
"""
import sys
from pathlib import Path
from src.hand_tracking import process_segment


def main():
    segments_dir = Path("data/segments")
    output_dir = Path("data/landmarks")
    
    # Get all segment video files
    segment_files = sorted(segments_dir.glob("segment_*.mp4"))
    
    if not segment_files:
        print(f"No segment files found in {segments_dir}")
        return
    
    print(f"Found {len(segment_files)} segments to process")
    print(f"Output directory: {output_dir}\n")
    
    # Check for GPU flag
    use_gpu = '--no-gpu' not in sys.argv
    if '--gpu' in sys.argv:
        use_gpu = True
    
    # Check save format
    save_format = 'json'
    if '--numpy' in sys.argv or '--npy' in sys.argv:
        save_format = 'numpy'
    
    successful = 0
    failed = 0
    
    for i, segment_path in enumerate(segment_files, 1):
        print(f"[{i}/{len(segment_files)}] Processing {segment_path.name}...")
        
        try:
            landmarks_path, preview_path = process_segment(
                str(segment_path),
                output_dir=str(output_dir),
                use_gpu=use_gpu,
                save_format=save_format,
                create_preview=True
            )
            print(f"  ✓ Landmarks saved to: {landmarks_path}")
            if preview_path:
                print(f"  ✓ Preview saved to: {preview_path}")
            print()
            successful += 1
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(segment_files)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

