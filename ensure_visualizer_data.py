"""
Utility script to ensure the clustered JSON file exists in visualizer/public/.
Can be run manually or imported by other scripts.
"""
from pathlib import Path
import shutil
import sys


def ensure_visualizer_file(source_path=None):
    """
    Ensure the clustered JSON file exists in visualizer/public/ directory.
    If source_path is provided, copy from there. Otherwise, look for it in data/landmarks/.
    
    Args:
        source_path: Optional path to source JSON file. If None, looks in data/landmarks/
    
    Returns:
        Path to visualizer JSON file if successful, None otherwise
    """
    visualizer_public = Path("visualizer/public")
    visualizer_json = visualizer_public / "all_segments_clustered_with_xy.json"
    
    # If visualizer file already exists, return it
    if visualizer_json.exists():
        print(f"Visualizer file already exists: {visualizer_json}")
        return visualizer_json
    
    # Find source file
    if source_path:
        source = Path(source_path)
    else:
        # Look in data/landmarks/
        landmarks_dir = Path("data/landmarks")
        source = landmarks_dir / "all_segments_clustered_with_xy.json"
        
        # Also check in old/ directory
        if not source.exists():
            old_source = landmarks_dir / "old" / "all_segments_clustered_with_xy.json"
            if old_source.exists():
                source = old_source
                print(f"Found source file in old/ directory: {source}")
    
    if not source.exists():
        print(f"Error: Source file not found: {source}")
        print("\nPlease run batch processing to generate the clustered data:")
        print("  python3 batch_process_all_segments.py")
        return None
    
    # Copy to visualizer
    try:
        visualizer_public.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, visualizer_json)
        print(f"✓ Copied clustered data to visualizer: {visualizer_json}")
        return visualizer_json
    except Exception as e:
        print(f"Error: Could not copy to visualizer/public/: {e}")
        return None


def main():
    """Main entry point for the script."""
    source_path = sys.argv[1] if len(sys.argv) > 1 else None
    result = ensure_visualizer_file(source_path)
    
    if result:
        print(f"\n✓ Visualizer data file ready at: {result}")
        sys.exit(0)
    else:
        print("\n✗ Failed to ensure visualizer data file exists.")
        sys.exit(1)


if __name__ == "__main__":
    main()

