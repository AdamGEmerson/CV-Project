"""
Batch process all clips: extract landmarks, normalize, extract features, and cluster together.
"""
import sys
from pathlib import Path
import time
import shutil
# Removed threading imports - using sequential processing for reliability
from src.hand_tracking import extract_landmarks, save_landmarks, smooth_landmarks
from src.normalization import normalize_from_json
from src.feature_extraction import extract_all_hands_features
from src.cluster import (
    cluster_features,
    load_features_from_npz,
    load_frame_landmarks,
    summarize_cluster_landmarks
)
import numpy as np
import json


def process_segment_pipeline(segment_path, clip_id, use_smoothing=True, smoothing_window=11, smoothing_polyorder=2):
    """
    Process a single segment/clip through the full pipeline.
    
    Args:
        segment_path: Path to the video file
        clip_id: Identifier for the clip (filename without extension)
    
    Returns:
        Tuple of (hand0_features_path, hand1_features_path, smoothed_json_path) or None if failed
        Note: Returns smoothed_json_path (original skeleton data) not normalized_json_path
    """
    segment_path = Path(segment_path)
    landmarks_dir = Path("data/landmarks")
    
    print(f"\n{'='*60}")
    print(f"Processing Clip: {clip_id}")
    print(f"{'='*60}")
    
    # Step 1: Extract landmarks
    landmarks_json = landmarks_dir / f"{clip_id}.json"
    needs_extraction = True
    if landmarks_json.exists():
        # Check if file has the frame indexing bug (all frames labeled 0)
        try:
            with open(landmarks_json, 'r') as f:
                data = json.load(f)
            frames = [entry.get('frame', 0) for entry in data.get('landmarks', [])]
            unique_frames = set(frames)
            # If we have multiple entries but only one unique frame index (and it's 0), it's buggy
            if len(frames) > 1 and len(unique_frames) == 1 and 0 in unique_frames:
                print(f"  [INFO] Detected frame indexing bug in {landmarks_json.name}, regenerating...")
                landmarks_json.unlink()  # Delete buggy file
                # Also delete dependent files that will be regenerated
                smoothed_json = landmarks_dir / f"{clip_id}_smoothed.json"
                if smoothed_json.exists():
                    smoothed_json.unlink()
                normalized_json = landmarks_dir / f"{clip_id}_smoothed_normalized.json"
                if normalized_json.exists():
                    normalized_json.unlink()
                normalized_json_raw = landmarks_dir / f"{clip_id}_normalized.json"
                if normalized_json_raw.exists():
                    normalized_json_raw.unlink()
                # Delete feature files
                for hand_idx in [0, 1]:
                    for suffix in ['_smoothed_normalized_features_distance_matrix', '_normalized_features_distance_matrix']:
                        feature_file = landmarks_dir / f"{clip_id}{suffix}_hand{hand_idx}.npz"
                        if feature_file.exists():
                            feature_file.unlink()
            else:
                needs_extraction = False
                fps = data.get('fps', 25.0)
                print(f"  [OK] Landmarks already exist: {landmarks_json.name} ({len(frames)} frames)")
        except Exception as e:
            print(f"  [WARN] Error checking existing landmarks file: {e}, regenerating...")
            if landmarks_json.exists():
                landmarks_json.unlink()
    
    if needs_extraction:
        print(f"  Step 1: Extracting landmarks...")
        try:
            landmarks, fps = extract_landmarks(str(segment_path), use_gpu=False, save_format='json')
            save_landmarks(landmarks, fps, landmarks_json, format='json')
            print(f"  [OK] Landmarks extracted: {len(landmarks)} frames")
        except Exception as e:
            print(f"  [FAIL] Failed to extract landmarks: {e}")
            return None, None, None
    
    # Step 2: Smooth landmarks (if requested)
    if use_smoothing:
        smoothed_json = landmarks_dir / f"{clip_id}_smoothed.json"
        if not smoothed_json.exists():
            print(f"  Step 2: Smoothing landmarks (window={smoothing_window}, polyorder={smoothing_polyorder})...")
            try:
                # Load landmarks with hand_labels if available
                with open(landmarks_json, 'r') as f:
                    data = json.load(f)
                landmarks_list = []
                for entry in data['landmarks']:
                    frame_idx = entry['frame']
                    hands = entry['hands']
                    hand_labels = entry.get('hand_labels', [])
                    if hand_labels:
                        landmarks_list.append((frame_idx, hands, hand_labels))
                    else:
                        landmarks_list.append((frame_idx, hands))
                
                # Smooth
                smoothed_landmarks = smooth_landmarks(landmarks_list, window_length=smoothing_window, polyorder=smoothing_polyorder)
                save_landmarks(smoothed_landmarks, fps, smoothed_json, format='json')
                print(f"  [OK] Landmarks smoothed")
            except Exception as e:
                print(f"  [FAIL] Failed to smooth landmarks: {e}")
                return None, None, None
        else:
            print(f"  [OK] Smoothed landmarks already exist: {smoothed_json.name}")
        
        normalized_json = landmarks_dir / f"{clip_id}_smoothed_normalized.json"
        input_json = smoothed_json
        skeleton_json = smoothed_json  # Use smoothed (original) skeleton data
    else:
        normalized_json = landmarks_dir / f"{clip_id}_normalized.json"
        input_json = landmarks_json
        skeleton_json = landmarks_json  # Use original skeleton data
    
    # Step 3: Normalize landmarks
    needs_normalization = not normalized_json.exists()
    
    if needs_normalization:
        print(f"  Step 3: Normalizing landmarks...")
        try:
            normalize_from_json(str(input_json), output_path=str(normalized_json), scale_method='palm')
            print(f"  [OK] Landmarks normalized")
        except Exception as e:
            print(f"  [FAIL] Failed to normalize landmarks: {e}")
            return None, None, None
    else:
        print(f"  [OK] Normalized landmarks already include centroid metadata: {normalized_json.name}")
    
    # Step 4: Extract features (consolidated into single NPZ file)
    if use_smoothing:
        features_file = landmarks_dir / f"{clip_id}_smoothed_normalized_features_distance_matrix.npz"
    else:
        features_file = landmarks_dir / f"{clip_id}_normalized_features_distance_matrix.npz"
    
    if not features_file.exists():
        print(f"  Step 4: Extracting features...")
        try:
            results = extract_all_hands_features(str(normalized_json), method='distance_matrix', output_path=str(features_file))
            print(f"  [OK] Features extracted: {len(results['common_frames'])} frames with both hands")
        except Exception as e:
            print(f"  [FAIL] Failed to extract features: {e}")
            return None, None, None
    else:
        print(f"  [OK] Features already exist: {features_file.name}")
    
    return features_file, features_file, skeleton_json  # Return same file twice for compatibility


def check_clip_has_both_hands(clip_file, clip_id, landmarks_dir, use_smoothing=True):
    """
    Quick check if a clip has any frames with both hands detected.
    This is a lightweight check to filter clips before full processing.
    
    Args:
        clip_file: Path to video file
        clip_id: Clip identifier
        landmarks_dir: Directory for landmarks
        use_smoothing: Whether to check smoothed landmarks
    
    Returns:
        Tuple of (has_both_hands: bool, frames_with_both_hands: int)
    """
    # Check if landmarks already exist
    if use_smoothing:
        landmarks_json = landmarks_dir / f"{clip_id}_smoothed.json"
        if not landmarks_json.exists():
            landmarks_json = landmarks_dir / f"{clip_id}.json"
    else:
        landmarks_json = landmarks_dir / f"{clip_id}.json"
    
    # If landmarks exist, check them
    if landmarks_json.exists():
        try:
            with open(landmarks_json, 'r') as f:
                data = json.load(f)
            
            frames_with_both_hands = 0
            for entry in data.get('landmarks', []):
                hands = entry.get('hands', [])
                # Count how many hands have landmarks
                valid_hands = sum(1 for hand in hands if hand and len(hand) > 0)
                if valid_hands >= 2:
                    frames_with_both_hands += 1
            
            return frames_with_both_hands > 0, frames_with_both_hands
        except Exception as e:
            # If we can't read the file, we'll need to extract landmarks
            pass
    
    # If landmarks don't exist, we need to extract them (but don't save)
    # This is slower but necessary for first pass
    try:
        from src.hand_tracking import extract_landmarks
        landmarks, fps = extract_landmarks(str(clip_file), use_gpu=False, save_format='json')
        
        frames_with_both_hands = 0
        for entry in landmarks:
            if len(entry) >= 2:
                hands = entry[1]
                valid_hands = sum(1 for hand in hands if hand and len(hand) > 0)
                if valid_hands >= 2:
                    frames_with_both_hands += 1
        
        # Save the landmarks since we extracted them
        landmarks_json = landmarks_dir / f"{clip_id}.json"
        from src.hand_tracking import save_landmarks
        save_landmarks(landmarks, fps, landmarks_json, format='json')
        
        return frames_with_both_hands > 0, frames_with_both_hands
    except Exception as e:
        return False, 0


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
    
    if not source.exists():
        print(f"Warning: Source file not found: {source}")
        print("Please run batch_process_all_segments.py to generate the clustered data.")
        return None
    
    # Copy to visualizer
    try:
        visualizer_public.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, visualizer_json)
        print(f"Copied clustered data to visualizer: {visualizer_json}")
        return visualizer_json
    except Exception as e:
        print(f"Warning: Could not copy to visualizer/public/: {e}")
        return None


def combine_hands_features(features_path, normalized_json_path=None):
    """
    Load combined features from consolidated NPZ file.
    
    Args:
        features_path: Path to consolidated features NPZ file (contains combined_features)
        normalized_json_path: Not used anymore (kept for compatibility)
    
    Returns:
        Tuple of (combined_features_array, common_frames)
        Combined features include: hand0_intra (441) + hand1_intra (441) + inter_hand (441) = 1323 dims
    """
    data = load_features_from_npz(features_path)
    
    # Check if this is the new consolidated format
    if 'combined_features' in data:
        return data['combined_features'], list(data['frame_indices'])
    
    # Fallback: old format (separate hand0/hand1 files) - shouldn't happen with new code
    return None, None


def main():
    clips_dir = Path("data/clips")
    landmarks_dir = Path("data/landmarks")
    
    # Clustering parameters
    pca_components = 11
    min_cluster_size = 40
    min_samples = 15
    use_smoothing = True
    smoothing_window = 11
    smoothing_polyorder = 2
    
    # Parse command line arguments
    if '--pca' in sys.argv:
        try:
            idx = sys.argv.index('--pca')
            pca_components = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --pca value, using default 11")
    
    if '--min-size' in sys.argv:
        try:
            idx = sys.argv.index('--min-size')
            min_cluster_size = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --min-size value, using default 50")
    
    if '--min-samples' in sys.argv:
        try:
            idx = sys.argv.index('--min-samples')
            min_samples = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --min-samples value, using default 10")
    
    if '--no-smoothing' in sys.argv:
        use_smoothing = False
    
    # Find all clip files
    clip_files = sorted(clips_dir.glob("*.mp4"))
    
    if len(clip_files) == 0:
        print("Error: No clip files found in data/clips/")
        return
    
    print("="*60)
    print("BATCH PROCESSING ALL CLIPS")
    print("="*60)
    print(f"Found {len(clip_files)} clips")
    print(f"Parameters:")
    print(f"  Smoothing: {use_smoothing} (window={smoothing_window}, polyorder={smoothing_polyorder})")
    print(f"  PCA components: {pca_components}")
    print(f"  Min cluster size: {min_cluster_size}")
    print(f"  Min samples: {min_samples}")
    print("="*60)
    
    # FIRST PASS: Check which clips have frames with both hands
    print("\n" + "="*60)
    print("PASS 1: Checking for clips with both hands")
    print("="*60)
    clips_with_both_hands = []
    clips_without_both_hands = []
    
    total_clips = len(clip_files)
    pass1_start = time.time()
    
    for idx, clip_file in enumerate(clip_files, 1):
        clip_id = clip_file.stem
        remaining = total_clips - idx
        
        print(f"[{idx}/{total_clips}] Checking: {clip_id} ({remaining} remaining)", end='', flush=True)
        
        try:
            has_both, count = check_clip_has_both_hands(
                clip_file, clip_id, landmarks_dir, use_smoothing=use_smoothing
            )
            
            if has_both:
                clips_with_both_hands.append(clip_file)
                print(f" ✓ {count} frames with both hands")
            else:
                clips_without_both_hands.append(clip_id)
                print(f" ✗ No frames with both hands")
        except Exception as e:
            clips_without_both_hands.append(clip_id)
            print(f" ✗ Error: {e}")
    
    pass1_elapsed = time.time() - pass1_start
    print(f"\nPass 1 complete in {pass1_elapsed:.1f} seconds")
    print(f"  Clips with both hands: {len(clips_with_both_hands)}")
    print(f"  Clips without both hands: {len(clips_without_both_hands)}")
    
    if len(clips_with_both_hands) == 0:
        print("\nERROR: No clips found with frames containing both hands!")
        print("Cannot proceed with clustering.")
        return
    
    # SECOND PASS: Process only clips with both hands
    print("\n" + "="*60)
    print("PASS 2: Processing clips with both hands")
    print("="*60)
    
    # Process each segment sequentially
    all_combined_features = []
    all_frame_info = []
    successful_segments = []
    failed_segments = []
    
    # Process clips sequentially (single-threaded) to avoid file I/O conflicts
    # OpenCV VideoCapture and MediaPipe have thread-safety issues
    # Sequential processing is more reliable for file operations
    print(f"\nProcessing {len(clips_with_both_hands)} clips sequentially (single-threaded for reliability)...")
    start_time = time.time()
    
    # Process each clip sequentially
    total_clips = len(clips_with_both_hands)
    for idx, clip_file in enumerate(clips_with_both_hands, 1):
        clip_id = clip_file.stem  # Use filename without extension as identifier
        remaining = total_clips - idx
        
        print(f"\n[{idx}/{total_clips}] Processing: {clip_id} ({remaining} remaining)")
        
        try:
            features_path, _, skeleton_json_path = process_segment_pipeline(
                clip_file,
                clip_id,
                use_smoothing=use_smoothing,
                smoothing_window=smoothing_window,
                smoothing_polyorder=smoothing_polyorder
            )
            
            if features_path is None or skeleton_json_path is None:
                failed_segments.append(clip_id)
                print(f"  [FAIL] Pipeline failed for {clip_id}")
                continue
            
            # Load combined features (already includes inter-hand distances in consolidated file)
            combined_features, common_frames = combine_hands_features(features_path)
            
            if combined_features is None or len(combined_features) == 0:
                print(f"  [WARN] Clip {clip_id}: No common frames with both hands")
                failed_segments.append(clip_id)
                continue
            
            # Load per-frame original skeleton landmarks (smoothed but NOT PCA-normalized)
            # These are the raw MediaPipe coordinates before normalization
            landmarks_lookup = load_frame_landmarks(skeleton_json_path)
            
            # Add to combined dataset
            for i, frame in enumerate(common_frames):
                all_combined_features.append(combined_features[i])
                all_frame_info.append({
                    'clip': clip_id,
                    'frame': int(frame),
                    'feature_idx': len(all_combined_features) - 1,
                    'hand_landmarks': landmarks_lookup.get(int(frame), [])
                })
            
            successful_segments.append(clip_id)
            elapsed_so_far = time.time() - start_time
            avg_time_per_clip = elapsed_so_far / idx if idx > 0 else 0
            remaining = total_clips - idx
            estimated_remaining = avg_time_per_clip * remaining if avg_time_per_clip > 0 else 0
            print(f"  [OK] {len(combined_features)} frames added | Progress: {idx}/{total_clips} ({idx/total_clips*100:.1f}%) | Est. remaining: {estimated_remaining/60:.1f} min")
            
        except Exception as e:
            print(f"  [FAIL] Clip {clip_id} failed: {e}")
            failed_segments.append(clip_id)
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total clips scanned: {len(clip_files)}")
    print(f"  Clips with both hands: {len(clips_with_both_hands)}")
    print(f"  Clips without both hands: {len(clips_without_both_hands)}")
    print(f"Successful clips processed: {len(successful_segments)}")
    print(f"Failed clips: {len(failed_segments)}")
    print(f"Total frames with both hands: {len(all_combined_features)}")
    print(f"Pass 1 time: {pass1_elapsed:.1f} seconds")
    print(f"Pass 2 time: {elapsed:.1f} seconds")
    print(f"Total processing time: {pass1_elapsed + elapsed:.1f} seconds")
    
    if len(all_combined_features) == 0:
        print("\nERROR: No features collected from any segment!")
        return
    
    # Cluster all segments together
    print("\n" + "="*60)
    print("CLUSTERING ALL SEGMENTS TOGETHER")
    print("="*60)
    print(f"Total features: {len(all_combined_features)}")
    print(f"Feature dimension: {all_combined_features[0].shape[0]} (1323 = 441 per hand + 441 inter-hand)")
    print(f"PCA components: {pca_components}")
    print(f"Min cluster size: {min_cluster_size}")
    print(f"Min samples: {min_samples}")
    print()
    
    all_combined_features = np.array(all_combined_features)
    
    try:
        cluster_result = cluster_features(
            all_combined_features,
            pca_components=pca_components,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        
        labels = cluster_result['labels']
        embeddings = cluster_result['X_reduced']
        
        # Add cluster labels and embedding coordinates to frame info
        for i, label in enumerate(labels):
            embedding_row = embeddings[i]
            x = float(embedding_row[0]) if len(embedding_row) > 0 else 0.0
            y = float(embedding_row[1]) if len(embedding_row) > 1 else 0.0
            z = float(embedding_row[2]) if len(embedding_row) > 2 else 0.0
            all_frame_info[i]['cluster'] = int(label)
            all_frame_info[i]['x'] = x
            all_frame_info[i]['y'] = y
            all_frame_info[i]['z'] = z
        
        # Create results
        results = {
            'n_clips': len(successful_segments),
            'successful_clips': successful_segments,
            'failed_clips': failed_segments,
            'n_frames': len(all_combined_features),
            'n_clusters': cluster_result['n_clusters'],
            'n_noise': cluster_result['n_noise'],
            'pca_components': pca_components,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'use_smoothing': use_smoothing,
            'smoothing_window': smoothing_window if use_smoothing else None,
            'smoothing_polyorder': smoothing_polyorder if use_smoothing else None,
            'feature_dim': all_combined_features.shape[1],
            'reduced_dim': cluster_result['X_reduced'].shape[1],
            'clustered_frames': all_frame_info,
            'cluster_distribution': {
                int(cluster): int(np.sum(labels == cluster))
                for cluster in set(labels)
            },
            'cluster_landmarks': summarize_cluster_landmarks(all_frame_info)
        }
        
        # Save results
        output_path = landmarks_dir / "all_segments_clustered_with_xy.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        labels_path = landmarks_dir / "all_segments_clustered_with_xy.npy"
        np.save(labels_path, labels)
        
        # Ensure visualizer has the file
        ensure_visualizer_file(output_path)
        
        print("="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Total clips processed: {len(successful_segments)}")
        print(f"Total frames (both hands): {results['n_frames']}")
        print(f"Number of clusters: {results['n_clusters']}")
        print(f"Noise/outliers: {results['n_noise']} ({results['n_noise']/results['n_frames']*100:.1f}%)")
        print(f"Feature dimension: {results['feature_dim']}")
        print(f"Reduced dimension: {results['reduced_dim']}")
        print()
        print("Cluster distribution:")
        for cluster_id, count in sorted(results['cluster_distribution'].items()):
            if cluster_id == -1:
                print(f"  Noise: {count} frames ({count/results['n_frames']*100:.1f}%)")
            else:
                print(f"  Cluster {cluster_id}: {count} frames ({count/results['n_frames']*100:.1f}%)")
        print()
        print(f"Results saved to: {output_path}")
        print(f"Labels saved to: {labels_path}")
        
        # Ensure visualizer has the file
        ensure_visualizer_file(output_path)
        
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR during clustering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

