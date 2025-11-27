"""
Batch process all segments: extract landmarks, normalize, extract features, and cluster together.
"""
import sys
from pathlib import Path
import time
from src.hand_tracking import extract_landmarks, save_landmarks, smooth_landmarks
from src.normalization import normalize_from_json
from src.feature_extraction import extract_all_hands_features
from src.cluster import cluster_features, load_features_from_npz
import numpy as np
import json


def process_segment_pipeline(segment_path, segment_num, use_smoothing=True, smoothing_window=11, smoothing_polyorder=2):
    """
    Process a single segment through the full pipeline.
    
    Returns:
        Tuple of (hand0_features_path, hand1_features_path) or None if failed
    """
    segment_path = Path(segment_path)
    landmarks_dir = Path("data/landmarks")
    
    print(f"\n{'='*60}")
    print(f"Processing Segment {segment_num:03d}")
    print(f"{'='*60}")
    
    # Step 1: Extract landmarks
    landmarks_json = landmarks_dir / f"segment_{segment_num:03d}.json"
    if not landmarks_json.exists():
        print(f"  Step 1: Extracting landmarks...")
        try:
            landmarks, fps = extract_landmarks(str(segment_path), use_gpu=True, save_format='json')
            save_landmarks(landmarks, fps, landmarks_json, format='json')
            print(f"  [OK] Landmarks extracted: {len(landmarks)} frames")
        except Exception as e:
            print(f"  [FAIL] Failed to extract landmarks: {e}")
            return None, None
    else:
        print(f"  [OK] Landmarks already exist: {landmarks_json.name}")
        with open(landmarks_json, 'r') as f:
            data = json.load(f)
            fps = data.get('fps', 25.0)
    
    # Step 2: Smooth landmarks (if requested)
    if use_smoothing:
        smoothed_json = landmarks_dir / f"segment_{segment_num:03d}_smoothed.json"
        if not smoothed_json.exists():
            print(f"  Step 2: Smoothing landmarks (window={smoothing_window}, polyorder={smoothing_polyorder})...")
            try:
                # Load landmarks
                with open(landmarks_json, 'r') as f:
                    data = json.load(f)
                landmarks_list = [(entry['frame'], entry['hands']) for entry in data['landmarks']]
                
                # Smooth
                smoothed_landmarks = smooth_landmarks(landmarks_list, window_length=smoothing_window, polyorder=smoothing_polyorder)
                save_landmarks(smoothed_landmarks, fps, smoothed_json, format='json')
                print(f"  [OK] Landmarks smoothed")
            except Exception as e:
                print(f"  [FAIL] Failed to smooth landmarks: {e}")
                return None, None
        else:
            print(f"  [OK] Smoothed landmarks already exist: {smoothed_json.name}")
        
        normalized_json = landmarks_dir / f"segment_{segment_num:03d}_smoothed_normalized.json"
        input_json = smoothed_json
    else:
        normalized_json = landmarks_dir / f"segment_{segment_num:03d}_normalized.json"
        input_json = landmarks_json
    
    # Step 3: Normalize landmarks
    if not normalized_json.exists():
        print(f"  Step 3: Normalizing landmarks...")
        try:
            normalize_from_json(str(input_json), output_path=str(normalized_json), scale_method='palm')
            print(f"  [OK] Landmarks normalized")
        except Exception as e:
            print(f"  [FAIL] Failed to normalize landmarks: {e}")
            return None, None
    else:
        print(f"  [OK] Normalized landmarks already exist: {normalized_json.name}")
    
    # Step 4: Extract features
    if use_smoothing:
        hand0_features = landmarks_dir / f"segment_{segment_num:03d}_smoothed_normalized_features_distance_matrix_hand0.npz"
        hand1_features = landmarks_dir / f"segment_{segment_num:03d}_smoothed_normalized_features_distance_matrix_hand1.npz"
    else:
        hand0_features = landmarks_dir / f"segment_{segment_num:03d}_normalized_features_distance_matrix_hand0.npz"
        hand1_features = landmarks_dir / f"segment_{segment_num:03d}_normalized_features_distance_matrix_hand1.npz"
    
    if not hand0_features.exists() or not hand1_features.exists():
        print(f"  Step 4: Extracting features...")
        try:
            results = extract_all_hands_features(str(normalized_json), method='distance_matrix')
            hand0_features = Path(results['hand0']['output_path'])
            hand1_features = Path(results['hand1']['output_path'])
            print(f"  [OK] Features extracted: Hand0={results['hand0']['metadata']['frames_with_features']} frames, "
                  f"Hand1={results['hand1']['metadata']['frames_with_features']} frames")
        except Exception as e:
            print(f"  [FAIL] Failed to extract features: {e}")
            return None, None
    else:
        print(f"  [OK] Features already exist: {hand0_features.name}, {hand1_features.name}")
    
    return hand0_features, hand1_features


def combine_hands_features(hand0_path, hand1_path):
    """Combine features from both hands for a segment."""
    hand0_data = load_features_from_npz(hand0_path)
    hand1_data = load_features_from_npz(hand1_path)
    
    # Get common frames
    hand0_frames = set(hand0_data['frame_indices'])
    hand1_frames = set(hand1_data['frame_indices'])
    common_frames = sorted(list(hand0_frames & hand1_frames))
    
    if len(common_frames) == 0:
        return None, None
    
    # Create mapping
    hand0_frame_to_idx = {frame: idx for idx, frame in enumerate(hand0_data['frame_indices'])}
    hand1_frame_to_idx = {frame: idx for idx, frame in enumerate(hand1_data['frame_indices'])}
    
    # Combine features
    combined_features = []
    for frame in common_frames:
        h0_idx = hand0_frame_to_idx[frame]
        h1_idx = hand1_frame_to_idx[frame]
        combined_feature = np.concatenate([
            hand0_data['features'][h0_idx],
            hand1_data['features'][h1_idx]
        ])
        combined_features.append(combined_feature)
    
    return np.array(combined_features), common_frames


def main():
    segments_dir = Path("data/segments")
    landmarks_dir = Path("data/landmarks")
    
    # Clustering parameters
    pca_components = 30
    min_cluster_size = 40
    min_samples = 40
    use_smoothing = True
    smoothing_window = 11
    smoothing_polyorder = 2
    
    # Parse command line arguments
    if '--pca' in sys.argv:
        try:
            idx = sys.argv.index('--pca')
            pca_components = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --pca value, using default 30")
    
    if '--min-size' in sys.argv:
        try:
            idx = sys.argv.index('--min-size')
            min_cluster_size = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --min-size value, using default 40")
    
    if '--min-samples' in sys.argv:
        try:
            idx = sys.argv.index('--min-samples')
            min_samples = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --min-samples value, using default 40")
    
    if '--no-smoothing' in sys.argv:
        use_smoothing = False
    
    # Find all segment files
    segment_files = sorted(segments_dir.glob("segment_*.mp4"))
    
    if len(segment_files) == 0:
        print("Error: No segment files found in data/segments/")
        return
    
    print("="*60)
    print("BATCH PROCESSING ALL SEGMENTS")
    print("="*60)
    print(f"Found {len(segment_files)} segments")
    print(f"Parameters:")
    print(f"  Smoothing: {use_smoothing} (window={smoothing_window}, polyorder={smoothing_polyorder})")
    print(f"  PCA components: {pca_components}")
    print(f"  Min cluster size: {min_cluster_size}")
    print(f"  Min samples: {min_samples}")
    print("="*60)
    
    # Process each segment
    all_combined_features = []
    all_frame_info = []
    successful_segments = []
    failed_segments = []
    
    start_time = time.time()
    
    for segment_file in segment_files:
        # Extract segment number from filename
        segment_num = int(segment_file.stem.split('_')[1])
        
        try:
            hand0_path, hand1_path = process_segment_pipeline(
                segment_file,
                segment_num,
                use_smoothing=use_smoothing,
                smoothing_window=smoothing_window,
                smoothing_polyorder=smoothing_polyorder
            )
            
            if hand0_path is None or hand1_path is None:
                failed_segments.append(segment_num)
                continue
            
            # Combine hands features
            combined_features, common_frames = combine_hands_features(hand0_path, hand1_path)
            
            if combined_features is None or len(combined_features) == 0:
                print(f"  [WARN] No common frames with both hands, skipping segment {segment_num}")
                failed_segments.append(segment_num)
                continue
            
            # Add to combined dataset
            for i, frame in enumerate(common_frames):
                all_combined_features.append(combined_features[i])
                all_frame_info.append({
                    'segment': segment_num,
                    'frame': int(frame),
                    'feature_idx': len(all_combined_features) - 1
                })
            
            successful_segments.append(segment_num)
            print(f"  [OK] Segment {segment_num:03d}: {len(combined_features)} frames added")
            
        except Exception as e:
            print(f"  [FAIL] Segment {segment_num:03d} failed: {e}")
            failed_segments.append(segment_num)
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Successful segments: {len(successful_segments)}")
    print(f"Failed segments: {len(failed_segments)}")
    print(f"Total frames with both hands: {len(all_combined_features)}")
    print(f"Processing time: {elapsed:.1f} seconds")
    
    if len(all_combined_features) == 0:
        print("\nERROR: No features collected from any segment!")
        return
    
    # Cluster all segments together
    print("\n" + "="*60)
    print("CLUSTERING ALL SEGMENTS TOGETHER")
    print("="*60)
    print(f"Total features: {len(all_combined_features)}")
    print(f"Feature dimension: {all_combined_features[0].shape[0]} (882 = 441 per hand)")
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
        
        # Add cluster labels to frame info
        for i, label in enumerate(labels):
            all_frame_info[i]['cluster'] = int(label)
        
        # Create results
        results = {
            'n_segments': len(successful_segments),
            'successful_segments': successful_segments,
            'failed_segments': failed_segments,
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
            }
        }
        
        # Save results
        output_path = landmarks_dir / "all_segments_clustered.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        labels_path = landmarks_dir / "all_segments_clustered.npy"
        np.save(labels_path, labels)
        
        print("="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Total segments processed: {len(successful_segments)}")
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
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR during clustering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

