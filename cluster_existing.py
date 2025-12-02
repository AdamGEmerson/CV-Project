"""
Cluster existing JSON files (raw landmarks, smoothed, or normalized).
This script processes JSON files through the pipeline: smoothing -> normalization -> feature extraction -> clustering.
"""
import sys
from pathlib import Path
import time
import shutil
from src.hand_tracking import smooth_landmarks, save_landmarks
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


def find_existing_feature_files(landmarks_dir, use_smoothing=True):
    """
    Find all existing feature NPZ files.
    
    Args:
        landmarks_dir: Directory containing feature files
        use_smoothing: Whether to look for smoothed features
    
    Returns:
        List of paths to feature NPZ files
    """
    landmarks_dir = Path(landmarks_dir)
    
    if use_smoothing:
        pattern = "*_smoothed_normalized_features_distance_matrix.npz"
    else:
        pattern = "*_normalized_features_distance_matrix.npz"
    
    feature_files = sorted(landmarks_dir.glob(pattern))
    return feature_files


def find_json_files_to_process(landmarks_dir, use_smoothing=True):
    """
    Find all JSON files that need processing (raw landmarks, smoothed, or normalized).
    Returns files in the order they should be processed.
    
    Args:
        landmarks_dir: Directory containing JSON files
        use_smoothing: Whether to use smoothing
    
    Returns:
        List of tuples (clip_id, raw_json_path, smoothed_json_path, normalized_json_path, feature_path)
    """
    landmarks_dir = Path(landmarks_dir)
    
    # Find all JSON files (excluding already processed ones)
    all_json_files = sorted(landmarks_dir.glob("*.json"))
    
    # Filter out processed files (smoothed_normalized, normalized, feature metadata)
    raw_json_files = [
        f for f in all_json_files 
        if not any(x in f.name for x in ['_smoothed_normalized', '_normalized', '_features_distance_matrix'])
    ]
    
    result = []
    for raw_json in raw_json_files:
        clip_id = raw_json.stem
        
        # Determine paths
        if use_smoothing:
            smoothed_json = landmarks_dir / f"{clip_id}_smoothed.json"
            normalized_json = landmarks_dir / f"{clip_id}_smoothed_normalized.json"
            feature_file = landmarks_dir / f"{clip_id}_smoothed_normalized_features_distance_matrix.npz"
        else:
            smoothed_json = None
            normalized_json = landmarks_dir / f"{clip_id}_normalized.json"
            feature_file = landmarks_dir / f"{clip_id}_normalized_features_distance_matrix.npz"
        
        result.append((clip_id, raw_json, smoothed_json, normalized_json, feature_file))
    
    return result


def combine_hands_features(features_path):
    """
    Load combined features from consolidated NPZ file.
    
    Args:
        features_path: Path to consolidated features NPZ file
    
    Returns:
        Tuple of (combined_features_array, common_frames)
    """
    data = load_features_from_npz(features_path)
    
    # load_features_from_npz already extracts combined_features and puts it in 'features'
    if 'features' in data and len(data['features']) > 0:
        return data['features'], list(data['frame_indices'])
    
    # Fallback: old format (separate hand0/hand1 files) - shouldn't happen with new code
    return None, None


def main():
    landmarks_dir = Path("data/landmarks")
    use_smoothing = True
    pca_components = 11
    min_cluster_size = 40
    min_samples = 15
    
    print("="*60)
    print("CLUSTERING FROM EXISTING FILES")
    print("="*60)
    print(f"Looking for {'smoothed' if use_smoothing else 'non-smoothed'} feature files...")
    print()
    
    # Step 1: Find existing feature files
    feature_files = find_existing_feature_files(landmarks_dir, use_smoothing=use_smoothing)
    print(f"Found {len(feature_files)} existing feature files")
    
    # Step 2: If no feature files, process JSON files through the pipeline
    if len(feature_files) == 0:
        print("\nNo feature files found. Processing JSON files through pipeline...")
        json_files_to_process = find_json_files_to_process(landmarks_dir, use_smoothing=use_smoothing)
        print(f"Found {len(json_files_to_process)} JSON files to process")
        
        if len(json_files_to_process) == 0:
            print("\nERROR: No JSON files found to process!")
            return
        
        # Process each JSON file: smoothing -> normalization -> feature extraction
        print("\nProcessing JSON files...")
        for clip_id, raw_json, smoothed_json, normalized_json, feature_file in json_files_to_process:
            if feature_file.exists():
                print(f"  [SKIP] {clip_id}: Features already exist")
                feature_files.append(feature_file)
                continue
            
            print(f"  Processing {clip_id}...")
            
            try:
                # Step 1: Smooth (if needed and use_smoothing is True)
                input_for_normalization = raw_json
                if use_smoothing:
                    if smoothed_json.exists():
                        print(f"    [OK] Smoothed file exists")
                        input_for_normalization = smoothed_json
                    else:
                        print(f"    Smoothing landmarks...")
                        # Load JSON file and convert to format expected by smooth_landmarks
                        with open(raw_json, 'r') as f:
                            data = json.load(f)
                        
                        # Convert JSON format to landmarks list format: [(frame_idx, hands, hand_labels), ...]
                        landmarks_list = []
                        for entry in data.get('landmarks', []):
                            frame_idx = entry['frame']
                            hands = entry.get('hands', [])
                            hand_labels = entry.get('hand_labels', ['unknown'] * len(hands))
                            landmarks_list.append((frame_idx, hands, hand_labels))
                        
                        fps = data.get('fps', 25.0)
                        smoothed_landmarks = smooth_landmarks(landmarks_list, window_length=11, polyorder=2)
                        save_landmarks(smoothed_landmarks, fps, smoothed_json, format='json')
                        print(f"    [OK] Smoothed and saved")
                        input_for_normalization = smoothed_json
                
                # Step 2: Normalize
                if normalized_json.exists():
                    print(f"    [OK] Normalized file exists")
                else:
                    print(f"    Normalizing landmarks...")
                    normalize_from_json(str(input_for_normalization), output_path=str(normalized_json), scale_method='palm')
                    print(f"    [OK] Normalized and saved")
                
                # Step 3: Extract features
                print(f"    Extracting features...")
                results = extract_all_hands_features(
                    str(normalized_json), 
                    method='distance_matrix', 
                    output_path=str(feature_file)
                )
                if len(results['common_frames']) > 0:
                    feature_files.append(feature_file)
                    print(f"    [OK] Extracted {len(results['common_frames'])} frames")
                else:
                    print(f"    [WARN] No frames with both hands")
                    
            except Exception as e:
                print(f"    [FAIL] Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nTotal feature files available: {len(feature_files)}")
    
    if len(feature_files) == 0:
        print("\nERROR: No feature files available for clustering!")
        return
    
    # Step 3: Load all features and combine
    print("\n" + "="*60)
    print("LOADING FEATURES")
    print("="*60)
    
    all_combined_features = []
    all_frame_info = []
    successful_clips = []
    failed_clips = []
    
    for features_path in feature_files:
        clip_id = features_path.stem.replace("_smoothed_normalized_features_distance_matrix", "").replace("_normalized_features_distance_matrix", "")
        
        try:
            combined_features, common_frames = combine_hands_features(features_path)
            
            if combined_features is None or len(combined_features) == 0:
                print(f"  [WARN] {clip_id}: No features available")
                failed_clips.append(clip_id)
                continue
            
            # Load metadata to get source file for landmarks
            data = load_features_from_npz(features_path)
            metadata = data.get('metadata', {})
            source_file = metadata.get('source_file')
            
            # Try to find smoothed JSON for landmarks (original skeleton data)
            landmarks_lookup = {}
            if source_file:
                source_path = Path(source_file)
                if '_normalized.json' in source_path.name:
                    smoothed_path = source_path.parent / source_path.name.replace('_normalized.json', '_smoothed.json')
                    if smoothed_path.exists():
                        landmarks_lookup = load_frame_landmarks(str(smoothed_path))
                    else:
                        # Fallback to normalized file
                        landmarks_lookup = load_frame_landmarks(source_file)
                else:
                    landmarks_lookup = load_frame_landmarks(source_file)
            
            # Add to combined dataset
            for i, frame in enumerate(common_frames):
                all_combined_features.append(combined_features[i])
                all_frame_info.append({
                    'clip': clip_id,
                    'frame': int(frame),
                    'feature_idx': len(all_combined_features) - 1,
                    'hand_landmarks': landmarks_lookup.get(int(frame), [])
                })
            
            successful_clips.append(clip_id)
            print(f"  [OK] {clip_id}: {len(common_frames)} frames")
            
        except Exception as e:
            print(f"  [FAIL] {clip_id}: {e}")
            failed_clips.append(clip_id)
            continue
    
    print(f"\nTotal clips processed: {len(successful_clips)}")
    print(f"Failed clips: {len(failed_clips)}")
    print(f"Total frames (both hands): {len(all_combined_features)}")
    
    if len(all_combined_features) == 0:
        print("\nERROR: No features collected from any clip!")
        return
    
    # Step 4: Cluster all segments together
    print("\n" + "="*60)
    print("CLUSTERING ALL SEGMENTS TOGETHER")
    print("="*60)
    print(f"Total features: {len(all_combined_features)}")
    print(f"Feature dimension: {all_combined_features[0].shape[0]}")
    print(f"PCA components: {pca_components}")
    print(f"Min cluster size: {min_cluster_size}")
    print(f"Min samples: {min_samples}")
    print()
    
    all_combined_features = np.array(all_combined_features)
    
    start_time = time.time()
    try:
        cluster_result = cluster_features(
            all_combined_features,
            pca_components=pca_components,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        
        elapsed = time.time() - start_time
        
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
            'n_clips': len(successful_clips),
            'successful_clips': successful_clips,
            'failed_clips': failed_clips,
            'n_frames': len(all_combined_features),
            'n_clusters': cluster_result['n_clusters'],
            'n_noise': cluster_result['n_noise'],
            'pca_components': pca_components,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'use_smoothing': use_smoothing,
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
        
        # Copy to visualizer (if not already there)
        visualizer_public = Path("visualizer/public")
        visualizer_json = visualizer_public / "all_segments_clustered_with_xy.json"
        try:
            if output_path.resolve() != visualizer_json.resolve():
                visualizer_public.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_path, visualizer_json)
                print(f"Copied to visualizer: {visualizer_json}")
            else:
                print(f"Results file already in visualizer location")
        except Exception as e:
            print(f"Warning: Could not copy to visualizer/public/: {e}")
        
        print("="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Clustering time: {elapsed:.1f} seconds")
        print(f"Total clips processed: {len(successful_clips)}")
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
        
    except Exception as e:
        print(f"\nERROR during clustering: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

