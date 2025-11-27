"""
Extract hand landmarks from video segments.
"""
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import json
from pathlib import Path
import time
from scipy.signal import savgol_filter
from src.gpu_utils import setup_gpu_environment


def extract_landmarks(video_path, use_gpu=True, save_format='json'):
    """
    Extract hand landmarks from a video segment.
    
    Args:
        video_path: Path to input video file
        use_gpu: Attempt to use GPU acceleration
        save_format: Format to save landmarks ('json' or 'numpy')
    
    Returns:
        landmarks: List of tuples (frame_idx, hands_data)
                  where hands_data is a list of hand landmarks
                  Each hand is a list of 21 (x, y, z) tuples
    """
    # Set up GPU environment if requested
    if use_gpu:
        setup_gpu_environment()
    
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    landmarks = []
    frame_idx = 0
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1.0

    print(f"  Processing {total_frames} frames @ {fps:.2f} FPS...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            hands = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract 21 landmarks: (x, y, z) coordinates
                hand_points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                hands.append(hand_points)
            landmarks.append((frame_idx, hands))
        else:
            # No hands detected in this frame
            landmarks.append((frame_idx, []))

        frame_idx += 1
        
        # Update progress periodically
        current_time = time.time()
        if current_time - last_update_time >= update_interval or frame_idx == total_frames:
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            elapsed = current_time - start_time
            fps_processing = frame_idx / elapsed if elapsed > 0 else 0
            
            if fps_processing > 0 and total_frames > 0:
                remaining_frames = total_frames - frame_idx
                eta_seconds = remaining_frames / fps_processing
                eta_str = f", ETA: {eta_seconds:.1f}s"
            else:
                eta_str = ""
            
            frames_with_hands = sum(1 for _, hands in landmarks if hands)
            print(f"\r    Progress: {frame_idx}/{total_frames} ({progress:.1f}%) | "
                  f"Speed: {fps_processing:.1f} fps | Frames with hands: {frames_with_hands}{eta_str}",
                  end='', flush=True)
            last_update_time = current_time

    cap.release()
    mp_hands.close()
    
    elapsed_total = time.time() - start_time
    frames_with_hands = sum(1 for _, hands in landmarks if hands)
    print(f"\r    Progress: {frame_idx}/{total_frames} (100.0%) | "
          f"Completed in {elapsed_total:.1f}s | Frames with hands: {frames_with_hands}",
          flush=True)
    
    return landmarks, fps


def save_landmarks(landmarks, fps, output_path, format='json'):
    """
    Save landmarks to file.
    
    Args:
        landmarks: List of tuples (frame_idx, hands_data)
        fps: Frames per second
        output_path: Path to save the landmarks file
        format: 'json' or 'numpy'
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert to JSON-serializable format
        landmarks_data = {
            "fps": fps,
            "total_frames": len(landmarks),
            "landmarks": [
                {
                    "frame": frame_idx,
                    "hands": hands  # List of hands, each hand is list of 21 (x,y,z) tuples
                }
                for frame_idx, hands in landmarks
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(landmarks_data, f, indent=2)
    
    elif format == 'numpy':
        # Save as numpy array: shape (num_frames, max_hands, 21, 3)
        # Pad to max 2 hands, use NaN for missing hands
        max_hands = 2
        num_frames = len(landmarks)
        landmarks_array = np.full((num_frames, max_hands, 21, 3), np.nan)
        
        for frame_idx, hands in landmarks:
            for hand_idx, hand_points in enumerate(hands[:max_hands]):
                landmarks_array[frame_idx, hand_idx] = np.array(hand_points)
        
        np.save(output_path, landmarks_array)
        
        # Also save metadata
        metadata_path = output_path.replace('.npy', '_metadata.json')
        metadata = {
            "fps": fps,
            "total_frames": num_frames,
            "max_hands": max_hands,
            "landmarks_per_hand": 21,
            "shape": list(landmarks_array.shape)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'numpy'")


def draw_landmarks_on_image(rgb_image, hand_landmarks_list):
    """
    Draw hand landmarks on an image.
    
    Args:
        rgb_image: RGB image (numpy array)
        hand_landmarks_list: List of hand landmarks (each is list of 21 (x,y,z) tuples)
    
    Returns:
        Annotated image with landmarks drawn
    """
    annotated_image = rgb_image.copy()
    
    # MediaPipe drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Convert landmarks to MediaPipe format
    for hand_landmarks_points in hand_landmarks_list:
        # Create a landmark list in MediaPipe format
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in hand_landmarks_points:
            landmark = hand_landmarks_proto.landmark.add()
            landmark.x = x
            landmark.y = y
            landmark.z = z
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    return annotated_image


def create_preview_image(video_path, output_path=None, frame_idx=None, use_gpu=True):
    """
    Create a preview image with landmarks overlaid.
    
    Args:
        video_path: Path to video segment
        output_path: Path to save preview image (if None, auto-generates)
        frame_idx: Specific frame to use (if None, uses middle frame with hands)
        use_gpu: Use GPU acceleration
    
    Returns:
        Path to saved preview image
    """
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = Path("data/landmarks") / f"{video_name}_preview.jpg"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Set up MediaPipe
    if use_gpu:
        setup_gpu_environment()
    
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine which frame to use
    if frame_idx is None:
        # Try to find a frame with hands, starting from middle
        target_frame = total_frames // 2
        frames_to_check = list(range(target_frame, total_frames)) + list(range(target_frame - 1, -1, -1))
    else:
        frames_to_check = [frame_idx]
    
    preview_frame = None
    preview_landmarks = None
    
    # Find a frame with hands
    for check_idx in frames_to_check:
        cap.set(cv2.CAP_PROP_POS_FRAMES, check_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Extract landmarks
            hands = []
            for hand_landmarks in results.multi_hand_landmarks:
                hand_points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                hands.append(hand_points)
            
            preview_frame = rgb_frame
            preview_landmarks = hands
            break
    
    # If no frame with hands found, use middle frame anyway
    if preview_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        if ret:
            preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_landmarks = []
    
    cap.release()
    mp_hands.close()
    
    # Draw landmarks on the frame
    if preview_frame is not None:
        annotated_image = draw_landmarks_on_image(preview_frame, preview_landmarks)
        
        # Convert back to BGR for saving
        bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # Save image
        cv2.imwrite(str(output_path), bgr_image)
        return str(output_path)
    else:
        raise ValueError("Could not extract frame from video")


def process_segment(video_path, output_dir="data/landmarks", use_gpu=True, save_format='json', create_preview=True):
    """
    Process a single video segment and extract landmarks.
    
    Args:
        video_path: Path to segment video file
        output_dir: Directory to save landmarks
        use_gpu: Use GPU acceleration
        save_format: 'json' or 'numpy'
        create_preview: Whether to create a preview image with landmarks
    
    Returns:
        Tuple of (landmarks_file_path, preview_image_path)
    """
    video_name = Path(video_path).stem
    extension = '.json' if save_format == 'json' else '.npy'
    output_path = Path(output_dir) / f"{video_name}{extension}"
    
    landmarks, fps = extract_landmarks(video_path, use_gpu=use_gpu, save_format=save_format)
    save_landmarks(landmarks, fps, output_path, format=save_format)
    
    preview_path = None
    if create_preview:
        preview_path = Path(output_dir) / f"{video_name}_preview.jpg"
        try:
            create_preview_image(video_path, output_path=preview_path, use_gpu=use_gpu)
        except Exception as e:
            print(f"    Warning: Could not create preview image: {e}")
            preview_path = None
    
    return str(output_path), str(preview_path) if preview_path else None


def _classify_hand(hand_points):
    """
    Classify a hand as 'left' or 'right' based on wrist position.
    Left hand typically has wrist x < 0.5, right hand has x > 0.5.
    
    Args:
        hand_points: List of 21 (x, y, z) tuples
    
    Returns:
        'left' or 'right'
    """
    if not hand_points:
        return None
    wrist_x = hand_points[0][0]  # Wrist is landmark 0
    return 'left' if wrist_x < 0.5 else 'right'


def smooth_landmarks(landmarks, window_length=7, polyorder=3):
    """
    Apply Savitzky-Golay filter to smooth hand landmarks temporally.
    
    This function properly handles missing hands by:
    1. Classifying each hand as left or right (not using raw index order)
    2. Maintaining separate smoothing buffers for left and right hands
    3. Clearing buffers when hands disappear (output NaNs)
    4. Restarting smoothing fresh when hands reappear (no drift)
    
    Args:
        landmarks: List of tuples (frame_idx, hands_data)
                  where hands_data is a list of hand landmarks
                  Each hand is a list of 21 (x, y, z) tuples
        window_length: Length of the filter window (must be odd, default=7)
        polyorder: Order of the polynomial used to fit samples (default=3)
    
    Returns:
        Smoothed landmarks in the same format as input
    """
    if len(landmarks) < window_length:
        # Not enough frames to smooth
        return landmarks
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure window_length doesn't exceed number of frames
    window_length = min(window_length, len(landmarks))
    # Ensure polyorder is less than window_length
    polyorder = min(polyorder, window_length - 1)
    
    if window_length <= polyorder:
        # Cannot smooth with these parameters
        return landmarks
    
    num_frames = len(landmarks)
    
    # Step 1: For each frame, determine {Left hand, Right hand, Missing}
    # Store as: frame_idx -> {'left': hand_points or None, 'right': hand_points or None}
    frame_hands = {}
    for frame_idx, (orig_frame_idx, hands) in enumerate(landmarks):
        frame_hands[frame_idx] = {'left': None, 'right': None}
        
        for hand_points in hands:
            hand_type = _classify_hand(hand_points)
            if hand_type:
                frame_hands[frame_idx][hand_type] = hand_points
    
    # Step 2: Maintain smoothing buffers separately for left and right
    # We'll process each hand type independently, tracking continuous segments
    smoothed_landmarks = []
    
    for hand_type in ['left', 'right']:
        # Extract hand data for this type: (num_frames, 21, 3)
        # Use NaN when hand is missing
        hand_data = np.full((num_frames, 21, 3), np.nan)
        
        for frame_idx in range(num_frames):
            hand_points = frame_hands[frame_idx][hand_type]
            if hand_points is not None:
                hand_data[frame_idx] = np.array(hand_points)
        
        # Find continuous segments where this hand is present
        valid_mask = ~np.isnan(hand_data[:, 0, 0])
        
        # Smooth each continuous segment independently
        # This ensures we restart smoothing when hand reappears
        i = 0
        while i < num_frames:
            if not valid_mask[i]:
                i += 1
                continue
            
            # Find the start and end of this continuous segment
            segment_start = i
            while i < num_frames and valid_mask[i]:
                i += 1
            segment_end = i
            
            segment_length = segment_end - segment_start
            
            if segment_length < window_length:
                # Segment too short to smooth, keep original
                continue
            
            # Extract segment data
            segment_data = hand_data[segment_start:segment_end]  # (segment_length, 21, 3)
            
            # Smooth each coordinate independently
            smoothed_segment = np.zeros_like(segment_data)
            
            for landmark_idx in range(21):
                for coord_idx in range(3):  # x, y, z
                    coord_series = segment_data[:, landmark_idx, coord_idx]
                    
                    # Apply Savitzky-Golay filter
                    try:
                        smoothed_coord = savgol_filter(
                            coord_series,
                            window_length=window_length,
                            polyorder=polyorder,
                            mode='nearest'
                        )
                        smoothed_segment[:, landmark_idx, coord_idx] = smoothed_coord
                    except ValueError:
                        # If smoothing fails, use original
                        smoothed_segment[:, landmark_idx, coord_idx] = coord_series
            
            # Write smoothed segment back
            hand_data[segment_start:segment_end] = smoothed_segment
        
        # Store smoothed data back to frame_hands
        for frame_idx in range(num_frames):
            if not np.all(np.isnan(hand_data[frame_idx])):
                hand_points = [(float(x), float(y), float(z)) 
                              for x, y, z in hand_data[frame_idx]]
                frame_hands[frame_idx][hand_type] = hand_points
            else:
                frame_hands[frame_idx][hand_type] = None
    
    # Step 3 & 4: Convert back to original format
    # When hand is missing, it won't appear in the output (empty list)
    for frame_idx, (orig_frame_idx, _) in enumerate(landmarks):
        hands = []
        for hand_type in ['left', 'right']:
            hand_points = frame_hands[frame_idx][hand_type]
            if hand_points is not None:
                hands.append(hand_points)
        smoothed_landmarks.append((orig_frame_idx, hands))
    
    return smoothed_landmarks


def create_landmark_video(video_path, output_path=None, use_gpu=True):
    """
    Create a video with hand landmarks overlaid on each frame.
    
    Args:
        video_path: Path to input video segment
        output_path: Path to save output video (if None, auto-generates)
        use_gpu: Use GPU acceleration
    
    Returns:
        Path to saved video file
    """
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = Path("data/landmarks") / f"{video_name}_landmarks.mp4"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Set up MediaPipe
    if use_gpu:
        setup_gpu_environment()
    
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"  Creating landmark video: {total_frames} frames @ {fps:.2f} FPS")
    print(f"  Output: {output_path}")
    
    frame_idx = 0
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = mp_hands.process(rgb_frame)
        
        # Draw landmarks on frame
        annotated_frame = rgb_frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Convert back to BGR for video writer
        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
        
        frame_idx += 1
        
        # Update progress
        current_time = time.time()
        if current_time - last_update_time >= update_interval or frame_idx == total_frames:
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            elapsed = current_time - start_time
            fps_processing = frame_idx / elapsed if elapsed > 0 else 0
            
            if fps_processing > 0 and total_frames > 0:
                remaining_frames = total_frames - frame_idx
                eta_seconds = remaining_frames / fps_processing
                eta_str = f", ETA: {eta_seconds:.1f}s"
            else:
                eta_str = ""
            
            print(f"\r    Progress: {frame_idx}/{total_frames} ({progress:.1f}%) | "
                  f"Speed: {fps_processing:.1f} fps{eta_str}", end='', flush=True)
            last_update_time = current_time
    
    # Cleanup
    cap.release()
    out.release()
    mp_hands.close()
    
    elapsed_total = time.time() - start_time
    print(f"\r    Progress: {frame_idx}/{total_frames} (100.0%) | "
          f"Completed in {elapsed_total:.1f}s", flush=True)
    
    return str(output_path)


def create_landmark_video_from_json(video_path, landmarks_json_path, output_path=None):
    """
    Create a video with hand landmarks overlaid from pre-saved landmarks JSON.
    
    Args:
        video_path: Path to input video segment
        landmarks_json_path: Path to JSON file containing landmarks
        output_path: Path to save output video (if None, auto-generates)
    
    Returns:
        Path to saved video file
    """
    # Load landmarks from JSON
    with open(landmarks_json_path, 'r') as f:
        landmarks_data = json.load(f)
    
    fps = landmarks_data.get('fps', 30.0)
    landmarks_list = landmarks_data.get('landmarks', [])
    
    # Create a dictionary mapping frame_idx to hands
    landmarks_dict = {entry['frame']: entry['hands'] for entry in landmarks_list}
    
    if output_path is None:
        video_name = Path(video_path).stem
        json_name = Path(landmarks_json_path).stem
        # Include "smoothed" in filename if it's in the JSON name
        if 'smoothed' in json_name:
            output_path = Path("data/landmarks") / f"{video_name}_landmarks_smoothed.mp4"
        else:
            output_path = Path("data/landmarks") / f"{video_name}_landmarks_from_json.mp4"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))
    
    print(f"  Creating landmark video from JSON: {total_frames} frames @ {video_fps:.2f} FPS")
    print(f"  Landmarks file: {landmarks_json_path}")
    print(f"  Output: {output_path}")
    
    frame_idx = 0
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1.0
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get landmarks for this frame
        hands = landmarks_dict.get(frame_idx, [])
        
        # Draw landmarks on frame
        annotated_frame = rgb_frame.copy()
        if hands:
            for hand_points in hands:
                # Convert to MediaPipe format
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                for x, y, z in hand_points:
                    landmark = hand_landmarks_proto.landmark.add()
                    landmark.x = x
                    landmark.y = y
                    landmark.z = z
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Convert back to BGR for video writer
        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
        
        frame_idx += 1
        
        # Update progress
        current_time = time.time()
        if current_time - last_update_time >= update_interval or frame_idx == total_frames:
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            elapsed = current_time - start_time
            fps_processing = frame_idx / elapsed if elapsed > 0 else 0
            
            if fps_processing > 0 and total_frames > 0:
                remaining_frames = total_frames - frame_idx
                eta_seconds = remaining_frames / fps_processing
                eta_str = f", ETA: {eta_seconds:.1f}s"
            else:
                eta_str = ""
            
            print(f"\r    Progress: {frame_idx}/{total_frames} ({progress:.1f}%) | "
                  f"Speed: {fps_processing:.1f} fps{eta_str}", end='', flush=True)
            last_update_time = current_time
    
    # Cleanup
    cap.release()
    out.release()
    
    elapsed_total = time.time() - start_time
    print(f"\r    Progress: {frame_idx}/{total_frames} (100.0%) | "
          f"Completed in {elapsed_total:.1f}s", flush=True)
    
    return str(output_path)

