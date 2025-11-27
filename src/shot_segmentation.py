import cv2
import mediapipe as mp
from pathlib import Path
import ffmpeg
import sys
import time
import shutil
from src.gpu_utils import check_gpu_available, setup_gpu_environment


def segment_by_hand_presence(video_path, min_active_frames=15, use_gpu=True):
    """
    Break video into segments containing actual hand activity.
    
    Args:
        video_path: Path to input video file
        min_active_frames: Minimum number of consecutive frames with hands
                          to consider a valid segment
        use_gpu: Attempt to use GPU acceleration (requires GPU support)
    
    Returns:
        segments: List of tuples (start_frame, end_frame)
        fps: Frames per second of the video
    """
    # Set up GPU environment if requested
    if use_gpu:
        gpu_available, gpu_status = setup_gpu_environment()
        print(f"GPU Status: {gpu_status}")
        if not gpu_available:
            print("Note: MediaPipe Python GPU support requires TensorFlow Lite GPU delegate.")
            print("Standard MediaPipe packages may not include GPU support.")
            print("For full GPU acceleration, consider using MediaPipe C++ API.")
    else:
        gpu_available = False
    
    # Use higher model complexity for better accuracy (can be slower)
    # model_complexity: 0=fastest, 1=balanced, 2=most accurate
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # Balanced mode - good accuracy/speed tradeoff
    )
    

    # Try to use GPU-accelerated video backend if available
    cap = None
    if use_gpu:
        # Try CUDA backend first (for NVIDIA GPUs)
        try:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            # Set backend preference (OpenCV will try GPU if available)
            cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_FFMPEG)
        except:
            pass
    
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {total_frames} frames @ {fps:.2f} FPS")
    print("Processing frames...")

    segments = []
    active_start = None
    frame_idx = 0
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1.0  # Update progress every second

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands_present = results.multi_hand_landmarks is not None

        if hands_present and active_start is None:
            active_start = frame_idx

        if not hands_present and active_start is not None:
            if frame_idx - active_start >= min_active_frames:
                segments.append((active_start, frame_idx))
            active_start = None

        frame_idx += 1
        
        # Update progress periodically
        current_time = time.time()
        if current_time - last_update_time >= update_interval or frame_idx == total_frames:
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            elapsed = current_time - start_time
            fps_processing = frame_idx / elapsed if elapsed > 0 else 0
            
            # Estimate remaining time
            if fps_processing > 0 and total_frames > 0:
                remaining_frames = total_frames - frame_idx
                eta_seconds = remaining_frames / fps_processing
                eta_str = f", ETA: {eta_seconds:.1f}s"
            else:
                eta_str = ""
            
            # Print progress (overwrite same line)
            print(f"\r  Progress: {frame_idx}/{total_frames} frames ({progress:.1f}%) | "
                  f"Speed: {fps_processing:.1f} fps | Segments: {len(segments)}{eta_str}", 
                  end='', flush=True)
            last_update_time = current_time
    
    # Final progress update
    elapsed_total = time.time() - start_time
    print(f"\r  Progress: {frame_idx}/{total_frames} frames (100.0%) | "
          f"Completed in {elapsed_total:.1f}s | Segments found: {len(segments)}", 
          flush=True)

    # Handle case where video ends with hands still present
    if active_start is not None and frame_idx - active_start >= min_active_frames:
        segments.append((active_start, frame_idx))

    cap.release()
    mp_hands.close()
    
    return segments, fps


def check_ffmpeg_available():
    """Check if ffmpeg is available in the system PATH."""
    return shutil.which('ffmpeg') is not None


def save_segments(video_path, segments, fps, out_dir="data/segments"):
    """
    Export video segments to individual files using ffmpeg.
    
    Args:
        video_path: Path to input video file
        segments: List of tuples (start_frame, end_frame)
        fps: Frames per second of the video
        out_dir: Output directory for segment files
    
    Returns:
        List of output file paths
    
    Raises:
        RuntimeError: If ffmpeg is not installed or not found in PATH
    """
    # Check if ffmpeg is available
    if not check_ffmpeg_available():
        error_msg = (
            "\n" + "="*70 + "\n"
            "ERROR: ffmpeg is not installed or not found in PATH.\n\n"
            "To install ffmpeg on Windows:\n"
            "  1. Download from: https://ffmpeg.org/download.html\n"
            "  2. Or use chocolatey: choco install ffmpeg\n"
            "  3. Or use winget: winget install ffmpeg\n"
            "  4. Or use conda: conda install -c conda-forge ffmpeg\n\n"
            "After installation, make sure ffmpeg is in your system PATH.\n"
            "You can verify by running: ffmpeg -version\n"
            + "="*70
        )
        raise RuntimeError(error_msg)
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    output_paths = []
    total_segments = len(segments)
    
    if total_segments == 0:
        print("No segments to export.")
        return output_paths
    
    print(f"\nExporting {total_segments} segments...")
    
    for i, (start, end) in enumerate(segments):
        ss = start / fps
        t = (end - start) / fps
        
        out_path = str(Path(out_dir) / f"segment_{i:03d}.mp4")
        
        try:
            (
                ffmpeg
                .input(video_path, ss=ss, t=t)
                .output(out_path, c="copy")
                .run(overwrite_output=True, quiet=True)
            )
            output_paths.append(out_path)
            progress = ((i + 1) / total_segments * 100)
            print(f"\r  Exporting: {i + 1}/{total_segments} ({progress:.1f}%)", end='', flush=True)
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            print(f"\nError processing segment {i}: {error_msg}")
            continue
        except FileNotFoundError:
            # This shouldn't happen if check_ffmpeg_available() passed, but handle it anyway
            raise RuntimeError(
                "ffmpeg executable not found. Please ensure ffmpeg is installed "
                "and available in your system PATH."
            )
    
    print(f"\r  Exporting: {total_segments}/{total_segments} (100.0%) - Complete!", flush=True)
    
    return output_paths

