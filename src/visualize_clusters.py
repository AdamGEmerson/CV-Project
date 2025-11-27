"""
Visualize clusters by overlaying cluster labels on video frames.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont


def create_cluster_labeled_video(video_path, cluster_json_path, output_path=None, fps=None):
    """
    Create a video with cluster labels overlaid on each frame.
    
    Args:
        video_path: Path to input video segment
        cluster_json_path: Path to clustering results JSON file
        output_path: Path to save output video (if None, auto-generates)
        fps: Frames per second (if None, reads from video)
    
    Returns:
        Path to saved video file
    """
    video_path = Path(video_path)
    cluster_json_path = Path(cluster_json_path)
    
    if output_path is None:
        output_path = cluster_json_path.parent / f"{video_path.stem}_clustered_labels.mp4"
    else:
        output_path = Path(output_path)
    
    # Load clustering results
    with open(cluster_json_path, 'r') as f:
        cluster_data = json.load(f)
    
    # Create frame to cluster mapping
    frame_to_cluster = {frame['frame']: frame['cluster'] for frame in cluster_data['clustered_frames']}
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = fps if fps else cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))
    
    print(f"Creating cluster-labeled video: {total_frames} frames @ {video_fps:.2f} FPS")
    print(f"Output: {output_path}")
    
    # Font settings - 25% bigger (1.2 * 1.25 = 1.5)
    font_scale = 1.5
    thickness = 3
    
    # Try to load Instrument Serif font, fallback to default serif
    try:
        # Try common system font paths for Instrument Serif
        font_paths = [
            "C:/Windows/Fonts/instrserif.ttf",
            "C:/Windows/Fonts/instrserif-regular.ttf",
            "/usr/share/fonts/truetype/instrserif.ttf",
        ]
        font_pil = None
        for path in font_paths:
            if Path(path).exists():
                font_pil = ImageFont.truetype(path, int(font_scale * 30))
                break
        
        if font_pil is None:
            # Fallback to default serif font
            font_pil = ImageFont.truetype("arial.ttf", int(font_scale * 30))
    except:
        # If all else fails, use default font
        font_pil = ImageFont.load_default()
    
    # Padding and margin settings
    padding = 15  # Internal padding around text
    margin = 80   # External margin from edges (80px both x and y)
    corner_radius = 10  # Rounded corner radius
    bg_alpha = 0.7  # Background transparency (0.0 = fully transparent, 1.0 = opaque)
    
    # Static label text for left side - read from clustering results if available
    try:
        with open(cluster_json_path, 'r') as f:
            cluster_meta = json.load(f)
        pca_comp = cluster_meta.get('pca_components', 'N/A')
        min_samples = cluster_meta.get('min_samples', cluster_meta.get('min_cluster_size', 'N/A'))
        static_label_text = f"PCA={pca_comp}, Min-Samples={min_samples}, Smoothed Landmarks"
    except:
        static_label_text = "Min-Samples=50, No Temporal Smoothing"
    
    frame_idx = 0
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get cluster label for this frame
        cluster_id = frame_to_cluster.get(frame_idx, -1)
        
        # Prepare label text
        if cluster_id == -1:
            label_text = "Noise"
            text_color = (128, 128, 128)  # Gray for noise
        else:
            label_text = f"Pose {cluster_id}"
            # Use different colors for different clusters
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (255, 165, 0),  # Orange
                (128, 0, 128),  # Purple
            ]
            text_color = colors[cluster_id % len(colors)]
        
        # Convert frame to PIL Image for better text rendering
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Create temporary image for overlays
        overlay_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay_img)
        
        # === Draw static label on the left side ===
        static_bbox = overlay_draw.textbbox((0, 0), static_label_text, font=font_pil)
        static_text_width = static_bbox[2] - static_bbox[0]
        static_text_height = static_bbox[3] - static_bbox[1]
        
        # Position static label on left side with margin
        static_x = margin
        static_y = margin
        
        # Calculate static label background rectangle (vertically centered)
        static_bg_x1 = static_x - padding
        static_bg_y1 = static_y - padding
        static_bg_x2 = static_x + static_text_width + padding
        static_bg_y2 = static_y + static_text_height + padding
        
        # Draw rounded rectangle for static label
        overlay_draw.rounded_rectangle(
            [(static_bg_x1, static_bg_y1), (static_bg_x2, static_bg_y2)],
            radius=corner_radius,
            fill=(0, 0, 0, int(255 * bg_alpha))
        )
        
        # Draw static label text (white color, vertically centered)
        overlay_draw.text((static_x, static_y), static_label_text, fill=(255, 255, 255, 255), font=font_pil)
        
        # === Draw cluster label on the right side ===
        # Get text size using PIL font
        bbox = overlay_draw.textbbox((0, 0), label_text, font=font_pil)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate background rectangle (with margin)
        bg_x1 = width - text_width - padding * 2 - margin
        bg_y1 = margin
        bg_x2 = width - margin
        bg_y2 = margin + text_height + padding * 2
        
        # Calculate text position (vertically centered in the box)
        text_x = bg_x1 + padding
        text_y = bg_y1 + padding  # Already centered since padding is equal top and bottom
        
        # Draw rounded rectangle background with transparency
        overlay_draw.rounded_rectangle(
            [(bg_x1, bg_y1), (bg_x2, bg_y2)],
            radius=corner_radius,
            fill=(0, 0, 0, int(255 * bg_alpha))  # Black with transparency
        )
        
        # Draw text (convert BGR to RGB for PIL)
        text_color_rgb = (text_color[2], text_color[1], text_color[0])
        overlay_draw.text((text_x, text_y), label_text, fill=text_color_rgb, font=font_pil)
        
        # Composite the overlay onto the frame
        frame_pil = Image.alpha_composite(frame_pil.convert('RGBA'), overlay_img).convert('RGB')
        
        # Convert back to OpenCV format
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        # Write frame
        out.write(frame)
        
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
            
            frames_labeled = sum(1 for f in range(frame_idx) if f in frame_to_cluster)
            print(f"\r    Progress: {frame_idx}/{total_frames} ({progress:.1f}%) | "
                  f"Speed: {fps_processing:.1f} fps | Labeled: {frames_labeled}{eta_str}",
                  end='', flush=True)
            last_update_time = current_time
    
    # Cleanup
    cap.release()
    out.release()
    
    elapsed_total = time.time() - start_time
    frames_labeled = sum(1 for f in range(frame_idx) if f in frame_to_cluster)
    print(f"\r    Progress: {frame_idx}/{total_frames} (100.0%) | "
          f"Completed in {elapsed_total:.1f}s | Labeled: {frames_labeled} frames",
          flush=True)
    
    return str(output_path)

