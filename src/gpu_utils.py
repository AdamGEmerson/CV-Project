"""
GPU utilities for MediaPipe acceleration.

Note: MediaPipe Python GPU support is limited. For full GPU acceleration,
consider using MediaPipe C++ API or ensure TensorFlow Lite GPU delegate
is properly compiled and available.
"""
import os
import subprocess
import sys
import warnings
from contextlib import contextmanager


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, None


def check_gpu_available():
    """
    Check GPU availability for MediaPipe/TensorFlow Lite.
    
    Returns:
        tuple: (is_available: bool, status_message: str)
    """
    # Check NVIDIA GPU
    nvidia_available, nvidia_info = check_nvidia_gpu()
    if nvidia_available:
        return True, "NVIDIA GPU detected (MediaPipe may use GPU if TensorFlow Lite GPU delegate is available)"
    
    # Check for CUDA environment
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if cuda_path:
        return True, f"CUDA environment detected at {cuda_path} (GPU support may be available)"
    
    return False, "CPU mode - GPU not detected. MediaPipe will use CPU inference."


def suppress_mediapipe_warnings():
    """
    Suppress MediaPipe's internal warnings and info messages.
    MediaPipe uses glog (Google Logging) which outputs to stderr.
    """
    # Set environment variable to reduce MediaPipe logging verbosity
    # GLOG_minloglevel: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
    # Set to 3 to suppress warnings (only show fatal errors)
    os.environ['GLOG_minloglevel'] = '3'
    
    # Also suppress Python warnings from MediaPipe
    warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')


@contextmanager
def suppress_stderr():
    """
    Context manager to temporarily suppress stderr output.
    Useful for suppressing MediaPipe's verbose logging during initialization.
    """
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def setup_gpu_environment():
    """
    Set up environment variables for GPU acceleration.
    These help TensorFlow Lite use GPU if available.
    Also suppresses MediaPipe warnings.
    """
    # Suppress MediaPipe warnings
    suppress_mediapipe_warnings()
    
    # Enable GPU memory growth (prevents TensorFlow from allocating all GPU memory)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Set CUDA visible devices (if needed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    
    # Note: MediaPipe Python uses TensorFlow Lite, which requires
    # the GPU delegate to be compiled separately. Standard MediaPipe
    # Python packages may not include GPU support.
    
    return check_gpu_available()

