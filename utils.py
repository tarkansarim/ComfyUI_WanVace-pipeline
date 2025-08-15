"""
Utility functions for WAN Vace Pipeline nodes
"""

import cv2
import numpy as np
import torch
from pathlib import Path


def tensor_to_cv2(tensor):
    """Convert ComfyUI tensor to OpenCV format
    Input: tensor [B, H, W, C] in range [0, 1]
    Output: numpy array [H, W, C] in range [0, 255] BGR format
    """
    # Take first image if batch
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    
    # Convert to numpy and scale to 0-255
    image = tensor.cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    
    return image


def cv2_to_tensor(image):
    """Convert OpenCV image to ComfyUI tensor format
    Input: numpy array [H, W, C] in BGR format with range [0, 255]
    Output: tensor [1, H, W, C] in RGB format with range [0, 1]
    """
    # Convert BGR to RGB
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    
    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(image).unsqueeze(0)
    
    return tensor


def frames_to_tensor(frames):
    """Convert list of OpenCV frames to ComfyUI tensor - OPTIMIZED VERSION
    Input: list of numpy arrays [H, W, C] in BGR format
    Output: tensor [B, H, W, C] in RGB format with range [0, 1]
    """
    if not frames:
        return None
    
    # Stack all frames into a single numpy array first
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)
    
    # Convert BGR to RGB and normalize in one operation
    # Using [..., ::-1] to reverse the last dimension (BGR to RGB)
    frames_rgb = frames[..., ::-1].astype(np.float32) / 255.0
    
    # Convert to tensor in one operation
    return torch.from_numpy(frames_rgb)


def get_video_info(video_path):
    """Get video information
    Returns: (frame_count, fps, width, height, duration)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return frame_count, fps, width, height, duration