"""
Video effects nodes for WAN Vace Pipeline
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import folder_paths
import json
from ..utils import get_video_info


class WANVaceOutpainting:
    """Expand or crop video canvas for outpainting workflows"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "canvas_width": ("INT", {
                    "default": 1920,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Width of the expanded canvas"
                }),
                "canvas_height": ("INT", {
                    "default": 1080,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Height of the expanded canvas"
                }),
                "video_x": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                    "tooltip": "X position of video within canvas (negative = shift left)"
                }),
                "video_y": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Y position of video within canvas (negative = shift up)"
                }),
                "feather_amount": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Feather edge pixels (0 = hard edge)"
                }),
                "padding_color": (["black", "white", "green", "magenta"], {
                    "default": "black",
                    "tooltip": "Color to use for padding areas"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("reference_frames", "preview_frames", "mask", "canvas_width", "canvas_height")
    FUNCTION = "expand_canvas"
    CATEGORY = "WAN Vace/Processing"
    
    def expand_canvas(self, frames, canvas_width, canvas_height, video_x, video_y, 
                     feather_amount, padding_color):
        """Expand or crop video canvas"""
        
        # Convert tensor to numpy array for processing
        batch_size, height, width, channels = frames.shape
        frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)
        
        print(f"\nOutpainting Configuration:")
        print(f"  Original video: {width}x{height}")
        print(f"  Canvas size: {canvas_width}x{canvas_height}")
        print(f"  Video position: ({video_x}, {video_y})")
        print(f"  Feather amount: {feather_amount}px")
        print(f"  Padding color: {padding_color}")
        
        # Determine if this is cropping-only (no padding needed)
        video_bounds = {
            'left': video_x,
            'top': video_y,
            'right': video_x + width,
            'bottom': video_y + height
        }
        
        canvas_bounds = {
            'left': 0,
            'top': 0,
            'right': canvas_width,
            'bottom': canvas_height
        }
        
        needs_padding = (
            canvas_bounds['left'] < video_bounds['left'] or
            canvas_bounds['top'] < video_bounds['top'] or
            canvas_bounds['right'] > video_bounds['right'] or
            canvas_bounds['bottom'] > video_bounds['bottom']
        )
        
        is_cropping_only = not needs_padding
        
        print(f"  Mode: {'Cropping only' if is_cropping_only else 'Expanding with padding'}")
        
        # Get padding color
        color_map = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "green": (0, 255, 0),
            "magenta": (255, 0, 255)
        }
        pad_color = color_map[padding_color]
        
        processed_frames = []
        mask_frames = []
        
        for i, frame_np in enumerate(frames_np):
            # Convert from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            if is_cropping_only:
                # Direct cropping without padding
                src_x = max(0, -video_x)
                src_y = max(0, -video_y)
                
                # Calculate crop region
                end_x = min(src_x + canvas_width, width)
                end_y = min(src_y + canvas_height, height)
                
                # Ensure valid crop region
                if end_x > src_x and end_y > src_y:
                    cropped = frame_bgr[src_y:end_y, src_x:end_x]
                    
                    # Resize if needed to match exact canvas size
                    if cropped.shape[0] != canvas_height or cropped.shape[1] != canvas_width:
                        cropped = cv2.resize(cropped, (canvas_width, canvas_height), 
                                           interpolation=cv2.INTER_LANCZOS4)
                else:
                    # Invalid crop region, use black frame
                    cropped = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                
                processed_frames.append(cropped)
                # No mask for cropping-only mode
                mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
                
            else:
                # Create expanded canvas with padding
                expanded = np.full((canvas_height, canvas_width, 3), pad_color, dtype=np.uint8)
                
                # Calculate copy regions
                src_x = max(0, -video_x)
                src_y = max(0, -video_y)
                dst_x = max(0, video_x)
                dst_y = max(0, video_y)
                
                copy_width = min(width - src_x, canvas_width - dst_x)
                copy_height = min(height - src_y, canvas_height - dst_y)
                
                # Copy video content to canvas
                if copy_width > 0 and copy_height > 0:
                    expanded[dst_y:dst_y+copy_height, dst_x:dst_x+copy_width] = \
                        frame_bgr[src_y:src_y+copy_height, src_x:src_x+copy_width]
                
                processed_frames.append(expanded)
                
                # Create mask (white for padding, black for original video)
                mask = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
                if copy_width > 0 and copy_height > 0:
                    mask[dst_y:dst_y+copy_height, dst_x:dst_x+copy_width] = 0
                
                # Apply feathering if requested
                if feather_amount > 0 and copy_width > 0 and copy_height > 0:
                    # Create a temporary mask for the video region
                    temp_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
                    temp_mask[dst_y:dst_y+copy_height, dst_x:dst_x+copy_width] = 255
                    
                    # Apply distance transform
                    dist_transform = cv2.distanceTransform(temp_mask, cv2.DIST_L2, 5)
                    
                    # Normalize and create feather gradient
                    feather_mask = np.clip(dist_transform / feather_amount, 0, 1)
                    
                    # Apply feather to mask
                    mask = ((1 - feather_mask) * 255).astype(np.uint8)
            
            mask_frames.append(mask)
        
        # Convert processed frames back to tensors
        processed_np = np.array(processed_frames)
        # Convert BGR back to RGB
        processed_rgb = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in processed_np])
        reference_tensor = torch.from_numpy(processed_rgb).float() / 255.0
        
        # Create preview frames (duplicate for consistency)
        preview_tensor = reference_tensor.clone()
        
        # Convert masks to tensor
        mask_np = np.array(mask_frames)
        mask_tensor = torch.from_numpy(mask_np).float() / 255.0
        
        # Add channel dimension to mask
        mask_tensor = mask_tensor.unsqueeze(-1)
        
        print(f"  Output shape: {reference_tensor.shape}")
        padding_pixels = int(torch.sum(mask_tensor).item())
        video_pixels = mask_tensor.numel() - padding_pixels
        print(f"  Padding pixels: {padding_pixels}, Video pixels: {video_pixels}")
        print(f"{'='*40}\n")
        
        return (reference_tensor, preview_tensor, mask_tensor, canvas_width, canvas_height)




