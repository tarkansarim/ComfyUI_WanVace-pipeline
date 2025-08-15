"""
WAN Match Batch Size Node - Automatically match batch sizes between images and masks
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union

class WANMatchBatchSize:
    """
    Automatically match batch sizes between images and masks by duplicating frames
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "mode": (["match_larger", "match_smaller", "match_images", "match_masks"], {"default": "match_larger"}),
                "duplication_mode": (["repeat_last", "repeat_first", "cycle"], {"default": "repeat_last"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("images", "masks", "batch_size")
    FUNCTION = "match_batch_size"
    CATEGORY = "WAN/utils"
    
    def match_batch_size(self, images: Optional[torch.Tensor] = None, 
                        masks: Optional[torch.Tensor] = None,
                        mode: str = "match_larger",
                        duplication_mode: str = "repeat_last") -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Match batch sizes between images and masks
        
        Args:
            images: Input images tensor [B, H, W, C]
            masks: Input masks tensor [B, H, W]
            mode: How to determine target batch size
            duplication_mode: How to duplicate frames
            
        Returns:
            Tuple of (matched_images, matched_masks, batch_size)
        """
        
        # Handle None inputs
        if images is None and masks is None:
            print("[WANMatchBatchSize] Both inputs are None, returning empty")
            return (None, None, 0)
        
        if images is None:
            batch_size = masks.shape[0] if masks is not None else 0
            print(f"[WANMatchBatchSize] Images is None, returning masks with batch size {batch_size}")
            return (None, masks, batch_size)
            
        if masks is None:
            batch_size = images.shape[0] if images is not None else 0
            print(f"[WANMatchBatchSize] Masks is None, returning images with batch size {batch_size}")
            return (images, None, batch_size)
        
        # Get batch sizes
        image_batch = images.shape[0]
        mask_batch = masks.shape[0]
        
        print(f"[WANMatchBatchSize] Input batch sizes - Images: {image_batch}, Masks: {mask_batch}")
        
        # Determine target batch size based on mode
        if mode == "match_larger":
            target_batch = max(image_batch, mask_batch)
        elif mode == "match_smaller":
            target_batch = min(image_batch, mask_batch)
        elif mode == "match_images":
            target_batch = image_batch
        elif mode == "match_masks":
            target_batch = mask_batch
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        print(f"[WANMatchBatchSize] Target batch size: {target_batch} (mode: {mode})")
        
        # Match images
        if image_batch < target_batch:
            images = self._duplicate_frames(images, target_batch, duplication_mode, "images")
        elif image_batch > target_batch:
            images = images[:target_batch]
            print(f"[WANMatchBatchSize] Truncated images from {image_batch} to {target_batch}")
            
        # Match masks
        if mask_batch < target_batch:
            masks = self._duplicate_frames(masks, target_batch, duplication_mode, "masks")
        elif mask_batch > target_batch:
            masks = masks[:target_batch]
            print(f"[WANMatchBatchSize] Truncated masks from {mask_batch} to {target_batch}")
        
        return (images, masks, target_batch)
    
    def _duplicate_frames(self, tensor: torch.Tensor, target_size: int, mode: str, tensor_name: str) -> torch.Tensor:
        """
        Duplicate frames to reach target size
        
        Args:
            tensor: Input tensor to duplicate
            target_size: Target batch size
            mode: Duplication mode
            tensor_name: Name for logging
            
        Returns:
            Tensor with duplicated frames
        """
        current_size = tensor.shape[0]
        needed = target_size - current_size
        
        if needed <= 0:
            return tensor
            
        print(f"[WANMatchBatchSize] Duplicating {needed} frames for {tensor_name} (mode: {mode})")
        
        if mode == "repeat_last":
            # Repeat the last frame
            last_frame = tensor[-1:].repeat(needed, *([1] * (len(tensor.shape) - 1)))
            result = torch.cat([tensor, last_frame], dim=0)
            
        elif mode == "repeat_first":
            # Repeat the first frame
            first_frame = tensor[0:1].repeat(needed, *([1] * (len(tensor.shape) - 1)))
            result = torch.cat([tensor, first_frame], dim=0)
            
        elif mode == "cycle":
            # Cycle through all frames
            if needed >= current_size:
                # Need to repeat the entire sequence one or more times
                full_repeats = needed // current_size
                remainder = needed % current_size
                
                repeated = tensor.repeat(full_repeats + 1, *([1] * (len(tensor.shape) - 1)))
                result = torch.cat([tensor, repeated[:needed]], dim=0)
            else:
                # Just need a portion of the sequence
                result = torch.cat([tensor, tensor[:needed]], dim=0)
        else:
            raise ValueError(f"Unknown duplication mode: {mode}")
            
        print(f"[WANMatchBatchSize] Result shape for {tensor_name}: {result.shape}")
        return result


# Node registration
NODE_CLASS_MAPPINGS = {
    "WANMatchBatchSize": WANMatchBatchSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANMatchBatchSize": "WAN Match Batch Size",
}