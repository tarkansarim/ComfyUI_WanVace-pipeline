"""
WAN Mask Viewer Node - Preview masks with overlay on images
"""

# Import torch only when needed
torch = None

class WANVaceMaskViewer:
    """
    Preview masks with overlay on images
    """
    
    @staticmethod
    def IS_CHANGED(images, masks, opacity=0.5, color="red", **kwargs):
        """ComfyUI caching: Return hash of inputs to detect changes"""
        import hashlib
        
        # Create a hash based on tensor shapes and other inputs since we can't hash tensors directly
        images_shape = tuple(images.shape) if hasattr(images, 'shape') else str(images)
        masks_shape = tuple(masks.shape) if hasattr(masks, 'shape') else str(masks)
        
        input_string = f"{images_shape}_{masks_shape}_{opacity}_{color}"
        input_hash = hashlib.md5(input_string.encode()).hexdigest()
        
        print(f"[WANVaceMaskViewer] IS_CHANGED hash: {input_hash[:8]}... (images: {images_shape}, masks: {masks_shape})")
        return input_hash
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "color": (["red", "green", "blue", "white", "black"], {"default": "red"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview_masks"
    CATEGORY = "WAN/mask"
    
    def preview_masks(self, images, masks, opacity=0.5, color="red"):
        """
        Create preview images with mask overlay
        """
        # Import torch when function is called
        global torch
        if torch is None:
            import torch
        
        # Check for cached output to avoid reprocessing
        # For tensors, we use shape and parameters as cache key since tensor content comparison is expensive
        images_shape = tuple(images.shape) if hasattr(images, 'shape') else str(images)
        masks_shape = tuple(masks.shape) if hasattr(masks, 'shape') else str(masks)
        current_inputs = (images_shape, masks_shape, opacity, color)
        
        if hasattr(self, '_last_inputs') and self._last_inputs == current_inputs:
            if hasattr(self, '_cached_output'):
                print(f"[WANVaceMaskViewer] Using cached output (inputs unchanged)")
                return self._cached_output
        
        print(f"[WANVaceMaskViewer] Processing mask overlay (inputs changed or no cache)")
        # Color mappings
        colors = {
            "red": [1.0, 0.0, 0.0],
            "green": [0.0, 1.0, 0.0],
            "blue": [0.0, 0.0, 1.0],
            "white": [1.0, 1.0, 1.0],
            "black": [0.0, 0.0, 0.0],
        }
        
        overlay_color = torch.tensor(colors[color], device=images.device)
        
        # Ensure masks have the right shape
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)
        
        # Create preview
        preview_list = []
        for i in range(images.shape[0]):
            img = images[i].clone()
            
            if i < masks.shape[0]:
                mask = masks[i].unsqueeze(-1)  # Add channel dimension
                # Apply colored overlay where mask is active
                img = img * (1 - mask * opacity) + overlay_color * mask * opacity
            
            preview_list.append(img)
        
        preview = torch.stack(preview_list)
        
        # Cache the result for future use
        result = (preview,)
        self._last_inputs = current_inputs
        self._cached_output = result
        return result


# Node mappings
NODE_CLASS_MAPPINGS = {
    "WANVaceMaskViewer": WANVaceMaskViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANVaceMaskViewer": "WAN Vace Mask Viewer",
}