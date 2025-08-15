"""
WAN Inpaint Conditioning Node for ComfyUI - Pure Denoising Approach
Provides inpainting through gradual denoising control without pixel masking.
The mask controls denoising strength: 0=preserve, 1=generate, gray=blend.
This avoids VAE encoding artifacts from traditional masked pixel approaches.
"""

import torch
import comfy.utils
import node_helpers
import hashlib
import uuid
import time


class WANInpaintConditioning:
    """
    Adds inpainting conditioning to WanWrapper embeddings using pure denoising approach.
    Unlike traditional inpainting, this does NOT modify pixels - the mask only controls
    denoising strength in the sampler for smooth, gradual inpainting without artifacts.
    
    Mask values:
    - 0 (black): Preserve original (0% denoising)
    - 1 (white): Full generation (100% denoising)
    - 0.5 (gray): 50% blend between original and generated
    """
    
    # CACHE ISOLATION: Prevent ComfyUI from caching/reusing outputs from other similar nodes
    NOT_IDEMPOTENT = False  # Enable caching but isolate by instance
    
    def __init__(self):
        # Create unique instance identifier for cache isolation
        self._instance_id = str(uuid.uuid4())[:8]
        self._instance_cache = {}
        print(f"🚀 WANInpaintConditioning: Initialized instance {self._instance_id}")
    
    @staticmethod
    def IS_CHANGED(**kwargs):
        """ComfyUI caching: Return hash of inputs to detect changes"""
        # Generate hash from all input parameters
        input_items = []
        
        # Handle each input type appropriately for hashing
        for key, value in kwargs.items():
            if key == "text_embeds" and isinstance(value, dict):
                # Hash text embed content
                text_content = str(sorted(value.items()))
                input_items.append(f"{key}:{hashlib.md5(text_content.encode()).hexdigest()[:8]}")
            elif key == "image_embeds" and isinstance(value, dict):  
                # Hash image embed content
                embed_content = str(sorted(value.items()))
                input_items.append(f"{key}:{hashlib.md5(embed_content.encode()).hexdigest()[:8]}")
            elif key == "vae":
                # Hash VAE identifier/type
                vae_id = str(type(value).__name__ + str(getattr(value, 'device', 'cpu')))
                input_items.append(f"{key}:{hashlib.md5(vae_id.encode()).hexdigest()[:8]}")
            elif key in ["pixels", "mask"] and hasattr(value, 'shape'):
                # Hash tensor content
                tensor_hash = WANInpaintConditioning.get_tensor_content_hash(value)
                input_items.append(f"{key}:{tensor_hash}")
            else:
                # Hash other values as strings
                input_items.append(f"{key}:{str(value)}")
        
        input_string = "|".join(sorted(input_items))
        input_hash = hashlib.md5(input_string.encode()).hexdigest()
        
        return input_hash
    
    @staticmethod
    def get_tensor_content_hash(tensor, context=""):
        """Generate hash from tensor content with execution context"""
        if tensor is None:
            return "none"
        
        # Convert to consistent format for hashing
        if hasattr(tensor, 'cpu'):
            tensor_data = tensor.cpu().detach()
        else:
            tensor_data = tensor
            
        # Create hash from shape, dtype, and content statistics
        content_parts = [
            f"shape:{tensor_data.shape}",
            f"dtype:{tensor_data.dtype}",
            f"mean:{tensor_data.float().mean().item():.6f}",
            f"std:{tensor_data.float().std().item():.6f}",
            f"min:{tensor_data.float().min().item():.6f}",
            f"max:{tensor_data.float().max().item():.6f}",
            context  # Include execution context
        ]
        
        content_string = "_".join(content_parts)
        return hashlib.md5(content_string.encode()).hexdigest()[:8]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "vae": ("WANVAE",),
                "pixels": ("IMAGE",),
                "mask": ("MASK",),
                "noise_mask": (
                    "BOOLEAN", 
                    {
                        "default": True, 
                        "tooltip": "Add a noise mask to the latent so sampling will only happen within the mask. Use with WANVideoSamplerInpaint for proper inpainting."
                    }
                ),
            }
        }
    
    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "WANVIDIMAGE_EMBEDS", "LATENT")
    RETURN_NAMES = ("text_embeds", "image_embeds", "latent")
    FUNCTION = "encode"
    CATEGORY = "WanVace/conditioning"
    
    def encode(self, text_embeds, image_embeds, vae, pixels, mask, noise_mask=True):
        # CACHE ISOLATION: Generate execution context for this specific node instance
        execution_context = f"{self._instance_id}"
        
        # Generate cache key from all inputs with execution context
        pixels_hash = self.get_tensor_content_hash(pixels, execution_context)
        mask_hash = self.get_tensor_content_hash(mask, execution_context)
        text_embeds_hash = hashlib.md5(str(sorted(text_embeds.items())).encode()).hexdigest()[:8]
        image_embeds_hash = hashlib.md5(str(sorted(image_embeds.items())).encode()).hexdigest()[:8]
        vae_hash = hashlib.md5(str(type(vae).__name__).encode()).hexdigest()[:8]
        
        cache_key = f"{pixels_hash}_{mask_hash}_{text_embeds_hash}_{image_embeds_hash}_{vae_hash}_{noise_mask}"
        
        # Check instance-level cache first
        if cache_key in self._instance_cache:
            cached_result, cached_info = self._instance_cache[cache_key]
            print(f"🚀 WANInpaintConditioning: Using instance cached output")
            print(f"  - Instance ID: {self._instance_id}")
            print(f"  - Execution context: {execution_context}")
            print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
            print(f"  - Cached info: {cached_info}")
            
            # Return cloned tensors to prevent modification of cached data
            cached_text_embeds, cached_image_embeds, cached_latent = cached_result
            
            # Deep clone the outputs to prevent cache corruption
            cloned_text_embeds = cached_text_embeds.copy() if isinstance(cached_text_embeds, dict) else cached_text_embeds
            cloned_image_embeds = cached_image_embeds.copy() if isinstance(cached_image_embeds, dict) else cached_image_embeds
            
            # Clone tensor content in image_embeds if present
            if isinstance(cloned_image_embeds, dict):
                for key, value in cloned_image_embeds.items():
                    if hasattr(value, 'clone'):
                        cloned_image_embeds[key] = value.clone().detach()
            
            # Clone latent tensors
            cloned_latent = {}
            for key, value in cached_latent.items():
                if hasattr(value, 'clone'):
                    cloned_latent[key] = value.clone().detach()
                else:
                    cloned_latent[key] = value
                    
            return (cloned_text_embeds, cloned_image_embeds, cloned_latent)
        
        print(f"🚀 WANInpaintConditioning: Computing output (cache miss)")
        print(f"  - Instance ID: {self._instance_id}")
        print(f"  - Execution context: {execution_context}")
        print(f"  - Input shapes: pixels={pixels.shape}, mask={mask.shape}")
        
        # PURE DENOISING APPROACH: No pixel masking at all
        # The mask will only be used to control denoising strength in the sampler
        
        # Ensure mask dimensions match pixels - make divisible by 8 for VAE
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
            size=(pixels.shape[1], pixels.shape[2]), 
            mode="bilinear"
        )
        
        # Keep original pixels unmodified
        orig_pixels = pixels
        
        # Crop to VAE-compatible dimensions if needed
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]
            orig_pixels = orig_pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        
        # Update image_embeds dimensions if they were changed by cropping
        if 'lat_h' in image_embeds and 'lat_w' in image_embeds:
            # Update latent dimensions to match what will be encoded
            # VAE downscales by factor of 8
            new_lat_h = pixels.shape[1] // 8
            new_lat_w = pixels.shape[2] // 8
            if image_embeds['lat_h'] != new_lat_h or image_embeds['lat_w'] != new_lat_w:
                print(f"[WANInpaintConditioning] Updating image_embeds dimensions after cropping:")
                print(f"  - Updated lat_h from {image_embeds['lat_h']} to {new_lat_h}")
                print(f"  - Updated lat_w from {image_embeds['lat_w']} to {new_lat_w}")
                image_embeds = image_embeds.copy()  # Make a copy before modifying
                image_embeds['lat_h'] = new_lat_h
                image_embeds['lat_w'] = new_lat_w
        
        # Log mask statistics for denoising control
        print(f"[WANInpaintConditioning] Denoising mask statistics:")
        print(f"  - Shape: {mask.shape}")
        print(f"  - Min/Max: {mask.min().item():.6f}/{mask.max().item():.6f}")
        print(f"  - Mean denoising strength: {mask.mean().item():.6f}")
        print(f"  - Unique values: {torch.unique(mask).numel()}")
        if torch.unique(mask).numel() > 10:
            print(f"  - ✅ Smooth gradient mask for gradual denoising")
        elif torch.unique(mask).numel() > 2:
            print(f"  - ⚠️ Limited gradient levels")
        else:
            print(f"  - ❌ Binary mask - may cause hard edges")
        
        # NO PIXEL MASKING - pixels remain unmodified for pure denoising approach
        
        # WanWrapper VAE handling
        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        
        # Move VAE to device for encoding
        vae.to(device)
        
        # Prepare pixels for WAN VAE format: [batch, channels, frames, height, width]
        # For pure denoising, we only encode the original unmasked pixels
        orig_pixels_prepared = orig_pixels.to(device=device, dtype=vae.dtype)
        
        # Handle different input dimensions
        if orig_pixels_prepared.dim() == 3:
            # Single image: [H, W, C] -> [1, C, 1, H, W]
            orig_pixels_prepared = orig_pixels_prepared.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(2)
        elif orig_pixels_prepared.dim() == 4:
            # Video: [T, H, W, C] -> [1, C, T, H, W]
            orig_pixels_prepared = orig_pixels_prepared.unsqueeze(0).permute(0, 4, 1, 2, 3)
        
        # Normalize to [-1, 1] for VAE
        orig_pixels_prepared = orig_pixels_prepared * 2.0 - 1.0
        
        # Encode original pixels only - no masked version needed for pure denoising
        orig_latent = vae.encode(orig_pixels_prepared, device)
        
        # For concat_latent, use the same original latent (no masking)
        concat_latent = orig_latent
        
        # Move VAE back to offload device
        vae.to(offload_device)
        comfy.model_management.soft_empty_cache()
        
        # Prepare output latent (mimics core ComfyUI)
        out_latent = {
            "samples": orig_latent
        }
        
        if noise_mask:
            out_latent["noise_mask"] = mask.to(device)
        
        # Add inpainting data to image_embeds for pure denoising approach
        modified_image_embeds = image_embeds.copy()
        modified_image_embeds["concat_latent_image"] = concat_latent  # Original unmasked latent
        # Pass mask for denoising strength control in sampler (not for pixel masking)
        # Mask values: 0=preserve (no denoise), 1=generate (full denoise), 0.5=blend
        modified_image_embeds["concat_mask"] = mask.to(device)
        
        # Text embeds pass through unchanged
        result = (text_embeds, modified_image_embeds, out_latent)
        
        # Cache the result for future use
        cache_info = f"pixels_shape={pixels.shape}, processed_mask_shape={mask.shape}"
        self._instance_cache[cache_key] = (result, cache_info)
        print(f"🚀 WANInpaintConditioning: Cached output for future use")
        print(f"  - Cache info: {cache_info}")
        
        return result


NODE_CLASS_MAPPINGS = {
    "WANInpaintConditioning": WANInpaintConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANInpaintConditioning": "WAN Inpaint Conditioning",
}