"""
WAN VACE Loop Encoder Node
Handles multi-batch VACE encoding for long videos with automatic reference frame management
"""

import torch
import math
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
import folder_paths
import comfy.model_management as mm
from importlib import import_module


class WANVACELoopEncoder:
    """
    Encodes long video sequences into VACE embeds with automatic batching and reference frame handling.
    This node sits before WANAutoLoopSampler and prepares the VACE context for multi-batch processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("WANVAE",),
                "controlnet_images": ("IMAGE",),
                "controlnet_masks": ("MASK",),
                "context_frames": ("INT", {
                    "default": 81,
                    "min": 2,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of frames per batch (WAN2.1 max: 81)"
                }),
                "overlap_frames": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Number of reference frames between batches"
                }),
                "vace_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "VACE encoding scale"
                }),
                "vace_start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Start percent for VACE"
                }),
                "vace_end_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "End percent for VACE"
                }),
                "has_ref": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether initial reference frames are provided"
                }),
                "verbose_logging": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed logging"
                }),
            },
            "optional": {
                "initial_ref_images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "STRING")
    RETURN_NAMES = ("vace_embeds", "batch_info")
    FUNCTION = "encode_batches"
    CATEGORY = "WAN Vace/Encoding"
    DESCRIPTION = "Encodes long videos into VACE embeds with automatic batching and reference frame management"
    
    def __init__(self):
        # Import WanWrapper nodes for orchestration
        self.nodes_available = False
        try:
            # Try to import WanWrapper nodes
            try:
                wan_wrapper_module = import_module("custom_nodes.ComfyUI-WanVideoWrapper.nodes")
                self.WanVideoVACEEncode = wan_wrapper_module.WanVideoVACEEncode
                self.WanVideoDecode = wan_wrapper_module.WanVideoDecode
                
                # Create instances
                self.vace_encoder = self.WanVideoVACEEncode()
                self.video_decoder = self.WanVideoDecode()
                
                self.nodes_available = True
                print("WANVACELoopEncoder: Successfully imported WanWrapper nodes")
            except Exception as e:
                # Try alternate import path
                wan_wrapper_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ComfyUI-WanVideoWrapper")
                if os.path.exists(wan_wrapper_path) and wan_wrapper_path not in sys.path:
                    sys.path.insert(0, wan_wrapper_path)
                    from nodes import WanVideoVACEEncode, WanVideoDecode
                    self.WanVideoVACEEncode = WanVideoVACEEncode
                    self.WanVideoDecode = WanVideoDecode
                    
                    self.vace_encoder = WanVideoVACEEncode()
                    self.video_decoder = WanVideoDecode()
                    
                    self.nodes_available = True
                    print("WANVACELoopEncoder: Successfully imported WanWrapper nodes (alternate path)")
                else:
                    raise ImportError(f"Could not import WanWrapper nodes: {e}")
                    
        except Exception as e:
            print(f"WANVACELoopEncoder: Failed to import required nodes: {e}")
            print("WANVACELoopEncoder: Please ensure ComfyUI-WanVideoWrapper is properly installed")
            print("WANVACELoopEncoder: Install from: https://github.com/kijai/ComfyUI-WanVideoWrapper")
    
    def calculate_batch_plan(self, total_frames: int, context_frames: int, overlap_frames: int) -> Dict:
        """
        Calculate batch processing plan with proper sliding window calculation.
        """
        if total_frames <= context_frames:
            return {
                "needs_batching": False,
                "total_batches": 1,
                "batches": [(0, total_frames)],
                "total_frames": total_frames
            }
        
        # Proper sliding window calculation
        # Each batch after the first processes (context_frames - overlap_frames) new frames
        effective_batch_size = context_frames - overlap_frames
        total_batches = math.ceil((total_frames - overlap_frames) / effective_batch_size)
        
        batches = []
        for batch_idx in range(total_batches):
            if batch_idx == 0:
                start_frame = 0
                end_frame = min(context_frames, total_frames)
            else:
                # Each subsequent batch starts where the previous batch ended minus overlap
                start_frame = batch_idx * effective_batch_size
                end_frame = min(start_frame + context_frames, total_frames)
            
            batches.append((start_frame, end_frame))
        
        return {
            "needs_batching": True,
            "total_batches": total_batches,
            "batches": batches,
            "total_frames": total_frames,
            "overlap_frames": overlap_frames
        }
    
    def extract_batch_frames(self, images: torch.Tensor, masks: torch.Tensor, 
                           start_frame: int, end_frame: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract frames and masks for a specific batch."""
        batch_images = images[start_frame:end_frame]
        batch_masks = masks[start_frame:end_frame]
        return batch_images, batch_masks
    
    def decode_reference_frames(self, vae: Any, latent_output: Dict, 
                              overlap_frames: int, width: int, height: int,
                              verbose_logging: bool = False) -> Optional[torch.Tensor]:
        """
        Decode the last N frames from a batch output to use as reference for next batch.
        """
        if not self.nodes_available or latent_output is None:
            return None
        
        try:
            # Extract latent samples
            if isinstance(latent_output, dict) and "samples" in latent_output:
                latents = latent_output["samples"]
            else:
                print("WANVACELoopEncoder: Unexpected latent format")
                return None
            
            # Get the last overlap_frames from the latent
            # Latent shape: [batch, channels, frames, height, width]
            num_frames = latents.shape[2]
            start_idx = max(0, num_frames - overlap_frames)
            
            # Create a new latent dict with just the reference frames
            reference_latents = {
                "samples": latents[:, :, start_idx:num_frames].clone()
            }
            
            # Decode using WanVideoDecode
            decoded_result = self.video_decoder.process(
                vae=vae,
                samples=reference_latents,
                enable_tiling=False,
                tile_sample_min_height=256,
                tile_sample_min_width=256,
                tile_overlap_factor_height=0.25,
                tile_overlap_factor_width=0.25
            )
            
            # Extract images from result
            if isinstance(decoded_result, tuple):
                reference_images = decoded_result[0]
            else:
                reference_images = decoded_result
            
            if verbose_logging:
                print(f"Decoded {reference_images.shape[0]} reference frames")
            
            return reference_images
            
        except Exception as e:
            print(f"WANVACELoopEncoder: Failed to decode reference frames: {e}")
            return None
    
    def encode_batches(self, vae, controlnet_images, controlnet_masks, 
                      context_frames=81, overlap_frames=4, vace_scale=1.0,
                      vace_start_percent=0.0, vace_end_percent=1.0, 
                      has_ref=False, verbose_logging=False,
                      initial_ref_images=None, width=832, height=480):
        """
        Encode multiple batches of frames with VACE, handling reference frames between batches.
        """
        
        if not self.nodes_available:
            raise RuntimeError("WanVideoWrapper nodes not available. Please install ComfyUI-WanVideoWrapper.")
        
        # Get total frames from input
        total_frames = controlnet_images.shape[0]
        
        if verbose_logging:
            print(f"WANVACELoopEncoder: Processing {total_frames} frames")
            print(f"Context frames: {context_frames}, Overlap: {overlap_frames}")
        
        # Calculate batch plan
        batch_plan = self.calculate_batch_plan(total_frames, context_frames, overlap_frames)
        
        if verbose_logging:
            print(f"Batch plan: {batch_plan['total_batches']} batches")
            for i, (start, end) in enumerate(batch_plan['batches']):
                print(f"  Batch {i+1}: frames {start}-{end-1} ({end-start} frames)")
        
        # Process each batch
        all_vace_embeds = []
        prev_vace_embeds = None
        reference_frames = initial_ref_images
        
        for batch_idx, (start_frame, end_frame) in enumerate(batch_plan['batches']):
            if verbose_logging:
                print(f"\nProcessing batch {batch_idx + 1}/{batch_plan['total_batches']}")
            
            # Extract frames for this batch
            batch_images, batch_masks = self.extract_batch_frames(
                controlnet_images, controlnet_masks, start_frame, end_frame
            )
            
            # Prepare inputs for WanVideoVACEEncode
            batch_frames = end_frame - start_frame
            
            # For first batch, use initial reference if provided
            # For subsequent batches, use decoded reference from previous batch
            if batch_idx == 0:
                ref_images = initial_ref_images if has_ref else None
            else:
                ref_images = reference_frames
            
            # Call WanVideoVACEEncode
            if verbose_logging:
                print(f"  Encoding {batch_frames} frames with VACE")
                print(f"  Reference frames: {'Yes' if ref_images is not None else 'No'}")
                print(f"  Previous VACE embeds: {'Yes' if prev_vace_embeds is not None else 'No'}")
            
            vace_result = self.vace_encoder.process(
                vae=vae,
                input_frames=batch_images,
                ref_images=ref_images,
                input_masks=batch_masks,
                prev_vace_embeds=prev_vace_embeds,
                width=width,
                height=height,
                num_frames=batch_frames,
                strength=vace_scale,
                vace_start_percent=vace_start_percent,
                vace_end_percent=vace_end_percent
            )
            
            if verbose_logging:
                print(f"  VACE encoding completed for batch {batch_idx + 1}")
            
            # Extract VACE embeds from result
            if isinstance(vace_result, tuple):
                batch_vace_embeds = vace_result[0]
            else:
                batch_vace_embeds = vace_result
            
            # Store for accumulation
            all_vace_embeds.append(batch_vace_embeds)
            
            # Update prev_vace_embeds for next batch
            prev_vace_embeds = batch_vace_embeds
            
            # Decode reference frames for next batch (if not last batch)
            if batch_idx < batch_plan['total_batches'] - 1:
                # First we need to get the latent output from sampling
                # For now, we'll extract from the VACE embeds if possible
                # This is a placeholder - in actual use, we'd need the sampled output
                if verbose_logging:
                    print("Note: Reference frame extraction requires sampled output")
                # For testing, we'll continue without reference frames
                reference_frames = None
        
        # Combine all VACE embeds
        # For now, return the last batch's embeds (with full chain)
        # In practice, we might need a different combination strategy
        final_vace_embeds = all_vace_embeds[-1] if all_vace_embeds else None
        
        # Generate batch info
        batch_info = self.generate_batch_info(batch_plan)
        
        return (final_vace_embeds, batch_info)
    
    def generate_batch_info(self, batch_plan: Dict) -> str:
        """Generate human-readable batch information."""
        info_lines = [
            f"Total frames: {batch_plan['total_frames']}",
            f"Total batches: {batch_plan['total_batches']}",
            f"Overlap frames: {batch_plan.get('overlap_frames', 0)}",
            ""
        ]
        
        for i, (start, end) in enumerate(batch_plan['batches']):
            info_lines.append(f"Batch {i+1}: frames {start}-{end-1} ({end-start} frames)")
        
        return "\n".join(info_lines)


# Node export
NODE_CLASS_MAPPINGS = {
    "WANVACELoopEncoder": WANVACELoopEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANVACELoopEncoder": "WAN VACE Loop Encoder"
}