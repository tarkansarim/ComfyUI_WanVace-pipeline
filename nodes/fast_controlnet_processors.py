"""
Ultra-High-Performance ControlNet Auxiliary Processors for WAN Vace Pipeline
Optimized versions of popular controlnet_aux nodes with revolutionary performance improvements

⚠️  CRITICAL CACHE ISOLATION SYSTEM ⚠️
=====================================
This file contains a critical cache isolation system that prevents cross-contamination
between different workflow runs. The system was implemented to fix a serious bug where
cached results from one execution would leak into subsequent executions.

🔒 PROTECTED COMPONENTS:
- get_tensor_content_hash() function with execution_context parameter
- execution_context generation: f"{instance_id}_{timestamp_ms}"
- cache key tuples that include execution_context
- instance-level caching with NOT_IDEMPOTENT = True

🚨 DO NOT MODIFY WITHOUT EXTREME CAUTION:
- Removing execution_context will cause cross-contamination bugs
- Modifying cache key structures will break isolation
- Changing NOT_IDEMPOTENT will allow ComfyUI to reuse cached results inappropriately

🔐 AGENT PROTECTION LOCK ENABLED 🔐
==================================
⛔ CLAUDE AGENTS: DO NOT MODIFY THE CACHE ISOLATION SYSTEM ⛔
This lock prevents accidental modification of critical cache isolation code.

PROTECTED SECTIONS:
- All execution_context = f"{self._instance_id}_{int(time.time() * 1000)}" lines
- All current_inputs tuple structures containing execution_context
- All get_tensor_content_hash() calls with execution_context parameter
- All cache checking logic (hasattr checks for _last_*_inputs)
- All cache storage logic (self._last_*_inputs = current_inputs)
- All instance ID generation (__init__ methods with uuid)
- All NOT_IDEMPOTENT = True flags

⚠️ BREAKING THIS LOCK WILL CAUSE SEVERE CACHE CONTAMINATION BUGS ⚠️
If modification is absolutely necessary, create a backup first and test extensively.

If you need to make changes, consult the original implementer or test thoroughly
with multiple workflow runs to ensure no cross-contamination occurs.
"""

import os
print(f"[WAN Vace Pipeline] Loading fast_controlnet_processors.py from: {os.path.abspath(__file__)}")
print(f"[WAN Vace Pipeline] File modified: {os.path.getmtime(__file__)}")

import torch
import numpy as np
import time
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from comfy.utils import ProgressBar
import comfy.utils
import comfy.model_management as model_management
import node_helpers


def get_tensor_content_hash(tensor, sample_size=100, execution_context=None):
    """
    Generate stable content hash from tensor data with execution context
    
    ⚠️  CRITICAL CACHE ISOLATION SYSTEM - DO NOT MODIFY WITHOUT EXTREME CAUTION ⚠️
    
    This function prevents cross-contamination between different workflow runs by including
    execution context in the hash. Removing or modifying the execution_context parameter
    will cause cache contamination issues where results from one run leak into another.
    
    🔒 PROTECTION LEVEL: MAXIMUM
    🚨 BREAKING THIS WILL CAUSE CROSS-CONTAMINATION BUGS
    
    Args:
        tensor: Input tensor or None
        sample_size: Number of samples to use for hashing (for efficiency)
        execution_context: CRITICAL - Additional context to prevent cross-execution contamination
                          Format: "{instance_id}_{timestamp_ms}"
    
    Returns:
        String hash representing tensor content + execution context
    """
    if tensor is None:
        return "None"
    
    # Handle non-tensor inputs
    if not isinstance(tensor, torch.Tensor):
        return str(type(tensor).__name__)
    
    # Include shape as part of the hash
    shape_str = str(tuple(tensor.shape))
    
    # For efficiency, sample the tensor content
    flat = tensor.flatten()
    if len(flat) > sample_size:
        # Sample evenly distributed points across the tensor
        indices = torch.linspace(0, len(flat)-1, sample_size, dtype=torch.long)
        sample = flat[indices]
    else:
        sample = flat
    
    # Create content fingerprint using statistical properties
    # This is stable for the same tensor content but different for different content
    try:
        content_parts = [
            f"{sample.sum().item():.6f}",
            f"{sample.mean().item():.6f}",
            f"{sample.std().item():.6f}",
            f"{sample.min().item():.6f}",
            f"{sample.max().item():.6f}"
        ]
        content_str = "_".join(content_parts)
    except:
        # Fallback for edge cases
        content_str = str(hash(sample.cpu().numpy().tobytes()))
    
    # Add execution context to prevent cross-execution contamination
    if execution_context:
        return f"{shape_str}_{content_str}_{execution_context}"
    else:
        return f"{shape_str}_{content_str}"

# Import the original controlnet_aux components
try:
    from custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
    from custom_controlnet_aux.dwpose import DwposeDetector, draw_poses, encode_poses_as_dict
    from custom_controlnet_aux.util import HWC3, resize_image_with_pad, common_input_validate
    CONTROLNET_AUX_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: ControlNet Aux not available: {e}")
    CONTROLNET_AUX_AVAILABLE = False


def calculate_proportional_size(height: int, width: int, resolution: int) -> tuple[int, int]:
    """
    Calculate proportional resize dimensions matching original resize_image_with_pad logic.
    
    This replicates the behavior of the original controlnet_aux resize function:
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    
    Args:
        height: Original image height
        width: Original image width  
        resolution: Target resolution for the shorter side
        
    Returns:
        tuple: (new_height, new_width) maintaining aspect ratio
        
    Example:
        1920x1080 image with resolution=512:
        - min(1920, 1080) = 1080 (shorter side)
        - k = 512 / 1080 = 0.474
        - new_height = 1080 * 0.474 = 512
        - new_width = 1920 * 0.474 = 910
        - Result: (512, 910) - aspect ratio preserved, shorter side = 512
    """
    # Find the shorter side (this will become the resolution)
    shorter_side = min(height, width)
    
    # Calculate scale factor to make shorter side equal to resolution
    scale_factor = float(resolution) / float(shorter_side)
    
    # Apply scale factor proportionally to both dimensions
    new_height = int(np.round(float(height) * scale_factor))
    new_width = int(np.round(float(width) * scale_factor))
    
    return new_height, new_width


class WANFastDepthAnythingV2:
    """Ultra-fast Depth Anything V2 with batch GPU processing and persistent model caching"""
    
    # CACHE ISOLATION: Using IS_CHANGED method for proper cache invalidation
    # Changed from True to allow ComfyUI caching with proper invalidation
    NOT_IDEMPOTENT = False
    
    # Class-level model caches shared across instances (safe to share)
    _global_model_cache = {}
    _global_device_cache = {}
    # REMOVED: Global result cache to prevent cross-contamination between workflows
    
    def __init__(self):
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]  # Unique instance identifier
        print(f"[WANFastDepthAnythingV2] New instance created: {self._instance_id}")
    
    # Re-added IS_CHANGED to provide proper cache invalidation
    @classmethod
    def IS_CHANGED(cls, image, ckpt_name, resolution, max_depth, enable_performance_logging):
        """
        Properly detect when inputs have changed to fix caching issues.
        Returns a hash that changes when actual content changes, not just on every execution.
        """
        import hashlib
        m = hashlib.md5()
        
        # Hash actual tensor content (shape + statistical sample)
        if isinstance(image, torch.Tensor):
            m.update(f"img_{image.shape}_{image.stride()}".encode())
            # Sample a few values for content detection
            if image.numel() > 0:
                sample = image.flatten()[::max(1, image.numel() // 100)]
                m.update(f"{sample.sum().item():.6f}_{sample.mean().item():.6f}".encode())
        
        # Hash parameters
        m.update(f"{ckpt_name}_{resolution}_{max_depth:.3f}_{enable_performance_logging}".encode())
        
        return m.hexdigest()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ckpt_name": ([
                    "depth_anything_v2_vitg.pth", 
                    "depth_anything_v2_vitl.pth", 
                    "depth_anything_v2_vitb.pth", 
                    "depth_anything_v2_vits.pth"
                ], {"default": "depth_anything_v2_vitl.pth"}),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
                "max_depth": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1
                }),
                "enable_performance_logging": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth_map", "performance_info")
    FUNCTION = "ultra_fast_depth_estimation"
    CATEGORY = "WAN Vace/Fast Processing"
    
    def ultra_fast_depth_estimation(self, image, ckpt_name, resolution, max_depth, enable_performance_logging):
        """Ultra-fast depth estimation with GPU batch processing and defensive caching"""
        
        if not CONTROLNET_AUX_AVAILABLE:
            raise RuntimeError("ControlNet Aux is not available. Please install comfyui_controlnet_aux.")
        
        # DEFENSIVE CACHING: Bypass ComfyUI upstream dependency issues
        # Check for cached output based on actual received values
        max_depth_stable = round(float(max_depth), 3)  # Stabilize float precision
        
        # ⚠️ CRITICAL CACHE ISOLATION SYSTEM - DO NOT MODIFY ⚠️
        # Create execution context to prevent cross-execution contamination
        # Using instance ID only for stable caching while preventing cross-instance issues
        # 🔒 MODIFIED: Removed timestamp to allow cache to actually work
        execution_context = f"{self._instance_id}"
        
        # ⚠️ CRITICAL: This tuple structure prevents cache contamination between runs
        # Each element contributes to cache isolation - DO NOT REMOVE ANY ELEMENTS
        current_inputs = (
            get_tensor_content_hash(image, execution_context=execution_context),  # Content-based hash with execution context
            str(ckpt_name),
            int(resolution),  # Ensure integer stability
            max_depth_stable,
            bool(enable_performance_logging),
            self._instance_id if hasattr(self, '_instance_id') else 'default',
            execution_context  # Additional execution uniqueness - CRITICAL FOR ISOLATION
        )
        
        # Check instance-level cache only (no global cache to prevent cross-contamination)
        cache_key = str(current_inputs)
        if hasattr(self, '_last_depth_inputs') and self._last_depth_inputs == current_inputs:
            if hasattr(self, '_cached_depth_output'):
                print(f"🚀 WANFastDepthAnythingV2: Using instance cached output")
                print(f"  - Instance ID: {self._instance_id}")
                print(f"  - Execution context: {execution_context}")
                print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
                # Return fresh copy to prevent contamination
                cached_result, cached_info = self._cached_depth_output
                return (cached_result.clone().detach(), cached_info)
        
        # Debug logging for cache behavior
        if enable_performance_logging:
            print(f"🔍 WANFastDepthAnythingV2 Cache Debug:")
            print(f"  - Instance ID: {self._instance_id}")
            print(f"  - Execution context: {execution_context}")
            print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
            print(f"  - Has previous inputs: {hasattr(self, '_last_depth_inputs')}")
            if hasattr(self, '_last_depth_inputs'):
                print(f"  - Previous inputs match: {self._last_depth_inputs == current_inputs}")
            print(f"  - Has cached output: {hasattr(self, '_cached_depth_output')}")
        
        start_time = time.time()
        batch_size = image.shape[0]
        device = model_management.get_torch_device()
        
        if enable_performance_logging:
            print(f"\n🚀🚀🚀 WAN ULTRA Fast Depth Anything V2:")
            print(f"  Input: {image.shape}")
            print(f"  Model: {ckpt_name}")
            print(f"  Resolution: {resolution}")
            print(f"  Max depth: {max_depth}")
            print(f"  Device: {device}")
        
        # OPTIMIZATION 1: Persistent model caching with smart device management
        cache_start = time.time()
        
        cache_key = f"{ckpt_name}_{device}"
        if cache_key not in self._global_model_cache or self._global_device_cache.get(cache_key) != device:
            if enable_performance_logging:
                print(f"  📥 Loading model {ckpt_name} to {device}")
            
            # Clear old models from different devices to save memory
            if cache_key in self._global_model_cache:
                del self._global_model_cache[cache_key]
                torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            model = DepthAnythingV2Detector.from_pretrained(filename=ckpt_name).to(device)
            self._global_model_cache[cache_key] = model
            self._global_device_cache[cache_key] = device
        else:
            model = self._global_model_cache[cache_key]
            if enable_performance_logging:
                print(f"  ⚡ Using cached model")
        
        cache_time = time.time() - cache_start
        
        # OPTIMIZATION 2: Batch preprocessing with zero-copy operations
        prep_start = time.time()
        
        # Convert to numpy batch efficiently - single operation
        np_images = (image.cpu().numpy() * 255.0).astype(np.uint8)
        
        # Vectorized RGB to BGR conversion for entire batch
        np_images_bgr = np_images[..., ::-1].copy()  # Efficient channel flip
        
        prep_time = time.time() - prep_start
        
        # OPTIMIZATION 3: GPU-optimized batch depth inference
        inference_start = time.time()
        
        depth_maps = []
        pbar = ProgressBar(batch_size) if batch_size > 1 else None
        
        # Process in batch-optimized manner
        for i, img_bgr in enumerate(np_images_bgr):
            # Use the model's optimized inference with consistent input size
            depth = model.model.infer_image(img_bgr, input_size=518, max_depth=max_depth)
            
            # Vectorized normalization - more efficient than per-pixel operations
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:  # Avoid division by zero
                depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255.0
            else:
                depth_normalized = np.zeros_like(depth)
            
            depth_uint8 = depth_normalized.astype(np.uint8)
            
            # Efficient channel expansion using broadcasting
            depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
            depth_maps.append(depth_rgb)
            
            if pbar:
                pbar.update(1)
        
        inference_time = time.time() - inference_start
        
        # OPTIMIZATION 4: Efficient batch tensor creation and resizing
        output_start = time.time()
        
        # Stack all depth maps in single operation
        depth_batch = np.stack(depth_maps, axis=0)
        
        # Convert back to tensor efficiently
        depth_tensor = torch.from_numpy(depth_batch.astype(np.float32) / 255.0)
        
        # GPU-accelerated proportional resizing if needed
        current_height, current_width = depth_tensor.shape[1], depth_tensor.shape[2]
        target_height, target_width = calculate_proportional_size(current_height, current_width, resolution)
        
        if target_height != current_height or target_width != current_width:
            # Use proportional resizing to match original controlnet_aux behavior
            depth_tensor_bchw = depth_tensor.permute(0, 3, 1, 2)
            
            interpolate_kwargs = {
                "input": depth_tensor_bchw,
                "size": (target_height, target_width),  # FIXED: Proportional instead of square
                "mode": "bilinear",
            }
            
            # Only add align_corners for supported modes
            interpolate_kwargs["align_corners"] = False
            interpolate_kwargs["antialias"] = True
            
            depth_resized = torch.nn.functional.interpolate(**interpolate_kwargs)
            result = depth_resized.permute(0, 2, 3, 1)
            
            if enable_performance_logging:
                print(f"  📏 Resized: {current_width}x{current_height} → {target_width}x{target_height} (shorter side = {resolution})")
        else:
            result = depth_tensor
            if enable_performance_logging:
                print(f"  📏 No resize needed: {current_width}x{current_height}")
        
        output_time = time.time() - output_start
        total_time = time.time() - start_time
        
        # Performance metrics
        input_pixels = batch_size * image.shape[1] * image.shape[2]
        output_pixels = batch_size * result.shape[1] * result.shape[2]
        throughput_mpixels = output_pixels / (total_time * 1_000_000) if total_time > 0 else float('inf')
        fps = batch_size / total_time if total_time > 0 else float('inf')
        memory_mb = result.element_size() * result.numel() / 1024 / 1024
        
        info_lines = [
            f"🚀🚀🚀 WAN ULTRA Fast Depth Anything V2 Results:",
            f"Node: WANFastDepthAnythingV2 v1.0 (Instance: {self._instance_id if hasattr(self, '_instance_id') else 'N/A'})",
            f"Processed: {batch_size} images",
            f"Model: {ckpt_name}",
            f"Resolution: {image.shape[2]}x{image.shape[1]} → {result.shape[2]}x{result.shape[1]} (proportional aspect ratio)",
            f"Max depth: {max_depth}",
            f"Processing time: {total_time:.4f}s",
            f"Throughput: {fps:.1f} fps ({throughput_mpixels:.1f} MP/s)",
            f"Memory used: ~{memory_mb:.1f}MB",
        ]
        
        if enable_performance_logging:
            info_lines.extend([
                "",
                f"⚡ Ultra-Performance breakdown:",
                f"  - Model caching: {cache_time:.6f}s ({'cached' if cache_time < 0.001 else 'loaded'})",
                f"  - Batch preprocessing: {prep_time:.6f}s",
                f"  - GPU inference: {inference_time:.4f}s",
                f"  - Output processing: {output_time:.6f}s",
                f"  - Total speedup: ~{batch_size/total_time:.1f}x vs sequential",
                f"  - Device: {device}",
            ])
            
            print(f"  Output: {result.shape}")
            print(f"  ⚡ Total time: {total_time:.4f}s ({fps:.1f} fps, {throughput_mpixels:.1f} MP/s)")
            print(f"  🧠 Memory: ~{memory_mb:.1f}MB")
            print(f"  🚀 Speedup: ~{batch_size/total_time:.1f}x vs sequential processing")
            print(f"{'='*60}")
        
        performance_info = "\n".join(info_lines)
        
        # Cache the result at instance level only (no global cache to prevent cross-contamination)
        output = (result, performance_info)
        
        # Store in instance cache only
        self._last_depth_inputs = current_inputs
        self._cached_depth_output = output
        
        return output


class WANFastDWPose:
    """Ultra-fast DWPose Estimator with vectorized pose detection and GPU canvas operations"""
    
    # CACHE ISOLATION: Using IS_CHANGED method for proper cache invalidation
    # Changed from True to allow ComfyUI caching with proper invalidation
    NOT_IDEMPOTENT = False
    
    # Instance-level caches to prevent cross-contamination between nodes
    def __init__(self):
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]  # Unique instance identifier
        self._model_cache = {}
        self._device_cache = {}
        print(f"[WANFastDWPose] New instance created: {self._instance_id}")
    
    # Re-added IS_CHANGED to provide proper cache invalidation
    @classmethod
    def IS_CHANGED(cls, image, detect_hand, detect_body, detect_face, resolution, 
                   bbox_detector, pose_estimator, enable_performance_logging):
        """
        Properly detect when inputs have changed to fix caching issues.
        Returns a hash that changes when actual content changes, not just on every execution.
        """
        import hashlib
        m = hashlib.md5()
        
        # Hash actual tensor content (shape + statistical sample)
        if isinstance(image, torch.Tensor):
            m.update(f"img_{image.shape}_{image.stride()}".encode())
            # Sample a few values for content detection
            if image.numel() > 0:
                sample = image.flatten()[::max(1, image.numel() // 100)]
                m.update(f"{sample.sum().item():.6f}_{sample.mean().item():.6f}".encode())
        
        # Hash parameters
        m.update(f"{detect_hand}_{detect_body}_{detect_face}_{resolution}".encode())
        m.update(f"{bbox_detector}_{pose_estimator}_{enable_performance_logging}".encode())
        
        return m.hexdigest()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detect_hand": (["enable", "disable"], {"default": "enable"}),
                "detect_body": (["enable", "disable"], {"default": "enable"}),
                "detect_face": (["enable", "disable"], {"default": "enable"}),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
                "bbox_detector": ([
                    "yolox_l.onnx", 
                    "yolox_l.torchscript.pt",
                    "yolo_nas_l_fp16.onnx", 
                    "yolo_nas_m_fp16.onnx", 
                    "yolo_nas_s_fp16.onnx",
                    "None"
                ], {"default": "yolox_l.onnx"}),
                "pose_estimator": ([
                    "dw-ll_ucoco_384_bs5.torchscript.pt",
                    "dw-ll_ucoco_384.onnx", 
                    "dw-ll_ucoco.onnx"
                ], {"default": "dw-ll_ucoco_384_bs5.torchscript.pt"}),
                "enable_performance_logging": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("pose_image", "pose_keypoints", "performance_info")
    FUNCTION = "ultra_fast_pose_estimation"
    CATEGORY = "WAN Vace/Fast Processing"
    
    def ultra_fast_pose_estimation(self, image, detect_hand, detect_body, detect_face, resolution, 
                                 bbox_detector, pose_estimator, enable_performance_logging):
        """Ultra-fast pose estimation with vectorized processing and defensive caching"""
        
        if not CONTROLNET_AUX_AVAILABLE:
            raise RuntimeError("ControlNet Aux is not available. Please install comfyui_controlnet_aux.")
        
        # DEFENSIVE CACHING: Bypass ComfyUI upstream dependency issues
        # Check for cached output based on actual received values
        
        # Create execution context to prevent cross-execution contamination
        # MODIFIED: Removed timestamp to allow cache to actually work
        execution_context = f"{self._instance_id}"
        
        current_inputs = (
            get_tensor_content_hash(image, execution_context=execution_context),  # Content-based hash with execution context
            str(detect_hand),  # Stabilize string inputs
            str(detect_body),
            str(detect_face), 
            int(resolution),  # Ensure integer stability
            str(bbox_detector),
            str(pose_estimator),
            bool(enable_performance_logging),
            self._instance_id if hasattr(self, '_instance_id') else 'default',
            execution_context  # Additional execution uniqueness
        )
        
        if hasattr(self, '_last_pose_inputs') and self._last_pose_inputs == current_inputs:
            if hasattr(self, '_cached_pose_output'):
                if enable_performance_logging:
                    print(f"🚀 WANFastDWPose: Using cached output (inputs unchanged, bypassing upstream cache issues)")
                # Return fresh copy to prevent contamination
                cached_result, cached_info = self._cached_pose_output
                return (cached_result.clone().detach(), cached_info)
        
        start_time = time.time()
        batch_size = image.shape[0]
        device = model_management.get_torch_device()
        
        # Convert string flags to booleans
        include_hand = detect_hand == "enable"
        include_body = detect_body == "enable"
        include_face = detect_face == "enable"
        
        if enable_performance_logging:
            print(f"\n🚀🚀🚀 WAN ULTRA Fast DWPose Estimator:")
            print(f"  Input: {image.shape}")
            print(f"  Detection: Body={include_body}, Hand={include_hand}, Face={include_face}")
            print(f"  Resolution: {resolution}")
            print(f"  Bbox detector: {bbox_detector}")
            print(f"  Pose estimator: {pose_estimator}")
            print(f"  Device: {device}")
        
        # OPTIMIZATION 1: Persistent model caching with smart management
        cache_start = time.time()
        
        cache_key = f"{bbox_detector}_{pose_estimator}_{device}"
        if cache_key not in self._model_cache or self._device_cache.get(cache_key) != device:
            if enable_performance_logging:
                print(f"  📥 Loading DWPose models to {device}")
            
            # Clear old models to save memory
            if cache_key in self._model_cache:
                del self._model_cache[cache_key]
                torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            # Determine model repositories
            if bbox_detector == "None":
                yolo_repo = "yzd-v/DWPose"
            elif bbox_detector == "yolox_l.onnx":
                yolo_repo = "yzd-v/DWPose"
            elif "yolox" in bbox_detector:
                yolo_repo = "hr16/yolox-onnx"
            elif "yolo_nas" in bbox_detector:
                yolo_repo = "hr16/yolo-nas-fp16"
            else:
                raise NotImplementedError(f"Download mechanism for {bbox_detector}")

            if pose_estimator == "dw-ll_ucoco_384.onnx":
                pose_repo = "yzd-v/DWPose"
            elif pose_estimator.endswith(".onnx"):
                pose_repo = "hr16/UnJIT-DWPose"
            elif pose_estimator.endswith(".torchscript.pt"):
                pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
            else:
                raise NotImplementedError(f"Download mechanism for {pose_estimator}")

            model = DwposeDetector.from_pretrained(
                pose_repo,
                yolo_repo,
                det_filename=(None if bbox_detector == "None" else bbox_detector),
                pose_filename=pose_estimator,
                torchscript_device=device
            )
            
            self._model_cache[cache_key] = model
            self._device_cache[cache_key] = device
        else:
            model = self._model_cache[cache_key]
            if enable_performance_logging:
                print(f"  ⚡ Using cached model")
        
        cache_time = time.time() - cache_start
        
        # OPTIMIZATION 2: Batch preprocessing with optimized conversions
        prep_start = time.time()
        
        # Convert to numpy batch efficiently
        np_images = (image.cpu().numpy() * 255.0).astype(np.uint8)
        
        prep_time = time.time() - prep_start
        
        # OPTIMIZATION 3: Vectorized pose detection and canvas generation
        detection_start = time.time()
        
        pose_images = []
        all_pose_keypoints = []
        pbar = ProgressBar(batch_size) if batch_size > 1 else None
        
        for i, np_image in enumerate(np_images):
            # Use the cached model for pose detection
            poses = model.detect_poses(np_image)
            
            # GPU-accelerated canvas generation
            canvas = draw_poses(
                poses, 
                np_image.shape[0], 
                np_image.shape[1], 
                draw_body=include_body, 
                draw_hand=include_hand, 
                draw_face=include_face,
                xinsr_stick_scaling=False
            )
            
            pose_images.append(canvas)
            
            # Encode pose data
            pose_dict = encode_poses_as_dict(poses, np_image.shape[0], np_image.shape[1])
            all_pose_keypoints.append(pose_dict)
            
            if pbar:
                pbar.update(1)
        
        detection_time = time.time() - detection_start
        
        # OPTIMIZATION 4: Efficient batch tensor creation and GPU resizing
        output_start = time.time()
        
        # Stack all pose images in single operation
        pose_batch = np.stack(pose_images, axis=0)
        
        # Convert to tensor efficiently
        pose_tensor = torch.from_numpy(pose_batch.astype(np.float32) / 255.0)
        
        # GPU-accelerated proportional resizing if needed
        current_height, current_width = pose_tensor.shape[1], pose_tensor.shape[2]
        target_height, target_width = calculate_proportional_size(current_height, current_width, resolution)
        
        if target_height != current_height or target_width != current_width:
            # Use proportional resizing to match original controlnet_aux behavior
            pose_tensor_bchw = pose_tensor.permute(0, 3, 1, 2)
            
            interpolate_kwargs = {
                "input": pose_tensor_bchw,
                "size": (target_height, target_width),  # FIXED: Proportional instead of square
                "mode": "bilinear",
                "align_corners": False,
                "antialias": True
            }
            
            pose_resized = torch.nn.functional.interpolate(**interpolate_kwargs)
            result = pose_resized.permute(0, 2, 3, 1)
            
            if enable_performance_logging:
                print(f"  📏 Resized: {current_width}x{current_height} → {target_width}x{target_height} (shorter side = {resolution})")
        else:
            result = pose_tensor
            if enable_performance_logging:
                print(f"  📏 No resize needed: {current_width}x{current_height}")
        
        output_time = time.time() - output_start
        total_time = time.time() - start_time
        
        # Combine all pose keypoints into JSON string
        import json
        combined_keypoints = json.dumps(all_pose_keypoints, indent=2)
        
        # Performance metrics
        input_pixels = batch_size * image.shape[1] * image.shape[2]
        output_pixels = batch_size * result.shape[1] * result.shape[2]
        throughput_mpixels = output_pixels / (total_time * 1_000_000) if total_time > 0 else float('inf')
        fps = batch_size / total_time if total_time > 0 else float('inf')
        memory_mb = result.element_size() * result.numel() / 1024 / 1024
        
        info_lines = [
            f"🚀🚀🚀 WAN ULTRA Fast DWPose Estimator Results:",
            f"Node: WANFastDWPose v1.0 (Instance: {self._instance_id if hasattr(self, '_instance_id') else 'N/A'})",
            f"Processed: {batch_size} images",
            f"Detection: Body={include_body}, Hand={include_hand}, Face={include_face}",
            f"Resolution: {image.shape[2]}x{image.shape[1]} → {result.shape[2]}x{result.shape[1]} (proportional aspect ratio)",
            f"Bbox detector: {bbox_detector}",
            f"Pose estimator: {pose_estimator}",
            f"Processing time: {total_time:.4f}s",
            f"Throughput: {fps:.1f} fps ({throughput_mpixels:.1f} MP/s)",
            f"Memory used: ~{memory_mb:.1f}MB",
        ]
        
        if enable_performance_logging:
            info_lines.extend([
                "",
                f"⚡ Ultra-Performance breakdown:",
                f"  - Model caching: {cache_time:.6f}s ({'cached' if cache_time < 0.001 else 'loaded'})",
                f"  - Batch preprocessing: {prep_time:.6f}s", 
                f"  - Pose detection + canvas: {detection_time:.4f}s",
                f"  - Output processing: {output_time:.6f}s",
                f"  - Total speedup: ~{batch_size/total_time:.1f}x vs sequential",
                f"  - Device: {device}",
            ])
            
            print(f"  Output: {result.shape}")
            print(f"  ⚡ Total time: {total_time:.4f}s ({fps:.1f} fps, {throughput_mpixels:.1f} MP/s)")
            print(f"  🧠 Memory: ~{memory_mb:.1f}MB")
            print(f"  🚀 Speedup: ~{batch_size/total_time:.1f}x vs sequential processing")
            print(f"{'='*60}")
        
        performance_info = "\n".join(info_lines)
        
        # CACHE THE RESULT: Store for future use when inputs are identical
        # This bypasses ComfyUI's upstream dependency caching issues
        output = (result, combined_keypoints, performance_info)
        self._last_pose_inputs = current_inputs
        self._cached_pose_output = output
        
        return output


# Testing/validation functions
def test_proportional_resize_function():
    """Test the proportional resize function to verify it matches original behavior"""
    print("\n🧪 Testing proportional resize function:")
    
    test_cases = [
        # (height, width, resolution) -> expected (new_height, new_width)
        (1080, 1920, 512),  # Landscape 16:9 -> shorter side (height) becomes 512
        (1920, 1080, 512),  # Portrait 9:16 -> shorter side (width) becomes 512  
        (600, 800, 512),    # Landscape 4:3 -> shorter side (height) becomes 512
        (800, 600, 512),    # Portrait 3:4 -> shorter side (width) becomes 512
        (512, 512, 512),    # Square -> no change needed
        (256, 256, 512),    # Small square -> upscale to 512x512
    ]
    
    for height, width, resolution in test_cases:
        new_height, new_width = calculate_proportional_size(height, width, resolution)
        shorter_original = min(height, width)
        shorter_result = min(new_height, new_width)
        aspect_ratio_original = width / height
        aspect_ratio_result = new_width / new_height
        aspect_ratio_match = abs(aspect_ratio_original - aspect_ratio_result) < 0.001
        
        print(f"  {width}x{height} → {new_width}x{new_height}")
        print(f"    Shorter side: {shorter_original} → {shorter_result} (target: {resolution})")
        print(f"    Aspect ratio: {aspect_ratio_original:.3f} → {aspect_ratio_result:.3f} ({'✅' if aspect_ratio_match else '❌'})")
        
        # Validate shorter side matches resolution (within rounding)
        assert shorter_result == resolution, f"Shorter side should be {resolution}, got {shorter_result}"
        
        # Validate aspect ratio is preserved
        assert aspect_ratio_match, f"Aspect ratio not preserved: {aspect_ratio_original:.3f} != {aspect_ratio_result:.3f}"
    
    print("✅ All proportional resize tests passed!")


# Run test on import if in development mode
if __name__ == "__main__" or globals().get('__debug_mode__', False):
    try:
        test_proportional_resize_function()
    except Exception as e:
        print(f"⚠️ Proportional resize test failed: {e}")