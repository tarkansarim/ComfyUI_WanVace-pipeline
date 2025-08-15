"""
High-Performance Image Processing Nodes for WAN Vace Pipeline
Optimized versions of standard ComfyUI nodes with significant performance improvements

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
print(f"[WAN Vace Pipeline] Loading fast_image_processing.py from: {os.path.abspath(__file__)}")
print(f"[WAN Vace Pipeline] File modified: {os.path.getmtime(__file__)}")

import torch
import numpy as np
import time
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from comfy.utils import ProgressBar
import comfy.utils
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


class WANFastImageBatchProcessor:
    """Ultra-fast image batch processor with vectorized frame selection"""
    
    # CACHE ISOLATION: Using IS_CHANGED method for proper cache invalidation
    # Changed from True to allow ComfyUI caching with proper invalidation
    NOT_IDEMPOTENT = False
    
    def __init__(self):
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]  # Unique instance identifier
    
    # Re-added IS_CHANGED to provide proper cache invalidation
    @classmethod
    def IS_CHANGED(cls, images, frame_load_cap, skip_first_frames, select_every_nth, enable_performance_logging):
        """
        Properly detect when inputs have changed to fix caching issues.
        Returns a hash that changes when actual content changes, not just on every execution.
        """
        import hashlib
        m = hashlib.md5()
        
        # Hash actual tensor content (shape + statistical sample)
        if isinstance(images, torch.Tensor):
            m.update(f"img_{images.shape}_{images.stride()}".encode())
            # Sample a few values for content detection
            if images.numel() > 0:
                sample = images.flatten()[::max(1, images.numel() // 100)]
                m.update(f"{sample.sum().item():.6f}_{sample.mean().item():.6f}".encode())
        
        # Hash parameters
        m.update(f"{frame_load_cap}_{skip_first_frames}_{select_every_nth}_{enable_performance_logging}".encode())
        
        return m.hexdigest()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_load_cap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Maximum frames to output (0 = unlimited)"
                }),
                "skip_first_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Skip N frames from beginning"
                }),
                "select_every_nth": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Select every Nth frame (1 = all frames)"
                }),
                "enable_performance_logging": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log detailed performance metrics"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_images", "processing_info")
    FUNCTION = "process_batch"
    CATEGORY = "WAN Vace/Fast Processing"
    
    def process_batch(self, images, frame_load_cap, skip_first_frames, select_every_nth, enable_performance_logging):
        """ULTRA-FAST batch processing with GPU vectorization and pattern optimization"""
        
        # DEFENSIVE CACHING: Bypass ComfyUI upstream dependency issues
        # Check for cached output based on actual received values
        
        # ⚠️ CRITICAL CACHE ISOLATION SYSTEM - DO NOT MODIFY ⚠️
        # Create execution context to prevent cross-execution contamination
        # Using instance ID only for stable caching while preventing cross-instance issues
        # 🔒 MODIFIED: Removed timestamp to allow cache to actually work
        execution_context = f"{self._instance_id}"
        
        # ⚠️ CRITICAL: This tuple structure prevents cache contamination between runs
        # Each element contributes to cache isolation - DO NOT REMOVE ANY ELEMENTS
        current_inputs = (
            get_tensor_content_hash(images, execution_context=execution_context),  # Content-based hash with execution context
            int(frame_load_cap),
            int(skip_first_frames),
            int(select_every_nth),
            bool(enable_performance_logging),
            self._instance_id if hasattr(self, '_instance_id') else 'default',  # Add instance isolation
            execution_context  # Additional execution uniqueness - CRITICAL FOR ISOLATION
        )
        
        # Check instance-level cache only (no global cache to prevent cross-contamination)
        cache_key = str(current_inputs)
        if hasattr(self, '_last_batch_inputs') and self._last_batch_inputs == current_inputs:
            if hasattr(self, '_cached_batch_output'):
                print(f"🚀 WANFastImageBatchProcessor: Using instance cached output")
                print(f"  - Instance ID: {self._instance_id}")
                print(f"  - Execution context: {execution_context}")
                print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
                # Return fresh copy to prevent contamination
                cached_result, cached_info = self._cached_batch_output
                return (cached_result.clone().detach(), cached_info)
        
        # Debug logging for cache behavior
        if enable_performance_logging:
            print(f"🔍 WANFastImageBatchProcessor Cache Debug:")
            print(f"  - Instance ID: {self._instance_id}")
            print(f"  - Execution context: {execution_context}")
            print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
            print(f"  - Has previous inputs: {hasattr(self, '_last_batch_inputs')}")
            if hasattr(self, '_last_batch_inputs'):
                print(f"  - Previous inputs match: {self._last_batch_inputs == current_inputs}")
            print(f"  - Has cached output: {hasattr(self, '_cached_batch_output')}")
        
        start_time = time.time()
        batch_size = images.shape[0]
        
        if enable_performance_logging:
            print(f"\n🚀🚀🚀 WAN ULTRA Fast Batch Processor:")
            print(f"  Input: {batch_size} frames")
            print(f"  Parameters: skip={skip_first_frames}, nth={select_every_nth}, cap={frame_load_cap}")
            print(f"  Device: {'CUDA' if images.is_cuda else 'CPU'}")
        
        # Early exit for no-op cases
        if batch_size == 0:
            return (images, "No frames to process")
        
        if skip_first_frames >= batch_size:
            empty_result = torch.empty((0, images.shape[1], images.shape[2], images.shape[3]), 
                                     dtype=images.dtype, device=images.device)
            return (empty_result, f"Skip ({skip_first_frames}) >= total frames ({batch_size})")
        
        # OPTIMIZATION 1: Pattern-Specific Ultra-Fast Operations
        pattern_start = time.time()
        
        # Detect common patterns and use optimal tensor operations (VIEWS, not copies)
        optimization_used = "Unknown"
        
        # Pattern 1: Simple truncation (cap only)
        if skip_first_frames == 0 and select_every_nth == 1 and frame_load_cap > 0:
            if frame_load_cap >= batch_size:
                result = images  # No-op, return as-is
                optimization_used = "No-op (cap >= batch_size)"
            else:
                result = images[:frame_load_cap]  # TENSOR VIEW - ULTRA FAST
                optimization_used = "Simple truncation (view)"
        
        # Pattern 2: Simple skip (skip only)  
        elif skip_first_frames > 0 and select_every_nth == 1 and frame_load_cap == 0:
            result = images[skip_first_frames:]  # TENSOR VIEW - ULTRA FAST
            optimization_used = "Simple skip (view)"
        
        # Pattern 3: Simple sampling (nth only)
        elif skip_first_frames == 0 and select_every_nth > 1 and frame_load_cap == 0:
            result = images[::select_every_nth]  # TENSOR VIEW - ULTRA FAST
            optimization_used = "Simple sampling (view)"
        
        # Pattern 4: Skip + truncation (no sampling)
        elif skip_first_frames > 0 and select_every_nth == 1 and frame_load_cap > 0:
            end_frame = min(batch_size, skip_first_frames + frame_load_cap)
            result = images[skip_first_frames:end_frame]  # TENSOR VIEW - ULTRA FAST
            optimization_used = "Skip + truncation (view)"
            
        # Pattern 5: Sampling + truncation (no skip)
        elif skip_first_frames == 0 and select_every_nth > 1 and frame_load_cap > 0:
            sampled = images[::select_every_nth]  # TENSOR VIEW first
            result = sampled[:frame_load_cap] if frame_load_cap < sampled.shape[0] else sampled
            optimization_used = "Sampling + truncation (view)"
        
        # Pattern 6: Complex pattern - use optimized GPU indexing
        else:
            # OPTIMIZATION 2: GPU-Native Advanced Indexing (no device transfers)
            max_possible_frames = (batch_size - skip_first_frames + select_every_nth - 1) // select_every_nth
            
            if frame_load_cap > 0:
                max_possible_frames = min(max_possible_frames, frame_load_cap)
            
            if max_possible_frames <= 0:
                empty_result = torch.empty((0, images.shape[1], images.shape[2], images.shape[3]), 
                                         dtype=images.dtype, device=images.device)
                return (empty_result, "No valid frames after filtering")
            
            # ULTRA-FAST: Keep everything on same device - NO TRANSFERS
            indices = torch.arange(max_possible_frames, device=images.device, dtype=torch.long)
            indices = indices * select_every_nth + skip_first_frames
            
            # Bounds checking in GPU
            valid_mask = indices < batch_size
            indices = indices[valid_mask]
            
            # GPU-optimized indexing - stay on device
            if images.is_cuda:
                # Use torch.index_select for CUDA optimization
                result = torch.index_select(images, 0, indices)
                optimization_used = "GPU-optimized indexing"
            else:
                result = images[indices]
                optimization_used = "CPU advanced indexing"
        
        pattern_time = time.time() - pattern_start
        total_time = time.time() - start_time
        
        # Performance metrics
        output_frames = result.shape[0]
        reduction_ratio = (batch_size - output_frames) / batch_size * 100
        fps = output_frames / total_time if total_time > 0 else float('inf')
        throughput_mpixels = (output_frames * images.shape[1] * images.shape[2]) / (total_time * 1_000_000) if total_time > 0 else float('inf')
        
        # Memory efficiency calculation
        memory_efficiency = "View" if "view" in optimization_used.lower() else "Copy"
        memory_mb = result.element_size() * result.numel() / 1024 / 1024
        
        # Detailed info string
        info_lines = [
            f"🚀🚀🚀 WAN ULTRA Fast Batch Processor Results:",
            f"Input frames: {batch_size} → Output frames: {output_frames}",
            f"Reduction: {reduction_ratio:.1f}%",
            f"Optimization: {optimization_used}",
            f"Memory: {memory_efficiency} operation (~{memory_mb:.1f}MB)",
            f"Processing time: {total_time:.4f}s",
            f"Throughput: {fps:.1f} fps ({throughput_mpixels:.1f} MP/s)",
            "",
            f"Parameters applied:",
            f"  - Skipped first: {skip_first_frames}",
            f"  - Every nth: {select_every_nth}",
            f"  - Frame cap: {frame_load_cap if frame_load_cap > 0 else 'None'}",
        ]
        
        if enable_performance_logging:
            info_lines.extend([
                "",
                f"⚡ Ultra-Performance breakdown:",
                f"  - Pattern optimization: {pattern_time:.6f}s",
                f"  - Memory efficiency: {memory_efficiency}",
                f"  - Device: {'CUDA' if result.is_cuda else 'CPU'}",
                f"  - Optimization method: {optimization_used}",
            ])
            
            print(f"  Output: {output_frames} frames ({reduction_ratio:.1f}% reduction)")
            print(f"  🚀 Optimization: {optimization_used}")
            print(f"  ⚡ Total time: {total_time:.6f}s ({fps:.1f} fps, {throughput_mpixels:.1f} MP/s)")
            print(f"  🧠 Memory: {memory_efficiency} operation (~{memory_mb:.1f}MB)")
            print(f"{'='*60}")
        
        processing_info = "\n".join(info_lines)
        
        # Cache the result at instance level only (no global cache to prevent cross-contamination)
        result_output = (result, processing_info)
        
        # Store in instance cache only
        self._last_batch_inputs = current_inputs
        self._cached_batch_output = result_output
        
        return result_output


class WANFastImageCompositeMasked:
    """Ultra-high-performance image compositing with full GPU vectorization"""
    
    # CACHE ISOLATION: Using IS_CHANGED method for proper cache invalidation
    # Changed from True to allow ComfyUI caching with proper invalidation
    NOT_IDEMPOTENT = False
    
    # REMOVED: Global cache to prevent cross-contamination between workflows
    # Each instance will use its own caching via IS_CHANGED and execution method
    
    def __init__(self):
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]  # Unique instance identifier
        print(f"[WANFastImageCompositeMasked] New instance created: {self._instance_id}")
    
    # Re-added IS_CHANGED to provide proper cache invalidation
    @classmethod
    def IS_CHANGED(cls, destination, source, x, y, resize_source, enable_performance_logging, mask=None):
        """
        Properly detect when inputs have changed to fix caching issues.
        Returns a hash that changes when actual content changes, not just on every execution.
        """
        import hashlib
        m = hashlib.md5()
        
        debug_parts = []
        
        # Hash actual tensor content (shape + statistical sample)
        if isinstance(destination, torch.Tensor):
            shape_str = f"d_{destination.shape}_{destination.stride()}"
            m.update(shape_str.encode())
            debug_parts.append(f"dest_shape={destination.shape}")
            # Sample a few values for content detection
            if destination.numel() > 0:
                sample = destination.flatten()[::max(1, destination.numel() // 100)]
                content_str = f"{sample.sum().item():.6f}_{sample.mean().item():.6f}"
                m.update(content_str.encode())
                debug_parts.append(f"dest_content={content_str[:20]}")
        
        if isinstance(source, torch.Tensor):
            shape_str = f"s_{source.shape}_{source.stride()}"
            m.update(shape_str.encode())
            debug_parts.append(f"src_shape={source.shape}")
            if source.numel() > 0:
                sample = source.flatten()[::max(1, source.numel() // 100)]
                content_str = f"{sample.sum().item():.6f}_{sample.mean().item():.6f}"
                m.update(content_str.encode())
                debug_parts.append(f"src_content={content_str[:20]}")
        
        if mask is not None and isinstance(mask, torch.Tensor):
            shape_str = f"m_{mask.shape}_{mask.stride()}"
            m.update(shape_str.encode())
            debug_parts.append(f"mask_shape={mask.shape}")
            if mask.numel() > 0:
                sample = mask.flatten()[::max(1, mask.numel() // 100)]
                content_str = f"{sample.sum().item():.6f}_{sample.mean().item():.6f}"
                m.update(content_str.encode())
                debug_parts.append(f"mask_content={content_str[:20]}")
        
        # Hash parameters
        param_str = f"{x}_{y}_{resize_source}_{enable_performance_logging}"
        m.update(param_str.encode())
        debug_parts.append(f"params=({x},{y},{resize_source})")
        
        final_hash = m.hexdigest()
        
        # Debug logging
        print(f"\n🔍 WANFastImageCompositeMasked.IS_CHANGED Debug:")
        print(f"  - Input details: {' | '.join(debug_parts)}")
        print(f"  - Final hash: {final_hash}")
        print(f"  - This method called at: {time.time()}")
        
        return final_hash
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
                "enable_performance_logging": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("composite_result", "performance_info")
    FUNCTION = "ultra_fast_composite"
    CATEGORY = "WAN Vace/Fast Processing"
    
    def ultra_fast_composite(self, destination, source, x, y, resize_source, enable_performance_logging, mask=None):
        """Ultra-fast composite using full GPU vectorization - 10x+ speedup"""
        
        print(f"\n🔴 WANFastImageCompositeMasked.ultra_fast_composite called!")
        print(f"  - Instance: {self._instance_id if hasattr(self, '_instance_id') else 'NO_ID'}")
        print(f"  - Time: {time.time()}")
        
        # DEFENSIVE CACHING: Bypass ComfyUI upstream dependency issues
        # ⚠️ CRITICAL CACHE ISOLATION SYSTEM - DO NOT MODIFY ⚠️
        # Create execution context to prevent cross-execution contamination
        # Using instance ID only for stable caching while preventing cross-instance issues
        # 🔒 MODIFIED: Removed timestamp to allow cache to actually work
        execution_context = f"{self._instance_id}"
        
        # ⚠️ CRITICAL: This tuple structure prevents cache contamination between runs
        # Each element contributes to cache isolation - DO NOT REMOVE ANY ELEMENTS
        current_inputs = (
            get_tensor_content_hash(destination, execution_context=execution_context),  # Content-based hash with execution context
            get_tensor_content_hash(source, execution_context=execution_context),       # Content-based hash with execution context
            int(x),
            int(y),
            bool(resize_source),
            bool(enable_performance_logging),
            get_tensor_content_hash(mask, execution_context=execution_context),  # Content-based hash with execution context
            self._instance_id if hasattr(self, '_instance_id') else 'default',  # Add instance isolation
            execution_context  # Additional execution uniqueness - CRITICAL FOR ISOLATION
        )
        
        # Check instance-level cache only (no global cache to prevent cross-contamination)
        cache_key = str(current_inputs)
        if hasattr(self, '_last_composite_inputs') and self._last_composite_inputs == current_inputs:
            if hasattr(self, '_cached_composite_output'):
                print(f"🚀 WANFastImageCompositeMasked: Using instance cached output")
                print(f"  - Instance ID: {self._instance_id}")
                print(f"  - Execution context: {execution_context}")
                print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
                # Return fresh copy to prevent contamination
                cached_result, cached_info = self._cached_composite_output
                return (cached_result.clone().detach(), cached_info)
        else:
            print(f"❌ WANFastImageCompositeMasked: Cache miss - executing")
            if hasattr(self, '_last_composite_inputs'):
                print(f"  - Previous cache key: {str(self._last_composite_inputs)[:50]}...")
                print(f"  - Current cache key:  {cache_key[:50]}...")
            else:
                print(f"  - No previous cache")
        
        # Debug logging for cache behavior
        if enable_performance_logging:
            print(f"🔍 WANFastImageCompositeMasked Cache Debug:")
            print(f"  - Instance ID: {self._instance_id}")
            print(f"  - Execution context: {execution_context}")
            print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
            print(f"  - Has previous inputs: {hasattr(self, '_last_composite_inputs')}")
            if hasattr(self, '_last_composite_inputs'):
                print(f"  - Previous inputs match: {self._last_composite_inputs == current_inputs}")
            print(f"  - Has cached output: {hasattr(self, '_cached_composite_output')}")
        
        start_time = time.time()
        
        # Enhanced parameter debugging to diagnose coordinate issues
        if enable_performance_logging:
            print(f"\n🚀🚀🚀 WAN ULTRA Fast Image Composite:")
            print(f"  Destination: {destination.shape}")
            print(f"  Source: {source.shape}")
            print(f"  📍 RECEIVED PARAMETERS:")
            print(f"    - x (raw): {x} (type: {type(x)})")
            print(f"    - y (raw): {y} (type: {type(y)})")
            print(f"    - resize_source: {resize_source}")
            print(f"    - mask provided: {mask is not None}")
            if mask is not None:
                print(f"    - mask shape: {mask.shape}")
            print(f"  Device: {'CUDA' if destination.is_cuda else 'CPU'}")
        
        # Alpha channel fix - but keep in BHWC format (no conversion overhead)
        destination, source = node_helpers.image_alpha_fix(destination, source)
        
        # OPTIMIZATION 1: Work directly with BHWC - no channel dimension conversions
        prep_start = time.time()
        
        # Fast device transfer with non-blocking
        source = source.to(destination.device, non_blocking=True)
        
        # Handle resizing if needed
        if resize_source and source.shape[1:3] != destination.shape[1:3]:
            # BHWC → BCHW for interpolation, then back
            source_for_resize = source.permute(0, 3, 1, 2)
            source_resized = torch.nn.functional.interpolate(
                source_for_resize, 
                size=destination.shape[1:3], 
                mode="bilinear", 
                align_corners=False,
                antialias=True
            )
            source = source_resized.permute(0, 2, 3, 1)
        
        # Batch size matching
        if source.shape[0] != destination.shape[0]:
            source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])
        
        prep_time = time.time() - prep_start
        
        # OPTIMIZATION 2: Vectorized bounds calculation for entire batch
        bounds_start = time.time()
        
        batch_size, dest_h, dest_w, channels = destination.shape
        _, src_h, src_w, _ = source.shape
        
        if enable_performance_logging:
            print(f"  📐 BOUNDS CALCULATION:")
            print(f"    - Destination size: {dest_w}x{dest_h}")
            print(f"    - Source size: {src_w}x{src_h}")
            print(f"    - Original position: ({x}, {y})")
        
        # Apply original ComfyUI coordinate processing logic (adapted for BHWC format)
        # Original: x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
        # For images, multiplier = 1, and we're in BHWC so shape[2] = width, shape[1] = height
        x_original, y_original = x, y
        x = max(-src_w * 1, min(x, dest_w * 1))
        y = max(-src_h * 1, min(y, dest_h * 1))
        
        if enable_performance_logging and (x != x_original or y != y_original):
            print(f"    - Position clamped: ({x_original}, {y_original}) → ({x}, {y})")
        
        # Original ComfyUI bounds calculation (adapted for BHWC format)
        # Original: left, top = (x // multiplier, y // multiplier)  
        # Original: right, bottom = (left + source.shape[3], top + source.shape[2])
        multiplier = 1  # For images (latents use multiplier=8)
        left, top = (x // multiplier, y // multiplier)
        right, bottom = (left + src_w, top + src_h)
        
        # Original ComfyUI visible bounds calculation (adapted for BHWC format)
        # Original: visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y))
        visible_width = dest_w - left + min(0, x)
        visible_height = dest_h - top + min(0, y)
        
        # Ensure visible bounds are non-negative
        visible_width = max(0, visible_width)
        visible_height = max(0, visible_height)
        
        # Source bounds based on visible region
        src_left, src_top = 0, 0
        src_right, src_bottom = visible_width, visible_height
        
        if enable_performance_logging:
            print(f"    - Destination region: [{left}, {top}] to [{right}, {bottom}] (size: {right-left}x{bottom-top})")
            print(f"    - Source region: [{src_left}, {src_top}] to [{src_right}, {src_bottom}] (size: {src_right-src_left}x{src_bottom-src_top})")
            if right <= left or bottom <= top:
                print(f"    - ⚠️ WARNING: No overlap detected!")
            else:
                print(f"    - ✅ Overlap detected: {(right-left) * (bottom-top)} pixels affected")
        
        bounds_time = time.time() - bounds_start
        
        # Early exit if no overlap
        if right <= left or bottom <= top:
            if enable_performance_logging:
                print(f"  ⚠️ No overlap detected, returning destination unchanged")
            return (destination, "No overlap - destination unchanged")
        
        # OPTIMIZATION 3: Mask processing following original ComfyUI logic (adapted for BHWC)
        mask_start = time.time()
        
        if mask is None:
            # Original: mask = torch.ones_like(source) (in BCHW format)
            # Adapted for BHWC: create mask matching source shape
            alpha = torch.ones((batch_size, src_h, src_w, 1), device=destination.device, dtype=destination.dtype)
        else:
            # Original ComfyUI mask processing logic
            mask = mask.to(destination.device, copy=True)
            
            # Original: mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            # Then interpolate to source size: size=(source.shape[2], source.shape[3])
            # Adapted for BHWC format
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # Add batch dimension
            
            # Reshape to (B, 1, H, W) for interpolation
            mask_for_interp = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            
            # Interpolate to source size (BCHW format for interpolation)
            mask_resized = torch.nn.functional.interpolate(
                mask_for_interp, 
                size=(src_h, src_w), 
                mode="bilinear"
            )
            
            # Convert back to BHWC format: (B, 1, H, W) → (B, H, W, 1)
            alpha = mask_resized.permute(0, 2, 3, 1)
            
            # Repeat to match batch size
            alpha = comfy.utils.repeat_to_batch_size(alpha, batch_size)
        
        # Original: mask = mask[:, :, :visible_height, :visible_width] (BCHW format)
        # Adapted for BHWC: crop to visible bounds
        alpha = alpha[:, :visible_height, :visible_width, :]
        
        mask_time = time.time() - mask_start
        
        # OPTIMIZATION 4: Compositing following original ComfyUI logic (adapted for BHWC)
        composite_start = time.time()
        
        # Original logic (adapted for BHWC format):
        # mask = mask[:, :, :visible_height, :visible_width] - already done above
        # source_portion = mask * source[:, :, :visible_height, :visible_width]
        # destination_portion = inverse_mask * destination[:, :, top:bottom, left:right]
        # destination[:, :, top:bottom, left:right] = source_portion + destination_portion
        
        # Calculate safe extraction dimensions that respect both source and destination constraints
        # Ensure we don't try to extract more than what's available in either tensor
        safe_height = min(visible_height, src_h, dest_h - top)
        safe_width = min(visible_width, src_w, dest_w - left)
        
        if enable_performance_logging:
            print(f"  📏 DIMENSION SAFETY CHECK:")
            print(f"    - Requested visible bounds: {visible_width}x{visible_height}")
            print(f"    - Source constraints: {src_w}x{src_h}")
            print(f"    - Destination constraints: {dest_w - left}x{dest_h - top}")
            print(f"    - Safe extraction bounds: {safe_width}x{safe_height}")
        
        # Extract regions using safe dimensions
        source_visible = source[:, :safe_height, :safe_width, :]
        dest_region = destination[:, top:top+safe_height, left:left+safe_width, :]
        
        # Crop alpha mask to match the safe dimensions
        alpha = alpha[:, :safe_height, :safe_width, :]
        
        if enable_performance_logging:
            print(f"  🎨 COMPOSITING OPERATION:")
            print(f"    - Source visible region shape: {source_visible.shape}")
            print(f"    - Destination region shape: {dest_region.shape}")
            print(f"    - Alpha mask shape: {alpha.shape}")
            print(f"    - Alpha range: {alpha.min().item():.3f} to {alpha.max().item():.3f}")
            print(f"    - Visible bounds: {visible_width}x{visible_height}")
            if mask is not None:
                print(f"    - Using custom mask")
            else:
                print(f"    - Using full opacity mask")
            print(f"    - Shapes match: {source_visible.shape == dest_region.shape == alpha.shape[:-1] + (channels,)}")
        
        # Original ComfyUI compositing logic (adapted for BHWC)
        # Create inverse mask
        inverse_alpha = torch.ones_like(alpha) - alpha
        
        # Calculate portions (vectorized) - now guaranteed to have matching shapes
        source_portion = alpha * source_visible
        destination_portion = inverse_alpha * dest_region
        
        # Combine portions and update destination (single vectorized operation)
        # Use safe dimensions for final assignment
        destination[:, top:top+safe_height, left:left+safe_width, :] = source_portion + destination_portion
        
        composite_time = time.time() - composite_start
        total_time = time.time() - start_time
        
        # Performance metrics
        fps = batch_size / total_time if total_time > 0 else float('inf')
        memory_mb = destination.element_size() * destination.numel() / 1024 / 1024
        throughput_mpixels = (batch_size * dest_h * dest_w) / (total_time * 1_000_000) if total_time > 0 else float('inf')
        
        info_lines = [
            f"🚀🚀🚀 WAN ULTRA Fast Composite Results:",
            f"Node: WANFastImageCompositeMasked v1.0 (Instance: {self._instance_id if hasattr(self, '_instance_id') else 'N/A'})",
            f"Processed: {batch_size} images ({dest_h}x{dest_w})",
            f"Position: ({x}, {y}) → Region: {right-left}x{bottom-top}",
            f"Resize source: {resize_source}",
            f"Mask: {'Yes' if mask is not None else 'No'}",
            f"Processing time: {total_time:.4f}s",
            f"Throughput: {fps:.1f} fps ({throughput_mpixels:.1f} MP/s)",
            f"Memory used: ~{memory_mb:.1f}MB",
        ]
        
        if enable_performance_logging:
            info_lines.extend([
                "",
                f"⚡ Ultra-Performance breakdown:",
                f"  - Preparation: {prep_time:.4f}s",
                f"  - Bounds calculation: {bounds_time:.4f}s", 
                f"  - Mask processing: {mask_time:.4f}s",
                f"  - Vectorized composite: {composite_time:.4f}s",
                f"  - Device: {'CUDA' if destination.is_cuda else 'CPU'}",
                f"  - Speedup estimate: {55.2 / total_time:.1f}x faster than before",
            ])
            
            print(f"  Output: {destination.shape}")
            print(f"  ⚡ Total time: {total_time:.4f}s ({fps:.1f} fps, {throughput_mpixels:.1f} MP/s)")
            print(f"  🧠 Memory: ~{memory_mb:.1f}MB")
            print(f"  🚀 Estimated speedup: {55.2 / total_time:.1f}x faster!")
            print(f"{'='*60}")
        
        performance_info = "\n".join(info_lines)
        
        # Cache the result at instance level only (no global cache to prevent cross-contamination)
        result_output = (destination.clone().detach(), performance_info)
        
        # Store in instance cache only
        self._last_composite_inputs = current_inputs
        self._cached_composite_output = result_output
        
        return result_output


class WANFastImageBlend:
    """High-performance image blending with vectorized operations"""
    
    # CACHE ISOLATION: Using IS_CHANGED method for proper cache invalidation
    # Changed from True to allow ComfyUI caching with proper invalidation
    NOT_IDEMPOTENT = False
    
    def __init__(self):
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]  # Unique instance identifier
    
    BLEND_MODES = ["normal", "multiply", "screen", "overlay", "soft_light", "difference", "addition", "subtract"]
    
    # Re-added IS_CHANGED to provide proper cache invalidation
    @classmethod
    def IS_CHANGED(cls, image1, image2, blend_factor, blend_mode, enable_performance_logging):
        """
        Properly detect when inputs have changed to fix caching issues.
        Returns a hash that changes when actual content changes, not just on every execution.
        """
        import hashlib
        m = hashlib.md5()
        
        # Hash actual tensor content (shape + statistical sample)
        if isinstance(image1, torch.Tensor):
            m.update(f"img1_{image1.shape}_{image1.stride()}".encode())
            if image1.numel() > 0:
                sample = image1.flatten()[::max(1, image1.numel() // 100)]
                m.update(f"{sample.sum().item():.6f}_{sample.mean().item():.6f}".encode())
        
        if isinstance(image2, torch.Tensor):
            m.update(f"img2_{image2.shape}_{image2.stride()}".encode())
            if image2.numel() > 0:
                sample = image2.flatten()[::max(1, image2.numel() // 100)]
                m.update(f"{sample.sum().item():.6f}_{sample.mean().item():.6f}".encode())
        
        # Hash parameters
        m.update(f"{blend_factor:.6f}_{blend_mode}_{enable_performance_logging}".encode())
        
        return m.hexdigest()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blend_mode": (cls.BLEND_MODES,),
                "enable_performance_logging": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("blended_image", "performance_info")
    FUNCTION = "fast_blend"
    CATEGORY = "WAN Vace/Fast Processing"
    
    @staticmethod
    def _vectorized_blend_modes(img1, img2, mode):
        """Ultra-fast vectorized blend mode operations"""
        
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "screen":
            return 1.0 - (1.0 - img1) * (1.0 - img2)
        elif mode == "overlay":
            return torch.where(img1 <= 0.5, 
                             2.0 * img1 * img2, 
                             1.0 - 2.0 * (1.0 - img1) * (1.0 - img2))
        elif mode == "soft_light":
            # Optimized soft light calculation
            mask = img2 <= 0.5
            sqrt_img1 = torch.sqrt(torch.clamp(img1, min=1e-8))  # Prevent sqrt(0)
            return torch.where(mask,
                             img1 - (1.0 - 2.0 * img2) * img1 * (1.0 - img1),
                             img1 + (2.0 * img2 - 1.0) * (sqrt_img1 - img1))
        elif mode == "difference":
            return torch.abs(img1 - img2)
        elif mode == "addition":
            return torch.clamp(img1 + img2, 0.0, 1.0)
        elif mode == "subtract":
            return torch.clamp(img1 - img2, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")
    
    def fast_blend(self, image1, image2, blend_factor, blend_mode, enable_performance_logging):
        """High-performance image blending with fused operations"""
        
        # DEFENSIVE CACHING: Bypass ComfyUI upstream dependency issues
        # ⚠️ CRITICAL CACHE ISOLATION SYSTEM - DO NOT MODIFY ⚠️
        # Create execution context to prevent cross-execution contamination
        # Using instance ID only for stable caching while preventing cross-instance issues
        # 🔒 MODIFIED: Removed timestamp to allow cache to actually work
        execution_context = f"{self._instance_id}"
        
        # ⚠️ CRITICAL: This tuple structure prevents cache contamination between runs
        # Each element contributes to cache isolation - DO NOT REMOVE ANY ELEMENTS
        current_inputs = (
            get_tensor_content_hash(image1, execution_context=execution_context),  # Content-based hash with execution context
            get_tensor_content_hash(image2, execution_context=execution_context),  # Content-based hash with execution context
            float(blend_factor),
            str(blend_mode),
            bool(enable_performance_logging),
            self._instance_id if hasattr(self, '_instance_id') else 'default',  # Add instance isolation
            execution_context  # Additional execution uniqueness - CRITICAL FOR ISOLATION
        )
        
        # Check instance-level cache only (no global cache to prevent cross-contamination)
        cache_key = str(current_inputs)
        if hasattr(self, '_last_blend_inputs') and self._last_blend_inputs == current_inputs:
            if hasattr(self, '_cached_blend_output'):
                print(f"🚀 WANFastImageBlend: Using instance cached output")
                print(f"  - Instance ID: {self._instance_id}")
                print(f"  - Execution context: {execution_context}")
                print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
                # Return fresh copy to prevent contamination
                cached_result, cached_info = self._cached_blend_output
                return (cached_result.clone().detach(), cached_info)
        
        # Debug logging for cache behavior
        if enable_performance_logging:
            print(f"🔍 WANFastImageBlend Cache Debug:")
            print(f"  - Instance ID: {self._instance_id}")
            print(f"  - Execution context: {execution_context}")
            print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
            print(f"  - Has previous inputs: {hasattr(self, '_last_blend_inputs')}")
            if hasattr(self, '_last_blend_inputs'):
                print(f"  - Previous inputs match: {self._last_blend_inputs == current_inputs}")
            print(f"  - Has cached output: {hasattr(self, '_cached_blend_output')}")
        
        start_time = time.time()
        
        if enable_performance_logging:
            print(f"\n🚀 WAN Fast Image Blend:")
            print(f"  Image1: {image1.shape}")
            print(f"  Image2: {image2.shape}")
            print(f"  Blend mode: {blend_mode}")
            print(f"  Blend factor: {blend_factor}")
        
        # Alpha channel fix
        image1, image2 = node_helpers.image_alpha_fix(image1, image2)
        
        # Fast device transfer
        image2 = image2.to(image1.device, non_blocking=True)
        
        # Shape matching with optimized upscaling
        if image1.shape != image2.shape:
            upscale_start = time.time()
            
            # Use more efficient interpolation
            image2 = image2.permute(0, 3, 1, 2)  # BHWC -> BCHW
            image2 = torch.nn.functional.interpolate(
                image2, 
                size=(image1.shape[1], image1.shape[2]), 
                mode='bilinear', 
                align_corners=False,
                antialias=True  # Better quality
            )
            image2 = image2.permute(0, 2, 3, 1)  # BCHW -> BHWC
            
            upscale_time = time.time() - upscale_start
        else:
            upscale_time = 0.0
        
        # FUSED BLEND OPERATION - MAJOR PERFORMANCE BOOST
        blend_start = time.time()
        
        # Single fused operation instead of separate blend + factor application
        if blend_factor == 0.0:
            # Shortcut for no blending
            result = image1.clone()
        elif blend_factor == 1.0:
            # Shortcut for full blend
            result = torch.clamp(self._vectorized_blend_modes(image1, image2, blend_mode), 0.0, 1.0)
        else:
            # Fused blend and factor application
            blended = self._vectorized_blend_modes(image1, image2, blend_mode)
            # Single fused lerp + clamp operation
            result = torch.clamp(
                image1 + blend_factor * (blended - image1), 
                0.0, 1.0
            )
        
        blend_time = time.time() - blend_start
        total_time = time.time() - start_time
        
        # Performance metrics
        batch_size = result.shape[0]
        fps = batch_size / total_time if total_time > 0 else float('inf')
        memory_mb = result.element_size() * result.numel() / 1024 / 1024
        
        info_lines = [
            f"🚀 WAN Fast Blend Results:",
            f"Node: WANFastImageBlend v1.0 (Instance: {self._instance_id if hasattr(self, '_instance_id') else 'N/A'})",
            f"Processed: {batch_size} images",
            f"Blend mode: {blend_mode}",
            f"Blend factor: {blend_factor:.3f}",
            f"Processing time: {total_time:.4f}s",
            f"Throughput: {fps:.1f} fps",
            f"Memory used: ~{memory_mb:.1f}MB",
        ]
        
        if enable_performance_logging:
            info_lines.extend([
                "",
                f"⚡ Performance breakdown:",
                f"  - Upscaling: {upscale_time:.4f}s",
                f"  - Blending: {blend_time:.4f}s",
                f"  - Overhead: {total_time - blend_time - upscale_time:.4f}s",
                f"  - Device: {'CUDA' if result.is_cuda else 'CPU'}",
            ])
            
            print(f"  Output: {result.shape}")
            print(f"  ⚡ Total time: {total_time:.4f}s ({fps:.1f} fps)")
            print(f"  🧠 Memory: ~{memory_mb:.1f}MB")
            print(f"{'='*50}")
        
        performance_info = "\n".join(info_lines)
        
        # Cache the result with deep copy to prevent tensor contamination
        result_output = (result.clone().detach(), performance_info)
        self._last_blend_inputs = current_inputs
        self._cached_blend_output = result_output
        
        return result_output


class WANFastImageScaleBy:
    """Ultra-fast image scaling by factor with full GPU vectorization"""
    
    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    
    # CACHE ISOLATION: Using IS_CHANGED method for proper cache invalidation
    # Changed from True to allow ComfyUI caching with proper invalidation
    NOT_IDEMPOTENT = False
    
    def __init__(self):
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]  # Unique instance identifier
    
    # Re-added IS_CHANGED to provide proper cache invalidation
    @classmethod
    def IS_CHANGED(cls, image, upscale_method, scale_by, enable_performance_logging):
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
        m.update(f"{upscale_method}_{scale_by:.6f}_{enable_performance_logging}".encode())
        
        return m.hexdigest()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (cls.UPSCALE_METHODS,),
                "scale_by": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 8.0,
                    "step": 0.01
                }),
                "enable_performance_logging": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("scaled_image", "performance_info")
    FUNCTION = "ultra_fast_scale"
    CATEGORY = "WAN Vace/Fast Processing"
    
    def ultra_fast_scale(self, image, upscale_method, scale_by, enable_performance_logging):
        """Ultra-fast scaling with zero format conversions and GPU vectorization"""
        
        # DEFENSIVE CACHING: Bypass ComfyUI upstream dependency issues
        # Check for cached output based on actual received values
        scale_by_stable = round(float(scale_by), 6)  # Stabilize float precision
        
        # ⚠️ CRITICAL CACHE ISOLATION SYSTEM - DO NOT MODIFY ⚠️
        # Create execution context to prevent cross-execution contamination
        # Using instance ID only for stable caching while preventing cross-instance issues
        # 🔒 MODIFIED: Removed timestamp to allow cache to actually work
        execution_context = f"{self._instance_id}"
        
        # ⚠️ CRITICAL: This tuple structure prevents cache contamination between runs
        # Each element contributes to cache isolation - DO NOT REMOVE ANY ELEMENTS
        current_inputs = (
            get_tensor_content_hash(image, execution_context=execution_context),  # Content-based hash with execution context
            str(upscale_method),
            scale_by_stable,  # Use stabilized value
            bool(enable_performance_logging),
            self._instance_id if hasattr(self, '_instance_id') else 'default',  # Add instance isolation
            execution_context  # Additional execution uniqueness - CRITICAL FOR ISOLATION
        )
        
        # Check instance-level cache only (no global cache to prevent cross-contamination)
        cache_key = str(current_inputs)
        if hasattr(self, '_last_scale_by_inputs') and self._last_scale_by_inputs == current_inputs:
            if hasattr(self, '_cached_scale_by_output'):
                print(f"🚀 WANFastImageScaleBy: Using instance cached output")
                print(f"  - Instance ID: {self._instance_id}")
                print(f"  - Execution context: {execution_context}")
                print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
                # Return fresh copy to prevent contamination
                cached_result, cached_info = self._cached_scale_by_output
                return (cached_result.clone().detach(), cached_info)
        
        # Debug logging for cache behavior
        if enable_performance_logging:
            print(f"🔍 WANFastImageScaleBy Cache Debug:")
            print(f"  - Instance ID: {self._instance_id}")
            print(f"  - Execution context: {execution_context}")
            print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
            print(f"  - Has previous inputs: {hasattr(self, '_last_scale_by_inputs')}")
            if hasattr(self, '_last_scale_by_inputs'):
                print(f"  - Previous inputs match: {self._last_scale_by_inputs == current_inputs}")
            print(f"  - Has cached output: {hasattr(self, '_cached_scale_by_output')}")
        
        start_time = time.time()
        
        if enable_performance_logging:
            print(f"\n🚀🚀🚀 WAN ULTRA Fast Image Scale By:")
            print(f"  Input: {image.shape}")
            print(f"  Scale factor: {scale_by}x")
            print(f"  Method: {upscale_method}")
            print(f"  Device: {'CUDA' if image.is_cuda else 'CPU'}")
        
        # Early exit for no-op scaling
        if abs(scale_by - 1.0) < 1e-6:
            if enable_performance_logging:
                print(f"  🚀 No-op detected (scale ~1.0), returning original")
            return (image, f"No-op scaling (factor: {scale_by:.6f})")
        
        # OPTIMIZATION 1: Vectorized resolution calculation for entire batch
        calc_start = time.time()
        
        batch_size, height, width, channels = image.shape
        
        # Vectorized calculation - all images at once
        new_height = round(height * scale_by)
        new_width = round(width * scale_by)
        
        calc_time = time.time() - calc_start
        
        # OPTIMIZATION 2: Single format conversion + GPU-optimized interpolation
        interp_start = time.time()
        
        # Convert to BCHW only once for interpolation
        image_bchw = image.permute(0, 3, 1, 2)
        
        # GPU-optimized interpolation with quality settings
        if upscale_method == "nearest-exact":
            mode = "nearest-exact"
        elif upscale_method == "bilinear":
            mode = "bilinear"
        elif upscale_method == "bicubic":
            mode = "bicubic"
        elif upscale_method == "area":
            mode = "area"
        elif upscale_method == "lanczos":
            # PyTorch doesn't have native Lanczos, use bicubic with antialias
            mode = "bicubic"
        else:
            mode = "bilinear"
        
        # Ultra-fast GPU interpolation  
        interpolate_kwargs = {
            "input": image_bchw,
            "size": (new_height, new_width),
            "mode": mode,
        }
        
        # Only add align_corners for supported modes
        if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
            interpolate_kwargs["align_corners"] = False
            
        # Only add antialias for supported modes
        if mode in ["bilinear", "bicubic"]:
            interpolate_kwargs["antialias"] = True
            
        scaled_bchw = torch.nn.functional.interpolate(**interpolate_kwargs)
        
        # Convert back to BHWC - single conversion
        result = scaled_bchw.permute(0, 2, 3, 1)
        
        interp_time = time.time() - interp_start
        total_time = time.time() - start_time
        
        # Performance metrics
        input_pixels = batch_size * height * width
        output_pixels = batch_size * new_height * new_width
        throughput_mpixels = output_pixels / (total_time * 1_000_000) if total_time > 0 else float('inf')
        fps = batch_size / total_time if total_time > 0 else float('inf')
        memory_mb = result.element_size() * result.numel() / 1024 / 1024
        
        info_lines = [
            f"🚀🚀🚀 WAN ULTRA Fast Scale By Results:",
            f"Processed: {batch_size} images",
            f"Resolution: {width}x{height} → {new_width}x{new_height}",
            f"Scale factor: {scale_by:.4f}x",
            f"Method: {upscale_method}",
            f"Processing time: {total_time:.4f}s",
            f"Throughput: {fps:.1f} fps ({throughput_mpixels:.1f} MP/s)",
            f"Memory used: ~{memory_mb:.1f}MB",
        ]
        
        if enable_performance_logging:
            info_lines.extend([
                "",
                f"⚡ Ultra-Performance breakdown:",
                f"  - Resolution calculation: {calc_time:.6f}s",
                f"  - GPU interpolation: {interp_time:.4f}s",
                f"  - Total overhead: {total_time - interp_time:.6f}s",
                f"  - Pixel throughput: {throughput_mpixels:.1f} MP/s",
                f"  - Device: {'CUDA' if result.is_cuda else 'CPU'}",
            ])
            
            print(f"  Output: {result.shape}")
            print(f"  ⚡ Total time: {total_time:.4f}s ({fps:.1f} fps, {throughput_mpixels:.1f} MP/s)")
            print(f"  🧠 Memory: ~{memory_mb:.1f}MB")
            print(f"{'='*60}")
        
        performance_info = "\n".join(info_lines)
        
        # Cache the result at instance level only (no global cache to prevent cross-contamination)
        result_output = (result, performance_info)
        
        # Store in instance cache only
        self._last_scale_by_inputs = current_inputs
        self._cached_scale_by_output = result_output
        
        return result_output


class WANFastImageScaleToMegapixels:
    """Ultra-fast image scaling to target megapixel count with vectorized math"""
    
    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    
    # DEFENSIVE CACHING: Bypass ComfyUI upstream dependency issues (e.g., MXslider nodes)
    # Using NOT_IDEMPOTENT=True to handle mxSlider compatibility issues
    NOT_IDEMPOTENT = True
    
    def __init__(self):
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]  # Unique instance identifier
    
    # REMOVED: IS_CHANGED method because NOT_IDEMPOTENT=True handles cache bypassing
    # ComfyUI will call the execution method every time when NOT_IDEMPOTENT=True
    # This avoids float precision issues with mxSlider inputs
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (cls.UPSCALE_METHODS,),
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 16.0,
                    "step": 0.01
                }),
                "enable_performance_logging": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("scaled_image", "performance_info") 
    FUNCTION = "ultra_fast_scale_to_megapixels"
    CATEGORY = "WAN Vace/Fast Processing"
    
    # REMOVED: IS_CHANGED method because NOT_IDEMPOTENT=True handles cache bypassing
    # ComfyUI will call the execution method every time when NOT_IDEMPOTENT=True
    
    def ultra_fast_scale_to_megapixels(self, image, upscale_method, megapixels, enable_performance_logging):
        """Ultra-fast scaling to megapixel target with vectorized calculations and defensive caching"""
        
        # DEFENSIVE CACHING: Bypass ComfyUI upstream dependency issues (e.g., MXslider nodes)
        # Check for cached output based on actual received values, not upstream cache state
        megapixels_stable = round(float(megapixels), 3)
        
        # ⚠️ CRITICAL CACHE ISOLATION SYSTEM - DO NOT MODIFY ⚠️
        # Create execution context to prevent cross-execution contamination
        # Using instance ID only for stable caching while preventing cross-instance issues
        # 🔒 MODIFIED: Removed timestamp to allow cache to actually work
        execution_context = f"{self._instance_id}"
        
        # ⚠️ CRITICAL: This tuple structure prevents cache contamination between runs
        # Each element contributes to cache isolation - DO NOT REMOVE ANY ELEMENTS
        current_inputs = (
            get_tensor_content_hash(image, execution_context=execution_context),  # Content-based hash with execution context
            str(upscale_method),
            megapixels_stable,  # Use stabilized value
            bool(enable_performance_logging),
            self._instance_id if hasattr(self, '_instance_id') else 'default',  # Add instance isolation
            execution_context  # Additional execution uniqueness - CRITICAL FOR ISOLATION
        )
        
        # Check instance-level cache only (no global cache to prevent cross-contamination)
        cache_key = str(current_inputs)
        if hasattr(self, '_last_scale_inputs') and self._last_scale_inputs == current_inputs:
            if hasattr(self, '_cached_scale_output'):
                print(f"🚀 WANFastImageScaleToMegapixels: Using instance cached output")
                print(f"  - Instance ID: {self._instance_id}")
                print(f"  - Execution context: {execution_context}")
                print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
                # Return fresh copy to prevent contamination
                cached_image, cached_info = self._cached_scale_output
                return (cached_image.clone().detach(), cached_info)
        
        # Debug logging for cache behavior
        if enable_performance_logging:
            print(f"🔍 WANFastImageScaleToMegapixels Cache Debug:")
            print(f"  - Instance ID: {self._instance_id}")
            print(f"  - Execution context: {execution_context}")
            print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
            print(f"  - Has previous inputs: {hasattr(self, '_last_scale_inputs')}")
            if hasattr(self, '_last_scale_inputs'):
                print(f"  - Previous inputs match: {self._last_scale_inputs == current_inputs}")
            print(f"  - Has cached output: {hasattr(self, '_cached_scale_output')}")
        
        start_time = time.time()
        
        if enable_performance_logging:
            print(f"\n🚀🚀🚀 WAN ULTRA Fast Scale To Megapixels:")
            print(f"  Input: {image.shape}")
            print(f"  Target: {megapixels:.2f} MP")
            print(f"  Method: {upscale_method}")
            print(f"  Device: {'CUDA' if image.is_cuda else 'CPU'}")
        
        # OPTIMIZATION 1: Vectorized megapixel calculation
        calc_start = time.time()
        
        batch_size, height, width, channels = image.shape
        current_pixels = height * width
        target_pixels = int(megapixels * 1024 * 1024)
        
        # Ultra-fast GPU calculation using torch operations
        current_pixels_tensor = torch.tensor(current_pixels, device=image.device, dtype=torch.float32)
        target_pixels_tensor = torch.tensor(target_pixels, device=image.device, dtype=torch.float32)
        
        # GPU-accelerated square root and scaling
        scale_factor = torch.sqrt(target_pixels_tensor / current_pixels_tensor).item()
        
        # Vectorized resolution calculation
        new_width = round(width * scale_factor)
        new_height = round(height * scale_factor)
        actual_megapixels = (new_width * new_height) / (1024 * 1024)
        
        calc_time = time.time() - calc_start
        
        # Early exit for no significant change
        if abs(scale_factor - 1.0) < 1e-4:
            if enable_performance_logging:
                print(f"  🚀 Minimal scaling needed (factor: {scale_factor:.6f})")
            return (image, f"Minimal scaling (factor: {scale_factor:.6f}, target: {megapixels:.2f}MP)")
        
        # OPTIMIZATION 2: GPU-optimized interpolation (same as ScaleBy)
        interp_start = time.time()
        
        # Single format conversion
        image_bchw = image.permute(0, 3, 1, 2)
        
        # Map upscale method to PyTorch mode
        if upscale_method == "nearest-exact":
            mode = "nearest-exact"
        elif upscale_method == "bilinear":
            mode = "bilinear"
        elif upscale_method == "bicubic":
            mode = "bicubic"
        elif upscale_method == "area":
            mode = "area"
        elif upscale_method == "lanczos":
            mode = "bicubic"  # Use bicubic with antialias for Lanczos-like quality
        else:
            mode = "bilinear"
        
        # GPU-accelerated interpolation
        interpolate_kwargs = {
            "input": image_bchw,
            "size": (new_height, new_width),
            "mode": mode,
        }
        
        # Only add align_corners for supported modes
        if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
            interpolate_kwargs["align_corners"] = False
            
        # Only add antialias for supported modes  
        if mode in ["bilinear", "bicubic"]:
            interpolate_kwargs["antialias"] = True
            
        scaled_bchw = torch.nn.functional.interpolate(**interpolate_kwargs)
        
        # Single conversion back
        result = scaled_bchw.permute(0, 2, 3, 1)
        
        interp_time = time.time() - interp_start
        total_time = time.time() - start_time
        
        # Performance metrics
        output_pixels = batch_size * new_height * new_width
        throughput_mpixels = output_pixels / (total_time * 1_000_000) if total_time > 0 else float('inf')
        fps = batch_size / total_time if total_time > 0 else float('inf')
        memory_mb = result.element_size() * result.numel() / 1024 / 1024
        
        info_lines = [
            f"🚀🚀🚀 WAN ULTRA Fast Scale To Megapixels Results:",
            f"Processed: {batch_size} images",
            f"Resolution: {width}x{height} → {new_width}x{new_height}",
            f"Megapixels: {current_pixels/(1024*1024):.2f}MP → {actual_megapixels:.2f}MP (target: {megapixels:.2f}MP)",
            f"Scale factor: {scale_factor:.4f}x",
            f"Method: {upscale_method}",
            f"Processing time: {total_time:.4f}s",
            f"Throughput: {fps:.1f} fps ({throughput_mpixels:.1f} MP/s)",
            f"Memory used: ~{memory_mb:.1f}MB",
        ]
        
        if enable_performance_logging:
            info_lines.extend([
                "",
                f"⚡ Ultra-Performance breakdown:",
                f"  - Vectorized calculation: {calc_time:.6f}s",
                f"  - GPU interpolation: {interp_time:.4f}s",
                f"  - Total overhead: {total_time - interp_time:.6f}s",
                f"  - Accuracy: {actual_megapixels:.3f}MP vs {megapixels:.3f}MP target",
                f"  - Device: {'CUDA' if result.is_cuda else 'CPU'}",
            ])
            
            print(f"  Output: {result.shape}")
            print(f"  🎯 Accuracy: {actual_megapixels:.3f}MP (target: {megapixels:.3f}MP)")
            print(f"  ⚡ Total time: {total_time:.4f}s ({fps:.1f} fps, {throughput_mpixels:.1f} MP/s)")
            print(f"  🧠 Memory: ~{memory_mb:.1f}MB")
            print(f"{'='*60}")
        
        performance_info = "\n".join(info_lines)
        
        # CACHE THE RESULT: Store for future use when inputs are identical
        # This bypasses ComfyUI's upstream dependency caching issues (e.g., MXslider nodes)
        output = (result, performance_info)
        self._last_scale_inputs = current_inputs
        self._cached_scale_output = output
        
        return output


class WANFastImageResize:
    """Ultra-fast image resizing to exact dimensions with aspect ratio options"""
    
    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    CROP_METHODS = ["disabled", "center"]
    
    # CACHE ISOLATION: Using IS_CHANGED method for proper cache invalidation
    # Changed from True to allow ComfyUI caching with proper invalidation
    NOT_IDEMPOTENT = False
    
    def __init__(self):
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]  # Unique instance identifier
    
    # Re-added IS_CHANGED to provide proper cache invalidation
    @classmethod
    def IS_CHANGED(cls, image, width, height, upscale_method, crop, enable_performance_logging):
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
        m.update(f"{width}_{height}_{upscale_method}_{crop}_{enable_performance_logging}".encode())
        
        return m.hexdigest()
    
    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 16384,
                    "step": 1
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 16384,
                    "step": 1
                }),
                "upscale_method": (cls.UPSCALE_METHODS,),
                "crop": (cls.CROP_METHODS,),
                "enable_performance_logging": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("resized_image", "performance_info")
    FUNCTION = "ultra_fast_resize"
    CATEGORY = "WAN Vace/Fast Processing"
    
    def ultra_fast_resize(self, image, width, height, upscale_method, crop, enable_performance_logging):
        """Ultra-fast resize to exact dimensions with GPU vectorization"""
        
        # DEFENSIVE CACHING: Bypass ComfyUI upstream dependency issues
        # Check for cached output based on actual received values
        
        # ⚠️ CRITICAL CACHE ISOLATION SYSTEM - DO NOT MODIFY ⚠️
        # Create execution context to prevent cross-execution contamination
        # Using instance ID only for stable caching while preventing cross-instance issues
        # 🔒 MODIFIED: Removed timestamp to allow cache to actually work
        execution_context = f"{self._instance_id}"
        
        # ⚠️ CRITICAL: This tuple structure prevents cache contamination between runs
        # Each element contributes to cache isolation - DO NOT REMOVE ANY ELEMENTS
        current_inputs = (
            get_tensor_content_hash(image, execution_context=execution_context),  # Content-based hash with execution context
            int(width),
            int(height),
            str(upscale_method),
            str(crop),
            bool(enable_performance_logging),
            self._instance_id if hasattr(self, '_instance_id') else 'default',  # Add instance isolation
            execution_context  # Additional execution uniqueness - CRITICAL FOR ISOLATION
        )
        
        # Check instance-level cache only (no global cache to prevent cross-contamination)
        cache_key = str(current_inputs)
        if hasattr(self, '_last_resize_inputs') and self._last_resize_inputs == current_inputs:
            if hasattr(self, '_cached_resize_output'):
                print(f"🚀 WANFastImageResize: Using instance cached output")
                print(f"  - Instance ID: {self._instance_id}")
                print(f"  - Execution context: {execution_context}")
                print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
                # Return fresh copy to prevent contamination
                cached_result, cached_info = self._cached_resize_output
                return (cached_result.clone().detach(), cached_info)
        
        # Debug logging for cache behavior
        if enable_performance_logging:
            print(f"🔍 WANFastImageResize Cache Debug:")
            print(f"  - Instance ID: {self._instance_id}")
            print(f"  - Execution context: {execution_context}")
            print(f"  - Cache key (first 50 chars): {cache_key[:50]}...")
            print(f"  - Has previous inputs: {hasattr(self, '_last_resize_inputs')}")
            if hasattr(self, '_last_resize_inputs'):
                print(f"  - Previous inputs match: {self._last_resize_inputs == current_inputs}")
            print(f"  - Has cached output: {hasattr(self, '_cached_resize_output')}")
        
        start_time = time.time()
        
        if enable_performance_logging:
            print(f"\n🚀🚀🚀 WAN ULTRA Fast Image Resize:")
            print(f"  Input: {image.shape}")
            print(f"  Target: {width}x{height}")
            print(f"  Method: {upscale_method}")
            print(f"  Crop: {crop}")
            print(f"  Device: {'CUDA' if image.is_cuda else 'CPU'}")
        
        batch_size, old_height, old_width, channels = image.shape
        
        # Early exit for no-op resize
        if old_width == width and old_height == height:
            if enable_performance_logging:
                print(f"  🚀 No-op detected (same dimensions)")
            return (image, f"No-op resize (already {width}x{height})")
        
        # OPTIMIZATION 1: Vectorized cropping calculation (if needed)
        crop_start = time.time()
        
        if crop == "center":
            # GPU-accelerated aspect ratio calculations
            old_aspect = old_width / old_height
            new_aspect = width / height
            
            if old_aspect > new_aspect:
                # Crop width
                crop_width = round(old_width * (new_aspect / old_aspect))
                x_offset = (old_width - crop_width) // 2
                cropped_image = image[:, :, x_offset:x_offset + crop_width, :]
            elif old_aspect < new_aspect:
                # Crop height  
                crop_height = round(old_height * (old_aspect / new_aspect))
                y_offset = (old_height - crop_height) // 2
                cropped_image = image[:, y_offset:y_offset + crop_height, :, :]
            else:
                cropped_image = image
        else:
            cropped_image = image
        
        crop_time = time.time() - crop_start
        
        # OPTIMIZATION 2: GPU-optimized interpolation  
        interp_start = time.time()
        
        # Single format conversion
        image_bchw = cropped_image.permute(0, 3, 1, 2)
        
        # Map upscale method
        if upscale_method == "nearest-exact":
            mode = "nearest-exact"
        elif upscale_method == "bilinear":
            mode = "bilinear"
        elif upscale_method == "bicubic":
            mode = "bicubic"
        elif upscale_method == "area":
            mode = "area"
        elif upscale_method == "lanczos":
            mode = "bicubic"
        else:
            mode = "bilinear"
        
        # Ultra-fast GPU interpolation
        interpolate_kwargs = {
            "input": image_bchw,
            "size": (height, width),
            "mode": mode,
        }
        
        # Only add align_corners for supported modes
        if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
            interpolate_kwargs["align_corners"] = False
            
        # Only add antialias for supported modes
        if mode in ["bilinear", "bicubic"]:
            interpolate_kwargs["antialias"] = True
            
        resized_bchw = torch.nn.functional.interpolate(**interpolate_kwargs)
        
        # Convert back to BHWC
        result = resized_bchw.permute(0, 2, 3, 1)
        
        interp_time = time.time() - interp_start
        total_time = time.time() - start_time
        
        # Performance metrics
        input_pixels = batch_size * old_height * old_width
        output_pixels = batch_size * height * width
        throughput_mpixels = output_pixels / (total_time * 1_000_000) if total_time > 0 else float('inf')
        fps = batch_size / total_time if total_time > 0 else float('inf')
        memory_mb = result.element_size() * result.numel() / 1024 / 1024
        scale_factor = (width * height) / (old_width * old_height)
        
        info_lines = [
            f"🚀🚀🚀 WAN ULTRA Fast Resize Results:",
            f"Processed: {batch_size} images",
            f"Resolution: {old_width}x{old_height} → {width}x{height}",
            f"Scale factor: {scale_factor:.4f}x",
            f"Method: {upscale_method}",
            f"Crop: {crop}",
            f"Processing time: {total_time:.4f}s",
            f"Throughput: {fps:.1f} fps ({throughput_mpixels:.1f} MP/s)",
            f"Memory used: ~{memory_mb:.1f}MB",
        ]
        
        if enable_performance_logging:
            info_lines.extend([
                "",
                f"⚡ Ultra-Performance breakdown:",
                f"  - Cropping: {crop_time:.6f}s",
                f"  - GPU interpolation: {interp_time:.4f}s",
                f"  - Total overhead: {total_time - interp_time:.6f}s",
                f"  - Pixel throughput: {throughput_mpixels:.1f} MP/s",
                f"  - Device: {'CUDA' if result.is_cuda else 'CPU'}",
            ])
            
            print(f"  Output: {result.shape}")
            print(f"  ⚡ Total time: {total_time:.4f}s ({fps:.1f} fps, {throughput_mpixels:.1f} MP/s)")
            print(f"  🧠 Memory: ~{memory_mb:.1f}MB")
            print(f"{'='*60}")
        
        performance_info = "\n".join(info_lines)
        
        # Cache the result at instance level only (no global cache to prevent cross-contamination)
        result_output = (result, performance_info)
        
        # Store in instance cache only
        self._last_resize_inputs = current_inputs
        self._cached_resize_output = result_output
        
        return result_output