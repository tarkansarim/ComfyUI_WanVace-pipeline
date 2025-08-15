"""
Timeline and keyframe management nodes for WAN Vace Pipeline
"""

import json
import numpy as np
import torch
from pathlib import Path
import cv2
import base64
from io import BytesIO
from PIL import Image


class WANVaceKeyframeTimeline:
    """Complete interactive keyframe timeline editor
    
    This node provides an interactive timeline widget for managing keyframes
    with drag-and-drop support, batch processing, and visual feedback.
    """
    
    @staticmethod
    def IS_CHANGED(timeline_frames, fps, frame_darkness, batch_size, keyframe_data="", **kwargs):
        """ComfyUI caching: Return hash of inputs to detect changes"""
        # Smart semantic caching: hash image content and structure separately
        import hashlib
        import json
        
        # Hash structural parameters
        structural_hash = f"{timeline_frames}_{fps}_{frame_darkness}_{batch_size}"
        
        # Hash keyframe content semantically (including key metadata)
        content_hash = ""
        if keyframe_data:
            try:
                data = json.loads(keyframe_data)
                keyframes = data.get("keyframes", {})
                ignore_held_frames_mask = data.get("ignore_held_frames_mask", False)
                # Include optional numeric metadata so changes invalidate cache even if widgets didn't emit
                meta_fps = data.get("fps", None)
                meta_batch = data.get("batch_size", None)
                meta_fd = data.get("frame_darkness", None)
                
                # Create stable content fingerprints for each keyframe
                content_parts = []
                for frame_str in sorted(keyframes.keys(), key=lambda s: int(float(s))):
                    kf_data = keyframes[frame_str]
                    kf_id = str(kf_data.get("id", ""))
                    kf_enabled = bool(kf_data.get("enabled", True))
                    kf_hold = int(kf_data.get("hold", 1))
                    # Hash image content snippet deterministically
                    image_data = kf_data.get("image", "") or ""
                    if image_data:
                        image_snippet = image_data[:100] + image_data[-100:] if len(image_data) > 200 else image_data
                        image_digest = hashlib.md5(image_snippet.encode()).hexdigest()[:16]
                    else:
                        image_digest = "no_image"
                    content_parts.append(f"frame={frame_str}|id={kf_id}|enabled={int(kf_enabled)}|hold={kf_hold}|img={image_digest}")
                # Include mask behavior flag in the hash
                content_parts.append(f"ignore_held_frames_mask={int(bool(ignore_held_frames_mask))}")
                # Include numeric metadata if present
                if meta_fps is not None:
                    content_parts.append(f"fps={int(round(float(meta_fps)))}")
                if meta_batch is not None:
                    content_parts.append(f"batch_size={int(round(float(meta_batch)))}")
                if meta_fd is not None:
                    try:
                        content_parts.append(f"frame_darkness={float(meta_fd):.6f}")
                    except Exception:
                        pass
                content_hash = hashlib.md5("|".join(content_parts).encode()).hexdigest()[:16]
                
            except Exception as e:
                # Fallback to simple keyframe_data hash if parsing fails
                content_hash = hashlib.md5(keyframe_data.encode()).hexdigest()[:16]
        
        # Combine structural and content hashes
        combined_input = f"{structural_hash}_{content_hash}"
        final_hash = hashlib.md5(combined_input.encode()).hexdigest()
        
        print(f"[WANVaceKeyframeTimeline] IS_CHANGED hash: {final_hash[:8]}... (content: {content_hash[:8]}, keyframe_data length: {len(keyframe_data)})")
        return final_hash
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_frames": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Total number of frames in the timeline"
                }),
                "fps": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Target FPS for timeline"
                }),
                "frame_darkness": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Darkness level for padded frames (0=black, 1=white)"
                }),
                "batch_size": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Batch size for processing"
                }),
            },
            "optional": {
                "keyframe_data": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Keyframe data from timeline widget"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("reference_frames", "preview_frames", "mask", "frame_count", "fps", "keyframe_info")
    FUNCTION = "create_timeline"
    CATEGORY = "WAN/timeline"
    OUTPUT_NODE = False

    def create_timeline(self, timeline_frames, fps, frame_darkness, batch_size, keyframe_data="", **kwargs):
        """Create timeline with reference frames, preview frames, and mask"""
        
        # Convert timeline_frames to integer to ensure PyTorch operations work
        timeline_frames = int(round(timeline_frames))
        # Convert batch_size to integer to fix boundary calculations
        batch_size = int(round(batch_size))
        # Convert fps to integer for consistent timing calculations
        fps = int(round(fps))
        
        # Smart caching: check if we can use cached output with semantic comparison
        import hashlib
        import json
        
        # Create semantic fingerprint for caching (same logic as IS_CHANGED)
        structural_fingerprint = f"{timeline_frames}_{fps}_{frame_darkness}_{batch_size}"
        content_fingerprint = ""
        
        if keyframe_data:
            try:
                data = json.loads(keyframe_data)
                keyframes = data.get("keyframes", {})
                ignore_held_frames_mask = data.get("ignore_held_frames_mask", False)
                meta_fps = data.get("fps", None)
                meta_batch = data.get("batch_size", None)
                meta_fd = data.get("frame_darkness", None)
                content_parts = []
                for frame_str in sorted(keyframes.keys(), key=lambda s: int(float(s))):
                    kf_data = keyframes[frame_str]
                    kf_id = str(kf_data.get("id", ""))
                    kf_enabled = bool(kf_data.get("enabled", True))
                    kf_hold = int(kf_data.get("hold", 1))
                    image_data = kf_data.get("image", "") or ""
                    if image_data:
                        image_snippet = image_data[:100] + image_data[-100:] if len(image_data) > 200 else image_data
                        image_digest = hashlib.md5(image_snippet.encode()).hexdigest()[:16]
                    else:
                        image_digest = "no_image"
                    content_parts.append(f"frame={frame_str}|id={kf_id}|enabled={int(kf_enabled)}|hold={kf_hold}|img={image_digest}")
                content_parts.append(f"ignore_held_frames_mask={int(bool(ignore_held_frames_mask))}")
                if meta_fps is not None:
                    content_parts.append(f"fps={int(round(float(meta_fps)))}")
                if meta_batch is not None:
                    content_parts.append(f"batch_size={int(round(float(meta_batch)))}")
                if meta_fd is not None:
                    try:
                        content_parts.append(f"frame_darkness={float(meta_fd):.6f}")
                    except Exception:
                        pass
                content_fingerprint = hashlib.md5("|".join(content_parts).encode()).hexdigest()[:16]
            except:
                content_fingerprint = hashlib.md5(keyframe_data.encode()).hexdigest()[:16]
        
        semantic_fingerprint = f"{structural_fingerprint}_{content_fingerprint}"
        
        # Check semantic cache
        if hasattr(self, '_last_semantic_fingerprint') and self._last_semantic_fingerprint == semantic_fingerprint:
            if hasattr(self, '_cached_output'):
                print(f"[WANVaceKeyframeTimeline] Using cached output (semantic fingerprint unchanged: {semantic_fingerprint[:16]}...)")
                return self._cached_output
        
        print(f"[WANVaceKeyframeTimeline] Processing timeline (semantic fingerprint changed or no cache: {semantic_fingerprint[:16]}...)")
        
        # Parse keyframe data if provided
        ignore_held_frames_mask = False  # Default value
        if keyframe_data:
            try:
                # Check if we have cached parsed data to avoid re-parsing
                if hasattr(self, '_cached_keyframe_data') and self._cached_keyframe_data_str == keyframe_data:
                    data = self._cached_keyframe_data
                    keyframes = data.get("keyframes", {})
                    timeline_frames = data.get("frames", timeline_frames)
                    # Optionally override numeric parameters from metadata if present
                    if 'fps' in data:
                        try:
                            fps = int(round(float(data['fps'])))
                        except Exception:
                            pass
                    if 'batch_size' in data:
                        try:
                            batch_size = int(round(float(data['batch_size'])))
                        except Exception:
                            pass
                    if 'frame_darkness' in data:
                        try:
                            frame_darkness = float(data['frame_darkness'])
                        except Exception:
                            pass
                    ignore_held_frames_mask = data.get("ignore_held_frames_mask", False)
                    print(f"[WANVaceKeyframeTimeline] Using cached keyframe data (cache hit)")
                    # Check if keyframes have images
                    image_count = sum(1 for kf in keyframes.values() if kf.get("image"))
                    print(f"  - Cached data has {image_count} keyframes with images")
                else:
                    # Parse and cache
                    data = json.loads(keyframe_data)
                    self._cached_keyframe_data = data
                    self._cached_keyframe_data_str = keyframe_data
                    keyframes = data.get("keyframes", {})
                    timeline_frames = data.get("frames", timeline_frames)
                    if 'fps' in data:
                        try:
                            fps = int(round(float(data['fps'])))
                        except Exception:
                            pass
                    if 'batch_size' in data:
                        try:
                            batch_size = int(round(float(data['batch_size'])))
                        except Exception:
                            pass
                    if 'frame_darkness' in data:
                        try:
                            frame_darkness = float(data['frame_darkness'])
                        except Exception:
                            pass
                    ignore_held_frames_mask = data.get("ignore_held_frames_mask", False)
                    
                    # Debug: print the structure (only on new data)
                    print(f"[WANVaceKeyframeTimeline] Received new keyframe_data:")
                    print(f"  - Timeline frames: {timeline_frames}")
                    print(f"  - FPS: {fps}")
                    print(f"  - Batch size: {batch_size}")
                    print(f"  - Frame darkness: {frame_darkness}")
                    print(f"  - Number of keyframes: {len(keyframes)}")
                    print(f"  - Ignore held frames for mask: {ignore_held_frames_mask}")
                    for frame, kf in list(keyframes.items())[:3]:  # Show first 3
                        print(f"  - Frame {frame}: has_image={kf.get('image') is not None}, path={kf.get('path', 'N/A')}")
            except Exception as e:
                print(f"[WANVaceKeyframeTimeline] Error parsing keyframe_data: {e}")
                keyframes = {}
        else:
            keyframes = {}
        
        # Extract enabled keyframes and their images
        enabled_keyframes = {}
        keyframe_images = {}  # Store as dict mapped by frame number
        keyframe_ids_by_frame = {}
        
        # Initialize persistent image cache if needed
        if not hasattr(self, '_keyframe_images'):
            self._keyframe_images = {}
            print(f"[WANVaceKeyframeTimeline] Initialized new _keyframe_images cache")
        # Initialize persistent ID-based cache if needed
        if not hasattr(self, '_keyframe_images_by_id'):
            self._keyframe_images_by_id = {}
            print(f"[WANVaceKeyframeTimeline] Initialized new _keyframe_images_by_id cache")
        
        # Initialize decoded image cache if needed
        if not hasattr(self, '_image_cache'):
            self._image_cache = {}
        
        # Log current cache state
        print(f"[WANVaceKeyframeTimeline] Cache state before processing:")
        print(f"  - _keyframe_images has {len(self._keyframe_images)} entries")
        print(f"  - _image_cache has {len(self._image_cache)} entries")
        
        # Process each keyframe
        missing_images = []
        for frame_str, kf_data in keyframes.items():
            # Only process enabled keyframes
            frame_num = int(float(frame_str))
            kf_enabled = bool(kf_data.get("enabled", True))
            if not kf_enabled:
                continue
            enabled_keyframes[frame_num] = kf_data
            keyframe_ids_by_frame[frame_num] = str(kf_data.get("id", ""))
            
            # Try multiple methods to get the image
            image_found = False
            
            # Method 1: Check if image is in current keyframe data
            if kf_data.get("image"):
                try:
                    image_data = kf_data["image"]
                    
                    # Check if we have this image cached
                    snippet = image_data[:100] + image_data[-100:] if len(image_data) > 200 else image_data
                    cache_key = f"img_{hashlib.md5(snippet.encode()).hexdigest()[:16]}"
                    if cache_key in self._image_cache:
                        img_tensor = self._image_cache[cache_key]
                        print(f"  - Frame {frame_num}: Using decoded cache (cache_key: {cache_key[:20]}...)")
                    else:
                        # Decode base64 image
                        if image_data.startswith("data:image"):
                            # Remove data URL prefix
                            image_data = image_data.split(",", 1)[1]
                        
                        # Decode base64
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(BytesIO(image_bytes))
                        
                        # Convert to RGB if necessary
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                    
                        # Convert to numpy array and then to tensor
                        img_array = np.array(image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array)
                        
                        # Cache the decoded image
                        self._image_cache[cache_key] = img_tensor
                        print(f"  - Frame {frame_num}: Decoded new image, shape={img_tensor.shape}")
                    
                    keyframe_images[frame_num] = img_tensor
                    # Also store in persistent caches
                    self._keyframe_images[frame_num] = img_tensor
                    kf_id = str(kf_data.get("id", ""))
                    if kf_id:
                        self._keyframe_images_by_id[kf_id] = img_tensor
                    image_found = True
                    
                except Exception as e:
                    print(f"[WANVaceKeyframeTimeline] Error decoding keyframe image at frame {frame_num}: {e}")
                    # Continue to fallback methods
            
            # Method 2: Check persistent ID-based cache
            if not image_found and hasattr(kf_data, 'get') and kf_data.get('id'):
                kf_id = str(kf_data['id'])
                if kf_id in self._keyframe_images_by_id:
                    keyframe_images[frame_num] = self._keyframe_images_by_id[kf_id]
                    print(f"  - Frame {frame_num}: Using ID cache (id={kf_id})")
                    image_found = True
            
            # Method 3: Check persistent frame cache (last resort)
            if not image_found and frame_num in self._keyframe_images:
                keyframe_images[frame_num] = self._keyframe_images[frame_num]
                print(f"  - Frame {frame_num}: Using persistent frame cache (last resort)")
                image_found = True
            
            if not image_found:
                missing_images.append(frame_num)
                print(f"  - Frame {frame_num}: WARNING - No image found in any cache!")
        
        # Log summary of image loading
        print(f"[WANVaceKeyframeTimeline] Image loading summary:")
        print(f"  - Total enabled keyframes: {len(enabled_keyframes)}")
        print(f"  - Successfully loaded images: {len(keyframe_images)}")
        print(f"  - Missing images: {len(missing_images)}")
        if missing_images:
            print(f"  - Missing frames: {missing_images}")
        
        # Update cache state
        print(f"[WANVaceKeyframeTimeline] Cache state after processing:")
        print(f"  - _keyframe_images has {len(self._keyframe_images)} entries")
        print(f"  - _image_cache has {len(self._image_cache)} entries")
        
        # Initialize with default dimensions - will update if we have keyframes
        height, width = 512, 512
        dimensions_set = False
        
        # Create initial tensors - may recreate if dimensions change
        reference_frames = torch.full((timeline_frames, height, width, 3), frame_darkness, dtype=torch.float32)
        preview_frames = torch.full((timeline_frames, height, width, 3), frame_darkness, dtype=torch.float32)
        mask = torch.ones((timeline_frames, height, width), dtype=torch.float32)
        
        # Sort keyframes by position
        sorted_keyframes = sorted(enabled_keyframes.items(), key=lambda x: x[0])
        
        # Place keyframes
        placed_keyframes = 0
        
        for frame_num, kf_data in sorted_keyframes:
            if frame_num >= timeline_frames:
                continue
                
            # Get the keyframe image
            if frame_num in keyframe_images:
                kf_image = keyframe_images[frame_num]
                
                # Update dimensions from first keyframe
                if not dimensions_set:
                    height, width = kf_image.shape[0:2]
                    dimensions_set = True
                    print(f"[WANVaceKeyframeTimeline] Updated dimensions from first keyframe: {height}x{width}")
                    
                    # Recreate tensors with correct dimensions
                    reference_frames = torch.full((timeline_frames, height, width, 3), frame_darkness, dtype=torch.float32)
                    preview_frames = torch.full((timeline_frames, height, width, 3), frame_darkness, dtype=torch.float32)
                    mask = torch.ones((timeline_frames, height, width), dtype=torch.float32)
                
                # Resize if needed
                if kf_image.shape[0] != height or kf_image.shape[1] != width:
                    # Use PIL for resizing
                    pil_img = Image.fromarray((kf_image.numpy() * 255).astype(np.uint8))
                    pil_img = pil_img.resize((width, height), Image.LANCZOS)
                    kf_image = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
                
                # Get hold duration for this keyframe
                keyframe_hold = int(kf_data.get("hold", 1))  # Default to 1 if not specified
                # Debug: print what's in kf_data
                if frame_num < 5:  # Only print first few to avoid spam
                    print(f"  - Frame {frame_num} kf_data keys: {list(kf_data.keys())}, hold value: {kf_data.get('hold', 'NOT PRESENT')}")
                
                # Place keyframe in reference frames - always at the keyframe position
                reference_frames[frame_num] = kf_image
                
                # Mark the keyframe position in the mask
                mask[frame_num] = 0.0
                
                # Handle hold duration if > 1
                if keyframe_hold > 1:
                    for j in range(1, keyframe_hold):  # Start from 1, not 0
                        if frame_num + j < timeline_frames:
                            # Place held frame
                            reference_frames[frame_num + j] = kf_image
                            
                            # Mark in mask only if NOT ignoring held frames
                            if not ignore_held_frames_mask:
                                mask[frame_num + j] = 0.0
                
                placed_keyframes += 1
                print(f"  - Placed keyframe at frame {frame_num} with hold duration {keyframe_hold}")
        
        # Fill preview frames - hold each keyframe until the next one (like video playback)
        current_keyframe = None
        
        # Final fallback: If we're missing images, try one more time from cache
        if len(keyframe_images) < len(enabled_keyframes) and hasattr(self, '_keyframe_images'):
            print(f"[WANVaceKeyframeTimeline] Final fallback: attempting to recover missing images from cache")
            for frame_num in enabled_keyframes:
                if frame_num not in keyframe_images and frame_num in self._keyframe_images:
                    keyframe_images[frame_num] = self._keyframe_images[frame_num]
                    print(f"  - Recovered frame {frame_num} from cache")
        
        # Ensure we have keyframe_images even if cache was used
        if not keyframe_images and hasattr(self, '_keyframe_images'):
            keyframe_images = self._keyframe_images
            print(f"[WANVaceKeyframeTimeline] Using entire cached keyframe images: {len(keyframe_images)} images")
        
        # Double-check we actually have images
        if not keyframe_images:
            print(f"[WANVaceKeyframeTimeline] CRITICAL WARNING: No keyframe images found!")
            print(f"  - enabled_keyframes: {list(enabled_keyframes.keys())}")
            print(f"  - has _keyframe_images: {hasattr(self, '_keyframe_images')}")
            if hasattr(self, '_keyframe_images'):
                print(f"  - _keyframe_images keys: {list(self._keyframe_images.keys())}")
            print(f"  - This will result in blank output!")
        else:
            print(f"[WANVaceKeyframeTimeline] Final image count: {len(keyframe_images)} images available")
        
        for i in range(timeline_frames):
            # Check if there's a new keyframe at this position
            if i in keyframe_images:
                current_keyframe = keyframe_images[i]
                
                # Resize if needed
                if current_keyframe.shape[0] != height or current_keyframe.shape[1] != width:
                    pil_img = Image.fromarray((current_keyframe.numpy() * 255).astype(np.uint8))
                    pil_img = pil_img.resize((width, height), Image.LANCZOS)
                    current_keyframe = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            
            # Use the current keyframe if we have one, otherwise use frame darkness
            if current_keyframe is not None:
                preview_frames[i] = current_keyframe
            else:
                # No keyframe yet, use frame darkness
                preview_frames[i] = torch.full((height, width, 3), frame_darkness, dtype=torch.float32)
        
        # Mask is already in correct format (frames, H, W) for ComfyUI
        # No need to add batch dimension
        
        # Create keyframe info string
        keyframe_info_str = f"Timeline: {timeline_frames} frames @ {fps} FPS ({timeline_frames/fps:.1f}s)\n"
        keyframe_info_str += f"Keyframes: {len(enabled_keyframes)} placed\n"
        if enabled_keyframes:
            positions = sorted(enabled_keyframes.keys())
            keyframe_info_str += f"Positions: {positions}"
        
        # Debug output types
        print(f"[WANVaceKeyframeTimeline] Returning:")
        print(f"  - reference_frames: {type(reference_frames)} shape={reference_frames.shape}")
        print(f"  - preview_frames: {type(preview_frames)} shape={preview_frames.shape}")
        print(f"  - mask: {type(mask)} shape={mask.shape}")
        print(f"  - frame_count: {type(timeline_frames)} value={timeline_frames}")
        print(f"  - fps: {type(fps)} value={fps}")
        print(f"  - keyframe_info: {type(keyframe_info_str)} length={len(keyframe_info_str)}")
        
        # Cache the result for future use with semantic fingerprint
        result = (reference_frames, preview_frames, mask, timeline_frames, fps, keyframe_info_str)
        self._last_semantic_fingerprint = semantic_fingerprint
        self._cached_output = result
        
        print(f"[WANVaceKeyframeTimeline] Cached output for future use (fingerprint: {semantic_fingerprint[:16]}...)")
        
        # Undo contact sheet patch: return all frames as before
        return result