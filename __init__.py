"""
ComfyUI WAN Vace Pipeline
Video frame processing nodes for AI interpolation workflows
"""

import os
import sys

# Print debug info about which file is being loaded
current_file = os.path.abspath(__file__)
print(f"[WAN Vace Pipeline] Loading from: {current_file}")
print(f"[WAN Vace Pipeline] Python version: {sys.version}")
print(f"[WAN Vace Pipeline] Module name: {__name__}")
print(f"[WAN Vace Pipeline] Package path: {os.path.dirname(current_file)}")

print("[WAN Vace Pipeline] Loading custom nodes...")

# Import server endpoints
try:
    from . import mask_editor_server
    print("[WAN Vace Pipeline] Mask editor server endpoints loaded")
except Exception as e:
    print(f"[WAN Vace Pipeline] Failed to load mask editor server: {e}")

try:
    from . import outpainting_editor_server
    print("[WAN Vace Pipeline] Outpainting editor server endpoints loaded")
except Exception as e:
    print(f"[WAN Vace Pipeline] Failed to load outpainting editor server: {e}")

# Define mask editor node directly here to ensure it loads
# TRULY PERSISTENT GLOBAL CACHE that survives module reloads
import sys
_CACHE_ATTR_NAME = "wan_vace_mask_editor_global_cache"

def get_persistent_cache():
    """Get cache that persists across ComfyUI module reloads"""
    if not hasattr(sys.modules[__name__], _CACHE_ATTR_NAME):
        setattr(sys.modules[__name__], _CACHE_ATTR_NAME, {})
    return getattr(sys.modules[__name__], _CACHE_ATTR_NAME)

class WANVaceMaskEditor:
    # BULLETPROOF CACHING: Enable proper ComfyUI caching
    NOT_IDEMPOTENT = False
    
    def __init__(self):
        """Bulletproof initialization - no objects that can cause formatting issues"""
        # Use simple counter instead of UUID to avoid any potential issues
        import time
        # Create simple, safe instance ID
        self._instance_id = f"mask_{int(time.time() * 1000) % 100000}"
        # Safe print with guaranteed string formatting
        try:
            print("[WANVaceMaskEditor] Initialized instance " + str(self._instance_id))
        except:
            print("[WANVaceMaskEditor] Initialized")
    
    @classmethod
    def IS_CHANGED(cls, mask_data="", **kwargs):
        """Deterministic IS_CHANGED - v9 proven pattern"""
        import hashlib
        if mask_data and str(mask_data).strip():
            return hashlib.md5(str(mask_data).encode('utf-8')).hexdigest()
        return ""
    
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "mask_data": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Mask data from mask editor"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "status")
    FUNCTION = "create_mask"
    CATEGORY = "WAN/mask"
    OUTPUT_NODE = True
    
    def create_mask(self, mask_data=""):
        import torch
        import json
        import cv2
        import numpy as np
        import os
        import tempfile
        from pathlib import Path

        # PERSISTENT CACHE SYSTEM: Use same logic as IS_CHANGED
        import hashlib
        
        # Generate cache key EXACTLY like IS_CHANGED method
        if mask_data and str(mask_data).strip():
            cache_key = hashlib.md5(str(mask_data).encode('utf-8')).hexdigest()
        else:
            cache_key = ""
        
        # Get persistent cache that survives module reloads
        persistent_cache = get_persistent_cache()
        
        # Check persistent cache first - BULLETPROOF formatting
        try:
            print("[WANVaceMaskEditor] Global cache check - Instance: " + str(self._instance_id))
            print("  - Cache key: " + str(cache_key)[:16] + "...")
            print("  - Global cache size: " + str(len(persistent_cache)) + " entries")
        except:
            print("[WANVaceMaskEditor] Global cache check")
        
        if cache_key and cache_key in persistent_cache:
            cached_result, cached_info = persistent_cache[cache_key]
            print("[WANVaceMaskEditor] ✅ Using global cached output (inputs unchanged)")
            try:
                print("  - Cache info: " + str(cached_info))
            except:
                print("  - Cache info: cached")
            print("  - Cache debug: Skipping expensive 121-frame processing")
            
            # Return cloned tensors to prevent modification of cached data
            cached_images, cached_masks, cached_status = cached_result
            
            # Clone tensor outputs to prevent cache corruption
            if hasattr(cached_images, 'clone'):
                cloned_images = cached_images.clone().detach()
            else:
                cloned_images = cached_images
                
            if hasattr(cached_masks, 'clone'):
                cloned_masks = cached_masks.clone().detach()
            else:
                cloned_masks = cached_masks
                
            return (cloned_images, cloned_masks, cached_status)
        
        print("[WANVaceMaskEditor] ❌ Global cache miss - Processing mask data")
        try:
            data_len = len(mask_data) if mask_data else 0
            print("  - Input data length: " + str(data_len))
        except:
            print("  - Input data length: unknown")
        print("  - This will process expensive 121-frame operation")
        
        print("[WANVaceMaskEditor] Processing mask (inputs changed or no cache)")

        # Helper for cycling dots
        def get_dot_count():
            counter_file = os.path.join(tempfile.gettempdir(), "wan_mask_status_counter.txt")
            try:
                if os.path.exists(counter_file):
                    with open(counter_file, "r") as f:
                        count = int(f.read().strip())
                else:
                    count = 1
                count = (count % 3) + 1
                with open(counter_file, "w") as f:
                    f.write(str(count))
            except Exception:
                count = 1
            return count

        dot_count = get_dot_count()
        dots = "." * dot_count

        print("[WAN Mask Editor] create_mask called")
        try:
            mask_preview = mask_data[:100] if mask_data else 'empty'
            print("[WAN Mask Editor] mask_data: " + str(mask_preview) + "...")
        except:
            print("[WAN Mask Editor] mask_data: provided")
        
        status = "Initializing" + dots
        # Try to load mask data from editor
        if mask_data:
            try:
                data = json.loads(mask_data)
                try:
                    print("[WAN Mask Editor] Data keys: " + str(list(data.keys())))
                except:
                    print("[WAN Mask Editor] Data loaded")
                
                project_data = data.get("project_data", {})
                if not project_data:
                    print("[WAN Mask Editor] No project data found")
                    raise ValueError("No project data")
                
                # Extract video info and shape keyframes
                video_info = project_data.get("video_info", {})
                shape_keyframes = project_data.get("shape_keyframes", {})
                
                try:
                    print("[WAN Mask Editor] Video info: " + str(video_info))
                    print("[WAN Mask Editor] Shape keyframes: " + str(len(shape_keyframes)))
                except:
                    print("[WAN Mask Editor] Video info and shape keyframes loaded")
                
                # Get video properties
                video_path = video_info.get("path")
                video_type = video_info.get("type")
                total_frames = video_info.get("total_frames", 1)
                width = video_info.get("width", 512)
                height = video_info.get("height", 512)
                
                # Load video frames
                frames = []
                if video_path and os.path.exists(video_path):
                    if video_type == "video":
                        status = "Loading Video File" + dots
                        try:
                            print("[WAN Mask Editor] Loading video from: " + str(video_path))
                        except:
                            print("[WAN Mask Editor] Loading video file")
                        cap = cv2.VideoCapture(video_path)
                        if cap.isOpened():
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                # Convert BGR to RGB and normalize
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                                frames.append(frame_tensor)
                            cap.release()
                            try:
                                print("[WAN Mask Editor] Loaded " + str(len(frames)) + " frames")
                            except:
                                print("[WAN Mask Editor] Video frames loaded")
                    elif video_type == "image_sequence":
                        status = "Loading Image Sequence" + dots
                        try:
                            print("[WAN Mask Editor] Loading image sequence from: " + str(video_path))
                        except:
                            print("[WAN Mask Editor] Loading image sequence")
                        import glob
                        image_files = sorted(glob.glob(os.path.join(video_path, "*.*")))
                        for img_path in image_files:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img_tensor = torch.from_numpy(img_rgb).float() / 255.0
                                frames.append(img_tensor)
                        try:
                            print("[WAN Mask Editor] Loaded " + str(len(frames)) + " frames from image sequence")
                        except:
                            print("[WAN Mask Editor] Image sequence loaded")
                else:
                    status = "No video/image sequence found" + dots
                
                # Stack frames into tensor
                if frames:
                    images = torch.stack(frames)
                    total_frames = len(frames)
                else:
                    print("[WAN Mask Editor] No frames loaded, creating empty tensor")
                    images = torch.zeros((total_frames, height, width, 3), dtype=torch.float32)
                
                # Generate masks
                status = "Generating Masks" + dots
                masks = torch.zeros((total_frames, height, width), dtype=torch.float32)
                
                # Convert shape keyframes to integer keys
                shape_keyframes_int = {int(k): v for k, v in shape_keyframes.items()}
                try:
                    print("[WAN Mask Editor] Processing shapes for frames: " + str(list(shape_keyframes_int.keys())))
                except:
                    print("[WAN Mask Editor] Processing shapes for frames")
                
                # Helper function to interpolate shapes
                def interpolate_shapes(shapes1, shapes2, t):
                    interpolated = []
                    max_shapes = max(len(shapes1), len(shapes2))
                    for i in range(max_shapes):
                        if i >= len(shapes1):
                            shape2 = shapes2[i]
                            if t > 0.5:
                                interpolated.append(shape2)
                        elif i >= len(shapes2):
                            shape1 = shapes1[i]
                            if t < 0.5:
                                interpolated.append(shape1)
                        else:
                            shape1 = shapes1[i]
                            shape2 = shapes2[i]
                            vertices1 = shape1['vertices']
                            vertices2 = shape2['vertices']
                            if len(vertices1) != len(vertices2):
                                try:
                                    print("[WAN Mask Editor] Warning: Vertex count mismatch in interpolation: " + str(len(vertices1)) + " vs " + str(len(vertices2)))
                                except:
                                    print("[WAN Mask Editor] Warning: Vertex count mismatch in interpolation")
                                continue
                            interp_vertices = [
                                [v1[0] * (1 - t) + v2[0] * t, v1[1] * (1 - t) + v2[1] * t]
                                for v1, v2 in zip(vertices1, vertices2)
                            ]
                            interpolated.append({
                                'vertices': interp_vertices,
                                'closed': shape1.get('closed', True),
                                'visible': True,
                                'vertex_count': len(interp_vertices)
                            })
                    return interpolated
                
                keyframe_indices = sorted(shape_keyframes_int.keys())
                for frame_idx in range(total_frames):
                    mask = np.zeros((height, width), dtype=np.uint8)
                    if frame_idx in shape_keyframes_int:
                        shapes = shape_keyframes_int[frame_idx]
                    else:
                        prev_frame = None
                        next_frame = None
                        for kf in keyframe_indices:
                            if kf <= frame_idx:
                                prev_frame = kf
                            if kf >= frame_idx and next_frame is None:
                                next_frame = kf
                        if prev_frame is not None and next_frame is not None and prev_frame != next_frame:
                            t = (frame_idx - prev_frame) / (next_frame - prev_frame)
                            shapes = interpolate_shapes(
                                shape_keyframes_int[prev_frame],
                                shape_keyframes_int[next_frame],
                                t
                            )
                        elif prev_frame is not None:
                            shapes = shape_keyframes_int[prev_frame]
                        elif next_frame is not None:
                            shapes = shape_keyframes_int[next_frame]
                        else:
                            shapes = []
                    if shapes:
                        for shape in shapes:
                            if 'vertices' in shape and shape.get('visible', True):
                                vertices = np.array(shape['vertices'], dtype=np.int32)
                                if len(vertices) >= 3:
                                    cv2.fillPoly(mask, [vertices], 255)
                    mask_tensor = torch.from_numpy(mask).float() / 255.0
                    masks[frame_idx] = mask_tensor
                # Set final status with frame count and type - BULLETPROOF
                if frames:
                    source_str = "Image sequence" if video_type == "image_sequence" else ("Video file" if video_type == "video" else "Source")
                    frame_word = "frame" if total_frames == 1 else "frames"
                    status = source_str + " and masks loaded (" + str(total_frames) + " " + frame_word + ")"
                else:
                    status = "No frames loaded."
                
                # Cache the result in persistent cache for future use
                result = (images, masks, status)
                try:
                    if isinstance(images, torch.Tensor):
                        cache_info = "frames=" + str(total_frames) + ", shape=" + str(images.shape)
                    else:
                        cache_info = "frames=" + str(total_frames) + ", shape=unknown"
                except:
                    cache_info = "frames=" + str(total_frames)
                
                # Store in persistent cache that survives module reloads
                if cache_key:  # Only cache if we have a valid key
                    persistent_cache[cache_key] = (result, cache_info)
                
                # Debug: Print tensor stats and cache confirmation - BULLETPROOF
                if isinstance(images, torch.Tensor) and images.numel() > 0:
                    print("[WANVaceMaskEditor] ✅ Processing complete - Result cached globally")
                    try:
                        print("  - Instance: " + str(self._instance_id))
                        print("  - Cache key: " + str(cache_key)[:16] + "...")
                        print("  - Output shape: " + str(images.shape))
                        mean_val = images.mean().item()
                        print("  - Mean: " + str(round(mean_val, 6)))
                        print("  - Persistent cache size: " + str(len(persistent_cache)) + " entries")
                    except:
                        print("  - Processing completed successfully")
                
                return result
            except Exception as e:
                try:
                    print("[WAN Mask Editor] Error processing mask data: " + str(e))
                except:
                    print("[WAN Mask Editor] Error processing mask data")
                import traceback
                traceback.print_exc()
                status = "Error: " + str(e)
                # Cache error result too to avoid reprocessing same error
                result = (torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1)), status)
                cache_info = "error_result"
                if cache_key:  # Only cache if we have a valid key
                    persistent_cache[cache_key] = (result, cache_info)
                try:
                    print("[WANVaceMaskEditor] ❌ Error cached globally for instance " + str(self._instance_id))
                except:
                    print("[WANVaceMaskEditor] ❌ Error cached globally")
                return result
        else:
            print("[WAN Mask Editor] No mask_data provided")
            status = "No mask data provided."
        
        # Cache the no-data result too (empty cache_key case)
        result = (torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1)), status)
        cache_info = "no_data_result"
        # For no data, we don't cache since cache_key is empty
        try:
            print("[WANVaceMaskEditor] ⚪ No-data result (not cached due to empty input)")
        except:
            print("[WANVaceMaskEditor] ⚪ No-data result")
        return result

# Define outpainting editor node
class WANVaceOutpaintingEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "canvas_data": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Canvas data from outpainting editor"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "status")
    FUNCTION = "process_outpainting"
    CATEGORY = "WAN/mask"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, canvas_data=""):
        # This helps ComfyUI know when to reprocess
        import hashlib
        if canvas_data:
            return hashlib.md5(canvas_data.encode()).hexdigest()
        return ""
    
    def process_outpainting(self, canvas_data=""):
        import torch
        import json
        import cv2
        import numpy as np
        import os
        import tempfile
        from pathlib import Path
        import glob
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Optimized loading functions for ComfyUI processing
        def load_single_image_comfyui(img_path):
            """Load a single image file - for parallel processing in ComfyUI"""
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img_rgb
            except Exception as e:
                print(f"[WAN Outpainting Editor] Error loading image {img_path}: {e}")
            return None

        def load_video_optimized_comfyui(video_path):
            """Optimized video loading for ComfyUI processing"""
            frames = []
            start_time = time.time()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames
            
            # Get total frame count for progress
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[WAN Outpainting Editor] Video has {total_frames} frames")
            
            # Pre-allocate list for better performance
            frames = [None] * total_frames
            frame_idx = 0
            
            # Process in batches for better performance
            batch_size = 30
            batch_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                batch_frames.append((frame_idx, frame))
                frame_idx += 1
                
                # Process batch
                if len(batch_frames) >= batch_size or frame_idx >= total_frames:
                    for idx, frame in batch_frames:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames[idx] = frame_rgb
                    batch_frames = []
            
            cap.release()
            
            # Remove any None values (in case frame count was wrong)
            frames = [f for f in frames if f is not None]
            
            load_time = time.time() - start_time
            print(f"[WAN Outpainting Editor] Video loaded in {load_time:.2f}s ({len(frames)} frames, {len(frames)/load_time:.1f} fps)")
            
            return frames

        def load_image_sequence_optimized_comfyui(dir_path):
            """Optimized image sequence loading for ComfyUI processing"""
            frames = []
            start_time = time.time()
            
            # Get all valid image files
            image_files = sorted(glob.glob(os.path.join(dir_path, "*.*")))
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            
            valid_files = [
                img_path for img_path in image_files 
                if Path(img_path).suffix.lower() in valid_extensions
            ]
            
            if not valid_files:
                return frames
            
            print(f"[WAN Outpainting Editor] Found {len(valid_files)} image files")
            
            # Use parallel processing for faster loading
            max_workers = min(8, len(valid_files))  # Don't use too many threads
            frames = [None] * len(valid_files)
            completed = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all loading tasks
                future_to_index = {
                    executor.submit(load_single_image_comfyui, img_path): idx 
                    for idx, img_path in enumerate(valid_files)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        if result is not None:
                            frames[idx] = result
                        completed += 1
                        
                        # Progress logging
                        if completed % 100 == 0 or completed == len(valid_files):
                            print(f"[WAN Outpainting Editor] Loaded {completed}/{len(valid_files)} images...")
                            
                    except Exception as e:
                        print(f"[WAN Outpainting Editor] Error loading image {idx}: {e}")
                        completed += 1
            
            # Remove any None values
            frames = [f for f in frames if f is not None]
            
            load_time = time.time() - start_time
            print(f"[WAN Outpainting Editor] Image sequence loaded in {load_time:.2f}s ({len(frames)} images, {len(frames)/load_time:.1f} imgs/s)")
            
            return frames
        
        # Helper for cycling dots
        def get_dot_count():
            counter_file = os.path.join(tempfile.gettempdir(), "wan_outpainting_status_counter.txt")
            try:
                if os.path.exists(counter_file):
                    with open(counter_file, "r") as f:
                        count = int(f.read().strip())
                else:
                    count = 1
                count = (count % 3) + 1
                with open(counter_file, "w") as f:
                    f.write(str(count))
            except Exception:
                count = 1
            return count
        
        dot_count = get_dot_count()
        dots = "." * dot_count
        
        print(f"[WAN Outpainting Editor] process_outpainting called")
        print(f"[WAN Outpainting Editor] canvas_data: {canvas_data[:100] if canvas_data else 'empty'}...")
        
        status = f"Initializing{dots}"
        # Try to load canvas data from editor
        if canvas_data:
            try:
                data = json.loads(canvas_data)
                print(f"[WAN Outpainting Editor] Data keys: {list(data.keys())}")
                
                project_data = data.get("project_data", {})
                if not project_data:
                    print(f"[WAN Outpainting Editor] No project data found")
                    raise ValueError("No project data")
                
                # Extract video info and canvas settings
                video_info = project_data.get("video_info", {})
                canvas_settings = project_data.get("canvas_settings", {})
                
                print(f"[WAN Outpainting Editor] Video info: {video_info}")
                print(f"[WAN Outpainting Editor] Canvas settings: {canvas_settings}")
                
                # Get video properties
                video_path = video_info.get("path")
                video_type = video_info.get("type")
                total_frames = video_info.get("total_frames", 1)
                width = video_info.get("width", 512)
                height = video_info.get("height", 512)
                
                # Get canvas settings
                canvas_width = int(canvas_settings.get("canvas_width", width))
                canvas_height = int(canvas_settings.get("canvas_height", height))
                video_x = int(canvas_settings.get("video_x", 0))
                video_y = int(canvas_settings.get("video_y", 0))
                video_width = int(canvas_settings.get("video_width", width))
                video_height = int(canvas_settings.get("video_height", height))
                feather_amount = int(canvas_settings.get("feather_amount", 0))
                
                # Load video frames using optimized loading
                frames = []
                if video_path and os.path.exists(video_path):
                    if video_type == "video":
                        status = f"Loading Video File{dots}"
                        print(f"[WAN Outpainting Editor] Loading video from: {video_path}")
                        frames = load_video_optimized_comfyui(video_path)
                        if frames:
                            print(f"[WAN Outpainting Editor] Loaded {len(frames)} frames")
                        else:
                            print(f"[WAN Outpainting Editor] Failed to load video")
                    elif video_type == "image_sequence":
                        status = f"Loading Image Sequence{dots}"
                        print(f"[WAN Outpainting Editor] Loading image sequence from: {video_path}")
                        frames = load_image_sequence_optimized_comfyui(video_path)
                        if frames:
                            print(f"[WAN Outpainting Editor] Loaded {len(frames)} frames from image sequence")
                        else:
                            print(f"[WAN Outpainting Editor] Failed to load image sequence")
                else:
                    status = f"No video/image sequence found{dots}"
                
                # Process outpainting
                if frames:
                    status = f"Applying Outpainting{dots}"
                    processed_frames = []
                    masks = []
                    
                    # Get padding color (frame darkness)
                    frame_darkness = 0.0  # Default to black padding
                    
                    # Get actual frame dimensions
                    frame_height, frame_width = frames[0].shape[0], frames[0].shape[1]
                    print(f"[WAN Outpainting Editor] Frame dimensions: {frame_width}x{frame_height}")
                    print(f"[WAN Outpainting Editor] Canvas dimensions: {canvas_width}x{canvas_height}")
                    print(f"[WAN Outpainting Editor] Video position: ({video_x}, {video_y})")
                    print(f"[WAN Outpainting Editor] Video display size: {video_width}x{video_height}")
                    
                    # Pre-calculate regions once (they're the same for all frames)
                    src_x = 0 if video_x >= 0 else -video_x
                    src_y = 0 if video_y >= 0 else -video_y
                    dst_x = max(0, video_x)
                    dst_y = max(0, video_y)
                    
                    # Check if we need to resize frames
                    need_resize = (video_width != frame_width) or (video_height != frame_height)
                    
                    # Calculate copy dimensions once
                    copy_width = min(frame_width - src_x, canvas_width - dst_x)
                    copy_height = min(frame_height - src_y, canvas_height - dst_y)
                    
                    print(f"[WAN Outpainting Editor] Copy region: src=({src_x},{src_y}), dst=({dst_x},{dst_y}), size={copy_width}x{copy_height}")
                    
                    # Create base mask once (same for all frames)
                    base_mask = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255  # Start with all white
                    
                    # Use video dimensions for mask if resizing
                    mask_width = video_width if need_resize else copy_width
                    mask_height = video_height if need_resize else copy_height
                    
                    if mask_width > 0 and mask_height > 0 and dst_x >= 0 and dst_y >= 0:
                        # Calculate the actual area to mark as black (video area)
                        mask_x_end = min(dst_x + mask_width, canvas_width)
                        mask_y_end = min(dst_y + mask_height, canvas_height)
                        base_mask[dst_y:mask_y_end, dst_x:mask_x_end] = 0  # Set video area to black
                    
                    # Apply feathering once if specified
                    if feather_amount > 0:
                        # Invert mask for distance transform (need white where video is)
                        inverted_mask = 255 - base_mask
                        # Create distance transform for feathering
                        dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
                        # Normalize and apply feather (inverted - closer to video = darker)
                        feather_mask = 255 - np.clip(dist_transform / feather_amount, 0, 1) * 255
                        final_mask = feather_mask.astype(np.uint8)
                    else:
                        final_mask = base_mask
                    
                    # Convert mask to tensor once
                    mask_tensor = torch.from_numpy(final_mask).float() / 255.0
                    
                    if need_resize:
                        print(f"[WAN Outpainting Editor] Video was resized from {frame_width}x{frame_height} to {video_width}x{video_height}")
                        # Recalculate copy dimensions for resized video
                        copy_width = min(video_width - src_x, canvas_width - dst_x)
                        copy_height = min(video_height - src_y, canvas_height - dst_y)
                        print(f"[WAN Outpainting Editor] Updated copy size for resized video: {copy_width}x{copy_height}")
                    
                    # Check for GPU availability
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    print(f"[WAN Outpainting Editor] Using device: {device}")
                    
                    # Process frames with parallelism for better performance
                    print(f"[WAN Outpainting Editor] Processing {len(frames)} frames...")
                    
                    # Use optimized vectorized processing for better performance
                    if len(frames) > 50:
                        # Use vectorized batch processing for large frame counts
                        print(f"[WAN Outpainting Editor] Using vectorized batch processing for {len(frames)} frames...")
                        
                        # Process in chunks to reduce memory usage
                        chunk_size = 64  # Increased chunk size for better GPU utilization
                        processed_frames = []
                        
                        for chunk_start in range(0, len(frames), chunk_size):
                            chunk_end = min(chunk_start + chunk_size, len(frames))
                            chunk_frames = frames[chunk_start:chunk_end]
                            
                            # Convert chunk to numpy array for vectorized operations
                            chunk_array = np.stack(chunk_frames)  # Shape: (N, H, W, 3)
                            
                            # Vectorized resize if needed
                            if need_resize:
                                resized_chunk = []
                                for frame in chunk_array:
                                    resized_frame = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_LANCZOS4)
                                    resized_chunk.append(resized_frame)
                                chunk_array = np.stack(resized_chunk)
                            
                            # Create expanded canvas for all frames at once
                            num_frames_in_chunk = chunk_array.shape[0]
                            expanded_chunk = np.full((num_frames_in_chunk, canvas_height, canvas_width, 3), 
                                                   int(frame_darkness * 255), dtype=np.uint8)
                            
                            # Vectorized copy operation for all frames
                            if copy_width > 0 and copy_height > 0:
                                expanded_chunk[:, dst_y:dst_y+copy_height, dst_x:dst_x+copy_width] = \
                                    chunk_array[:, src_y:src_y+copy_height, src_x:src_x+copy_width]
                            
                            # Convert entire chunk to tensor in one operation
                            chunk_tensor = torch.from_numpy(expanded_chunk).float() / 255.0
                            
                            # Move to GPU if available
                            if device == 'cuda':
                                chunk_tensor = chunk_tensor.cuda()
                            
                            # Add individual frames to list (keep as tensors)
                            for frame_tensor in chunk_tensor:
                                processed_frames.append(frame_tensor)
                            
                            # Progress update
                            if chunk_end % 200 == 0 or chunk_end == len(frames):
                                print(f"[WAN Outpainting Editor] Processed {chunk_end}/{len(frames)} frames...")
                        
                        # Create masks efficiently (same for all frames) - vectorized
                        if device == 'cuda':
                            mask_tensor = mask_tensor.cuda()
                        
                        # Create all masks at once
                        mask_batch = mask_tensor.unsqueeze(0).repeat(len(processed_frames), 1, 1)
                        masks = [mask_batch[i] for i in range(len(processed_frames))]
                        
                    else:
                        # Optimized processing for small frame counts
                        print(f"[WAN Outpainting Editor] Using optimized processing for {len(frames)} frames...")
                        
                        # Process all frames at once for small counts
                        frames_array = np.stack(frames)
                        
                        # Vectorized resize if needed
                        if need_resize:
                            resized_frames = []
                            for frame in frames_array:
                                resized_frame = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_LANCZOS4)
                                resized_frames.append(resized_frame)
                            frames_array = np.stack(resized_frames)
                        
                        # Create expanded canvas for all frames
                        expanded_frames = np.full((len(frames), canvas_height, canvas_width, 3), 
                                               int(frame_darkness * 255), dtype=np.uint8)
                        
                        # Vectorized copy operation
                        if copy_width > 0 and copy_height > 0:
                            expanded_frames[:, dst_y:dst_y+copy_height, dst_x:dst_x+copy_width] = \
                                frames_array[:, src_y:src_y+copy_height, src_x:src_x+copy_width]
                        
                        # Convert all frames to tensors at once
                        frames_tensor = torch.from_numpy(expanded_frames).float() / 255.0
                        if device == 'cuda':
                            frames_tensor = frames_tensor.cuda()
                        
                        # Split into individual frame tensors
                        for frame_tensor in frames_tensor:
                            processed_frames.append(frame_tensor)
                        
                        # Create all masks efficiently
                        if device == 'cuda':
                            mask_tensor = mask_tensor.cuda()
                        mask_batch = mask_tensor.unsqueeze(0).repeat(len(frames), 1, 1)
                        masks = [mask_batch[i] for i in range(len(frames))]
                    
                    # Stack into tensors
                    images = torch.stack(processed_frames)
                    mask_stack = torch.stack(masks)
                    
                    # Move tensors back to CPU for ComfyUI compatibility
                    if device == 'cuda':
                        images = images.cpu()
                        mask_stack = mask_stack.cpu()
                    
                    # Set final status
                    source_str = "Image sequence" if video_type == "image_sequence" else "Video file"
                    frame_word = "frame" if len(frames) == 1 else "frames"
                    perf_info = f" | {device.upper()}"
                    status = f"{source_str} outpainted ({len(frames)} {frame_word}, canvas: {canvas_width}x{canvas_height}){perf_info}"
                    
                    return (images, mask_stack, status)
                else:
                    print(f"[WAN Outpainting Editor] No frames loaded, creating empty tensor")
                    images = torch.zeros((1, canvas_height, canvas_width, 3), dtype=torch.float32)
                    masks = torch.zeros((1, canvas_height, canvas_width), dtype=torch.float32)
                    status = "No frames loaded."
                    return (images, masks, status)
                    
            except Exception as e:
                print(f"[WAN Outpainting Editor] Error processing canvas data: {e}")
                import traceback
                traceback.print_exc()
                status = f"Error: {e}"
                return (torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1)), status)
        else:
            print(f"[WAN Outpainting Editor] No canvas_data provided")
            status = "No canvas data provided."
        return (torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1)), status)

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add the mask editor and outpainting editor first
NODE_CLASS_MAPPINGS["WANVaceMaskEditor"] = WANVaceMaskEditor
NODE_DISPLAY_NAME_MAPPINGS["WANVaceMaskEditor"] = "WanVace-pipeline Mask Editor 🎭"

NODE_CLASS_MAPPINGS["WANVaceOutpaintingEditor"] = WANVaceOutpaintingEditor
NODE_DISPLAY_NAME_MAPPINGS["WANVaceOutpaintingEditor"] = "WanVace-pipeline Outpainting Editor 🎨"

# Try to load crop and stitch nodes first (these should work independently)
print("[WAN Vace Pipeline] Loading crop and stitch nodes...")
try:
    from .wan_cropandstitch import WanCropImproved
    from .wan_cropandstitch import WanStitchImproved
    
    NODE_CLASS_MAPPINGS["WanCropImproved"] = WanCropImproved
    NODE_CLASS_MAPPINGS["WanStitchImproved"] = WanStitchImproved
    
    NODE_DISPLAY_NAME_MAPPINGS["WanCropImproved"] = "WanVace-pipeline Crop ✂️"
    NODE_DISPLAY_NAME_MAPPINGS["WanStitchImproved"] = "WanVace-pipeline Stitch ✂️"
    
    print("[WAN Vace Pipeline] ✅ Successfully loaded crop and stitch nodes")
except Exception as crop_e:
    print(f"[WAN Vace Pipeline] ❌ ERROR loading crop and stitch nodes: {crop_e}")
    import traceback
    traceback.print_exc()

# Try to load main nodes
try:
    from .node_mappings import NODE_CLASS_MAPPINGS as MAIN_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MAIN_DISPLAY_MAPPINGS
    print(f"[WAN Vace Pipeline] Successfully loaded {len(MAIN_NODE_MAPPINGS)} nodes from node_mappings")
    
    # Merge the main nodes with our existing mappings
    NODE_CLASS_MAPPINGS.update(MAIN_NODE_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(MAIN_DISPLAY_MAPPINGS)
    
    # List all loaded nodes
    print("[WAN Vace Pipeline] All loaded nodes:")
    for node_name in NODE_CLASS_MAPPINGS:
        print(f"  - {node_name}")
        
except Exception as e:
    print(f"[WAN Vace Pipeline] ERROR loading main nodes: {e}")
    import traceback
    traceback.print_exc()
    print("[WAN Vace Pipeline] Continuing with mask editor and crop/stitch nodes only")

import os
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]