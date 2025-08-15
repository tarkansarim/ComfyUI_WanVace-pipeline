"""
Video Input/Output nodes for WAN Vace Pipeline
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import folder_paths
import json
from ..utils import get_video_info


class WANSaveVideo:
    """Save frames as video file or image sequence"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "output_mode": (["video", "image_sequence"], {
                    "default": "video",
                    "tooltip": "Choose output format"
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Full path to output"
                }),
                # Common video settings
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.1,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Frames per second (video mode only)"
                }),
                "video_format": (["mp4", "avi", "mov", "mkv", "webm"], {
                    "default": "mp4",
                    "tooltip": "Video container format (video mode only)"
                }),
                "video_codec": (["h264", "hevc", "mp4v", "xvid", "mjpeg"], {
                    "default": "h264",
                    "tooltip": "Video codec (video mode only)"
                }),
                "video_quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Video compression quality (video mode only)"
                }),
                # Image sequence settings
                "image_format": (["png", "jpg", "jpeg", "bmp", "tiff"], {
                    "default": "png",
                    "tooltip": "Image format (image sequence mode only)"
                }),
                "filename_prefix": ("STRING", {
                "default": "frame.",
                    "tooltip": "Prefix for image filenames (image sequence mode only)"
                }),
                "start_number": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Starting frame number (image sequence mode only)"
                }),
                "number_padding": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of digits, e.g., 5 = frame_00001.png (image sequence mode only)"
                }),
                # Versioning feature
                "enable_versioning": ("BOOLEAN", {
                "default": True,
                    "tooltip": "Enable automatic versioning - creates version folders (v1, v2, etc.) for each run"
                }),
                "version_prefix": ("STRING", {
                    "default": "v",
                    "multiline": False,
                    "tooltip": "Prefix for version folders (e.g., 'v' creates v1, v2; 'test' creates test1, test2)"
                }),
            },
            "optional": {
                "batch_info_json": ("STRING", {
                    "default": "",
                    "tooltip": "Connect from Split node's batch_info_json output for multi-batch saves"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("output_path", "frames_saved", "save_type")
    FUNCTION = "save_video"
    CATEGORY = "WAN Vace/IO"
    OUTPUT_NODE = True
    
    def get_versioned_path(self, base_path, version_prefix, is_video_mode):
        """Generate versioned path using custom prefix and folder structure"""
        base_path = Path(base_path)
        
        # For both video and image modes, we create version folders
        if is_video_mode:
            # Extract the filename to put inside the version folder
            filename = base_path.name
            output_dir = base_path.parent
        else:
            # For image sequences, the base_path should be the output directory
            output_dir = base_path.parent if base_path.suffix else base_path
            filename = None  # Will be handled in save method
        
        # Scan for existing version folders
        existing_versions = []
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir() and item.name.startswith(version_prefix):
                    # Extract version number
                    version_part = item.name[len(version_prefix):]
                    if version_part.isdigit():
                        existing_versions.append(int(version_part))
        
        # Find next version number
        if existing_versions:
            next_version = max(existing_versions) + 1
        else:
            next_version = 1
        
        # Create version folder path
        version_folder = output_dir / f"{version_prefix}{next_version}"
        
        if is_video_mode:
            # Return path to video file inside version folder
            return version_folder / filename
        else:
            # Return path to version folder for image sequence
            return version_folder
    
    def save_video(self, frames, output_mode, output_path,
                   fps=30.0, video_format="mp4", video_codec="h264", video_quality=95,
                   image_format="png", filename_prefix="frame_", 
                   start_number=0, number_padding=5, enable_versioning=False, 
                   version_prefix="v", batch_info_json=""):
        
        print(f"\n=== SAVE VIDEO CALLED ===")
        print(f"Output mode: {output_mode}")
        print(f"Output path: {output_path}")
        print(f"Enable versioning: {enable_versioning}")
        print(f"Version prefix: {version_prefix}")
        print(f"Batch info json length: {len(batch_info_json) if batch_info_json else 0}")
        print(f"Batch info preview: {batch_info_json[:100] if batch_info_json else 'None'}...")
        
        if not output_path or not output_path.strip():
            raise ValueError("Please provide a valid output path")
        
        output_path = Path(output_path.strip()).expanduser()
        
        # If a relative path is provided, resolve it against ComfyUI's output directory
        if not output_path.is_absolute():
            try:
                base_output_dir = Path(folder_paths.get_output_directory())
            except Exception:
                # Fallback to current working directory if ComfyUI's output dir is unavailable
                base_output_dir = Path.cwd()
            output_path = (base_output_dir / output_path).resolve()
        
        # Apply versioning logic if enabled
        if enable_versioning:
            original_path = output_path
            output_path = self.get_versioned_path(output_path, version_prefix, output_mode == "video")
            print(f"📁 Versioning enabled: {original_path} -> {output_path}")
        
        # Convert tensor to numpy array
        if isinstance(frames, torch.Tensor):
            # Ensure we have the right shape (B, H, W, C)
            if frames.dim() == 3:
                frames = frames.unsqueeze(0)
            
            # Convert from normalized float to uint8
            frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)
        else:
            frames_np = frames
        
        num_frames = frames_np.shape[0]
        height, width = frames_np.shape[1:3]
        
        # Parse batch info if provided
        batch_info = None
        if batch_info_json and batch_info_json.strip():
            try:
                batch_info = json.loads(batch_info_json)
                print(f"DEBUG: Batch info parsed: multi_batch={batch_info.get('multi_batch', False)}, total_batches={batch_info.get('total_batches', 0)}")
            except json.JSONDecodeError:
                print(f"WARNING: Failed to parse batch_info_json")
                batch_info = None
        
        # Check if this is a batched save
        if batch_info is not None and isinstance(batch_info, dict):
            # Check if this is a multi-batch save (all batches at once)
            if batch_info.get('multi_batch', False):
                # This contains multiple batches that need to be saved separately
                batch_boundaries = batch_info.get('batch_boundaries', [])
                fps_to_use = fps  # Use the fps from save node parameter
                
                output_paths = []
                total_frames_saved = 0
                
                print(f"\n{'='*40}")
                print(f"📹 SAVING MULTIPLE BATCHES")
                print(f"{'='*40}")
                print(f"Total batches to save: {len(batch_boundaries)}")
                print(f"Output path base: {output_path}")
                print(f"Batch boundaries: {batch_boundaries}")
                
                for boundary in batch_boundaries:
                    batch_idx = boundary['batch_index']
                    start = boundary['start']
                    end = boundary['end']
                    
                    # Extract frames for this batch
                    batch_frames = frames_np[start:end]
                    
                    # Modify output path to include batch number
                    if enable_versioning:
                        # For versioned saves, put batches inside the version folder
                        version_folder = output_path.parent
                        base_name = output_path.stem if output_mode == "video" else "batch"
                        if output_mode == "video":
                            batch_output_path = version_folder / f"{base_name}_batch_{batch_idx+1:03d}{output_path.suffix}"
                        else:
                            batch_output_path = version_folder / f"batch_{batch_idx+1:03d}"
                    else:
                        # Original batch naming logic
                        base_path = output_path.stem
                        if output_mode == "video":
                            batch_output_path = output_path.parent / f"{base_path}_batch_{batch_idx+1:03d}{output_path.suffix}"
                            if not batch_output_path.suffix or batch_output_path.suffix.lower()[1:] != video_format:
                                batch_output_path = batch_output_path.with_suffix(f'.{video_format}')
                        else:
                            batch_output_path = output_path.parent / f"{base_path}_batch_{batch_idx+1:03d}"
                    
                    if output_mode == "video":
                        if not batch_output_path.suffix or batch_output_path.suffix.lower()[1:] != video_format:
                            batch_output_path = batch_output_path.with_suffix(f'.{video_format}')
                        
                        batch_path, frames_written, _ = self.save_as_video(
                            batch_frames, fps_to_use, batch_output_path, 
                            video_format, video_codec, video_quality
                        )
                    else:
                        batch_path, frames_written, _ = self.save_as_image_sequence(
                            batch_frames, batch_output_path, image_format,
                            filename_prefix, start_number, number_padding
                        )
                    
                    output_paths.append(batch_path)
                    total_frames_saved += frames_written
                
                print(f"Total frames saved across all batches: {total_frames_saved}")
                print(f"{'='*40}\n")
                
                # Return the first output path as the main output
                return (output_paths[0] if output_paths else "", total_frames_saved, output_mode)
            
            else:
                # Single batch save
                batch_index = batch_info.get('batch_index', 0)
                total_batches = batch_info.get('total_batches', 1)
                batch_enabled = batch_info.get('batching_enabled', False)
                
                if batch_enabled and total_batches > 1:
                    # Modify output path to include batch number
                    if enable_versioning:
                        # For versioned saves, put batches inside the version folder
                        version_folder = output_path.parent
                        base_name = output_path.stem if output_mode == "video" else "batch"
                        if output_mode == "video":
                            output_path = version_folder / f"{base_name}_batch_{batch_index+1:03d}{output_path.suffix}"
                        else:
                            output_path = version_folder / f"batch_{batch_index+1:03d}"
                    else:
                        # Original batch naming logic
                        base_path = output_path.stem
                        if output_mode == "video":
                            output_path = output_path.parent / f"{base_path}_batch_{batch_index+1:03d}{output_path.suffix}"
                        else:
                            output_path = output_path.parent / f"{base_path}_batch_{batch_index+1:03d}"
        
        # Continue with normal save for non-batch or single batch
        if output_mode == "video":
            # For video mode, ensure path has extension matching format
            if not output_path.suffix or output_path.suffix.lower()[1:] != video_format:
                output_path = output_path.with_suffix(f'.{video_format}')
            return self.save_as_video(frames_np, fps, output_path, video_format, video_codec, video_quality)
        else:
            # For image sequence mode, path should be a directory
            return self.save_as_image_sequence(frames_np, output_path, image_format, 
                                             filename_prefix, start_number, number_padding)
    
    def save_as_video(self, frames_np, fps, output_path, video_format, codec_name, quality):
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Map codec names to fourcc codes
        # Use more compatible codec settings
        if video_format == "mp4":
            if codec_name == "h264":
                # Try different H264 variations
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                except:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*'H264')
                    except:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            codec_map = {
                "h264": cv2.VideoWriter_fourcc(*'H264'),
                "hevc": cv2.VideoWriter_fourcc(*'HEVC'),
                "mp4v": cv2.VideoWriter_fourcc(*'mp4v'),
                "xvid": cv2.VideoWriter_fourcc(*'XVID'),
                "mjpeg": cv2.VideoWriter_fourcc(*'MJPG'),
            }
            fourcc = codec_map.get(codec_name, cv2.VideoWriter_fourcc(*'mp4v'))
        
        # Get frame dimensions
        height, width = frames_np.shape[1:3]
        
        # Create video writer
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Failed to create video writer for: {output_path}")
        
        # Write frames
        frames_written = 0
        for i, frame in enumerate(frames_np):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            frames_written += 1
        
        out.release()
        
        # Print save info
        print(f"\n{'='*40}")
        print(f"📹 VIDEO SAVED")
        print(f"{'='*40}")
        print(f"Path: {output_path}")
        print(f"Format: {video_format.upper()}")
        print(f"Codec: {codec_name}")
        print(f"FPS: {fps:.2f}")
        print(f"Frames: {frames_written}")
        print(f"Resolution: {width}x{height}")
        print(f"Duration: {frames_written/fps:.2f} seconds")
        print(f"{'='*40}\n")
        
        return (str(output_path), frames_written, "video")
    
    def save_as_image_sequence(self, frames_np, output_dir, image_format, 
                             filename_prefix, start_number, number_padding):
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use maximum quality for all formats
        if image_format in ['jpg', 'jpeg']:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 100]  # Max quality
        elif image_format == 'png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # No compression = max quality
        else:
            encode_params = []
        
        frames_written = 0
        
        # Save each frame
        for i, frame in enumerate(frames_np):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Generate filename
            frame_number = start_number + i
            filename = f"{filename_prefix}{frame_number:0{number_padding}d}.{image_format}"
            filepath = output_dir / filename
            
            # Save image
            success = cv2.imwrite(str(filepath), frame_bgr, encode_params)
            if not success:
                raise ValueError(f"Failed to save image: {filepath}")
            
            frames_written += 1
        
        # Print save info
        print(f"\n{'='*40}")
        print(f"🖼️ IMAGE SEQUENCE SAVED")
        print(f"{'='*40}")
        print(f"Path: {output_dir}")
        print(f"Format: {image_format}")
        print(f"Quality: Maximum")
        print(f"Frames: {frames_written}")
        print(f"Pattern: {filename_prefix}{'0' * number_padding}.{image_format}")
        print(f"{'='*40}\n")
        
        return (str(output_dir), frames_written, "image_sequence")




class WANLoadVideo:
    """Load video files or image sequences into ComfyUI"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "Enter path to video file or image sequence folder",
                    "multiline": False,
                    "tooltip": "Enter full path to video file (e.g., C:/videos/video.mp4) or image sequence folder"
                }),
                "frame_load_cap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Maximum number of frames to load (0 = load all frames)"
                }),
                "skip_first_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of frames to skip from beginning"
                }),
                "select_every_nth": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Select every Nth frame (1 = every frame, 2 = every other frame, etc.)"
                }),
            },
            "optional": {
                "target_fps": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Target FPS (0 = use original FPS). Automatically drops/samples frames to maintain playback speed"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT", "STRING", "DICT")
    RETURN_NAMES = ("frames", "frame_count", "fps", "width", "height", "source_type", "video_info")
    FUNCTION = "load_video"
    CATEGORY = "WAN Vace/IO"
    OUTPUT_NODE = False
    
    def load_video(self, path, frame_load_cap=0, skip_first_frames=0, select_every_nth=1, target_fps=0.0):
        if not path or not path.strip():
            raise ValueError("Please provide a path to a video file or image sequence folder")
        
        target_path = Path(path.strip())
        
        if not target_path.exists():
            raise ValueError(f"Path not found: {target_path}")
        
        # Auto-detect if it's a video file or image sequence folder
        if target_path.is_file():
            # It's a video file
            return self.load_video_file(target_path, frame_load_cap, skip_first_frames, select_every_nth, target_fps)
        elif target_path.is_dir():
            # It's an image sequence folder
            return self.load_image_sequence(target_path, frame_load_cap, skip_first_frames, select_every_nth, target_fps)
        else:
            raise ValueError(f"Path is neither a file nor a directory: {target_path}")
    
    def load_video_file(self, video_file, frame_load_cap, skip_first_frames, select_every_nth, target_fps):
        
        # Open video with hardware acceleration if available
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_file}")
        
        # Set buffer size for faster reading
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frames to load
        start_frame = skip_first_frames
        if start_frame >= total_frames:
            cap.release()
            raise ValueError(f"Skip frames ({start_frame}) exceeds total frames ({total_frames})")
        
        # Calculate frame sampling based on target FPS
        if target_fps > 0 and target_fps != fps:
            # Calculate frame step to achieve target FPS while maintaining playback speed
            fps_ratio = fps / target_fps
            effective_select_every_nth = int(round(fps_ratio * select_every_nth))
            if effective_select_every_nth < 1:
                effective_select_every_nth = 1
            output_fps = target_fps
        else:
            effective_select_every_nth = select_every_nth
            output_fps = fps / select_every_nth  # Adjust FPS based on frame selection
        
        # Calculate total frames we'll actually load
        frames_after_skip = total_frames - start_frame
        max_frames = frames_after_skip // effective_select_every_nth
        if frame_load_cap > 0:
            max_frames = min(max_frames, frame_load_cap)
        
        # Pre-allocate numpy array for maximum speed
        frame_shape = (height, width, 3)
        frames = np.empty((max_frames, height, width, 3), dtype=np.uint8)
        
        # Set start position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Load frames directly into pre-allocated array
        frame_idx = 0
        read_count = 0
        
        while frame_idx < max_frames:
            # Skip frames if needed
            if effective_select_every_nth > 1 and read_count > 0:
                for _ in range(effective_select_every_nth - 1):
                    ret = cap.grab()  # grab() is faster than read() for skipping
                    if not ret:
                        break
                    read_count += 1
            
            # Read the frame we want
            ret, frame = cap.read()
            if not ret:
                break
            
            # Copy directly into pre-allocated array
            frames[frame_idx] = frame
            frame_idx += 1
            read_count += 1
        
        cap.release()
        
        # Trim array if we read fewer frames than expected
        if frame_idx < max_frames:
            frames = frames[:frame_idx]
        
        if frame_idx == 0:
            raise ValueError("No frames were loaded from the video")
        
        # Convert frames to tensor using optimized batch conversion
        # Convert BGR to RGB and normalize in one operation
        frames_rgb = frames[..., ::-1].astype(np.float32) / 255.0
        tensor_frames = torch.from_numpy(frames_rgb)
        
        # Calculate actual frame count
        actual_frame_count = frame_idx
        
        # Create video info dictionary
        video_info = {
            "source_fps": fps,
            "source_frame_count": total_frames,
            "source_duration": total_frames / fps if fps > 0 else 0,
            "source_width": width,
            "source_height": height,
            "loaded_fps": output_fps,
            "loaded_frame_count": actual_frame_count,
            "loaded_duration": actual_frame_count / output_fps if output_fps > 0 else 0,
            "loaded_width": width,
            "loaded_height": height,
        }
        
        # Print detection info
        print(f"\n{'='*40}")
        print(f"🎬 VIDEO FILE DETECTED")
        print(f"{'='*40}")
        print(f"Path: {video_file}")
        print(f"Original FPS: {fps:.2f}")
        if target_fps > 0 and target_fps != fps:
            print(f"Target FPS: {output_fps:.2f} (sampling every {effective_select_every_nth} frames)")
        print(f"Frames loaded: {actual_frame_count}")
        print(f"Output FPS: {output_fps:.2f}")
        print(f"Resolution: {width}x{height}")
        print(f"Duration: {actual_frame_count/output_fps:.2f} seconds")
        print(f"{'='*40}\n")
        
        return (tensor_frames, actual_frame_count, output_fps, width, height, "video", video_info)
    
    def load_image_sequence(self, folder_path, frame_load_cap, skip_first_frames, select_every_nth, target_fps):
        # Get all image files in the folder
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        # Use a single case-insensitive search to avoid duplicates
        image_files = []
        for file in folder_path.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                image_files.append(file)
        
        # Sort files naturally (handle numeric ordering)
        def natural_sort_key(path):
            import re
            parts = re.split(r'(\d+)', path.stem)
            return [int(part) if part.isdigit() else part.lower() for part in parts]
        
        image_files = sorted(image_files, key=natural_sort_key)
        
        if not image_files:
            raise ValueError(f"No image files found in folder: {folder_path}")
        
        # Apply frame selection parameters
        total_frames = len(image_files)
        
        # Skip frames
        if skip_first_frames >= total_frames:
            raise ValueError(f"Skip frames ({skip_first_frames}) exceeds total frames ({total_frames})")
        
        image_files = image_files[skip_first_frames:]
        
        # Select every nth frame
        if select_every_nth > 1:
            image_files = image_files[::select_every_nth]
        
        # Apply frame cap
        if frame_load_cap > 0 and len(image_files) > frame_load_cap:
            image_files = image_files[:frame_load_cap]
        
        # Load first image to get dimensions
        first_image = cv2.imread(str(image_files[0]))
        if first_image is None:
            raise ValueError(f"Failed to load first image: {image_files[0]}")
        
        height, width = first_image.shape[:2]
        
        # Pre-allocate array for all images
        frames = np.empty((len(image_files), height, width, 3), dtype=np.uint8)
        frames[0] = first_image
        
        # Load remaining images
        for i, img_path in enumerate(image_files[1:], 1):
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Check dimensions match
            if img.shape[:2] != (height, width):
                raise ValueError(f"Image dimensions mismatch: {img_path} has shape {img.shape[:2]}, expected ({height}, {width})")
            
            frames[i] = img
        
        # Convert to tensor
        frames_rgb = frames[..., ::-1].astype(np.float32) / 255.0
        tensor_frames = torch.from_numpy(frames_rgb)
        
        # Determine FPS
        fps = target_fps if target_fps > 0 else 30.0
        actual_frame_count = len(image_files)
        
        # Create video info dictionary
        video_info = {
            "source_fps": 30.0,  # Default assumption for image sequences
            "source_frame_count": total_frames,
            "source_duration": total_frames / 30.0,
            "source_width": width,
            "source_height": height,
            "loaded_fps": fps,
            "loaded_frame_count": actual_frame_count,
            "loaded_duration": actual_frame_count / fps,
            "loaded_width": width,
            "loaded_height": height,
        }
        
        # Print detection info
        print(f"\n{'='*40}")
        print(f"🖼️ IMAGE SEQUENCE DETECTED")
        print(f"{'='*40}")
        print(f"Path: {folder_path}")
        print(f"Frames loaded: {actual_frame_count}")
        print(f"FPS: {fps:.2f} (default: 30.0)" if target_fps == 0 else f"FPS: {fps:.2f}")
        print(f"Resolution: {width}x{height}")
        print(f"Duration: {actual_frame_count/fps:.2f} seconds")
        print(f"{'='*40}\n")
        
        return (tensor_frames, actual_frame_count, fps, width, height, "image_sequence", video_info)




