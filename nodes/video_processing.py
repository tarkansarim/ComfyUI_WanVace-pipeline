"""
Video processing nodes for WAN Vace Pipeline
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import folder_paths
import json
from ..utils import get_video_info


class WANVaceSplitReferenceVideo:
    """Split video into batches for processing
    
    Two modes:
    1. Single batch mode (batch_index >= 0): Output one batch at a time
    2. All batches mode (batch_index = -1): Output all batches concatenated
    
    When using batch_index = -1:
    - All batches are concatenated and sent to save node
    - Save node automatically splits and saves each batch separately
    - Files are saved as: name_batch_001.mp4, name_batch_002.mp4, etc.
    
    When using specific batch_index:
    - Only that batch is output
    - Must run workflow multiple times for all batches
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "batch_size": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of frames per batch"
                }),
                "overlap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of overlapping frames between batches"
                }),
                "batch_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Which batch to output (0 = first batch, -1 = all batches)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("frames", "mask", "batch_count", "frames_in_batch", "batch_info_json")
    FUNCTION = "split_video"
    CATEGORY = "WAN Vace/Processing"
    
    def split_video(self, frames, batch_size, overlap, batch_index):
        # Ensure frames is a tensor
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames)
        
        # Get total frame count
        total_frames = frames.shape[0]
        
        # Calculate batch boundaries
        batches = []
        current_pos = 0
        
        while current_pos < total_frames:
            # Calculate end position for this batch
            end_pos = min(current_pos + batch_size, total_frames)
            
            batches.append((current_pos, end_pos))
            
            # Move to next batch position, accounting for overlap
            current_pos += batch_size - overlap
            
            # Ensure we don't go backwards
            if current_pos <= batches[-1][0]:
                current_pos = batches[-1][1]
        
        total_batches = len(batches)
        
        # Handle "all batches" mode
        if batch_index == -1:
            # Process all batches
            all_batch_frames = []
            all_masks = []
            batch_boundaries = []  # Store where each batch starts/ends in the combined output
            current_pos = 0
            
            for idx, (start_idx, end_idx) in enumerate(batches):
                batch_frames = frames[start_idx:end_idx]
                frames_in_batch = batch_frames.shape[0]
                
                # Create mask for this batch
                mask = torch.zeros((frames_in_batch, frames.shape[1], frames.shape[2]), dtype=torch.float32)
                
                # Mark overlap frames if this is not the first batch
                if idx > 0 and overlap > 0:
                    mask[:overlap] = 1.0
                
                all_batch_frames.append(batch_frames)
                all_masks.append(mask)
                
                # Record batch boundary
                batch_boundaries.append({
                    "batch_index": idx,
                    "start": current_pos,
                    "end": current_pos + frames_in_batch,
                    "original_start": start_idx,
                    "original_end": end_idx
                })
                current_pos += frames_in_batch
            
            # Concatenate all batches
            combined_frames = torch.cat(all_batch_frames, dim=0)
            combined_masks = torch.cat(all_masks, dim=0)
            
            # Create batch info with all boundaries
            batch_info = {
                "multi_batch": True,
                "batch_boundaries": batch_boundaries,
                "total_batches": total_batches,
                "batching_enabled": True,
                "batch_size": batch_size,
                "overlap": overlap
            }
            
            # Print batch information
            print(f"\n{'='*40}")
            print(f"🎬 VIDEO SPLIT INTO {total_batches} BATCHES")
            print(f"{'='*40}")
            print(f"Total frames: {total_frames}")
            print(f"Batch size: {batch_size}")
            print(f"Overlap: {overlap}")
            print(f"Total batches: {total_batches}")
            print(f"Combined output frames: {combined_frames.shape[0]}")
            
            for i, boundary in enumerate(batch_boundaries):
                print(f"Batch {i+1}: frames {boundary['start']}-{boundary['end']-1} (original {boundary['original_start']}-{boundary['original_end']-1})")
            
            print(f"{'='*40}\n")
            
            # Convert batch_info to JSON string
            batch_info_json = json.dumps(batch_info)
            
            print(f"Generated batch_info_json length: {len(batch_info_json)}")
            print(f"JSON preview: {batch_info_json[:200]}...")
            
            return (combined_frames, combined_masks, total_batches, combined_frames.shape[0], batch_info_json)
        
        else:
            # Single batch mode
            # Check if requested batch index is valid
            if batch_index >= total_batches:
                raise ValueError(f"Batch index {batch_index} out of range. Only {total_batches} batches available.")
            
            # Get the requested batch
            start_idx, end_idx = batches[batch_index]
            batch_frames = frames[start_idx:end_idx]
            frames_in_batch = batch_frames.shape[0]
            
            # Create mask for this batch
            # Black (0) = original frame, White (255) = overlap frame
            mask = torch.zeros((frames_in_batch, frames.shape[1], frames.shape[2]), dtype=torch.float32)
            
            # Mark overlap frames if this is not the first batch
            if batch_index > 0 and overlap > 0:
                # First 'overlap' frames are from previous batch
                mask[:overlap] = 1.0
            
            # Create batch info for the save node
            batch_info = {
                "batch_index": batch_index,
                "total_batches": total_batches,
                "batching_enabled": total_batches > 1,
                "start_frame": start_idx,
                "end_frame": end_idx,
                "overlap": overlap
            }
            
            # Print batch information
            print(f"\n{'='*40}")
            print(f"🎬 VIDEO SPLIT INTO BATCHES")
            print(f"{'='*40}")
            print(f"Total frames: {total_frames}")
            print(f"Batch size: {batch_size}")
            print(f"Overlap: {overlap}")
            print(f"Total batches: {total_batches}")
            print(f"Current batch: {batch_index + 1}/{total_batches}")
            print(f"Frames in batch: {frames_in_batch} (frames {start_idx}-{end_idx-1})")
            
            if overlap > 0 and batch_index > 0:
                print(f"Overlap frames: {overlap} (from previous batch)")
            
            print(f"{'='*40}\n")
            
            # Convert batch_info to JSON string
            batch_info_json = json.dumps(batch_info)
            
            return (batch_frames, mask, total_batches, frames_in_batch, batch_info_json)




class WANVaceJoinVideos:
    """Join two videos with a gap between them
    
    Takes frames from the end of video 1 and beginning of video 2,
    with a configurable gap in between filled with black/held frames.
    
    Parameters use frame numbers (not time/seconds):
    - video1_join_at_frame: Join point in video 1 (takes frames before this)
    - video2_join_at_frame: Join point in video 2 (takes frames from this point)
    
    Example: video1_join_at_frame=30, video2_join_at_frame=10
    - Takes frames ending at frame 30 from video 1
    - Adds gap frames
    - Takes frames starting at frame 10 from video 2
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video1_frames": ("IMAGE",),
                "video2_frames": ("IMAGE",),
                "video1_join_at_frame": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Join at this frame number in video 1 (0=first frame)"
                }),
                "video2_join_at_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Join at this frame number in video 2 (0=first frame)"
                }),
                "gap_frames": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of gap frames between videos"
                }),
                "target_length": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Total length of output video"
                }),
                "frame_darkness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Darkness of gap frames (0=black, 1=hold last frame)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("joined_frames", "preview_frames", "mask", "video1_unused", "video2_unused")
    FUNCTION = "join_videos"
    CATEGORY = "WAN Vace/Processing"
    
    def join_videos(self, video1_frames, video2_frames, video1_join_at_frame, video2_join_at_frame, 
                    gap_frames, target_length, frame_darkness):
        # Ensure frames are tensors
        if not isinstance(video1_frames, torch.Tensor):
            video1_frames = torch.from_numpy(video1_frames)
        if not isinstance(video2_frames, torch.Tensor):
            video2_frames = torch.from_numpy(video2_frames)
        
        # Get dimensions
        height = video1_frames.shape[1]
        width = video1_frames.shape[2]
        channels = video1_frames.shape[3]
        
        # Frame positions are now direct inputs
        
        # Calculate how many frames from each video
        frames_per_video = (target_length - gap_frames) // 2
        
        # Calculate actual frame ranges
        end_frame1 = min(video1_join_at_frame, video1_frames.shape[0])
        start_frame1 = max(0, end_frame1 - frames_per_video)
        actual_frames1 = end_frame1 - start_frame1
        
        start_frame2 = min(video2_join_at_frame, video2_frames.shape[0])
        # Adjust frames from video 2 to fill remaining space
        remaining_frames = target_length - actual_frames1 - gap_frames
        end_frame2 = min(start_frame2 + remaining_frames, video2_frames.shape[0])
        actual_frames2 = end_frame2 - start_frame2
        
        # Calculate actual output length
        actual_length = actual_frames1 + gap_frames + actual_frames2
        
        # Pre-allocate output tensors
        joined_frames = torch.zeros((actual_length, height, width, channels), dtype=video1_frames.dtype)
        preview_frames = torch.zeros((actual_length, height, width, channels), dtype=video1_frames.dtype)
        mask = torch.zeros((actual_length, height, width), dtype=torch.float32)
        
        current_idx = 0
        
        # Add frames from video 1
        if actual_frames1 > 0:
            joined_frames[current_idx:current_idx + actual_frames1] = video1_frames[start_frame1:end_frame1]
            preview_frames[current_idx:current_idx + actual_frames1] = video1_frames[start_frame1:end_frame1]
            mask[current_idx:current_idx + actual_frames1] = 0.0  # Black mask for original frames
            current_idx += actual_frames1
        
        # Add gap frames
        if gap_frames > 0:
            # Get the last frame from video 1 for holding
            if actual_frames1 > 0:
                hold_frame = video1_frames[end_frame1 - 1]
            else:
                # If no frames from video 1, use first frame of video 2
                hold_frame = video2_frames[start_frame2] if actual_frames2 > 0 else torch.zeros((height, width, channels), dtype=video1_frames.dtype)
            
            # Reference version: black or darkened frames
            if frame_darkness == 0.0:
                gap_frame = torch.zeros((height, width, channels), dtype=video1_frames.dtype)
                for i in range(gap_frames):
                    joined_frames[current_idx + i] = gap_frame
            else:
                for i in range(gap_frames):
                    joined_frames[current_idx + i] = hold_frame * frame_darkness
            
            # Preview version: held frames
            for i in range(gap_frames):
                preview_frames[current_idx + i] = hold_frame
            
            # White mask for gap
            mask[current_idx:current_idx + gap_frames] = 1.0
            current_idx += gap_frames
        
        # Add frames from video 2
        if actual_frames2 > 0:
            joined_frames[current_idx:current_idx + actual_frames2] = video2_frames[start_frame2:end_frame2]
            preview_frames[current_idx:current_idx + actual_frames2] = video2_frames[start_frame2:end_frame2]
            mask[current_idx:current_idx + actual_frames2] = 0.0  # Black mask for original frames
        
        # Extract unused portions
        video1_unused = video1_frames[:start_frame1] if start_frame1 > 0 else torch.zeros((1, height, width, channels), dtype=video1_frames.dtype)
        video2_unused = video2_frames[end_frame2:] if end_frame2 < video2_frames.shape[0] else torch.zeros((1, height, width, channels), dtype=video2_frames.dtype)
        
        # Print info
        print(f"\n{'='*40}")
        print(f"🔗 VIDEO JOIN")
        print(f"{'='*40}")
        print(f"Video 1: {actual_frames1} frames (from {start_frame1} to {end_frame1})")
        print(f"Gap: {gap_frames} frames")
        print(f"Video 2: {actual_frames2} frames (from {start_frame2} to {end_frame2})")
        print(f"Total output: {actual_length} frames")
        print(f"Frame darkness: {frame_darkness:.1f}")
        print(f"Video 1 unused: {video1_unused.shape[0]} frames")
        print(f"Video 2 unused: {video2_unused.shape[0]} frames")
        print(f"{'='*40}\n")
        
        return (joined_frames, preview_frames, mask, video1_unused, video2_unused)




class WANVaceVideoExtension:
    """Move last N frames to beginning and pad remaining with black/held frames
    
    Useful for creating seamless loops or preparing videos for certain workflows.
    Takes the last N frames and places them at the start, then fills the rest
    with padding frames up to the target length.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "extension_frames": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 500,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of frames from the end to move to beginning"
                }),
                "target_length": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Total length of output video (including extension frames)"
                }),
                "frame_darkness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Darkness of padding frames (0=black, 1=hold last frame)"
                }),
                "from_beginning": ("BOOLEAN", {
                    "default": False,
                    "display": "switch",
                    "tooltip": "If true, use the first N frames as extension frames; otherwise use the last N frames."
                }),
            },
            "optional": {
                "padding_frames": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "IMAGE")
    RETURN_NAMES = ("reference_frames", "preview_frames", "mask", "frames_moved", "last_extension_frame")
    FUNCTION = "extend_video"
    CATEGORY = "WAN Vace/Processing"
    
    def extend_video(self, frames, extension_frames, target_length, frame_darkness, from_beginning=False, padding_frames=None):
        # Ensure frames is a tensor
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames)
        # Get dimensions
        num_frames = frames.shape[0]
        height = frames.shape[1]
        width = frames.shape[2]
        channels = frames.shape[3]
        # Select extension frames block from input, independent of target_length
        print(f"[WANVaceVideoExtension] Input frames shape: {frames.shape}")
        if from_beginning:
            extension_block = frames[:extension_frames]
            print(f"[WANVaceVideoExtension] from_beginning=True: Taking frames 0 to {extension_frames-1} as extension frames.")
        else:
            extension_block = frames[-extension_frames:]
            print(f"[WANVaceVideoExtension] from_beginning=False: Taking frames {num_frames-extension_frames} to {num_frames-1} as extension frames.")
        print(f"[WANVaceVideoExtension] extension_block shape: {extension_block.shape}")
        if extension_block.shape[0] > 0:
            print(f"[WANVaceVideoExtension] First extension_block index: {0 if from_beginning else num_frames-extension_frames}")
            print(f"[WANVaceVideoExtension] Last extension_block index: {extension_block.shape[0]-1 if from_beginning else num_frames-1}")
        # Determine how many extension frames will fit in the output
        actual_extension = min(extension_block.shape[0], target_length)
        extension_frames_tensor = extension_block[:actual_extension]
        # Calculate how many padding frames we need
        padding_needed = target_length - actual_extension
        if padding_needed < 0:
            print(f"Warning: Target length ({target_length}) is less than extension frames ({actual_extension}). Truncating extension frames.")
            reference_frames = extension_frames_tensor[:target_length]
            preview_frames = reference_frames.clone()
            mask = torch.zeros((target_length, height, width), dtype=torch.float32)
            last_extension_frame = reference_frames[target_length-1].unsqueeze(0) if target_length > 0 else torch.zeros((1, height, width, channels), dtype=frames.dtype)
            return (reference_frames, preview_frames, mask, actual_extension, last_extension_frame)
        # Pre-allocate output tensors
        reference_frames = torch.zeros((target_length, height, width, channels), dtype=frames.dtype)
        preview_frames = torch.zeros((target_length, height, width, channels), dtype=frames.dtype)
        mask = torch.zeros((target_length, height, width), dtype=torch.float32)
        # Copy the extension frames to the beginning
        reference_frames[:actual_extension] = extension_frames_tensor
        preview_frames[:actual_extension] = extension_frames_tensor
        mask[:actual_extension] = 0.0  # Black mask for original frames
        # Fill the remaining frames
        if padding_needed > 0:
            hold_frame = extension_frames_tensor[-1]
            if padding_frames is not None:
                # Ensure padding_frames is a tensor
                if not isinstance(padding_frames, torch.Tensor):
                    padding_frames = torch.from_numpy(padding_frames)
                pad_count = padding_frames.shape[0]
                for i in range(actual_extension, target_length):
                    # Cycle through padding_frames if not enough
                    reference_frames[i] = padding_frames[(i - actual_extension) % pad_count]
            elif frame_darkness == 0.0:
                padding_frame = torch.zeros((height, width, channels), dtype=frames.dtype)
                for i in range(actual_extension, target_length):
                    reference_frames[i] = padding_frame
            else:
                for i in range(actual_extension, target_length):
                    reference_frames[i] = hold_frame * frame_darkness
            for i in range(actual_extension, target_length):
                preview_frames[i] = hold_frame
                mask[i] = 1.0  # White mask for padded frames
        print(f"\n{'='*40}")
        print(f"🔄 VIDEO EXTENSION")
        print(f"{'='*40}")
        print(f"Input frames: {num_frames}")
        print(f"Extension frames selected: {actual_extension}")
        print(f"Target length: {target_length}")
        print(f"Padding frames added: {padding_needed}")
        print(f"Frame darkness: {frame_darkness:.1f}")
        print(f"{'='*40}\n")
        last_extension_frame = extension_frames_tensor[-1].unsqueeze(0) if actual_extension > 0 else torch.zeros((1, height, width, channels), dtype=frames.dtype)
        return (reference_frames, preview_frames, mask, actual_extension, last_extension_frame)




class WANVaceFrameInterpolation:
    """Insert black frames between each frame for interpolation workflows
    
    Creates alternating pattern of original and black frames:
    - Original frames get black masks (0)
    - Black frames get white masks (255) to indicate where interpolation should happen
    
    Also outputs a preview version with held frames instead of black frames
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "frame_darkness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Darkness of interpolation frames (0=black, 1=original frame)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("reference_frames", "preview_frames", "mask", "original_count", "output_count")
    FUNCTION = "interpolate_frames"
    CATEGORY = "WAN Vace/Processing"
    
    def interpolate_frames(self, frames, frame_darkness):
        # Ensure frames is a tensor
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames)
        
        # Get dimensions
        num_frames = frames.shape[0]
        height = frames.shape[1] 
        width = frames.shape[2]
        channels = frames.shape[3]
        
        # Calculate output size (double the input minus 1)
        output_count = num_frames * 2 - 1
        
        # Pre-allocate output tensors
        reference_frames = torch.zeros((output_count, height, width, channels), dtype=frames.dtype)
        preview_frames = torch.zeros((output_count, height, width, channels), dtype=frames.dtype)
        mask = torch.zeros((output_count, height, width), dtype=torch.float32)
        
        # Create padding frame based on darkness setting
        if frame_darkness == 0.0:
            # Pure black
            padding_frame = torch.zeros((height, width, channels), dtype=frames.dtype)
        else:
            # We'll create padding frames individually based on each frame
            padding_frame = None
        
        # Process frames
        for i in range(num_frames):
            # Calculate position in output
            output_idx = i * 2
            
            # Add original frame to both reference and preview
            reference_frames[output_idx] = frames[i]
            preview_frames[output_idx] = frames[i]
            mask[output_idx] = 0.0  # Black mask for original frames
            
            # Add interpolation frame (except after last frame)
            if i < num_frames - 1:
                interp_idx = output_idx + 1
                
                # Reference version: padding frame
                if frame_darkness == 0.0:
                    reference_frames[interp_idx] = padding_frame
                else:
                    # Blend between black and current frame
                    reference_frames[interp_idx] = frames[i] * frame_darkness
                
                # Preview version: hold current frame
                preview_frames[interp_idx] = frames[i]
                
                # White mask for interpolation positions
                mask[interp_idx] = 1.0
        
        # Print info
        print(f"\n{'='*40}")
        print(f"🎞️ FRAME INTERPOLATION")
        print(f"{'='*40}")
        print(f"Input frames: {num_frames}")
        print(f"Output frames: {output_count}")
        print(f"Frame darkness: {frame_darkness:.1f}")
        print(f"Pattern: Original → Black → Original → Black...")
        print(f"{'='*40}\n")
        
        return (reference_frames, preview_frames, mask, num_frames, output_count)




class WANVaceBatchStartIndex:
    """Calculate the starting frame index for a given batch (1-based, outputs batch_size for index 1).
    
    Given a batch size and a batch index (1-based), outputs the starting frame index for that batch.
    Example: batch_size=81, batch_index=1 -> start_frame=81
    Example: batch_size=81, batch_index=2 -> start_frame=162
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 1000000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of frames per batch"
                }),
                "batch_index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Batch index (1 = first batch)"
                }),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("start_frame",)
    FUNCTION = "calculate_start_index"
    CATEGORY = "WAN Vace/Utility"

    def calculate_start_index(self, batch_size, batch_index):
        start_frame = batch_size * batch_index
        print(f"[WANVaceBatchStartIndex] batch_size={batch_size}, batch_index={batch_index}, start_frame={start_frame}")
        return (start_frame,)


# Node mappings


