"""
Frame utility nodes for WAN Vace Pipeline
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import folder_paths
import json
from PIL import Image
from ..utils import get_video_info


class WANVaceFrameSampler:
    """Extract frames at specific intervals from video"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "sample_mode": (["interval", "total_count", "specific_frames"], {
                    "default": "interval",
                    "tooltip": "How to sample frames"
                }),
                "interval": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Sample every Nth frame (interval mode)"
                }),
                "total_count": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Total number of frames to sample evenly (total_count mode)"
                }),
                "specific_frames": ("STRING", {
                    "default": "0,10,20,30",
                    "multiline": False,
                    "tooltip": "Comma-separated frame numbers (specific_frames mode)"
                }),
                "include_last_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Always include the last frame"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sampled_frames", "frame_info")
    FUNCTION = "sample_frames"
    CATEGORY = "WAN Vace/Processing"
    
    def sample_frames(self, frames, sample_mode, interval, total_count, 
                     specific_frames, include_last_frame):
        """Sample frames based on specified mode"""
        
        batch_size = frames.shape[0]
        
        print(f"\nFrame Sampling:")
        print(f"  Total frames: {batch_size}")
        print(f"  Mode: {sample_mode}")
        
        sampled_indices = []
        
        if sample_mode == "interval":
            # Sample every Nth frame
            print(f"  Interval: every {interval} frames")
            sampled_indices = list(range(0, batch_size, interval))
            
        elif sample_mode == "total_count":
            # Sample total_count frames evenly distributed
            print(f"  Target count: {total_count} frames")
            if total_count >= batch_size:
                # If requesting more frames than available, return all
                sampled_indices = list(range(batch_size))
            else:
                # Calculate step to evenly distribute samples
                step = (batch_size - 1) / (total_count - 1) if total_count > 1 else 0
                sampled_indices = [int(i * step) for i in range(total_count)]
                
        elif sample_mode == "specific_frames":
            # Parse specific frame numbers
            try:
                frame_numbers = [int(f.strip()) for f in specific_frames.split(',')]
                # Filter out invalid frame numbers
                sampled_indices = [f for f in frame_numbers if 0 <= f < batch_size]
                print(f"  Requested frames: {frame_numbers}")
                print(f"  Valid frames: {sampled_indices}")
            except ValueError:
                print(f"  Error parsing frame numbers: {specific_frames}")
                sampled_indices = [0]  # Default to first frame on error
        
        # Ensure we always include the last frame if requested
        last_frame_idx = batch_size - 1
        if include_last_frame and last_frame_idx not in sampled_indices:
            sampled_indices.append(last_frame_idx)
        
        # Remove duplicates and sort
        sampled_indices = sorted(list(set(sampled_indices)))
        
        print(f"  Sampled frames: {len(sampled_indices)}")
        print(f"  Frame indices: {sampled_indices[:10]}{'...' if len(sampled_indices) > 10 else ''}")
        
        # Extract the sampled frames
        sampled_frames = frames[sampled_indices]
        
        # Create info string
        info_lines = [
            f"Sampled {len(sampled_indices)} frames from {batch_size} total frames",
            f"Mode: {sample_mode}",
            f"Indices: {', '.join(map(str, sampled_indices[:20]))}{'...' if len(sampled_indices) > 20 else ''}"
        ]
        
        if sample_mode == "interval":
            info_lines.append(f"Interval: every {interval} frames")
        elif sample_mode == "total_count":
            info_lines.append(f"Target count: {total_count} frames")
        
        frame_info = "\n".join(info_lines)
        
        print(f"{'='*40}\n")
        
        return (sampled_frames, frame_info)




class WANVaceFrameInjector:
    """Insert frames at specific positions in a video"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_frames": ("IMAGE",),
                "inject_frames": ("IMAGE",),
                "injection_mode": (["manual", "nth_frame"], {
                    "default": "manual",
                    "tooltip": "Manual: use position lists, Nth Frame: inject at regular intervals"
                }),
                "injection_positions": ("STRING", {
                    "default": "10,20,30",
                    "multiline": False,
                    "tooltip": "Comma-separated base frame positions (manual mode only)"
                }),
                "inject_indices": ("STRING", {
                    "default": "0,1,2",
                    "multiline": False,
                    "tooltip": "Comma-separated indices of inject frames (manual mode only)"
                }),
                "nth_interval": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Inject every Nth frame (nth_frame mode only)"
                }),
                "start_offset": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Starting frame offset (nth_frame mode only)"
                }),
                "include_last_frame": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force include last frame if not in sequence (nth_frame mode only)"
                }),
                "inject_skip_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Use every Nth inject frame (nth_frame mode only). 1=use all, 2=use every 2nd, etc."
                }),
                "replace_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Replace frames instead of inserting (maintains total frame count)"
                }),
                "truncate_after_last": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove frames after the last injection position (replace mode only)"
                }),
            },
            "optional": {
                "base_mask": ("MASK", {
                    "tooltip": "Optional mask video to inject black/white frames into at same positions"
                }),
                "inject_white_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Inject white mask frames instead of black (default: black)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("frames", "injection_mask", "injection_info")
    FUNCTION = "inject_frames"
    CATEGORY = "WAN Vace/Processing"
    
    def inject_frames(self, base_frames, inject_frames, injection_mode, injection_positions, inject_indices, 
                     nth_interval, start_offset, include_last_frame, inject_skip_frames, replace_mode, 
                     truncate_after_last, base_mask=None, inject_white_mask=False):
        """Insert or replace frames at specified positions"""
        
        base_count = base_frames.shape[0]
        inject_count = inject_frames.shape[0]
        
        # Extract image dimensions for proper mask creation
        height, width = base_frames.shape[1:3]
        
        print(f"\nFrame Injection:")
        print(f"  Base video: {base_count} frames")
        print(f"  Inject frames: {inject_count} frames")
        print(f"  Frame dimensions: {height}x{width}")
        print(f"  Injection mode: {injection_mode}")
        print(f"  Mode: {'Replace' if replace_mode else 'Insert'}")
        
        # Validate base_mask if provided
        if base_mask is not None:
            print(f"  Base mask provided: {base_mask.shape}")
            if base_mask.shape[0] != base_count:
                print(f"  Warning: Base mask frame count ({base_mask.shape[0]}) doesn't match base frames ({base_count})")
                # Continue anyway, we'll handle it
        
        # Handle different injection modes
        if injection_mode == "nth_frame":
            # Calculate positions based on nth interval
            print(f"  Nth interval: every {nth_interval} frames")
            print(f"  Start offset: {start_offset}")
            print(f"  Include last frame: {include_last_frame}")
            print(f"  Inject skip frames: {inject_skip_frames} (use every {inject_skip_frames} inject frame)")
            
            # When using inject_skip_frames, we need to adjust the interval
            # If original sampling was every 3rd frame, and we skip every 2nd inject frame,
            # then we need to inject at every 6th position (3 * 2)
            effective_interval = nth_interval * inject_skip_frames
            
            # Calculate expected positions with the effective interval
            positions = list(range(start_offset, base_count, effective_interval))
            
            # Include last frame if requested and not already included
            last_frame_idx = base_count - 1
            if include_last_frame and last_frame_idx not in positions and last_frame_idx >= start_offset:
                positions.append(last_frame_idx)
                positions.sort()
            
            # Apply inject_skip_frames to determine which inject frames to use
            inject_indices_to_use = list(range(0, inject_count, inject_skip_frames))
            actual_inject_count = len(inject_indices_to_use)
            actual_positions = positions[:actual_inject_count]  # Take only as many positions as we have inject frames
            
            # Validate we have enough positions
            if len(actual_positions) < actual_inject_count:
                error_msg = (f"Not enough base positions for inject frames:\n"
                           f"  - Base video: {base_count} frames\n"
                           f"  - Effective interval: every {effective_interval} frames (nth {nth_interval} × skip {inject_skip_frames})\n"
                           f"  - Positions available: {len(positions)}\n"
                           f"  - Inject frames to use: {actual_inject_count} (every {inject_skip_frames} from {inject_count})\n"
                           f"  - Need at least {actual_inject_count} positions")
                print(f"  Error: {error_msg}")
                
                if base_mask is not None:
                    return (base_frames, base_mask, error_msg)
                else:
                    mask = torch.zeros(base_count, height, width, dtype=torch.float32)
                    return (base_frames, mask, error_msg)
            
            # Update for actual usage
            positions = actual_positions
            indices = inject_indices_to_use
            
            print(f"  Effective interval: every {effective_interval} frames (nth_interval {nth_interval} × inject_skip {inject_skip_frames})")
            print(f"  ✓ Using {actual_inject_count} inject frames (indices: {indices[:10]}{'...' if len(indices) > 10 else ''})")
            print(f"  ✓ Injecting at {len(positions)} positions: {positions[:10]}{'...' if len(positions) > 10 else ''}")
            
        else:
            # Manual mode - parse injection positions and indices
            try:
                positions = [int(p.strip()) for p in injection_positions.split(',')]
                indices = [int(i.strip()) for i in inject_indices.split(',')]
                
                print(f"  Base positions: {positions}")
                print(f"  Inject indices: {indices}")
                
                # Validate matching lengths
                if len(positions) != len(indices):
                    print(f"  Error: Mismatch - {len(positions)} positions but {len(indices)} inject indices")
                    # Return original mask or create empty one
                    if base_mask is not None:
                        return (base_frames, base_mask, f"Error: Mismatch - {len(positions)} positions but {len(indices)} inject indices")
                    else:
                        mask = torch.zeros(base_count, height, width, dtype=torch.float32)
                        return (base_frames, mask, f"Error: Mismatch - {len(positions)} positions but {len(indices)} inject indices")
                
            except ValueError as e:
                print(f"  Error parsing positions or indices: {e}")
                # Return original frames on error with correct mask dimensions
                if base_mask is not None:
                    return (base_frames, base_mask, f"Error: Invalid format - {str(e)}")
                else:
                    mask = torch.zeros(base_count, height, width, dtype=torch.float32)
                    return (base_frames, mask, f"Error: Invalid format - {str(e)}")
        
        # Create frame pairs based on mode
        if injection_mode == "nth_frame":
            # Already validated, create pairs
            frame_pairs = list(zip(indices, positions))
        else:
            # Manual mode - create pairs from parsed lists
            frame_pairs = list(zip(indices, positions))
            print(f"  Frame pairs (inject_idx → base_pos): {frame_pairs}")
        
        # Validate all inject indices are in range
        for inject_idx, _ in frame_pairs:
            if inject_idx < 0 or inject_idx >= inject_count:
                print(f"  Warning: Inject index {inject_idx} out of range (0-{inject_count-1})")
        
        if replace_mode:
            # Replace mode: Replace frames at specified positions
            result_frames = base_frames.clone()
            
            # Handle base mask if provided
            if base_mask is not None:
                # Use base mask and inject black frames at injection positions
                result_mask = base_mask.clone()
                # Ensure mask has correct dimensions
                if len(result_mask.shape) == 3:  # frames, height, width
                    pass  # Already correct
                elif len(result_mask.shape) == 2:  # Single mask, height, width
                    result_mask = result_mask.unsqueeze(0)  # Add frame dimension
                else:
                    print(f"  Warning: Unexpected mask shape {result_mask.shape}, creating new mask")
                    result_mask = torch.zeros(base_count, height, width, dtype=torch.float32)
            else:
                # Create injection position mask (original behavior)
                result_mask = torch.zeros(base_count, height, width, dtype=torch.float32)
            
            replaced_pairs = []
            for inject_idx, base_pos in frame_pairs:
                # Validate indices
                if inject_idx < 0 or inject_idx >= inject_count:
                    print(f"  Skipping: Inject index {inject_idx} out of range (0-{inject_count-1})")
                    continue
                if base_pos < 0 or base_pos >= base_count:
                    print(f"  Skipping: Base position {base_pos} out of range (0-{base_count-1})")
                    continue
                
                # Get inject frame
                inject_frame = inject_frames[inject_idx]
                
                # Check if inject frame needs resizing
                if inject_frame.shape[0] != height or inject_frame.shape[1] != width:
                    print(f"  Warning: Resizing inject frame {inject_idx} from {inject_frame.shape[:2]} to {height}x{width}")
                    inject_pil = Image.fromarray((inject_frame.cpu().numpy() * 255).astype('uint8'))
                    inject_pil = inject_pil.resize((width, height), Image.LANCZOS)
                    inject_frame = torch.from_numpy(np.array(inject_pil).astype(np.float32) / 255.0)
                
                # Apply replacement
                result_frames[base_pos] = inject_frame
                
                # Handle mask injection
                if base_mask is not None:
                    # Inject white or black mask frame at this position
                    result_mask[base_pos] = 1.0 if inject_white_mask else 0.0
                else:
                    # Original behavior - mark injection position
                    result_mask[base_pos] = 1.0  # Full mask for replaced frame
                    
                replaced_pairs.append((inject_idx, base_pos))
            
            # Handle truncation if requested
            if truncate_after_last and replaced_pairs:
                # Find the last injection position
                last_injection_pos = max(pos for _, pos in replaced_pairs)
                truncate_at = last_injection_pos + 1
                
                if truncate_at < base_count:
                    # Truncate frames and mask
                    frames_removed = base_count - truncate_at
                    result_frames = result_frames[:truncate_at]
                    result_mask = result_mask[:truncate_at]
                    
                    print(f"  ✓ Truncated output to {truncate_at} frames (removed {frames_removed} frames after position {last_injection_pos})")
                    
                    info_lines = [
                        f"Replaced {len(replaced_pairs)} frames in {base_count} frame video",
                        f"Truncated to {truncate_at} frames (removed {frames_removed} trailing frames)",
                        f"Frame pairs applied:"
                    ]
                else:
                    info_lines = [
                        f"Replaced {len(replaced_pairs)} frames in {base_count} frame video",
                        f"Frame pairs applied:"
                    ]
            else:
                info_lines = [
                    f"Replaced {len(replaced_pairs)} frames in {base_count} frame video",
                    f"Frame pairs applied:"
                ]
            
            for inject_idx, base_pos in replaced_pairs:
                info_lines.append(f"  Inject frame {inject_idx} → Base position {base_pos}")
            
        else:
            # Insert mode: Insert frames, shifting existing frames
            if truncate_after_last:
                print(f"  ⚠️ Note: truncate_after_last is ignored in insert mode")
            
            # Sort pairs by base position for correct insertion order
            sorted_pairs = sorted(frame_pairs, key=lambda x: x[1])
            
            frame_list = []
            mask_list = []
            
            # Prepare base mask if provided
            if base_mask is not None:
                # Ensure mask has correct dimensions
                if len(base_mask.shape) == 3:  # frames, height, width
                    working_base_mask = base_mask
                elif len(base_mask.shape) == 2:  # Single mask, height, width
                    working_base_mask = base_mask.unsqueeze(0).expand(base_count, -1, -1)
                else:
                    print(f"  Warning: Unexpected mask shape {base_mask.shape}, creating new mask")
                    working_base_mask = None
            else:
                working_base_mask = None
            
            # Track offset due to insertions
            offset = 0
            last_pos = 0
            
            inserted_pairs = []
            
            for inject_idx, base_pos in sorted_pairs:
                # Validate indices
                if inject_idx < 0 or inject_idx >= inject_count:
                    print(f"  Skipping: Inject index {inject_idx} out of range (0-{inject_count-1})")
                    continue
                if base_pos < 0 or base_pos > base_count:  # Allow position at end
                    print(f"  Skipping: Base position {base_pos} out of range (0-{base_count})")
                    continue
                    
                # Adjust position for previous insertions
                adjusted_pos = base_pos + offset
                
                # Add frames up to insertion point
                if last_pos < base_count and base_pos > last_pos:
                    frame_count = min(base_pos, base_count) - last_pos
                    frame_list.append(base_frames[last_pos:min(base_pos, base_count)])
                    
                    # Add corresponding mask frames
                    if working_base_mask is not None:
                        mask_list.append(working_base_mask[last_pos:min(base_pos, base_count)])
                    else:
                        mask_list.append(torch.zeros(frame_count, height, width, dtype=torch.float32))
                
                # Get and prepare inject frame
                inject_frame = inject_frames[inject_idx:inject_idx+1]
                if inject_frame.shape[1] != height or inject_frame.shape[2] != width:
                    print(f"  Warning: Resizing inject frame {inject_idx} from {inject_frame.shape[1:3]} to {height}x{width}")
                    inject_pil = Image.fromarray((inject_frame[0].cpu().numpy() * 255).astype('uint8'))
                    inject_pil = inject_pil.resize((width, height), Image.LANCZOS)
                    inject_frame = torch.from_numpy(np.array(inject_pil).astype(np.float32) / 255.0).unsqueeze(0)
                
                # Insert the frame
                frame_list.append(inject_frame)
                
                # Insert mask (white/black frame if base_mask provided, else injection marker)
                if working_base_mask is not None:
                    # Insert white or black mask frame
                    mask_value = 1.0 if inject_white_mask else 0.0
                    mask_list.append(torch.full((1, height, width), mask_value, dtype=torch.float32))
                else:
                    # Original behavior - mark injection position
                    mask_list.append(torch.ones(1, height, width, dtype=torch.float32))
                    
                inserted_pairs.append((inject_idx, adjusted_pos))
                
                last_pos = base_pos
                offset += 1
            
            # Add remaining frames
            if last_pos < base_count:
                remaining_count = base_count - last_pos
                frame_list.append(base_frames[last_pos:])
                
                # Add corresponding mask frames
                if working_base_mask is not None:
                    mask_list.append(working_base_mask[last_pos:])
                else:
                    mask_list.append(torch.zeros(remaining_count, height, width, dtype=torch.float32))
            
            # Concatenate all frames and masks
            if frame_list:
                result_frames = torch.cat(frame_list, dim=0)
                result_mask = torch.cat(mask_list, dim=0)
            else:
                result_frames = base_frames
                if working_base_mask is not None:
                    result_mask = working_base_mask
                else:
                    result_mask = torch.zeros(base_count, height, width, dtype=torch.float32)
            
            info_lines = [
                f"Inserted {len(inserted_pairs)} frames into {base_count} frame video",
                f"New total: {result_frames.shape[0]} frames",
                f"Frame pairs applied (with adjusted positions):"
            ]
            for inject_idx, adj_pos in inserted_pairs:
                info_lines.append(f"  Inject frame {inject_idx} → Position {adj_pos}")
        
        injection_info = "\n".join(info_lines)
        print(f"  Result: {result_frames.shape[0]} frames")
        print(f"  Output mask shape: {result_mask.shape} (should be frames x height x width)")
        print(f"  Output frames shape: {result_frames.shape}")
        if base_mask is not None:
            mask_color = "White" if inject_white_mask else "Black"
            print(f"  Mask mode: {mask_color} frames injected at positions")
        else:
            print(f"  Mask mode: Injection position markers")
        print(f"{'='*40}\n")
        
        return (result_frames, result_mask, injection_info)
