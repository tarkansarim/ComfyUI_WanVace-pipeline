import comfy.utils
import math
import nodes
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, binary_closing, binary_fill_holes
import json
import gc

print("[WanCropAndStitch] Module loaded successfully!")
print("[WanCropAndStitch] Debug: This module is being imported by ComfyUI")

# Define a custom type for lazy evaluation
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")


def rescale_i(samples, width, height, algorithm: str):
    batch_size = samples.shape[0]
    algorithm_fn = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
    
    processed_frames = []
    for i in range(batch_size):
        # Process each frame individually
        frame = samples[i:i+1]  # Keep batch dimension: [1, H, W, C]
        frame = frame.movedim(-1, 1)  # [1, H, W, C] -> [1, C, H, W]
        frame_pil = F.to_pil_image(frame[0].cpu()).resize((width, height), algorithm_fn)
        frame_tensor = F.to_tensor(frame_pil).unsqueeze(0)  # [1, C, H, W]
        frame_tensor = frame_tensor.movedim(1, -1)  # [1, C, H, W] -> [1, H, W, C]
        processed_frames.append(frame_tensor)
    
    # Concatenate all processed frames back into batch
    result = torch.cat(processed_frames, dim=0)  # [B, H, W, C]
    return result


def rescale_m_batch(samples, width, height, algorithm: str):
    batch_size = samples.shape[0]
    algorithm_fn = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
    
    processed_frames = []
    for i in range(batch_size):
        # Process each frame individually
        frame = samples[i:i+1]  # Keep batch dimension: [1, H, W]
        frame = frame.unsqueeze(1)  # [1, H, W] -> [1, 1, H, W]
        frame_pil = F.to_pil_image(frame[0].cpu()).resize((width, height), algorithm_fn)
        frame_tensor = F.to_tensor(frame_pil).unsqueeze(0)  # [1, 1, H, W]
        frame_tensor = frame_tensor.squeeze(1)  # [1, 1, H, W] -> [1, H, W]
        processed_frames.append(frame_tensor)
    
    # Concatenate all processed frames back into batch
    result = torch.cat(processed_frames, dim=0)  # [B, H, W]
    return result


def rescale_m(samples, width, height, algorithm: str):
    # Handle both single-frame [1, H, W] and batch [B, H, W] inputs
    original_shape = samples.shape
    batch_size = original_shape[0]
    
    if batch_size == 1:
        # Original single-frame processing logic (exactly as in backup)
        samples = samples.unsqueeze(1)
        algorithm = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
        samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
        samples = F.to_tensor(samples_pil).unsqueeze(0)
        samples = samples.squeeze(1)
        return samples
    else:
        # Multi-frame batch processing - iterate through each frame
        algorithm = getattr(Image, algorithm.upper())
        processed_frames = []
        
        for i in range(batch_size):
            # Extract single frame [H, W] and convert to [1, H, W]
            frame = samples[i:i+1]  # [1, H, W]
            frame = frame.unsqueeze(1)  # [1, 1, H, W]
            frame_pil = F.to_pil_image(frame[0].cpu()).resize((width, height), algorithm)
            frame_tensor = F.to_tensor(frame_pil).unsqueeze(0)  # [1, 1, H, W]
            frame_tensor = frame_tensor.squeeze(1)  # [1, H, W]
            processed_frames.append(frame_tensor)
        
        # Concatenate all processed frames back into batch
        result = torch.cat(processed_frames, dim=0)  # [B, H, W]
        return result


def validate_aspect_ratio(image_width, image_height, mask_width, mask_height, tolerance=0.01):
    """Validate that mask and image have the same aspect ratio within tolerance"""
    image_aspect = image_width / image_height
    mask_aspect = mask_width / mask_height
    
    aspect_diff = abs(image_aspect - mask_aspect)
    max_allowed_diff = tolerance * max(image_aspect, mask_aspect)
    
    if aspect_diff > max_allowed_diff:
        raise ValueError(f"Mask aspect ratio ({mask_aspect:.4f}) doesn't match image aspect ratio ({image_aspect:.4f}). "
                        f"Difference: {aspect_diff:.4f}, Maximum allowed: {max_allowed_diff:.4f}")
    
    return True


def resize_mask_to_image(mask, image_width, image_height):
    """Resize mask to match image dimensions after aspect ratio validation"""
    # Get current mask dimensions
    mask_height, mask_width = mask.shape[1], mask.shape[2]
    
    # Validate aspect ratio first
    validate_aspect_ratio(image_width, image_height, mask_width, mask_height)
    
    # Resize mask to match image dimensions
    if mask_width != image_width or mask_height != image_height:
        mask = rescale_m(mask, image_width, image_height, 'bilinear')
    
    return mask


def preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height):
    current_width, current_height = image.shape[2], image.shape[1]  # Image size [batch, height, width, channels]
    
    if preresize_mode == "ensure minimum resolution":
        if current_width >= preresize_min_width and current_height >= preresize_min_height:
            return image, mask, optional_context_mask

        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height

        scale_factor = max(scale_factor_min_width, scale_factor_min_height)

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, upscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'bilinear')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
        
        assert target_width >= preresize_min_width and target_height >= preresize_min_height, \
            f"Internal error: After resizing, target size {target_width}x{target_height} is smaller than min size {preresize_min_width}x{preresize_min_height}"

    elif preresize_mode == "ensure minimum and maximum resolution":
        if preresize_min_width <= current_width <= preresize_max_width and preresize_min_height <= current_height <= preresize_max_height:
            return image, mask, optional_context_mask

        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height
        scale_factor_min = max(scale_factor_min_width, scale_factor_min_height)

        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

        if scale_factor_min > 1 and scale_factor_max < 1:
            assert False, "Cannot meet both minimum and maximum resolution requirements with aspect ratio preservation."
        
        if scale_factor_min > 1:  # We're upscaling to meet min resolution
            scale_factor = scale_factor_min
            rescale_algorithm = upscale_algorithm  # Use upscale algorithm for min resolution
        else:  # We're downscaling to meet max resolution
            scale_factor = scale_factor_max
            rescale_algorithm = downscale_algorithm  # Use downscale algorithm for max resolution

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, rescale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'bilinear')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
        
        assert preresize_min_width <= target_width <= preresize_max_width and preresize_min_height <= target_height <= preresize_max_height, \
            f"Internal error: After resizing, target size {target_width}x{target_height} is outside the allowed range {preresize_min_width}x{preresize_min_height} to {preresize_max_width}x{preresize_max_height}"

    elif preresize_mode == "ensure maximum resolution":
        if current_width <= preresize_max_width and current_height <= preresize_max_height:
            return image, mask, optional_context_mask

        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height

        scale_factor = min(scale_factor_max_width, scale_factor_max_height)

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, downscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'bilinear')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
        
        assert target_width <= preresize_max_width and target_height <= preresize_max_height, \
            f"Internal error: After resizing, target size {target_width}x{target_height} is larger than max size {preresize_max_width}x{preresize_max_height}"

    return image, mask, optional_context_mask


def fillholes_iterative_hipass_fill_m(samples):
    thresholds = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    mask_np = samples.squeeze(0).cpu().numpy()

    for threshold in thresholds:
        thresholded_mask = mask_np >= threshold
        closed_mask = binary_closing(thresholded_mask, structure=np.ones((3, 3)), border_value=1)
        filled_mask = binary_fill_holes(closed_mask)
        mask_np = np.maximum(mask_np, np.where(filled_mask != 0, threshold, 0))

    final_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

    return final_mask


def hipassfilter_m(samples, threshold):
    if samples is None:
        return None
    filtered_mask = samples.clone()
    filtered_mask[filtered_mask < threshold] = 0
    return filtered_mask


def expand_m(mask, pixels):
    sigma = pixels / 4
    mask_np = mask.squeeze(0).cpu().numpy()
    kernel_size = math.ceil(sigma * 1.5 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_mask = grey_dilation(mask_np, footprint=kernel)
    dilated_mask = dilated_mask.astype(np.float32)
    dilated_mask = torch.from_numpy(dilated_mask)
    dilated_mask = torch.clamp(dilated_mask, 0.0, 1.0)
    return dilated_mask.unsqueeze(0)


def invert_m(samples):
    inverted_mask = samples.clone()
    inverted_mask = 1.0 - inverted_mask
    return inverted_mask


def blur_m(samples, pixels):
    mask = samples.squeeze(0)
    sigma = pixels / 4 
    mask_np = mask.cpu().numpy()
    blurred_mask = gaussian_filter(mask_np, sigma=sigma)
    blurred_mask = torch.from_numpy(blurred_mask).float()
    blurred_mask = torch.clamp(blurred_mask, 0.0, 1.0)
    return blurred_mask.unsqueeze(0)


def extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
    B, H, W, C = image.shape

    new_H = int(H * (1.0 + extend_up_factor - 1.0 + extend_down_factor - 1.0))
    new_W = int(W * (1.0 + extend_left_factor - 1.0 + extend_right_factor - 1.0))

    assert new_H >= 0, f"Error: Trying to crop too much, height ({new_H}) must be >= 0"
    assert new_W >= 0, f"Error: Trying to crop too much, width ({new_W}) must be >= 0"

    expanded_image = torch.zeros(1, new_H, new_W, C, device=image.device)
    expanded_mask = torch.ones(1, new_H, new_W, device=mask.device)
    expanded_optional_context_mask = torch.zeros(1, new_H, new_W, device=optional_context_mask.device)

    up_padding = int(H * (extend_up_factor - 1.0))
    down_padding = new_H - H - up_padding
    left_padding = int(W * (extend_left_factor - 1.0))
    right_padding = new_W - W - left_padding

    slice_target_up = max(0, up_padding)
    slice_target_down = min(new_H, up_padding + H)
    slice_target_left = max(0, left_padding)
    slice_target_right = min(new_W, left_padding + W)

    slice_source_up = max(0, -up_padding)
    slice_source_down = min(H, new_H - up_padding)
    slice_source_left = max(0, -left_padding)
    slice_source_right = min(W, new_W - left_padding)

    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    expanded_image[:, :, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = image[:, :, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    if up_padding > 0:
        expanded_image[:, :, :up_padding, slice_target_left:slice_target_right] = image[:, :, 0:1, slice_source_left:slice_source_right].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, slice_target_left:slice_target_right] = image[:, :, -1:, slice_source_left:slice_source_right].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, :, :left_padding] = expanded_image[:, :, :, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, :, -right_padding:] = expanded_image[:, :, :, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    expanded_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    expanded_optional_context_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = optional_context_mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]

    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    return expanded_image, expanded_mask, expanded_optional_context_mask


def debug_context_location_in_image(image, x, y, w, h):
    debug_image = image.clone()
    debug_image[:, y:y+h, x:x+w, :] = 1.0 - debug_image[:, y:y+h, x:x+w, :]
    return debug_image


def findcontextarea_m(mask):
    mask_squeezed = mask[0]  # Now shape is [H, W]
    non_zero_indices = torch.nonzero(mask_squeezed)

    H, W = mask_squeezed.shape

    if non_zero_indices.numel() == 0:
        x, y = -1, -1
        w, h = -1, -1
    else:
        y = torch.min(non_zero_indices[:, 0]).item()
        x = torch.min(non_zero_indices[:, 1]).item()
        y_max = torch.max(non_zero_indices[:, 0]).item()
        x_max = torch.max(non_zero_indices[:, 1]).item()
        w = x_max - x + 1  # +1 to include the max index
        h = y_max - y + 1  # +1 to include the max index

    # Handle edge case where context would be empty
    if w <= 0 or h <= 0:
        x, y = -1, -1
        w, h = -1, -1
        context = torch.zeros(1, 0, 0, device=mask.device)
    else:
        context = mask[:, y:y+h, x:x+w]
    
    return context, x, y, w, h


def growcontextarea_m(context, mask, x, y, w, h, extend_factor):
    img_h, img_w = mask.shape[1], mask.shape[2]

    # Compute intended growth in each direction
    grow_left = int(round(w * (extend_factor-1.0) / 2.0))
    grow_right = int(round(w * (extend_factor-1.0) / 2.0))
    grow_up = int(round(h * (extend_factor-1.0) / 2.0))
    grow_down = int(round(h * (extend_factor-1.0) / 2.0))

    # Try to grow left, but clamp at 0
    new_x = x - grow_left
    if new_x < 0:
        new_x = 0

    # Try to grow up, but clamp at 0
    new_y = y - grow_up
    if new_y < 0:
        new_y = 0

    # Right edge
    new_x2 = x + w + grow_right
    if new_x2 > img_w:
        new_x2 = img_w

    # Bottom edge
    new_y2 = y + h + grow_down
    if new_y2 > img_h:
        new_y2 = img_h

    # New width and height
    new_w = new_x2 - new_x
    new_h = new_y2 - new_y

    # Extract the context
    new_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]

    if new_h < 0 or new_w < 0:
        new_x = 0
        new_y = 0
        new_w = mask.shape[2]
        new_h = mask.shape[1]
        new_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]

    return new_context, new_x, new_y, new_w, new_h


def combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask):
    _, x_opt, y_opt, w_opt, h_opt = findcontextarea_m(optional_context_mask)
    if x == -1:
        x, y, w, h = x_opt, y_opt, w_opt, h_opt
    if x_opt == -1:
        x_opt, y_opt, w_opt, h_opt = x, y, w, h
    if x == -1:
        return torch.zeros(1, 0, 0, device=mask.device), -1, -1, -1, -1
    new_x = min(x, x_opt)
    new_y = min(y, y_opt)
    new_x_max = max(x + w, x_opt + w_opt)
    new_y_max = max(y + h, y_opt + h_opt)
    new_w = new_x_max - new_x
    new_h = new_y_max - new_y
    combined_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]
    return combined_context, new_x, new_y, new_w, new_h


def pad_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


# (No-op helper for identity stitchers removed; native bypass handled in UI/engine)

def expand_image_to_canvas_like(reference_canvas_h, reference_canvas_w, cto_x, cto_y, image, mask):
    """Expand a single-frame image/mask to match a reference canvas size and offsets.

    - reference_canvas_h, reference_canvas_w: target canvas spatial size
    - cto_x, cto_y: where the original image's top-left should be placed in the canvas
    - image: [1, H, W, C]
    - mask: [1, H, W]
    Returns: expanded_image [1, Hc, Wc, C], expanded_mask [1, Hc, Wc]
    """
    _, img_h, img_w, img_c = image.shape
    canvas_h, canvas_w = int(reference_canvas_h), int(reference_canvas_w)

    # Create empty canvas tensors
    expanded_image = torch.zeros((1, canvas_h, canvas_w, img_c), device=image.device, dtype=image.dtype)
    expanded_mask = torch.ones((1, canvas_h, canvas_w), device=mask.device, dtype=mask.dtype)

    up_padding = int(cto_y)
    left_padding = int(cto_x)
    down_padding = max(0, canvas_h - img_h - up_padding)
    right_padding = max(0, canvas_w - img_w - left_padding)

    # Channel-first for padding operations
    image_chw = image.permute(0, 3, 1, 2)
    expanded_image_chw = expanded_image.permute(0, 3, 1, 2)

    # Paste original image region
    expanded_image_chw[:, :, up_padding:up_padding + img_h, left_padding:left_padding + img_w] = image_chw

    # Edge-repeat fills to match original behavior
    if up_padding > 0:
        expanded_image_chw[:, :, :up_padding, left_padding:left_padding + img_w] = image_chw[:, :, 0:1, :].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image_chw[:, :, -down_padding:, left_padding:left_padding + img_w] = image_chw[:, :, -1:, :].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image_chw[:, :, up_padding:up_padding + img_h, :left_padding] = expanded_image_chw[:, :, up_padding:up_padding + img_h, left_padding:left_padding + 1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image_chw[:, :, up_padding:up_padding + img_h, -right_padding:] = expanded_image_chw[:, :, up_padding:up_padding + img_h, -right_padding - 1:-right_padding].repeat(1, 1, 1, right_padding)

    # Back to BHWC
    expanded_image = expanded_image_chw.permute(0, 2, 3, 1)

    # Mask placement (no edge-repeat needed; keep 1s outside original image)
    expanded_mask[:, up_padding:up_padding + img_h, left_padding:left_padding + img_w] = mask

    return expanded_image, expanded_mask


def crop_magic_im(image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm, minimize_horizontal_padding=False):
    image = image.clone()
    mask = mask.clone()
    
    # Ok this is the most complex function in this node. The one that does the magic after all the preparation done by the other nodes.
    # Basically this function determines the right context area that encompasses the whole context area (mask+optional_context_mask),
    # that is ideally within the bounds of the original image, and that has the right aspect ratio to match target width and height.
    # It may grow the image if the aspect ratio wouldn't fit in the original image.
    # It keeps track of that growing to then be able to crop the image in the stitch node.
    # Finally, it crops the context area and resizes it to be exactly target_w and target_h.
    # It keeps track of that resize to be able to revert it in the stitch node.

    # Check for invalid inputs
    if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
        return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

    # Step 1: Pad target dimensions to be multiples of padding
    if padding != 0:
        padding = int(padding)
        target_w = pad_to_multiple(target_w, padding)
        target_h = pad_to_multiple(target_h, padding)

    # Step 2: Calculate target aspect ratio
    target_aspect_ratio = target_w / target_h

    # Step 3: Grow current context area to meet the target aspect ratio
    B, image_h, image_w, C = image.shape
    context_aspect_ratio = w / h
    
    # Debug: Print aspect ratio information
    print(f"[crop_magic_im] Input context: x={x}, y={y}, w={w}, h={h}")
    print(f"[crop_magic_im] Context aspect ratio: {context_aspect_ratio:.2f}, Target aspect ratio: {target_aspect_ratio:.2f}")
    print(f"[crop_magic_im] Target dimensions: {target_w}x{target_h}")
    print(f"[crop_magic_im] minimize_horizontal_padding: {minimize_horizontal_padding}")
    
    if context_aspect_ratio < target_aspect_ratio:
        # Grow width to meet aspect ratio
        new_w = int(h * target_aspect_ratio)
        new_h = h
        
        # Always center the expansion first (like original implementation)
        new_x = x - (new_w - w) // 2
        new_y = y
        
        if minimize_horizontal_padding:
            # Special optimization: try to reduce padding by smart positioning
            # Only activate if the new width exceeds image bounds
            if new_x < 0 or new_x + new_w > image_w:
                # Try to fit within image bounds if possible
                if new_w <= image_w:
                    # Width fits, just need to reposition
                    if new_x < 0:
                        new_x = 0
                    elif new_x + new_w > image_w:
                        new_x = image_w - new_w
                else:
                    # Width doesn't fit, minimize overflow equally
                    overflow = (new_w - image_w) // 2
                    new_x = -overflow
        else:
            # Original behavior: adjust centered position to keep within bounds
            if new_x < 0:
                shift = -new_x
                if new_x + new_w + shift <= image_w:
                    new_x += shift
                else:
                    overflow = (new_w - image_w) // 2
                    new_x = -overflow
            elif new_x + new_w > image_w:
                overflow = new_x + new_w - image_w
                if new_x - overflow >= 0:
                    new_x -= overflow
                else:
                    overflow = (new_w - image_w) // 2
                    new_x = -overflow

    else:
        # Grow height to meet aspect ratio
        new_w = w
        new_h = int(w / target_aspect_ratio)
        new_x = x
        new_y = y - (new_h - h) // 2

        # Adjust new_y to keep within bounds
        if new_y < 0:
            shift = -new_y
            if new_y + new_h + shift <= image_h:
                new_y += shift
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow
        elif new_y + new_h > image_h:
            overflow = new_y + new_h - image_h
            if new_y - overflow >= 0:
                new_y -= overflow
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow
    
    # Debug: Print adjusted dimensions
    print(f"[crop_magic_im] After aspect ratio adjustment: new_x={new_x}, new_y={new_y}, new_w={new_w}, new_h={new_h}")

    # Step 4: Grow the image to accommodate the new context area
    up_padding, down_padding, left_padding, right_padding = 0, 0, 0, 0

    expanded_image_w = image_w
    expanded_image_h = image_h

    # Adjust width for left overflow (x < 0) and right overflow (x + w > image_w)
    if new_x < 0:
        left_padding = -new_x
        expanded_image_w += left_padding
    if new_x + new_w > image_w:
        right_padding = (new_x + new_w - image_w)
        expanded_image_w += right_padding
    # Adjust height for top overflow (y < 0) and bottom overflow (y + h > image_h)
    if new_y < 0:
        up_padding = -new_y
        expanded_image_h += up_padding 
    if new_y + new_h > image_h:
        down_padding = (new_y + new_h - image_h)
        expanded_image_h += down_padding

    # Step 5: Create the new image and mask
    expanded_image = torch.zeros((image.shape[0], expanded_image_h, expanded_image_w, image.shape[3]), device=image.device)
    expanded_mask = torch.ones((mask.shape[0], expanded_image_h, expanded_image_w), device=mask.device)

    # Reorder the tensors to match the required dimension format for padding
    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    # Ensure the expanded image has enough room to hold the padded version of the original image
    expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = image

    # Fill the new extended areas with the edge values of the image
    if up_padding > 0:
        expanded_image[:, :, :up_padding, left_padding:left_padding + image_w] = image[:, :, 0:1, left_padding:left_padding + image_w].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, left_padding:left_padding + image_w] = image[:, :, -1:, left_padding:left_padding + image_w].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, :left_padding] = expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, -right_padding:] = expanded_image[:, :, up_padding:up_padding + image_h, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    # Reorder the tensors back to [B, H, W, C] format
    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    # Same for the mask
    expanded_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

    # Record the cto values (canvas to original)
    cto_x = left_padding
    cto_y = up_padding
    cto_w = image_w
    cto_h = image_h

    # The final expanded image and mask
    canvas_image = expanded_image
    canvas_mask = expanded_mask

    # Step 6: Crop the image and mask around x, y, w, h
    ctc_x = new_x+left_padding
    ctc_y = new_y+up_padding
    ctc_w = new_w
    ctc_h = new_h

    # Crop the image and mask
    cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    # Step 7: Resize image and mask to the target width and height
    # Decide which algorithm to use based on the scaling direction
    if target_w > ctc_w or target_h > ctc_h:  # Upscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
        cropped_mask = rescale_m_batch(cropped_mask, target_w, target_h, upscale_algorithm)
    else:  # Downscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
        cropped_mask = rescale_m_batch(cropped_mask, target_w, target_h, downscale_algorithm)

    return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h


def stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
    canvas_image = canvas_image.clone()
    inpainted_image = inpainted_image.clone()
    mask = mask.clone()

    # Resize inpainted image and mask to match the context size
    _, h, w, _ = inpainted_image.shape
    if ctc_w > w or ctc_h > h:  # Upscaling
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
    else:  # Downscaling
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)

    # Clamp mask to [0, 1] and expand to match image channels
    resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)  # shape: [B, H, W, 1]

    # Extract the canvas region we're about to overwrite
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    # Blend: new = mask * inpainted + (1 - mask) * canvas
    blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop

    # Paste the blended region back onto the canvas
    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended

    # Final crop to get back the original image area
    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]

    return output_image


class WanCropImproved:
    print("[WanCropImproved] Class definition loaded!")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # NEW: Multi-input control
                "num_inputs": ("INT", {"default": 1, "min": 1, "max": 5, "tooltip": "Number of image/mask pairs to process"}),
                
                # Required inputs (first pair always required)
                "image": ("IMAGE",),
                "mask": ("MASK",),

                # Resize algorithms
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bilinear"}),
                "upscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bicubic"}),

                # Pre-resize input image
                "preresize": ("BOOLEAN", {"default": False, "tooltip": "Resize the original image before processing."}),
                "preresize_mode": (["ensure minimum resolution", "ensure maximum resolution", "ensure minimum and maximum resolution"], {"default": "ensure minimum resolution"}),
                "preresize_min_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_min_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_width": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_height": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),

                # Mask manipulation
                "mask_fill_holes": ("BOOLEAN", {"default": True, "tooltip": "Mark as masked any areas fully enclosed by mask."}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Expand the mask by a certain amount of pixels before processing."}),
                "mask_invert": ("BOOLEAN", {"default": False,"tooltip": "Invert mask so that anything masked will be kept."}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1, "tooltip": "How many pixels to blend into the original image."}),
                "mask_hipass_filter": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01, "tooltip": "Ignore mask values lower than this value."}),

                # Extend image for outpainting
                "extend_for_outpainting": ("BOOLEAN", {"default": False, "tooltip": "Extend the image for outpainting."}),
                "extend_up_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_down_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_left_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_right_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),

                # Context
                "context_from_mask_extend_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01, "tooltip": "Grow the context area from the mask by a certain factor in every direction. For example, 1.5 grabs extra 50% up, down, left, and right as context."}),

                # Output
                "output_resize_to_target_size": ("BOOLEAN", {"default": True, "tooltip": "Force a specific resolution for sampling."}),
                "output_target_width": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_target_height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_padding": (["0", "8", "16", "32", "64", "128", "256", "512"], {"default": "0"}),
                "minimize_horizontal_padding": ("BOOLEAN", {"default": True, "tooltip": "When adjusting aspect ratio, minimize padding on left/right boundaries."}),
                "reuse_first_input_coordinates": ("BOOLEAN", {"default": True, "tooltip": "Apply crop/stitch coordinates from input 1 to all additional inputs."}),
                

           },
           "optional": {
                # NEW: Dynamic image/mask inputs (only shown based on num_inputs)
                "image_2": ("IMAGE",),
                "mask_2": ("MASK",),
                "image_3": ("IMAGE",),
                "mask_3": ("MASK",),
                "image_4": ("IMAGE",),
                "mask_4": ("MASK",),
                "image_5": ("IMAGE",),
                "mask_5": ("MASK",),
                
                # EXISTING: Optional inputs with lazy evaluation
                "optional_context_mask": (any_type, {"lazy": True}),
           }
        }

    FUNCTION = "inpaint_crop"
    CATEGORY = "inpaint"
    
    @classmethod
    def check_lazy_status(cls, **kwargs):
        """Handle lazy evaluation status check - accept all parameters including num_inputs"""
        return []
    DESCRIPTION = "Crops an image around a mask for inpainting, the optional context mask defines an extra area to keep for the context."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Deterministic cache key based on relevant widget parameters only
        import json
        tracked_keys = [
            'num_inputs',
            'downscale_algorithm', 'upscale_algorithm',
            'preresize', 'preresize_mode', 'preresize_min_width', 'preresize_min_height', 'preresize_max_width', 'preresize_max_height',
            'extend_for_outpainting', 'extend_up_factor', 'extend_down_factor', 'extend_left_factor', 'extend_right_factor',
            'mask_hipass_filter', 'mask_fill_holes', 'mask_expand_pixels', 'mask_invert', 'mask_blend_pixels',
            'context_from_mask_extend_factor',
            'output_resize_to_target_size', 'output_target_width', 'output_target_height', 'output_padding',
            'minimize_horizontal_padding', 'reuse_first_input_coordinates',
        ]
        filtered = {k: kwargs.get(k) for k in tracked_keys if k in kwargs}
        # Return a stable JSON string; ComfyUI will include upstream input cache keys separately
        return json.dumps(filtered, sort_keys=True)
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Always return True to ensure validation passes
        return True
    
    def check_lazy_status(self, image, mask, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, minimize_horizontal_padding, reuse_first_input_coordinates=True, num_inputs=1, image_2=None, mask_2=None, image_3=None, mask_3=None, image_4=None, mask_4=None, image_5=None, mask_5=None, optional_context_mask=None):
        needed = []
        
        # optional_context_mask is truly optional - we can provide a default
        # So we don't need to require it in the lazy evaluation
        # The node will handle the case where it's None by creating a default empty mask
        
        return needed

    # REMOVED: calculate_crop_coordinates method - replaced with per-frame processing
    # This method was removed because it calculated coordinates from only the first frame
    # and applied them to all frames, which is incorrect for video processing where
    # each frame may have different mask areas requiring different crop regions.
    # Now using per-frame processing like the original implementation.
    def calculate_crop_coordinates_removed(self):
        """Deprecated stub kept for backward compatibility in serialized graphs."""
        return {}

    # REMOVED: apply_crop_coordinates method - replaced with per-frame processing
    def apply_crop_coordinates_removed(self):
        """Deprecated stub kept for backward compatibility in serialized graphs."""
        return None

    # Remove the following # to turn on debug mode (extra outputs, print statements)
    #'''
    DEBUG_MODE = False
    RETURN_TYPES = ("IMAGE", "MASK", "STITCHER") * 5
    RETURN_NAMES = (
        "cropped_image_1", "cropped_mask_1", "stitcher_1",
        "cropped_image_2", "cropped_mask_2", "stitcher_2",
        "cropped_image_3", "cropped_mask_3", "stitcher_3",
        "cropped_image_4", "cropped_mask_4", "stitcher_4",
        "cropped_image_5", "cropped_mask_5", "stitcher_5",
    )

    def inpaint_crop(self, num_inputs, image, mask, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, minimize_horizontal_padding, reuse_first_input_coordinates, optional_context_mask=None, **kwargs):
        
        # OPTIMIZATION: Collect active image/mask pairs
        image_mask_pairs = [(image, mask)]  # Always include the first pair
        print(f"[WanCropImproved] First pair - image: {image.shape}, mask: {mask.shape if mask is not None else 'None'}")
        
        # Extract additional image/mask pairs from kwargs
        for i in range(2, num_inputs + 1):
            img = kwargs.get(f"image_{i}")
            msk = kwargs.get(f"mask_{i}")
            if img is not None and msk is not None:
                image_mask_pairs.append((img, msk))
                print(f"[WanCropImproved] Found image/mask pair {i} - image: {img.shape}, mask: {msk.shape}")
            else:
                print(f"[WanCropImproved] Warning: image_{i} or mask_{i} is None, skipping pair {i}")
        
        print(f"[WanCropImproved] Processing {len(image_mask_pairs)} image/mask pairs")
        print(f"[WanCropImproved] Optional context mask: {optional_context_mask.shape if optional_context_mask is not None else 'None'}")
        
        # Prepare first image/mask for coordinate calculation
        first_image, first_mask = image_mask_pairs[0]
        first_image = first_image.clone()
        first_mask = first_mask.clone() if first_mask is not None else None
        first_optional_context_mask = optional_context_mask.clone() if optional_context_mask is not None else None
        
        # Handle basic validations and preparations
        output_padding = int(output_padding)
        
        # Check that some parameters make sense
        if preresize and preresize_mode == "ensure minimum and maximum resolution":
            assert preresize_max_width >= preresize_min_width, "Preresize maximum width must be greater than or equal to minimum width"
            assert preresize_max_height >= preresize_min_height, "Preresize maximum height must be greater than or equal to minimum height"

        # Handle mask shape mismatches for first pair
        if first_mask is not None and (first_image.shape[0] == 1 or first_mask.shape[0] == 1 or first_mask.shape[0] == first_image.shape[0]):
            if first_mask.shape[1] != first_image.shape[1] or first_mask.shape[2] != first_image.shape[2]:
                if torch.count_nonzero(first_mask) == 0:
                    first_mask = torch.zeros((first_mask.shape[0], first_image.shape[1], first_image.shape[2]), device=first_image.device, dtype=first_image.dtype)
                else:
                    first_mask = resize_mask_to_image(first_mask, first_image.shape[2], first_image.shape[1])

        if first_optional_context_mask is not None and (first_image.shape[0] == 1 or first_optional_context_mask.shape[0] == 1 or first_optional_context_mask.shape[0] == first_image.shape[0]):
            if first_optional_context_mask.shape[1] != first_image.shape[1] or first_optional_context_mask.shape[2] != first_image.shape[2]:
                if torch.count_nonzero(first_optional_context_mask) == 0:
                    first_optional_context_mask = torch.zeros((first_optional_context_mask.shape[0], first_image.shape[1], first_image.shape[2]), device=first_image.device, dtype=first_image.dtype)
                else:
                    first_optional_context_mask = resize_mask_to_image(first_optional_context_mask, first_image.shape[2], first_image.shape[1])

        # If no mask is provided, create one with the shape of the image
        if first_mask is None:
            first_mask = torch.zeros_like(first_image[:, :, :, 0])
        
        if first_optional_context_mask is None:
            first_optional_context_mask = torch.zeros_like(first_image[:, :, :, 0])

        # PER-FRAME PROCESSING: Process each frame of each input individually (like original implementation)
        print("[WanCropImproved] Starting per-frame processing for all inputs...")
        
        results = []
        for i, (img, msk) in enumerate(image_mask_pairs):
            print(f"[WanCropImproved] Processing image/mask pair {i+1} - image: {img.shape}, mask: {msk.shape}")
            
            # CRITICAL FIX: Use original context mask for first pair, create fresh empty one for others
            # This prevents all inputs from using the same context mask and producing identical outputs
            if i == 0:
                context_mask_for_this_pair = optional_context_mask
                print(f"[WanCropImproved] Using original context mask for first pair")
            else:
                # Create empty context mask matching this image's dimensions
                context_mask_for_this_pair = torch.zeros_like(img[:, :, :, 0]) if img is not None else None
                print(f"[WanCropImproved] Created fresh empty context mask for pair {i+1}: {context_mask_for_this_pair.shape if context_mask_for_this_pair is not None else 'None'}")
            
            # Process each frame individually (like original implementation)
            batch_size = img.shape[0]
            print(f"[WanCropImproved] Processing {batch_size} frames individually for input {i+1}")
            
            # Initialize result collectors for this input
            input_stitcher = {
                'downscale_algorithm': downscale_algorithm,
                'upscale_algorithm': upscale_algorithm,
                'blend_pixels': mask_blend_pixels,
                'canvas_to_orig_x': [],
                'canvas_to_orig_y': [],
                'canvas_to_orig_w': [],
                'canvas_to_orig_h': [],
                'canvas_image': [],
                'cropped_to_canvas_x': [],
                'cropped_to_canvas_y': [],
                'cropped_to_canvas_w': [],
                'cropped_to_canvas_h': [],
                'cropped_mask_for_blend': [],
            }
            
            input_cropped_images = []
            input_cropped_masks = []
            
            # Process each frame individually
            for b in range(batch_size):
                one_image = img[b].unsqueeze(0)  # [1, H, W, C]
                one_mask = msk[b].unsqueeze(0)   # [1, H, W]
                one_context_mask = context_mask_for_this_pair[b].unsqueeze(0) if context_mask_for_this_pair is not None else None

                print(f"[WanCropImproved] Processing frame {b+1}/{batch_size} for input {i+1}")

                if reuse_first_input_coordinates and i > 0 and len(results) > 0:
                    # Use coordinates from input 1, same frame index b
                    ref_stitcher = results[0][0]
                    # Guard in case of mismatched lengths
                    ref_index = min(b, len(ref_stitcher['canvas_image']) - 1)
                    ref_canvas = ref_stitcher['canvas_image'][ref_index]
                    # ref_canvas is [1, H, W, C]
                    ref_h, ref_w = int(ref_canvas.shape[1]), int(ref_canvas.shape[2])
                    cto_x = int(ref_stitcher['canvas_to_orig_x'][ref_index])
                    cto_y = int(ref_stitcher['canvas_to_orig_y'][ref_index])
                    ctc_x = int(ref_stitcher['cropped_to_canvas_x'][ref_index])
                    ctc_y = int(ref_stitcher['cropped_to_canvas_y'][ref_index])
                    ctc_w = int(ref_stitcher['cropped_to_canvas_w'][ref_index])
                    ctc_h = int(ref_stitcher['cropped_to_canvas_h'][ref_index])

                    # Expand this frame to the same canvas layout as reference (but using this input's content)
                    expanded_image, expanded_mask = expand_image_to_canvas_like(ref_h, ref_w, cto_x, cto_y, one_image, one_mask)

                    # Crop using reference crop box
                    cropped_image = expanded_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
                    cropped_mask = expanded_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

                    # Resize to reference target size
                    ref_target_h = int(results[0][1].shape[1]) if results[0][1].dim() == 4 else ctc_h
                    ref_target_w = int(results[0][1].shape[2]) if results[0][1].dim() == 4 else ctc_w
                    if ref_target_w != ctc_w or ref_target_h != ctc_h:
                        cropped_image = rescale_i(cropped_image, ref_target_w, ref_target_h, upscale_algorithm)
                        cropped_mask = rescale_m_batch(cropped_mask, ref_target_w, ref_target_h, upscale_algorithm)

                    # Build blend mask from this input's cropped mask (match original behavior)
                    cropped_mask_blend = cropped_mask.clone()
                    if mask_blend_pixels > 0:
                        cropped_mask_blend = blur_m(cropped_mask_blend, mask_blend_pixels * 0.5)

                    # Append stitcher data: reuse coordinates from reference, but use this input's canvas and blend mask
                    input_stitcher['canvas_to_orig_x'].append(cto_x)
                    input_stitcher['canvas_to_orig_y'].append(cto_y)
                    input_stitcher['canvas_to_orig_w'].append(int(ref_stitcher['canvas_to_orig_w'][ref_index]))
                    input_stitcher['canvas_to_orig_h'].append(int(ref_stitcher['canvas_to_orig_h'][ref_index]))
                    input_stitcher['canvas_image'].append(expanded_image)
                    input_stitcher['cropped_to_canvas_x'].append(ctc_x)
                    input_stitcher['cropped_to_canvas_y'].append(ctc_y)
                    input_stitcher['cropped_to_canvas_w'].append(ctc_w)
                    input_stitcher['cropped_to_canvas_h'].append(ctc_h)
                    input_stitcher['cropped_mask_for_blend'].append(cropped_mask_blend)

                else:
                    # Compute coordinates from this frame's own mask (original behavior)
                    frame_outputs = self.inpaint_crop_single_image(
                        one_image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode,
                        preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height,
                        extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor,
                        mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels,
                        context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height,
                        output_padding, minimize_horizontal_padding, one_mask, one_context_mask
                    )
                    frame_stitcher, cropped_image, cropped_mask = frame_outputs[:3]

                    # Collect frame stitcher arrays
                    for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h',
                                'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w',
                                'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                        input_stitcher[key].append(frame_stitcher[key])

                # Collect cropped results (common path)
                frame_cropped_image = cropped_image.squeeze(0)
                frame_cropped_mask = cropped_mask.squeeze(0)
                input_cropped_images.append(frame_cropped_image)
                input_cropped_masks.append(frame_cropped_mask)
                print(f"  - Frame {b+1} result: cropped_image {frame_cropped_image.shape}, cropped_mask {frame_cropped_mask.shape}")
            
            # Combine frame results into batch tensors for this input
            combined_cropped_image = torch.stack(input_cropped_images, dim=0)
            combined_cropped_mask = torch.stack(input_cropped_masks, dim=0)
            
            print(f"[WanCropImproved] Input {i+1} combined results:")
            print(f"  - Combined cropped_image: {combined_cropped_image.shape}")
            print(f"  - Combined cropped_mask: {combined_cropped_mask.shape}")
            print(f"  - Stitcher arrays: {len(input_stitcher['canvas_image'])} frames")
            
            results.append((input_stitcher, combined_cropped_image, combined_cropped_mask))
        
        # Prepare outputs: interleaved per input -> image_i, mask_i, stitcher_i
        print(f"[WanCropImproved] ✅ Multi-crop processing complete ({len(results)} active crops)")
        all_outputs = []
        print(f"[WanCropImproved] Preparing outputs (interleaved per input):")
        for i in range(5):
            if i < len(results):
                stitcher_i, cropped_image, cropped_mask = results[i]
                all_outputs.extend([cropped_image, cropped_mask, stitcher_i])
                print(f"  - Output {i+1}: cropped_image {cropped_image.shape}, cropped_mask {cropped_mask.shape}, stitcher dict")
            else:
                if results:
                    first_cropped_image = results[0][1]
                    target_h, target_w = first_cropped_image.shape[1], first_cropped_image.shape[2]
                    batch_size = first_cropped_image.shape[0]
                else:
                    target_h, target_w = 512, 512
                    batch_size = 1
                empty_image = torch.zeros((batch_size, target_h, target_w, 3), dtype=torch.float32)
                empty_mask = torch.zeros((batch_size, target_h, target_w), dtype=torch.float32)
                all_outputs.extend([empty_image, empty_mask, {}])
                print(f"  - Output {i+1}: empty_image {empty_image.shape}, empty_mask {empty_mask.shape}, empty stitcher")

        print(f"[WanCropImproved] Total outputs returned: {len(all_outputs)} (5 x [image, mask, stitcher])")
        return tuple(all_outputs)

    def inpaint_crop_single_image(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, minimize_horizontal_padding, mask, optional_context_mask):
        # If no optional_context_mask is provided, create one with the shape of the image
        if optional_context_mask is None:
            optional_context_mask = torch.zeros_like(image[:, :, :, 0])
        
        if preresize:
            image, mask, optional_context_mask = preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height)
        if self.DEBUG_MODE:
            DEBUG_preresize_image = image.clone()
            DEBUG_preresize_mask = mask.clone()
       
        if mask_fill_holes:
           mask = fillholes_iterative_hipass_fill_m(mask)
        if self.DEBUG_MODE:
            DEBUG_fillholes_mask = mask.clone()

        if mask_expand_pixels > 0:
            mask = expand_m(mask, mask_expand_pixels)
        if self.DEBUG_MODE:
            DEBUG_expand_mask = mask.clone()

        if mask_invert:
            mask = invert_m(mask)
        if self.DEBUG_MODE:
            DEBUG_invert_mask = mask.clone()

        if mask_blend_pixels > 0:
            mask = expand_m(mask, mask_blend_pixels)
            mask = blur_m(mask, mask_blend_pixels*0.5)
        if self.DEBUG_MODE:
            DEBUG_blur_mask = mask.clone()

        if mask_hipass_filter >= 0.01:
            mask = hipassfilter_m(mask, mask_hipass_filter)
            optional_context_mask = hipassfilter_m(optional_context_mask, mask_hipass_filter)
        if self.DEBUG_MODE:
            DEBUG_hipassfilter_mask = mask.clone()

        if extend_for_outpainting:
            image, mask, optional_context_mask = extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor)
        if self.DEBUG_MODE:
            DEBUG_extend_image = image.clone()
            DEBUG_extend_mask = mask.clone()

        context, x, y, w, h = findcontextarea_m(mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        
        # Debug: Print mask boundaries
        print(f"[WanCropImproved] Mask boundaries found: x={x}, y={y}, w={w}, h={h}")
        print(f"[WanCropImproved] Mask aspect ratio: {w/h if h > 0 else 'undefined':.2f}")
        if self.DEBUG_MODE:
            DEBUG_context_from_mask = context.clone()
            DEBUG_context_from_mask_location = debug_context_location_in_image(image, x, y, w, h)

        if context_from_mask_extend_factor >= 1.01:
            context, x, y, w, h = growcontextarea_m(context, mask, x, y, w, h, context_from_mask_extend_factor)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_expand = context.clone()
            DEBUG_context_expand_location = debug_context_location_in_image(image, x, y, w, h)

        context, x, y, w, h = combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_with_context_mask = context.clone()
            DEBUG_context_with_context_mask_location = debug_context_location_in_image(image, x, y, w, h)

        if not output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, w, h, output_padding, downscale_algorithm, upscale_algorithm, minimize_horizontal_padding)
        else: # if output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, output_target_width, output_target_height, output_padding, downscale_algorithm, upscale_algorithm, minimize_horizontal_padding)
        if self.DEBUG_MODE:
            DEBUG_context_to_target = context.clone()
            DEBUG_context_to_target_location = debug_context_location_in_image(image, x, y, w, h)
            DEBUG_context_to_target_image = image.clone()
            DEBUG_context_to_target_mask = mask.clone()
            DEBUG_canvas_image = canvas_image.clone()
            DEBUG_orig_in_canvas_location = debug_context_location_in_image(canvas_image, cto_x, cto_y, cto_w, cto_h)
            DEBUG_cropped_in_canvas_location = debug_context_location_in_image(canvas_image, ctc_x, ctc_y, ctc_w, ctc_h)

        # For blending, grow the mask even further and make it blurrier.
        cropped_mask_blend = cropped_mask.clone()
        if mask_blend_pixels > 0:
           cropped_mask_blend = blur_m(cropped_mask_blend, mask_blend_pixels*0.5)
        if self.DEBUG_MODE:
            DEBUG_cropped_mask_blend = cropped_mask_blend.clone()

        stitcher = {
            'canvas_to_orig_x': cto_x,
            'canvas_to_orig_y': cto_y,
            'canvas_to_orig_w': cto_w,
            'canvas_to_orig_h': cto_h,
            'canvas_image': canvas_image,
            'cropped_to_canvas_x': ctc_x,
            'cropped_to_canvas_y': ctc_y,
            'cropped_to_canvas_w': ctc_w,
            'cropped_to_canvas_h': ctc_h,
            'cropped_mask_for_blend': cropped_mask_blend,
        }

        if not self.DEBUG_MODE:
            return stitcher, cropped_image, cropped_mask
        else:
            return stitcher, cropped_image, cropped_mask, DEBUG_preresize_image, DEBUG_preresize_mask, DEBUG_fillholes_mask, DEBUG_expand_mask, DEBUG_invert_mask, DEBUG_blur_mask, DEBUG_hipassfilter_mask, DEBUG_extend_image, DEBUG_extend_mask, DEBUG_context_from_mask, DEBUG_context_from_mask_location, DEBUG_context_expand, DEBUG_context_expand_location, DEBUG_context_with_context_mask, DEBUG_context_with_context_mask_location, DEBUG_context_to_target, DEBUG_context_to_target_location, DEBUG_context_to_target_image, DEBUG_context_to_target_mask, DEBUG_canvas_image, DEBUG_orig_in_canvas_location, DEBUG_cropped_in_canvas_location, DEBUG_cropped_mask_blend 


class WanStitchImproved:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch

    This node stitches the inpainted image without altering unmasked areas.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Put original_images first so native bypass forwards this input
                "original_images": ("IMAGE",),
                "inpainted_image": ("IMAGE",),
                "stitcher": ("STITCHER",),
            },
            "optional": {}
        }

    CATEGORY = "inpaint"
    DESCRIPTION = "Stitches an image cropped with Inpaint Crop back into the original image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "inpaint_stitch"


    def inpaint_stitch(self, original_images, inpainted_image, stitcher):
        inpainted_image = inpainted_image.clone()
        results = []

        # Robust bypass/fallback: if stitcher is missing/empty, just pass through the input image
        if not stitcher or not isinstance(stitcher, dict) or 'cropped_to_canvas_x' not in stitcher or len(stitcher.get('cropped_to_canvas_x', [])) == 0:
            return (inpainted_image.clone(),)

        batch_size = inpainted_image.shape[0]
        arrays_len = len(stitcher['cropped_to_canvas_x']) if stitcher else 0
        if arrays_len == 0:
            return (inpainted_image.clone(),)
        use_idx = 0 if arrays_len == 1 else None
        for b in range(batch_size):
            one_image = inpainted_image[b]
            one_stitcher = {}
            for key in ['downscale_algorithm', 'upscale_algorithm', 'blend_pixels']:
                one_stitcher[key] = stitcher[key]
            for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                if use_idx is not None:
                    idx = use_idx
                else:
                    idx = b if b < arrays_len else arrays_len - 1
                one_stitcher[key] = stitcher[key][idx]
            one_image = one_image.unsqueeze(0)
            one_image, = self.inpaint_stitch_single_image(one_stitcher, one_image)
            one_image = one_image.squeeze(0)
            one_image = one_image.clone()
            results.append(one_image)

        result_batch = torch.stack(results, dim=0)

        return (result_batch,)

    def inpaint_stitch_single_image(self, stitcher, inpainted_image):
        downscale_algorithm = stitcher['downscale_algorithm']
        upscale_algorithm = stitcher['upscale_algorithm']
        canvas_image = stitcher['canvas_image']

        ctc_x = stitcher['cropped_to_canvas_x']
        ctc_y = stitcher['cropped_to_canvas_y']
        ctc_w = stitcher['cropped_to_canvas_w']
        ctc_h = stitcher['cropped_to_canvas_h']

        cto_x = stitcher['canvas_to_orig_x']
        cto_y = stitcher['canvas_to_orig_y']
        cto_w = stitcher['canvas_to_orig_w']
        cto_h = stitcher['canvas_to_orig_h']

        mask = stitcher['cropped_mask_for_blend']  # shape: [1, H, W]

        output_image = stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm)

        return (output_image,) 