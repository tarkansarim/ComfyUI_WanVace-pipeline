"""
Utility functions for the WANTiledSampler node
Extracted from ComfyUI-WanVideoWrapper to make the node self-contained
"""

import torch
import numpy as np


def get_sigmas(scheduler, steps, shift, riflex_freq_index=0):
    """
    Returns the sigmas for the given scheduler.
    """
    # For now, we'll use a simple linear sigma schedule as fallback
    # This can be expanded later with proper scheduler implementations
    if "unipc" in scheduler or "euler" in scheduler or "dpm++" in scheduler:
        # Simple linear sigma schedule from 1.0 to 0.0
        sigmas_np = np.linspace(1.0, 0.0, steps + 1)
        sigmas = torch.from_numpy(sigmas_np).float()
    else:
        # Default fallback
        sigmas_np = np.linspace(1.0, 0.0, steps + 1)
        sigmas = torch.from_numpy(sigmas_np).float()
    
    return sigmas


def sigmas_to_timesteps(sigmas, steps):
    """
    Converts sigmas to timesteps.
    """
    timesteps = (sigmas * 1000).to(torch.int64)
    return timesteps


def apply_lora(patcher, device, offload_device, low_mem_load=False):
    """
    Stub for apply_lora function - returns the patcher unchanged
    In a full implementation, this would apply LoRA weights to the model
    """
    return patcher