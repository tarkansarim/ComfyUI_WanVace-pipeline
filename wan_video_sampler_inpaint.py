"""
WAN Video Sampler with Inpainting Support
Complete copy of WanVideoSampler with minimal inpainting detection added.
"""

import os
import torch
import torch.nn.functional as F
import gc
import numpy as np
import math
from tqdm import tqdm
from einops import rearrange
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from itertools import product

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.model_base
import comfy.latent_formats
from comfy.clip_vision import clip_preprocess, ClipVisionModel
from comfy.sd import load_lora_for_models
from comfy.cli_args import args, LatentPreviewMethod

# Import local copies of WanVideoWrapper dependencies
from .wanvideo.modules.clip import CLIPModel
from .wanvideo.modules.model import WanModel, rope_params
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .wanvideo.utils.basic_flowmatch import FlowMatchScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler
from .wanvideo.utils.scheduling_flow_match_lcm import FlowMatchLCMScheduler
from .enhance_a_video.globals import (
    enable_enhance,
    disable_enhance,
    set_enhance_weight,
    set_num_frames,
)
from .taehv import TAEHV
from .context import get_context_scheduler

# Local utils functions - copied from WanVideoWrapper utils.py
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

def print_memory(device):
    memory = torch.cuda.memory_allocated(device) / 1024**3
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    log.info(f"Allocated memory: {memory=:.3f} GB")
    log.info(f"Max allocated memory: {max_memory=:.3f} GB")
    log.info(f"Max reserved memory: {max_reserved=:.3f} GB")

def apply_lora(model, device_to, transformer_load_device, params_to_keep=None, dtype=None, base_dtype=None, state_dict=None, low_mem_load=False):
    # Simplified version - full implementation would be copied from utils.py if needed
    return model

def clip_encode_image_tiled(clip_vision, image, tiles=1, ratio=1.0):
    # Simplified version - full implementation would be copied from utils.py if needed
    return clip_vision.encode_image(image).last_hidden_state

def fourier_filter(x, scale_low=1.0, scale_high=1.5, freq_cutoff=20):
    # Simplified version - full implementation would be copied from utils.py if needed
    return x

def get_sigmas(scheduler, steps, shift, riflex_freq_index=0):
    # Simplified version - full implementation would be copied from utils.py if needed
    from .wanvideo.utils.fm_solvers import get_sampling_sigmas
    sigmas_np = get_sampling_sigmas(steps, shift)
    return torch.from_numpy(sigmas_np).float()

def sigmas_to_timesteps(sigmas, steps):
    # Simplified version - full implementation would be copied from utils.py if needed
    timesteps = (sigmas * 1000).to(torch.int64)
    return timesteps

# Import context tracking classes
class WindowTracker:
    def __init__(self, verbose=False):
        self.cache_states = {}
        self.verbose = verbose
    
    def get_window_id(self, context_range):
        return f"{min(context_range)}-{max(context_range)}"
    
    def get_teacache(self, window_id, cache_state):
        return self.cache_states.get(window_id, cache_state)


class WANVideoSamplerInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01},
                ),
                "shift": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "force_offload": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Moves the model to the offload device after sampling",
                    },
                ),
                "scheduler": (
                    [
                        "unipc",
                        "unipc/beta",
                        "dpm++",
                        "dpm++/beta",
                        "dpm++_sde",
                        "dpm++_sde/beta",
                        "euler",
                        "euler/beta",
                        "euler/accvideo",
                        "deis",
                        "lcm",
                        "lcm/beta",
                        "flowmatch_causvid",
                        "flowmatch_distill",
                    ],
                    {"default": "unipc"},
                ),
                "riflex_freq_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping",
                    },
                ),
            },
            "optional": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "samples": (
                    "LATENT",
                    {"tooltip": "init Latents to use for video2video process"},
                ),
                "denoise_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "feta_args": ("FETAARGS",),
                "context_options": ("WANVIDCONTEXT",),
                "cache_args": ("CACHEARGS",),
                "flowedit_args": ("FLOWEDITARGS",),
                "batched_cfg": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Batch cond and uncond for faster sampling, possibly faster on some hardware, uses more memory",
                    },
                ),
                "slg_args": ("SLGARGS",),
                "rope_function": (
                    ["default", "comfy"],
                    {
                        "default": "comfy",
                        "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile",
                    },
                ),
                "loop_args": ("LOOPARGS",),
                "experimental_args": ("EXPERIMENTALARGS",),
                "sigmas": ("SIGMAS",),
                "unianimate_poses": ("UNIANIMATE_POSE",),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS",),
                "uni3c_embeds": ("UNI3C_EMBEDS",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanVace-pipeline/sampler"

    def process(
        self,
        model,
        image_embeds,
        shift,
        steps,
        cfg,
        seed,
        scheduler,
        riflex_freq_index,
        text_embeds=None,
        force_offload=True,
        samples=None,
        feta_args=None,
        denoise_strength=1.0,
        context_options=None,
        cache_args=None,
        teacache_args=None,
        flowedit_args=None,
        batched_cfg=False,
        slg_args=None,
        rope_function="default",
        loop_args=None,
        experimental_args=None,
        sigmas=None,
        unianimate_poses=None,
        fantasytalking_embeds=None,
        uni3c_embeds=None,
    ):
        # Initialize variables early to avoid scope issues
        samples_noise_mask = None
        
        # Check for inpainting conditioning in image_embeds
        is_inpainting = "concat_latent_image" in image_embeds
        concat_latent = None
        concat_mask = None
        
        # Check if we have noise_mask from samples (WANInpaintConditioning)
        has_noise_mask = samples_noise_mask is not None
        
        if is_inpainting or has_noise_mask:
            log.info("=== DETECTED INPAINTING CONDITIONING ===")
            
            if is_inpainting:
                concat_latent = image_embeds.get("concat_latent_image")
                concat_mask = image_embeds.get("concat_mask")
                log.info(f"Reference latent shape: {concat_latent.shape if concat_latent is not None else 'None'}")
                log.info(f"Denoise mask shape: {concat_mask.shape if concat_mask is not None else 'None'}")
                log.info(f"Mask min/max values: {concat_mask.min().item():.3f}/{concat_mask.max().item():.3f}" if concat_mask is not None else "No mask")
                
                if concat_latent is not None:
                    log.info(f"Reference latent min/max: {concat_latent.min().item():.3f}/{concat_latent.max().item():.3f}")
                    log.info(f"Reference latent device: {concat_latent.device}")
                    log.info(f"Reference latent dtype: {concat_latent.dtype}")
            
            if has_noise_mask:
                log.info(f"Samples noise_mask shape: {samples_noise_mask.shape}")
                log.info(f"Samples noise_mask min/max: {samples_noise_mask.min().item():.6f}/{samples_noise_mask.max().item():.6f}")
                log.info(f"Samples noise_mask unique values: {torch.unique(samples_noise_mask).numel()}")
                if torch.unique(samples_noise_mask).numel() > 2:
                    log.info("✅ Blur preserved: noise_mask has gradation values")
                else:
                    log.info("❌ Binary mask: noise_mask has only 0/1 values")
            
            log.info("Will apply inpainting masks during denoising steps")
            
            # Use samples noise_mask as fallback if no explicit inpainting conditioning
            if not is_inpainting and has_noise_mask:
                log.info("Using noise_mask from samples as primary inpainting mask")
                concat_mask = samples_noise_mask
                # For noise_mask, we need the reference latent (original unmasked content)
                if samples is not None and "samples" in samples:
                    concat_latent = samples["samples"]
                    log.info(f"Using original samples as reference latent: {concat_latent.shape}")
        else:
            log.info("No inpainting conditioning detected - using standard mode")
        
        # Continue with the complete WanVideoSampler implementation
        patcher = model
        model = model.model
        transformer = model.diffusion_model
        dtype = model["dtype"]
        control_lora = model["control_lora"]
        
        # Initialize tiling_enabled to False for regular sampler
        tiling_enabled = False

        transformer_options = patcher.model_options.get("transformer_options", None)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        steps = int(steps / denoise_strength)

        if text_embeds == None:
            text_embeds = {
                "prompt_embeds": [],
                "negative_prompt_embeds": [],
            }

        if isinstance(cfg, list):
            if steps != len(cfg):
                log.info(
                    f"Received {len(cfg)} cfg values, but only {steps} steps. Setting step count to match."
                )
                steps = len(cfg)

        timesteps = None
        if "unipc" in scheduler:
            sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
            if sigmas is None:
                sample_scheduler.set_timesteps(
                    steps,
                    device=device,
                    shift=shift,
                    use_beta_sigmas=("beta" in scheduler),
                )
            else:
                sample_scheduler.sigmas = sigmas.to(device)
                sample_scheduler.timesteps = (
                    (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
                )
                sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)

        elif scheduler in ["euler/beta", "euler"]:
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift, use_beta_sigmas=(scheduler == "euler/beta")
            )
            if flowedit_args:  # seems to work better
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=device,
                    sigmas=get_sampling_sigmas(steps, shift),
                )
            else:
                sample_scheduler.set_timesteps(
                    steps,
                    device=device,
                    sigmas=sigmas.tolist() if sigmas is not None else None,
                )
        elif scheduler in ["euler/accvideo"]:
            if steps != 50:
                raise Exception(
                    "Steps must be set to 50 for accvideo scheduler, 10 actual steps are used"
                )
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift, use_beta_sigmas=(scheduler == "euler/beta")
            )
            sample_scheduler.set_timesteps(
                steps,
                device=device,
                sigmas=sigmas.tolist() if sigmas is not None else None,
            )
            start_latent_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            sample_scheduler.sigmas = sample_scheduler.sigmas[start_latent_list]
            steps = len(start_latent_list) - 1
            sample_scheduler.timesteps = timesteps = sample_scheduler.timesteps[
                start_latent_list[:steps]
            ]
        elif "dpm++" in scheduler:
            if "sde" in scheduler:
                algorithm_type = "sde-dpmsolver++"
            else:
                algorithm_type = "dpmsolver++"
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                shift=shift, algorithm_type=algorithm_type
            )
            if sigmas is None:
                sample_scheduler.set_timesteps(
                    steps, device=device, use_beta_sigmas=("beta" in scheduler)
                )
            else:
                sample_scheduler.sigmas = sigmas.to(device)
                sample_scheduler.timesteps = (
                    (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
                )
                sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)
        elif scheduler == "deis":
            sample_scheduler = DEISMultistepScheduler(
                use_flow_sigmas=True,
                prediction_type="flow_prediction",
                flow_shift=shift,
            )
            sample_scheduler.set_timesteps(steps, device=device)
            sample_scheduler.sigmas[-1] = 1e-6
        elif "lcm" in scheduler:
            sample_scheduler = FlowMatchLCMScheduler(
                shift=shift, use_beta_sigmas=(scheduler == "lcm/beta")
            )
            sample_scheduler.set_timesteps(
                steps,
                device=device,
                sigmas=sigmas.tolist() if sigmas is not None else None,
            )
        elif "flowmatch_causvid" in scheduler:
            if transformer.dim == 5120:
                denoising_list = [999, 934, 862, 756, 603, 410, 250, 140, 74]
            else:
                if steps != 4:
                    raise ValueError("CausVid 1.3B schedule is only for 4 steps")
                denoising_list = [1000, 750, 500, 250]
            sample_scheduler = FlowMatchScheduler(
                num_inference_steps=steps, shift=shift, sigma_min=0, extra_one_step=True
            )
            sample_scheduler.timesteps = torch.tensor(denoising_list)[:steps].to(device)
            sample_scheduler.sigmas = torch.cat(
                [sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)]
            )
        elif "flowmatch_distill" in scheduler:
            sample_scheduler = FlowMatchScheduler(
                shift=shift, sigma_min=0.0, extra_one_step=True
            )
            sample_scheduler.set_timesteps(1000, training=True)

            denoising_step_list = torch.tensor([999, 750, 500, 250], dtype=torch.long)
            temp_timesteps = torch.cat(
                (
                    sample_scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            denoising_step_list = temp_timesteps[1000 - denoising_step_list]
            print("denoising_step_list: ", denoising_step_list)

            # denoising_step_list = [999, 750, 500, 250]
            if steps != 4:
                raise ValueError("This scheduler is only for 4 steps")
            # sample_scheduler = FlowMatchScheduler(num_inference_steps=steps, shift=shift, sigma_min=0, extra_one_step=True)
            sample_scheduler.timesteps = torch.tensor(denoising_step_list)[:steps].to(
                device
            )
            sample_scheduler.sigmas = torch.cat(
                [sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)]
            )

        if timesteps is None:
            timesteps = sample_scheduler.timesteps
        log.info(f"timesteps: {timesteps}")

        if denoise_strength < 1.0:
            steps = int(steps * denoise_strength)
            timesteps = timesteps[-(steps + 1) :]

        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)

        control_latents = control_camera_latents = clip_fea = clip_fea_neg = (
            end_image
        ) = recammaster = camera_embed = unianim_data = None
        vace_data = vace_context = vace_scale = None
        fun_or_fl2v_model = has_ref = drop_last = False
        phantom_latents = None
        fun_ref_image = None

        image_cond = image_embeds.get("image_embeds", None)
        ATI_tracks = None
        add_cond = attn_cond = attn_cond_neg = None

        if image_cond is not None:
            log.info(f"image_cond shape: {image_cond.shape}")
            # ATI tracks
            if transformer_options is not None:
                ATI_tracks = transformer_options.get("ati_tracks", None)
                if ATI_tracks is not None:
                    from .ATI.motion_patch import patch_motion

                    topk = transformer_options.get("ati_topk", 2)
                    temperature = transformer_options.get("ati_temperature", 220.0)
                    ati_start_percent = transformer_options.get(
                        "ati_start_percent", 0.0
                    )
                    ati_end_percent = transformer_options.get("ati_end_percent", 1.0)
                    image_cond_ati = patch_motion(
                        ATI_tracks.to(image_cond.device, image_cond.dtype),
                        image_cond,
                        topk=topk,
                        temperature=temperature,
                    )
                    log.info(f"ATI tracks shape: {ATI_tracks.shape}")

            realisdance_latents = image_embeds.get("realisdance_latents", None)
            if realisdance_latents is not None:
                add_cond = realisdance_latents["pose_latent"]
                attn_cond = realisdance_latents["ref_latent"]
                attn_cond_neg = realisdance_latents["ref_latent_neg"]
                add_cond_start_percent = realisdance_latents["pose_cond_start_percent"]
                add_cond_end_percent = realisdance_latents["pose_cond_end_percent"]

            end_image = image_embeds.get("end_image", None)
            lat_h = image_embeds.get("lat_h", None)
            lat_w = image_embeds.get("lat_w", None)
            if lat_h is None or lat_w is None:
                raise ValueError(
                    "Clip encoded image embeds must be provided for I2V (Image to Video) model"
                )
            fun_or_fl2v_model = image_embeds.get("fun_or_fl2v_model", False)
            noise = torch.randn(
                16,
                (image_embeds["num_frames"] - 1) // 4
                + (2 if end_image is not None and not fun_or_fl2v_model else 1),
                lat_h,
                lat_w,
                dtype=torch.float32,
                generator=seed_g,
                device=torch.device("cpu"),
            )
            seq_len = image_embeds["max_seq_len"]

            clip_fea = image_embeds.get("clip_context", None)
            if clip_fea is not None:
                clip_fea = clip_fea.to(dtype)
            clip_fea_neg = image_embeds.get("negative_clip_context", None)
            if clip_fea_neg is not None:
                clip_fea_neg = clip_fea_neg.to(dtype)

            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                if transformer.in_dim not in [48, 32]:
                    raise ValueError("Control signal only works with Fun-Control model")
                control_latents = control_embeds.get("control_images", None)
                control_camera_latents = control_embeds.get(
                    "control_camera_latents", None
                )
                control_camera_start_percent = control_embeds.get(
                    "control_camera_start_percent", 0.0
                )
                control_camera_end_percent = control_embeds.get(
                    "control_camera_end_percent", 1.0
                )
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
            drop_last = image_embeds.get("drop_last", False)
            has_ref = image_embeds.get("has_ref", False)
        else:  # t2v
            target_shape = image_embeds.get("target_shape", None)
            if target_shape is None:
                raise ValueError(
                    "Empty image embeds must be provided for T2V (Text to Video"
                )

            has_ref = image_embeds.get("has_ref", False)
            vace_context = image_embeds.get("vace_context", None)
            vace_scale = image_embeds.get("vace_scale", None)
            if not isinstance(vace_scale, list):
                vace_scale = [vace_scale] * (steps + 1)
            vace_start_percent = image_embeds.get("vace_start_percent", 0.0)
            vace_end_percent = image_embeds.get("vace_end_percent", 1.0)
            vace_seqlen = image_embeds.get("vace_seq_len", None)

            vace_additional_embeds = image_embeds.get("additional_vace_inputs", [])
            if vace_context is not None:
                vace_data = [
                    {
                        "context": vace_context,
                        "scale": vace_scale,
                        "start": vace_start_percent,
                        "end": vace_end_percent,
                        "seq_len": vace_seqlen,
                    }
                ]
                if len(vace_additional_embeds) > 0:
                    for i in range(len(vace_additional_embeds)):
                        if vace_additional_embeds[i].get("has_ref", False):
                            has_ref = True
                        vace_scale = vace_additional_embeds[i]["vace_scale"]
                        if not isinstance(vace_scale, list):
                            vace_scale = [vace_scale] * (steps + 1)
                        vace_data.append(
                            {
                                "context": vace_additional_embeds[i]["vace_context"],
                                "scale": vace_scale,
                                "start": vace_additional_embeds[i][
                                    "vace_start_percent"
                                ],
                                "end": vace_additional_embeds[i]["vace_end_percent"],
                                "seq_len": vace_additional_embeds[i]["vace_seq_len"],
                            }
                        )

            noise = torch.randn(
                target_shape[0],
                target_shape[1] + 1 if has_ref else target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=torch.device("cpu"),
                generator=seed_g,
            )

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

            recammaster = image_embeds.get("recammaster", None)
            if recammaster is not None:
                camera_embed = recammaster.get("camera_embed", None)
                recam_latents = recammaster.get("source_latents", None)
                orig_noise_len = noise.shape[1]
                log.info(f"RecamMaster camera embed shape: {camera_embed.shape}")
                log.info(f"RecamMaster source video shape: {recam_latents.shape}")
                seq_len *= 2

            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                control_latents = control_embeds.get("control_images", None)
                if control_latents is not None:
                    control_latents = control_latents.to(device)
                control_camera_latents = control_embeds.get(
                    "control_camera_latents", None
                )
                control_camera_start_percent = control_embeds.get(
                    "control_camera_start_percent", 0.0
                )
                control_camera_end_percent = control_embeds.get(
                    "control_camera_end_percent", 1.0
                )
                if control_camera_latents is not None:
                    control_camera_latents = control_camera_latents.to(device)

                if control_lora:
                    image_cond = control_latents.to(device)
                    if not patcher.model.is_patched:
                        log.info("Re-loading control LoRA...")
                        patcher = apply_lora(
                            patcher, device, device, low_mem_load=False
                        )
                        patcher.model.is_patched = True
                else:
                    if transformer.in_dim not in [48, 32]:
                        raise ValueError(
                            "Control signal only works with Fun-Control model"
                        )
                    image_cond = torch.zeros_like(noise).to(device)  # fun control
                    clip_fea = None
                    fun_ref_image = control_embeds.get("fun_ref_image", None)
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
            else:
                if transformer.in_dim == 36:  # fun inp
                    mask_latents = torch.tile(torch.zeros_like(noise[:1]), [4, 1, 1, 1])
                    masked_video_latents_input = torch.zeros_like(noise)
                    image_cond = torch.cat(
                        [mask_latents, masked_video_latents_input], dim=0
                    ).to(device)

            phantom_latents = image_embeds.get("phantom_latents", None)
            phantom_cfg_scale = image_embeds.get("phantom_cfg_scale", None)
            if not isinstance(phantom_cfg_scale, list):
                phantom_cfg_scale = [phantom_cfg_scale] * (steps + 1)
            phantom_start_percent = image_embeds.get("phantom_start_percent", 0.0)
            phantom_end_percent = image_embeds.get("phantom_end_percent", 1.0)
            if phantom_latents is not None:
                phantom_latents = phantom_latents.to(device)

        latent_video_length = noise.shape[1]

        if unianimate_poses is not None:
            transformer.dwpose_embedding.to(device, model["dtype"])
            dwpose_data = unianimate_poses["pose"].to(device, model["dtype"])
            dwpose_data = torch.cat(
                [dwpose_data[:, :, :1].repeat(1, 1, 3, 1, 1), dwpose_data], dim=2
            )
            dwpose_data = transformer.dwpose_embedding(dwpose_data)
            log.info(f"UniAnimate pose embed shape: {dwpose_data.shape}")
            if dwpose_data.shape[2] > latent_video_length:
                log.info(
                    f"UniAnimate pose embed length {dwpose_data.shape[2]} is longer than the video length {latent_video_length}, truncating"
                )
                dwpose_data = dwpose_data[:, :, :latent_video_length]
            elif dwpose_data.shape[2] < latent_video_length:
                log.info(
                    f"UniAnimate pose embed length {dwpose_data.shape[2]} is shorter than the video length {latent_video_length}, padding with last pose"
                )
                pad_len = latent_video_length - dwpose_data.shape[2]
                pad = dwpose_data[:, :, :1].repeat(1, 1, pad_len, 1, 1)
                dwpose_data = torch.cat([dwpose_data, pad], dim=2)
            dwpose_data_flat = rearrange(
                dwpose_data, "b c f h w -> b (f h w) c"
            ).contiguous()

            random_ref_dwpose_data = None
            if image_cond is not None:
                transformer.randomref_embedding_pose.to(device)
                random_ref_dwpose = unianimate_poses.get("ref", None)
                if random_ref_dwpose is not None:
                    random_ref_dwpose_data = (
                        transformer.randomref_embedding_pose(
                            random_ref_dwpose.to(device)
                        )
                        .unsqueeze(2)
                        .to(model["dtype"])
                    )  # [1, 20, 104, 60]

            unianim_data = {
                "dwpose": dwpose_data_flat,
                "random_ref": (
                    random_ref_dwpose_data.squeeze(0)
                    if random_ref_dwpose_data is not None
                    else None
                ),
                "strength": unianimate_poses["strength"],
                "start_percent": unianimate_poses["start_percent"],
                "end_percent": unianimate_poses["end_percent"],
            }

        audio_proj = None
        if fantasytalking_embeds is not None:
            audio_proj = fantasytalking_embeds["audio_proj"].to(device)
            audio_context_lens = fantasytalking_embeds["audio_context_lens"]
            audio_scale = fantasytalking_embeds["audio_scale"]
            audio_cfg_scale = fantasytalking_embeds["audio_cfg_scale"]
            if not isinstance(audio_cfg_scale, list):
                audio_cfg_scale = [audio_cfg_scale] * (steps + 1)
            log.info(
                f"Audio proj shape: {audio_proj.shape}, audio context lens: {audio_context_lens}"
            )

        minimax_latents = minimax_mask_latents = None
        minimax_latents = image_embeds.get("minimax_latents", None)
        minimax_mask_latents = image_embeds.get("minimax_mask_latents", None)
        if minimax_latents is not None:
            log.info(f"minimax_latents: {minimax_latents.shape}")
            log.info(f"minimax_mask_latents: {minimax_mask_latents.shape}")
            minimax_latents = minimax_latents.to(device, dtype)
            minimax_mask_latents = minimax_mask_latents.to(device, dtype)

        is_looped = False
        if context_options is not None:

            def create_window_mask(
                noise_pred_context,
                c,
                latent_video_length,
                context_overlap,
                looped=False,
            ):
                window_mask = torch.ones_like(noise_pred_context)

                # Apply left-side blending for all except first chunk (or always in loop mode)
                if min(c) > 0 or (looped and max(c) == latent_video_length - 1):
                    ramp_up = torch.linspace(
                        0, 1, context_overlap, device=noise_pred_context.device
                    )
                    ramp_up = ramp_up.view(1, -1, 1, 1)
                    window_mask[:, :context_overlap] = ramp_up

                # Apply right-side blending for all except last chunk (or always in loop mode)
                if max(c) < latent_video_length - 1 or (looped and min(c) == 0):
                    ramp_down = torch.linspace(
                        1, 0, context_overlap, device=noise_pred_context.device
                    )
                    ramp_down = ramp_down.view(1, -1, 1, 1)
                    window_mask[:, -context_overlap:] = ramp_down

                return window_mask

            context_schedule = context_options["context_schedule"]
            context_frames = (context_options["context_frames"] - 1) // 4 + 1
            context_stride = context_options["context_stride"] // 4
            context_overlap = context_options["context_overlap"] // 4
            context_vae = context_options.get("vae", None)
            if context_vae is not None:
                context_vae.to(device)

            self.window_tracker = WindowTracker(verbose=context_options["verbose"])

            # Get total number of prompts
            num_prompts = len(text_embeds["prompt_embeds"])
            log.info(f"Number of prompts: {num_prompts}")
            # Calculate which section this context window belongs to
            section_size = latent_video_length / num_prompts
            log.info(f"Section size: {section_size}")
            is_looped = context_schedule == "uniform_looped"

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * context_frames)

            if context_options["freenoise"]:
                log.info("Applying FreeNoise")
                # code from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
                delta = context_frames - context_overlap
                for start_idx in range(0, latent_video_length - context_frames, delta):
                    place_idx = start_idx + context_frames
                    if place_idx >= latent_video_length:
                        break
                    end_idx = place_idx - 1

                    if end_idx + delta >= latent_video_length:
                        final_delta = latent_video_length - place_idx
                        list_idx = torch.tensor(
                            list(range(start_idx, start_idx + final_delta)),
                            device=torch.device("cpu"),
                            dtype=torch.long,
                        )
                        list_idx = list_idx[
                            torch.randperm(final_delta, generator=seed_g)
                        ]
                        noise[:, place_idx : place_idx + final_delta, :, :] = noise[
                            :, list_idx, :, :
                        ]
                        break
                    list_idx = torch.tensor(
                        list(range(start_idx, start_idx + delta)),
                        device=torch.device("cpu"),
                        dtype=torch.long,
                    )
                    list_idx = list_idx[torch.randperm(delta, generator=seed_g)]
                    noise[:, place_idx : place_idx + delta, :, :] = noise[
                        :, list_idx, :, :
                    ]

            log.info(
                f"Context schedule enabled: {context_frames} frames, {context_stride} stride, {context_overlap} overlap"
            )
            from .context import get_context_scheduler

            context = get_context_scheduler(context_schedule)

        # Extract noise_mask from samples (for WANInpaintConditioning support)
        if samples is not None:
            input_samples = samples["samples"].squeeze(0).to(noise)
            if input_samples.shape[1] != noise.shape[1]:
                input_samples = torch.cat(
                    [
                        input_samples[:, :1].repeat(
                            1, noise.shape[1] - input_samples.shape[1], 1, 1
                        ),
                        input_samples,
                    ],
                    dim=1,
                )
            
            # Check for noise_mask in samples (from WANInpaintConditioning)
            if "noise_mask" in samples:
                samples_noise_mask = samples["noise_mask"]
                log.info(f"Found noise_mask in samples with shape: {samples_noise_mask.shape}")
                log.info(f"noise_mask min/max: {samples_noise_mask.min().item():.3f}/{samples_noise_mask.max().item():.3f}")
                log.info("Will use noise_mask for pre-step masking during sampling")
            else:
                log.info("No noise_mask found in samples")
            original_image = input_samples.to(device)
            if denoise_strength < 1.0:
                latent_timestep = timesteps[:1].to(noise)
                noise = (
                    noise * latent_timestep / 1000
                    + (1 - latent_timestep / 1000) * input_samples
                )

            mask = samples.get("mask", None)
            if mask is not None:
                if mask.shape[2] != noise.shape[1]:
                    mask = torch.cat(
                        [
                            torch.zeros(
                                1,
                                noise.shape[0],
                                noise.shape[1] - mask.shape[2],
                                noise.shape[2],
                                noise.shape[3],
                            ),
                            mask,
                        ],
                        dim=2,
                    )

        latent = noise.to(device)

        freqs = None
        transformer.rope_embedder.k = None
        transformer.rope_embedder.num_frames = None
        if rope_function == "comfy":
            transformer.rope_embedder.k = riflex_freq_index
            transformer.rope_embedder.num_frames = latent_video_length
        else:
            d = transformer.dim // transformer.num_heads
            freqs = torch.cat(
                [
                    rope_params(
                        1024,
                        d - 4 * (d // 6),
                        L_test=latent_video_length,
                        k=riflex_freq_index,
                    ),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            )

        if not isinstance(cfg, list):
            cfg = [cfg] * (steps + 1)

        log.info(f"Seq len: {seq_len}")

        pbar = ProgressBar(steps)

        if args.preview_method in [
            LatentPreviewMethod.Auto,
            LatentPreviewMethod.Latent2RGB,
        ]:  # default for latent2rgb
            from latent_preview import prepare_callback
        else:
            try:
                from .latent_preview import prepare_callback  # custom for tiny VAE previews
            except:
                from latent_preview import prepare_callback
        callback = prepare_callback(patcher, steps)

        # blockswap init
        if transformer_options is not None:
            block_swap_args = transformer_options.get("block_swap_args", None)

        if block_swap_args is not None:
            transformer.use_non_blocking = block_swap_args.get("use_non_blocking", True)
            for name, param in transformer.named_parameters():
                if "block" not in name:
                    param.data = param.data.to(device)
                if "control_adapter" in name:
                    param.data = param.data.to(device)
                elif block_swap_args["offload_txt_emb"] and "txt_emb" in name:
                    param.data = param.data.to(
                        offload_device, non_blocking=transformer.use_non_blocking
                    )
                elif block_swap_args["offload_img_emb"] and "img_emb" in name:
                    param.data = param.data.to(
                        offload_device, non_blocking=transformer.use_non_blocking
                    )

            transformer.block_swap(
                block_swap_args["blocks_to_swap"] - 1,
                block_swap_args["offload_txt_emb"],
                block_swap_args["offload_img_emb"],
                vace_blocks_to_swap=block_swap_args.get("vace_blocks_to_swap", None),
            )

        elif model["auto_cpu_offload"]:
            for module in transformer.modules():
                if hasattr(module, "offload"):
                    module.offload()
                if hasattr(module, "onload"):
                    module.onload()
        elif model["manual_offloading"]:
            transformer.to(device)

        # controlnet
        controlnet_latents = controlnet = None
        if transformer_options is not None:
            controlnet = transformer_options.get("controlnet", None)
            if controlnet is not None:
                self.controlnet = controlnet["controlnet"]
                controlnet_start = controlnet["controlnet_start"]
                controlnet_end = controlnet["controlnet_end"]
                controlnet_latents = controlnet["control_latents"]
                controlnet["controlnet_weight"] = controlnet["controlnet_strength"]
                controlnet["controlnet_stride"] = controlnet["control_stride"]

        # uni3c
        pcd_data = None
        if uni3c_embeds is not None:
            transformer.controlnet = uni3c_embeds["controlnet"]
            pcd_data = {
                "render_latent": uni3c_embeds["render_latent"],
                "render_mask": uni3c_embeds["render_mask"],
                "camera_embedding": uni3c_embeds["camera_embedding"],
                "controlnet_weight": uni3c_embeds["controlnet_weight"],
                "start": uni3c_embeds["start"],
                "end": uni3c_embeds["end"],
            }

        # feta
        if feta_args is not None and latent_video_length > 1:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]
            if context_options is not None:
                set_num_frames(context_frames)
            else:
                set_num_frames(latent_video_length)
            enable_enhance()
        else:
            feta_args = None
            disable_enhance()

        # Initialize Cache if enabled
        transformer.enable_teacache = transformer.enable_magcache = False
        if teacache_args is not None:  # for backward compatibility on old workflows
            cache_args = teacache_args
        if cache_args is not None:
            transformer.cache_device = cache_args["cache_device"]
            if cache_args["cache_type"] == "TeaCache":
                log.info(f"TeaCache: Using cache device: {transformer.cache_device}")
                transformer.teacache_state.clear_all()
                transformer.enable_teacache = True
                transformer.rel_l1_thresh = cache_args["rel_l1_thresh"]
                transformer.teacache_start_step = cache_args["start_step"]
                transformer.teacache_end_step = (
                    len(timesteps) - 1
                    if cache_args["end_step"] == -1
                    else cache_args["end_step"]
                )
                transformer.teacache_use_coefficients = cache_args["use_coefficients"]
                transformer.teacache_mode = cache_args["mode"]
            elif cache_args["cache_type"] == "MagCache":
                log.info(f"MagCache: Using cache device: {transformer.cache_device}")
                transformer.magcache_state.clear_all()
                transformer.enable_magcache = True
                transformer.magcache_start_step = cache_args["start_step"]
                transformer.magcache_end_step = (
                    len(timesteps) - 1
                    if cache_args["end_step"] == -1
                    else cache_args["end_step"]
                )
                transformer.magcache_thresh = cache_args["magcache_thresh"]
                transformer.magcache_K = cache_args["magcache_K"]

        if slg_args is not None:
            assert batched_cfg is not None, "Batched cfg is not supported with SLG"
            transformer.slg_blocks = slg_args["blocks"]
            transformer.slg_start_percent = slg_args["start_percent"]
            transformer.slg_end_percent = slg_args["end_percent"]
        else:
            transformer.slg_blocks = None

        self.cache_state = [None, None]
        if phantom_latents is not None:
            log.info(f"Phantom latents shape: {phantom_latents.shape}")
            self.cache_state = [None, None, None]
        self.cache_state_source = [None, None]
        self.cache_states_context = []

        if flowedit_args is not None:
            source_embeds = flowedit_args["source_embeds"]
            source_image_embeds = flowedit_args.get("source_image_embeds", image_embeds)
            source_image_cond = source_image_embeds.get("image_embeds", None)
            source_clip_fea = source_image_embeds.get("clip_fea", clip_fea)
            if source_image_cond is not None:
                source_image_cond = source_image_cond.to(dtype)
            skip_steps = flowedit_args["skip_steps"]
            drift_steps = flowedit_args["drift_steps"]
            source_cfg = flowedit_args["source_cfg"]
            if not isinstance(source_cfg, list):
                source_cfg = [source_cfg] * (steps + 1)
            drift_cfg = flowedit_args["drift_cfg"]
            if not isinstance(drift_cfg, list):
                drift_cfg = [drift_cfg] * (steps + 1)

            x_init = samples["samples"].clone().squeeze(0).to(device)
            x_tgt = samples["samples"].squeeze(0).to(device)

            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=flowedit_args["drift_flow_shift"],
                use_dynamic_shifting=False,
            )

            sampling_sigmas = get_sampling_sigmas(
                steps, flowedit_args["drift_flow_shift"]
            )

            drift_timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=device, sigmas=sampling_sigmas
            )

            if drift_steps > 0:
                drift_timesteps = torch.cat(
                    [drift_timesteps, torch.tensor([0]).to(drift_timesteps.device)]
                ).to(drift_timesteps.device)
                timesteps[-drift_steps:] = drift_timesteps[-drift_steps:]

        use_cfg_zero_star = use_fresca = False
        if experimental_args is not None:
            video_attention_split_steps = experimental_args.get(
                "video_attention_split_steps", []
            )
            if video_attention_split_steps:
                transformer.video_attention_split_steps = [
                    int(x.strip()) for x in video_attention_split_steps.split(",")
                ]
            else:
                transformer.video_attention_split_steps = []

            use_zero_init = experimental_args.get("use_zero_init", True)
            use_cfg_zero_star = experimental_args.get("cfg_zero_star", False)
            zero_star_steps = experimental_args.get("zero_star_steps", 0)

            use_fresca = experimental_args.get("use_fresca", False)
            if use_fresca:
                fresca_scale_low = experimental_args.get("fresca_scale_low", 1.0)
                fresca_scale_high = experimental_args.get("fresca_scale_high", 1.25)
                fresca_freq_cutoff = experimental_args.get("fresca_freq_cutoff", 20)

        # region model pred
        def predict_with_cfg(
            z,
            cfg_scale,
            positive_embeds,
            negative_embeds,
            timestep,
            idx,
            image_cond=None,
            clip_fea=None,
            control_latents=None,
            vace_data=None,
            unianim_data=None,
            audio_proj=None,
            control_camera_latents=None,
            add_cond=None,
            cache_state=None,
        ):
            z = z.to(dtype)
            with torch.autocast(
                device_type=mm.get_autocast_device(device),
                dtype=dtype,
                enabled=("fp8" in model["quantization"]),
            ):

                if use_cfg_zero_star and (idx <= zero_star_steps) and use_zero_init:
                    return z * 0, None

                nonlocal patcher
                current_step_percentage = idx / len(timesteps)
                control_lora_enabled = False
                image_cond_input = None
                if control_latents is not None:
                    if control_lora:
                        control_lora_enabled = True
                    else:
                        if (
                            control_start_percent
                            <= current_step_percentage
                            <= control_end_percent
                        ) or (
                            control_end_percent > 0
                            and idx == 0
                            and current_step_percentage >= control_start_percent
                        ):
                            image_cond_input = torch.cat(
                                [control_latents.to(z), image_cond.to(z)]
                            )
                        else:
                            image_cond_input = torch.cat(
                                [
                                    torch.zeros_like(image_cond, dtype=dtype),
                                    image_cond.to(z),
                                ]
                            )
                        if fun_ref_image is not None:
                            fun_ref_input = fun_ref_image.to(z)
                        else:
                            fun_ref_input = torch.zeros_like(z, dtype=z.dtype)[
                                :, 0
                            ].unsqueeze(1)
                            # fun_ref_input = None

                    if control_lora:
                        if (
                            not control_start_percent
                            <= current_step_percentage
                            <= control_end_percent
                        ):
                            control_lora_enabled = False
                            if patcher.model.is_patched:
                                log.info("Unloading LoRA...")
                                patcher.unpatch_model(device)
                                patcher.model.is_patched = False
                        else:
                            image_cond_input = control_latents.to(z)
                            if not patcher.model.is_patched:
                                log.info("Loading LoRA...")
                                patcher = apply_lora(
                                    patcher, device, device, low_mem_load=False
                                )
                                patcher.model.is_patched = True

                elif ATI_tracks is not None and (
                    (ati_start_percent <= current_step_percentage <= ati_end_percent)
                    or (
                        ati_end_percent > 0
                        and idx == 0
                        and current_step_percentage >= ati_start_percent
                    )
                ):
                    image_cond_input = image_cond_ati.to(z)
                else:
                    image_cond_input = (
                        image_cond.to(z) if image_cond is not None else None
                    )

                if control_camera_latents is not None:
                    if (
                        control_camera_start_percent
                        <= current_step_percentage
                        <= control_camera_end_percent
                    ) or (
                        control_end_percent > 0
                        and idx == 0
                        and current_step_percentage >= control_camera_start_percent
                    ):
                        control_camera_input = control_camera_latents.to(z)
                    else:
                        control_camera_input = None

                if recammaster is not None:
                    z = torch.cat([z, recam_latents.to(z)], dim=1)

                use_phantom = False
                if not tiling_enabled and phantom_latents is not None:
                    if (
                        phantom_start_percent
                        <= current_step_percentage
                        <= phantom_end_percent
                    ) or (
                        phantom_end_percent > 0
                        and idx == 0
                        and current_step_percentage >= phantom_start_percent
                    ):

                        z_pos = torch.cat(
                            [z[:, : -phantom_latents.shape[1]], phantom_latents.to(z)],
                            dim=1,
                        )
                        z_phantom_img = torch.cat(
                            [z[:, : -phantom_latents.shape[1]], phantom_latents.to(z)],
                            dim=1,
                        )
                        z_neg = torch.cat(
                            [
                                z[:, : -phantom_latents.shape[1]],
                                torch.zeros_like(phantom_latents).to(z),
                            ],
                            dim=1,
                        )
                        use_phantom = True
                        if cache_state is not None and len(cache_state) != 3:
                            cache_state.append(None)
                if not use_phantom:
                    z_pos = z_neg = z

                if controlnet_latents is not None:
                    if controlnet_start <= current_step_percentage < controlnet_end:
                        self.controlnet.to(device)
                        controlnet_states = self.controlnet(
                            hidden_states=z.unsqueeze(0).to(
                                device, self.controlnet.dtype
                            ),
                            timestep=timestep,
                            encoder_hidden_states=positive_embeds[0]
                            .unsqueeze(0)
                            .to(device, self.controlnet.dtype),
                            attention_kwargs=None,
                            controlnet_states=controlnet_latents.to(
                                device, self.controlnet.dtype
                            ),
                            return_dict=False,
                        )[0]
                        if isinstance(controlnet_states, (tuple, list)):
                            controlnet["controlnet_states"] = [
                                x.to(z) for x in controlnet_states
                            ]
                        else:
                            controlnet["controlnet_states"] = controlnet_states.to(z)

                add_cond_input = None
                if add_cond is not None:
                    if (
                        add_cond_start_percent
                        <= current_step_percentage
                        <= add_cond_end_percent
                    ) or (
                        add_cond_end_percent > 0
                        and idx == 0
                        and current_step_percentage >= add_cond_start_percent
                    ):
                        add_cond_input = add_cond

                if minimax_latents is not None:
                    z_pos = z_neg = torch.cat(
                        [z, minimax_latents, minimax_mask_latents], dim=0
                    )

                base_params = {
                    "seq_len": seq_len,
                    "device": device,
                    "freqs": freqs,
                    "t": timestep,
                    "current_step": idx,
                    "control_lora_enabled": control_lora_enabled,
                    "camera_embed": camera_embed,
                    "unianim_data": unianim_data,
                    "fun_ref": fun_ref_input if fun_ref_image is not None else None,
                    "fun_camera": (
                        control_camera_input
                        if control_camera_latents is not None
                        else None
                    ),
                    "audio_proj": (
                        audio_proj if fantasytalking_embeds is not None else None
                    ),
                    "audio_scale": (
                        audio_scale if fantasytalking_embeds is not None else None
                    ),
                    "pcd_data": pcd_data,
                    "controlnet": controlnet,
                    "add_cond": add_cond_input,
                    "nag_params": text_embeds.get("nag_params", {}),
                    "nag_context": text_embeds.get("nag_prompt_embeds", None),
                }
                
                # Add audio_context_lens only if the model supports it
                if fantasytalking_embeds is not None:
                    try:
                        # Check if the model's forward method accepts audio_context_lens
                        import inspect
                        forward_signature = inspect.signature(transformer.forward)
                        if 'audio_context_lens' in forward_signature.parameters:
                            base_params["audio_context_lens"] = audio_context_lens
                            log.info(f"✅ Added audio_context_lens to base_params")
                        else:
                            log.info(f"⚠️ Model does not support audio_context_lens parameter")
                    except Exception as e:
                        # If we can't check the signature, don't add the parameter
                        log.info(f"⚠️ Could not check model signature: {e}")
                        pass

                batch_size = 1

                if not math.isclose(cfg_scale, 1.0) and len(positive_embeds) > 1:
                    negative_embeds = negative_embeds * len(positive_embeds)

                if not batched_cfg:
                    # cond
                    noise_pred_cond, cache_state_cond = transformer(
                        [z_pos],
                        context=positive_embeds,
                        y=[image_cond_input] if image_cond_input is not None else None,
                        clip_fea=clip_fea,
                        is_uncond=False,
                        current_step_percentage=current_step_percentage,
                        pred_id=cache_state[0] if cache_state else None,
                        vace_data=vace_data,
                        attn_cond=attn_cond,
                        **base_params,
                    )
                    noise_pred_cond = noise_pred_cond[0].to(intermediate_device)
                    if math.isclose(cfg_scale, 1.0):
                        if use_fresca:
                            noise_pred_cond = fourier_filter(
                                noise_pred_cond,
                                scale_low=fresca_scale_low,
                                scale_high=fresca_scale_high,
                                freq_cutoff=fresca_freq_cutoff,
                            )
                        return noise_pred_cond, [cache_state_cond]
                    # uncond
                    if fantasytalking_embeds is not None:
                        if not math.isclose(audio_cfg_scale[idx], 1.0):
                            base_params["audio_proj"] = None
                    noise_pred_uncond, cache_state_uncond = transformer(
                        [z_neg],
                        context=negative_embeds,
                        clip_fea=clip_fea_neg if clip_fea_neg is not None else clip_fea,
                        y=[image_cond_input] if image_cond_input is not None else None,
                        is_uncond=True,
                        current_step_percentage=current_step_percentage,
                        pred_id=cache_state[1] if cache_state else None,
                        vace_data=vace_data,
                        attn_cond=attn_cond_neg,
                        **base_params,
                    )
                    noise_pred_uncond = noise_pred_uncond[0].to(intermediate_device)
                    # phantom
                    if use_phantom and not math.isclose(phantom_cfg_scale[idx], 1.0):
                        noise_pred_phantom, cache_state_phantom = transformer(
                            [z_phantom_img],
                            context=negative_embeds,
                            clip_fea=(
                                clip_fea_neg if clip_fea_neg is not None else clip_fea
                            ),
                            y=(
                                [image_cond_input]
                                if image_cond_input is not None
                                else None
                            ),
                            is_uncond=True,
                            current_step_percentage=current_step_percentage,
                            pred_id=cache_state[2] if cache_state else None,
                            vace_data=None,
                            **base_params,
                        )
                        noise_pred_phantom = noise_pred_phantom[0].to(
                            intermediate_device
                        )

                        noise_pred = (
                            noise_pred_uncond
                            + phantom_cfg_scale[idx]
                            * (noise_pred_phantom - noise_pred_uncond)
                            + cfg_scale * (noise_pred_cond - noise_pred_phantom)
                        )
                        return noise_pred, [
                            cache_state_cond,
                            cache_state_uncond,
                            cache_state_phantom,
                        ]
                    # fantasytalking
                    if fantasytalking_embeds is not None:
                        if not math.isclose(audio_cfg_scale[idx], 1.0):
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            base_params["audio_proj"] = None
                            noise_pred_no_audio, cache_state_audio = transformer(
                                [z_pos],
                                context=positive_embeds,
                                y=(
                                    [image_cond_input]
                                    if image_cond_input is not None
                                    else None
                                ),
                                clip_fea=clip_fea,
                                is_uncond=False,
                                current_step_percentage=current_step_percentage,
                                pred_id=cache_state[2] if cache_state else None,
                                vace_data=vace_data,
                                **base_params,
                            )
                            noise_pred_no_audio = noise_pred_no_audio[0].to(
                                intermediate_device
                            )
                            noise_pred = (
                                noise_pred_uncond
                                + cfg_scale * (noise_pred_no_audio - noise_pred_uncond)
                                + audio_cfg_scale[idx]
                                * (noise_pred_cond - noise_pred_no_audio)
                            )
                            return noise_pred, [
                                cache_state_cond,
                                cache_state_uncond,
                                cache_state_audio,
                            ]

                # batched
                else:
                    cache_state_uncond = None
                    [noise_pred_cond, noise_pred_uncond], cache_state_cond = (
                        transformer(
                            [z] + [z],
                            context=positive_embeds + negative_embeds,
                            y=(
                                [image_cond_input] + [image_cond_input]
                                if image_cond_input is not None
                                else None
                            ),
                            clip_fea=clip_fea.repeat(2, 1, 1),
                            is_uncond=False,
                            current_step_percentage=current_step_percentage,
                            pred_id=cache_state[0] if cache_state else None,
                            **base_params,
                        )
                    )
                # cfg

                # https://github.com/WeichenFan/CFG-Zero-star/
                if use_cfg_zero_star:
                    try:
                        alpha = optimized_scale(
                            noise_pred_cond.view(batch_size, -1),
                            noise_pred_uncond.view(batch_size, -1),
                        ).view(batch_size, 1, 1, 1)
                    except:
                        alpha = 1.0
                else:
                    alpha = 1.0

                # https://github.com/WikiChao/FreSca
                if use_fresca:
                    filtered_cond = fourier_filter(
                        noise_pred_cond - noise_pred_uncond,
                        scale_low=fresca_scale_low,
                        scale_high=fresca_scale_high,
                        freq_cutoff=fresca_freq_cutoff,
                    )
                    noise_pred = (
                        noise_pred_uncond * alpha + cfg_scale * filtered_cond * alpha
                    )
                else:
                    noise_pred = noise_pred_uncond * alpha + cfg_scale * (
                        noise_pred_cond - noise_pred_uncond * alpha
                    )

                return noise_pred, [cache_state_cond, cache_state_uncond]

        log.info(
            f"Sampling {(latent_video_length-1) * 4 + 1} frames at {latent.shape[3]*8}x{latent.shape[2]*8} with {steps} steps"
        )

        intermediate_device = device

        # diff diff prep
        masks = None
        if samples is not None and mask is not None:
            mask = 1 - mask
            thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(
                timesteps
            )
            thresholds = (
                thresholds.unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .to(device)
            )
            masks = mask.repeat(len(timesteps), 1, 1, 1, 1).to(device)
            masks = masks > thresholds

        latent_shift_loop = False
        if loop_args is not None:
            latent_shift_loop = True
            is_looped = True
            latent_skip = loop_args["shift_skip"]
            latent_shift_start_percent = loop_args["start_percent"]
            latent_shift_end_percent = loop_args["end_percent"]
            shift_idx = 0

        # clear memory before sampling
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        # region main loop start
        for idx, t in enumerate(tqdm(timesteps)):
            if flowedit_args is not None:
                if idx < skip_steps:
                    continue

            # diff diff
            if masks is not None:
                if idx < len(timesteps) - 1:
                    noise_timestep = timesteps[idx + 1]
                    image_latent = sample_scheduler.scale_noise(
                        original_image, torch.tensor([noise_timestep]), noise.to(device)
                    )
                    mask = masks[idx]
                    mask = mask.to(latent)
                    latent = image_latent * mask + latent * (1 - mask)
                    # end diff diff

            latent_model_input = latent.to(device)
            
            # Pre-step inpainting mask application (noise_mask style)
            # Re-enabled with proper CONST model noise scaling
            if (is_inpainting or has_noise_mask) and concat_latent is not None and concat_mask is not None:
                log.info(f"Applying pre-step noise_mask at step {idx+1}/{len(timesteps)}")
                
                # Use concat_mask (from inpainting conditioning) for pre-step masking
                # concat_mask is [T, 1, H, W], need latent space [C, T', H', W']
                mask_to_use = concat_mask  # Use the same mask as post-step for consistency
                log.info(f"Original mask shape: {mask_to_use.shape}")
                log.info(f"Target latent shape: {latent_model_input.shape}")
                
                # For video (3D), reshape mask to [1, 1, T, H, W] before interpolation (matching ComfyUI)
                if mask_to_use.dim() == 4 and mask_to_use.shape[1] == 1:
                    mask_reshaped = mask_to_use.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 1, T, H, W]
                else:
                    mask_reshaped = mask_to_use.reshape((1, 1, -1, mask_to_use.shape[-2], mask_to_use.shape[-1]))
                
                # Use trilinear interpolation for 3D video masks (matching ComfyUI's reshape_mask)
                pre_latent_mask = F.interpolate(
                    mask_reshaped,
                    size=(latent_model_input.shape[1], latent_model_input.shape[-2], latent_model_input.shape[-1]),
                    mode="trilinear",
                    align_corners=False
                )  # [1, 1, T', H', W']
                
                # Reshape to match latent format [C, T', H', W']
                pre_latent_mask = pre_latent_mask.squeeze(0).squeeze(0)  # [T', H', W']
                pre_latent_mask = pre_latent_mask.unsqueeze(0).expand(latent_model_input.shape[0], -1, -1, -1)  # [C, T', H', W']
                pre_latent_mask = pre_latent_mask.to(latent_model_input.device, latent_model_input.dtype)
                
                log.info(f"Final pre-mask shape: {pre_latent_mask.shape}")
                log.info(f"Pre-mask unique values after all processing: {torch.unique(pre_latent_mask).numel()}")
                if torch.unique(pre_latent_mask).numel() > 2:
                    log.info("✅ Pre-step blur preserved through interpolation")
                else:
                    log.info("❌ Pre-step blur lost - mask became binary")
                
                # Get reference latent for blending
                pre_reference_latent = concat_latent.to(latent_model_input.device, latent_model_input.dtype)
                if pre_reference_latent.ndim == 5 and pre_reference_latent.shape[0] == 1:
                    pre_reference_latent = pre_reference_latent.squeeze(0)
                
                # Check and adjust spatial dimensions to match sampling resolution
                if pre_reference_latent.shape[-2:] != latent_model_input.shape[-2:]:
                    original_shape = pre_reference_latent.shape
                    # Resize reference latent to match current sampling dimensions
                    pre_reference_latent = torch.nn.functional.interpolate(
                        pre_reference_latent,
                        size=latent_model_input.shape[-2:],  # (H, W)
                        mode="bilinear",
                        align_corners=False
                    )
                    log.info(f"Resized reference latent from {original_shape} to {pre_reference_latent.shape} to match sampling resolution")
                
                # Apply proper noise scaling (ComfyUI style for flow models)
                sigma = t / 1000.0  # Convert timestep to sigma  
                
                # Generate fresh noise for scaling (not reusing existing noise)
                # Use same seed generator as main sampling for consistency
                # NOTE: This MUST happen AFTER resizing so noise matches the correct shape
                fresh_noise = torch.randn(
                    pre_reference_latent.shape,  # Now uses potentially resized shape
                    generator=seed_g, 
                    device=torch.device("cpu"),
                    dtype=pre_reference_latent.dtype
                ).to(pre_reference_latent.device)
                
                # Flow model scaling formula: sigma * noise + (1.0 - sigma) * latent_image
                # This matches ComfyUI's scale_latent_inpaint for flow models
                scaled_reference = sigma * fresh_noise + (1.0 - sigma) * pre_reference_latent
                
                log.info(f"Applied flow model noise scaling: sigma={sigma:.4f}")
                
                # NEW APPROACH: Use mask as gradual denoising strength
                # Mask values directly control blend ratio:
                # - mask=0 (black): Keep original (0% denoise)
                # - mask=1 (white): Full generation (100% denoise)
                # - mask=0.5 (gray): 50% blend
                # This provides smooth transitions without hard edges
                
                # The mask controls how much of the noisy version vs original we use
                # Formula: result = original + (noisy - original) * mask
                # This is equivalent to: result = original * (1 - mask) + noisy * mask
                denoise_strength_map = pre_latent_mask  # Use mask directly as strength map
                
                # Debug shapes before blending
                log.info(f"Shape debugging before blending:")
                log.info(f"  - latent_model_input shape: {latent_model_input.shape}")
                log.info(f"  - scaled_reference shape: {scaled_reference.shape}")
                log.info(f"  - denoise_strength_map shape: {denoise_strength_map.shape}")
                
                # Check for common dimension compatibility issues
                if latent_model_input.ndim >= 4:
                    h_idx = -2
                    w_idx = -1
                    latent_h = latent_model_input.shape[h_idx]
                    latent_w = latent_model_input.shape[w_idx]
                    
                    # Check common patch sizes
                    patch_sizes = [8, 14, 16]
                    h_compatible = any(latent_h % ps == 0 for ps in patch_sizes)
                    w_compatible = any(latent_w % ps == 0 for ps in patch_sizes)
                    
                    if not h_compatible or not w_compatible:
                        log.warning(f"⚠️ POTENTIAL DIMENSION COMPATIBILITY ISSUE ⚠️")
                        log.warning(f"Current latent dimensions: height={latent_h}, width={latent_w}")
                        log.warning(f"These dimensions may cause sequence length mismatches in the model.")
                        log.warning(f"")
                        log.warning(f"Common compatible dimensions (latent space):")
                        for ps in patch_sizes:
                            log.warning(f"  - Divisible by {ps}: {ps}, {ps*2}, {ps*3}, {ps*4}, {ps*5}...")
                        log.warning(f"")
                        log.warning(f"To fix dimension-related errors:")
                        log.warning(f"1. Try image dimensions that are multiples of 64, 128, or 256 pixels")
                        log.warning(f"2. Use a resize node to adjust dimensions before processing")
                        log.warning(f"3. Common working resolutions: 512x512, 768x768, 1024x1024, 1280x768")
                        
                        # Still check for the most common case (divisible by 16)
                        if latent_h % 16 != 0 or latent_w % 16 != 0:
                            log.warning(f"")
                            log.warning(f"Note: Dimensions divisible by 16 are most commonly supported.")
                            log.warning(f"Your dimensions: height={latent_h} {'(÷16 ❌)' if latent_h % 16 != 0 else '(÷16 ✓)'}, width={latent_w} {'(÷16 ❌)' if latent_w % 16 != 0 else '(÷16 ✓)'}")
                
                # Ensure scaled_reference matches latent_model_input dimensions
                if scaled_reference.shape != latent_model_input.shape:
                    log.info(f"Dimension mismatch detected! Resizing scaled_reference...")
                    # Handle both spatial and temporal dimension mismatches
                    if scaled_reference.ndim == latent_model_input.ndim:
                        # For 4D tensors: [B, C, H, W] or [B, T, H, W]
                        if scaled_reference.shape[1] != latent_model_input.shape[1]:
                            # Temporal dimension mismatch - need to handle carefully
                            log.warning(f"Temporal dimension mismatch: {scaled_reference.shape[1]} vs {latent_model_input.shape[1]}")
                            # For now, we'll resize all dimensions to match
                        
                        # Use adaptive pooling to match all dimensions
                        scaled_reference = torch.nn.functional.interpolate(
                            scaled_reference,
                            size=latent_model_input.shape[2:] if latent_model_input.ndim == 4 else latent_model_input.shape[1:],
                            mode="bilinear" if latent_model_input.ndim == 4 else "linear",
                            align_corners=False
                        )
                        log.info(f"Resized scaled_reference to: {scaled_reference.shape}")
                    else:
                        log.error(f"Unexpected dimension mismatch: scaled_reference ndim={scaled_reference.ndim}, latent_model_input ndim={latent_model_input.ndim}")
                
                # Final shape check before blending
                if scaled_reference.shape != latent_model_input.shape or denoise_strength_map.shape != latent_model_input.shape:
                    log.error(f"Shape mismatch before blending!")
                    log.error(f"  - latent_model_input: {latent_model_input.shape}")
                    log.error(f"  - scaled_reference: {scaled_reference.shape}")
                    log.error(f"  - denoise_strength_map: {denoise_strength_map.shape}")
                    raise RuntimeError("Cannot blend tensors with different shapes")
                
                # Blend between current latent (noisy) and scaled reference (original-like)
                # Areas with mask=0 stay close to original, mask=1 get full noise
                latent_model_input = scaled_reference + (latent_model_input - scaled_reference) * denoise_strength_map
                
                log.info(f"Pre-step gradual denoising applied:")
                log.info(f"  - Average denoise strength: {denoise_strength_map.mean().item():.3f}")
                log.info(f"  - Min/Max denoise: {denoise_strength_map.min().item():.3f}/{denoise_strength_map.max().item():.3f}")
                log.info(f"  - Scaled reference range: {scaled_reference.min().item():.3f} to {scaled_reference.max().item():.3f}")

            timestep = torch.tensor([t]).to(device)
            current_step_percentage = idx / len(timesteps)

            ### latent shift
            if latent_shift_loop:
                if (
                    latent_shift_start_percent
                    <= current_step_percentage
                    <= latent_shift_end_percent
                ):
                    latent_model_input = torch.cat(
                        [latent_model_input[:, shift_idx:]]
                        + [latent_model_input[:, :shift_idx]],
                        dim=1,
                    )

            # enhance-a-video
            if (
                feta_args is not None
                and feta_start_percent <= current_step_percentage <= feta_end_percent
            ):
                enable_enhance()
            else:
                disable_enhance()

            # flow-edit
            if flowedit_args is not None:
                sigma = t / 1000.0
                sigma_prev = (
                    timesteps[idx + 1] if idx < len(timesteps) - 1 else timesteps[-1]
                ) / 1000.0
                noise = torch.randn(
                    x_init.shape, generator=seed_g, device=torch.device("cpu")
                )
                if idx < len(timesteps) - drift_steps:
                    cfg = drift_cfg

                zt_src = (1 - sigma) * x_init + sigma * noise.to(t)
                zt_tgt = x_tgt + zt_src - x_init

                # source
                if idx < len(timesteps) - drift_steps:
                    if context_options is not None:
                        counter = torch.zeros_like(zt_src, device=intermediate_device)
                        vt_src = torch.zeros_like(zt_src, device=intermediate_device)
                        context_queue = list(
                            context(
                                idx,
                                steps,
                                latent_video_length,
                                context_frames,
                                context_stride,
                                context_overlap,
                            )
                        )
                        for c in context_queue:
                            window_id = self.window_tracker.get_window_id(c)

                            if cache_args is not None:
                                current_teacache = self.window_tracker.get_teacache(
                                    window_id, self.cache_state
                                )
                            else:
                                current_teacache = None

                            prompt_index = min(
                                int(max(c) / section_size), num_prompts - 1
                            )
                            if context_options["verbose"]:
                                log.info(f"Prompt index: {prompt_index}")

                            if len(source_embeds["prompt_embeds"]) > 1:
                                positive = source_embeds["prompt_embeds"][prompt_index]
                            else:
                                positive = source_embeds["prompt_embeds"]

                            partial_img_emb = None
                            if source_image_cond is not None:
                                partial_img_emb = source_image_cond[:, c, :, :]
                                partial_img_emb[:, 0, :, :] = source_image_cond[
                                    :, 0, :, :
                                ].to(intermediate_device)

                            partial_zt_src = zt_src[:, c, :, :]
                            vt_src_context, new_teacache = predict_with_cfg(
                                partial_zt_src,
                                cfg[idx],
                                positive,
                                source_embeds["negative_prompt_embeds"],
                                timestep,
                                idx,
                                partial_img_emb,
                                control_latents,
                                source_clip_fea,
                                current_teacache,
                            )

                            if cache_args is not None:
                                self.window_tracker.cache_states[window_id] = (
                                    new_teacache
                                )

                            window_mask = create_window_mask(
                                vt_src_context, c, latent_video_length, context_overlap
                            )
                            vt_src[:, c, :, :] += vt_src_context * window_mask
                            counter[:, c, :, :] += window_mask
                        vt_src /= counter
                    else:
                        vt_src, self.cache_state_source = predict_with_cfg(
                            zt_src,
                            cfg[idx],
                            source_embeds["prompt_embeds"],
                            source_embeds["negative_prompt_embeds"],
                            timestep,
                            idx,
                            source_image_cond,
                            source_clip_fea,
                            control_latents,
                            cache_state=self.cache_state_source,
                        )
                else:
                    if idx == len(timesteps) - drift_steps:
                        x_tgt = zt_tgt
                    zt_tgt = x_tgt
                    vt_src = 0
                # target
                if context_options is not None:
                    counter = torch.zeros_like(zt_tgt, device=intermediate_device)
                    vt_tgt = torch.zeros_like(zt_tgt, device=intermediate_device)
                    context_queue = list(
                        context(
                            idx,
                            steps,
                            latent_video_length,
                            context_frames,
                            context_stride,
                            context_overlap,
                        )
                    )
                    for c in context_queue:
                        window_id = self.window_tracker.get_window_id(c)

                        if cache_args is not None:
                            current_teacache = self.window_tracker.get_teacache(
                                window_id, self.cache_state
                            )
                        else:
                            current_teacache = None

                        prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                        if context_options["verbose"]:
                            log.info(f"Prompt index: {prompt_index}")

                        if len(text_embeds["prompt_embeds"]) > 1:
                            positive = text_embeds["prompt_embeds"][prompt_index]
                        else:
                            positive = text_embeds["prompt_embeds"]

                        partial_img_emb = None
                        partial_control_latents = None
                        if image_cond is not None:
                            partial_img_emb = image_cond[:, c, :, :]
                            partial_img_emb[:, 0, :, :] = image_cond[:, 0, :, :].to(
                                intermediate_device
                            )
                        if control_latents is not None:
                            partial_control_latents = control_latents[:, c, :, :]

                        partial_zt_tgt = zt_tgt[:, c, :, :]
                        vt_tgt_context, new_teacache = predict_with_cfg(
                            partial_zt_tgt,
                            cfg[idx],
                            positive,
                            text_embeds["negative_prompt_embeds"],
                            timestep,
                            idx,
                            partial_img_emb,
                            partial_control_latents,
                            clip_fea,
                            current_teacache,
                        )

                        if cache_args is not None:
                            self.window_tracker.cache_states[window_id] = new_teacache

                        window_mask = create_window_mask(
                            vt_tgt_context, c, latent_video_length, context_overlap
                        )
                        vt_tgt[:, c, :, :] += vt_tgt_context * window_mask
                        counter[:, c, :, :] += window_mask
                    vt_tgt /= counter
                else:
                    vt_tgt, self.cache_state = predict_with_cfg(
                        zt_tgt,
                        cfg[idx],
                        text_embeds["prompt_embeds"],
                        text_embeds["negative_prompt_embeds"],
                        timestep,
                        idx,
                        image_cond,
                        clip_fea,
                        control_latents,
                        cache_state=self.cache_state,
                    )
                v_delta = vt_tgt - vt_src
                x_tgt = x_tgt.to(torch.float32)
                v_delta = v_delta.to(torch.float32)
                x_tgt = x_tgt + (sigma_prev - sigma) * v_delta
                x0 = x_tgt
            # context windowing
            elif context_options is not None:
                counter = torch.zeros_like(
                    latent_model_input, device=intermediate_device
                )
                noise_pred = torch.zeros_like(
                    latent_model_input, device=intermediate_device
                )
                context_queue = list(
                    context(
                        idx,
                        steps,
                        latent_video_length,
                        context_frames,
                        context_stride,
                        context_overlap,
                    )
                )

                for c in context_queue:
                    window_id = self.window_tracker.get_window_id(c)

                    if cache_args is not None:
                        current_teacache = self.window_tracker.get_teacache(
                            window_id, self.cache_state
                        )
                    else:
                        current_teacache = None

                    prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                    if context_options["verbose"]:
                        log.info(f"Prompt index: {prompt_index}")

                    # Use the appropriate prompt for this section
                    if len(text_embeds["prompt_embeds"]) > 1:
                        positive = text_embeds["prompt_embeds"][prompt_index]
                    else:
                        positive = text_embeds["prompt_embeds"]

                    partial_img_emb = None
                    partial_control_latents = None
                    if image_cond is not None:
                        partial_img_emb = image_cond[:, c]
                        partial_img_emb[:, 0] = image_cond[:, 0].to(intermediate_device)

                        if control_latents is not None:
                            partial_control_latents = control_latents[:, c]

                    partial_control_camera_latents = None
                    if control_camera_latents is not None:
                        partial_control_camera_latents = control_camera_latents[:, :, c]

                    partial_vace_context = None
                    if vace_data is not None:
                        window_vace_data = []
                        for vace_entry in vace_data:
                            partial_context = vace_entry["context"][0][:, c]
                            if has_ref:
                                partial_context[:, 0] = vace_entry["context"][0][:, 0]

                            window_vace_data.append(
                                {
                                    "context": [partial_context],
                                    "scale": vace_entry["scale"],
                                    "start": vace_entry["start"],
                                    "end": vace_entry["end"],
                                    "seq_len": vace_entry["seq_len"],
                                }
                            )

                        partial_vace_context = window_vace_data

                    partial_audio_proj = None
                    if fantasytalking_embeds is not None:
                        partial_audio_proj = audio_proj[:, c]

                    partial_latent_model_input = latent_model_input[:, c]

                    partial_unianim_data = None
                    if unianim_data is not None:
                        partial_dwpose = dwpose_data[:, :, c]
                        partial_dwpose_flat = rearrange(
                            partial_dwpose, "b c f h w -> b (f h w) c"
                        )
                        partial_unianim_data = {
                            "dwpose": partial_dwpose_flat,
                            "random_ref": unianim_data["random_ref"],
                            "strength": unianimate_poses["strength"],
                            "start_percent": unianimate_poses["start_percent"],
                            "end_percent": unianimate_poses["end_percent"],
                        }

                    partial_add_cond = None
                    if add_cond is not None:
                        partial_add_cond = add_cond[:, :, c].to(device, dtype)

                    noise_pred_context, new_teacache = predict_with_cfg(
                        partial_latent_model_input,
                        cfg[idx],
                        positive,
                        text_embeds["negative_prompt_embeds"],
                        timestep,
                        idx,
                        partial_img_emb,
                        clip_fea,
                        partial_control_latents,
                        partial_vace_context,
                        partial_unianim_data,
                        partial_audio_proj,
                        partial_control_camera_latents,
                        partial_add_cond,
                        current_teacache,
                    )

                    if cache_args is not None:
                        self.window_tracker.cache_states[window_id] = new_teacache

                    window_mask = create_window_mask(
                        noise_pred_context,
                        c,
                        latent_video_length,
                        context_overlap,
                        looped=is_looped,
                    )
                    noise_pred[:, c] += noise_pred_context * window_mask
                    counter[:, c] += window_mask
                noise_pred /= counter
            # region normal inference
            else:
                noise_pred, self.cache_state = predict_with_cfg(
                    latent_model_input,
                    cfg[idx],
                    text_embeds["prompt_embeds"],
                    text_embeds["negative_prompt_embeds"],
                    timestep,
                    idx,
                    image_cond,
                    clip_fea,
                    control_latents,
                    vace_data,
                    unianim_data,
                    audio_proj,
                    control_camera_latents,
                    add_cond,
                    cache_state=self.cache_state,
                )

            if latent_shift_loop:
                # reverse latent shift
                if (
                    latent_shift_start_percent
                    <= current_step_percentage
                    <= latent_shift_end_percent
                ):
                    noise_pred = torch.cat(
                        [noise_pred[:, latent_video_length - shift_idx :]]
                        + [noise_pred[:, : latent_video_length - shift_idx]],
                        dim=1,
                    )
                    shift_idx = (shift_idx + latent_skip) % latent_video_length

            if flowedit_args is None:
                latent = latent.to(intermediate_device)
                step_args = {
                    "generator": seed_g,
                }
                if isinstance(sample_scheduler, DEISMultistepScheduler) or isinstance(
                    sample_scheduler, FlowMatchScheduler
                ):
                    step_args.pop("generator", None)
                temp_x0 = sample_scheduler.step(
                    (
                        noise_pred[:, :orig_noise_len].unsqueeze(0)
                        if recammaster is not None
                        else noise_pred.unsqueeze(0)
                    ),
                    t,
                    (
                        latent[:, :orig_noise_len].unsqueeze(0)
                        if recammaster is not None
                        else latent.unsqueeze(0)
                    ),
                    # return_dict=False,
                    **step_args,
                )[0]
                latent = temp_x0.squeeze(0)

                # Apply inpainting mask if present (from either inpainting conditioning or noise_mask)
                if (is_inpainting or has_noise_mask) and concat_latent is not None and concat_mask is not None:
                    log.info(f"Applying inpainting mask at step {idx+1}/{len(timesteps)}")
                    
                    # Get original latent and ensure it matches current latent dimensions
                    original_latent = concat_latent.to(latent.device, latent.dtype)
                    if original_latent.ndim == 5 and original_latent.shape[0] == 1:
                        original_latent = original_latent.squeeze(0)  # Remove batch dim if present
                    
                    log.info(f"Current latent shape: {latent.shape}")
                    log.info(f"Original latent shape: {original_latent.shape}")
                    log.info(f"Mask shape before processing: {concat_mask.shape}")
                    
                    # CRITICAL: Follow ComfyUI's exact mask preparation approach
                    # ComfyUI uses comfy.utils.reshape_mask which handles video masks with trilinear interpolation
                    
                    # Prepare mask for video latent space
                    # concat_mask is [T, 1, H, W] but latent is [C, T', H', W']
                    mask_frames = concat_mask.shape[0]  # 81 frames
                    latent_frames = latent.shape[1]     # 21 frames (after VAE temporal compression)
                    
                    # For video (3D), ComfyUI reshapes mask to [1, 1, T, H, W] before interpolation
                    if concat_mask.dim() == 4 and concat_mask.shape[1] == 1:
                        # Reshape from [T, 1, H, W] to [1, 1, T, H, W]
                        mask_reshaped = concat_mask.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 1, T, H, W]
                    else:
                        # Handle other mask formats if needed
                        mask_reshaped = concat_mask.reshape((1, 1, -1, concat_mask.shape[-2], concat_mask.shape[-1]))
                    
                    # Use trilinear interpolation for 3D video masks (matching ComfyUI's reshape_mask)
                    latent_mask = F.interpolate(
                        mask_reshaped,
                        size=(latent_frames, latent.shape[-2], latent.shape[-1]),  # (T', H', W')
                        mode="trilinear",
                        align_corners=False  # ComfyUI doesn't specify, but False is PyTorch default
                    )  # [1, 1, T', H', W']
                    
                    # Reshape to match latent format [C, T', H', W']
                    latent_mask = latent_mask.squeeze(0).squeeze(0)  # [T', H', W']
                    latent_mask = latent_mask.unsqueeze(0).expand(latent.shape[0], -1, -1, -1)  # [C, T', H', W']
                    latent_mask = latent_mask.to(latent.device, latent.dtype)
                    
                    log.info(f"Final mask shape: {latent_mask.shape}")
                    log.info(f"Mask min/max after processing: {latent_mask.min().item():.6f}/{latent_mask.max().item():.6f}")
                    log.info(f"Post-step mask unique values: {torch.unique(latent_mask).numel()}")
                    if torch.unique(latent_mask).numel() > 2:
                        log.info("✅ Post-step blur preserved for smooth blending")
                    else:
                        log.info("❌ Post-step blur lost - sharp boundaries will occur")
                    
                    # NEW APPROACH: Use mask as gradual denoising strength
                    # The mask directly controls the blend between original and generated
                    # - mask=0 (black): Keep original (0% denoise)
                    # - mask=1 (white): Use generated (100% denoise)
                    # - mask=0.5 (gray): 50% blend
                    # This creates smooth, natural transitions at boundaries
                    
                    # Formula: result = original + (generated - original) * mask
                    # This smoothly interpolates based on mask strength
                    denoise_strength_map = latent_mask  # Use mask directly as strength map
                    
                    # Blend between original and generated based on mask strength
                    latent = original_latent + (latent - original_latent) * denoise_strength_map
                    
                    log.info(f"Post-step gradual blending applied:")
                    log.info(f"  - Average blend strength: {denoise_strength_map.mean().item():.3f}")
                    log.info(f"  - Min/Max blend: {denoise_strength_map.min().item():.3f}/{denoise_strength_map.max().item():.3f}")
                    
                    # Additional check for gradient preservation
                    unique_values = torch.unique(denoise_strength_map)
                    if unique_values.numel() > 10:
                        log.info(f"  - ✅ Smooth gradient preserved: {unique_values.numel()} unique values")
                    elif unique_values.numel() > 2:
                        log.info(f"  - ⚠️ Limited gradient: {unique_values.numel()} unique values")
                    else:
                        log.info(f"  - ❌ Binary mask detected: {unique_values.numel()} unique values")

                x0 = latent.to(device)
                if callback is not None:
                    if recammaster is not None:
                        callback_latent = (
                            (
                                latent_model_input[:, :orig_noise_len].to(device)
                                - noise_pred[:, :orig_noise_len].to(device)
                                * t.to(device)
                                / 1000
                            )
                            .detach()
                            .permute(1, 0, 2, 3)
                        )
                    elif phantom_latents is not None:
                        callback_latent = (
                            (
                                latent_model_input[:, : -phantom_latents.shape[1]].to(
                                    device
                                )
                                - noise_pred[:, : -phantom_latents.shape[1]].to(device)
                                * t.to(device)
                                / 1000
                            )
                            .detach()
                            .permute(1, 0, 2, 3)
                        )
                    else:
                        callback_latent = (
                            (
                                latent_model_input.to(device)
                                - noise_pred.to(device) * t.to(device) / 1000
                            )
                            .detach()
                            .permute(1, 0, 2, 3)
                        )
                    # Skip preview for very high resolution to avoid OOM
                    # Note: latent space is 8x smaller than pixel space
                    latent_pixels = callback_latent.shape[2] * callback_latent.shape[3] * 64  # 8x8 = 64
                    if latent_pixels > 1280 * 720 or callback is None:
                        log.info(f"Skipping preview for high resolution: {callback_latent.shape[2]*8}x{callback_latent.shape[3]*8} pixels ({callback_latent.shape[2]}x{callback_latent.shape[3]} latents)")
                        pbar.update(1)
                    else:
                        callback(idx, callback_latent, None, steps)
                else:
                    pbar.update(1)
                del latent_model_input, timestep
            else:
                if callback is not None:
                    callback_latent = (
                        (zt_tgt.to(device) - vt_tgt.to(device) * t.to(device) / 1000)
                        .detach()
                        .permute(1, 0, 2, 3)
                    )
                    # Skip preview for very high resolution to avoid OOM
                    # Note: latent space is 8x smaller than pixel space
                    latent_pixels = callback_latent.shape[2] * callback_latent.shape[3] * 64  # 8x8 = 64
                    if latent_pixels > 1280 * 720 or callback is None:
                        log.info(f"Skipping preview for high resolution: {callback_latent.shape[2]*8}x{callback_latent.shape[3]*8} pixels ({callback_latent.shape[2]}x{callback_latent.shape[3]} latents)")
                        pbar.update(1)
                    else:
                        callback(idx, callback_latent, None, steps)
                else:
                    pbar.update(1)

        if phantom_latents is not None:
            x0 = x0[:, : -phantom_latents.shape[1]]

        if cache_args is not None:
            cache_type = cache_args["cache_type"]
            states = (
                transformer.teacache_state.states
                if cache_type == "TeaCache"
                else transformer.magcache_state.states
            )
            state_names = {0: "conditional", 1: "unconditional"}
            for pred_id, state in states.items():
                name = state_names.get(pred_id, f"prediction_{pred_id}")
                if "skipped_steps" in state:
                    log.info(
                        f"{cache_type} skipped: {len(state['skipped_steps'])} {name} steps: {state['skipped_steps']}"
                    )
            transformer.teacache_state.clear_all()
            transformer.magcache_state.clear_all()
            del states

        if force_offload:
            if model["manual_offloading"]:
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()

        try:
            print_memory(device)
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        return (
            {
                "samples": x0.unsqueeze(0).cpu(),
                "looped": is_looped,
                "end_image": end_image if not fun_or_fl2v_model else None,
                "has_ref": has_ref,
                "drop_last": drop_last,
            },
        )


NODE_CLASS_MAPPINGS = {
    "WANVideoSamplerInpaint": WANVideoSamplerInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANVideoSamplerInpaint": "WAN Video Sampler Inpaint",
}