# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat, rearrange
try:
    from ...enhance_a_video.enhance import get_feta_scores
    from ...enhance_a_video.globals import is_enhance_enabled
except ImportError:
    # Fallback if enhance_a_video is not available
    def get_feta_scores(*args, **kwargs):
        return None
    def is_enhance_enabled():
        return False

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask
    create_block_mask = torch.compile(create_block_mask)
    flex_attention = torch.compile(flex_attention)
except:
    BlockMask = create_block_mask = flex_attention = None
    pass

from .attention import attention
import numpy as np
__all__ = ['WanModel']

from tqdm import tqdm
import gc
import comfy.model_management as mm
# Removed utils import - using local logging
import logging
log = logging.getLogger(__name__)

def get_module_memory_mb(module):
    memory = 0
    for param in module.parameters():
        if param.data is not None:
            memory += param.nelement() * param.element_size()
    return memory / (1024 * 1024)  # Convert to MB

from comfy.ldm.flux.math import apply_rope as apply_rope_comfy

def rope_riflex(pos, dim, theta, L_test, k, temporal):
    assert dim % 2 == 0
    if mm.is_device_mps(pos.device) or mm.is_intel_xpu() or mm.is_directml_enabled():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)

    # RIFLEX modification - adjust last frequency component if L_test and k are provided
    if temporal and k > 0 and L_test:
        omega[k-1] = 0.9 * 2 * torch.pi / L_test

    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)

class EmbedND_RifleX(nn.Module):
    def __init__(self, dim, theta, axes_dim, num_frames, k):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.num_frames = num_frames
        self.k = k

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope_riflex(ids[..., i], self.axes_dim[i], self.theta, self.num_frames, self.k, temporal=True if i == 0 else False) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result.abs()

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000, L_test=25, k=0):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    if k > 0:
        print(f"RifleX: Using {k}th freq")
        inv_theta_pow[k-1] = 0.9 * 2 * torch.pi / L_test
        
    freqs = torch.outer(torch.arange(max_seq_len), inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

from comfy.model_management import get_torch_device, get_autocast_device
@torch.autocast(device_type=get_autocast_device(get_torch_device()), enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x)* self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 attention_mode='sdpa'):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, rope_func = "default", block_mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)


        if is_enhance_enabled():
            feta_scores = get_feta_scores(q, k)

        if self.attention_mode == 'flex_attention':
            if rope_func == "comfy":
                roped_query, roped_key = apply_rope_comfy(q, k, freqs)
            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                 torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                             device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)
        
        else:
            if rope_func == "comfy":
                q, k = apply_rope_comfy(q, k, freqs)
            else:
                q=rope_apply(q, grid_sizes, freqs)
                k=rope_apply(k, grid_sizes, freqs)
                
            x = attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
                attention_mode=self.attention_mode)

        # output
        x = x.flatten(2)
        x = self.o(x)

        if is_enhance_enabled():
            x *= feta_scores

        return x
    
    def forward_split(self, x, seq_lens, grid_sizes, freqs, seq_chunks=1,current_step=0, video_attention_split_steps = [], rope_func = "default"):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if rope_func == "comfy":
            q, k = apply_rope_comfy(q, k, freqs)
        else:
            q=rope_apply(q, grid_sizes, freqs)
            k=rope_apply(k, grid_sizes, freqs)

        if is_enhance_enabled():
            feta_scores = get_feta_scores(q, k)

        # Split by frames if multiple prompts are provided
        if seq_chunks > 1 and current_step in video_attention_split_steps:
            outputs = []
            # Extract frame, height, width from grid_sizes - force to CPU scalars
            frames = grid_sizes[0][0].item()
            height = grid_sizes[0][1].item()
            width = grid_sizes[0][2].item()
            tokens_per_frame = height * width
            
            actual_chunks = min(seq_chunks, frames)
            if isinstance(actual_chunks, torch.Tensor):
                actual_chunks = actual_chunks.item()
            
            frame_chunks = []  # Pre-calculate all chunk boundaries
            start_frame = 0
            base_frames_per_chunk = frames // actual_chunks
            extra_frames = frames % actual_chunks
            
            # Pre-calculate all chunks
            for i in range(actual_chunks):
                chunk_size = base_frames_per_chunk + (1 if i < extra_frames else 0)
                end_frame = start_frame + chunk_size
                frame_chunks.append((start_frame, end_frame))
                start_frame = end_frame
            
            # Process each chunk using the pre-calculated boundaries
            for start_frame, end_frame in frame_chunks:
                # Convert to token indices
                start_idx = int(start_frame * tokens_per_frame)
                end_idx = int(end_frame * tokens_per_frame)
                
                chunk_q = q[:, start_idx:end_idx, :, :]
                chunk_k = k[:, start_idx:end_idx, :, :]
                chunk_v = v[:, start_idx:end_idx, :, :]
                
                chunk_out = attention(
                    q=chunk_q,
                    k=chunk_k,
                    v=chunk_v,
                    k_lens=seq_lens,
                    window_size=self.window_size,
                    attention_mode=self.attention_mode)
                
                outputs.append(chunk_out)
            
            # Concatenate outputs along the sequence dimension
            x = torch.cat(outputs, dim=1)
        else:
            # Original attention computation
            x = attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
                attention_mode=self.attention_mode)

        # output
        x = x.flatten(2)
        x = self.o(x)

        if is_enhance_enabled():
            x *= feta_scores

        return x
    
    def normalized_attention_guidance(self, b, n, d, q, context, nag_context=None, nag_params={}):
        # NAG text attention
        context_positive = context
        context_negative = nag_context
        nag_scale = nag_params['nag_scale']
        nag_alpha = nag_params['nag_alpha']
        nag_tau = nag_params['nag_tau']

        k_positive = self.norm_k(self.k(context_positive)).view(b, -1, n, d)
        v_positive = self.v(context_positive).view(b, -1, n, d)
        k_negative = self.norm_k(self.k(context_negative)).view(b, -1, n, d)
        v_negative = self.v(context_negative).view(b, -1, n, d)

        x_positive = attention(q, k_positive, v_positive, k_lens=None, attention_mode=self.attention_mode)
        x_positive = x_positive.flatten(2)

        x_negative = attention(q, k_negative, v_negative, k_lens=None, attention_mode=self.attention_mode)
        x_negative = x_negative.flatten(2)

        nag_guidance = x_positive * nag_scale - x_negative * (nag_scale - 1)
        
        norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
        norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)
        
        scale = norm_guidance / norm_positive
        scale = torch.nan_to_num(scale, nan=10.0)
        
        mask = scale > nag_tau
        adjustment = (norm_positive * nag_tau) / (norm_guidance + 1e-7)
        nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)
        del mask, adjustment
        
        return nag_guidance * nag_alpha + x_positive * (1 - nag_alpha)

#region T2V crossattn
class WanT2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attention_mode='sdpa'):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.attention_mode = attention_mode

    def forward(self, x, context, context_lens, clip_embed=None, audio_proj=None, audio_context_lens=None, audio_scale=1.0, 
                num_latent_frames=21, nag_params={}, nag_context=None, is_uncond=False):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if nag_context is not None and not is_uncond:
            x_text = self.normalized_attention_guidance(b, n, d, q, context, nag_context, nag_params)
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            x_text = attention(q, k, v, k_lens=None, attention_mode=self.attention_mode)
            x_text = x_text.flatten(2)

        x = x_text

        # FantasyTalking audio attention
        if audio_proj is not None:
            if len(audio_proj.shape) == 4:
                audio_q = q.view(b * num_latent_frames, -1, n, d)
                ip_key = self.k_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                audio_x = attention(
                    audio_q, ip_key, ip_value, k_lens=audio_context_lens, attention_mode=self.attention_mode
                )
                audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
            elif len(audio_proj.shape) == 3:
                ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
                audio_x = attention(q, ip_key, ip_value, k_lens=audio_context_lens, attention_mode=self.attention_mode).flatten(2)
            
            x = x + audio_x * audio_scale

        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attention_mode='sdpa'):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attention_mode = attention_mode

    def forward(self, x, context, context_lens, clip_embed, audio_proj=None, audio_context_lens=None, 
                audio_scale=1.0, num_latent_frames=21, nag_params={}, nag_context=None, is_uncond=False):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if nag_context is not None and not is_uncond:
            x_text = self.normalized_attention_guidance(b, n, d, q, context, nag_context, nag_params)
        else:
            # text attention
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            x_text = attention(q, k, v, k_lens=context_lens, attention_mode=self.attention_mode).flatten(2)

        #img attention
        if clip_embed is not None:
            k_img = self.norm_k_img(self.k_img(clip_embed)).view(b, -1, n, d)
            v_img = self.v_img(clip_embed).view(b, -1, n, d)
            img_x = attention(q, k_img, v_img, k_lens=None, attention_mode=self.attention_mode).flatten(2)
            x = x_text + img_x
        else:
            x = x_text

        # FantasyTalking audio attention
        if audio_proj is not None:
            if len(audio_proj.shape) == 4:
                audio_q = q.view(b * num_latent_frames, -1, n, d)
                ip_key = self.k_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                audio_x = attention(
                    audio_q, ip_key, ip_value, k_lens=audio_context_lens, attention_mode=self.attention_mode
                )
                audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
            elif len(audio_proj.shape) == 3:
                ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
                audio_x = attention(q, ip_key, ip_value, k_lens=audio_context_lens, attention_mode=self.attention_mode).flatten(2)
            
            x = x + audio_x * audio_scale

        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 attention_mode='sdpa'):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, self.attention_mode)
        if cross_attn_type != "no_cross_attn":
            self.norm3 = WanLayerNorm(
                dim, eps,
                elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                          num_heads,
                                                                          (-1, -1),
                                                                          qk_norm,
                                                                          eps,#attention_mode=attention_mode sageattn doesn't seem faster here
                                                                          )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    @torch.compiler.disable()
    def get_mod(self, e):
        if e.dim() == 3:
            modulation = self.modulation  # 1, 6, dim
            e = (modulation.to(e.device) + e).chunk(6, dim=1)
        elif e.dim() == 4:
            modulation = self.modulation.unsqueeze(2)  # 1, 6, 1, dim
            e = (modulation.to(e.device) + e).chunk(6, dim=1)
            e = [ei.squeeze(1) for ei in e]
        return e
    
    def modulate(self, x, e):
        return x * (1 + e[1]) + e[0]

    #region attention forward
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        current_step,
        video_attention_split_steps=[],
        rope_func = "default",
        clip_embed=None,
        camera_embed=None,
        audio_proj=None,
        audio_context_lens=None,
        audio_scale=1.0,
        num_latent_frames=21,
        block_mask=None,
        nag_params={},
        nag_context=None,
        is_uncond=False
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        #e = (self.modulation.to(e.device) + e).chunk(6, dim=1)
        e = self.get_mod(e)

        input_x = self.modulate(self.norm1(x), e)

        if camera_embed is not None:
            # encode ReCamMaster camera
            camera_embed = self.cam_encoder(camera_embed.to(x))
            camera_embed = camera_embed.repeat(1, 2, 1)
            camera_embed = camera_embed.unsqueeze(2).unsqueeze(3).repeat(1, 1, grid_sizes[0][1], grid_sizes[0][2], 1)
            camera_embed = rearrange(camera_embed, 'b f h w d -> b (f h w) d')
            input_x += camera_embed

        # self-attention
        if context is not None and (context.shape[0] > 1 or (clip_embed is not None and clip_embed.shape[0] > 1)) and x.shape[0] == 1:
            y = self.self_attn.forward_split(
            input_x, 
            seq_lens, grid_sizes,
            freqs, rope_func=rope_func, 
            seq_chunks=max(context.shape[0], clip_embed.shape[0] if clip_embed is not None else 0),
            current_step=current_step,
            video_attention_split_steps=video_attention_split_steps
            )
        else:
            y = self.self_attn.forward(
            input_x, 
            seq_lens, grid_sizes,
            freqs, rope_func=rope_func,
            block_mask=block_mask,
            )
        #ReCamMaster
        if camera_embed is not None:
            y = self.projector(y)

        del input_x

        x = x + (y * e[2])
        del y

        # cross-attention & ffn function
        if context is not None:
            if (context.shape[0] > 1 or (clip_embed is not None and clip_embed.shape[0] > 1)) and x.shape[0] == 1:
                if nag_context is not None:
                    raise NotImplementedError("nag_context is not supported in split_cross_attn_ffn")
                x = self.split_cross_attn_ffn(x, context, context_lens, e, clip_embed=clip_embed, grid_sizes=grid_sizes)
            else:
                x = self.cross_attn_ffn(x, context, context_lens, e, clip_embed=clip_embed, grid_sizes=grid_sizes, 
                                        audio_proj=audio_proj, audio_context_lens=audio_context_lens, audio_scale=audio_scale, 
                                        num_latent_frames=num_latent_frames, nag_params=nag_params, nag_context=nag_context, is_uncond=is_uncond)
        else:
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + (y * e[5])
        del e
        return x
    #@torch.compiler.disable()
    def cross_attn_ffn(self, x, context, context_lens, e, clip_embed=None, grid_sizes=None, 
                       audio_proj=None, audio_context_lens=None, audio_scale=1.0, num_latent_frames=21, nag_params={}, 
                       nag_context=None, is_uncond=False):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, clip_embed=clip_embed,
                                    audio_proj=audio_proj, audio_context_lens=audio_context_lens, audio_scale=audio_scale, 
                                    num_latent_frames=num_latent_frames, nag_params=nag_params, nag_context=nag_context, is_uncond=is_uncond)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + (y * e[5])
            return x
    
    @torch.compiler.disable()
    def split_cross_attn_ffn(self, x, context, context_lens, e, clip_embed=None, grid_sizes=None):
        # Get number of prompts
        num_prompts = context.shape[0]
        num_clip_embeds = 0 if clip_embed is None else clip_embed.shape[0]
        num_segments = max(num_prompts, num_clip_embeds)
        
        # Extract spatial dimensions
        frames, height, width = grid_sizes[0]  # Assuming batch size 1
        tokens_per_frame = height * width
        
        # Distribute frames across prompts
        frames_per_segment = max(1, frames // num_segments)
        
        # Process each prompt segment
        x_combined = torch.zeros_like(x)
        
        for i in range(num_segments):
            # Calculate frame boundaries for this segment
            start_frame = i * frames_per_segment
            end_frame = min((i+1) * frames_per_segment, frames) if i < num_segments-1 else frames
            
            # Convert frame indices to token indices
            start_idx = start_frame * tokens_per_frame
            end_idx = end_frame * tokens_per_frame
            segment_indices = torch.arange(start_idx, end_idx, device=x.device, dtype=torch.long)
            
            # Get prompt segment (cycle through available prompts if needed)
            prompt_idx = i % num_prompts
            segment_context = context[prompt_idx:prompt_idx+1]
            segment_context_lens = None
            if context_lens is not None:
                segment_context_lens = context_lens[prompt_idx:prompt_idx+1]
            
            # Handle clip_embed for this segment (cycle through available embeddings)
            segment_clip_embed = None
            if clip_embed is not None:
                clip_idx = i % num_clip_embeds
                segment_clip_embed = clip_embed[clip_idx:clip_idx+1]
            
            # Get tensor segment
            x_segment = x[:, segment_indices, :]
            
            # Process segment with its prompt and clip embedding
            processed_segment = self.cross_attn(self.norm3(x_segment), segment_context, segment_context_lens, clip_embed=segment_clip_embed)
            processed_segment = processed_segment.to(x.dtype)
            
            # Add to combined result
            x_combined[:, segment_indices, :] = processed_segment
        
        # Continue with FFN
        x = x + x_combined
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + (y * e[5])
        return x

class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c_list, x, intermediate_device=None, nonblocking=True, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c_list[0]) + x
            all_c = []
        else:
            all_c = c_list
            c = all_c.pop(-1)
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)

        all_c += [c_skip.to(intermediate_device, non_blocking=nonblocking), c]
        
        return all_c

class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None,
        attention_mode='sdpa'
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, attention_mode)
        self.block_id = block_id

    def forward(self, x, vace_hints=None, vace_context_scale=[1.0], **kwargs):
        x = super().forward(x, **kwargs)
        if vace_hints is None:
            return x
        
        if self.block_id is not None:
            for i in range(len(vace_hints)):
                x = x + vace_hints[i][self.block_id].to(x.device) * vace_context_scale[i]
        return x

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def get_mod(self, e):
        if e.dim() == 2:
            modulation = self.modulation.to(e.device)  # 1, 2, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:
            modulation = self.modulation.to(e.device).unsqueeze(2)  # 1, 2, seq, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            e = [ei.squeeze(1) for ei in e]
        return e

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        
        # e = (self.modulation.to(e.device) + e.unsqueeze(1)).chunk(2, dim=1)
        # normed = self.norm(x)
        # x = self.head(normed * (1 + e[1]) + e[0])

        e = self.get_mod(e)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, fl_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if fl_pos_emb:  # NOTE: we only use this for `fl2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, 257 * 2, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            image_embeds = image_embeds + self.emb_pos.to(image_embeds.device)
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 attention_mode='sdpa',
                 main_device=torch.device('cuda'),
                 offload_device=torch.device('cpu'),
                 teacache_coefficients=[],
                 magcache_ratios=[],
                 vace_layers=None,
                 vace_in_dim=None,
                 inject_sample_info=False,
                 add_ref_conv=False,
                 in_dim_ref_conv=16,
                 add_control_adapter=False,
                 in_dim_control_adapter=24,
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.main_device = main_device
        self.offload_device = offload_device

        self.blocks_to_swap = -1
        self.offload_txt_emb = False
        self.offload_img_emb = False
        self.vace_blocks_to_swap = -1

        self.cache_device = offload_device

        #init TeaCache variables
        self.enable_teacache = False
        self.rel_l1_thresh = 0.15
        self.teacache_start_step= 0
        self.teacache_end_step = -1
        self.teacache_state = TeaCacheState(cache_device=self.cache_device)
        self.teacache_coefficients = teacache_coefficients
        self.teacache_use_coefficients = False
        self.teacache_mode = 'e'

        #init MagCache variables
        self.enable_magcache = False
        self.magcache_state = MagCacheState(cache_device=self.cache_device)
        self.magcache_thresh = 0.24
        self.magcache_K = 4
        self.magcache_start_step = 0
        self.magcache_end_step = -1
        self.magcache_ratios = magcache_ratios

        self.slg_blocks = None
        self.slg_start_percent = 0.0
        self.slg_end_percent = 1.0

        self.use_non_blocking = True

        self.video_attention_split_steps = []

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        self.original_patch_embedding = self.patch_embedding
        self.expanded_patch_embedding = self.patch_embedding

        if model_type != 'no_cross_attn':
            self.text_embedding = nn.Sequential(
                nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        if vace_layers is not None:
            self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
            self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

            self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

            # vace blocks
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                        self.cross_attn_norm, self.eps, block_id=i)
                for i in self.vace_layers
            ])

            # vace patch embeddings
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
            )
            self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              attention_mode=self.attention_mode,
                              block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None)
            for i in range(num_layers)
            ])
        else:
            # blocks
            if model_type == 't2v':
                cross_attn_type = 't2v_cross_attn'
            elif model_type == 'i2v' or model_type == 'fl2v':
                cross_attn_type = 'i2v_cross_attn'
            else:
                cross_attn_type = 'no_cross_attn'

            self.blocks = nn.ModuleList([
                WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                window_size, qk_norm, cross_attn_norm, eps,
                                attention_mode=self.attention_mode)
                for _ in range(num_layers)
            ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)
        

        d = self.dim // self.num_heads
        self.rope_embedder = EmbedND_RifleX(
            d, 
            10000.0, 
            [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
            num_frames=None,
            k=None,
            )

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        
        if model_type == 'i2v' or model_type == 'fl2v':
            self.img_emb = MLPProj(1280, dim, fl_pos_emb=model_type == 'fl2v')

        #skyreels v2
        if inject_sample_info:
            self.fps_embedding = nn.Embedding(2, dim)
            self.fps_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))
        #fun 1.1
        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        if add_control_adapter:
            from .wan_camera_adapter import SimpleAdapter
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        self.block_mask=None

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ):
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        print("num_frames", num_frames)
        print("frame_seqlen", frame_seqlen)
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        return block_mask

    def block_swap(self, blocks_to_swap, offload_txt_emb=False, offload_img_emb=False, vace_blocks_to_swap=None):
        log.info(f"Swapping {blocks_to_swap + 1} transformer blocks")
        self.blocks_to_swap = blocks_to_swap
        
        self.offload_img_emb = offload_img_emb
        self.offload_txt_emb = offload_txt_emb

        total_offload_memory = 0
        total_main_memory = 0
       
        for b, block in tqdm(enumerate(self.blocks), total=len(self.blocks), desc="Initializing block swap"):
            block_memory = get_module_memory_mb(block)
            
            if b > self.blocks_to_swap:
                block.to(self.main_device)
                total_main_memory += block_memory
            else:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)
                total_offload_memory += block_memory

        if blocks_to_swap != -1 and vace_blocks_to_swap == 0:
            vace_blocks_to_swap = 1

        if vace_blocks_to_swap > 0 and self.vace_layers is not None:
            self.vace_blocks_to_swap = vace_blocks_to_swap

            for b, block in tqdm(enumerate(self.vace_blocks), total=len(self.vace_blocks), desc="Initializing vace block swap"):
                block_memory = get_module_memory_mb(block)
                
                if b > self.vace_blocks_to_swap:
                    block.to(self.main_device)
                    total_main_memory += block_memory
                else:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)
                    total_offload_memory += block_memory

        mm.soft_empty_cache()
        gc.collect()

        log.info("----------------------")
        log.info(f"Block swap memory summary:")
        log.info(f"Transformer blocks on {self.offload_device}: {total_offload_memory:.2f}MB")
        log.info(f"Transformer blocks on {self.main_device}: {total_main_memory:.2f}MB")
        log.info(f"Total memory used by transformer blocks: {(total_offload_memory + total_main_memory):.2f}MB")
        log.info(f"Non-blocking memory transfer: {self.use_non_blocking}")
        log.info("----------------------")

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs
    ):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0).float()).to(x.dtype) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        if x.shape[1] > c.shape[1]:
            c = torch.cat([c.new_zeros(x.shape[0], x.shape[1] - c.shape[1], c.shape[2]), c], dim=1)
        if c.shape[1] > x.shape[1]:
            c = c[:, :x.shape[1]]
        
        c_list = [c]
        for b, block in enumerate(self.vace_blocks):
            if b <= self.vace_blocks_to_swap and self.vace_blocks_to_swap >= 0:
                block.to(self.main_device)
            c_list = block(
                c_list, x, 
                intermediate_device=self.offload_device if self.vace_blocks_to_swap != -1 else self.main_device, 
                nonblocking=self.use_non_blocking,
                **kwargs)
            if b <= self.vace_blocks_to_swap and self.vace_blocks_to_swap >= 0:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)

        hints = c_list[:-1]
        
        return hints

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        is_uncond=False,
        current_step_percentage=0.0,
        current_step=0,
        total_steps=50,
        clip_fea=None,
        y=None,
        device=torch.device('cuda'),
        freqs=None,
        pred_id=None,
        control_lora_enabled=False,
        vace_data=None,
        camera_embed=None,
        unianim_data=None,
        fps_embeds=None,
        fun_ref=None,
        fun_camera=None,
        audio_proj=None,
        audio_context_lens=None,
        audio_scale=1.0,
        pcd_data=None,
        controlnet=None,
        add_cond=None,
        attn_cond=None,
        nag_params={},
        nag_context=None
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """        
        # params
        device = self.patch_embedding.weight.device
        if freqs is not None and freqs.device != device:
           freqs = freqs.to(device)

        _, F, H, W = x[0].shape
 
        # Construct blockwise causal attn mask
        if self.attention_mode == 'flex_attention' and current_step == 0:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device, num_frames=F,
                frame_seqlen=H * W // (self.patch_size[1] * self.patch_size[2]),
                num_frame_per_block=3
            )
            
        if y is not None:
            if hasattr(self, "randomref_embedding_pose") and unianim_data is not None:
                if unianim_data['start_percent'] <= current_step_percentage <= unianim_data['end_percent']:
                    random_ref_emb = unianim_data["random_ref"]
                    if random_ref_emb is not None:
                        y[0] = y[0] + random_ref_emb * unianim_data["strength"]
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        #uni3c controlnet
        if pcd_data is not None:
            hidden_states = x[0].unsqueeze(0).clone().float()
            render_latent = torch.cat([hidden_states[:, :20], pcd_data["render_latent"]], dim=1)

        # embeddings
        if control_lora_enabled:
            self.expanded_patch_embedding.to(device)
            x = [
            self.expanded_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype)
            for u in x
            ]
        else:
            self.original_patch_embedding.to(self.main_device)
            x = [
            self.original_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype)
            for u in x
            ]

        if self.control_adapter is not None and fun_camera is not None:
            fun_camera = self.control_adapter(fun_camera)
            x = [u + v for u, v in zip(x, fun_camera)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]

        x_len = x[0].shape[1]

        if add_cond is not None:
            add_cond = self.add_conv_in(add_cond.to(self.add_conv_in.weight.dtype)).to(x[0].dtype)
            add_cond = add_cond.flatten(2).transpose(1, 2)
            x[0] = x[0] + self.add_proj(add_cond)
        if attn_cond is not None:
            F_cond, H_cond, W_cond = attn_cond.shape[2], attn_cond.shape[3], attn_cond.shape[4]
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            attn_cond = self.attn_conv_in(attn_cond.to(self.attn_conv_in.weight.dtype)).to(x[0].dtype)
            attn_cond = attn_cond.flatten(2).transpose(1, 2)
            x[0] = torch.cat([x[0], attn_cond], dim=1)
            seq_len += attn_cond.size(1)

        if self.ref_conv is not None and fun_ref is not None:
            fun_ref = self.ref_conv(fun_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += fun_ref.size(1)
            F += 1
            x = [torch.concat([_fun_ref.unsqueeze(0), u], dim=1) for _fun_ref, u in zip(fun_ref, x)]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        if freqs is None: #comfy rope
            rope_func = "comfy"
            f_len = ((F + (self.patch_size[0] // 2)) // self.patch_size[0])
            h_len = ((H + (self.patch_size[1] // 2)) // self.patch_size[1])
            w_len = ((W + (self.patch_size[2] // 2)) // self.patch_size[2])
            img_ids = torch.zeros((f_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, f_len - 1, steps=f_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
            img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
            img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)

            if attn_cond is not None:   
                cond_f_len = ((F_cond + (self.patch_size[0] // 2)) // self.patch_size[0])
                cond_h_len = ((H_cond + (self.patch_size[1] // 2)) // self.patch_size[1])
                cond_w_len = ((W_cond + (self.patch_size[2] // 2)) // self.patch_size[2])
                cond_img_ids = torch.zeros((cond_f_len, cond_h_len, cond_w_len, 3), device=x.device, dtype=x.dtype)
                
                #shift
                shift_f_size = 81 # Default value
                shift_f = False
                if shift_f:
                    cond_img_ids[:, :, :, 0] = cond_img_ids[:, :, :, 0] + torch.linspace(shift_f_size, shift_f_size + cond_f_len - 1,steps=cond_f_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
                else:
                    cond_img_ids[:, :, :, 0] = cond_img_ids[:, :, :, 0] + torch.linspace(0, cond_f_len - 1, steps=cond_f_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
                cond_img_ids[:, :, :, 1] = cond_img_ids[:, :, :, 1] + torch.linspace(h_len, h_len + cond_h_len - 1, steps=cond_h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
                cond_img_ids[:, :, :, 2] = cond_img_ids[:, :, :, 2] + torch.linspace(w_len, w_len + cond_w_len - 1, steps=cond_w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
              
                # Combine original and conditional position ids
                img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=1)
                cond_img_ids = repeat(cond_img_ids, "t h w c -> b (t h w) c", b=1)
                combined_img_ids = torch.cat([img_ids, cond_img_ids], dim=1)
                
                # Generate RoPE frequencies for the combined positions
                freqs = self.rope_embedder(combined_img_ids).movedim(1, 2)
            else:
                img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=1)
                freqs = self.rope_embedder(img_ids).movedim(1, 2)
        else:
            rope_func = "default"

        # time embeddings
        if t.dim() == 2:
            b, f = t.shape
            _flag_df = True
        else:
            _flag_df = False

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(x.dtype)
        )  # b, dim
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # b, 6, dim

        if fps_embeds is not None:
            fps_embeds = torch.tensor(fps_embeds, dtype=torch.long, device=device)

            fps_emb = self.fps_embedding(fps_embeds).to(e0.dtype)
            if _flag_df:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim)).repeat(t.shape[1], 1, 1)
            else:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim))

        if _flag_df:
            e = e.view(b, f, 1, 1, self.dim).expand(b, f, grid_sizes[0][1], grid_sizes[0][2], self.dim)
            e0 = e0.view(b, f, 1, 1, 6, self.dim).expand(b, f, grid_sizes[0][1], grid_sizes[0][2], 6, self.dim)
            
            e = e.flatten(1, 3)
            e0 = e0.flatten(1, 3)
            
            e0 = e0.transpose(1, 2)
            if not e0.is_contiguous():
                e0 = e0.contiguous()
            
            e = e.to(self.offload_device, non_blocking=self.use_non_blocking)

        # context (test embedding)
        context_lens = None
        if hasattr(self, "text_embedding"):
            if self.offload_txt_emb:
                self.text_embedding.to(self.main_device)
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]).to(x.dtype))
            # NAG
            if nag_context is not None:
                nag_context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in nag_context
                ]).to(x.dtype))
            
            if self.offload_txt_emb:
                self.text_embedding.to(self.offload_device, non_blocking=self.use_non_blocking)
        else:
            context = None

        clip_embed = None
        if clip_fea is not None and hasattr(self, "img_emb"):
            clip_fea = clip_fea.to(self.main_device)
            if self.offload_img_emb:
                self.img_emb.to(self.main_device)
            clip_embed = self.img_emb(clip_fea)  # bs x 257 x dim
            #context = torch.concat([context_clip, context], dim=1)
            if self.offload_img_emb:
                self.img_emb.to(self.offload_device, non_blocking=self.use_non_blocking)

        should_calc = True
        accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
        if self.enable_teacache and self.teacache_start_step <= current_step <= self.teacache_end_step:
            if pred_id is None:
                pred_id = self.teacache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True                
            else:
                previous_modulated_input = self.teacache_state.get(pred_id)['previous_modulated_input']
                previous_modulated_input = previous_modulated_input.to(device)
                previous_residual = self.teacache_state.get(pred_id)['previous_residual']
                accumulated_rel_l1_distance = self.teacache_state.get(pred_id)['accumulated_rel_l1_distance']

                if self.teacache_use_coefficients:
                    rescale_func = np.poly1d(self.teacache_coefficients[self.teacache_mode])
                    temb = e if self.teacache_mode == 'e' else e0
                    accumulated_rel_l1_distance += rescale_func((
                        (temb.to(device) - previous_modulated_input).abs().mean() / previous_modulated_input.abs().mean()
                        ).cpu().item())
                    del temb
                else:
                    temb_relative_l1 = relative_l1_distance(previous_modulated_input, e0)
                    accumulated_rel_l1_distance = accumulated_rel_l1_distance.to(e0.device) + temb_relative_l1
                    del temb_relative_l1


                if accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
                accumulated_rel_l1_distance = accumulated_rel_l1_distance.to(self.cache_device)

            previous_modulated_input = e.to(self.cache_device).clone() if (self.teacache_use_coefficients and self.teacache_mode == 'e') else e0.to(self.cache_device).clone()
           
            if not should_calc:
                x = x.to(previous_residual.dtype) + previous_residual.to(x.device)
                self.teacache_state.update(
                    pred_id,
                    accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                )
                self.teacache_state.get(pred_id)['skipped_steps'].append(current_step)

        # enable magcache
        if self.enable_magcache and self.magcache_start_step <= current_step <= self.magcache_end_step:
            if pred_id is None:
                pred_id = self.magcache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True
            else:
                accumulated_ratio = self.magcache_state.get(pred_id)['accumulated_ratio']
                accumulated_err = self.magcache_state.get(pred_id)['accumulated_err']
                accumulated_steps = self.magcache_state.get(pred_id)['accumulated_steps']

                calibration_len = len(self.magcache_ratios) // 2
                cur_mag_ratio = self.magcache_ratios[int((current_step*(calibration_len/total_steps)))]

                accumulated_ratio *= cur_mag_ratio
                accumulated_err += np.abs(1-accumulated_ratio)
                accumulated_steps += 1

                self.magcache_state.update(
                    pred_id,
                    accumulated_ratio=accumulated_ratio,
                    accumulated_steps=accumulated_steps,
                    accumulated_err=accumulated_err
                )

                if accumulated_err<=self.magcache_thresh and accumulated_steps<=self.magcache_K:
                    should_calc = False
                    x += self.magcache_state.get(pred_id)['residual_cache'].to(x.device)
                    self.magcache_state.get(pred_id)['skipped_steps'].append(current_step)
                else:
                    should_calc = True
                    self.magcache_state.update(
                        pred_id,
                        accumulated_ratio=1.0,
                        accumulated_steps=0,
                        accumulated_err=0
                    )

        if should_calc:
            if self.enable_teacache or self.enable_magcache:
                original_x = x.to(self.cache_device).clone()

            if hasattr(self, "dwpose_embedding") and unianim_data is not None:
                if unianim_data['start_percent'] <= current_step_percentage <= unianim_data['end_percent']:
                    dwpose_emb = unianim_data['dwpose']
                    x += dwpose_emb * unianim_data['strength']
            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                context_lens=context_lens,
                clip_embed=clip_embed,
                rope_func=rope_func,
                current_step=current_step,
                video_attention_split_steps=self.video_attention_split_steps,
                camera_embed=camera_embed,
                audio_proj=audio_proj,
                audio_context_lens=audio_context_lens,
                num_latent_frames = F,
                audio_scale=audio_scale,
                block_mask=self.block_mask,
                nag_params=nag_params,
                nag_context=nag_context,
                is_uncond = is_uncond
                )
            
            if vace_data is not None:
                vace_hint_list = []
                vace_scale_list = []
                if isinstance(vace_data[0], dict):
                    for data in vace_data:
                        if (data["start"] <= current_step_percentage <= data["end"]) or \
                            (data["end"] > 0 and current_step == 0 and current_step_percentage >= data["start"]):

                            vace_hints = self.forward_vace(x, data["context"], data["seq_len"], kwargs)
                            vace_hint_list.append(vace_hints)
                            vace_scale_list.append(data["scale"][current_step])
                else:
                    vace_hints = self.forward_vace(x, vace_data, seq_len, kwargs)
                    vace_hint_list.append(vace_hints)
                    vace_scale_list.append(1.0)
                
                kwargs['vace_hints'] = vace_hint_list
                kwargs['vace_context_scale'] = vace_scale_list

            #uni3c controlnet
            pdc_controlnet_states = None
            if pcd_data is not None:
                if (pcd_data["start"] <= current_step_percentage <= pcd_data["end"]) or \
                            (pcd_data["end"] > 0 and current_step == 0 and current_step_percentage >= pcd_data["start"]):
                    self.controlnet.to(self.main_device)
                    pdc_controlnet_states = self.controlnet(
                        render_latent=render_latent.to(self.main_device, self.controlnet.dtype), 
                        render_mask=pcd_data["render_mask"], 
                        camera_embedding=pcd_data["camera_embedding"], 
                        temb=e.to(self.main_device),
                        device=self.offload_device)
                    self.controlnet.to(self.offload_device)

            for b, block in enumerate(self.blocks):
                #skip layer guidance
                if self.slg_blocks is not None:
                    if b in self.slg_blocks and is_uncond:
                        if self.slg_start_percent <= current_step_percentage <= self.slg_end_percent:
                            continue
                if b <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                    block.to(self.main_device)
                x = block(x, **kwargs)

                #uni3c controlnet
                if pdc_controlnet_states is not None and b < len(pdc_controlnet_states):
                    x[:, :x_len] += pdc_controlnet_states[b].to(x) * pcd_data["controlnet_weight"]
                #controlnet
                if (controlnet is not None) and (b % controlnet["controlnet_stride"] == 0) and (b // controlnet["controlnet_stride"] < len(controlnet["controlnet_states"])):
                    x[:, :x_len] += controlnet["controlnet_states"][b // controlnet["controlnet_stride"]].to(x) * controlnet["controlnet_weight"]

                if b <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)

            if self.enable_teacache and (self.teacache_start_step <= current_step <= self.teacache_end_step) and pred_id is not None:
                self.teacache_state.update(
                    pred_id,
                    previous_residual=(x.to(original_x.device) - original_x),
                    accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                    previous_modulated_input=previous_modulated_input
                )
            elif self.enable_magcache and (self.magcache_start_step <= current_step <= self.magcache_end_step) and pred_id is not None:
                self.magcache_state.update(
                    pred_id,
                    residual_cache=(x.to(original_x.device) - original_x)
                )
                
        if self.ref_conv is not None and fun_ref is not None:
            full_ref_length = fun_ref.size(1)
            x = x[:, full_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        if attn_cond is not None:
            x = x[:, :x_len]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        x = self.head(x, e.to(x.device))
        x = self.unpatchify(x, grid_sizes) # type: ignore[arg-type]
        x = [u.float() for u in x]
        return (x, pred_id) if pred_id is not None else (x, None)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

class TeaCacheState:
    def __init__(self, cache_device='cpu'):
        self.cache_device = cache_device
        self.states = {}
        self._next_pred_id = 0
    
    def new_prediction(self, cache_device='cpu'):
        """Create new prediction state and return its ID"""
        self.cache_device = cache_device
        pred_id = self._next_pred_id
        self._next_pred_id += 1
        self.states[pred_id] = {
            'previous_residual': None,
            'accumulated_rel_l1_distance': 0,
            'previous_modulated_input': None,
            'skipped_steps': [],
        }
        return pred_id
    
    def update(self, pred_id, **kwargs):
        """Update state for specific prediction"""
        if pred_id not in self.states:
            return None
        for key, value in kwargs.items():
            self.states[pred_id][key] = value
    
    def get(self, pred_id):
        return self.states.get(pred_id, {})
    
    def clear_all(self):
        self.states = {}
        self._next_pred_id = 0

class MagCacheState:
    def __init__(self, cache_device='cpu'):
        self.cache_device = cache_device
        self.states = {}
        self._next_pred_id = 0
    
    def new_prediction(self, cache_device='cpu'):
        """Create new prediction state and return its ID"""
        self.cache_device = cache_device
        pred_id = self._next_pred_id
        self._next_pred_id += 1
        self.states[pred_id] = {
            'residual_cache': None,
            'accumulated_ratio': 1.0,
            'accumulated_steps': 0,
            'accumulated_err': 0,
            'skipped_steps': [],
        }
        return pred_id
    
    def update(self, pred_id, **kwargs):
        """Update state for specific prediction"""
        if pred_id not in self.states:
            return None
        for key, value in kwargs.items():
            self.states[pred_id][key] = value
    
    def get(self, pred_id):
        return self.states.get(pred_id, {})
    
    def clear_all(self):
        self.states = {}
        self._next_pred_id = 0

def relative_l1_distance(last_tensor, current_tensor):
    l1_distance = torch.abs(last_tensor.to(current_tensor.device) - current_tensor).mean()
    norm = torch.abs(last_tensor).mean()
    relative_l1_distance = l1_distance / norm
    return relative_l1_distance.to(torch.float32).to(current_tensor.device)

def get_tensor_memory(tensor):
    memory_bytes = tensor.element_size() * tensor.nelement()
    return f"{memory_bytes / (1024 * 1024):.2f} MB"