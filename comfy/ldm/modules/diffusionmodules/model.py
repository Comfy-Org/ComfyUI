# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import logging

from comfy import model_management
import comfy.ops
import comfy.quant_ops
ops = comfy.ops.disable_weight_init

if model_management.xformers_enabled_vae():
    import xformers
    import xformers.ops

def torch_cat_if_needed(xl, dim):
    xl = [x for x in xl if x is not None and x.shape[dim] > 0]
    if len(xl) > 1:
        return torch.cat(xl, dim)
    elif len(xl) == 1:
        return xl[0]
    else:
        return None

def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=1):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - downscale_freq_shift)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return torch.nn.functional.silu(x)


def Normalize(in_channels, num_groups=32):
    return ops.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def fused_norm_served(device, dtype):
    # MIOpen runs a channels_last conv through NCHW and loses the gain; the kitchen fp16 conv on RDNA3+ does not
    ck = getattr(comfy.quant_ops, "ck", None)
    if ck is None or device.type != "cuda":
        return False
    if torch.version.hip:
        return dtype == torch.float16 and ck.registry.get_constraints("hip", "fp16_conv3d") is not None
    return dtype in (torch.float16, torch.bfloat16)



STRIP_ELEMS = 2 ** 24


def strip_rows(fn, x, out_elems, scale=1, out=None, accumulate=False):
    # fn(rows, top, bottom) runs a 3x3 conv on a strip with a one-row halo, padding only at the image edge
    size = x.shape[2]
    step = -(-size // -(-max(x.numel(), out_elems) // STRIP_ELEMS))
    for a in range(0, size, step):
        b = min(size, a + step)
        lo, hi = max(0, a - 1), min(size, b + 1)
        top = int(lo == 0)
        rows = fn(x.narrow(2, lo, hi - lo), top, int(hi == size))
        rows = rows.narrow(2, scale * (a - lo) + top - 1, scale * (b - a))
        if out is None:
            out = torch.empty((rows.shape[0], rows.shape[1], size * scale, rows.shape[3]), dtype=rows.dtype, device=rows.device, memory_format=torch.channels_last)
        dst = out.narrow(2, scale * a, scale * (b - a))
        if accumulate:
            dst.add_(rows)
        else:
            dst.copy_(rows)
    return out


def interleaved_view(x, channels, part):
    # x's storage as two half-width tensors whose rows alternate inside each row slot
    b, c, h, w = x.shape
    flat = x.permute(0, 2, 3, 1).reshape(-1)
    return flat.as_strided((1, channels, h, w), (h * w * c, 1, w * c, channels), flat.storage_offset() + part * w * channels)


def region(buf, end, channels, h, w):
    # the channels_last tensor of that shape ending at element `end` of the flat buffer
    n = channels * h * w
    return buf.narrow(0, end - n, n).view(1, h, w, channels).permute(0, 3, 1, 2)


def padded_conv(padded, weight, bias):
    # a 5-D weight means the fp16-accumulate opt-in: the kitchen conv takes the padded strip as a depth-1 conv3d
    if weight.ndim == 5:
        return comfy.quant_ops.ck.fp16_conv3d(padded, weight, bias, None, 1).squeeze(2)
    return torch.nn.functional.conv2d(padded.squeeze(2), weight, bias)


def conv_weight(weight, x):
    # the ROCm kitchen conv accumulates in fp32, so it needs no fp16-accumulation opt-in
    if comfy.ops._fp16_linear_wanted(x) or (torch.version.hip and x.dtype == torch.float16):
        return weight.unsqueeze(2).contiguous(memory_format=torch.channels_last_3d)
    return weight


def strip_conv(conv, x, norm=None, scale=1, out=None, accumulate=False):
    # conv(silu(norm(x))) in row strips, after a nearest upsample when scale > 1
    ck = comfy.quant_ops.ck
    groups, eps = (1, 0.0) if norm is None else (norm.num_groups, norm.eps)
    stats = None if norm is None else ck.group_norm_stats3d(x.unsqueeze(2), groups, eps)
    with comfy.ops.CastBiasWeightContext(norm, x, offloadable=True) as (norm_weight, norm_bias), comfy.ops.CastBiasWeightContext(conv, x, offloadable=True) as (weight, bias):
        weight = conv_weight(weight, x)

        def fn(rows, top, bottom):
            if scale > 1:
                rows = interpolate_up(rows, scale)
            rows = ck.group_norm_silu_pad3d(rows.unsqueeze(2), norm_weight, norm_bias, groups, eps, [1, 1, top, bottom, 0], norm is not None, zero_pad=True, stats=stats)
            return padded_conv(rows, weight, bias)
        return strip_rows(fn, x, x.shape[0] * conv.out_channels * x.shape[2] * x.shape[3] * scale * scale, scale, out, accumulate)


class CarriedConv3d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, **kwargs):
        super().__init__()
        self.conv = ops.Conv3d(n_channels, out_channels, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        return self.conv(x)


def conv_carry_causal_3d(xl, op, conv_carry_in=None, conv_carry_out=None):

    x = xl[0]
    xl.clear()

    if isinstance(op, CarriedConv3d):
        if conv_carry_in is None:
            x = torch.nn.functional.pad(x, (1, 1, 1, 1, 2, 0), mode = 'replicate')
        else:
            carry_len = conv_carry_in[0].shape[2]
            x = torch.nn.functional.pad(x, (1, 1, 1, 1, 2 - carry_len, 0), mode = 'replicate')
            x = torch.cat([conv_carry_in.pop(0), x], dim=2)

        if conv_carry_out is not None:
            to_push = x[:, :, -2:, :, :].clone()
            conv_carry_out.append(to_push)

    out = op(x)

    return out


class VideoConv3d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1, dilation=1, padding_mode='replicate', padding=1, **kwargs):
        super().__init__()

        self.padding_mode = padding_mode
        if padding != 0:
            padding = (padding, padding, padding, padding, kernel_size - 1, 0)
        else:
            kwargs["padding"] = padding

        self.padding = padding
        self.conv = ops.Conv3d(n_channels, out_channels, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        if self.padding != 0:
            x = torch.nn.functional.pad(x, self.padding, mode=self.padding_mode)
        return self.conv(x)

def interpolate_up(x, scale_factor):
    return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="nearest")

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, conv_op=ops.Conv2d, scale_factor=2.0, fused_norm=False):
        super().__init__()
        self.with_conv = with_conv
        self.scale_factor = scale_factor
        self.fused_norm = fused_norm and with_conv

        if self.with_conv:
            self.conv = conv_op(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, conv_carry_in=None, conv_carry_out=None, out=None):
        scale_factor = self.scale_factor
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor,) * (x.ndim - 2)

        if self.fused_norm and fused_norm_served(x.device, x.dtype):
            return strip_conv(self.conv, x, scale=2, out=out)

        if x.ndim == 5 and scale_factor[0] > 1.0:
            results = []
            if conv_carry_in is None:
                first = x[:, :, :1, :, :]
                results.append(interpolate_up(first.squeeze(2), scale_factor=scale_factor[1:]).unsqueeze(2))
                x = x[:, :, 1:, :, :]
            if x.shape[2] > 0:
                results.append(interpolate_up(x, scale_factor))
            x = torch_cat_if_needed(results, dim=2)
        else:
            x = interpolate_up(x, scale_factor)
        if self.with_conv:
            x = conv_carry_causal_3d([x], self.conv, conv_carry_in, conv_carry_out)
        return x


def strip_down(conv, x):
    # the stride-2 conv with its (0, 1, 0, 1) pad in strips: output rows [a, b) read input rows [2a, 2b + 1)
    ck = comfy.quant_ops.ck
    rows = x.shape[2] // 2
    step = -(-rows // -(-x.numel() // STRIP_ELEMS))
    out = None
    with comfy.ops.CastBiasWeightContext(conv, x, offloadable=True) as (weight, bias):
        for a in range(0, rows, step):
            b = min(rows, a + step)
            hi = min(x.shape[2], 2 * b + 1)
            padded = ck.group_norm_silu_pad3d(x.narrow(2, 2 * a, hi - 2 * a).unsqueeze(2), None, None, 1, 0.0, [0, 1, 0, int(hi == x.shape[2]), 0], False, zero_pad=True).squeeze(2)
            y = torch.nn.functional.conv2d(padded, weight, bias, stride=2)
            if out is None:
                out = torch.empty((x.shape[0], y.shape[1], rows, y.shape[3]), dtype=y.dtype, device=y.device, memory_format=torch.channels_last)
            out.narrow(2, a, b - a).copy_(y)
    return out


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride=2, conv_op=ops.Conv2d, fused_norm=False):
        super().__init__()
        self.with_conv = with_conv
        self.fused_norm = fused_norm and with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = conv_op(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=0)

    def forward(self, x, conv_carry_in=None, conv_carry_out=None):
        if self.fused_norm and fused_norm_served(x.device, x.dtype):
            return strip_down(self.conv, x)
        if self.with_conv:
            if isinstance(self.conv, CarriedConv3d):
                x = conv_carry_causal_3d([x], self.conv, conv_carry_in, conv_carry_out)
            elif x.ndim == 4:
                pad = (0, 1, 0, 1)
                mode = "constant"
                x = torch.nn.functional.pad(x, pad, mode=mode, value=0)
                x = self.conv(x)
            elif x.ndim == 5:
                pad = (1, 1, 1, 1, 2, 0)
                mode = "replicate"
                x = torch.nn.functional.pad(x, pad, mode=mode)
                x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, temb_channels=512, conv_op=ops.Conv2d, norm_op=Normalize, fused_norm=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.fused_norm = fused_norm

        self.swish = torch.nn.SiLU(inplace=True)
        self.norm1 = norm_op(in_channels)
        self.conv1 = conv_op(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = ops.Linear(temb_channels,
                                             out_channels)
        self.norm2 = norm_op(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = conv_op(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_op(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = conv_op(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def halves_in_place(self, x):
        return x.shape[0] == 1 and self.in_channels == 2 * self.out_channels and x.is_contiguous(memory_format=torch.channels_last)

    def halve_in_place(self, x):
        # C -> C/2: hidden and output interleave inside x; a strip's last row waits until the next strip has read its halo
        ck = comfy.quant_ops.ck
        h = interleaved_view(x, self.out_channels, 0)
        out = interleaved_view(x, self.out_channels, 1)
        stats = ck.group_norm_stats3d(x.unsqueeze(2), self.norm1.num_groups, self.norm1.eps)
        rows = x.shape[2]
        step = -(-rows // -(-x.numel() // STRIP_ELEMS))
        held = None
        with comfy.ops.CastBiasWeightContext(self.norm1, x, offloadable=True) as (norm_weight, norm_bias), comfy.ops.CastBiasWeightContext(self.conv1, x, offloadable=True) as (weight, bias), comfy.ops.CastBiasWeightContext(self.nin_shortcut, x, offloadable=True) as (sc_weight, sc_bias):
            weight = conv_weight(weight, x)
            for a in range(0, rows, step):
                b = min(rows, a + step)
                lo, hi = max(0, a - 1), min(rows, b + 1)
                padded = ck.group_norm_silu_pad3d(x.narrow(2, lo, hi - lo).unsqueeze(2), norm_weight, norm_bias, self.norm1.num_groups, self.norm1.eps, [1, 1, int(lo == 0), int(hi == rows), 0], True, zero_pad=True, stats=stats)
                h_rows = padded_conv(padded, weight, bias)
                out_rows = torch.nn.functional.conv2d(x.narrow(2, a, b - a), sc_weight, sc_bias)
                if held is not None:
                    h.narrow(2, a - 1, 1).copy_(held[0])
                    out.narrow(2, a - 1, 1).copy_(held[1])
                h.narrow(2, a, b - a - 1).copy_(h_rows.narrow(2, 0, b - a - 1))
                out.narrow(2, a, b - a - 1).copy_(out_rows.narrow(2, 0, b - a - 1))
                held = (h_rows.narrow(2, b - a - 1, 1).clone(), out_rows.narrow(2, b - a - 1, 1).clone())
            h.narrow(2, rows - 1, 1).copy_(held[0])
            out.narrow(2, rows - 1, 1).copy_(held[1])
        return strip_conv(self.conv2, h, self.norm2, out=out, accumulate=True)

    def forward(self, x, temb=None, conv_carry_in=None, conv_carry_out=None, hidden=None):
        if self.fused_norm and fused_norm_served(x.device, x.dtype):
            # the block consumes x: the residual stream is updated in place
            if x.stride(1) != 1:  # NCHW, after the mid-block attention
                x = x.contiguous(memory_format=torch.channels_last)
            if self.in_channels == self.out_channels:
                h = strip_conv(self.conv1, x, self.norm1, out=hidden)
                return strip_conv(self.conv2, h, self.norm2, out=x, accumulate=True)
            if self.halves_in_place(x):
                return self.halve_in_place(x)
            h = strip_conv(self.conv1, x, self.norm1)
            return strip_conv(self.conv2, h, self.norm2, out=self.nin_shortcut(x), accumulate=True)

        h = x
        h = self.norm1(h)
        h = [ self.swish(h) ]
        h = conv_carry_causal_3d(h, self.conv1, conv_carry_in=conv_carry_in, conv_carry_out=conv_carry_out)

        if temb is not None:
            h = h + self.temb_proj(self.swish(temb))[:,:,None,None]

        h = self.norm2(h)
        h = self.swish(h)
        h = [ self.dropout(h) ]
        h = conv_carry_causal_3d(h, self.conv2, conv_carry_in=conv_carry_in, conv_carry_out=conv_carry_out)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = conv_carry_causal_3d([x], self.conv_shortcut, conv_carry_in=conv_carry_in, conv_carry_out=conv_carry_out)
            else:
                x = self.nin_shortcut(x)

        return x+h

def slice_attention(q, k, v):
    r1 = torch.zeros_like(k, device=q.device)
    scale = (int(q.shape[-1])**(-0.5))

    mem_free_total = model_management.get_free_memory(q.device)

    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

    while True:
        try:
            slice_size = math.ceil(q.shape[1] / steps)
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                s1 = torch.bmm(q[:, i:end], k) * scale

                s2 = torch.nn.functional.softmax(s1, dim=2).permute(0,2,1)
                del s1

                r1[:, :, i:end] = torch.bmm(v, s2)
                del s2
            break
        except Exception as e:
            model_management.raise_non_oom(e)
            model_management.soft_empty_cache(True)
            steps *= 2
            if steps > 128:
                raise e
            logging.warning("out of memory error, increasing steps and trying again {}".format(steps))

    return r1

def normal_attention(q, k, v):
    # compute attention
    orig_shape = q.shape
    b = orig_shape[0]
    c = orig_shape[1]

    q = q.reshape(b, c, -1)
    q = q.permute(0, 2, 1)   # b,hw,c
    k = k.reshape(b, c, -1) # b,c,hw
    v = v.reshape(b, c, -1)

    r1 = slice_attention(q, k, v)
    h_ = r1.reshape(orig_shape)
    del r1
    return h_

def xformers_attention(q, k, v):
    # compute attention
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )

    try:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        out = out.transpose(1, 2).reshape(orig_shape)
    except NotImplementedError:
        out = slice_attention(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(orig_shape)
    return out

def pytorch_attention(q, k, v):
    # compute attention
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    oom_fallback = False
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )

    try:
        out = comfy.ops.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).reshape(orig_shape)
    except Exception as e:
        model_management.raise_non_oom(e)
        logging.warning("scaled_dot_product_attention OOMed: switched to slice attention")
        oom_fallback = True
    if oom_fallback:
        out = slice_attention(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(orig_shape)
    return out


def vae_attention():
    if model_management.xformers_enabled_vae():
        logging.info("Using xformers attention in VAE")
        return xformers_attention
    elif model_management.pytorch_attention_enabled_vae():
        logging.info("Using pytorch attention in VAE")
        return pytorch_attention
    else:
        logging.info("Using split attention in VAE")
        return normal_attention

class AttnBlock(nn.Module):
    def __init__(self, in_channels, conv_op=ops.Conv2d, norm_op=Normalize):
        super().__init__()
        self.in_channels = in_channels

        self.norm = norm_op(in_channels)
        self.q = conv_op(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = conv_op(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = conv_op(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = conv_op(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        self.optimized_attention = vae_attention()

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        h_ = self.optimized_attention(q, k, v)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None, conv_op=ops.Conv2d):
    return AttnBlock(in_channels, conv_op=conv_op)


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                ops.Linear(self.ch,
                                self.temb_ch),
                ops.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = ops.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = ops.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 conv3d=False, time_compress=None, fused_norm=False,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.fused_norm = fused_norm = fused_norm and not conv3d
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.carried = False

        if conv3d:
            if not attn_resolutions:
                conv_op = CarriedConv3d
                self.carried = True
            else:
                conv_op = VideoConv3d
            mid_attn_conv_op = ops.Conv3d
        else:
            conv_op = ops.Conv2d
            mid_attn_conv_op = ops.Conv2d

        # downsampling
        self.conv_in = conv_op(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.time_compress = 1
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         conv_op=conv_op,
                                         fused_norm=fused_norm))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, conv_op=conv_op))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                stride = 2
                if time_compress is not None:
                    if (self.num_resolutions - 1 - i_level) > math.log2(time_compress):
                        stride = (1, 2, 2)
                else:
                    self.time_compress *= 2
                down.downsample = Downsample(block_in, resamp_with_conv, stride=stride, conv_op=conv_op, fused_norm=fused_norm)
                curr_res = curr_res // 2
            self.down.append(down)

        if time_compress is not None:
            self.time_compress = time_compress

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op,
                                       fused_norm=fused_norm)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, conv_op=mid_attn_conv_op)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op,
                                       fused_norm=fused_norm)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_op(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        if self.carried:
            xl = [x[:, :, :1, :, :]]
            if x.shape[2] > self.time_compress:
                tc = self.time_compress
                xl += torch.split(x[:, :, 1: 1 + ((x.shape[2] - 1) // tc) * tc, :, :], tc * 2, dim = 2)
            x = xl
        else:
            x = [x]
        out = []

        conv_carry_in = None

        for i, x1 in enumerate(x):
            conv_carry_out = []
            if i == len(x) - 1:
                conv_carry_out = None

            # downsampling
            if self.fused_norm and fused_norm_served(x1.device, x1.dtype):
                x1 = x1.contiguous(memory_format=torch.channels_last)
            x1 = [ x1 ]
            h1 = conv_carry_causal_3d(x1, self.conv_in, conv_carry_in, conv_carry_out)

            for i_level in range(self.num_resolutions):
                for i_block in range(self.num_res_blocks):
                    h1 = self.down[i_level].block[i_block](h1, temb, conv_carry_in, conv_carry_out)
                    if len(self.down[i_level].attn) > 0:
                        assert i == 0 #carried should not happen if attn exists
                        h1 = self.down[i_level].attn[i_block](h1)
                if i_level != self.num_resolutions-1:
                    h1 = self.down[i_level].downsample(h1, conv_carry_in, conv_carry_out)

            out.append(h1)
            conv_carry_in = conv_carry_out

        h = torch_cat_if_needed(out, dim=2)
        del out

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        if self.fused_norm and fused_norm_served(h.device, h.dtype):
            return strip_conv(self.conv_out, h, self.norm_out)
        h = [ nonlinearity(self.norm_out(h)) ]
        h = conv_carry_causal_3d(h, self.conv_out)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, tanh_out=False, use_linear_attn=False,
                 conv_out_op=ops.Conv2d,
                 resnet_op=ResnetBlock,
                 attn_op=AttnBlock,
                 conv3d=False,
                 time_compress=None,
                 fused_norm=False,
                **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.tanh_out = tanh_out
        self.carried = False
        self.fused_norm = fused_norm = fused_norm and not conv3d

        if conv3d:
            if not attn_resolutions and resnet_op == ResnetBlock:
                conv_op = CarriedConv3d
                conv_out_op = CarriedConv3d
                self.carried = True
            else:
                conv_op = VideoConv3d
                conv_out_op = VideoConv3d

            mid_attn_conv_op = ops.Conv3d
        else:
            conv_op = ops.Conv2d
            mid_attn_conv_op = ops.Conv2d

        # compute block_in and curr_res at lowest res
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        logging.debug("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = conv_op(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = resnet_op(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op,
                                       fused_norm=fused_norm)
        self.mid.attn_1 = attn_op(block_in, conv_op=mid_attn_conv_op)
        self.mid.block_2 = resnet_op(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op,
                                       fused_norm=fused_norm)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(resnet_op(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         conv_op=conv_op,
                                         fused_norm=fused_norm))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(attn_op(block_in, conv_op=conv_op))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                scale_factor = 2.0
                if time_compress is not None:
                    if i_level > math.log2(time_compress):
                        scale_factor = (1.0, 2.0, 2.0)

                up.upsample = Upsample(block_in, resamp_with_conv, conv_op=conv_op, scale_factor=scale_factor, fused_norm=fused_norm)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_out_op(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward_fused(self, z):
        # one buffer the size of the largest upsample output holds every level: each upsample writes it from
        # the front while reading its input at the back, and the blocks take their hidden tensors from the rest
        h = self.conv_in(z.contiguous(memory_format=torch.channels_last))
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        n = self.num_resolutions
        buf = torch.empty(max(self.up[i].block[-1].out_channels * (h.shape[2] << (n - i)) * (h.shape[3] << (n - i)) for i in range(1, n)), dtype=h.dtype, device=h.device)
        for i_level in reversed(range(n)):
            level = self.up[i_level]
            hidden = None
            for i_block, block in enumerate(level.block):
                if hidden is None:
                    if block.halves_in_place(h):
                        hidden = interleaved_view(h, block.out_channels, 0)
                    else:
                        hidden = region(buf, buf.numel() - (h.numel() if i_level < n - 1 else 0), block.out_channels, h.shape[2], h.shape[3])
                h = block(h, None, hidden=hidden)
                if len(level.attn) > 0:
                    h = level.attn[i_block](h)
            if i_level != 0:
                h = level.upsample(h, out=region(buf, buf.numel(), h.shape[1], 2 * h.shape[2], 2 * h.shape[3]))
        h = strip_conv(self.conv_out, h, self.norm_out)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

    def forward(self, z, **kwargs):
        if self.fused_norm and fused_norm_served(z.device, z.dtype):
            out = [self.forward_fused(z[i:i + 1]) for i in range(z.shape[0])]
            return out[0] if len(out) == 1 else torch.cat(out)

        # timestep embedding
        temb = None

        # z to block_in
        h = conv_carry_causal_3d([z], self.conv_in)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        if self.carried:
            h = torch.split(h, 2, dim=2)
        else:
            h = [ h ]
        out = []

        conv_carry_in = None

        # upsampling
        for i, h1 in enumerate(h):
            conv_carry_out = []
            if i == len(h) - 1:
                conv_carry_out = None
            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks+1):
                    h1 = self.up[i_level].block[i_block](h1, temb, conv_carry_in, conv_carry_out, **kwargs)
                    if len(self.up[i_level].attn) > 0:
                        assert i == 0 #carried should not happen if attn exists
                        h1 = self.up[i_level].attn[i_block](h1, **kwargs)
                if i_level != 0:
                    h1 = self.up[i_level].upsample(h1, conv_carry_in, conv_carry_out)

            h1 = self.norm_out(h1)
            h1 = [ nonlinearity(h1) ]
            h1 = conv_carry_causal_3d(h1, self.conv_out, conv_carry_in, conv_carry_out)
            if self.tanh_out:
                h1 = torch.tanh(h1)
            out.append(h1)
            conv_carry_in = conv_carry_out

        out = torch_cat_if_needed(out, dim=2)

        return out
