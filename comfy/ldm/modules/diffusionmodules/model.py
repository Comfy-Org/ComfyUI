# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import logging

from comfy import model_management
import comfy.ops
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

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return torch.nn.functional.silu(x)


def Normalize(in_channels, num_groups=32):
    return ops.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


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
    def __init__(self, in_channels, with_conv, conv_op=ops.Conv2d, scale_factor=2.0):
        super().__init__()
        self.with_conv = with_conv
        self.scale_factor = scale_factor

        if self.with_conv:
            self.conv = conv_op(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, conv_carry_in=None, conv_carry_out=None):
        scale_factor = self.scale_factor
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor,) * (x.ndim - 2)

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


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride=2, conv_op=ops.Conv2d):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = conv_op(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=0)

    def forward(self, x, conv_carry_in=None, conv_carry_out=None):
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
                 dropout=0.0, temb_channels=512, conv_op=ops.Conv2d, norm_op=Normalize):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

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

    def forward(self, x, temb=None, conv_carry_in=None, conv_carry_out=None):
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
            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
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
                 conv3d=False, time_compress=None,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
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
                                         conv_op=conv_op))
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
                down.downsample = Downsample(block_in, resamp_with_conv, stride=stride, conv_op=conv_op)
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
                                       conv_op=conv_op)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, conv_op=mid_attn_conv_op)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op)

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
        h = self.norm_out(h)
        h = [ nonlinearity(h) ]
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
                                       conv_op=conv_op)
        self.mid.attn_1 = attn_op(block_in, conv_op=mid_attn_conv_op)
        self.mid.block_2 = resnet_op(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op)

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
                                         conv_op=conv_op))
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

                up.upsample = Upsample(block_in, resamp_with_conv, conv_op=conv_op, scale_factor=scale_factor)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_out_op(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, **kwargs):
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
