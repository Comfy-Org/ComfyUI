# Tiny AutoEncoder for HunyuanVideo and WanVideo https://github.com/madebyollin/taehv

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import namedtuple, deque

import comfy.ops
import comfy.model_management
operations=comfy.ops.disable_weight_init

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

def conv(n_in, n_out, **kwargs):
    return operations.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out, act_func):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), act_func, conv(n_out, n_out), act_func, conv(n_out, n_out))
        self.skip = operations.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act_func
    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = operations.Conv2d(n_f*stride,n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = operations.Conv2d(n_f, n_f*stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)

def apply_model_with_memblocks(model, x, parallel, show_progress_bar, output_device=None,
                               patch_size=1, decode=False):

    B, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(B*T, C, H, W)
        if not decode and patch_size > 1:
            x = F.pixel_unshuffle(x, patch_size)
        # parallel over input timesteps, iterate over blocks
        for b in tqdm(model, disable=not show_progress_bar):
            if isinstance(b, MemBlock):
                BT, C, H, W = x.shape
                T = BT // B
                _x = x.reshape(B, T, C, H, W)
                mem = F.pad(_x, (0,0,0,0,0,0,1,0), value=0)[:,:T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)
        if decode and patch_size > 1:
            x = F.pixel_shuffle(x, patch_size)
        x = x.view(B, x.shape[0] // B, *x.shape[1:])
        x = x.to(output_device)
    else:
        out = []
        # Chunk along the time dim directly (chunks are [B,1,C,H,W] views, squeeze to [B,C,H,W] views).
        # Avoids forcing a contiguous copy when x is non-contiguous (e.g. after movedim in encode/decode).
        work_queue = deque([TWorkItem(xt.squeeze(1), 0) for xt in x.chunk(T, dim=1)])
        progress_bar = tqdm(range(T), disable=not show_progress_bar)
        mem = [None] * len(model)
        while work_queue:
            xt, i = work_queue.popleft()
            if i == 0:
                progress_bar.update(1)
                if not decode and patch_size > 1:
                    xt = F.pixel_unshuffle(xt, patch_size)
            if i == len(model):
                if decode and patch_size > 1:
                    xt = F.pixel_shuffle(xt, patch_size)
                out.append(xt.to(output_device))
                del xt
            else:
                b = model[i]
                if isinstance(b, MemBlock):
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt.detach().clone()
                    else:
                        xt_new = b(xt, mem[i])
                        mem[i] = xt.detach().clone()
                    del xt
                    work_queue.appendleft(TWorkItem(xt_new, i+1))
                elif isinstance(b, TPool):
                    if mem[i] is None:
                        mem[i] = []
                    mem[i].append(xt.detach().clone())
                    if len(mem[i]) == b.stride:
                        B, C, H, W = xt.shape
                        xt = b(torch.cat(mem[i], 1).view(B*b.stride, C, H, W))
                        mem[i] = []
                        work_queue.appendleft(TWorkItem(xt, i+1))
                elif isinstance(b, TGrow):
                    xt = b(xt)
                    NT, C, H, W = xt.shape
                    for xt_next in reversed(xt.view(B, b.stride*C, H, W).chunk(b.stride, 1)):
                        work_queue.appendleft(TWorkItem(xt_next, i+1))
                    del xt
                else:
                    xt = b(xt)
                    work_queue.appendleft(TWorkItem(xt, i+1))
        progress_bar.close()
        x = torch.stack(out, 1)
    return x


class TAEHV(nn.Module):
    def __init__(self, latent_channels, parallel=False, encoder_time_downscale=(True, True, False), decoder_time_upscale=(False, True, True), decoder_space_upscale=(True, True, True),
                 latent_format=None, show_progress_bar=False):
        super().__init__()
        self.image_channels = 3
        self.patch_size = 1
        self.latent_channels = latent_channels
        self.parallel = parallel
        self.latent_format = latent_format
        self.show_progress_bar = show_progress_bar
        self.process_in = latent_format().process_in if latent_format is not None else (lambda x: x)
        self.process_out = latent_format().process_out if latent_format is not None else (lambda x: x)
        if self.latent_channels in [48, 32]: # Wan 2.2 and HunyuanVideo1.5
            self.patch_size = 2
        elif self.latent_channels == 128: # LTX2
            self.patch_size, self.latent_channels, encoder_time_downscale, decoder_time_upscale = 4, 128, (True, True, True), (True, True, True)

        if self.latent_channels == 32: # HunyuanVideo1.5
            act_func = nn.LeakyReLU(0.2, inplace=True)
        else: # HunyuanVideo, Wan 2.1
            act_func = nn.ReLU(inplace=True)

        self.encoder = nn.Sequential(
            conv(self.image_channels*self.patch_size**2, 64), act_func,
            TPool(64, 2 if encoder_time_downscale[0] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64, act_func), MemBlock(64, 64, act_func), MemBlock(64, 64, act_func),
            TPool(64, 2 if encoder_time_downscale[1] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64, act_func), MemBlock(64, 64, act_func), MemBlock(64, 64, act_func),
            TPool(64, 2 if encoder_time_downscale[2] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64, act_func), MemBlock(64, 64, act_func), MemBlock(64, 64, act_func),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]

        self.decoder = nn.Sequential(
            Clamp(), conv(self.latent_channels, n_f[0]), act_func,
            MemBlock(n_f[0], n_f[0], act_func), MemBlock(n_f[0], n_f[0], act_func), MemBlock(n_f[0], n_f[0], act_func), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1], act_func), MemBlock(n_f[1], n_f[1], act_func), MemBlock(n_f[1], n_f[1], act_func), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2], act_func), MemBlock(n_f[2], n_f[2], act_func), MemBlock(n_f[2], n_f[2], act_func), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
            act_func, conv(n_f[3], self.image_channels*self.patch_size**2),
        )

        self.t_downscale = 2**sum(t.stride == 2 for t in self.encoder if isinstance(t, TPool))
        self.t_upscale = 2**sum(t.stride == 2 for t in self.decoder if isinstance(t, TGrow))
        self.frames_to_trim = self.t_upscale - 1
        self._show_progress_bar = show_progress_bar

    @property
    def show_progress_bar(self):
        return self._show_progress_bar

    @show_progress_bar.setter
    def show_progress_bar(self, value):
        self._show_progress_bar = value

    def encode(self, x, **kwargs):
        x = x.movedim(2, 1)  # [B, C, T, H, W] -> [B, T, C, H, W]
        if x.shape[1] % self.t_downscale != 0:
            # pad at end to multiple of t_downscale
            n_pad = self.t_downscale - x.shape[1] % self.t_downscale
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        x = apply_model_with_memblocks(self.encoder, x, self.parallel, self.show_progress_bar,
                                        patch_size=self.patch_size).movedim(2, 1)
        return self.process_out(x)

    def decode(self, x, **kwargs):
        x = x.unsqueeze(0) if x.ndim == 4 else x  # [T, C, H, W] -> [1, T, C, H, W]
        x = x.movedim(1, 2) if x.shape[1] != self.latent_channels else x  # [B, T, C, H, W] or [B, C, T, H, W]
        x = self.process_in(x).movedim(2, 1)  # [B, C, T, H, W] -> [B, T, C, H, W]
        x = apply_model_with_memblocks(self.decoder, x, self.parallel, self.show_progress_bar,
                                        output_device=comfy.model_management.intermediate_device(),
                                        patch_size=self.patch_size, decode=True)
        return x[:, self.frames_to_trim:].movedim(2, 1)
