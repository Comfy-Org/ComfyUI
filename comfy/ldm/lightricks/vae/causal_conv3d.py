from typing import Tuple, Union

import threading
import torch
import torch.nn as nn
import comfy.ops
ops = comfy.ops.disable_weight_init

class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: Union[int, Tuple[int]] = 1,
        dilation: int = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(stride, int):
            self.time_stride = stride
        else:
            self.time_stride = stride[0]

        kernel_size = (kernel_size, kernel_size, kernel_size)
        self.time_kernel_size = kernel_size[0]

        dilation = (dilation, 1, 1)

        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = ops.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=spatial_padding_mode,
            groups=groups,
        )
        self.temporal_cache_state={}

    def forward(self, x, causal: bool = True):
        tid = threading.get_ident()

        cached, is_end = self.temporal_cache_state.get(tid, (None, False))
        if cached is None:
            padding_length = self.time_kernel_size - 1
            if not causal:
                padding_length = padding_length // 2
            if x.shape[2] == 0:
                return x
            cached = x[:, :, :1, :, :].repeat((1, 1, padding_length, 1, 1))
        pieces = [ cached, x ]
        if is_end and not causal:
            pieces.append(x[:, :, -1:, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1)))
        input_length = sum([piece.shape[2] for piece in pieces])
        cache_length = (self.time_kernel_size - self.time_stride) + ((input_length - self.time_kernel_size) % self.time_stride)

        needs_caching = not is_end
        if needs_caching and cache_length == 0:
            self.temporal_cache_state[tid] = (x[:, :, :0, :, :], False)
            needs_caching = False
        if needs_caching and x.shape[2] >= cache_length:
            needs_caching = False
            self.temporal_cache_state[tid] = (x[:, :, -cache_length:, :, :], False)

        x = torch.cat(pieces, dim=2)
        del pieces
        del cached

        if needs_caching:
            self.temporal_cache_state[tid] = (x[:, :, -cache_length:, :, :], False)
        elif is_end:
            self.temporal_cache_state[tid] = (None, True)

        return self.conv(x) if x.shape[2] >= self.time_kernel_size else x[:, :, :0, :, :]

    @property
    def weight(self):
        return self.conv.weight
