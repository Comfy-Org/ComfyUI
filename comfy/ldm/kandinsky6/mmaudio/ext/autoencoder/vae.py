import logging
from typing import Optional

import torch
import torch.nn as nn

from .edm2_utils import MPConv1D
from .vae_modules import (AttnBlock1D, Downsample1D, ResnetBlock1D,
                                                 Upsample1D, nonlinearity)
from .distributions import DiagonalGaussianDistribution

log = logging.getLogger()

DATA_MEAN_128D = [
    -3.3462, -2.6723, -2.4893, -2.3143, -2.2664, -2.3317, -2.1802, -2.4006, -2.2357, -2.4597,
    -2.3717, -2.4690, -2.5142, -2.4919, -2.6610, -2.5047, -2.7483, -2.5926, -2.7462, -2.7033,
    -2.7386, -2.8112, -2.7502, -2.9594, -2.7473, -3.0035, -2.8891, -2.9922, -2.9856, -3.0157,
    -3.1191, -2.9893, -3.1718, -3.0745, -3.1879, -3.2310, -3.1424, -3.2296, -3.2791, -3.2782,
    -3.2756, -3.3134, -3.3509, -3.3750, -3.3951, -3.3698, -3.4505, -3.4509, -3.5089, -3.4647,
    -3.5536, -3.5788, -3.5867, -3.6036, -3.6400, -3.6747, -3.7072, -3.7279, -3.7283, -3.7795,
    -3.8259, -3.8447, -3.8663, -3.9182, -3.9605, -3.9861, -4.0105, -4.0373, -4.0762, -4.1121,
    -4.1488, -4.1874, -4.2461, -4.3170, -4.3639, -4.4452, -4.5282, -4.6297, -4.7019, -4.7960,
    -4.8700, -4.9507, -5.0303, -5.0866, -5.1634, -5.2342, -5.3242, -5.4053, -5.4927, -5.5712,
    -5.6464, -5.7052, -5.7619, -5.8410, -5.9188, -6.0103, -6.0955, -6.1673, -6.2362, -6.3120,
    -6.3926, -6.4797, -6.5565, -6.6511, -6.8130, -6.9961, -7.1275, -7.2457, -7.3576, -7.4663,
    -7.6136, -7.7469, -7.8815, -8.0132, -8.1515, -8.3071, -8.4722, -8.7418, -9.3975, -9.6628,
    -9.7671, -9.8863, -9.9992, -10.0860, -10.1709, -10.5418, -11.2795, -11.3861
]

DATA_STD_128D = [
    2.3804, 2.4368, 2.3772, 2.3145, 2.2803, 2.2510, 2.2316, 2.2083, 2.1996, 2.1835, 2.1769, 2.1659,
    2.1631, 2.1618, 2.1540, 2.1606, 2.1571, 2.1567, 2.1612, 2.1579, 2.1679, 2.1683, 2.1634, 2.1557,
    2.1668, 2.1518, 2.1415, 2.1449, 2.1406, 2.1350, 2.1313, 2.1415, 2.1281, 2.1352, 2.1219, 2.1182,
    2.1327, 2.1195, 2.1137, 2.1080, 2.1179, 2.1036, 2.1087, 2.1036, 2.1015, 2.1068, 2.0975, 2.0991,
    2.0902, 2.1015, 2.0857, 2.0920, 2.0893, 2.0897, 2.0910, 2.0881, 2.0925, 2.0873, 2.0960, 2.0900,
    2.0957, 2.0958, 2.0978, 2.0936, 2.0886, 2.0905, 2.0845, 2.0855, 2.0796, 2.0840, 2.0813, 2.0817,
    2.0838, 2.0840, 2.0917, 2.1061, 2.1431, 2.1976, 2.2482, 2.3055, 2.3700, 2.4088, 2.4372, 2.4609,
    2.4731, 2.4847, 2.5072, 2.5451, 2.5772, 2.6147, 2.6529, 2.6596, 2.6645, 2.6726, 2.6803, 2.6812,
    2.6899, 2.6916, 2.6931, 2.6998, 2.7062, 2.7262, 2.7222, 2.7158, 2.7041, 2.7485, 2.7491, 2.7451,
    2.7485, 2.7233, 2.7297, 2.7233, 2.7145, 2.6958, 2.6788, 2.6439, 2.6007, 2.4786, 2.2469, 2.1877,
    2.1392, 2.0717, 2.0107, 1.9676, 1.9140, 1.7102, 0.9101, 0.7164
]


class VAE(nn.Module):

    def __init__(
        self,
        *,
        data_dim: int,
        embed_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        if data_dim != 128:
            raise ValueError(f"Only 44k audio VAE data is supported, got data_dim={data_dim}")
        self.data_mean = nn.Buffer(torch.tensor(DATA_MEAN_128D, dtype=torch.float32))
        self.data_std = nn.Buffer(torch.tensor(DATA_STD_128D, dtype=torch.float32))

        self.data_mean = self.data_mean.view(1, -1, 1)
        self.data_std = self.data_std.view(1, -1, 1)

        self.encoder = Encoder1D(
            dim=hidden_dim,
            ch_mult=(1, 2, 4),
            num_res_blocks=2,
            attn_layers=[3],
            down_layers=[0],
            in_dim=data_dim,
            embed_dim=embed_dim,
        )
        self.decoder = Decoder1D(
            dim=hidden_dim,
            ch_mult=(1, 2, 4),
            num_res_blocks=2,
            attn_layers=[3],
            down_layers=[0],
            in_dim=data_dim,
            out_dim=data_dim,
            embed_dim=embed_dim,
        )

        self.embed_dim = embed_dim
        # self.quant_conv = nn.Conv1d(2 * embed_dim, 2 * embed_dim, 1)
        # self.post_quant_conv = nn.Conv1d(embed_dim, embed_dim, 1)

    def encode(self, x: torch.Tensor, normalize: bool = True) -> DiagonalGaussianDistribution:
        if normalize:
            x = self.normalize(x)
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor, unnormalize: bool = True) -> torch.Tensor:
        dec = self.decoder(z)
        if unnormalize:
            dec = self.unnormalize(dec)
        return dec

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.data_mean) / self.data_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.data_std + self.data_mean

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        rng: Optional[torch.Generator] = None,
        normalize: bool = True,
        unnormalize: bool = True,
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:

        posterior = self.encode(x, normalize=normalize)
        if sample_posterior:
            z = posterior.sample(rng)
        else:
            z = posterior.mode()
        dec = self.decode(z, unnormalize=unnormalize)
        return dec, posterior

    def load_weights(self, src_dict) -> None:
        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def remove_weight_norm(self):
        for name, m in self.named_modules():
            if isinstance(m, MPConv1D):
                m.remove_weight_norm()
                log.debug(f"Removed weight norm from {name}")
        return self


class Encoder1D(nn.Module):

    def __init__(self,
                 *,
                 dim: int,
                 ch_mult: tuple[int] = (1, 2, 4, 8),
                 num_res_blocks: int,
                 attn_layers: list[int] = [],
                 down_layers: list[int] = [],
                 resamp_with_conv: bool = True,
                 in_dim: int,
                 embed_dim: int,
                 double_z: bool = True,
                 kernel_size: int = 3,
                 clip_act: float = 256.0):
        super().__init__()
        self.dim = dim
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_dim
        self.clip_act = clip_act
        self.down_layers = down_layers
        self.attn_layers = attn_layers
        self.conv_in = MPConv1D(in_dim, self.dim, kernel_size=kernel_size)

        in_ch_mult = (1, ) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        # downsampling
        self.down = nn.ModuleList()
        for i_level in range(self.num_layers):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = dim * in_ch_mult[i_level]
            block_out = dim * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock1D(in_dim=block_in,
                                  out_dim=block_out,
                                  kernel_size=kernel_size,
                                  use_norm=True))
                block_in = block_out
                if i_level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level in down_layers:
                down.downsample = Downsample1D(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_dim=block_in,
                                         out_dim=block_in,
                                         kernel_size=kernel_size,
                                         use_norm=True)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_dim=block_in,
                                         out_dim=block_in,
                                         kernel_size=kernel_size,
                                         use_norm=True)

        # end
        self.conv_out = MPConv1D(block_in,
                                 2 * embed_dim if double_z else embed_dim,
                                 kernel_size=kernel_size)

        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, x):

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                h = h.clamp(-self.clip_act, self.clip_act)
                hs.append(h)
            if i_level in self.down_layers:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.clamp(-self.clip_act, self.clip_act)

        # end
        h = nonlinearity(h)
        h = self.conv_out(h, gain=(self.learnable_gain + 1))
        return h

class Decoder1D(nn.Module):

    def __init__(self,
                 *,
                 dim: int,
                 out_dim: int,
                 ch_mult: tuple[int] = (1, 2, 4, 8),
                 num_res_blocks: int,
                 attn_layers: list[int] = [],
                 down_layers: list[int] = [],
                 kernel_size: int = 3,
                 resamp_with_conv: bool = True,
                 in_dim: int,
                 embed_dim: int,
                 clip_act: float = 256.0):
        super().__init__()
        self.ch = dim
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_dim
        self.clip_act = clip_act
        self.down_layers = [i + 1 for i in down_layers]  # each downlayer add one

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = dim * ch_mult[self.num_layers - 1]

        # z to block_in
        self.conv_in = MPConv1D(embed_dim, block_in, kernel_size=kernel_size)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, use_norm=True)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, use_norm=True)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_layers)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = dim * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1D(in_dim=block_in, out_dim=block_out, use_norm=True))
                block_in = block_out
                if i_level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level in self.down_layers:
                up.upsample = Upsample1D(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.conv_out = MPConv1D(block_in, out_dim, kernel_size=kernel_size)
        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.clamp(-self.clip_act, self.clip_act)

        # upsampling
        for i_level in reversed(range(self.num_layers)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                h = h.clamp(-self.clip_act, self.clip_act)
            if i_level in self.down_layers:
                h = self.up[i_level].upsample(h)

        h = nonlinearity(h)
        h = self.conv_out(h, gain=(self.learnable_gain + 1))
        return h
