"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from torch import nn
from torch.autograd import Function
import comfy.ops

ops = comfy.ops.disable_weight_init


class vector_quantize(Function):
    @staticmethod
    def forward(ctx, x, codebook):
        with torch.no_grad():
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            x_sqr = torch.sum(x ** 2, dim=1, keepdim=True)

            dist = torch.addmm(codebook_sqr + x_sqr, x, codebook.t(), alpha=-2.0, beta=1.0)
            _, indices = dist.min(dim=1)

            ctx.save_for_backward(indices, codebook)
            ctx.mark_non_differentiable(indices)

            nn = torch.index_select(codebook, 0, indices)
            return nn, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors

            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output)

        return (grad_inputs, grad_codebook)


class VectorQuantize(nn.Module):
    def __init__(self, embedding_size, k, ema_decay=0.99, ema_loss=False):
        """
        Takes an input of variable size (as long as the last dimension matches the embedding size).
        Returns one tensor containing the nearest neigbour embeddings to each of the inputs,
        with the same size as the input, vq and commitment components for the loss as a touple
        in the second output and the indices of the quantized vectors in the third:
        quantized, (vq_loss, commit_loss), indices
        """
        super(VectorQuantize, self).__init__()

        self.codebook = nn.Embedding(k, embedding_size)
        self.codebook.weight.data.uniform_(-1./k, 1./k)
        self.vq = vector_quantize.apply

        self.ema_decay = ema_decay
        self.ema_loss = ema_loss
        if ema_loss:
            self.register_buffer('ema_element_count', torch.ones(k))
            self.register_buffer('ema_weight_sum', torch.zeros_like(self.codebook.weight))

    def _laplace_smoothing(self, x, epsilon):
        n = torch.sum(x)
        return ((x + epsilon) / (n + x.size(0) * epsilon) * n)

    def _updateEMA(self, z_e_x, indices):
        mask = nn.functional.one_hot(indices, self.ema_element_count.size(0)).float()
        elem_count = mask.sum(dim=0)
        weight_sum = torch.mm(mask.t(), z_e_x)

        self.ema_element_count = (self.ema_decay * self.ema_element_count) + ((1-self.ema_decay) * elem_count)
        self.ema_element_count = self._laplace_smoothing(self.ema_element_count, 1e-5)
        self.ema_weight_sum = (self.ema_decay * self.ema_weight_sum) + ((1-self.ema_decay) * weight_sum)

        self.codebook.weight.data = self.ema_weight_sum / self.ema_element_count.unsqueeze(-1)

    def idx2vq(self, idx, dim=-1):
        q_idx = self.codebook(idx)
        if dim != -1:
            q_idx = q_idx.movedim(-1, dim)
        return q_idx

    def forward(self, x, get_losses=True, dim=-1):
        if dim != -1:
            x = x.movedim(dim, -1)
        z_e_x = x.contiguous().view(-1, x.size(-1)) if len(x.shape) > 2 else x
        z_q_x, indices = self.vq(z_e_x, self.codebook.weight.detach())
        vq_loss, commit_loss = None, None
        if self.ema_loss and self.training:
            self._updateEMA(z_e_x.detach(), indices.detach())
        # pick the graded embeddings after updating the codebook in order to have a more accurate commitment loss
        z_q_x_grd = torch.index_select(self.codebook.weight, dim=0, index=indices)
        if get_losses:
            vq_loss = (z_q_x_grd - z_e_x.detach()).pow(2).mean()
            commit_loss = (z_e_x - z_q_x_grd.detach()).pow(2).mean()

        z_q_x = z_q_x.view(x.shape)
        if dim != -1:
            z_q_x = z_q_x.movedim(-1, dim)
        return z_q_x, (vq_loss, commit_loss), indices.view(x.shape[:-1])


class ResBlock(nn.Module):
    def __init__(self, c, c_hidden):
        super().__init__()
        # depthwise/attention
        self.norm1 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.depthwise = nn.Sequential(
            nn.ReplicationPad2d(1),
            ops.Conv2d(c, c, kernel_size=3, groups=c)
        )

        # channelwise
        self.norm2 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            ops.Linear(c, c_hidden),
            nn.GELU(),
            ops.Linear(c_hidden, c),
        )

        self.gammas = nn.Parameter(torch.zeros(6), requires_grad=False)

    def _norm(self, x, norm):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x):
        mods = self.gammas

        x_temp = self._norm(x, self.norm1) * (1 + mods[0]) + mods[1]
        try:
            x = x + self.depthwise(x_temp) * mods[2]
        except: #operation not implemented for bf16
            x_temp = self.depthwise[0](x_temp.float()).to(x.dtype)
            x = x + self.depthwise[1](x_temp) * mods[2]

        x_temp = self._norm(x, self.norm2) * (1 + mods[3]) + mods[4]
        x = x + self.channelwise(x_temp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * mods[5]

        return x


class StageA(nn.Module):
    def __init__(self, levels=2, bottleneck_blocks=12, c_hidden=384, c_latent=4, codebook_size=8192):
        super().__init__()
        self.c_latent = c_latent
        c_levels = [c_hidden // (2 ** i) for i in reversed(range(levels))]

        # Encoder blocks
        self.in_block = nn.Sequential(
            nn.PixelUnshuffle(2),
            ops.Conv2d(3 * 4, c_levels[0], kernel_size=1)
        )
        down_blocks = []
        for i in range(levels):
            if i > 0:
                down_blocks.append(ops.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            block = ResBlock(c_levels[i], c_levels[i] * 4)
            down_blocks.append(block)
        down_blocks.append(nn.Sequential(
            ops.Conv2d(c_levels[-1], c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
        ))
        self.down_blocks = nn.Sequential(*down_blocks)
        self.down_blocks[0]

        self.codebook_size = codebook_size
        self.vquantizer = VectorQuantize(c_latent, k=codebook_size)

        # Decoder blocks
        up_blocks = [nn.Sequential(
            ops.Conv2d(c_latent, c_levels[-1], kernel_size=1)
        )]
        for i in range(levels):
            for j in range(bottleneck_blocks if i == 0 else 1):
                block = ResBlock(c_levels[levels - 1 - i], c_levels[levels - 1 - i] * 4)
                up_blocks.append(block)
            if i < levels - 1:
                up_blocks.append(
                    ops.ConvTranspose2d(c_levels[levels - 1 - i], c_levels[levels - 2 - i], kernel_size=4, stride=2,
                                       padding=1))
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_block = nn.Sequential(
            ops.Conv2d(c_levels[0], 3 * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    def encode(self, x, quantize=False):
        x = self.in_block(x)
        x = self.down_blocks(x)
        if quantize:
            qe, (vq_loss, commit_loss), indices = self.vquantizer.forward(x, dim=1)
            return qe, x, indices, vq_loss + commit_loss * 0.25
        else:
            return x

    def decode(self, x):
        x = self.up_blocks(x)
        x = self.out_block(x)
        return x

    def forward(self, x, quantize=False):
        qe, x, _, vq_loss = self.encode(x, quantize)
        x = self.decode(qe)
        return x, vq_loss


class Discriminator(nn.Module):
    def __init__(self, c_in=3, c_cond=0, c_hidden=512, depth=6):
        super().__init__()
        d = max(depth - 3, 3)
        layers = [
            nn.utils.spectral_norm(ops.Conv2d(c_in, c_hidden // (2 ** d), kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        ]
        for i in range(depth - 1):
            c_in = c_hidden // (2 ** max((d - i), 0))
            c_out = c_hidden // (2 ** max((d - 1 - i), 0))
            layers.append(nn.utils.spectral_norm(ops.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(*layers)
        self.shuffle = ops.Conv2d((c_hidden + c_cond) if c_cond > 0 else c_hidden, 1, kernel_size=1)
        self.logits = nn.Sigmoid()

    def forward(self, x, cond=None):
        x = self.encoder(x)
        if cond is not None:
            cond = cond.view(cond.size(0), cond.size(1), 1, 1, ).expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, cond], dim=1)
        x = self.shuffle(x)
        x = self.logits(x)
        return x
