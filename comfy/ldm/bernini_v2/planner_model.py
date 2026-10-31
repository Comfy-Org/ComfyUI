# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Native PyTorch modules for Bernini v2 semantic planning.

The parameter layout matches ByteDance/Bernini exactly. The implementation is
adapted from the official Apache-2.0 source and cross-checked against the
ComfyUI-oriented reference published in rzgar/Bernini-v2-ComfyUI.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLPConnector(nn.Module):
    def __init__(
        self,
        in_dim: int = 3584,
        out_dim_for_gen: int = 4096,
        out_dim_for_vit: int = 3584,
        *,
        device=None,
        dtype=None,
        operations,
    ):
        super().__init__()
        self.proj_gen = nn.Sequential(
            operations.Linear(in_dim, out_dim_for_gen, device=device, dtype=dtype),
            nn.GELU(),
            operations.RMSNorm(out_dim_for_gen, eps=1e-6, device=device, dtype=dtype),
            operations.Linear(
                out_dim_for_gen, out_dim_for_gen, device=device, dtype=dtype
            ),
        )
        self.pred_vit = nn.Sequential(
            operations.Linear(in_dim, out_dim_for_vit, device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(
                out_dim_for_vit, out_dim_for_vit, device=device, dtype=dtype
            ),
            operations.RMSNorm(out_dim_for_vit, eps=1e-6, device=device, dtype=dtype),
            operations.Linear(
                out_dim_for_vit, out_dim_for_vit, device=device, dtype=dtype
            ),
        )

    def for_gen(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_gen(x)

    def for_vit(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred_vit(x)


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        *,
        device=None,
        dtype=None,
        operations,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
                device=device,
                dtype=dtype,
            ),
            nn.SiLU(),
            operations.Linear(
                hidden_size, hidden_size, bias=True, device=device, dtype=dtype
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        frequency = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(frequency.to(t.dtype))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, channels: int, *, device=None, dtype=None, operations):
        super().__init__()
        self.in_ln = operations.LayerNorm(
            channels, eps=1e-6, device=device, dtype=dtype
        )
        self.mlp = nn.Sequential(
            operations.Linear(
                channels, channels, bias=True, device=device, dtype=dtype
            ),
            nn.SiLU(),
            operations.Linear(
                channels, channels, bias=True, device=device, dtype=dtype
            ),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(
                channels, 3 * channels, bias=True, device=device, dtype=dtype
            ),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        hidden = self.mlp(modulate(self.in_ln(x), shift, scale))
        return x + gate * hidden


class FinalLayer(nn.Module):
    def __init__(
        self,
        model_channels: int,
        out_channels: int,
        *,
        device=None,
        dtype=None,
        operations,
    ):
        super().__init__()
        self.norm_final = operations.LayerNorm(
            model_channels,
            elementwise_affine=False,
            eps=1e-6,
            device=device,
            dtype=dtype,
        )
        self.linear = operations.Linear(
            model_channels, out_channels, bias=True, device=device, dtype=dtype
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(
                model_channels,
                2 * model_channels,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        *,
        device=None,
        dtype=None,
        operations,
    ):
        super().__init__()
        settings = {"device": device, "dtype": dtype, "operations": operations}
        self.in_channels = in_channels
        self.time_embed = TimestepEmbedder(model_channels, **settings)
        self.cond_embed = operations.Linear(
            z_channels, model_channels, device=device, dtype=dtype
        )
        self.input_proj = operations.Linear(
            in_channels, model_channels, device=device, dtype=dtype
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(model_channels, **settings) for _ in range(num_res_blocks)]
        )
        self.final_layer = FinalLayer(model_channels, out_channels, **settings)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_proj(x)
        condition = self.time_embed(t) + self.cond_embed(c)
        for block in self.res_blocks:
            x = block(x, condition)
        return self.final_layer(x, condition)

    def forward_with_cfg(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        half = x[: len(x) // 2]
        output = self.forward(torch.cat([half, half], dim=0), t, c)
        cond, uncond = torch.split(
            output[:, : self.in_channels], len(output) // 2, dim=0
        )
        guided = uncond + cfg_scale * (cond - uncond)
        return torch.cat([guided, guided], dim=0)

    def forward_with_txt_img_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        txt_cfg_scale: float,
        img_cfg_scale: float,
    ) -> torch.Tensor:
        part = x[: len(x) // 3]
        output = self.forward(torch.cat([part, part, part], dim=0), t, c)
        cond, uncond, imgcond = torch.split(
            output[:, : self.in_channels], len(output) // 3, dim=0
        )
        guided = (
            uncond
            + img_cfg_scale * (imgcond - uncond)
            + txt_cfg_scale * (cond - imgcond)
        )
        return torch.cat([guided, guided, guided], dim=0)


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.sigmas = torch.empty(0)
        self.timesteps = torch.empty(0)
        self.set_timesteps(num_inference_steps, device="cpu")

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        shift: float | None = None,
        device=None,
    ) -> None:
        if shift is not None:
            self.shift = shift
        device = device or "cpu"
        sigma_start = (
            self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        )
        count = num_inference_steps + 1 if self.extra_one_step else num_inference_steps
        sigmas = torch.linspace(
            sigma_start,
            self.sigma_min,
            count,
            device=device,
            dtype=torch.float32,
        )
        if self.extra_one_step:
            sigmas = sigmas[:-1]
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        to_final: bool = False,
    ) -> torch.Tensor:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=self.timesteps.device)
        else:
            timestep = timestep.to(self.timesteps.device)
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            next_sigma = sample.new_zeros(())
        else:
            next_sigma = self.sigmas[timestep_id + 1]
        return sample + model_output * (next_sigma - sigma)


class DiffLossFM(nn.Module):
    def __init__(
        self,
        target_channels: int = 3584,
        z_channels: int = 3584,
        depth: int = 16,
        width: int = 4096,
        shift: float = 2.0,
        extra_one_step: bool = True,
        *,
        device=None,
        dtype=None,
        operations,
    ):
        super().__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            target_channels,
            width,
            target_channels,
            z_channels,
            depth,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.scheduler_shift = shift
        self.scheduler_extra_one_step = extra_one_step

    def sample(
        self,
        z: torch.Tensor,
        *,
        cfg: float,
        num_inference_steps: int,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        img_cfg: float | None = None,
    ) -> torch.Tensor:
        device = z.device
        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(
                0 if seed is None else seed
            )
        branch_count = 3 if img_cfg is not None else 2 if cfg > 1.0 else 1
        noise = torch.randn(
            z.shape[0] // branch_count, self.in_channels, generator=generator
        )
        samples = torch.cat([noise] * branch_count, dim=0).to(
            device=device, dtype=z.dtype
        )

        if branch_count == 3:
            sample_fn = self.net.forward_with_txt_img_cfg
            kwargs = {"c": z, "txt_cfg_scale": cfg, "img_cfg_scale": img_cfg}
        elif branch_count == 2:
            sample_fn = self.net.forward_with_cfg
            kwargs = {"c": z, "cfg_scale": cfg}
        else:
            sample_fn = self.net.forward
            kwargs = {"c": z}

        scheduler = FlowMatchScheduler(
            shift=self.scheduler_shift,
            extra_one_step=self.scheduler_extra_one_step,
        )
        scheduler.set_timesteps(num_inference_steps, device=device)
        for timestep in scheduler.timesteps:
            prediction = sample_fn(samples, timestep.unsqueeze(0).to(z.dtype), **kwargs)
            samples = scheduler.step(prediction, timestep, samples)
        return samples
