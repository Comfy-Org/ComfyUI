"""PiFlow sampling for the distilled Kandinsky 6 release.

Each segment costs one DiT evaluation; the n_grid velocity grids it emits are
integrated over the segment by a network-free policy, so the distilled model
runs with the vendor's step budget instead of the base model's.
"""

import torch

import comfy.model_management as mm

# Vendor settings of the released distilled checkpoints.
PIFLOW_DEFAULTS = {
    "eps": 1e-06,
    "final_step_size_scale": 0.5,
    "num_policy_substeps": 128,
    "num_steps": 10,
    "shift": 5.0,
}


class DXPolicy:
    """Network-free DX policy over one flow-matching segment."""

    def __init__(self, denoising_output, x_t_src, sigma_t_src, segment_size=1.0, shift=1.0, eps=1e-4):
        self.ndim = x_t_src.dim()
        self.shift = shift
        self.eps = eps
        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.raw_t_src = self._unwarp_t(self.sigma_t_src)
        if isinstance(segment_size, torch.Tensor) and segment_size.dim() < self.raw_t_src.dim():
            segment_size = segment_size.reshape(*segment_size.size(), *((self.raw_t_src.dim() - segment_size.dim()) * [1]))
        self.raw_t_dst = (self.raw_t_src - segment_size).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)
        self.denoising_output_x_0 = x_t_src.unsqueeze(1) - self.sigma_t_src.unsqueeze(1) * denoising_output

    def _unwarp_t(self, sigma_t):
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    @staticmethod
    def _interpolate(x, t):
        n = x.size(1)
        if n < 2:
            return x.squeeze(1)
        t = t.clamp(min=0, max=1) * (n - 1)
        t0 = t.floor().to(torch.long).clamp(min=0, max=n - 2)
        t1 = t0 + 1
        values = torch.gather(x, dim=1, index=torch.stack([t0, t1], dim=1).expand(-1, -1, *x.shape[2:]))
        return (t1 - t) * values[:, 0] + (t - t0) * values[:, 1]

    def pi(self, x_t, sigma_t):
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        raw_t = self._unwarp_t(sigma_t)
        x_0 = self._interpolate(self.denoising_output_x_0, (raw_t - self.raw_t_dst) / self.segment_size)
        return (x_t - x_0) / sigma_t.clamp(min=self.eps)


def shift_timesteps(t, shift):
    """Map raw flow-matching time to the shifted DiT time."""
    return shift * t / (1 + (shift - 1) * t)


def policy_rollout_fm(x_t_start, sigma_t_start, raw_t_start, raw_t_end, total_substeps, policy):
    """Integrate ``policy.pi`` from ``raw_t_start`` to ``raw_t_end``."""
    num_batches = x_t_start.size(0)
    ndim = x_t_start.dim()
    shape = (num_batches, *((ndim - 1) * [1]))
    raw_t_start = raw_t_start.reshape(shape)
    raw_t_end = raw_t_end.reshape(shape)
    sigma_t = sigma_t_start.reshape(shape)

    delta_raw_t = raw_t_start - raw_t_end
    num_substeps = (delta_raw_t * total_substeps).round().to(torch.long).clamp(min=1)
    substep_size = delta_raw_t / num_substeps
    max_num_substeps = num_substeps.max()

    raw_t = raw_t_start
    x_t = x_t_start
    for substep_id in range(max_num_substeps.item()):
        velocity = policy.pi(x_t, sigma_t)
        raw_t_minus = (raw_t - substep_size).clamp(min=0)
        sigma_t_minus = shift_timesteps(raw_t_minus, policy.shift)
        x_t_minus = x_t + velocity * (sigma_t_minus - sigma_t)

        active_mask = num_substeps > substep_id
        x_t = torch.where(active_mask, x_t_minus, x_t)
        sigma_t = torch.where(active_mask, sigma_t_minus, sigma_t)
        raw_t = torch.where(active_mask, raw_t_minus, raw_t)

    return x_t, sigma_t


def rollout(dit, video, audio, context, pooled, *, steps, dtype, reference=None, transformer_options=None, callback=None):
    """Run the PiFlow segments; one DiT evaluation per segment."""
    params = PIFLOW_DEFAULTS
    eps = float(params["eps"])
    shift = float(params["shift"])
    final_scale = max(float(params["final_step_size_scale"]), eps)
    base_segment = 1.0 / (steps - (1.0 - final_scale))
    raw_src = 1.0
    batch = video.shape[0]
    device = video.device
    n_grid = dit.n_grid
    options = dict(transformer_options or {})
    options.pop("k6_magcache", None)

    for index in range(steps):
        mm.throw_exception_if_processing_interrupted()
        segment = base_segment * (final_scale if index == steps - 1 else 1.0)
        raw_dst = max(raw_src - segment, eps)
        sigma_src = shift_timesteps(torch.full((batch,), raw_src, device=device), shift)

        v_out, a_out = dit(
            [video.to(dtype), audio.to(dtype)],
            sigma_src * 1000.0,
            context=context,
            y=pooled,
            k6_reference=reference,
            transformer_options=options,
            collapse_grids=False,
        )
        video_grid = v_out.reshape(batch, n_grid, video.shape[1], *video.shape[2:])
        audio_grid = a_out.reshape(batch, audio.shape[1], n_grid, audio.shape[2]).movedim(2, 1)

        updated = []
        x0_streams = []
        for state, grid in ((video, video_grid), (audio, audio_grid)):
            sigma = sigma_src.reshape(batch, *((state.ndim - 1) * [1]))
            policy = DXPolicy(grid, state, sigma, segment, shift=shift, eps=eps)
            result, _ = policy_rollout_fm(
                state,
                sigma,
                torch.full((batch,), raw_src, device=device),
                torch.full((batch,), raw_dst, device=device),
                int(params["num_policy_substeps"]),
                policy,
            )
            updated.append(result)
            x0_streams.append(policy.denoising_output_x_0[:, 0])
        video, audio = updated
        if callback is not None:
            callback(index, (x0_streams[0], x0_streams[1]), (video, audio), steps)
        raw_src = raw_dst

    return video, audio
