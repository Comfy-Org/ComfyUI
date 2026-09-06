# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Bernini v2's flow-prediction UniPC BH2 solver.

The update equations are adapted from Diffusers' Apache-2.0 licensed
``UniPCMultistepScheduler``.  They intentionally operate on flow sigmas where
``alpha = 1 - sigma``; ComfyUI's generic UniPC sampler instead assumes a VP
noise schedule and is therefore not interchangeable with the released model.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _alpha_sigma(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return 1.0 - sigma, sigma


def _lambda(sigma: torch.Tensor) -> torch.Tensor:
    alpha, sigma = _alpha_sigma(sigma)
    return torch.log(alpha) - torch.log(sigma)


def _coefficients(
    h: torch.Tensor, order: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the UniPC BH2 Vandermonde system and right-hand side."""
    hh = -h
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1.0
    b_h = torch.expm1(hh)
    factorial_i = 1
    values = []
    # The caller supplies the actual r_k values for the rows.  This helper only
    # builds b; keeping it separate makes the flow-specific sign explicit.
    for index in range(1, order + 1):
        values.append(h_phi_k * factorial_i / b_h)
        factorial_i *= index + 1
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i
    return h_phi_1, torch.stack(values).to(device=device)


def _uni_p_bh2_update(
    sample: torch.Tensor,
    model_outputs: list[torch.Tensor | None],
    sigmas: torch.Tensor,
    step_index: int,
    order: int,
) -> torch.Tensor:
    """UniP predictor for an x0-prediction representation."""
    m0 = model_outputs[-1]
    if m0 is None:
        raise RuntimeError("UniPC predictor is missing the current model output")

    sigma_t = sigmas[step_index + 1].to(device=sample.device, dtype=torch.float32)
    sigma_s0 = sigmas[step_index].to(device=sample.device, dtype=torch.float32)
    alpha_t, sigma_t = _alpha_sigma(sigma_t)
    _, sigma_s0 = _alpha_sigma(sigma_s0)
    h = _lambda(sigmas[step_index + 1].to(sample.device)) - _lambda(
        sigmas[step_index].to(sample.device)
    )

    rks = []
    d1s = []
    for history_index in range(1, order):
        previous_step = step_index - history_index
        previous = model_outputs[-(history_index + 1)]
        if previous is None:
            raise RuntimeError("UniPC predictor history is incomplete")
        rk = (
            _lambda(sigmas[previous_step].to(sample.device))
            - _lambda(sigmas[step_index].to(sample.device))
        ) / h
        rks.append(rk)
        d1s.append((previous - m0) / rk)
    rks.append(torch.ones((), device=sample.device, dtype=h.dtype))
    rks_tensor = torch.stack(rks)

    h_phi_1, b = _coefficients(h, order, sample.device)
    b_h = torch.expm1(-h)
    rows = torch.stack([rks_tensor**index for index in range(order)])

    pred_res: torch.Tensor | float = 0.0
    if d1s:
        stacked = torch.stack(d1s, dim=1)
        if order == 2:
            rhos_p = torch.tensor([0.5], dtype=sample.dtype, device=sample.device)
        else:  # Kept for clarity if the published solver order ever changes.
            rhos_p = torch.linalg.solve(rows[:-1, :-1], b[:-1]).to(sample.dtype)
        pred_res = torch.einsum("k,bkc...->bc...", rhos_p, stacked)

    result = sigma_t / sigma_s0 * sample - alpha_t * h_phi_1 * m0
    result = result - alpha_t * b_h * pred_res
    return result.to(sample.dtype)


def _uni_c_bh2_update(
    this_model_output: torch.Tensor,
    last_sample: torch.Tensor,
    this_sample: torch.Tensor,
    model_outputs: list[torch.Tensor | None],
    sigmas: torch.Tensor,
    step_index: int,
    order: int,
) -> torch.Tensor:
    """UniC corrector for an x0-prediction representation."""
    m0 = model_outputs[-1]
    if m0 is None:
        raise RuntimeError("UniPC corrector is missing its previous model output")

    sigma_t_value = sigmas[step_index].to(
        device=this_sample.device, dtype=torch.float32
    )
    sigma_s0_value = sigmas[step_index - 1].to(
        device=this_sample.device, dtype=torch.float32
    )
    alpha_t, sigma_t = _alpha_sigma(sigma_t_value)
    _, sigma_s0 = _alpha_sigma(sigma_s0_value)
    h = _lambda(sigma_t_value) - _lambda(sigma_s0_value)

    rks = []
    d1s = []
    for history_index in range(1, order):
        previous_step = step_index - (history_index + 1)
        previous = model_outputs[-(history_index + 1)]
        if previous is None:
            raise RuntimeError("UniPC corrector history is incomplete")
        rk = (
            _lambda(sigmas[previous_step].to(this_sample.device))
            - _lambda(sigma_s0_value)
        ) / h
        rks.append(rk)
        d1s.append((previous - m0) / rk)
    rks.append(torch.ones((), device=this_sample.device, dtype=h.dtype))
    rks_tensor = torch.stack(rks)

    h_phi_1, b = _coefficients(h, order, this_sample.device)
    b_h = torch.expm1(-h)
    rows = torch.stack([rks_tensor**index for index in range(order)])
    if order == 1:
        rhos_c = torch.tensor([0.5], dtype=last_sample.dtype, device=this_sample.device)
    else:
        rhos_c = torch.linalg.solve(rows, b).to(last_sample.dtype)

    corr_res: torch.Tensor | float = 0.0
    if d1s:
        corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], torch.stack(d1s, dim=1))
    d1_t = this_model_output - m0
    result = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
    result = result - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
    return result.to(this_sample.dtype)


def sample_flow_unipc_bh2(
    model: Callable,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict | None = None,
    callback: Callable | None = None,
    disable: bool = False,
) -> torch.Tensor:
    """Sample with the released order-2 flow UniPC predictor/corrector.

    ``model`` follows Comfy's sampler contract and returns denoised/x0.  Comfy's
    sampler wrapper pre-multiplies the initial noise by the first flow sigma, so
    it is divided back out to match Diffusers' unit-normal initial latent.
    """
    del disable
    extra_args = {} if extra_args is None else extra_args
    if len(sigmas) <= 1:
        return noise

    if (
        not bool(torch.isfinite(sigmas).all())
        or bool((sigmas < 0).any())
        or bool((sigmas >= 1).any())
    ):
        raise ValueError("flow UniPC requires finite sigmas in [0, 1)")
    if bool((sigmas[:-1] <= sigmas[1:]).any()):
        raise ValueError("flow UniPC requires strictly decreasing sigmas")

    first_sigma = sigmas[0].to(device=noise.device, dtype=noise.dtype)
    sample = noise / first_sigma
    model_outputs: list[torch.Tensor | None] = [None, None]
    last_sample = None
    lower_order_nums = 0
    previous_order = 1
    total_steps = len(sigmas) - 1

    for step_index in range(total_steps):
        sigma = sigmas[step_index].to(device=sample.device)
        sigma_batch = sigma * sample.new_ones([sample.shape[0]])
        denoised = model(sample, sigma_batch, **extra_args)

        if step_index > 0 and last_sample is not None:
            sample = _uni_c_bh2_update(
                denoised,
                last_sample,
                sample,
                model_outputs,
                sigmas,
                step_index,
                previous_order,
            )

        model_outputs[0] = model_outputs[1]
        model_outputs[1] = denoised
        this_order = min(2, total_steps - step_index, lower_order_nums + 1)
        last_sample = sample
        sample = _uni_p_bh2_update(
            sample, model_outputs, sigmas, step_index, this_order
        )
        previous_order = this_order
        lower_order_nums = min(lower_order_nums + 1, 2)

        if callback is not None:
            callback(
                {
                    "x": sample,
                    "i": step_index,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                }
            )
    return sample
