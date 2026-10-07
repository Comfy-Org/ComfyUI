"""MiniMax H3 Flow Scheduler for ComfyUI.

High-performance, shift-preserving sigma scheduler specifically engineered for
MiniMax H3 (video + audio DiT) in the 6-12 step range.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import comfy.samplers
import comfy.model_sampling


def _get_model_sigma_table(model_sampling: Any) -> Optional[torch.Tensor]:
    """Retrieve the ascending sigma table the model was configured with."""
    table = getattr(model_sampling, "sigmas", None)
    if table is not None:
        try:
            table = table.detach().to(device="cpu", dtype=torch.float32).flatten()
            if table.numel() >= 2:
                return table
        except Exception:
            pass
    return None


def calculate_h3_sigmas(
    model_sampling: Any,
    steps: int,
    mode: str = "6-12_steps_optimal",
    detail_boost: float = 0.45,
    structure_hold: float = 0.05,
) -> torch.Tensor:
    """Calculates optimal sigmas for MiniMax H3.

    Warps the model's native sigma table (shift 12.0 for video, 3.0 for audio)
    using a monotonic Kumaraswamy transformation:
        w(u) = 1 - (1 - u^a)^b
    where a controls structure/motion budget (high sigma) and b controls
    detail/terminal refinement (low sigma).

    In the 6-12 step range, stock `simple` drops ~63% of the flow in the
    final step (0.632 -> 0.0 at 8 steps). This scheduler smoothly pulls
    terminal sigma down to ~0.15-0.25, giving the DiT critical steps to
    resolve facial clarity, sharp prompt features, and clean audio.
    """
    steps = max(int(steps), 1)
    table = _get_model_sigma_table(model_sampling)

    # Preset profiles
    if mode == "turbo_lora_6-8_steps":
        # Tuned specifically for MiniMax H3 Turbo LoRA (v4-600 / 8-step)
        # Keeps steps aligned with distillation anchors while softening the terminal jump
        a = 1.0
        b = 1.25
    elif mode == "6-12_steps_optimal":
        # Recommended default: balanced prompt-following and fine detail
        a = 1.0 + 2.0 * max(0.0, min(1.0, structure_hold))
        b = 1.0 + 2.0 * max(0.0, min(1.0, detail_boost))
    elif mode == "base_model_10-16_steps":
        # For base MiniMax H3 without Turbo LoRA
        a = 1.05
        b = 1.60
    elif mode == "stock_simple":
        # Exact stock ComfyUI simple schedule
        a = 1.0
        b = 1.0
    else:
        # Custom parameters
        a = 1.0 + 2.0 * max(0.0, min(1.0, structure_hold))
        b = 1.0 + 2.0 * max(0.0, min(1.0, detail_boost))

    if table is None:
        # Fallback: construct exact shift-12 table
        n = 1000
        table = torch.empty(n, dtype=torch.float32)
        for i in range(n):
            t = (i + 1) / n
            table[i] = (12.0 * t) / (1.0 + 11.0 * t)

    n = table.numel()
    sigs = []
    neutral = (a == 1.0 and b == 1.0)

    for i in range(steps):
        u = i / steps
        w = u if neutral else 1.0 - (1.0 - (u ** a)) ** b
        idx = int(w * n + 1e-9)
        idx = max(0, min(n - 1, idx))
        sigs.append(float(table[-(1 + idx)]))

    # Guard monotonic descent
    for i in range(1, steps):
        if sigs[i] >= sigs[i - 1]:
            sigs[i] = sigs[i - 1] * 0.999

    sigs.append(0.0)
    return torch.FloatTensor(sigs)


class MiniMaxH3SchedulerNode:
    """MiniMax H3 Flow Scheduler Node for ComfyUI.

    Outputs optimized SIGMAS for SamplerCustomAdvanced, SamplerCustom, or KSampler.
    Specifically calibrated for MiniMax H3 video+audio in the 6-12 step range.
    """

    MODES = [
        "6-12_steps_optimal",
        "turbo_lora_6-8_steps",
        "base_model_10-16_steps",
        "custom",
        "stock_simple",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Connect the MiniMax H3 model (passes model_sampling carrying shift 12/3)."
                }),
                "steps": ("INT", {"default": 8, "min": 1, "max": 1000, "step": 1,
                                  "tooltip": "Optimal range for MiniMax H3 is 6-12 steps."}),
                "mode": (cls.MODES, {
                    "default": "6-12_steps_optimal",
                    "tooltip": (
                        "6-12_steps_optimal: Best overall quality & prompt following.\n"
                        "turbo_lora_6-8_steps: Aligned with Turbo LoRA distillation anchors.\n"
                        "base_model_10-16_steps: Extended detail for non-distilled base model.\n"
                        "custom: Manually tune detail_boost and structure_hold.\n"
                        "stock_simple: Stock ComfyUI simple schedule."
                    )
                }),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "detail_boost": ("FLOAT", {
                    "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Active when mode is 'custom'. Higher allocates more steps to low sigmas for finer detail."
                }),
                "structure_hold": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Active when mode is 'custom'. Higher holds high-noise steps for global composition."
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"
    DESCRIPTION = (
        "MiniMax H3 Flow Scheduler: Optimized 6-12 step schedule preserving the DiT's "
        "native shift (12 video / 3 audio) while eliminating terminal detail starvation."
    )

    def get_sigmas(
        self,
        model: Any,
        steps: int,
        mode: str,
        denoise: float = 1.0,
        detail_boost: float = 0.45,
        structure_hold: float = 0.05,
    ) -> Tuple[torch.Tensor]:
        if denoise <= 0.0:
            return (torch.FloatTensor([]),)

        total_steps = steps
        if denoise < 1.0:
            total_steps = max(int(steps / denoise), steps)

        ms = model.get_model_object("model_sampling")
        sigmas = calculate_h3_sigmas(
            ms,
            total_steps,
            mode=mode,
            detail_boost=detail_boost,
            structure_hold=structure_hold,
        )

        if denoise < 1.0:
            sigmas = sigmas[-(steps + 1):]

        return (sigmas.cpu(),)
