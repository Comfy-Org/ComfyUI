"""ComfyUI-OmniFlow: MiniMax H3 Dedicated Scheduler and Sampler.

1 dedicated Scheduler and 1 dedicated Sampler engineered specifically for MiniMax H3
video + synchronized audio generation in the 6-12 step range.
"""

from __future__ import annotations

import comfy.samplers
from .scheduler import (
    MiniMaxH3SchedulerNode,
    calculate_h3_sigmas,
)
from .sampler import (
    MiniMaxH3SamplerNode,
    MiniMaxH3Sampler,
    sample_minimax_h3_flow2m,
)

# Export the single dedicated scheduler and single dedicated sampler nodes
NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Scheduler": MiniMaxH3SchedulerNode,
    "MiniMaxH3Sampler": MiniMaxH3SamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Scheduler": "MiniMax H3 Flow Scheduler (6-12 Steps)",
    "MiniMaxH3Sampler": "MiniMax H3 Flow Sampler (Flow-2M)",
}

# --- Register into ComfyUI's core scheduler dictionary ---
try:
    def _h3_scheduler_handler(model_sampling, steps):
        return calculate_h3_sigmas(model_sampling, steps, mode="6-12_steps_optimal")

    def _h3_turbo_handler(model_sampling, steps):
        return calculate_h3_sigmas(model_sampling, steps, mode="turbo_lora_6-8_steps")

    sched_map = {
        "h3_flow_optimal": _h3_scheduler_handler,
        "h3_flow_turbo": _h3_turbo_handler,
    }

    if hasattr(comfy.samplers, "SCHEDULER_HANDLERS") and hasattr(comfy.samplers, "SchedulerHandler"):
        for name, handler_fn in sched_map.items():
            if name not in comfy.samplers.SCHEDULER_HANDLERS:
                comfy.samplers.SCHEDULER_HANDLERS[name] = comfy.samplers.SchedulerHandler(
                    handler_fn, use_ms=True
                )
            if name not in comfy.samplers.SCHEDULER_NAMES:
                comfy.samplers.SCHEDULER_NAMES.append(name)

    if hasattr(comfy.samplers, "KSampler") and hasattr(comfy.samplers.KSampler, "SCHEDULERS"):
        for name in sched_map:
            if name not in comfy.samplers.KSampler.SCHEDULERS:
                comfy.samplers.KSampler.SCHEDULERS.append(name)

except Exception as e:
    print(f"[MiniMax-H3] Core scheduler registration note: {e}")


# --- Register into ComfyUI's core sampler dictionary ---
# Makes the sampler appear directly in KSamplerSelect and KSampler dropdown menus!
try:
    SAMPLER_NAME = "minimax_h3_flow2m"

    if SAMPLER_NAME not in comfy.samplers.SAMPLER_NAMES:
        comfy.samplers.SAMPLER_NAMES.append(SAMPLER_NAME)
    if hasattr(comfy.samplers, "KSAMPLER_NAMES") and SAMPLER_NAME not in comfy.samplers.KSAMPLER_NAMES:
        comfy.samplers.KSAMPLER_NAMES.append(SAMPLER_NAME)
    if hasattr(comfy.samplers, "KSampler") and hasattr(comfy.samplers.KSampler, "SAMPLERS"):
        if SAMPLER_NAME not in comfy.samplers.KSampler.SAMPLERS:
            comfy.samplers.KSampler.SAMPLERS.append(SAMPLER_NAME)

    _orig_sampler_object = comfy.samplers.sampler_object

    def _hooked_sampler_object(name):
        if name == SAMPLER_NAME:
            return MiniMaxH3Sampler(damp=0.15, max_extrap=0.75)
        return _orig_sampler_object(name)

    comfy.samplers.sampler_object = _hooked_sampler_object

except Exception as e:
    print(f"[MiniMax-H3] Core sampler registration note: {e}")


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MiniMaxH3SchedulerNode",
    "MiniMaxH3SamplerNode",
    "MiniMaxH3Sampler",
    "sample_minimax_h3_flow2m",
]
