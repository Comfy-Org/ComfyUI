"""Ideogram 4 sampling helper
"""

import enum
import math

import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

_LOGSNR_MIN = -15.0
_LOGSNR_MAX = 18.0

class Ideogram4Enum(enum.Enum):
    QUALITY = "Quality"
    HIGH = "High"
    DEFAULT = "Default"
    FAST = "Fast"
    TURBO = "Turbo"

IDEOGRAM4_PRESET_CONFIGS = {
  Ideogram4Enum.QUALITY.value: {
    "num_steps": 48,
    "mu": 0.0,
    "std": 1.5,
    "preset_id": "V4_QUALITY_48"
  },
  Ideogram4Enum.HIGH.value: {
    "num_steps": 34,
    "mu": 0.0,
    "std": 1.6875,
    "preset_id": "V4_HIGH_34"
  },
  Ideogram4Enum.DEFAULT.value: {
    "num_steps": 20,
    "mu": 0.0,
    "std": 1.75,
    "preset_id": "V4_DEFAULT_20"
  },
  Ideogram4Enum.FAST.value: {
    "num_steps": 16,
    "mu": 0.25,
    "std": 1.8375,
    "preset_id": "V4_FAST_16"
  },
  Ideogram4Enum.TURBO.value: {
    "num_steps": 12,
    "mu": 0.5,
    "std": 1.75,
    "preset_id": "V4_TURBO_12"
  }
}

def _logit_normal_schedule(u, mean, std):
    # Reference time (0=noise..1=clean) via the probit/ndtri quantile.
    u = torch.as_tensor(u, dtype=torch.float64)
    t = 1.0 - torch.special.expit(mean + std * torch.special.ndtri(u))
    t_min = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MAX))
    t_max = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MIN))
    return t.clamp(t_min, t_max)


def ideogram4_sigmas(num_steps, width, height, mu, std):
    """Descending sigmas (len num_steps+1) for the reference schedule.

    mu + the resolution term form the logSNR shift; std is the spread.
    """
    mean = mu + 0.5 * math.log((width * height) / (512 * 512))
    u = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float64)
    sigmas = (1.0 - _logit_normal_schedule(u, mean, std)).flip(0)
    sigmas[-1] = 0.0                                      # clamp leaves ~6e-4; force full denoise
    return sigmas.to(torch.float32)


class Ideogram4Scheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Ideogram4Scheduler",
            display_name="Ideogram 4 Scheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=200),
                io.Int.Input("width", default=1024, min=256, max=8192, step=16),
                io.Int.Input("height", default=1024, min=256, max=8192, step=16),
                io.Float.Input("mu", default=0.0, min=-10.0, max=10.0, step=0.05),
                io.Float.Input("std", default=1.75, min=0.1, max=5.0, step=0.05),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, steps, width, height, mu, std) -> io.NodeOutput:
        return io.NodeOutput(ideogram4_sigmas(steps, width, height, mu, std))


class Ideogram4SchedulerPreset(Ideogram4Scheduler):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Ideogram4SchedulerPreset",
            display_name="Ideogram 4 Scheduler (Presets)",
            category="sampling/custom_sampling/schedulers",
            description="Schedule Presets for Ideogram 4. They are as follows: Quality=48, High=34, Default=20, Fast=16, Turbo=12",
            inputs=[
                io.Combo.Input("preset", options=[e.value for e in Ideogram4Enum], default=Ideogram4Enum.DEFAULT.value),
                io.Int.Input("width", default=1024, min=256, max=8192, step=16),
                io.Int.Input("height", default=1024, min=256, max=8192, step=16),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, preset, width, height) -> io.NodeOutput:
        config = IDEOGRAM4_PRESET_CONFIGS.get(preset)
        if not config:
            raise ValueError(f"Invalid preset: {preset}")

        return super().execute(
            steps=config["num_steps"],
            width=width,
            height=height,
            mu=config["mu"],
            std=config["std"]
        )


class Ideogram4Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Ideogram4Scheduler, Ideogram4SchedulerPreset]


async def comfy_entrypoint() -> Ideogram4Extension:
    return Ideogram4Extension()
