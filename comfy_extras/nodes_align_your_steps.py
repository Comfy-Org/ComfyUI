#from: https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
import numpy as np
import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys

NOISE_LEVELS = {"SD1": [14.6146412293, 6.4745760956,  3.8636745985,  2.6946151520, 1.8841921177,  1.3943805092,  0.9642583904,  0.6523686016, 0.3977456272,  0.1515232662,  0.0291671582],
                "SDXL":[14.6146412293, 6.3184485287,  3.7681790315,  2.1811480769, 1.3405244945,  0.8620721141,  0.5550693289,  0.3798540708, 0.2332364134,  0.1114188177,  0.0291671582],
                "SVD": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002]}

class AlignYourStepsScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AlignYourStepsScheduler",
            search_aliases=["AYS scheduler"],
            category="model/sampling/schedulers",
            inputs=[
                io.Combo.Input("model_type", options=["SD1", "SDXL", "SVD"]),
                io.Int.Input("steps", default=10, min=1, max=10000),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Sigmas.Output()],
        )

    def get_sigmas(self, model_type, steps, denoise):
        # Deprecated: use the V3 schema's `execute` method instead of this.
        return AlignYourStepsScheduler().execute(model_type, steps, denoise).result

    @classmethod
    def execute(cls, model_type, steps, denoise) -> io.NodeOutput:
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return io.NodeOutput(torch.FloatTensor([]))
            total_steps = round(steps * denoise)

        sigmas = NOISE_LEVELS[model_type][:]
        if (steps + 1) != len(sigmas):
            sigmas = loglinear_interp(sigmas, steps + 1)

        sigmas = sigmas[-(total_steps + 1):]
        sigmas[-1] = 0
        return io.NodeOutput(torch.FloatTensor(sigmas))


class AlignYourStepsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AlignYourStepsScheduler,
        ]

async def comfy_entrypoint() -> AlignYourStepsExtension:
    return AlignYourStepsExtension()
