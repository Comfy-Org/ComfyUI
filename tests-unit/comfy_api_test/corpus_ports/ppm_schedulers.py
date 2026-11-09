"""ComfyUI-ppm's schedulers (src/schedulers.py), ported to the V2 API.

The original updates ``comfy.samplers.SCHEDULER_HANDLERS`` at import and imports ``simple_scheduler``,
``beta_scheduler`` and ``loglinear_interp`` from ComfyUI. The port declares its schedulers
(``ComfyExtension.get_schedulers``) and carries those three small helpers, written against the
model's sigma table (``model_sampling["sigmas"]``) instead of the engine's object.
"""
from functools import partial

import numpy as np
import scipy.stats
import torch

from comfy_api.index import ComfyExtension


def _simple(model_sampling, steps):
    sigmas_table = model_sampling["sigmas"]
    sigs = []
    ss = len(sigmas_table) / steps
    for x in range(steps):
        sigs += [float(sigmas_table[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def _beta(model_sampling, steps, alpha=0.6, beta=0.6):
    sigmas_table = model_sampling["sigmas"]
    total_timesteps = len(sigmas_table) - 1
    ts = 1 - np.linspace(0, 1, steps, endpoint=False)
    ts = np.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)
    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(sigmas_table[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def _loglinear_interp(t_steps, num_steps):
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    return np.exp(new_ys)[::-1].copy()


def _ays_scheduler(model_sampling, steps, force_sigma_min=False, model_type="SDXL"):
    timestep_indices = {
        "SDXL": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0],
        "SDXL_30": [999, 953, 904, 850, 813, 777, 738, 695, 650, 602, 556, 510, 462, 417, 374, 331, 290, 250, 214, 182, 155, 131, 108, 85, 66, 49, 32, 20, 12, 3, 0],
    }  # fmt: skip
    indices = timestep_indices[model_type]
    sigmas = _simple(model_sampling, 1000).flip(0)[1:][indices]
    sigmas = _loglinear_interp(sigmas.tolist(), steps + 1 if not force_sigma_min else steps)
    sigmas = torch.FloatTensor(sigmas)
    sigmas = torch.cat([sigmas[:-1] if not force_sigma_min else sigmas, torch.FloatTensor([0.0])])
    return torch.FloatTensor(sigmas)


SCHEDULERS = {
    "ays": partial(_ays_scheduler, force_sigma_min=False, model_type="SDXL"),
    "ays+": partial(_ays_scheduler, force_sigma_min=True, model_type="SDXL_30"),
    "beta_1_1": partial(_beta, alpha=1.0, beta=1.0),
}


class PpmSchedulers(ComfyExtension):
    async def get_node_list(self):
        return []

    async def get_schedulers(self):
        return dict(SCHEDULERS)
