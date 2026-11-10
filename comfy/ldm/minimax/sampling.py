import torch

import comfy.model_sampling


class MiniMaxH3ImageModelSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    """Flow sampling on the PDD sigma grid of a MiniMax-H3-Image checkpoint.

    sigma() is linear in the grid index, so the stock schedulers at pdd_steps steps land on the grid the
    heads were distilled for; the flow shift is unused.
    """

    def __init__(self, model_config):
        torch.nn.Module.__init__(self)
        # the grid decreases from 1 to 0; stored ascending so index 0 is sigma 0
        self.register_buffer("grid", torch.tensor(model_config.unet_config["pdd_sigmas"], dtype=torch.float32).flip(0))
        self.set_noise_scale(model_config.sampling_settings.get("noise_scale", 1.0))
        self.set_parameters()

    def sigma(self, timestep):
        grid = self.grid.to(timestep.device)
        n = grid.shape[0] - 1
        pos = (timestep / self.multiplier).clamp(0.0, 1.0) * n
        i = pos.floor().long().clamp(max=n - 1)
        return torch.lerp(grid[i], grid[i + 1], pos - i)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return float(self.sigma(torch.tensor((1.0 - percent) * self.multiplier)))
