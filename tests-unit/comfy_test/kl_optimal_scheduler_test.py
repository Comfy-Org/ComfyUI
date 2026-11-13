import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.samplers  # noqa: E402


def test_kl_optimal_scheduler_single_step():
    sigmas = comfy.samplers.kl_optimal_scheduler(n=1, sigma_min=0.03, sigma_max=14.6)
    assert torch.allclose(sigmas, torch.tensor([14.6, 0.0]))


def test_kl_optimal_scheduler_endpoints():
    sigmas = comfy.samplers.kl_optimal_scheduler(n=5, sigma_min=0.03, sigma_max=14.6)
    assert sigmas.shape == (6,)
    assert torch.allclose(sigmas, torch.tensor([14.6, 2.14361826, 0.96232334, 0.42055038, 0.03, 0.0]))
