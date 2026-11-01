import torch

from comfy.cli_args import args

args.cpu = True

from comfy.samplers import _batch_timestep


def test_batch_timestep_expands_singleton_batch():
    timestep = torch.tensor([[1.0, 2.0]])

    batched = _batch_timestep(timestep, 3)

    assert torch.equal(batched, torch.tensor([[1.0, 2.0]]).expand(3, 2))


def test_batch_timestep_repeats_full_per_sample_batch():
    timestep = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    batched = _batch_timestep(timestep, 3)

    assert torch.equal(
        batched,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]] * 3),
    )
