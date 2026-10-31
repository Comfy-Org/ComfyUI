from unittest import mock

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_patcher


def test_restore_loaded_backups_restores_buffer_to_its_own_module(monkeypatch):
    """Regression test for #16490.

    ModelPatcherDynamic.load() backs up every buffer by its dotted attribute
    path. If an object patch (e.g. ModelSamplingDiscrete) swaps in a new
    module at that path before restore_loaded_backups() runs, the backup must
    not be written into the new module - it belongs to the module it was
    taken from.
    """
    class Sampling(nn.Module):
        def __init__(self, sigmas):
            super().__init__()
            self.register_buffer("sigmas", sigmas)

    monkeypatch.setattr(comfy.model_patcher.comfy_aimdo.host_buffer, "HostBuffer", mock.MagicMock())
    cpu = torch.device("cpu")
    model = nn.Module()
    model.model_loaded_weight_memory = 0
    old_sampling = Sampling(torch.tensor([1.0, 2.0, 3.0]))
    model.model_sampling = old_sampling
    original_sigmas = old_sampling.sigmas

    # CPU construction reroutes to ModelPatcher, so build the dynamic patcher directly.
    patcher = object.__new__(comfy.model_patcher.ModelPatcherDynamic)
    comfy.model_patcher.ModelPatcher.__init__(patcher, model, cpu, cpu)
    comfy.model_patcher.ModelPatcherDynamic.__init__(patcher, model, cpu, cpu)
    patcher.load(device_to=cpu)

    # Simulate an object patch swapping in a different module at the same path
    # (e.g. ModelSamplingDiscrete replacing the checkpoint's own model_sampling).
    new_sampling = Sampling(torch.tensor([4.0, 5.0, 6.0]))
    model.model_sampling = new_sampling

    # Overwrite old_sampling's buffer so restoring it back is observable.
    old_sampling.register_buffer("sigmas", torch.tensor([7.0, 8.0, 9.0]))

    patcher.restore_loaded_backups()

    assert model.model_sampling is new_sampling
    assert torch.equal(model.model_sampling.sigmas, torch.tensor([4.0, 5.0, 6.0]))
    assert old_sampling.sigmas is original_sigmas
    assert torch.equal(old_sampling.sigmas, torch.tensor([1.0, 2.0, 3.0]))
