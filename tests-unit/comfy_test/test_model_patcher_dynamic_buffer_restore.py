import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_patcher


def test_restore_loaded_backups_restores_buffer_to_its_own_module():
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

    model = nn.Module()
    model.model_loaded_weight_memory = 0
    old_sampling = Sampling(torch.tensor([1.0, 2.0, 3.0]))
    model.model_sampling = old_sampling

    patcher = object.__new__(comfy.model_patcher.ModelPatcherDynamic)
    patcher.model = model
    patcher.backup = {}
    patcher.backup_buffers = {"model_sampling.sigmas": (old_sampling, old_sampling.sigmas.clone())}

    # Simulate an object patch swapping in a different module at the same path
    # (e.g. ModelSamplingDiscrete replacing the checkpoint's own model_sampling).
    new_sampling = Sampling(torch.tensor([4.0, 5.0, 6.0]))
    model.model_sampling = new_sampling

    patcher.restore_loaded_backups()

    assert model.model_sampling is new_sampling
    assert torch.equal(model.model_sampling.sigmas, torch.tensor([4.0, 5.0, 6.0]))
    assert torch.equal(old_sampling.sigmas, torch.tensor([1.0, 2.0, 3.0]))
