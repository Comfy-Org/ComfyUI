import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.model_patcher import ModelPatcher, ModelPatcherBackup


def test_backup_entries_share_one_type():
    model = torch.nn.Module()
    keys = []
    for index in range(2):
        key = "weight_{}".format(index)
        keys.append(key)
        model.register_parameter(key, torch.nn.Parameter(torch.ones(1)))

    patcher = ModelPatcher(
        model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )

    for key in keys:
        patcher.patch_weight_to_device(key, device_to=torch.device("cpu"), force_cast=True)

    assert len(patcher.backup) == len(keys)
    for entry in patcher.backup.values():
        assert isinstance(entry, ModelPatcherBackup)
        assert entry.weight is not None
        assert entry.inplace_update is False
    assert len({type(entry) for entry in patcher.backup.values()}) == 1
