import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.wan.model_animate2 import PoseBranchCache  # noqa: E402


def test_select_promotes_matching_slot_when_earlier_slot_has_different_shape():
    cache = PoseBranchCache(store_device="cpu")
    cache.select(torch.randn(1, 8), create=True)
    cache.select(torch.randn(1, 4), create=True)
    second_key = cache.slots[1]["key"].clone()

    found = cache.select(second_key, create=False)

    assert found is True
    assert len(cache.slots) == 2
    assert cache.slot is cache.slots[-1]
    assert torch.equal(cache.slots[-1]["key"], second_key)
