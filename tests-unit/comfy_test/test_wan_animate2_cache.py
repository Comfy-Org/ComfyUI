import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.wan.model_animate2 import PoseBranchCache


def test_select_moves_matching_tensor_key_to_most_recently_used():
    cache = PoseBranchCache()
    first = {"key": torch.tensor([1, 2, 3])}
    selected = {"key": torch.tensor([4, 5, 6])}
    last = {"key": torch.tensor([7, 8, 9])}
    cache.slots = [first, selected, last]

    assert cache.select(selected["key"], create=False)
    assert cache.slots[0] is first
    assert cache.slots[1] is last
    assert cache.slots[2] is selected
    assert cache.slot is selected
