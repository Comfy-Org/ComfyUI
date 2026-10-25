import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.wan.model_animate2 import PoseBranchCache


def test_select_moves_matched_slot_to_end_across_different_key_shapes():
    cache = PoseBranchCache()
    first, second, third = torch.zeros(1, 4), torch.ones(1, 7), torch.full((1, 5), 2.0)
    for key in (first, second, third):
        assert not cache.select(key, create=False)
        cache.select(key)
    slots = list(cache.slots)

    assert cache.select(second, create=False)
    assert [id(s) for s in cache.slots] == [id(slots[0]), id(slots[2]), id(slots[1])] and cache.slot is slots[1]
    assert cache.select(first, create=False)
    assert cache.slots[-1] is slots[0] and cache.slot is slots[0]
