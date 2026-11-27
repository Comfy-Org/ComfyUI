import weakref

import torch

from comfy.ldm.minimax.model import PackedLayout
from comfy_extras.nodes_sparse_attention import SparseAttnPatch


def test_vsa_rope_reuses_table_for_same_layout_and_rebuilds_for_new_positions():
    patch = SparseAttnPatch(0.5, 0.1, True, 1.0, 0.0, 0, set(), "off", 0, False)
    latents = torch.empty(1, 1, 1, 4, 4)
    layouts = [PackedLayout(2, 1, 4, 4, 1,
                            keyframes=[{"resolved_frame_index": frame, "latent": latents}])
               for frame in (0, 4)]
    assert not torch.equal(layouts[0].position_ids, layouts[1].position_ids)

    plan = patch.vsa_plan(layouts[0], torch.device("cpu"))
    freqs = layouts[0].position_ids.float().unsqueeze(0)
    first = patch.vsa_rope_freqs(freqs, plan)
    torch.testing.assert_close(first[0, plan["inv"]], freqs[0])

    second = patch.vsa_rope_freqs(freqs.clone(), plan)
    assert second is first
    old_table = weakref.ref(first)
    del first, second

    assert patch.vsa_plan(layouts[1], torch.device("cpu")) is plan
    changed_freqs = layouts[1].position_ids.float().unsqueeze(0)
    changed = patch.vsa_rope_freqs(changed_freqs, plan)
    assert old_table() is None
    torch.testing.assert_close(changed[0, plan["inv"]], changed_freqs[0])
