import sys
from unittest.mock import MagicMock, patch

import torch

# comfy.model_management initializes the torch device at import time and requires
# CUDA (or --cpu), which the unit-test environment does not provide. Stub it so
# importing comfy.context_windows does not trigger device initialization.
with patch.dict(sys.modules, {"comfy.model_management": MagicMock()}):
    from comfy.context_windows import create_weights_overlap_linear


class TestCreateWeightsOverlapLinear:
    def test_zero_overlap_returns_ones(self):
        # context_overlap == 0 is reachable (the context-window nodes allow min=0,
        # and the WAN/LTXV nodes derive it via max(overlap // 4, 0)). With no overlap
        # band there is nothing to blend, so every weight is 1.
        w = create_weights_overlap_linear(16, 100, list(range(20, 36)), 0)
        assert w.shape == (16,)
        assert torch.equal(w, torch.ones(16))

    def test_overlap_ramps_edges(self):
        w = create_weights_overlap_linear(16, 100, list(range(20, 36)), 4)
        assert w.shape == (16,)
        # middle (non-first) window: left edge ramps up to 1 over the overlap width
        assert w[0] < w[3]
        assert torch.isclose(w[3], torch.tensor(1.0))
        # right edge ramps back down
        assert w[-1] < w[-4]
