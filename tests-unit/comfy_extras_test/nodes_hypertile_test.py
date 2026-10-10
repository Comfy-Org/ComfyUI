from unittest.mock import patch, MagicMock

import torch

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_hypertile import random_divisor


class TestRandomDivisor:
    def test_all_options_are_reachable(self):
        # value=8, min=2, max_options=2 -> candidate tile counts {4, 2}; both must be
        # selectable. torch.randint's high is exclusive, so the last option was unreachable.
        torch.manual_seed(0)
        results = {random_divisor(8, 2, 2) for _ in range(200)}
        assert results == {2, 4}

    def test_single_option_is_deterministic(self):
        assert random_divisor(8, 8, 1) == 1
