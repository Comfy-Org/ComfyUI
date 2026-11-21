import pytest
import torch
from unittest.mock import patch, MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_mask import MaskComposite


class TestMaskCompositeOperations:
    @staticmethod
    def _exec(destination, source, operation, threshold=None):
        d = torch.tensor([[destination]])
        s = torch.tensor([[source]])
        if threshold is None:
            out = MaskComposite.execute(d, s, 0, 0, operation)
        else:
            out = MaskComposite.execute(d, s, 0, 0, operation, threshold)
        return out.result[0].flatten().tolist()

    def test_max_is_union_of_soft_masks(self):
        result = self._exec([0.0, 0.25, 0.75, 1.0], [0.5, 0.5, 0.5, 0.5], "max")
        assert result == pytest.approx([0.5, 0.5, 0.75, 1.0])

    def test_min_is_intersection_of_soft_masks(self):
        result = self._exec([0.0, 0.25, 0.75, 1.0], [0.5, 0.5, 0.5, 0.5], "min")
        assert result == pytest.approx([0.0, 0.25, 0.5, 0.5])

    def test_max_preserves_intermediate_values(self):
        # Unlike "or", max must not round feathered values to 0 or 1.
        result = self._exec([0.25, 0.75], [0.0, 0.0], "max")
        assert result == pytest.approx([0.25, 0.75])

    def test_min_preserves_intermediate_values(self):
        result = self._exec([0.25, 0.75], [1.0, 1.0], "min")
        assert result == pytest.approx([0.25, 0.75])

    def test_or_binarizes(self):
        # Documents existing behaviour that motivates max/min.
        result = self._exec([0.25, 0.75], [0.0, 0.0], "or")
        assert result == pytest.approx([0.0, 1.0])

    def test_max_is_commutative(self):
        a = self._exec([0.3, 0.8], [0.6, 0.1], "max")
        b = self._exec([0.6, 0.1], [0.3, 0.8], "max")
        assert a == pytest.approx(b)

    def test_min_is_commutative(self):
        a = self._exec([0.3, 0.8], [0.6, 0.1], "min")
        b = self._exec([0.6, 0.1], [0.3, 0.8], "min")
        assert a == pytest.approx(b)

    def test_max_with_empty_mask_is_identity(self):
        result = self._exec([0.0, 0.4, 1.0], [0.0, 0.0, 0.0], "max")
        assert result == pytest.approx([0.0, 0.4, 1.0])

    def test_min_with_full_mask_is_identity(self):
        result = self._exec([0.0, 0.4, 1.0], [1.0, 1.0, 1.0], "min")
        assert result == pytest.approx([0.0, 0.4, 1.0])


class TestMaskCompositeThreshold:
    _exec = staticmethod(TestMaskCompositeOperations._exec)

    def test_default_threshold_matches_legacy_rounding(self):
        # The default must reproduce the old .round() exactly, including its
        # round-half-to-even behaviour at 0.5 (0.5 -> 0, not 1).
        values = [0.0, 0.25, 0.5, 0.5001, 0.75, 1.0]
        zeros = [0.0] * len(values)
        legacy = torch.tensor(values).round().tolist()
        assert self._exec(values, zeros, "or") == pytest.approx(legacy)
        assert self._exec(values, zeros, "or", 0.5) == pytest.approx(legacy)

    def test_low_threshold_keeps_faint_pixels(self):
        # threshold=0.0 makes "or" a union of the two footprints instead of a
        # union of only their confident areas.
        result = self._exec([0.0, 0.05, 0.4], [0.0, 0.0, 0.0], "or", 0.0)
        assert result == pytest.approx([0.0, 1.0, 1.0])

    def test_high_threshold_keeps_only_strong_pixels(self):
        result = self._exec([0.5, 0.85, 1.0], [1.0, 1.0, 1.0], "and", 0.9)
        assert result == pytest.approx([0.0, 0.0, 1.0])

    def test_threshold_applies_to_both_operands(self):
        result = self._exec([0.3, 0.3, 0.0], [0.3, 0.0, 0.0], "and", 0.2)
        assert result == pytest.approx([1.0, 0.0, 0.0])

    def test_threshold_changes_xor(self):
        # At 0.5 neither 0.3 nor 0.4 is set, so xor is 0; at 0.2 only the
        # destination pixel is set on the second element, so xor is 1.
        assert self._exec([0.3, 0.3], [0.4, 0.1], "xor") == pytest.approx([0.0, 0.0])
        assert self._exec([0.3, 0.3], [0.4, 0.1], "xor", 0.2) == pytest.approx([0.0, 1.0])

    @pytest.mark.parametrize("operation", ["multiply", "add", "subtract", "max", "min"])
    def test_arithmetic_operations_ignore_threshold(self, operation):
        d, s = [0.2, 0.6, 0.9], [0.5, 0.5, 0.5]
        assert self._exec(d, s, operation, 0.05) == pytest.approx(self._exec(d, s, operation))
        assert self._exec(d, s, operation, 0.95) == pytest.approx(self._exec(d, s, operation))
