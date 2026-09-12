from unittest.mock import patch, MagicMock

import torch

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

# comfy.model_management initializes the torch device at import and requires CUDA
# (or --cpu), which the unit-test environment does not provide; stub it for the import.
patched_modules = {"nodes": mock_nodes, "server": mock_server, "comfy.model_management": MagicMock()}
with patch.dict("sys.modules", patched_modules):
    # imported first to satisfy the nodes_post_processing <-> nodes_latent circular import
    import comfy_extras.nodes_post_processing  # noqa: F401
    from comfy_extras.nodes_latent import LatentInterpolate


class TestLatentInterpolate:
    def test_batch_size_one(self):
        s = {"samples": torch.randn(1, 4, 16, 16)}
        out = LatentInterpolate.execute(s, s, 0.5)[0]
        assert out["samples"].shape == (1, 4, 16, 16)

    def test_batched_latent_does_not_crash(self):
        s1 = {"samples": torch.randn(2, 4, 16, 16)}
        s2 = {"samples": torch.randn(2, 4, 16, 16)}
        out = LatentInterpolate.execute(s1, s2, 0.5)[0]
        assert out["samples"].shape == (2, 4, 16, 16)

    def test_each_batch_element_is_interpolated_independently(self):
        # normalization must be per latent, not mixed across the batch. batch == channels
        # (4) is the case that does not crash but silently corrupts without the fix.
        a = torch.randn(1, 4, 8, 8)
        b = torch.randn(1, 4, 8, 8)
        filler = torch.randn(3, 4, 8, 8)
        single = LatentInterpolate.execute({"samples": a}, {"samples": b}, 0.3)[0]["samples"]
        batched = LatentInterpolate.execute(
            {"samples": torch.cat([a, filler])},
            {"samples": torch.cat([b, filler])},
            0.3,
        )[0]["samples"]
        assert torch.allclose(single[0], batched[0], atol=1e-5)
