"""
Tests for mask handling across arbitrary spatial dimensions.

Covers resolve_areas_and_cond_masks_multidim (conditioning masks) and
reshape_mask (denoise masks) for 1D (audio), 2D (image), and 3D (video)
spatial dims, including edge cases around batch size, spurious unsqueeze
from upstream nodes, and size mismatches.
"""

import torch

from comfy.samplers import resolve_areas_and_cond_masks_multidim
from comfy.utils import reshape_mask


def make_cond(mask):
    """Create a minimal conditioning dict with a mask."""
    return {"mask": mask, "model_conds": {}}


def run_resolve(mask, dims, device="cpu"):
    """Run resolve on a single condition and return the resolved mask."""
    conds = [make_cond(mask)]
    resolve_areas_and_cond_masks_multidim(conds, dims, device)
    return conds[0]["mask"]


# ============================================================
# 1D spatial dims (audio models like AceStep v1.5)
# dims = (length,), expected mask output: [batch, length]
# ============================================================

class Test1DSpatial:
    """Tests for 1D spatial models (e.g. audio with noise shape [B, C, L])."""

    def test_correct_shape_same_length(self):
        """[B, L] mask with matching length — should pass through unchanged."""
        mask = torch.ones(2, 100)
        result = run_resolve(mask, dims=(100,))
        assert result.shape == (2, 100)

    def test_correct_shape_resize(self):
        """[B, L] mask with different length — should resize via linear interp."""
        mask = torch.ones(1, 50)
        result = run_resolve(mask, dims=(100,))
        assert result.shape == (1, 100)

    def test_bare_spatial_mask(self):
        """[L] mask (no batch) — should get batch dim added."""
        mask = torch.ones(50)
        result = run_resolve(mask, dims=(100,))
        assert result.shape == (1, 100)

    def test_spurious_unsqueeze_from_hooks(self):
        """[1, B, L] mask (from set_mask_for_conditioning unsqueezing a [B, L] mask)
        — should squeeze back to [B, L]."""
        # Simulates: mask is [B, L], hooks.py does unsqueeze(0) -> [1, B, L]
        mask = torch.ones(1, 2, 100)
        result = run_resolve(mask, dims=(100,))
        assert result.shape == (2, 100)

    def test_spurious_unsqueeze_batch1(self):
        """[1, 1, L] mask (batch=1, hooks added extra dim) — should become [1, L]."""
        mask = torch.ones(1, 1, 50)
        result = run_resolve(mask, dims=(100,))
        assert result.shape == (1, 100)

    def test_batch_gt1_same_length(self):
        """[B, L] mask with batch=4 and matching length — no changes needed."""
        mask = torch.rand(4, 100)
        result = run_resolve(mask, dims=(100,))
        assert result.shape == (4, 100)
        torch.testing.assert_close(result, mask)

    def test_batch_gt1_resize(self):
        """[B, L] mask with batch=4 and different length — should resize each batch."""
        mask = torch.rand(4, 50)
        result = run_resolve(mask, dims=(100,))
        assert result.shape == (4, 100)

    def test_values_preserved_no_resize(self):
        """Mask values should be preserved when no resize is needed."""
        mask = torch.tensor([[0.0, 0.5, 1.0]])
        result = run_resolve(mask, dims=(3,))
        torch.testing.assert_close(result, mask)

    def test_linear_interpolation_values(self):
        """Check that linear interpolation produces sensible values."""
        mask = torch.tensor([[0.0, 1.0]])  # [1, 2]
        result = run_resolve(mask, dims=(5,))
        assert result.shape == (1, 5)
        # Should interpolate from 0 to 1
        assert result[0, 0].item() < result[0, -1].item()

    def test_set_area_to_bounds_skipped_for_1d(self):
        """set_area_to_bounds should be skipped for 1D (no crash)."""
        mask = torch.zeros(1, 100)
        mask[0, 10:50] = 1.0
        conds = [{"mask": mask, "model_conds": {}, "set_area_to_bounds": True}]
        resolve_areas_and_cond_masks_multidim(conds, (100,), "cpu")
        assert "area" not in conds[0]


# ============================================================
# 2D spatial dims (image models) — regression tests
# dims = (H, W), expected mask output: [batch, H, W]
# ============================================================

class Test2DSpatial:
    """Regression tests for standard 2D image models."""

    def test_correct_shape_same_size(self):
        """[B, H, W] mask matching dims — pass through."""
        mask = torch.ones(1, 64, 64)
        result = run_resolve(mask, dims=(64, 64))
        assert result.shape == (1, 64, 64)

    def test_bare_spatial_mask(self):
        """[H, W] mask — should get batch dim added."""
        mask = torch.ones(64, 64)
        result = run_resolve(mask, dims=(64, 64))
        assert result.shape == (1, 64, 64)

    def test_resize_different_resolution(self):
        """[B, H1, W1] mask with different size than dims — should bilinear resize."""
        mask = torch.ones(1, 32, 32)
        result = run_resolve(mask, dims=(64, 64))
        assert result.shape == (1, 64, 64)

    def test_4d_mask(self):
        """[B, C, H, W] mask (4D) — should resize via common_upscale 4D path."""
        mask = torch.ones(1, 1, 32, 32)
        result = run_resolve(mask, dims=(64, 64))
        assert result.shape == (1, 64, 64)

    def test_batch_gt1(self):
        """[B, H, W] mask with batch > 1."""
        mask = torch.rand(4, 64, 64)
        result = run_resolve(mask, dims=(64, 64))
        assert result.shape == (4, 64, 64)

    def test_batch_gt1_resize(self):
        """[B, H, W] mask with batch > 1 and different resolution."""
        mask = torch.rand(4, 32, 32)
        result = run_resolve(mask, dims=(64, 64))
        assert result.shape == (4, 64, 64)

    def test_set_area_to_bounds(self):
        """set_area_to_bounds should work for 2D masks."""
        mask = torch.zeros(1, 64, 64)
        mask[0, 10:20, 10:30] = 1.0
        conds = [{"mask": mask, "model_conds": {}, "set_area_to_bounds": True}]
        resolve_areas_and_cond_masks_multidim(conds, (64, 64), "cpu")
        assert "area" in conds[0]

    def test_non_square_resize(self):
        """[B, H1, W1] mask resized to non-square dims."""
        mask = torch.ones(1, 16, 32)
        result = run_resolve(mask, dims=(64, 128))
        assert result.shape == (1, 64, 128)


# ============================================================
# 3D spatial dims (video models)
# dims = (T, H, W), expected mask output: [batch, T, H, W]
# ============================================================

class Test3DSpatial:
    """Tests for 3D spatial models (e.g. video with noise shape [B, C, T, H, W])."""

    def test_correct_shape_same_size(self):
        """[B, T, H, W] mask matching dims — pass through."""
        mask = torch.ones(1, 8, 64, 64)
        result = run_resolve(mask, dims=(8, 64, 64))
        assert result.shape == (1, 8, 64, 64)

    def test_bare_spatial_mask(self):
        """[T, H, W] mask — should get batch dim added."""
        mask = torch.ones(8, 64, 64)
        result = run_resolve(mask, dims=(8, 64, 64))
        assert result.shape == (1, 8, 64, 64)

    def test_resize_hw(self):
        """[B, T, H1, W1] mask with different H, W — should resize last 2 dims."""
        mask = torch.ones(1, 8, 32, 32)
        result = run_resolve(mask, dims=(8, 64, 64))
        assert result.shape == (1, 8, 64, 64)

    def test_set_area_to_bounds_skipped_for_3d(self):
        """set_area_to_bounds should be skipped for 3D (no crash)."""
        mask = torch.zeros(1, 8, 64, 64)
        mask[0, :, 10:20, 10:30] = 1.0
        conds = [{"mask": mask, "model_conds": {}, "set_area_to_bounds": True}]
        resolve_areas_and_cond_masks_multidim(conds, (8, 64, 64), "cpu")
        assert "area" not in conds[0]


class TestNoMask:
    """Conditions without masks should pass through untouched."""

    def test_no_mask_key(self):
        """Condition with no mask key — untouched."""
        conds = [{"model_conds": {}}]
        resolve_areas_and_cond_masks_multidim(conds, (64, 64), "cpu")
        assert "mask" not in conds[0]

    def test_empty_conditions(self):
        """Empty conditions list — no crash."""
        conds = []
        resolve_areas_and_cond_masks_multidim(conds, (64, 64), "cpu")
        assert len(conds) == 0


# ============================================================
# Area resolution (percentage-based)
# ============================================================

class TestAreaResolution:
    """Test that percentage-based area resolution works for different dims."""

    def test_percentage_area_2d(self):
        """Percentage area for 2D should resolve to pixel coords."""
        conds = [{"area": ("percentage", 0.5, 0.5, 0.25, 0.25), "model_conds": {}}]
        resolve_areas_and_cond_masks_multidim(conds, (64, 64), "cpu")
        area = conds[0]["area"]
        assert area == (32, 32, 16, 16)

    def test_percentage_area_1d(self):
        """Percentage area for 1D should resolve to frame coords."""
        conds = [{"area": ("percentage", 0.5, 0.25), "model_conds": {}}]
        resolve_areas_and_cond_masks_multidim(conds, (100,), "cpu")
        area = conds[0]["area"]
        assert area == (50, 25)


# ============================================================
# reshape_mask — mask reshaping for F.interpolate
# ============================================================

class TestReshapeMask1D:
    """Tests for reshape_mask with 1D output (e.g. audio with noise shape [B, C, L])."""

    def test_4d_input_same_length(self):
        """[1, 1, 1, L] input (typical from pipeline) — should reshape and expand channels."""
        mask = torch.ones(1, 1, 1, 100)
        result = reshape_mask(mask, torch.Size([1, 64, 100]))
        assert result.shape == (1, 64, 100)

    def test_4d_input_resize(self):
        """[1, 1, 1, L1] input resized to different length."""
        mask = torch.ones(1, 1, 1, 50)
        result = reshape_mask(mask, torch.Size([1, 64, 100]))
        assert result.shape == (1, 64, 100)

    def test_3d_input(self):
        """[1, 1, L] input — should work directly."""
        mask = torch.ones(1, 1, 100)
        result = reshape_mask(mask, torch.Size([1, 64, 100]))
        assert result.shape == (1, 64, 100)

    def test_2d_input(self):
        """[B, L] input — should reshape to [B, 1, L]."""
        mask = torch.ones(1, 50)
        result = reshape_mask(mask, torch.Size([1, 64, 100]))
        assert result.shape == (1, 64, 100)

    def test_1d_input(self):
        """[L] input — should reshape to [1, 1, L]."""
        mask = torch.ones(50)
        result = reshape_mask(mask, torch.Size([1, 64, 100]))
        assert result.shape == (1, 64, 100)

    def test_channel_repeat(self):
        """Mask with 1 channel should repeat to match output channels."""
        mask = torch.full((1, 1, 1, 100), 0.5)
        result = reshape_mask(mask, torch.Size([1, 32, 100]))
        assert result.shape == (1, 32, 100)
        torch.testing.assert_close(result, torch.full_like(result, 0.5))

    def test_batch_repeat(self):
        """Single-batch mask should repeat to match output batch size."""
        mask = torch.full((1, 1, 1, 100), 0.7)
        result = reshape_mask(mask, torch.Size([4, 64, 100]))
        assert result.shape == (4, 64, 100)

    def test_values_preserved_no_resize(self):
        """Values should be preserved when no resize is needed."""
        values = torch.tensor([[[0.0, 0.5, 1.0]]])  # [1, 1, 3]
        result = reshape_mask(values, torch.Size([1, 1, 3]))
        torch.testing.assert_close(result, values)

    def test_interpolation_values(self):
        """Linear interpolation should produce sensible intermediate values."""
        mask = torch.tensor([[[[0.0, 1.0]]]])  # [1, 1, 1, 2]
        result = reshape_mask(mask, torch.Size([1, 1, 4]))
        assert result.shape == (1, 1, 4)
        # Should interpolate from 0 to 1
        assert result[0, 0, 0].item() < result[0, 0, -1].item()


class TestReshapeMask2D:
    """Regression tests for reshape_mask with 2D output (image models)."""

    def test_standard_resize(self):
        """[1, 1, H, W] mask resized to different resolution."""
        mask = torch.ones(1, 1, 32, 32)
        result = reshape_mask(mask, torch.Size([1, 4, 64, 64]))
        assert result.shape == (1, 4, 64, 64)

    def test_same_size(self):
        """[1, 1, H, W] mask with matching size — no resize needed."""
        mask = torch.rand(1, 1, 64, 64)
        result = reshape_mask(mask, torch.Size([1, 4, 64, 64]))
        assert result.shape == (1, 4, 64, 64)

    def test_3d_input(self):
        """[B, H, W] input — should reshape to [B, 1, H, W]."""
        mask = torch.ones(1, 32, 32)
        result = reshape_mask(mask, torch.Size([1, 4, 64, 64]))
        assert result.shape == (1, 4, 64, 64)


class TestReshapeMask3D:
    """Regression tests for reshape_mask with 3D output (video models)."""

    def test_standard_resize(self):
        """[1, 1, T, H, W] mask resized to different resolution."""
        mask = torch.ones(1, 1, 8, 32, 32)
        result = reshape_mask(mask, torch.Size([1, 4, 8, 64, 64]))
        assert result.shape == (1, 4, 8, 64, 64)

    def test_4d_input(self):
        """[B, T, H, W] input — should reshape to [1, 1, T, H, W]."""
        mask = torch.ones(1, 8, 32, 32)
        result = reshape_mask(mask, torch.Size([1, 4, 8, 64, 64]))
        assert result.shape == (1, 4, 8, 64, 64)
