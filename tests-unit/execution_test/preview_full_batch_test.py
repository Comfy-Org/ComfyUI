"""Unit tests for the --preview-full-batch CLI option and its grid-preview rendering."""
import numpy as np
import torch

from comfy.cli_args import args, parser
from comfy.taesd.taesd import TAESD
from latent_preview import Latent2RGBPreviewer, TAEHVPreviewerImpl, TAESDPreviewerImpl


class TestPreviewFullBatchArg:
    def test_default_is_false(self):
        assert args.preview_full_batch is False

    def test_parser_sets_true(self):
        ns = parser.parse_args(["--preview-full-batch"])
        assert ns.preview_full_batch is True

    def test_parser_without_flag_is_false(self):
        ns = parser.parse_args([])
        assert ns.preview_full_batch is False


class _FakeTAESD:
    """Test double for the TAEHV (video TAE) decoder.

    The real decoder upsamples 8x and returns [1, 3, 8H, 8W]; this double
    reproduces that contract with deterministic output. It also accepts the
    5-D slice [1, C, 1, H, W] produced by the default TAEHV path and uses
    its first temporal frame.
    """

    def decode(self, x):
        if x.ndim == 5:
            x = x[:, :, 0]
        b, c, h, w = x.shape
        return torch.ones(b, 3, h * 8, w * 8)


def _assert_every_cell_filled(img, cells_x, cells_y):
    """Assert each grid cell carries content, i.e. every batch item was composited."""
    arr = np.array(img)
    cell_h, cell_w = arr.shape[0] // cells_y, arr.shape[1] // cells_x
    for y in range(cells_y):
        for x in range(cells_x):
            cell = arr[y * cell_h:(y + 1) * cell_h, x * cell_w:(x + 1) * cell_w]
            assert cell.max() > 0, f"grid cell ({x}, {y}) is empty"


class TestFullBatchRendering:
    """Rendering tests for the --preview-full-batch grid preview.

    Covers both previewer paths (TAESD/TAESD-based and Latent2RGB) and the
    TAEHV (video TAE) path: with the flag off, only the first batch item is
    previewed (existing behavior); with the flag on, every batch item is
    composited into a single grid, including 5-D video latent input where
    the first temporal frame of each item is used.
    """

    # --- TAESD path ---

    def test_taesd_default_preserves_first_item(self, monkeypatch):
        monkeypatch.setattr(args, "preview_full_batch", False)
        prev = TAESDPreviewerImpl(TAESD(None, None, latent_channels=16))
        x0 = torch.randn(4, 16, 64, 64)
        img = prev.decode_latent_to_preview(x0)
        # Existing behavior: a single 8x-upscaled tile, not a grid
        assert img.size == (512, 512)

    def test_taesd_full_batch_grid_includes_all_items(self, monkeypatch):
        monkeypatch.setattr(args, "preview_full_batch", True)
        # Deterministic decoder: every decoded tile is all-ones, so each grid
        # cell must be non-empty, proving every batch item was composited.
        prev = TAESDPreviewerImpl(_FakeTAESD())
        x0 = torch.randn(4, 16, 64, 64)
        img = prev.decode_latent_to_preview(x0)
        # 2x2 grid of 512x512 tiles -> 1024x1024
        assert img.size == (1024, 1024)
        _assert_every_cell_filled(img, 2, 2)

    def test_taesd_full_batch_5d_video_latent(self, monkeypatch):
        monkeypatch.setattr(args, "preview_full_batch", True)
        prev = TAESDPreviewerImpl(TAESD(None, None, latent_channels=16))
        x0 = torch.randn(4, 16, 8, 64, 64)
        img = prev.decode_latent_to_preview(x0)
        # First temporal frame of every batch item, tiled into a 2x2 grid
        assert img.size == (1024, 1024)

    # --- TAEHV (video TAE) path ---

    def test_taehv_default_preserves_first_item(self, monkeypatch):
        monkeypatch.setattr(args, "preview_full_batch", False)
        prev = TAEHVPreviewerImpl(_FakeTAESD())
        x0 = torch.randn(4, 16, 64, 64)
        img = prev.decode_latent_to_preview(x0)
        # Existing behavior: x0[:1, :, :1] slices H to 1 -> [1, C, 1, W]
        assert img.size == (512, 8)

    def test_taehv_default_5d_uses_first_frame(self, monkeypatch):
        monkeypatch.setattr(args, "preview_full_batch", False)
        prev = TAEHVPreviewerImpl(_FakeTAESD())
        x0 = torch.randn(4, 16, 8, 64, 64)
        img = prev.decode_latent_to_preview(x0)
        # First batch item, first temporal frame
        assert img.size == (512, 512)

    def test_taehv_full_batch_grid_includes_all_items(self, monkeypatch):
        monkeypatch.setattr(args, "preview_full_batch", True)
        prev = TAEHVPreviewerImpl(_FakeTAESD())
        x0 = torch.randn(4, 16, 64, 64)
        img = prev.decode_latent_to_preview(x0)
        assert img.size == (1024, 1024)
        _assert_every_cell_filled(img, 2, 2)

    def test_taehv_full_batch_5d_video_latent(self, monkeypatch):
        monkeypatch.setattr(args, "preview_full_batch", True)
        prev = TAEHVPreviewerImpl(_FakeTAESD())
        x0 = torch.randn(4, 16, 8, 64, 64)
        img = prev.decode_latent_to_preview(x0)
        assert img.size == (1024, 1024)

    # --- Latent2RGB path ---

    def test_latent2rgb_default_preserves_first_item(self, monkeypatch):
        monkeypatch.setattr(args, "preview_full_batch", False)
        prev = Latent2RGBPreviewer(torch.eye(16, 3))
        x0 = torch.randn(4, 16, 64, 64)
        img = prev.decode_latent_to_preview(x0)
        # Existing behavior: only the first batch member is projected to RGB
        assert img.size == (64, 64)

    def test_latent2rgb_full_batch_grid_includes_all_items(self, monkeypatch):
        monkeypatch.setattr(args, "preview_full_batch", True)
        prev = Latent2RGBPreviewer(torch.eye(16, 3))
        x0 = torch.randn(4, 16, 64, 64)
        img = prev.decode_latent_to_preview(x0)
        # 2x2 grid of 64x64 RGB tiles -> 128x128
        assert img.size == (128, 128)
        _assert_every_cell_filled(img, 2, 2)

    def test_latent2rgb_full_batch_5d_video_latent(self, monkeypatch):
        monkeypatch.setattr(args, "preview_full_batch", True)
        prev = Latent2RGBPreviewer(torch.eye(16, 3))
        x0 = torch.randn(4, 16, 8, 64, 64)
        img = prev.decode_latent_to_preview(x0)
        assert img.size == (128, 128)

