from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

from comfy_extras.nodes_video_deflicker import warp


def _make_warp_inputs():
    torch.manual_seed(0)
    b, c, h, w = 1, 3, 8, 8
    x = torch.randn(b, c, h, w)
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    grid = torch.stack((xs, ys), dim=-1).unsqueeze(0)
    norm = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)]).view(1, 1, 1, 2)
    # Large flow values push sampling coordinates well outside [-1, 1]
    flow = torch.randn(b, 2, h, w) * 6.0
    return x, flow, grid, norm


def test_warp_mps_avoids_border_padding_and_matches_cpu(monkeypatch):
    x, flow, grid, norm = _make_warp_inputs()
    expected = warp(x, flow, grid, norm)

    recorded = {}
    real_grid_sample = F.grid_sample

    def spy_grid_sample(inp, sample_grid, **kwargs):
        recorded["padding_mode"] = kwargs.get("padding_mode")
        recorded["min_coord"] = sample_grid.min().item()
        recorded["max_coord"] = sample_grid.max().item()
        return real_grid_sample(x, sample_grid, **kwargs)

    monkeypatch.setattr(F, "grid_sample", spy_grid_sample)
    fake_mps_x = SimpleNamespace(device=SimpleNamespace(type="mps"))
    actual = warp(fake_mps_x, flow, grid, norm)

    assert recorded["padding_mode"] == "zeros"
    assert recorded["min_coord"] >= -1.0
    assert recorded["max_coord"] <= 1.0
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device not available")
def test_warp_on_mps_device_matches_cpu():
    x, flow, grid, norm = _make_warp_inputs()
    expected = warp(x, flow, grid, norm)
    actual = warp(x.to("mps"), flow.to("mps"), grid.to("mps"), norm.to("mps")).cpu()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
