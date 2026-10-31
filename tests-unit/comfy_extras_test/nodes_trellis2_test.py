from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

with patch.dict("sys.modules", {"nodes": MagicMock(), "server": MagicMock()}):
    from comfy_extras.nodes_trellis2 import Trellis2UpsampleStage


SIDE = 64  # upsampled points fill a SIDE^3 cube of the 512 grid: ceil(SIDE * (res / 16) / 512)^3 tokens per resolution


class StubShapeVae:
    def __init__(self):
        grid = torch.arange(SIDE, dtype=torch.int32)
        xyz = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1).reshape(-1, 3)
        self.hr_coords = torch.cat([torch.zeros(xyz.shape[0], 1, dtype=torch.int32), xyz], dim=1)

    def upsample_shape(self, slat, upsample_times):
        return self.hr_coords.to(slat.feats.device)


def run(target_resolution, **kwargs):
    vae = SimpleNamespace(first_stage_model=StubShapeVae(), vae_dtype=torch.float32,
                          prepare_decode=lambda *args, **kw: None)
    shape_latent = {"samples": torch.zeros(1, 32, 8, 1),
                    "coords": torch.cat([torch.zeros(8, 1, dtype=torch.int32),
                                         torch.randint(0, 32, (8, 3), dtype=torch.int32)], dim=1)}
    cond = [[torch.zeros(1, 1, 8), {}]]
    _, _, latent = Trellis2UpsampleStage.execute(cond, cond, shape_latent, vae, target_resolution, **kwargs)
    return latent["coord_resolution"] * 16, latent["coords"].shape[0]


def tokens(resolution):
    return -(-SIDE * (resolution // 16) // 512) ** 3


@pytest.mark.parametrize("max_tokens,expected", [
    (0, 1536),                   # disabled: always the target
    (tokens(1536) + 1, 1536),    # fits at the target
    (tokens(1536), 1408),        # one step down: the budget is strict, like the original
    (tokens(1280), 1152),
    (1, 1024),                   # never below 1024, even if still over budget
])
def test_upsample_stage_max_tokens(max_tokens, expected):
    resolution, count = run(1536, max_tokens=max_tokens)
    assert resolution == expected
    assert count == tokens(expected)


def test_upsample_stage_default_budget_matches_original_pipeline():
    resolution, _ = run(1536)
    assert tokens(1536) < 49152 and resolution == 1536
