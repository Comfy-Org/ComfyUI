"""SeedVR2 temporal chunk/merge node regression tests."""

import pytest
import torch

from comfy.cli_args import args as cli_args
from comfy.ldm.seedvr.constants import (
    BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE,
    SEEDVR2_CHUNK_GIB_PER_MPX_FRAME,
    SEEDVR2_CHUNK_RESERVED_GIB,
    SEEDVR2_CHUNK_SIGMA_GIB,
    SEEDVR2_CHUNK_SIGMA_K,
    SEEDVR2_LATENT_CHANNELS,
)

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management  # noqa: E402
from comfy_extras.nodes_seedvr import SeedVR2TemporalChunk, SeedVR2TemporalMerge, _seedvr2_chunk_crossfade_weights  # noqa: E402

def _latent(t_latent, h=8, w=8, b=1):
    g = torch.Generator().manual_seed(7)
    return {"samples": torch.randn(b, SEEDVR2_LATENT_CHANNELS, t_latent, h, w, generator=g)}

def _split(latent, frames_per_chunk, temporal_overlap, chunking_mode="manual"):
    combo = {"chunking_mode": chunking_mode}
    if chunking_mode != "auto":
        combo["frames_per_chunk"] = frames_per_chunk
    return SeedVR2TemporalChunk.execute(latent, temporal_overlap, combo).args

def _merge(chunks, temporal_overlap):
    return SeedVR2TemporalMerge.execute(chunks, [temporal_overlap]).args[0]

def test_chunk_temporal_windows_and_validation():
    with pytest.raises(ValueError, match="4n\\+1"):
        _split(_latent(9), 20, 0)
    with pytest.raises(ValueError, match="5-D"):
        _split({"samples": torch.zeros(1, SEEDVR2_LATENT_CHANNELS * 9, 8, 8)}, 21, 0)
    with pytest.raises(ValueError, match="chunking_mode"):
        _split(_latent(13), 21, 0, "adaptive")
    latent = _latent(13)
    chunks, overlap = _split(latent, 21, 2)  # chunk_latent=6, step=4 -> [0:6], [4:10], [8:13]
    assert overlap == 2 and [c["samples"].shape[2] for c in chunks] == [6, 6, 5]
    assert all(torch.equal(c["samples"], latent["samples"][:, :, s:e]) for c, (s, e) in zip(chunks, [(0, 6), (4, 10), (8, 13)]))
    assert len(_split(_latent(13), 21, 999)[0]) == 8  # overlap clamps to chunk_latent-1 -> step=1
    assert (r := _split(_latent(5), 21, 3)) and len(r[0]) == 1 and r[1] == 0  # t_pixel <= 21: passthrough

def test_chunk_auto_mode_applies_vram_law(monkeypatch):
    mpx_per_frame = (32 * 32) * (BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE ** 2) / 1e6
    free_gb = (
        SEEDVR2_CHUNK_RESERVED_GIB
        + SEEDVR2_CHUNK_SIGMA_K * SEEDVR2_CHUNK_SIGMA_GIB
        + 5.1 * SEEDVR2_CHUNK_GIB_PER_MPX_FRAME * mpx_per_frame
    )
    monkeypatch.setattr(comfy.model_management, "get_free_memory", lambda dev=None: free_gb * (1024 ** 3))
    assert [c["samples"].shape[2] for c in _split(_latent(13, h=32, w=32), 1, 0, "auto")[0]] == [5, 5, 3]
    assert _split(_latent(13, h=32, w=32, b=2), 1, 0, "auto")[0][0]["samples"].shape[2] == 2  # batch halves the chunk

def test_merge_crossfade_and_reassembly():
    latent = _latent(13)
    latent["noise_mask"] = torch.rand(1, 1, 13, 8, 8)
    latent["batch_index"] = [0]
    merged = _merge(_split(latent, 21, 0)[0], 0)
    assert torch.equal(merged["samples"], latent["samples"])
    assert "noise_mask" not in merged and merged["batch_index"] == [0]
    assert torch.allclose(_merge(_split(latent, 21, 3)[0], 3)["samples"], latent["samples"], atol=1e-6)
    w = _seedvr2_chunk_crossfade_weights(3, merged["samples"].device, merged["samples"].dtype)
    assert w[0] == 1.0 and w[-1] == 0.0 and torch.all(w[:-1] >= w[1:])
    ones, zeros = {"samples": torch.ones(1, SEEDVR2_LATENT_CHANNELS, 6, 8, 8)}, {"samples": torch.zeros(1, SEEDVR2_LATENT_CHANNELS, 6, 8, 8)}
    fused = _merge([ones, zeros], 3)["samples"]  # overlap equals w: prev fades out, next fades in
    assert torch.equal(fused[:, :, 3:6], w.view(1, 1, 3, 1, 1).expand(1, SEEDVR2_LATENT_CHANNELS, 3, 8, 8))
    assert torch.equal(fused[:, :, :3], ones["samples"][:, :, :3]) and torch.equal(fused[:, :, 6:], zeros["samples"][:, :, :3])
    short = _split(latent, 21, 2)[0]
    short[0]["samples"] = short[0]["samples"][:, :, :4]
    with pytest.raises(ValueError, match="only the final chunk may be shorter"):
        _merge(short, 2)
