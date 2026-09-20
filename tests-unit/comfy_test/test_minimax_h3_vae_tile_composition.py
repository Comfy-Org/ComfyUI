import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.minimax.vae as vae_mod  # noqa: E402
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE  # noqa: E402


def _bare_model(overlap=64):
    model = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(model)
    model.vae_ratio = 16
    model.vae_ratio_t = 4
    model.tile_size = 256
    model.tile_overlap_min = overlap
    model.tiling = True
    return model


def _axis_oracle(starts, lengths, overlaps, axis_length):
    weights = torch.zeros((len(starts), axis_length), dtype=torch.float64)
    for i, (start, length) in enumerate(zip(starts, lengths)):
        u = torch.arange(length, dtype=torch.float64)
        q = torch.ones(length, dtype=torch.float64)
        if i > 0 and overlaps[i - 1] > 0:
            q *= torch.clamp(u / overlaps[i - 1], max=1.0)
        if i < len(starts) - 1 and overlaps[i] > 0:
            q *= torch.clamp((length - u) / overlaps[i], max=1.0)
        weights[i, start:start + length] = q
    denominator = weights.sum(dim=0)
    assert torch.all(denominator > 0)
    return weights / denominator


def _compose_oracle(model, tile_rows, height, width):
    y_idx, y_len, y_overlap = model.split_tiles(height)
    x_idx, x_len, x_overlap = model.split_tiles(width)
    wy = _axis_oracle(y_idx, y_len, y_overlap, height)
    wx = _axis_oracle(x_idx, x_len, x_overlap, width)
    first = tile_rows[0][0]
    out = torch.zeros(
        *first.shape[:-2], height, width, dtype=torch.float64, device=first.device
    )
    for i, row in enumerate(tile_rows):
        local_y = wy[i, y_idx[i]:y_idx[i] + y_len[i]]
        for j, tile in enumerate(row):
            local_x = wx[j, x_idx[j]:x_idx[j] + x_len[j]]
            weight = local_y.view(-1, 1) * local_x.view(1, -1)
            out[..., y_idx[i]:y_idx[i] + y_len[i], x_idx[j]:x_idx[j] + x_len[j]] += (
                tile.to(torch.float64) * weight
            )
    return out


def _run_tile_rows(model, tile_rows, height, width, dtype=torch.float32):
    rows = [[tile.clone() for tile in row] for row in tile_rows]
    row_index = 0

    def decode_row(_z_row, _x_idx, _x_len):
        nonlocal row_index
        row = rows[row_index]
        row_index += 1
        return iter(row)

    if len(rows) == 1 and len(rows[0]) == 1:
        model._decode_pixels = MagicMock(return_value=rows[0][0])
    else:
        model._decode_tile_row = decode_row

    z = torch.zeros((tile_rows[0][0].shape[0], 24, 1, height // 16, width // 16), dtype=dtype)
    before = z.clone()
    out = model.tiled_decode(z)
    torch.testing.assert_close(z, before, rtol=0, atol=0)

    if len(rows) == 1 and len(rows[0]) == 1:
        model._decode_pixels.assert_called_once_with(z)
    else:
        assert row_index == len(rows)
    return out


def _constant_rows(model, height, width, fn, batch=1, dtype=torch.float32):
    y_idx, y_len, _ = model.split_tiles(height)
    x_idx, x_len, _ = model.split_tiles(width)
    return [
        [
            torch.full((batch, 1, 1, yl, xl), float(fn(i, j)), dtype=dtype)
            for j, xl in enumerate(x_len)
        ]
        for i, yl in enumerate(y_len)
    ]


def _fp32_bound(tile_rows):
    terms = len(tile_rows) * len(tile_rows[0])
    maximum = max(float(tile.abs().max()) for row in tile_rows for tile in row)
    return 32 * torch.finfo(torch.float32).eps * max(1.0, maximum) * max(1, terms)


def test_normalized_composition_keeps_x_invariant_across_y_blend():
    model = _bare_model()
    rows = _constant_rows(model, 448, 448, lambda i, _j: i)
    out = _run_tile_rows(model, rows, 448, 448)

    torch.testing.assert_close(out[..., 224, :], torch.full_like(out[..., 224, :], 0.5), rtol=0, atol=2e-7)
    assert float(torch.max(torch.abs(out[..., 224, 1:] - out[..., 224, :-1]))) <= 2e-7


def test_normalized_composition_keeps_y_invariant_across_x_blend():
    model = _bare_model()
    rows = _constant_rows(model, 448, 448, lambda _i, j: j)
    out = _run_tile_rows(model, rows, 448, 448)

    torch.testing.assert_close(out[..., :, 224], torch.full_like(out[..., :, 224], 0.5), rtol=0, atol=2e-7)
    assert float(torch.max(torch.abs(out[..., 1:, 224] - out[..., :-1, 224]))) <= 2e-7


def test_diagonal_tile_has_bilinear_weight_at_overlap_intersection():
    model = _bare_model()
    values = ((0.0, 10.0), (20.0, 30.0))
    rows = _constant_rows(model, 448, 448, lambda i, j: values[i][j])
    out = _run_tile_rows(model, rows, 448, 448)

    assert float(out[0, 0, 0, 224, 224]) == pytest.approx(15.0, abs=2e-6)


@pytest.mark.parametrize(
    ("length", "overlap"),
    [(464, 64), (1216, 128), (512, 240)],
)
def test_triple_and_higher_overlap_matches_all_contributor_oracle(length, overlap):
    model = _bare_model(overlap)
    rows = _constant_rows(model, 16, length, lambda _i, j: j)
    out = _run_tile_rows(model, rows, 16, length)
    expected = _compose_oracle(model, rows, 16, length)

    torch.testing.assert_close(out.double(), expected, rtol=0, atol=_fp32_bound(rows))


@pytest.mark.parametrize(
    ("height", "width", "overlap"),
    [(464, 464, 64), (896, 1216, 128)],
)
def test_two_dimensional_multi_overlap_matches_all_contributor_oracle(height, width, overlap):
    model = _bare_model(overlap)
    rows = _constant_rows(model, height, width, lambda i, j: i * 10 + j)
    out = _run_tile_rows(model, rows, height, width)
    expected = _compose_oracle(model, rows, height, width)

    torch.testing.assert_close(out.double(), expected, rtol=0, atol=_fp32_bound(rows))


def test_float64_weight_oracle_partition_sweep():
    worst = 0.0
    cases = 0
    for overlap in (64, 128, 240):
        model = _bare_model(overlap)
        for length in range(16, 2065, 16):
            starts, lengths, overlaps = model.split_tiles(length)
            weights = _axis_oracle(starts, lengths, overlaps, length)
            error = float((weights.sum(dim=0) - 1.0).abs().max())
            worst = max(worst, error)
            cases += 1

    assert cases == 387
    assert worst <= 1e-12


@pytest.mark.parametrize("overlap", [0, 64, 128, 240])
@pytest.mark.parametrize("length", [16, 256, 272, 448, 464, 512, 1216, 2064])
def test_axis_weights_are_nonnegative_partition_of_unity(length, overlap):
    model = _bare_model(overlap)
    starts, lengths, overlaps = model.split_tiles(length)
    weights = model._tile_axis_weights(starts, lengths, overlaps, length, torch.device("cpu"))

    reconstructed = torch.zeros(length, dtype=torch.float32)
    for start, weight in zip(starts, weights):
        assert weight.dtype == torch.float32
        assert bool(torch.all(weight >= 0))
        reconstructed[start:start + weight.numel()] += weight
    torch.testing.assert_close(reconstructed, torch.ones_like(reconstructed), rtol=0, atol=4 * torch.finfo(torch.float32).eps)


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (16, 16),
        (256, 256),
        (256, 448),
        (448, 256),
        (272, 272),
        (464, 464),
        (512, 512),
        (1216, 896),
        (896, 1216),
    ],
)
def test_constant_tiles_reproduce_constant_without_gaps_or_offsets(height, width):
    model = _bare_model()
    rows = _constant_rows(model, height, width, lambda _i, _j: 0.375)
    out = _run_tile_rows(model, rows, height, width)

    assert tuple(out.shape[-2:]) == (height, width)
    torch.testing.assert_close(out, torch.full_like(out, 0.375), rtol=0, atol=2e-7)


def test_batch_two_and_float16_output_dtype_match_oracle():
    model = _bare_model()
    rows = _constant_rows(model, 464, 464, lambda i, j: 1 + i * 3 + j, batch=2, dtype=torch.float16)
    for row in rows:
        for tile in row:
            tile[1].mul_(0.5)
    out = _run_tile_rows(model, rows, 464, 464, dtype=torch.float16)
    expected = _compose_oracle(model, rows, 464, 464)

    assert out.dtype == torch.float16
    cast_expected = expected.to(torch.float16)
    tolerance = torch.finfo(torch.float16).eps * max(1.0, float(cast_expected.abs().max()))
    torch.testing.assert_close(out.double(), cast_expected.double(), rtol=0, atol=tolerance)


def test_single_spatial_tile_is_direct_decoder_result_without_recomposition():
    model = _bare_model()
    z = torch.randn((1, 24, 2, 16, 16), dtype=torch.float32)
    raw = torch.randn((1, 3, 8, 256, 256), dtype=torch.float32)
    model._decode_pixels = MagicMock(return_value=raw)

    out = model.tiled_decode(z)

    model._decode_pixels.assert_called_once_with(z)
    assert out.data_ptr() == raw.data_ptr()
    assert out.dtype == raw.dtype


def test_decode_tile_row_preserves_exact_planned_windows(monkeypatch):
    model = _bare_model()
    monkeypatch.setattr(vae_mod.comfy.model_management, "get_free_memory", lambda _device: 0)
    height, width = 896, 1216
    y_idx, y_len, _ = model.split_tiles(height)
    x_idx, x_len, _ = model.split_tiles(width)
    z = torch.arange(24 * (height // 16) * (width // 16), dtype=torch.float32).reshape(
        1, 24, 1, height // 16, width // 16
    )
    calls = []

    def decode_pixels(tile):
        calls.append(tile.clone())
        return torch.zeros((tile.shape[0], 3, 4, tile.shape[-2] * 16, tile.shape[-1] * 16))

    model._decode_pixels = decode_pixels
    yi, yl = y_idx[2], y_len[2]
    z_row = z[..., yi // 16:(yi + yl) // 16, :]
    list(model._decode_tile_row(z_row, x_idx, x_len))

    assert len(calls) == len(x_idx)
    for call, start, length in zip(calls, x_idx, x_len):
        expected = z_row[..., start // 16:(start + length) // 16]
        torch.testing.assert_close(call, expected, rtol=0, atol=0)


def test_noncontiguous_input_and_global_affine_decoder_preserve_coordinates(monkeypatch):
    model = _bare_model()
    monkeypatch.setattr(vae_mod.comfy.model_management, "get_free_memory", lambda _device: 0)
    latent_h, latent_w = 28, 29
    yy = torch.arange(latent_h, dtype=torch.float32).view(1, 1, 1, latent_h, 1)
    xx = torch.arange(latent_w, dtype=torch.float32).view(1, 1, 1, 1, latent_w)
    base = yy * 100 + xx
    z = base.repeat(1, 24, 1, 1, 1).transpose(-1, -2)
    assert not z.is_contiguous()

    def decode_pixels(tile):
        data = tile[:, :1].repeat(1, 3, 4, 1, 1)
        return data.repeat_interleave(16, -2).repeat_interleave(16, -1)

    model._decode_pixels = decode_pixels
    out = model.tiled_decode(z)
    expected = decode_pixels(z)

    maximum = float(expected.abs().max())
    tolerance = 32 * torch.finfo(torch.float32).eps * max(1.0, maximum) * 9
    torch.testing.assert_close(out, expected, rtol=0, atol=tolerance)


def test_repeat_and_interleaved_calls_do_not_share_composition_state():
    model64 = _bare_model(64)
    model128 = _bare_model(128)
    rows64 = _constant_rows(model64, 464, 464, lambda i, j: i * 10 + j)
    rows128 = _constant_rows(model128, 464, 464, lambda i, j: i * 10 + j)

    first = _run_tile_rows(model64, rows64, 464, 464)
    middle = _run_tile_rows(model128, rows128, 464, 464)
    second = _run_tile_rows(model64, rows64, 464, 464)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(middle.double(), _compose_oracle(model128, rows128, 464, 464), rtol=0, atol=_fp32_bound(rows128))


def test_failure_leaves_no_partial_composition_state_on_model():
    model = _bare_model()
    calls = 0

    def fail_row(_z_row, _x_idx, _x_len):
        nonlocal calls
        calls += 1
        if calls == 1:
            def generator():
                yield torch.zeros((1, 1, 1, 256, 256))
                raise RuntimeError("injected")
            return generator()
        raise AssertionError("unexpected row")

    model._decode_tile_row = fail_row
    z = torch.zeros((1, 24, 1, 28, 28))
    with pytest.raises(RuntimeError, match="injected"):
        model.tiled_decode(z)

    rows = _constant_rows(model, 448, 448, lambda i, j: i + j)
    clean = _run_tile_rows(model, rows, 448, 448)
    torch.testing.assert_close(clean.double(), _compose_oracle(model, rows, 448, 448), rtol=0, atol=_fp32_bound(rows))


def _temporal_model():
    model = _bare_model()
    model.clip_length = 17
    model.token_drop = 3
    model.frame_pre_padding = (-model.clip_length) % model.vae_ratio_t
    model.tokens_chunk_size = math.ceil(model.clip_length / model.vae_ratio_t)
    model.token_overlap = (-model.token_drop) % model.tokens_chunk_size
    model.frame_overlap = max(model.token_overlap * model.vae_ratio_t - model.frame_pre_padding, 0)
    model.register_buffer("latents_mean", torch.zeros(24))
    model.register_buffer("latents_std", torch.ones(24))
    model.register_buffer("pixel_mean", torch.zeros((1, 3, 1, 1, 1)), persistent=False)
    model.register_buffer("pixel_std", torch.ones((1, 3, 1, 1, 1)), persistent=False)
    model.decoder = SimpleNamespace(out_channels=3)

    def decode_pixels(tile):
        return torch.full(
            (tile.shape[0], 3, tile.shape[2] * 4, tile.shape[-2] * 16, tile.shape[-1] * 16),
            0.25,
            dtype=tile.dtype,
            device=tile.device,
        )

    model._decode_pixels = decode_pixels
    return model


@pytest.mark.parametrize("latent_t", [1, 2, 7, 12])
def test_decode_output_buffer_identity_full_overwrite_and_temporal_shape(latent_t):
    model = _temporal_model()
    z = torch.zeros((1, 24, latent_t, 16, 16), dtype=torch.float32)
    expected_shape = model.decode_output_shape(z.shape)
    output = torch.full(expected_shape, float("nan"), dtype=torch.float32)

    returned = model.decode(z, output_buffer=output)

    assert returned.data_ptr() == output.data_ptr()
    assert tuple(returned.shape) == expected_shape
    assert returned.dtype == torch.float32
    assert bool(torch.isfinite(returned).all())
    torch.testing.assert_close(returned, torch.full_like(returned, 0.25), rtol=0, atol=0)


def test_decode_output_buffer_uses_compositor_and_fully_overwrites():
    model = _temporal_model()
    z = torch.zeros((1, 24, 1, 17, 17), dtype=torch.float32)
    expected_shape = model.decode_output_shape(z.shape)
    output = torch.full(expected_shape, float("nan"), dtype=torch.float32)

    returned = model.decode(z, output_buffer=output)

    assert returned.data_ptr() == output.data_ptr()
    assert tuple(returned.shape) == (1, 3, 1, 272, 272)
    assert bool(torch.isfinite(returned).all())
    expected = torch.full_like(returned, 0.25)
    tolerance = 12 * torch.finfo(torch.float32).eps * 0.25
    torch.testing.assert_close(returned, expected, rtol=0, atol=tolerance)


def test_streamed_float32_composition_matches_float64_oracle_with_roundoff_bound():
    model = _bare_model()
    rows = _constant_rows(model, 896, 1216, lambda i, j: (i + 1) * 0.125 + (j + 1) * 0.0625)
    out = _run_tile_rows(model, rows, 896, 1216)
    expected = _compose_oracle(model, rows, 896, 1216)

    torch.testing.assert_close(out.double(), expected, rtol=0, atol=_fp32_bound(rows))
