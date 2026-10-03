import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.minimax.vae import create_token_ids  # noqa: E402


def test_create_token_ids_defaults_match_full_range():
    # full_dims/offset omitted must reproduce the original (pre-tiling-fix) behavior.
    baseline = create_token_ids((1, 4, 4), device="cpu", dtype=torch.float32)
    same = create_token_ids((1, 4, 4), device="cpu", dtype=torch.float32, full_dims=(1, 4, 4), offset=(0, 0, 0))
    torch.testing.assert_close(baseline, same)


def test_create_token_ids_tile_offset_matches_global_slice():
    # A spatial tile decoded at its true offset within the full latent grid must get the
    # same position ids as the corresponding slice of a single-shot full-grid decode.
    # Without full_dims/offset, every tile is normalized to its own local -1..1 range
    # regardless of where it sits in the frame, which is the root cause of ComfyUI/#15548
    # (MiniMax H3 tiled decode producing a grid of independently textured tiles).
    full = create_token_ids((1, 8, 8), device="cpu", dtype=torch.float32).view(1, 8, 8, 3)

    top_left = create_token_ids(
        (1, 4, 4), device="cpu", dtype=torch.float32, full_dims=(1, 8, 8), offset=(0, 0, 0)
    ).view(1, 4, 4, 3)
    bottom_right = create_token_ids(
        (1, 4, 4), device="cpu", dtype=torch.float32, full_dims=(1, 8, 8), offset=(0, 4, 4)
    ).view(1, 4, 4, 3)

    torch.testing.assert_close(top_left, full[:, 0:4, 0:4, :])
    torch.testing.assert_close(bottom_right, full[:, 4:8, 4:8, :])
