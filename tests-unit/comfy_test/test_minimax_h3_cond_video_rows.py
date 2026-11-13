import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.common_dit
import comfy.ldm.minimax.model as minimax_model


def test_cond_video_rows_pads_odd_spatial_grid():
    model = minimax_model.MiniMaxH3Model.__new__(minimax_model.MiniMaxH3Model)
    model.patch_size = (1, 2, 2)

    # odd latent grid (e.g. from a source width/height whose VAE-cropped
    # size divides down to an odd value) must not crash patchify_video
    z = torch.randn(1, 24, 1, 31, 33)
    payload = {"cond_video_latents": [z], "visual_cond_noise_aug": 1.0}

    rows = model._cond_video_rows(payload, torch.device("cpu"))

    padded_t, padded_h, padded_w = 1, 32, 34
    expected_rows = padded_t * (padded_h // 2) * (padded_w // 2)
    expected_dim = 24 * 1 * 2 * 2
    assert rows.shape == (expected_rows, expected_dim)


def test_cond_video_rows_aligned_grid_is_noop():
    model = minimax_model.MiniMaxH3Model.__new__(minimax_model.MiniMaxH3Model)
    model.patch_size = (1, 2, 2)

    # already patch-aligned latent: padding must not change the patchified output
    z = torch.randn(1, 24, 1, 32, 34)
    payload = {"cond_video_latents": [z], "visual_cond_noise_aug": 1.0}

    rows = model._cond_video_rows(payload, torch.device("cpu"))
    expected = minimax_model.patchify_video(z, model.patch_size)
    assert torch.equal(rows, expected)


def test_packed_layout_ref_image_row_count_matches_padded_patchify():
    # PackedLayout must size the "ref_img" segment for an odd-latent ref block
    # to match the row count _cond_video_rows actually produces after padding
    # to the patch grid, or all_video_rows[~img_update] = cond_video_rows in
    # MiniMaxH3Model._forward crashes on a shape mismatch.
    latent_h, latent_w = 5, 7
    refs = [{"kind": "image", "latent_h": latent_h, "latent_w": latent_w}]
    layout = minimax_model.PackedLayout(text_len=4, latent_t=1, latent_h=8, latent_w=8,
                                        audio_t=0, refs=refs)
    ref_rows = next(b - a for a, b, kind in layout.segments if kind == "ref_img")

    z = torch.randn(1, 24, 1, latent_h, latent_w)
    padded = comfy.ldm.common_dit.pad_to_patch_size(z, (1, 2, 2))
    actual_rows = minimax_model.patchify_video(padded, (1, 2, 2)).shape[0]

    assert ref_rows == actual_rows
