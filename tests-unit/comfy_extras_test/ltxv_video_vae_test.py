from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_lt import LTXVAddGuide, LTXVImgToVideoInplace, video_vae_downscale_factors  # noqa: E402


IMAGE_VAE = SimpleNamespace(downscale_index_formula=None)


def _latent():
    return {"samples": torch.zeros(1, 128, 3, 4, 4)}


def test_img_to_video_inplace_names_the_wrong_vae():
    with pytest.raises(ValueError, match="needs a video VAE"):
        LTXVImgToVideoInplace.execute(IMAGE_VAE, torch.zeros(1, 128, 128, 3), _latent(), 1.0)


def test_add_guide_names_the_wrong_vae():
    with pytest.raises(ValueError, match="needs a video VAE"):
        LTXVAddGuide.execute([], [], IMAGE_VAE, _latent(), torch.zeros(1, 128, 128, 3), 0, 1.0)


def test_video_vae_still_unpacks():
    assert video_vae_downscale_factors(SimpleNamespace(downscale_index_formula=(8, 32, 32))) == (8, 32, 32)
