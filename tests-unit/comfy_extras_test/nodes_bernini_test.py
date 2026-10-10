from unittest.mock import MagicMock

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_bernini import BerniniConditioning


def _run(reference_images):
    vae = MagicMock()
    vae.encode.side_effect = lambda pixels: pixels.sum()
    positive = [[torch.zeros(1, 1), {}]]
    negative = [[torch.zeros(1, 1), {}]]

    out = BerniniConditioning.execute(
        positive, negative, vae,
        width=16, height=16, length=1, batch_size=1,
        reference_images=reference_images,
    )
    return out, vae


def test_reference_images_as_tensor_batch():
    """A raw IMAGE batch (e.g. wired directly instead of through the autogrow dict)
    must not crash on the ambiguous multi-element tensor boolean check."""
    images = torch.zeros(3, 16, 16, 3)

    (positive, _negative, _latent), vae = _run(images)

    assert vae.encode.call_count == 3
    assert len(positive[0][1]["context_latents"]) == 3


def test_reference_images_as_dict():
    images = {"reference_image_0": torch.zeros(2, 16, 16, 3)}

    (positive, _negative, _latent), vae = _run(images)

    assert vae.encode.call_count == 2
    assert len(positive[0][1]["context_latents"]) == 2


def test_reference_images_none():
    (positive, _negative, _latent), vae = _run(None)

    assert vae.encode.call_count == 0
    assert "context_latents" not in positive[0][1]
