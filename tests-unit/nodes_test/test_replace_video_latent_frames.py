import logging

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

# nodes_post_processing imports from nodes_latent while nodes_latent imports it back, so it has to be
# the module that starts the pair.
import comfy_extras.nodes_post_processing  # noqa: E402,F401
from comfy_extras.nodes_latent import ReplaceVideoLatentFrames  # noqa: E402


def _latent(batch: int, frames: int) -> dict:
    return {"samples": torch.zeros(batch, 4, frames, 8, 8)}


def test_mismatched_source_batch_returns_destination_unchanged(caplog):
    destination = _latent(2, 4)
    source = _latent(3, 2)

    with caplog.at_level(logging.WARNING):
        result = ReplaceVideoLatentFrames.execute(destination, 0, source=source)

    assert torch.equal(result.result[0]["samples"], destination["samples"])
    assert "batch 3 does not match destination latent batch 2" in caplog.text


def test_single_item_source_still_broadcasts_over_the_destination_batch():
    destination = _latent(3, 4)
    source = _latent(1, 2)
    source["samples"].fill_(1.0)

    result = ReplaceVideoLatentFrames.execute(destination, 0, source=source)

    samples = result.result[0]["samples"]
    assert torch.equal(samples[:, :, 0:2], torch.ones(3, 4, 2, 8, 8))
    assert torch.equal(samples[:, :, 2:4], torch.zeros(3, 4, 2, 8, 8))


def test_matching_batch_replaces_only_the_source_frames():
    destination = _latent(2, 4)
    source = _latent(2, 2)
    source["samples"].fill_(1.0)

    result = ReplaceVideoLatentFrames.execute(destination, 1, source=source)

    samples = result.result[0]["samples"]
    assert torch.equal(samples[:, :, 1:3], torch.ones(2, 4, 2, 8, 8))
    assert torch.equal(samples[:, :, 0:1], torch.zeros(2, 4, 1, 8, 8))
    assert torch.equal(samples[:, :, 3:4], torch.zeros(2, 4, 1, 8, 8))
