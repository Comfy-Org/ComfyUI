"""First-frame anchoring of LTXVImgToVideo output through the real sampling stack.

``LTXVImgToVideo`` anchors the encoded image by zeroing ``noise_mask`` on its latent
frames. These tests build that output with the real node and run it through
``CFGGuider -> KSAMPLER -> KSamplerX0Inpaint -> LTXV`` with a small ``LTXVModel``,
asserting the anchored frames leave the sampler as exactly the conditioning latent.

The third test reproduces the failure signature of issue #17618: the report's output
holds non-finite values (VideoHelperSuite logs ``invalid value encountered in cast``
while converting frames that render black). A non-finite prediction at the conditioned
frames leaks through ``nan * 0`` in KSamplerX0Inpaint's mask blend and destroys the
anchor, which the test asserts must not happen.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

# Stub nodes/server for the import only, then restore exactly those keys. Same shape as
# tests-unit/comfy_extras_test/nodes_lt_test.py (see the comment there for why).
_stubs = {"nodes": MagicMock(MAX_RESOLUTION=16384), "server": MagicMock()}
_saved = {name: sys.modules.get(name) for name in _stubs}
sys.modules.update(_stubs)
try:
    import comfy_extras.nodes_lt as nodes_lt
finally:
    for _name, _original in _saved.items():
        if _original is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _original

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import comfy.sample
import comfy.samplers
from comfy.ldm.lightricks.model import LTXVModel

LATENT_CHANNELS = 128
FRAMES, HEIGHT, WIDTH = 4, 2, 3
ANCHOR_FRAMES = 1  # one encoded image is one latent frame


class _ModelConfig:
    unet_config = dict(
        in_channels=LATENT_CHANNELS,
        cross_attention_dim=16,
        attention_head_dim=8,
        num_attention_heads=2,
        caption_channels=16,
        num_layers=1,
        dtype=torch.float32,
    )
    latent_format = comfy.latent_formats.LTXV()
    manual_cast_dtype = None
    custom_operations = torch.nn
    optimizations = {}
    sampling_settings = {}
    memory_usage_factor = 1.0
    model_type = "ltxv"

    @staticmethod
    def get_dtype(*args, **kwargs):
        return torch.float32


def _tiny_ltxv():
    torch.manual_seed(0)
    model = comfy.model_base.LTXV.__new__(comfy.model_base.LTXV)
    comfy.model_base.BaseModel.__init__(
        model,
        _ModelConfig(),
        model_type=comfy.model_base.ModelType.FLOW,
        device=torch.device("cpu"),
        unet_model=LTXVModel,
    )
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 0:
                torch.nn.init.normal_(p, std=0.02)
            else:
                p.zero_()
    return model


def _image_to_video_latent():
    """The real node's output: the encoded image in frame 0, noise_mask zeroed there."""
    vae = SimpleNamespace(
        encode=lambda pixels: torch.randn(1, LATENT_CHANNELS, ANCHOR_FRAMES, pixels.shape[1] // 32, pixels.shape[2] // 32)
    )
    image = torch.rand(1, HEIGHT * 32, WIDTH * 32, 3)
    _, _, latent = nodes_lt.LTXVImgToVideo.execute(
        positive=[],
        negative=[],
        image=image,
        vae=vae,
        width=WIDTH * 32,
        height=HEIGHT * 32,
        length=(FRAMES - 1) * 8 + 1,
        batch_size=1,
        strength=1.0,
    )
    return latent


def _sample(latent, poison_conditioned_frames=False):
    torch.manual_seed(1)
    model = _tiny_ltxv()
    if poison_conditioned_frames:
        real_forward = model.diffusion_model._forward

        def poisoned_forward(*fargs, **fkwargs):
            out = real_forward(*fargs, **fkwargs)
            return torch.cat(
                [torch.full_like(out[:, :, :ANCHOR_FRAMES], float("nan")), out[:, :, ANCHOR_FRAMES:]],
                dim=2,
            )

        model.diffusion_model._forward = poisoned_forward

    patcher = comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    cond = [[torch.randn(1, 4, 16), {"attention_mask": torch.ones(1, 4, dtype=torch.long)}]]
    guider = comfy.samplers.CFGGuider(patcher)
    guider.set_conds(cond, cond)
    guider.set_cfg(1.0)
    sampler = comfy.samplers.ksampler("euler", {})
    sigmas = torch.FloatTensor([1.0, 0.75, 0.5, 0.25, 0.0])
    noise = comfy.sample.prepare_noise(latent["samples"], 42)
    return guider.sample(
        noise,
        latent["samples"].clone(),
        sampler,
        sigmas,
        denoise_mask=latent["noise_mask"].clone(),
        seed=42,
    )


def test_node_masks_only_the_encoded_frames():
    latent = _image_to_video_latent()
    mask = latent["noise_mask"]
    assert mask.shape == (1, 1, FRAMES, 1, 1)
    assert (mask[:, :, :ANCHOR_FRAMES] == 0.0).all()
    assert (mask[:, :, ANCHOR_FRAMES:] == 1.0).all()
    assert latent["samples"][:, :, :ANCHOR_FRAMES].abs().sum() > 0


def test_anchored_frames_are_restored_exactly():
    latent = _image_to_video_latent()
    out = _sample(latent)
    torch.testing.assert_close(out[:, :, :ANCHOR_FRAMES], latent["samples"][:, :, :ANCHOR_FRAMES], atol=1e-6, rtol=0)
    assert (out[:, :, ANCHOR_FRAMES:] - latent["samples"][:, :, ANCHOR_FRAMES:]).abs().max() > 0


def test_anchored_frames_survive_non_finite_predictions():
    latent = _image_to_video_latent()
    out = _sample(latent, poison_conditioned_frames=True)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, :, :ANCHOR_FRAMES], latent["samples"][:, :, :ANCHOR_FRAMES], atol=1e-6, rtol=0)
