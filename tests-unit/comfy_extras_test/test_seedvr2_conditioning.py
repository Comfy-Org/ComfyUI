"""SeedVR2 conditioning node regression tests."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args
from comfy.ldm.seedvr.constants import SEEDVR2_LATENT_CHANNELS

if not torch.cuda.is_available():
    cli_args.cpu = True


_SENTINEL = object()
_TARGETS = (
    ("comfy.model_management", "comfy"),
    ("comfy_extras.nodes_seedvr", "comfy_extras"),
)


def _import_nodes_seedvr_isolated():
    """Import comfy_extras.nodes_seedvr with comfy.model_management mocked."""
    priors = []
    for mod_name, parent_name in _TARGETS:
        prior_mod = sys.modules.get(mod_name, _SENTINEL)
        parent = sys.modules.get(parent_name)
        attr = mod_name.split(".")[-1]
        prior_attr = (
            getattr(parent, attr, _SENTINEL) if parent is not None else _SENTINEL
        )
        priors.append((mod_name, parent_name, attr, prior_mod, prior_attr))

    mock_mm = MagicMock()
    for fn in (
        "xformers_enabled", "xformers_enabled_vae",
        "pytorch_attention_enabled", "pytorch_attention_enabled_vae",
        "sage_attention_enabled", "flash_attention_enabled",
        "is_intel_xpu",
    ):
        getattr(mock_mm, fn).return_value = False
    tv = torch.version.__version__.split(".")
    mock_mm.torch_version_numeric = (int(tv[0]), int(tv[1]))
    mock_mm.WINDOWS = False
    sys.modules["comfy.model_management"] = mock_mm
    if sys.modules.get("comfy") is None:
        importlib.import_module("comfy")
    comfy_pkg = sys.modules.get("comfy")
    if comfy_pkg is not None:
        setattr(comfy_pkg, "model_management", mock_mm)
    nodes_seedvr = sys.modules.get("comfy_extras.nodes_seedvr") or (
        importlib.import_module("comfy_extras.nodes_seedvr")
    )

    def _restore():
        for mod_name, parent_name, attr, prior_mod, prior_attr in priors:
            if prior_mod is _SENTINEL:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = prior_mod
            parent = sys.modules.get(parent_name)
            if parent is None:
                continue
            if prior_attr is _SENTINEL:
                if hasattr(parent, attr):
                    delattr(parent, attr)
            else:
                setattr(parent, attr, prior_attr)

    return nodes_seedvr, _restore


class _Rope(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = nn.Parameter(torch.zeros(4))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = _Rope()


class _DiffusionModel(nn.Module):
    def __init__(self, n_blocks=3, conditioning_dtype=torch.float32):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])
        self.register_buffer("positive_conditioning", torch.ones((2, 4), dtype=conditioning_dtype))
        self.register_buffer("negative_conditioning", torch.zeros((3, 4), dtype=conditioning_dtype))


class _ModelInner:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class _ModelPatcher:
    def __init__(self, diffusion_model):
        self.model = _ModelInner(diffusion_model)


def test_seedvr2_conditioning_schema_exposes_conditioning_outputs():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        schema = nodes_seedvr.SeedVR2Conditioning.define_schema()
        assert [input_item.id for input_item in schema.inputs] == [
            "model",
            "vae_conditioning",
        ]
        assert schema.inputs[1].display_name == "latent"
        assert [output.display_name for output in schema.outputs] == [
            "positive",
            "negative",
        ]
    finally:
        restore()


def test_seedvr2_conditioning_rejects_wrong_latent_channels():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        patcher = _ModelPatcher(_DiffusionModel())
        vae_conditioning = {"samples": torch.zeros(1, 8, 2, 2, 2)}

        with pytest.raises(ValueError, match=f"{SEEDVR2_LATENT_CHANNELS} channels"):
            nodes_seedvr.SeedVR2Conditioning.execute(patcher, vae_conditioning)
    finally:
        restore()


def test_seedvr2_conditioning_returns_conditioning_deterministically():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()
        patcher = _ModelPatcher(diffusion_model)
        samples = torch.arange(
            1,
            1 + SEEDVR2_LATENT_CHANNELS * 3 * 2 * 2,
            dtype=torch.float32,
        ).reshape(1, SEEDVR2_LATENT_CHANNELS, 3, 2, 2)
        vae_conditioning = {"samples": samples}

        first_positive, first_negative = (
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher,
                vae_conditioning,
            )
        )
        second_positive, second_negative = (
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher,
                vae_conditioning,
            )
        )

        channel_last = samples.movedim(1, -1).contiguous()
        expected_condition = torch.cat(
            [
                channel_last,
                torch.ones((*channel_last.shape[:-1], 1)),
            ],
            dim=-1,
        ).movedim(-1, 1)

        assert torch.equal(
            first_positive[0][1]["condition"],
            expected_condition,
        )
        assert torch.equal(
            second_positive[0][1]["condition"],
            expected_condition,
        )
        assert torch.equal(
            first_negative[0][1]["condition"],
            expected_condition,
        )
        assert torch.equal(
            second_negative[0][1]["condition"],
            expected_condition,
        )
    finally:
        restore()
