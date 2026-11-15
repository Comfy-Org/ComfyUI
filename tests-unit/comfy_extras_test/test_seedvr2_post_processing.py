from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras import nodes_seedvr  # noqa: E402


def _schema_ids(items):
    return [item.id for item in items]


def test_seedvr2_post_processing_schema():
    schema = nodes_seedvr.SeedVR2PostProcessing.define_schema()

    assert _schema_ids(schema.inputs) == ["images", "original_resized_images", "color_correction_method"]
    assert schema.inputs[2].options == ["lab", "wavelet", "adain", "none"]
    assert schema.inputs[2].default == "lab"
    assert schema.outputs[0].get_io_type() == "IMAGE"


def test_seedvr2_post_processing_oom_error_uses_color_correction_method(monkeypatch):
    decoded = torch.full((1, 3, 4, 4), 0.25)
    reference = torch.full((1, 3, 4, 4), 0.75)

    def _lab(content, style):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(nodes_seedvr.comfy.model_management, "vae_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes_seedvr.comfy.model_management, "get_free_memory", lambda device: 1_000_000)

    with patch.object(nodes_seedvr, "lab_color_transfer", _lab):
        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr.SeedVR2PostProcessing._color_transfer_chunked(
                decoded, reference, torch.device("cpu"), "lab",
            )
    assert "color_correction_method=lab" in str(excinfo.value)
    assert " method=lab" not in str(excinfo.value)


def test_seedvr2_post_processing_unknown_color_correction_method_raises():
    decoded = torch.zeros(1, 2, 4, 4, 3)
    original = torch.zeros(1, 2, 4, 4, 3)
    with pytest.raises(ValueError) as excinfo:
        nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, "bogus")
    assert "color_correction_method" in str(excinfo.value)
