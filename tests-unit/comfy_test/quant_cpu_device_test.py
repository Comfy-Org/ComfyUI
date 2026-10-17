import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ops  # noqa: E402


def test_disabled_quant_formats_cpu_device_skips_cuda_checks(monkeypatch):
    def fail(device=None):
        raise ValueError("Expected a cuda device, but got: cpu")

    for name in ("supports_nvfp4_compute", "supports_mxfp8_compute", "supports_fp8_compute", "supports_int8_compute"):
        monkeypatch.setattr(comfy.model_management, name, fail)

    disabled = comfy.ops.get_disabled_quant_formats(torch.device("cpu"))
    assert set(comfy.ops.QUANT_ALGOS) <= disabled
    assert set(comfy.ops._GROUPED_INT8_FORMATS) <= disabled
