import torch

from comfy.cli_args import args as cli_args

# Only force CPU state when neither CUDA nor MPS is available (plain
# CPU-only CI runners), where model_management would otherwise crash at
# import time. Leaving real MPS machines alone avoids permanently
# stamping cpu_state as CPU for the rest of the test session.
if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    cli_args.cpu = True

import comfy.model_management as model_management


def test_force_upcast_includes_bfloat16_on_affected_macos(monkeypatch):
    """LTX 2.5 ships bf16-only checkpoints and renders black frames on MPS
    without this upcast (issue: black video unless --use-split-cross-attention).
    bfloat16 has less mantissa precision than float16, so it needs the same
    macOS attention-upcast workaround as float16."""
    monkeypatch.setattr(model_management, "mac_version", lambda: (15, 5))
    monkeypatch.setattr(model_management.args, "force_upcast_attention", False)

    result = model_management.force_upcast_attention_dtype()

    assert result is not None
    assert result.get(torch.bfloat16) == torch.float32
    assert result.get(torch.float16) == torch.float32
