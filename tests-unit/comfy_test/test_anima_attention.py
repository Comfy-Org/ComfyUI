import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.anima.model as anima_model  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.ldm.anima.model import Attention  # noqa: E402
from comfy.ldm.modules.attention import attention_sub_quad  # noqa: E402


def make_attention():
    attn = Attention(
        query_dim=8,
        context_dim=8,
        n_heads=2,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        operations=comfy.ops.disable_weight_init,
    )
    # disable_weight_init leaves parameters as uninitialized (torch.empty)
    # memory since it expects a checkpoint to populate them; give them real
    # values since no checkpoint is loaded here.
    for p in attn.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return attn


def test_attention_does_not_call_raw_sdpa_directly(monkeypatch):
    # Anima previously called torch's SDPA directly, bypassing ComfyUI's
    # optimized_attention dispatch. On hardware/driver combos where ComfyUI's
    # own capability detection has already determined SDPA is broken (e.g.
    # ROCm gfx1201, see #16526), that raw call is exactly what crashes with
    # hipErrorInvalidValue. Simulate a broken SDPA and confirm Anima no
    # longer depends on it. Force a non-SDPA backend for the dispatch itself,
    # since on hardware where ComfyUI legitimately selects attention_pytorch
    # this simulated failure would also break the shared dispatch.
    def broken_sdpa(*args, **kwargs):
        raise RuntimeError("simulated hipErrorInvalidValue")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", broken_sdpa)
    monkeypatch.setattr(anima_model, "optimized_attention_masked", attention_sub_quad)

    attn = make_attention()
    x = torch.randn(1, 5, 8)
    out = attn(x)

    assert out.shape == (1, 5, 8)
    assert torch.isfinite(out).all()
