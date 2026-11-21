"""Regression test for aliased intermediate_output="all" snapshots (issue #16610).

TransformerBlockGemma2 writes each layer's result back into the storage of the
tensor it was given (`output = x` ... `torch.add(residual, x, out=output)`), so the
same hidden-state buffer is reused for every layer. The per-layer snapshot taken
by Llama2_.forward relies on `.clone()` to detach a real copy from that buffer
before the next layer overwrites it. A tensor subclass that overrides `clone()`
to return `self` (as ComfyUI-GGUF's GGMLTensor does) turns every snapshot into a
view of the same buffer, so all of them end up showing the final layer's output.
"""

import types

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops as comfy_ops  # noqa: E402
import comfy.text_encoders.llama as llama  # noqa: E402


class _CloneAliasesSelfTensor(torch.Tensor):
    """Stands in for ComfyUI-GGUF's GGMLTensor, whose clone() returns self."""

    def clone(self, *args, **kwargs):
        return self


def _build_tiny_gemma3_model():
    config = types.SimpleNamespace(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=1e-6,
        rms_norm_add=True,
        mlp_activation="gelu_pytorch_tanh",
        qkv_bias=False,
        rope_dims=None,
        q_norm="gemma3",
        k_norm="gemma3",
        transformer_type="gemma3",
        sliding_attention=[False, False, False],
        rope_theta=[1000000.0, 10000.0],
        rope_scale=[8.0, 1.0],
        final_norm=True,
        lm_head=False,
    )
    model = llama.Llama2_(config, device="cpu", dtype=torch.float32, ops=comfy_ops.disable_weight_init)
    model.eval()

    generator = torch.Generator().manual_seed(0)
    for p in model.parameters():
        p.data.copy_(torch.empty(p.shape).normal_(0, 0.02, generator=generator))
        p.requires_grad_(False)
    return model


def test_all_intermediate_layers_are_not_aliased_when_clone_is_a_no_op():
    model = _build_tiny_gemma3_model()
    embeds = torch.randn(1, 5, 8).as_subclass(_CloneAliasesSelfTensor)

    _, intermediate = model(None, embeds=embeds, intermediate_output="all", final_layer_norm_intermediate=False)

    # intermediate has shape (batch, num_layers + 1, seq, hidden); entries 1 and 2
    # are the hidden state captured before layer 1 and before layer 2 respectively.
    # They must reflect different amounts of transformer processing.
    assert not torch.equal(intermediate[:, 1], intermediate[:, 2]), (
        "per-layer intermediate snapshots aliased the same buffer instead of "
        "each holding the hidden state at its own layer"
    )
