"""Regression tests for the Flux 3 block gate residual under training.

`comfy/ldm/flux3/model.py::Block.forward` folds each span's gated residual into
the running stream with an in-place `addcmul_` over row slices. The block norm
and the gradient checkpointing recompute both read the same stream, so training
backward raises "one of the variables needed for gradient computation has been
modified by an inplace operation": the slice update is recorded as a CopySlices
node that bumps the stream's version once per block.

The training branch builds the gated residual out of place; inference keeps the
fused in-place update. `Flux3.preprocess_conds` keeps each early block result so
the training branch's new stream tensors are not dropped.

Run:  python -m pytest tests-unit/comfy_test/flux3_block_training_test.py
"""
import pytest
import torch
import torch.utils.checkpoint as cp

import comfy.model_management
import comfy.ops as comfy_ops
from comfy.ldm.flux.layers import EmbedND, ModulationOut
from comfy.ldm.flux3.model import Block, Flux3

NUM_BLOCKS = 32
DIM = 16
HEADS = 2
SEQ = 16

BOUNDS_SINGLE = [(0, SEQ)]
BOUNDS_MULTI = [(0, 3), (3, SEQ)]


def _make_block():
    block = Block(DIM, HEADS, 2.0, fused_qkv=False, dtype=torch.float32,
                  device=torch.device("cpu"), operations=comfy_ops.disable_weight_init)
    torch.manual_seed(0)
    for p in block.parameters():
        torch.nn.init.normal_(p, 0.0, 0.1)
    return block


def _make_model():
    # EmbedND's axes_dim sums to 128, so head_dim has to stay 128
    model = Flux3(hidden_size=128, num_heads=1, depth=2, depth_single_blocks=1, fused_qkv=False,
                  dtype=torch.float32, device=torch.device("cpu"), operations=comfy_ops.disable_weight_init)
    torch.manual_seed(0)
    for p in model.parameters():
        torch.nn.init.normal_(p, 0.0, 0.1)
    return model


def _make_pe():
    rope = EmbedND(dim=DIM // HEADS, theta=10000, axes_dim=[DIM // HEADS // 4] * 4)
    return rope(torch.zeros(SEQ, 4).unsqueeze(0)).transpose(1, 2).contiguous()


def _spans(bounds, gate=0.1):
    # training gates come from the modulation linears, so they carry gradients
    return [(a, b, ModulationOut(shift=torch.zeros(1, 1, DIM), scale=torch.zeros(1, 1, DIM),
                                 gate=torch.full((1, 1, DIM), gate * (1.0 + 0.1 * i)).requires_grad_()))
            for i, (a, b) in enumerate(bounds)]


def _upstream_gate_update(x, out, spans):
    """The pre-fix gate loop, kept here to prove the fixture can fail."""
    for start, end, mod in spans:
        x[:, start:end].addcmul_(out[:, start:end], mod.gate)
    return x


def _run_chain(monkeypatch, bounds, use_checkpoint, reentrant=False):
    """Thread a stream through NUM_BLOCKS block calls and backprop."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    block = _make_block()
    pe = _make_pe()
    spans = _spans(bounds)
    torch.manual_seed(0)
    base = torch.randn(1, SEQ, DIM, requires_grad=True)

    h = base * 1.0
    for _ in range(NUM_BLOCKS):
        if use_checkpoint:
            h = cp.checkpoint(block, h, spans, pe, {}, use_reentrant=reentrant)
        else:
            h = block(h, spans, pe, {})

    h.square().mean().backward()
    return base.grad, [p.grad for p in block.parameters() if p.grad is not None]


@pytest.mark.parametrize("bounds", [BOUNDS_SINGLE, BOUNDS_MULTI])
def test_training_chain_backward_produces_finite_gradients(monkeypatch, bounds):
    base_grad, param_grads = _run_chain(monkeypatch, bounds, use_checkpoint=False)
    assert base_grad is not None
    assert torch.isfinite(base_grad).all()
    assert param_grads and all(torch.isfinite(g).all() for g in param_grads)


@pytest.mark.parametrize("reentrant", [False, True])
def test_training_chain_survives_gradient_checkpointing(monkeypatch, reentrant):
    """The pre-fix in-place loop raises here; the out-of-place path must not."""
    base_grad, param_grads = _run_chain(monkeypatch, BOUNDS_MULTI, use_checkpoint=True, reentrant=reentrant)
    assert base_grad is not None
    assert torch.isfinite(base_grad).all()
    assert param_grads and all(torch.isfinite(g).all() for g in param_grads)


def test_pre_fix_implementation_actually_fails(monkeypatch):
    """Guards the guard: confirms the in-place update is the thing that breaks."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    torch.manual_seed(0)
    base = torch.randn(1, SEQ, DIM, requires_grad=True)
    spans = _spans(BOUNDS_SINGLE)

    h = base * 1.0
    for _ in range(NUM_BLOCKS):
        h = cp.checkpoint(_upstream_gate_update, h, torch.randn(1, SEQ, DIM), spans, use_reentrant=False)

    with pytest.raises(RuntimeError, match="inplace"):
        h.square().mean().backward()


def test_training_block_does_not_mutate_input(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    block = _make_block()
    pe = _make_pe()
    x = torch.randn(1, SEQ, DIM)
    original = x.clone()

    out = block(x, _spans(BOUNDS_MULTI), pe, {})

    assert torch.equal(x, original)
    assert out is not x


def test_training_block_keeps_every_row(monkeypatch):
    """Zero gates must return the block input: the reconstruction cannot drop rows."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    block = _make_block()
    pe = _make_pe()
    x = torch.randn(1, SEQ, DIM)

    for bounds in (BOUNDS_SINGLE, BOUNDS_MULTI, [(0, 5), (5, 12), (12, SEQ)]):
        out = block(x.clone(), _spans(bounds, gate=0.0), pe, {})
        assert torch.allclose(out, x, atol=1e-6)


def test_gradients_match_between_checkpointed_and_plain(monkeypatch):
    """Checkpointing must not change the gradients the update is based on."""
    plain_base, plain = _run_chain(monkeypatch, BOUNDS_MULTI, use_checkpoint=False)
    ckpt_base, ckpt = _run_chain(monkeypatch, BOUNDS_MULTI, use_checkpoint=True)

    assert torch.allclose(plain_base, ckpt_base, atol=1e-5)
    assert len(plain) == len(ckpt)
    for a, b in zip(plain, ckpt):
        assert torch.allclose(a, b, atol=1e-5)


def test_preprocess_conds_keeps_the_early_stream_updates(monkeypatch):
    """The training branch returns new stream tensors, so each early block result must be kept."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    model = _make_model()
    context = torch.randn(1, 4, 20480)
    ref = torch.randn(1, 128, 2, 2)

    txt_states, ref_states = model.preprocess_conds(context, [ref])

    assert not torch.allclose(txt_states, model.txt_in(context))
    assert not torch.allclose(ref_states, model.emb_in["image_cond"](ref.flatten(2).transpose(1, 2)))

    (txt_states.square().mean() + ref_states.square().mean()).backward()
    assert model.txt_mode_blocks[0].q_proj.weight.grad is not None
    assert model.content_mode_blocks["image_cond"][0].q_proj.weight.grad is not None
