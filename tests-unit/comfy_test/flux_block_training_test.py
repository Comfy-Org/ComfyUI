"""Regression tests for the Flux block gated residuals under training.

`comfy/ldm/flux/layers.py::DoubleStreamBlock` and `SingleStreamBlock` fold the
gated attention/MLP residual into the block inputs with an in-place `+=`. The
block norms and the gradient checkpointing recompute read those same tensors,
so the training backward pass raises "one of the variables needed for gradient
computation has been modified by an inplace operation" once per block.

Training now builds the gated residual out of place; inference keeps the
in-place update.

Run:  python -m pytest tests-unit/comfy_test/flux_block_training_test.py
"""
import pytest
import torch
import torch.utils.checkpoint as cp

import comfy.model_management
import comfy.ops as comfy_ops
from comfy.ldm.flux.layers import DoubleStreamBlock, EmbedND, SingleStreamBlock

NUM_BLOCKS = 32
DIM = 32
HEADS = 4
SEQ = 16
TXT_SEQ = 8


def _make_block(cls):
    block = cls(DIM, HEADS, 2.0, dtype=torch.float32, device=torch.device("cpu"), operations=comfy_ops.disable_weight_init)
    torch.manual_seed(0)
    for p in block.parameters():
        torch.nn.init.normal_(p, 0.0, 0.1)
    return block


def _make_pe(seq):
    rope = EmbedND(dim=DIM // HEADS, theta=10000, axes_dim=[DIM // HEADS // 4] * 4)
    return rope(torch.zeros(1, seq, 4))


def _upstream_gated_residual(x, out, gate):
    """The pre-fix residual, kept here to prove the fixture can fail."""
    x += out * gate
    return x


def _single_chain(monkeypatch, use_checkpoint, reentrant=False):
    """Thread a stream through NUM_BLOCKS single-stream blocks and backprop."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    block = _make_block(SingleStreamBlock)
    pe = _make_pe(SEQ)
    vec = torch.randn(1, DIM)
    torch.manual_seed(0)
    base = torch.randn(1, SEQ, DIM, requires_grad=True)

    h = base * 1.0
    for _ in range(NUM_BLOCKS):
        if use_checkpoint:
            h = cp.checkpoint(block, h, vec, pe, use_reentrant=reentrant)
        else:
            h = block(h, vec, pe)

    h.square().mean().backward()
    return base.grad, [p.grad for p in block.parameters() if p.grad is not None]


def _double_chain(monkeypatch, use_checkpoint, reentrant=False):
    """Thread an image and a text stream through NUM_BLOCKS double-stream blocks and backprop."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    block = _make_block(DoubleStreamBlock)
    pe = _make_pe(TXT_SEQ + SEQ)
    vec = torch.randn(1, DIM)
    torch.manual_seed(0)
    img_base = torch.randn(1, SEQ, DIM, requires_grad=True)
    txt_base = torch.randn(1, TXT_SEQ, DIM, requires_grad=True)

    img, txt = img_base * 1.0, txt_base * 1.0
    for _ in range(NUM_BLOCKS):
        if use_checkpoint:
            img, txt = cp.checkpoint(block, img, txt, vec, pe, use_reentrant=reentrant)
        else:
            img, txt = block(img, txt, vec, pe)

    (img.square().mean() + txt.square().mean()).backward()
    return img_base.grad, txt_base.grad, [p.grad for p in block.parameters() if p.grad is not None]


def test_single_block_training_chain_backward_produces_finite_gradients(monkeypatch):
    base_grad, param_grads = _single_chain(monkeypatch, use_checkpoint=False)
    assert base_grad is not None
    assert torch.isfinite(base_grad).all()
    assert param_grads and all(torch.isfinite(g).all() for g in param_grads)


def test_double_block_training_chain_backward_produces_finite_gradients(monkeypatch):
    img_grad, txt_grad, param_grads = _double_chain(monkeypatch, use_checkpoint=False)
    assert img_grad is not None and torch.isfinite(img_grad).all()
    assert txt_grad is not None and torch.isfinite(txt_grad).all()
    assert param_grads and all(torch.isfinite(g).all() for g in param_grads)


@pytest.mark.parametrize("reentrant", [False, True])
def test_single_block_chain_survives_gradient_checkpointing(monkeypatch, reentrant):
    """The pre-fix in-place update raises here; the out-of-place path must not."""
    base_grad, param_grads = _single_chain(monkeypatch, use_checkpoint=True, reentrant=reentrant)
    assert base_grad is not None
    assert torch.isfinite(base_grad).all()
    assert param_grads and all(torch.isfinite(g).all() for g in param_grads)


@pytest.mark.parametrize("reentrant", [False, True])
def test_double_block_chain_survives_gradient_checkpointing(monkeypatch, reentrant):
    """The pre-fix in-place update raises here; the out-of-place path must not."""
    img_grad, txt_grad, param_grads = _double_chain(monkeypatch, use_checkpoint=True, reentrant=reentrant)
    assert img_grad is not None and torch.isfinite(img_grad).all()
    assert txt_grad is not None and torch.isfinite(txt_grad).all()
    assert param_grads and all(torch.isfinite(g).all() for g in param_grads)


def test_pre_fix_implementation_actually_fails(monkeypatch):
    """Guards the guard: confirms the in-place residual is the thing that breaks."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    torch.manual_seed(0)
    base = torch.randn(1, SEQ, DIM, requires_grad=True)
    # training gates come from the modulation linears, so they carry gradients
    gate = torch.full((1, 1, DIM), 0.2).requires_grad_()

    h = base * 1.0
    for _ in range(NUM_BLOCKS):
        h = cp.checkpoint(_upstream_gated_residual, h, torch.randn(1, SEQ, DIM), gate, use_reentrant=False)

    with pytest.raises(RuntimeError, match="inplace"):
        h.square().mean().backward()


def test_training_blocks_do_not_mutate_inputs(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    vec = torch.randn(1, DIM)

    single = _make_block(SingleStreamBlock)
    x = torch.randn(1, SEQ, DIM)
    original = x.clone()
    single_out = single(x, vec, _make_pe(SEQ))
    assert torch.equal(x, original)
    assert single_out is not x

    double = _make_block(DoubleStreamBlock)
    img, txt = torch.randn(1, SEQ, DIM), torch.randn(1, TXT_SEQ, DIM)
    img_original, txt_original = img.clone(), txt.clone()
    img_out, txt_out = double(img, txt, vec, _make_pe(TXT_SEQ + SEQ))
    assert torch.equal(img, img_original)
    assert torch.equal(txt, txt_original)
    assert img_out is not img
    assert txt_out is not txt


def test_zero_gates_return_the_block_inputs(monkeypatch):
    """Zero gates must return the block inputs: the out-of-place residual cannot drop rows."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    vec = torch.randn(1, DIM)

    single = _make_block(SingleStreamBlock)
    torch.nn.init.zeros_(single.modulation.lin.weight)
    torch.nn.init.zeros_(single.modulation.lin.bias)
    x = torch.randn(1, SEQ, DIM)
    assert torch.allclose(single(x.clone(), vec, _make_pe(SEQ)), x)

    double = _make_block(DoubleStreamBlock)
    for mod in (double.img_mod, double.txt_mod):
        torch.nn.init.zeros_(mod.lin.weight)
        torch.nn.init.zeros_(mod.lin.bias)
    img, txt = torch.randn(1, SEQ, DIM), torch.randn(1, TXT_SEQ, DIM)
    img_out, txt_out = double(img.clone(), txt.clone(), vec, _make_pe(TXT_SEQ + SEQ))
    assert torch.allclose(img_out, img)
    assert torch.allclose(txt_out, txt)


def test_checkpointed_gradients_match_plain(monkeypatch):
    """Checkpointing must not change the gradients the residuals are based on."""
    plain_base, plain = _single_chain(monkeypatch, use_checkpoint=False)
    ckpt_base, ckpt = _single_chain(monkeypatch, use_checkpoint=True)

    assert torch.allclose(plain_base, ckpt_base, atol=1e-5)
    assert len(plain) == len(ckpt)
    for a, b in zip(plain, ckpt):
        assert torch.allclose(a, b, atol=1e-5)
