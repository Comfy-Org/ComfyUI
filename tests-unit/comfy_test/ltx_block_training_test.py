"""Regression tests for the LTX transformer block residual under training.

`comfy/ldm/lightricks/model.py::BasicTransformerBlock` folds the gated
attention/MLP residuals into the block input stream with in-place ops. The MLP
norm and the gradient checkpointing recompute read that same tensor, so the
training backward pass raises "one of the variables needed for gradient
computation has been modified by an inplace operation" after a few blocks.

Training now builds the residual out of place; inference keeps the in-place
update.

Run:  python -m pytest tests-unit/comfy_test/ltx_block_training_test.py
"""
import pytest
import torch
import torch.utils.checkpoint as cp

import comfy.model_management
import comfy.ops as comfy_ops
from comfy.ldm.lightricks.model import BasicTransformerBlock, freqs_cis_matrix

NUM_BLOCKS = 32
DIM = 32
HEADS = 4
SEQ = 16
CTX_SEQ = 8


def _make_block(cross_attention_adaln):
    block = BasicTransformerBlock(
        DIM,
        HEADS,
        DIM // HEADS,
        context_dim=DIM,
        cross_attention_adaln=cross_attention_adaln,
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=comfy_ops.disable_weight_init,
    )
    torch.manual_seed(0)
    for p in block.parameters():
        torch.nn.init.normal_(p, 0.0, 0.1)
    return block


def _block_args(cross_attention_adaln):
    pe = freqs_cis_matrix(torch.randn(1, SEQ, DIM // 2), 0, False, HEADS, torch.float32)
    context = torch.randn(1, CTX_SEQ, DIM)
    timestep = torch.randn(1, SEQ, (9 if cross_attention_adaln else 6) * DIM)
    prompt_timestep = torch.randn(1, 1, 2 * DIM) if cross_attention_adaln else None
    return context, timestep, pe, prompt_timestep


def _call_block(block, x, args, use_checkpoint, reentrant=False):
    context, timestep, pe, prompt_timestep = args
    if use_checkpoint:
        return cp.checkpoint(block, x, context, None, timestep, pe, {}, None, prompt_timestep, use_reentrant=reentrant)
    return block(x, context, None, timestep, pe, {}, None, prompt_timestep)


def _run_chain(monkeypatch, cross_attention_adaln, use_checkpoint, reentrant=False):
    """Thread a stream through NUM_BLOCKS blocks and backprop through the result."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    block = _make_block(cross_attention_adaln)
    args = _block_args(cross_attention_adaln)
    torch.manual_seed(0)
    base = torch.randn(1, SEQ, DIM, requires_grad=True)

    h = base * 1.0
    for _ in range(NUM_BLOCKS):
        h = _call_block(block, h, args, use_checkpoint, reentrant)

    h.square().mean().backward()
    return base.grad, [p.grad for p in block.parameters() if p.grad is not None]


def _upstream_gated_residual(x, out, gate=None):
    """The pre-fix residual application, kept here to prove the fixture can fail."""
    if gate is None:
        x += out
    else:
        x += out * gate
    return x


@pytest.mark.parametrize("cross_attention_adaln", [False, True])
def test_training_chain_backward_produces_finite_gradients(monkeypatch, cross_attention_adaln):
    base_grad, param_grads = _run_chain(monkeypatch, cross_attention_adaln, use_checkpoint=False)
    assert base_grad is not None
    assert torch.isfinite(base_grad).all()
    assert param_grads and all(torch.isfinite(g).all() for g in param_grads)


@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("cross_attention_adaln", [False, True])
def test_training_chain_survives_gradient_checkpointing(monkeypatch, cross_attention_adaln, reentrant):
    """The pre-fix in-place update raises here; the out-of-place path must not."""
    base_grad, param_grads = _run_chain(monkeypatch, cross_attention_adaln, use_checkpoint=True, reentrant=reentrant)
    assert base_grad is not None
    assert torch.isfinite(base_grad).all()
    assert param_grads and all(torch.isfinite(g).all() for g in param_grads)


@pytest.mark.parametrize("cross_attention_adaln", [False, True])
def test_checkpointed_gradients_match_plain(monkeypatch, cross_attention_adaln):
    """Checkpointing must not change the gradients the residual is based on."""
    plain_base, plain = _run_chain(monkeypatch, cross_attention_adaln, use_checkpoint=False)
    ckpt_base, ckpt = _run_chain(monkeypatch, cross_attention_adaln, use_checkpoint=True)

    assert torch.allclose(plain_base, ckpt_base, atol=1e-5)
    assert len(plain) == len(ckpt)
    for a, b in zip(plain, ckpt):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("cross_attention_adaln", [False, True])
def test_training_block_does_not_mutate_input(monkeypatch, cross_attention_adaln):
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    block = _make_block(cross_attention_adaln)
    x = torch.randn(1, SEQ, DIM)
    original = x.clone()

    out = _call_block(block, x, _block_args(cross_attention_adaln), use_checkpoint=False)
    assert torch.equal(x, original)
    assert out is not x


def test_pre_fix_implementation_actually_fails(monkeypatch):
    """Guards the guard: confirms the in-place residual is the thing that breaks."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    torch.manual_seed(0)
    base = torch.randn(1, SEQ, DIM, requires_grad=True)
    # training gates come from the modulation tables, so they carry gradients
    gate = torch.full((1, 1, DIM), 0.2).requires_grad_()

    h = base * 1.0
    for _ in range(NUM_BLOCKS):
        h = cp.checkpoint(_upstream_gated_residual, h, torch.randn(1, SEQ, DIM), gate, use_reentrant=False)

    with pytest.raises(RuntimeError, match="inplace"):
        h.square().mean().backward()
