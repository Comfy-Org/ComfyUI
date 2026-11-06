"""Regression tests for the LTX AV transformer block residuals under training.

`comfy/ldm/lightricks/av_model.py::BasicAVTransformerBlock` folds the gated
attention/MLP residuals into the video and audio streams with in-place ops.
The block norms and the gradient checkpointing recompute read those same
tensors, so the training backward pass raises "one of the variables needed for
gradient computation has been modified by an inplace operation".

Training now builds the residuals out of place; inference keeps the in-place
update.

Run:  python -m pytest tests-unit/comfy_test/ltx_av_block_training_test.py
"""
import pytest
import torch
import torch.utils.checkpoint as cp

import comfy.ldm.common_dit
import comfy.model_management
import comfy.ops as comfy_ops
from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock
from comfy.ldm.lightricks.model import freqs_cis_matrix

NUM_BLOCKS = 16
V_DIM = 32
A_DIM = 16
V_HEADS = 4
A_HEADS = 2
HEAD_DIM = 8
V_SEQ = 16
A_SEQ = 8
CTX_SEQ = 6


def _make_block(cross_attention_adaln):
    block = BasicAVTransformerBlock(
        V_DIM,
        A_DIM,
        V_HEADS,
        A_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        v_context_dim=V_DIM,
        a_context_dim=A_DIM,
        cross_attention_adaln=cross_attention_adaln,
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=comfy_ops.disable_weight_init,
    )
    torch.manual_seed(0)
    for p in block.parameters():
        torch.nn.init.normal_(p, 0.0, 0.1)
    return block


def _pe(seq, heads):
    return freqs_cis_matrix(torch.randn(1, seq, heads * HEAD_DIM // 2), 0, False, heads, torch.float32)


def _block_args(cross_attention_adaln):
    num_ada = 9 if cross_attention_adaln else 6
    return dict(
        v_context=torch.randn(1, CTX_SEQ, V_DIM),
        a_context=torch.randn(1, CTX_SEQ, A_DIM),
        v_timestep=torch.randn(1, 1, num_ada * V_DIM),
        a_timestep=torch.randn(1, 1, num_ada * A_DIM),
        v_pe=_pe(V_SEQ, V_HEADS),
        a_pe=_pe(A_SEQ, A_HEADS),
        v_cross_pe=_pe(V_SEQ, A_HEADS),
        a_cross_pe=_pe(A_SEQ, A_HEADS),
        v_cross_scale_shift_timestep=torch.randn(1, 1, 4 * V_DIM),
        a_cross_scale_shift_timestep=torch.randn(1, 1, 4 * A_DIM),
        v_cross_gate_timestep=torch.randn(1, 1, V_DIM),
        a_cross_gate_timestep=torch.randn(1, 1, A_DIM),
        transformer_options={},
        v_prompt_timestep=torch.randn(1, 1, 2 * V_DIM) if cross_attention_adaln else None,
        a_prompt_timestep=torch.randn(1, 1, 2 * A_DIM) if cross_attention_adaln else None,
    )


def _call_block(block, x, args, use_checkpoint):
    if use_checkpoint:
        return cp.checkpoint(block, x, use_reentrant=False, **args)
    return block(x, **args)


def _run_chain(monkeypatch, cross_attention_adaln, use_checkpoint):
    """Thread video and audio streams through NUM_BLOCKS blocks and backprop."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    block = _make_block(cross_attention_adaln)
    args = _block_args(cross_attention_adaln)
    torch.manual_seed(0)
    vx_base = torch.randn(1, V_SEQ, V_DIM, requires_grad=True)
    ax_base = torch.randn(1, A_SEQ, A_DIM, requires_grad=True)

    vx, ax = vx_base * 1.0, ax_base * 1.0
    for _ in range(NUM_BLOCKS):
        vx, ax = _call_block(block, (vx, ax), args, use_checkpoint)

    (vx.square().mean() + ax.square().mean()).backward()
    return vx_base.grad, ax_base.grad, [p.grad for p in block.parameters() if p.grad is not None]


def _upstream_block_residual(x, gate):
    """The pre-fix residual, kept here to prove the fixture can fail: the block norm reads x first."""
    return x.addcmul_(comfy.ldm.common_dit.rms_norm(x), gate)


def _assert_finite(grads):
    for grad in grads:
        assert grad is not None
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize("cross_attention_adaln", [False, True])
def test_training_chain_backward_produces_finite_gradients(monkeypatch, cross_attention_adaln):
    vx_grad, ax_grad, param_grads = _run_chain(monkeypatch, cross_attention_adaln, use_checkpoint=False)
    _assert_finite((vx_grad, ax_grad))
    assert param_grads
    _assert_finite(param_grads)


@pytest.mark.parametrize("cross_attention_adaln", [False, True])
def test_training_chain_survives_gradient_checkpointing(monkeypatch, cross_attention_adaln):
    """The pre-fix in-place update raises here; the out-of-place path must not."""
    vx_grad, ax_grad, param_grads = _run_chain(monkeypatch, cross_attention_adaln, use_checkpoint=True)
    _assert_finite((vx_grad, ax_grad))
    assert param_grads
    _assert_finite(param_grads)


@pytest.mark.parametrize("cross_attention_adaln", [False, True])
def test_checkpointed_gradients_match_plain(monkeypatch, cross_attention_adaln):
    """Checkpointing must not change the gradients the residuals are based on."""
    plain_vx, plain_ax, plain = _run_chain(monkeypatch, cross_attention_adaln, use_checkpoint=False)
    ckpt_vx, ckpt_ax, ckpt = _run_chain(monkeypatch, cross_attention_adaln, use_checkpoint=True)

    assert torch.allclose(plain_vx, ckpt_vx, atol=1e-5)
    assert torch.allclose(plain_ax, ckpt_ax, atol=1e-5)
    assert len(plain) == len(ckpt)
    for a, b in zip(plain, ckpt):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("cross_attention_adaln", [False, True])
def test_training_block_does_not_mutate_inputs(monkeypatch, cross_attention_adaln):
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    block = _make_block(cross_attention_adaln)
    vx, ax = torch.randn(1, V_SEQ, V_DIM), torch.randn(1, A_SEQ, A_DIM)
    vx_original, ax_original = vx.clone(), ax.clone()

    out_vx, out_ax = _call_block(block, (vx, ax), _block_args(cross_attention_adaln), use_checkpoint=False)
    assert torch.equal(vx, vx_original)
    assert torch.equal(ax, ax_original)
    assert out_vx is not vx
    assert out_ax is not ax


def test_pre_fix_implementation_actually_fails(monkeypatch):
    """Guards the guard: confirms the in-place residual is the thing that breaks."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    torch.manual_seed(0)
    base = torch.randn(1, V_SEQ, V_DIM, requires_grad=True)
    # training gates come from the modulation tables, so they carry gradients
    gate = torch.full((1, 1, V_DIM), 0.2).requires_grad_()

    h = base * 1.0
    for _ in range(NUM_BLOCKS):
        h = _upstream_block_residual(h, gate)

    with pytest.raises(RuntimeError, match="inplace"):
        h.square().mean().backward()
