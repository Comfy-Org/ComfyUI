"""Regression tests for the Qwen-Image-2.1 in-place gated residual.

`comfy/ldm/qwen_image21/model.py::_gated_residual` folds each block's gated
residual into the running hidden state with an in-place `addcmul_`. The same
tensor is then read by backward, which makes the training path invalid:

  * with gradient checkpointing (reentrant or not) the backward pass raises
    "one of the variables needed for gradient computation has been modified
    by an inplace operation"
  * the in-place mutation is 32 blocks deep, so the autograd graph keeps
    aliasing the same storage

The fix makes the training branch out-of-place and leaves the inference
branch on the original in-place path, so image output is bit-identical.

Run:  python -m pytest tests-unit/comfy_test/qwen_image21_gated_residual_test.py
"""
import pytest
import torch
import torch.utils.checkpoint as cp

import comfy.model_management
from comfy.ldm.qwen_image21.model import _gated_residual

NUM_BLOCKS = 32
DIM = 32
SEQ = 16
PREFIX_LEN = 4


def _upstream_gated_residual(x, y, gate, prefix_len):
    """The pre-fix implementation, kept here to prove the test can fail."""
    g_prefix, g_target = gate
    x[:, prefix_len:].addcmul_(y[:, prefix_len:], g_target)
    if prefix_len:
        x[:, :prefix_len].addcmul_(y[:, :prefix_len], g_prefix)
    return x


def _run_chain(fn, num_blocks, prefix_len, use_checkpoint, reentrant=False):
    """Thread a residual stream through `num_blocks` calls of `fn` and backprop."""
    torch.manual_seed(0)
    base = torch.randn(1, SEQ, DIM, requires_grad=True)
    gates = [
        (torch.full((1, DIM), 0.1 + 0.001 * i, requires_grad=True),
         torch.full((1, DIM), 0.2 + 0.001 * i, requires_grad=True))
        for i in range(num_blocks)
    ]
    branch = [
        torch.randn(1, SEQ, DIM) for _ in range(num_blocks)
    ]

    h = base * 1.0
    for i in range(num_blocks):
        g_prefix, g_target = gates[i]
        if use_checkpoint:
            h = cp.checkpoint(fn, h, branch[i], (g_prefix, g_target), prefix_len,
                              use_reentrant=reentrant)
        else:
            h = fn(h, branch[i], (g_prefix, g_target), prefix_len)

    h.square().mean().backward()
    return base.grad, [g.grad for pair in gates for g in pair]


def _all_finite(grads):
    """True when every gradient is finite."""
    return all(torch.isfinite(g).all() for g in grads)


def test_training_chain_backward_produces_finite_gradients(monkeypatch):
    """Without checkpointing the pre-fix chain survives, so gradients must be usable."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    base_grad, grads = _run_chain(_gated_residual, NUM_BLOCKS, PREFIX_LEN,
                                  use_checkpoint=False)
    assert base_grad is not None
    assert _all_finite(grads)
    assert torch.isfinite(base_grad).all()


@pytest.mark.parametrize("reentrant", [False, True])
def test_training_chain_survives_gradient_checkpointing(monkeypatch, reentrant):
    """The pre-fix in-place chain raises here; the out-of-place path must not."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    base_grad, grads = _run_chain(_gated_residual, NUM_BLOCKS, PREFIX_LEN,
                                  use_checkpoint=True, reentrant=reentrant)
    assert base_grad is not None
    assert _all_finite(grads)


def test_pre_fix_implementation_actually_fails(monkeypatch):
    """Guards the guard: confirms the in-place form is the thing that breaks.

    Without this, the tests above could pass simply because the fixture is
    too weak to exercise the defect.
    """
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    with pytest.raises(RuntimeError, match="inplace"):
        _run_chain(_upstream_gated_residual, NUM_BLOCKS, PREFIX_LEN,
                   use_checkpoint=True, reentrant=False)


def test_training_branch_does_not_mutate_its_input(monkeypatch):
    """The training path must leave the caller's tensor alone."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    x = torch.randn(1, SEQ, DIM)
    original = x.clone()
    y = torch.randn(1, SEQ, DIM)
    g = (torch.full((1, DIM), 0.1), torch.full((1, DIM), 0.2))

    out = _gated_residual(x, y, g, PREFIX_LEN)

    assert torch.equal(x, original)
    assert out is not x


def test_inference_branch_still_mutates_in_place(monkeypatch):
    """Inference keeps the original fused path: no extra allocation."""
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    x = torch.randn(1, SEQ, DIM)
    original = x.clone()
    y = torch.randn(1, SEQ, DIM)
    g = (torch.full((1, DIM), 0.1), torch.full((1, DIM), 0.2))

    out = _gated_residual(x, y, g, PREFIX_LEN)

    assert out is x
    expected = original.clone()
    expected[:, PREFIX_LEN:] += y[:, PREFIX_LEN:] * 0.2
    expected[:, :PREFIX_LEN] += y[:, :PREFIX_LEN] * 0.1
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("prefix_len", [0, 1, PREFIX_LEN, SEQ - 1])
def test_training_branch_matches_inference_values(monkeypatch, prefix_len):
    """Both branches must compute the same thing; only the aliasing differs."""
    torch.manual_seed(1)
    x = torch.randn(1, SEQ, DIM)
    y = torch.randn(1, SEQ, DIM)
    g = (torch.full((1, DIM), 0.1), torch.full((1, DIM), 0.2))

    monkeypatch.setattr(comfy.model_management, "in_training", False)
    inference = _gated_residual(x.clone(), y, g, prefix_len).clone()

    monkeypatch.setattr(comfy.model_management, "in_training", True)
    training = _gated_residual(x.clone(), y, g, prefix_len)

    assert torch.allclose(training, inference, atol=1e-6)


def test_gradients_match_between_checkpointed_and_plain(monkeypatch):
    """Checkpointing must not change the gradients the update is based on."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    plain_base, plain = _run_chain(_gated_residual, NUM_BLOCKS, PREFIX_LEN,
                                   use_checkpoint=False)
    ckpt_base, ckpt = _run_chain(_gated_residual, NUM_BLOCKS, PREFIX_LEN,
                                 use_checkpoint=True, reentrant=False)

    assert torch.allclose(plain_base, ckpt_base, atol=1e-5)
    for a, b in zip(plain, ckpt, strict=True):
        assert torch.allclose(a, b, atol=1e-5)
