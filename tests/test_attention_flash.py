import torch
import torch.nn.functional as F

from comfy.ldm.modules import attention
from comfy.ldm.qwen_image21 import model as qwen_image21


def _inputs():
    torch.manual_seed(0)
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 3, 8)
    v = torch.randn(1, 3, 8)
    mask = torch.tril(torch.ones(2, 3, dtype=torch.bool), diagonal=1)
    return q, k, v, mask


def test_attention_flash_uses_causal_flash_for_right_aligned_causal_mask(monkeypatch):
    q, k, v, mask = _inputs()
    calls = []

    def flash_attn(q, k, v, dropout_p, causal, softmax_scale):
        calls.append(causal)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        causal_mask = torch.tril(
            torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device),
            diagonal=k.shape[-2] - q.shape[-2],
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
        return out.transpose(1, 2)

    monkeypatch.setattr(attention, "flash_attn_wrapper", flash_attn)
    actual = attention.attention_flash(q[:, :2], k, v, heads=2, mask=mask, causal=True)
    expected = attention.attention_pytorch(q[:, :2], k, v, heads=2, mask=mask)

    assert calls == [True]
    torch.testing.assert_close(actual, expected)


def test_qwen21_causal_segment_uses_flash_attention(monkeypatch):
    q, k, v, mask = _inputs()
    calls = []

    def flash_attn(q, k, v, dropout_p, causal, softmax_scale):
        calls.append(causal)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        causal_mask = torch.tril(
            torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device),
            diagonal=k.shape[-2] - q.shape[-2],
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
        return out.transpose(1, 2)

    monkeypatch.setattr(attention, "flash_attn_wrapper", flash_attn)
    monkeypatch.setattr(qwen_image21, "optimized_attention", attention.attention_flash)
    q, k, v = (tensor.reshape(1, tensor.shape[1], 2, 4) for tensor in (q, k, v))
    actual = qwen_image21.block_causal_attention([(1, 3, mask)])(q, k, v, heads=2)
    expected = attention.attention_pytorch(
        q[:, 1:3].flatten(2), k.flatten(2), v.flatten(2), heads=2, mask=mask
    )

    assert calls == [True]
    torch.testing.assert_close(actual, expected)


def test_attention_flash_keeps_fallback_for_noncausal_mask(monkeypatch):
    q, k, v, _ = _inputs()
    mask = torch.tensor([[True, False, True], [False, True, True], [True, True, False]])
    calls = []

    def flash_attn(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Flash Attention cannot apply an arbitrary mask")

    monkeypatch.setattr(attention, "flash_attn_wrapper", flash_attn)
    actual = attention.attention_flash(q, k, v, heads=2, mask=mask)
    expected = attention.attention_pytorch(q, k, v, heads=2, mask=mask)

    assert calls == []
    torch.testing.assert_close(actual, expected)
