import math

import torch
import torch.nn.functional as F

import comfy.ldm.modules.attention as attention
import comfy.ldm.modules.diffusionmodules.model as vae_model


def test_attention_split_slices_uneven_sequence(monkeypatch):
    heads, dim_head, tokens = 2, 8, 1001
    q, k, v = torch.randn(3, 1, tokens, heads * dim_head).unbind(0)
    # a third of the memory the full attention matrix needs, so the query is split into 4 slices
    mem_required = heads * tokens * tokens * q.element_size() * 3
    monkeypatch.setattr(attention.model_management, "get_free_memory", lambda device: mem_required // 3)

    slices = []
    einsum = attention.einsum

    def recording_einsum(equation, a, b):
        if equation == 'b i d, b j d -> b i j':
            slices.append(a.shape[1])
        return einsum(equation, a, b)

    monkeypatch.setattr(attention, "einsum", recording_einsum)

    out = attention.attention_split(q, k, v, heads)

    assert slices == [math.ceil(tokens / 4)] * 3 + [tokens - 3 * math.ceil(tokens / 4)]
    q, k, v = (t.view(1, tokens, heads, dim_head).transpose(1, 2) for t in (q, k, v))
    expected = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, tokens, heads * dim_head)
    torch.testing.assert_close(out, expected)


def test_vae_split_attention_slices_uneven_sequence(monkeypatch):
    channels, h, w = 16, 7, 143
    tokens = h * w
    q, k, v = torch.randn(3, 1, channels, h, w).unbind(0)
    mem_required = tokens * tokens * q.element_size() * 2.5
    monkeypatch.setattr(vae_model.model_management, "get_free_memory", lambda device: mem_required / 3)

    slices = []
    bmm = torch.bmm

    def recording_bmm(a, b):
        if a.shape[2] == channels:
            slices.append(a.shape[1])
        return bmm(a, b)

    monkeypatch.setattr(torch, "bmm", recording_bmm)

    out = vae_model.normal_attention(q, k, v)

    assert slices == [math.ceil(tokens / 4)] * 3 + [tokens - 3 * math.ceil(tokens / 4)]
    q, k, v = (t.reshape(1, channels, tokens).transpose(1, 2) for t in (q, k, v))
    expected = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, channels, h, w)
    torch.testing.assert_close(out, expected)
