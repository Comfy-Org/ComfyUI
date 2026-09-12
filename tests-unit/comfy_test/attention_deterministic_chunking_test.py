import sys

import torch

sys.argv = ["pytest", "--cpu"]
from comfy.options import enable_args_parsing  # noqa: E402
enable_args_parsing()

import comfy.ldm.modules.attention as attention  # noqa: E402


# A 2176-token bf16 sequence at this batch_x_heads lands on the 4096 vs. 2048
# query_chunk_size threshold depending on whether ~2.14GB or ~1.07GB is free.
BATCH = 30
TOKENS = 2176
DIM = 8
FREE_MEMORY_FORWARD = 2_200_000_000  # picks query_chunk_size=4096
FREE_MEMORY_RECOMPUTE = 1_500_000_000  # picks query_chunk_size=2048


def _make_qkv(batch=BATCH, tokens=TOKENS, dim=DIM, dtype=torch.bfloat16):
    q = torch.randn(batch, tokens, dim, dtype=dtype)
    k = torch.randn(batch, tokens, dim, dtype=dtype)
    v = torch.randn(batch, tokens, dim, dtype=dtype)
    return q, k, v


def _patch_free_memory_and_capture_chunk_sizes(monkeypatch):
    values = iter([FREE_MEMORY_FORWARD, FREE_MEMORY_RECOMPUTE])

    def fake_get_free_memory(device, torch_free_too=False):
        value = next(values)
        return (value, value) if torch_free_too else value

    chosen_query_chunk_sizes = []

    def fake_efficient_dot_product_attention(query, key, value, query_chunk_size=None, **kwargs):
        chosen_query_chunk_sizes.append(query_chunk_size)
        return torch.zeros(query.shape[0], query.shape[1], value.shape[-1], dtype=query.dtype)

    monkeypatch.setattr(attention.model_management, "get_free_memory", fake_get_free_memory)
    monkeypatch.setattr(attention, "efficient_dot_product_attention", fake_efficient_dot_product_attention)
    return chosen_query_chunk_sizes


def test_deterministic_memory_chunking_reuses_first_choice(monkeypatch):
    """A gradient-checkpoint recomputation calls attention_sub_quad again with the
    same shapes as the original forward, but free memory can have dropped in the
    meantime; inside the context both calls must pick the same chunk size."""
    chosen_query_chunk_sizes = _patch_free_memory_and_capture_chunk_sizes(monkeypatch)

    q, k, v = _make_qkv()
    with attention.deterministic_memory_chunking():
        attention.attention_sub_quad(q, k, v, heads=1)
        attention.attention_sub_quad(q, k, v, heads=1)

    assert len(chosen_query_chunk_sizes) == 2
    assert chosen_query_chunk_sizes[0] == chosen_query_chunk_sizes[1]


def test_memory_chunking_still_reacts_to_free_memory_outside_context(monkeypatch):
    """Sanity check that normal (non-checkpointed) calls keep querying live free
    memory, i.e. the fix does not change behavior outside the training path."""
    chosen_query_chunk_sizes = _patch_free_memory_and_capture_chunk_sizes(monkeypatch)

    q, k, v = _make_qkv()
    attention.attention_sub_quad(q, k, v, heads=1)
    attention.attention_sub_quad(q, k, v, heads=1)

    assert chosen_query_chunk_sizes[0] != chosen_query_chunk_sizes[1]


# batch_x_heads=2, tokens=8, dim=4, bf16 -> mem_required=768 bytes, matching SPLIT_FREE_MEMORY.
SPLIT_QKV_SHAPE = (2, 8, 4)
SPLIT_FREE_MEMORY = 768


def _patch_free_memory(monkeypatch, free_memory_values):
    values = iter(free_memory_values)

    def fake_get_free_memory(device, torch_free_too=False):
        value = next(values)
        return (value, value) if torch_free_too else value

    monkeypatch.setattr(attention.model_management, "get_free_memory", fake_get_free_memory)


def test_deterministic_memory_chunking_reuses_first_choice_for_split(monkeypatch):
    """Same recomputation scenario as above, for attention_split's `steps` count.
    Only one free-memory value is queued: a second live query (i.e. a cache miss
    on the recompute) would raise StopIteration and fail the test."""
    _patch_free_memory(monkeypatch, [SPLIT_FREE_MEMORY])
    q, k, v = _make_qkv(*SPLIT_QKV_SHAPE)

    with attention.deterministic_memory_chunking():
        attention.attention_split(q, k, v, heads=1)
        attention.attention_split(q, k, v, heads=1)
        assert len(attention._memory_chunk_cache_var.get()) == 1


def test_split_oom_retry_updates_cache_with_final_steps(monkeypatch):
    """An OOM during the forward pass doubles `steps` past the value cached before
    the retry; the cache reused by checkpoint recomputation must reflect the final,
    successful `steps`, not the stale one."""
    _patch_free_memory(monkeypatch, [SPLIT_FREE_MEMORY])
    real_einsum = attention.einsum
    failures_left = [2]

    def flaky_einsum(equation, *operands, **kwargs):
        if failures_left[0] > 0:
            failures_left[0] -= 1
            raise torch.OutOfMemoryError("simulated OOM")
        return real_einsum(equation, *operands, **kwargs)

    monkeypatch.setattr(attention, "einsum", flaky_einsum)
    q, k, v = _make_qkv(*SPLIT_QKV_SHAPE)

    with attention.deterministic_memory_chunking():
        attention.attention_split(q, k, v, heads=1)
        cache = attention._memory_chunk_cache_var.get()

    assert list(cache.values()) == [2]


def test_memory_chunking_still_reacts_to_free_memory_outside_context_for_split(monkeypatch):
    """Sanity check for attention_split, mirroring the sub_quad test above: outside
    deterministic_memory_chunking(), two calls with different free memory can still
    pick different `steps`, i.e. the fix does not change non-training behavior."""
    _patch_free_memory(monkeypatch, [SPLIT_FREE_MEMORY, SPLIT_FREE_MEMORY // 2])
    q, k, v = _make_qkv(*SPLIT_QKV_SHAPE)

    real_einsum = attention.einsum
    qk_call_counts = []

    def counting_einsum(equation, *operands, **kwargs):
        if equation == 'b i d, b j d -> b i j':
            qk_call_counts[-1] += 1
        return real_einsum(equation, *operands, **kwargs)

    monkeypatch.setattr(attention, "einsum", counting_einsum)

    qk_call_counts.append(0)
    attention.attention_split(q, k, v, heads=1)
    qk_call_counts.append(0)
    attention.attention_split(q, k, v, heads=1)

    assert qk_call_counts[0] != qk_call_counts[1]
