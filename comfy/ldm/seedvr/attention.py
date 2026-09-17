import torch

from comfy.ldm.modules import attention as _attention


def _var_attention_qkv(q, k, v, heads, skip_reshape):
    if skip_reshape:
        return q, k, v, q.shape[-1]
    total_tokens, embed_dim = q.shape
    head_dim = embed_dim // heads
    return (
        q.view(total_tokens, heads, head_dim),
        k.view(k.shape[0], heads, head_dim),
        v.view(v.shape[0], heads, head_dim),
        head_dim,
    )


def _var_attention_output(out, heads, head_dim, skip_output_reshape):
    if skip_output_reshape:
        return out
    return out.reshape(-1, heads * head_dim)


def _equal_length_runs(lengths):
    """(start, stop) sequence spans over which the sequence length is constant."""
    runs = []
    start = 0
    for i in range(1, len(lengths) + 1):
        if i == len(lengths) or lengths[i] != lengths[start]:
            runs.append((start, i))
            start = i
    return runs


def var_attention_optimized_split(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, *args, skip_reshape=False, skip_output_reshape=False, **kwargs):
    q, k, v, head_dim = _var_attention_qkv(q, k, v, heads, skip_reshape)

    if k.shape[0] != v.shape[0]:
        raise ValueError("cu_seqlens_k does not match v token count")
    if len(cu_seqlens_q) != len(cu_seqlens_k):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same sequence count")

    q_lens = [end - start for start, end in zip(cu_seqlens_q, cu_seqlens_q[1:])]
    k_lens = [end - start for start, end in zip(cu_seqlens_k, cu_seqlens_k[1:])]

    out = torch.empty((cu_seqlens_q[-1], heads, head_dim), device=q.device, dtype=q.dtype)
    # Windows of matching length are adjacent, so each run becomes one batched attention call.
    for first, last in _equal_length_runs(list(zip(q_lens, k_lens))):
        n = last - first
        len_q, len_k = q_lens[first], k_lens[first]
        q_slice = slice(cu_seqlens_q[first], cu_seqlens_q[last])
        k_slice = slice(cu_seqlens_k[first], cu_seqlens_k[last])
        q_i = q[q_slice].reshape(n, len_q, heads, head_dim).transpose(1, 2)
        k_i = k[k_slice].reshape(n, len_k, heads, head_dim).transpose(1, 2)
        v_i = v[k_slice].reshape(n, len_k, heads, head_dim).transpose(1, 2)
        out_i = _attention.optimized_attention(q_i, k_i, v_i, heads, skip_reshape=True, skip_output_reshape=True)
        out[q_slice] = out_i.transpose(1, 2).reshape(n * len_q, heads, head_dim)

    return _var_attention_output(out, heads, head_dim, skip_output_reshape)


optimized_var_attention = var_attention_optimized_split
