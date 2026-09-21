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


def var_attention_optimized_split(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, *args, skip_reshape=False, skip_output_reshape=False, **kwargs):
    q, k, v, head_dim = _var_attention_qkv(q, k, v, heads, skip_reshape)

    q_split_indices = cu_seqlens_q[1:-1]
    k_split_indices = cu_seqlens_k[1:-1]
    if k.shape[0] != v.shape[0]:
        raise ValueError("cu_seqlens_k does not match v token count")

    q_splits = torch.tensor_split(q, q_split_indices, dim=0)
    k_splits = torch.tensor_split(k, k_split_indices, dim=0)
    v_splits = torch.tensor_split(v, k_split_indices, dim=0)
    if len(q_splits) != len(k_splits) or len(q_splits) != len(v_splits):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same sequence count")

    out = []
    for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits):
        q_i = q_i.permute(1, 0, 2).unsqueeze(0)
        k_i = k_i.permute(1, 0, 2).unsqueeze(0)
        v_i = v_i.permute(1, 0, 2).unsqueeze(0)
        out_i = _attention.optimized_attention(q_i, k_i, v_i, heads, skip_reshape=True, skip_output_reshape=True)
        out.append(out_i.squeeze(0).permute(1, 0, 2))

    out = torch.cat(out, dim=0)
    return _var_attention_output(out, heads, head_dim, skip_output_reshape)


optimized_var_attention = var_attention_optimized_split
