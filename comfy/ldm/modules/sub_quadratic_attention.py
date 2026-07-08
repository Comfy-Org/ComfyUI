# original source:
#   https://github.com/AminRezaei0x443/memory-efficient-attention/blob/1bc0d9e6ac5f82ea43a375135c4e1d3896ee1694/memory_efficient_attention/attention_torch.py
# license:
#   MIT
# credit:
#   Amin Rezaei (original author)
#   Alex Birch (optimized algorithm for 3D tensors, at the expense of removing bias, masking and callbacks)
# implementation of:
#   Self-attention Does Not Need O(n2) Memory":
#   https://arxiv.org/abs/2112.05682v2

from functools import partial
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import math
import logging

try:
    from typing import Optional, NamedTuple, List, Protocol
except ImportError:
    from typing import Optional, NamedTuple, List
    from typing_extensions import Protocol

from typing import List

from comfy import model_management

def dynamic_slice(
    x: Tensor,
    starts: List[int],
    sizes: List[int],
) -> Tensor:
    slicing = tuple(slice(start, start + size) for start, size in zip(starts, sizes))
    return x[slicing]

class AttnChunk(NamedTuple):
    exp_values: Tensor
    exp_weights_sum: Tensor
    max_score: Tensor

class SummarizeChunk(Protocol):
    @staticmethod
    def __call__(
        query: Tensor,
        key_t: Tensor,
        value: Tensor,
    ) -> AttnChunk: ...

class ComputeQueryChunkAttn(Protocol):
    @staticmethod
    def __call__(
        query: Tensor,
        key_t: Tensor,
        value: Tensor,
    ) -> Tensor: ...

def _summarize_chunk(
    query: Tensor,
    key_t: Tensor,
    value: Tensor,
    scale: float,
    upcast_attention: bool,
    mask,
) -> AttnChunk:
    if upcast_attention:
        with torch.autocast(enabled=False, device_type = 'cuda'):
            query = query.float()
            key_t = key_t.float()
            attn_weights = torch.baddbmm(
                torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
                query,
                key_t,
                alpha=scale,
                beta=0,
            )
    else:
        attn_weights = torch.baddbmm(
            torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
            query,
            key_t,
            alpha=scale,
            beta=0,
        )
    max_score, _ = torch.max(attn_weights, -1, keepdim=True)
    max_score = max_score.detach()
    attn_weights -= max_score
    if mask is not None:
        attn_weights += mask
    torch.exp(attn_weights, out=attn_weights)
    exp_weights = attn_weights.to(value.dtype)
    exp_values = torch.bmm(exp_weights, value)
    max_score = max_score.squeeze(-1)
    return AttnChunk(exp_values, exp_weights.sum(dim=-1), max_score)

def _query_chunk_attention(
    query: Tensor,
    key_t: Tensor,
    value: Tensor,
    summarize_chunk: SummarizeChunk,
    kv_chunk_size: int,
    mask,
) -> Tensor:
    batch_x_heads, k_channels_per_head, k_tokens = key_t.shape
    _, _, v_channels_per_head = value.shape

    def chunk_scanner(chunk_idx: int, mask) -> AttnChunk:
        key_chunk = dynamic_slice(
            key_t,
            (0, 0, chunk_idx),
            (batch_x_heads, k_channels_per_head, kv_chunk_size)
        )
        value_chunk = dynamic_slice(
            value,
            (0, chunk_idx, 0),
            (batch_x_heads, kv_chunk_size, v_channels_per_head)
        )
        if mask is not None:
            mask = mask[:,:,chunk_idx:chunk_idx + kv_chunk_size]

        return summarize_chunk(query, key_chunk, value_chunk, mask=mask)

    chunks: List[AttnChunk] = [
        chunk_scanner(chunk, mask) for chunk in torch.arange(0, k_tokens, kv_chunk_size)
    ]
    acc_chunk = AttnChunk(*map(torch.stack, zip(*chunks)))
    chunk_values, chunk_weights, chunk_max = acc_chunk

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights

# TODO: refactor CrossAttention#get_attention_scores to share code with this
def _get_attention_scores_no_kv_chunking(
    query: Tensor,
    key_t: Tensor,
    value: Tensor,
    scale: float,
    upcast_attention: bool,
    mask,
) -> Tensor:
    if upcast_attention:
        with torch.autocast(enabled=False, device_type = 'cuda'):
            query = query.float()
            key_t = key_t.float()
            attn_scores = torch.baddbmm(
                torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
                query,
                key_t,
                alpha=scale,
                beta=0,
            )
    else:
        attn_scores = torch.baddbmm(
            torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
            query,
            key_t,
            alpha=scale,
            beta=0,
        )

    if mask is not None:
        attn_scores += mask
    try:
        attn_probs = attn_scores.softmax(dim=-1)
        del attn_scores
    except Exception as e:
        model_management.raise_non_oom(e)
        logging.warning("ran out of memory while running softmax in  _get_attention_scores_no_kv_chunking, trying slower in place softmax instead")
        attn_scores -= attn_scores.max(dim=-1, keepdim=True).values # noqa: F821 attn_scores is not defined
        torch.exp(attn_scores, out=attn_scores)
        summed = torch.sum(attn_scores, dim=-1, keepdim=True)
        attn_scores /= summed
        attn_probs = attn_scores

    hidden_states_slice = torch.bmm(attn_probs.to(value.dtype), value)
    return hidden_states_slice

class ScannedChunk(NamedTuple):
    chunk_idx: int
    attn_chunk: AttnChunk

def efficient_dot_product_attention(
    query: Tensor,
    key_t: Tensor,
    value: Tensor,
    query_chunk_size=1024,
    kv_chunk_size: Optional[int] = None,
    kv_chunk_size_min: Optional[int] = None,
    use_checkpoint=True,
    upcast_attention=False,
    mask = None,
):
    """Computes efficient dot-product attention given query, transposed key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Args:
        query: queries for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        key_t: keys for calculating attention with shape of
          `[batch * num_heads, channels_per_head, tokens]`.
        value: values to be used in attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        query_chunk_size: int: query chunks size
        kv_chunk_size: Optional[int]: key/value chunks size. if None: defaults to sqrt(key_tokens)
        kv_chunk_size_min: Optional[int]: key/value minimum chunk size. only considered when kv_chunk_size is None. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
        use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)
      Returns:
        Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.
      """
    batch_x_heads, q_tokens, q_channels_per_head = query.shape
    _, _, k_tokens = key_t.shape
    scale = q_channels_per_head ** -0.5

    kv_chunk_size = min(kv_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    if kv_chunk_size_min is not None:
        kv_chunk_size = max(kv_chunk_size, kv_chunk_size_min)

    if mask is not None and len(mask.shape) == 2:
        mask = mask.unsqueeze(0)

    def get_query_chunk(chunk_idx: int) -> Tensor:
        return dynamic_slice(
            query,
            (0, chunk_idx, 0),
            (batch_x_heads, min(query_chunk_size, q_tokens), q_channels_per_head)
        )

    def get_mask_chunk(chunk_idx: int) -> Tensor:
        if mask is None:
            return None
        if mask.shape[1] == 1:
            return mask
        chunk = min(query_chunk_size, q_tokens)
        return mask[:,chunk_idx:chunk_idx + chunk]

    summarize_chunk: SummarizeChunk = partial(_summarize_chunk, scale=scale, upcast_attention=upcast_attention)
    summarize_chunk: SummarizeChunk = partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk
    compute_query_chunk_attn: ComputeQueryChunkAttn = partial(
        _get_attention_scores_no_kv_chunking,
        scale=scale,
        upcast_attention=upcast_attention
    ) if k_tokens <= kv_chunk_size else (
        # fast-path for when there's just 1 key-value chunk per query chunk (this is just sliced attention btw)
        partial(
            _query_chunk_attention,
            kv_chunk_size=kv_chunk_size,
            summarize_chunk=summarize_chunk,
        )
    )

    if q_tokens <= query_chunk_size:
        # fast-path for when there's just 1 query chunk
        return compute_query_chunk_attn(
            query=query,
            key_t=key_t,
            value=value,
            mask=mask,
        )

    # TODO: maybe we should use torch.empty_like(query) to allocate storage in-advance,
    # and pass slices to be mutated, instead of torch.cat()ing the returned slices
    res = torch.cat([
        compute_query_chunk_attn(
            query=get_query_chunk(i * query_chunk_size),
            key_t=key_t,
            value=value,
            mask=get_mask_chunk(i * query_chunk_size)
        ) for i in range(math.ceil(q_tokens / query_chunk_size))
    ], dim=1)
    return res
