import math

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention


def fractal_flatten(x, rope, shape):
    pixel_size = 8
    x = local_patching(x, shape, (1, pixel_size, pixel_size), dim=1)
    rope = local_patching(rope, shape, (1, pixel_size, pixel_size), dim=1)
    x = x.flatten(1, 2)
    rope = rope.flatten(1, 2)
    return x, rope


def fractal_unflatten(x, shape):
    pixel_size = 8
    x = x.reshape(x.shape[0], -1, pixel_size**2, x.shape[-1])
    x = local_merge(x, shape, (1, pixel_size, pixel_size), dim=1)
    return x

def local_patching(x, shape, group_size, dim=0):
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        g1,
        height // g2,
        g2,
        width // g3,
        g3,
        *x.shape[dim + 3 :]
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 2,
        dim + 4,
        dim + 1,
        dim + 3,
        dim + 5,
        *range(dim + 6, len(x.shape))
    )
    x = x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)
    return x


def local_merge(x, shape, group_size, dim=0):
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        height // g2,
        width // g3,
        g1,
        g2,
        g3,
        *x.shape[dim + 2 :]
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 3,
        dim + 1,
        dim + 4,
        dim + 2,
        dim + 5,
        *range(dim + 6, len(x.shape))
    )
    x = x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)
    return x

def fast_sta_nabla(T: int, H: int, W: int, wT: int = 3, wH: int = 3, wW: int = 3, device="cuda") -> Tensor:
    l = torch.Tensor([T, H, W]).amax()
    r = torch.arange(0, l, 1, dtype=torch.int16, device=device)
    mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
    sta_t, sta_h, sta_w = (
        mat[:T, :T].flatten(),
        mat[:H, :H].flatten(),
        mat[:W, :W].flatten(),
    )
    sta_t = sta_t <= wT // 2
    sta_h = sta_h <= wH // 2
    sta_w = sta_w <= wW // 2
    sta_hw = (
        (sta_h.unsqueeze(1) * sta_w.unsqueeze(0))
        .reshape(H, H, W, W)
        .transpose(1, 2)
        .flatten()
    )
    sta = (
        (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0))
        .reshape(T, T, H * W, H * W)
        .transpose(1, 2)
    )
    return sta.reshape(T * H * W, T * H * W)

def nablaT_v2(q: Tensor, k: Tensor, sta: Tensor, thr: float = 0.9) -> BlockMask:
    # Map estimation
    B, h, S, D = q.shape
    s1 = S // 64
    qa = q.reshape(B, h, s1, 64, D).mean(-2)
    ka = k.reshape(B, h, s1, 64, D).mean(-2).transpose(-2, -1)
    map = qa @ ka

    map = torch.softmax(map / math.sqrt(D), dim=-1)
    # Map binarization
    vals, inds = map.sort(-1)
    cvals = vals.cumsum_(-1)
    mask = (cvals >= 1 - thr).int()
    mask = mask.gather(-1, inds.argsort(-1))
    mask = torch.logical_or(mask, sta)

    # BlockMask creation
    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=64, mask_mod=None
    )

@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def nabla(query, key, value, sparse_params=None):
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    block_mask = nablaT_v2(
        query,
        key,
        sparse_params["sta_mask"],
        thr=sparse_params["P"],
    )
    out = (
        flex_attention(
            query,
            key,
            value,
            block_mask=block_mask
        )
        .transpose(1, 2)
        .contiguous()
    )
    out = out.flatten(-2, -1)
    return out
