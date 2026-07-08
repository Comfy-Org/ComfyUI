import math
import sys

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any, Callable, Union
import logging
import functools

from .diffusionmodules.util import AlphaBlender, timestep_embedding
from .sub_quadratic_attention import efficient_dot_product_attention

from comfy import model_management

TORCH_HAS_GQA = model_management.torch_version_numeric >= (2, 5)

if model_management.xformers_enabled():
    import xformers
    import xformers.ops

SAGE_ATTENTION_IS_AVAILABLE = False
try:
    from sageattention import sageattn
    SAGE_ATTENTION_IS_AVAILABLE = True
except ImportError as e:
    if model_management.sage_attention_enabled():
        if e.name == "sageattention":
            logging.error(f"\n\nTo use the `--use-sage-attention` feature, the `sageattention` package must be installed first.\ncommand:\n\t{sys.executable} -m pip install sageattention")
        else:
            raise e
        exit(-1)

SAGE_ATTENTION3_IS_AVAILABLE = False
try:
    from sageattn3 import sageattn3_blackwell
    SAGE_ATTENTION3_IS_AVAILABLE = True
except ImportError:
    pass

FLASH_ATTENTION_IS_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_IS_AVAILABLE = True
except ImportError:
    if model_management.flash_attention_enabled():
        logging.error(f"\n\nTo use the `--use-flash-attention` feature, the `flash-attn` package must be installed first.\ncommand:\n\t{sys.executable} -m pip install flash-attn")
        exit(-1)

REGISTERED_ATTENTION_FUNCTIONS = {}
def register_attention_function(name: str, func: Callable):
    # avoid replacing existing functions
    if name not in REGISTERED_ATTENTION_FUNCTIONS:
        REGISTERED_ATTENTION_FUNCTIONS[name] = func
    else:
        logging.warning(f"Attention function {name} already registered, skipping registration.")

def get_attention_function(name: str, default: Any=...) -> Union[Callable, None]:
    if name == "optimized":
        return optimized_attention
    elif name not in REGISTERED_ATTENTION_FUNCTIONS:
        if default is ...:
            raise KeyError(f"Attention function {name} not found.")
        else:
            return default
    return REGISTERED_ATTENTION_FUNCTIONS[name]

from comfy.cli_args import args
import comfy.ops
ops = comfy.ops.disable_weight_init

FORCE_UPCAST_ATTENTION_DTYPE = model_management.force_upcast_attention_dtype()

def get_attn_precision(attn_precision, current_dtype):
    if args.dont_upcast_attention:
        return None

    if FORCE_UPCAST_ATTENTION_DTYPE is not None and current_dtype in FORCE_UPCAST_ATTENTION_DTYPE:
        return FORCE_UPCAST_ATTENTION_DTYPE[current_dtype]
    return attn_precision

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None, operations=ops):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out * 2, dtype=dtype, device=device)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            operations.Linear(dim, inner_dim, dtype=dtype, device=device),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim, dtype=dtype, device=device, operations=operations)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim_out, dtype=dtype, device=device)
        )

    def forward(self, x):
        return self.net(x)

def Normalize(in_channels, dtype=None, device=None):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)


def wrap_attn(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        remove_attn_wrapper_key = False
        try:
            if "_inside_attn_wrapper" not in kwargs:
                transformer_options = kwargs.get("transformer_options", None)
                remove_attn_wrapper_key = True
                kwargs["_inside_attn_wrapper"] = True
                if transformer_options is not None:
                    if "optimized_attention_override" in transformer_options:
                        return transformer_options["optimized_attention_override"](func, *args, **kwargs)
            return func(*args, **kwargs)
        finally:
            if remove_attn_wrapper_key:
                del kwargs["_inside_attn_wrapper"]
    return wrapper

@wrap_attn
def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    attn_precision = get_attn_precision(attn_precision, q.dtype)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    if kwargs.get("enable_gqa", False) and q.shape[-3] != k.shape[-3]:
        n_rep = q.shape[-3] // k.shape[-3]
        k = k.repeat_interleave(n_rep, dim=-3)
        v = v.repeat_interleave(n_rep, dim=-3)

    scale = kwargs.get("scale", dim_head ** -0.5)

    h = heads
    if skip_reshape:
         q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

    # force cast to fp32 to avoid overflowing
    if attn_precision == torch.float32:
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if exists(mask):
        if mask.dtype == torch.bool:
            mask = rearrange(mask, 'b ... -> b (...)') #TODO: check if this bool part matches pytorch attention
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        else:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
            sim.add_(mask)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)

    if skip_output_reshape:
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
        )
    else:
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
    return out

@wrap_attn
def attention_sub_quad(query, key, value, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    attn_precision = get_attn_precision(attn_precision, query.dtype)

    if skip_reshape:
        b, _, _, dim_head = query.shape
    else:
        b, _, dim_head = query.shape
        dim_head //= heads

    if "scale" in kwargs:
        # Pre-scale query to match requested scale (cancels internal 1/sqrt(dim_head))
        query = query * (kwargs["scale"] * dim_head ** 0.5)

    if skip_reshape:
        query = query.reshape(b * heads, -1, dim_head)
        value = value.reshape(b * heads, -1, dim_head)
        key = key.reshape(b * heads, -1, dim_head).movedim(1, 2)
    else:
        query = query.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
        value = value.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
        key = key.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 3, 1).reshape(b * heads, dim_head, -1)


    dtype = query.dtype
    upcast_attention = attn_precision == torch.float32 and query.dtype != torch.float32
    if upcast_attention:
        bytes_per_token = torch.finfo(torch.float32).bits//8
    else:
        bytes_per_token = torch.finfo(query.dtype).bits//8
    batch_x_heads, q_tokens, _ = query.shape
    _, _, k_tokens = key.shape

    mem_free_total, _ = model_management.get_free_memory(query.device, True)

    kv_chunk_size_min = None
    kv_chunk_size = None
    query_chunk_size = None

    for x in [4096, 2048, 1024, 512, 256]:
        count = mem_free_total / (batch_x_heads * bytes_per_token * x * 4.0)
        if count >= k_tokens:
            kv_chunk_size = k_tokens
            query_chunk_size = x
            break

    if query_chunk_size is None:
        query_chunk_size = 512

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

    hidden_states = efficient_dot_product_attention(
        query,
        key,
        value,
        query_chunk_size=query_chunk_size,
        kv_chunk_size=kv_chunk_size,
        kv_chunk_size_min=kv_chunk_size_min,
        use_checkpoint=False,
        upcast_attention=upcast_attention,
        mask=mask,
    )

    hidden_states = hidden_states.to(dtype)
    if skip_output_reshape:
        hidden_states = hidden_states.unflatten(0, (-1, heads))
    else:
        hidden_states = hidden_states.unflatten(0, (-1, heads)).transpose(1,2).flatten(start_dim=2)
    return hidden_states

@wrap_attn
def attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    attn_precision = get_attn_precision(attn_precision, q.dtype)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    scale = kwargs.get("scale", dim_head ** -0.5)

    if skip_reshape:
         q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

    mem_free_total = model_management.get_free_memory(q.device)

    if attn_precision == torch.float32:
        element_size = 4
        upcast = True
    else:
        element_size = q.element_size()
        upcast = False

    gb = 1024 ** 3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * element_size
    modifier = 3
    mem_required = tensor_size * modifier
    steps = 1


    if mem_required > mem_free_total:
        steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))
        # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
        #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

    if steps > 64:
        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
        raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                            f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

    # print("steps", steps, mem_required, mem_free_total, modifier, q.element_size(), tensor_size)
    first_op_done = False
    cleared_cache = False
    while True:
        try:
            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                if upcast:
                    with torch.autocast(enabled=False, device_type = 'cuda'):
                        s1 = einsum('b i d, b j d -> b i j', q[:, i:end].float(), k.float()) * scale
                else:
                    s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * scale

                if mask is not None:
                    if len(mask.shape) == 2:
                        s1 += mask[i:end]
                    else:
                        if mask.shape[1] == 1:
                            s1 += mask
                        else:
                            s1 += mask[:, i:end]

                s2 = s1.softmax(dim=-1).to(v.dtype)
                del s1
                first_op_done = True

                r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
                del s2
            break
        except Exception as e:
            model_management.raise_non_oom(e)
            if first_op_done == False:
                model_management.soft_empty_cache(True)
                if cleared_cache == False:
                    cleared_cache = True
                    logging.warning("out of memory error, emptying cache and trying again")
                    continue
                steps *= 2
                if steps > 64:
                    raise e
                logging.warning("out of memory error, increasing steps and trying again {}".format(steps))
            else:
                raise e

    del q, k, v

    if skip_output_reshape:
        r1 = (
            r1.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
        )
    else:
        r1 = (
            r1.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
    return r1

BROKEN_XFORMERS = False
try:
    x_vers = xformers.__version__
    # XFormers bug confirmed on all versions from 0.0.21 to 0.0.26 (q with bs bigger than 65535 gives CUDA error)
    BROKEN_XFORMERS = x_vers.startswith("0.0.2") and not x_vers.startswith("0.0.20")
except:
    pass

@wrap_attn
def attention_xformers(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    b = q.shape[0]
    dim_head = q.shape[-1]
    # check to make sure xformers isn't broken
    disabled_xformers = False

    if BROKEN_XFORMERS:
        if b * heads > 65535:
            disabled_xformers = True

    if not disabled_xformers:
        if torch.jit.is_tracing() or torch.jit.is_scripting():
            disabled_xformers = True

    if disabled_xformers:
        return attention_pytorch(q, k, v, heads, mask, skip_reshape=skip_reshape, **kwargs)

    if skip_reshape:
        # b h k d -> b k h d
        q, k, v = map(
            lambda t: t.permute(0, 2, 1, 3),
            (q, k, v),
        )
    # actually do the reshaping
    else:
        dim_head //= heads
        q, k, v = map(
            lambda t: t.reshape(b, -1, heads, dim_head),
            (q, k, v),
        )

    if mask is not None:
        # add a singleton batch dimension
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a singleton heads dimension
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # pad to a multiple of 8
        pad = 8 - mask.shape[-1] % 8
        # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
        # but when using separated heads, the shape has to be (B, H, Nq, Nk)
        # in flux, this matrix ends up being over 1GB
        # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
        mask_out = torch.empty([mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device)

        mask_out[..., :mask.shape[-1]] = mask
        # doesn't this remove the padding again??
        mask = mask_out[..., :mask.shape[-1]]
        mask = mask.expand(b, heads, -1, -1)

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    if skip_output_reshape:
        out = out.permute(0, 2, 1, 3)
    else:
        out = (
            out.reshape(b, -1, heads * dim_head)
        )

    return out

if model_management.is_nvidia(): #pytorch 2.3 and up seem to have this issue.
    SDP_BATCH_LIMIT = 2**15
else:
    #TODO: other GPUs ?
    SDP_BATCH_LIMIT = 2**31

@wrap_attn
def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    # Pass through extra SDPA kwargs (scale, enable_gqa) if provided
    # enable_gqa requires PyTorch 2.5+; older versions use manual KV expansion above
    sdpa_keys = ("scale", "enable_gqa") if TORCH_HAS_GQA else ("scale",)
    sdpa_extra = {k: v for k, v in kwargs.items() if k in sdpa_keys}

    if SDP_BATCH_LIMIT >= b:
        out = comfy.ops.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, **sdpa_extra)
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
    else:
        out = torch.empty((b, q.shape[2], heads * dim_head), dtype=q.dtype, layout=q.layout, device=q.device)
        for i in range(0, b, SDP_BATCH_LIMIT):
            m = mask
            if mask is not None:
                if mask.shape[0] > 1:
                    m = mask[i : i + SDP_BATCH_LIMIT]

            out[i : i + SDP_BATCH_LIMIT] = comfy.ops.scaled_dot_product_attention(
                q[i : i + SDP_BATCH_LIMIT],
                k[i : i + SDP_BATCH_LIMIT],
                v[i : i + SDP_BATCH_LIMIT],
                attn_mask=m,
                dropout_p=0.0, is_causal=False, **sdpa_extra
            ).transpose(1, 2).reshape(-1, q.shape[2], heads * dim_head)
    return out

@wrap_attn
def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    if kwargs.get("low_precision_attention", True) is False:
        return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

    exception_fallback = False
    if skip_reshape:
        b, _, _, dim_head = q.shape
        tensor_layout = "HND"
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head),
            (q, k, v),
        )
        tensor_layout = "NHD"

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    try:
        out = sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
    except Exception as e:
        logging.error("Error running sage attention: {}, using pytorch attention instead.".format(e))
        exception_fallback = True
    if exception_fallback:
        if tensor_layout == "NHD":
            q, k, v = map(
                lambda t: t.transpose(1, 2),
                (q, k, v),
            )
        return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)

    if tensor_layout == "HND":
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
    else:
        if skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(b, -1, heads * dim_head)
    return out

@wrap_attn
def attention3_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    exception_fallback = False
    if (q.device.type != "cuda" or
        q.dtype not in (torch.float16, torch.bfloat16) or
        mask is not None):
        return attention_pytorch(
            q, k, v, heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs
        )

    if skip_reshape:
        B, H, L, D = q.shape
        if H != heads:
            return attention_pytorch(
                q, k, v, heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=True,
                skip_output_reshape=skip_output_reshape,
                **kwargs
            )
        q_s, k_s, v_s = q, k, v
        N = q.shape[2]
        dim_head = D
    else:
        B, N, inner_dim = q.shape
        if inner_dim % heads != 0:
            return attention_pytorch(
                q, k, v, heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=False,
                skip_output_reshape=skip_output_reshape,
                **kwargs
            )
        dim_head = inner_dim // heads

    if dim_head >= 256 or N <= 1024:
        return attention_pytorch(
                q, k, v, heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs
            )

    if not skip_reshape:
        q_s, k_s, v_s = map(
            lambda t: t.view(B, -1, heads, dim_head).permute(0, 2, 1, 3).contiguous(),
            (q, k, v),
        )
        B, H, L, D = q_s.shape

    try:
        out = sageattn3_blackwell(q_s, k_s, v_s, is_causal=False)
    except Exception as e:
        exception_fallback = True
        logging.error("Error running SageAttention3: %s, falling back to pytorch attention.", e)

    if exception_fallback:
        if not skip_reshape:
            del q_s, k_s, v_s
        return attention_pytorch(
                q, k, v, heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=False,
                skip_output_reshape=skip_output_reshape,
                **kwargs
            )

    if skip_reshape:
        if not skip_output_reshape:
            out = out.permute(0, 2, 1, 3).reshape(B, L, H * D)
    else:
        if skip_output_reshape:
            pass
        else:
            out = out.permute(0, 2, 1, 3).reshape(B, L, H * D)

    return out

try:
    @torch.library.custom_op("flash_attention::flash_attn", mutates_args=())
    def flash_attn_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    dropout_p: float = 0.0, causal: bool = False) -> torch.Tensor:
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)


    @flash_attn_wrapper.register_fake
    def flash_attn_fake(q, k, v, dropout_p=0.0, causal=False):
        # Output shape is the same as q
        return q.new_empty(q.shape)
except AttributeError as error:
    FLASH_ATTN_ERROR = error

    def flash_attn_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    dropout_p: float = 0.0, causal: bool = False) -> torch.Tensor:
        assert False, f"Could not define flash_attn_wrapper: {FLASH_ATTN_ERROR}"

@wrap_attn
def attention_flash(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    try:
        if mask is not None:
            raise RuntimeError("Mask must not be set for Flash attention")
        out = flash_attn_wrapper(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            causal=False,
        ).transpose(1, 2)
    except Exception as e:
        logging.warning(f"Flash Attention failed, using default SDPA: {e}")
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    if not skip_output_reshape:
        out = (
            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        )
    return out


optimized_attention = attention_basic

if model_management.sage_attention_enabled():
    logging.info("Using sage attention")
    optimized_attention = attention_sage
elif model_management.flash_attention_enabled():
    logging.info("Using Flash Attention")
    optimized_attention = attention_flash
elif model_management.xformers_enabled():
    logging.info("Using xformers attention")
    optimized_attention = attention_xformers
elif model_management.pytorch_attention_enabled():
    logging.info("Using pytorch attention")
    optimized_attention = attention_pytorch
else:
    if args.use_split_cross_attention:
        logging.info("Using split optimization for attention")
        optimized_attention = attention_split
    else:
        logging.info("Using sub quadratic optimization for attention, if you have memory or speed issues try using: --use-split-cross-attention")
        optimized_attention = attention_sub_quad

optimized_attention_masked = optimized_attention


# register core-supported attention functions
if SAGE_ATTENTION_IS_AVAILABLE:
    register_attention_function("sage", attention_sage)
if SAGE_ATTENTION3_IS_AVAILABLE:
    register_attention_function("sage3", attention3_sage)
if FLASH_ATTENTION_IS_AVAILABLE:
    register_attention_function("flash", attention_flash)
if model_management.xformers_enabled():
    register_attention_function("xformers", attention_xformers)
register_attention_function("pytorch", attention_pytorch)
register_attention_function("sub_quad", attention_sub_quad)
register_attention_function("split", attention_split)


def optimized_attention_for_device(device, mask=False, small_input=False):
    if small_input:
        if model_management.pytorch_attention_enabled():
            return attention_pytorch #TODO: need to confirm but this is probably slightly faster for small inputs in all cases
        else:
            return attention_basic

    if device == torch.device("cpu"):
        return attention_sub_quad

    if mask:
        return optimized_attention_masked

    return optimized_attention


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., attn_precision=None, dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.attn_precision = attn_precision

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

    def forward(self, x, context=None, value=None, mask=None, transformer_options={}):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        if mask is None:
            out = optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision, transformer_options=transformer_options)
        else:
            out = optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision, transformer_options=transformer_options)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, ff_in=False, inner_dim=None,
                 disable_self_attn=False, disable_temporal_crossattention=False, switch_temporal_ca_to_sa=False, attn_precision=None, dtype=None, device=None, operations=ops):
        super().__init__()

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim
        self.attn_precision = attn_precision

        if self.ff_in:
            self.norm_in = operations.LayerNorm(dim, dtype=dtype, device=device)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device, operations=operations)

        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None, attn_precision=self.attn_precision, dtype=dtype, device=device, operations=operations)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device, operations=operations)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            context_dim_attn2 = None
            if not switch_temporal_ca_to_sa:
                context_dim_attn2 = context_dim

            self.attn2 = CrossAttention(query_dim=inner_dim, context_dim=context_dim_attn2,
                                heads=n_heads, dim_head=d_head, dropout=dropout, attn_precision=self.attn_precision, dtype=dtype, device=device, operations=operations)  # is self-attn if context is none
            self.norm2 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)

        self.norm1 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head
        extra_options["attn_precision"] = self.attn_precision

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1, transformer_options=transformer_options)

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x = n + x
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if self.attn2 is not None:
            n = self.norm2(x)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2, transformer_options=transformer_options)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x = n + x
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x = x_skip + x

        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, attn_precision=None, dtype=None, device=None, operations=ops):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
        if not use_linear:
            self.proj_in = operations.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0, dtype=dtype, device=device)
        else:
            self.proj_in = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(inner_dim,in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0, dtype=dtype, device=device)
        else:
            self.proj_out = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        transformer_options["activations_shape"] = list(x.shape)
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.movedim(1, 3).flatten(1, 2).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(x.shape[0], h, w, x.shape[-1]).movedim(3, 1).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
        attn_precision=None,
        dtype=None, device=None, operations=ops
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            attn_precision=attn_precision,
            dtype=dtype, device=device, operations=operations
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    # timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    attn_precision=attn_precision,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(
            operations.Linear(self.in_channels, time_embed_dim, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(time_embed_dim, self.in_channels, dtype=dtype, device=device),
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        transformer_options={}
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        transformer_options["activations_shape"] = list(x.shape)
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            if time_context is None:
                time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False, max_period=self.max_time_embed_period).to(x.dtype)
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_stack)
        ):
            transformer_options["block_index"] = it_
            x = block(
                x,
                context=spatial_context,
                transformer_options=transformer_options,
            )

            x_mix = x
            x_mix = x_mix + emb

            B, S, C = x_mix.shape
            x_mix = rearrange(x_mix, "(b t) s c -> (b s) t c", t=timesteps)
            x_mix = mix_block(x_mix, context=time_context, transformer_options=transformer_options)
            x_mix = rearrange(
                x_mix, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
            )

            x = self.time_mixer(x_spatial=x, x_temporal=x_mix, image_only_indicator=image_only_indicator)

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


