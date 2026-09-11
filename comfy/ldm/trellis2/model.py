import torch
import torch.nn.functional as F
import torch.nn as nn
import comfy.model_management
import comfy.ops
from comfy.ldm.trellis2.vae import SparseTensor, SparseLinear, sparse_cat, VarLenTensor
from typing import Optional, Tuple, Literal, Union, List
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.genmo.joint_model.layers import TimestepEmbedder
from comfy.ldm.flux.math import apply_rope, apply_rope1


def dense_attention(q, k, v, **kwargs):
    heads = q.shape[2]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = optimized_attention(q, k, v, heads, skip_output_reshape=True, skip_reshape=True, **kwargs)
    return out.permute(0, 2, 1, 3)


def _to_padded(t):
    if not isinstance(t, VarLenTensor):
        return t, None
    seqlens = [item.stop - item.start for item in t.layout]
    if len(set(seqlens)) == 1:
        return t.feats.view(len(t.layout), seqlens[0], *t.feats.shape[1:]), None
    max_seqlen = max(seqlens)
    padded = t.feats.new_zeros((len(t.layout), max_seqlen, *t.feats.shape[1:]))
    valid = torch.zeros((len(t.layout), max_seqlen), dtype=torch.bool, device=t.device)
    for i, (item, seqlen) in enumerate(zip(t.layout, seqlens)):
        padded[i, :seqlen] = t.feats[item]
        valid[i, :seqlen] = True
    return padded, valid


def sparse_attention(q, k, v, **kwargs):
    q_padded, q_valid = _to_padded(q)
    k_padded, k_valid = _to_padded(k)
    v_padded, _ = _to_padded(v)
    if k_valid is not None:
        mask = torch.zeros((k_valid.shape[0], 1, 1, k_valid.shape[1]),
                           dtype=q_padded.dtype, device=q_padded.device)
        mask.masked_fill_(~k_valid[:, None, None, :], -torch.finfo(q_padded.dtype).max)
        kwargs["mask"] = mask
    out = dense_attention(q_padded, k_padded, v_padded, **kwargs)
    if isinstance(q, VarLenTensor):
        if q_valid is not None:
            return q.replace(out[q_valid])
        return q.replace(out.reshape(-1, *out.shape[2:]))
    return out


class SparseGELU(nn.GELU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))

class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, device=None, dtype=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio), device=device, dtype=dtype, operations=operations),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels, device=device, dtype=dtype, operations=operations),
        )

    def forward(self, x: VarLenTensor) -> VarLenTensor:
        return self.mlp(x)

class MultiHeadRMSNorm(nn.Module):
    # Per-head qk-norm for both sparse (VarLenTensor) and dense inputs. gamma is [heads, dim]
    # (per-head), so it's a broadcast multiply rather than F.rms_norm's 1-D weight
    def __init__(self, dim: int, heads: int, device=None, dtype=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(heads, dim, device=device, dtype=dtype))

    def forward(self, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            return x.replace(F.rms_norm(x.feats, (x.feats.shape[-1],)) * comfy.ops.cast_to_input(self.gamma, x.feats))
        return F.rms_norm(x, (x.shape[-1],)) * comfy.ops.cast_to_input(self.gamma, x)

class SparseRotaryPositionEmbedder(nn.Module):
    def __init__(self, head_dim: int, dim: int = 3, rope_freq: Tuple[float, float] = (1.0, 10000.0), device=None):
        super().__init__()
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32, device=device) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** self.freqs)

    def _get_freqs_cis(self, coords: torch.Tensor) -> torch.Tensor:
        phases_list = []
        for i in range(self.dim):
            phases_list.append(torch.outer(coords[..., i], self.freqs.to(coords.device)))

        phases = torch.cat(phases_list, dim=-1)

        if phases.shape[-1] < self.head_dim // 2:
            padn = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat([phases, torch.zeros(*phases.shape[:-1], padn, device=phases.device)], dim=-1)

        cos = torch.cos(phases)
        sin = torch.sin(phases)

        f_cis_0 = torch.stack([cos, sin], dim=-1)
        f_cis_1 = torch.stack([-sin, cos], dim=-1)
        freqs_cis = torch.stack([f_cis_0, f_cis_1], dim=-1)

        return freqs_cis

    def forward(self, q, k=None):
        cache_name = f'rope_cis_{self.dim}d_f{self.rope_freq[1]}_hd{self.head_dim}'
        freqs_cis = q.get_spatial_cache(cache_name)

        if freqs_cis is None:
            coords = q.coords[..., 1:].to(torch.float32)
            freqs_cis = self._get_freqs_cis(coords)
            q.register_spatial_cache(cache_name, freqs_cis)

        if q.feats.ndim == 3:
            f_cis = freqs_cis.unsqueeze(1)
        else:
            f_cis = freqs_cis

        if k is None:
            return q.replace(apply_rope1(q.feats, f_cis))

        q_feats, k_feats = apply_rope(q.feats, k.feats, f_cis)
        return q.replace(q_feats), k.replace(k_feats)


class RotaryPositionEmbedder(SparseRotaryPositionEmbedder):
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self._get_freqs_cis(coords) # [L, head_dim/2, 2, 2]

class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = operations.Linear(channels, channels * 3, bias=qkv_bias, device=device, dtype=dtype)
        else:
            self.to_q = operations.Linear(channels, channels, bias=qkv_bias, device=device, dtype=dtype)
            self.to_kv = operations.Linear(self.ctx_channels, channels * 2, bias=qkv_bias, device=device, dtype=dtype)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, device=device, dtype=dtype)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, device=device, dtype=dtype)

        self.to_out = operations.Linear(channels, channels, device=device, dtype=dtype)

        if self._type == "self":
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, device=device)

    @staticmethod
    def _linear(module: nn.Linear, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[VarLenTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[VarLenTensor, torch.Tensor], num_fused: int) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, VarLenTensor) else x_feats

    def forward(self, x: SparseTensor, context: Optional[Union[VarLenTensor, torch.Tensor]] = None, transformer_options=None) -> SparseTensor:
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            q, k, v = qkv.unbind(dim=-3)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            q, k = self.rope(q, k)
            h = sparse_attention(q, k, v, transformer_options=transformer_options)
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            k, v = kv.unbind(dim=-3)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            h = sparse_attention(q, k, v, transformer_options=transformer_options)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h

def _split_proj_context(context):
    if not isinstance(context, dict):
        return context, None
    global_ctx = context["global"]
    if "proj" in context:
        return global_ctx, context["proj"]
    if "proj_semantic" in context and "proj_color" in context:
        return global_ctx, (context["proj_semantic"], context["proj_color"])
    return global_ctx, None


class ProjectAttentionSparse(nn.Module):
    def __init__(self, cross_attn_block: nn.Module, channels: int, proj_in_channels: int,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = operations.Linear(proj_in_channels, channels, bias=True,
                                             device=device, dtype=dtype)

    def forward(self, x: SparseTensor, context, transformer_options=None) -> SparseTensor:
        global_ctx, proj_in = _split_proj_context(context)
        global_out = self.cross_attn_block(x, global_ctx, transformer_options=transformer_options)
        if isinstance(proj_in, tuple):
            proj_in = torch.cat([proj_in[0], proj_in[1]], dim=-1)
        proj_out = self.proj_linear(proj_in)
        return global_out.replace(global_out.feats + proj_out.to(global_out.feats.dtype))


class ProjectAttentionDense(nn.Module):
    def __init__(self, cross_attn_block: nn.Module, channels: int, proj_in_channels: int,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = operations.Linear(proj_in_channels, channels, bias=True,
                                             device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, context, transformer_options=None) -> torch.Tensor:
        global_ctx, proj_in = _split_proj_context(context)
        global_out = self.cross_attn_block(x, global_ctx, transformer_options=transformer_options)
        if isinstance(proj_in, tuple):
            proj_in = torch.cat([proj_in[0], proj_in[1]], dim=-1)
        proj_out = self.proj_linear(proj_in)
        return global_out + proj_out.to(global_out.dtype)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        image_attn_mode: Literal["global", "proj"] = "global",
        proj_in_channels: Optional[int] = None,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.share_mod = share_mod
        self.image_attn_mode = image_attn_mode
        self.norm1 = operations.LayerNorm(channels, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(channels, elementwise_affine=True, eps=1e-6, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(channels, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm,
            device=device, dtype=dtype, operations=operations
        )
        cross_inner = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            device=device, dtype=dtype, operations=operations
        )
        if image_attn_mode == "global":
            self.cross_attn = cross_inner
        else:
            if proj_in_channels is None:
                raise ValueError("proj_in_channels must be set when image_attn_mode != 'global'")
            self.cross_attn = ProjectAttentionSparse(
                cross_inner, channels, proj_in_channels,
                device=device, dtype=dtype, operations=operations,
            )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            device=device, dtype=dtype, operations=operations
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(channels, 6 * channels, bias=True, device=device, dtype=dtype)
            )
        else:
            self.modulation = nn.Parameter(torch.empty(6 * channels, device=device, dtype=dtype))

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context, transformer_options=None) -> SparseTensor:
        if self.share_mod:
            modulation = comfy.ops.cast_to_input(self.modulation, mod)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (modulation + mod).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        # Fuse the (mul + add) and (mul + residual) pairs into addcmul
        b_map = x.batch_boardcast_map

        h_feats = self.norm1(x.feats)
        h_feats = torch.addcmul(shift_msa[b_map], h_feats, (1 + scale_msa)[b_map])
        h = self.self_attn(x.replace(h_feats), transformer_options=transformer_options)
        x = x.replace(torch.addcmul(x.feats, h.feats, gate_msa[b_map]))

        h = x.replace(self.norm2(x.feats))
        if self.image_attn_mode == "global":
            global_ctx, _ = _split_proj_context(context)
            h = self.cross_attn(h, global_ctx, transformer_options=transformer_options)
        else:
            h = self.cross_attn(h, context, transformer_options=transformer_options)
        x = x + h

        h_feats = self.norm3(x.feats)
        h_feats = torch.addcmul(shift_mlp[b_map], h_feats, (1 + scale_mlp)[b_map])
        h = self.mlp(x.replace(h_feats))
        x = x.replace(torch.addcmul(x.feats, h.feats, gate_mlp[b_map]))
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context, transformer_options=None) -> SparseTensor:
        return self._forward(x, mod, context, transformer_options=transformer_options)


class SLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        image_attn_mode: Literal["global", "proj"] = "global",
        proj_in_channels: Optional[int] = None,
        dtype = None, device = None, operations = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.image_attn_mode = image_attn_mode
        self.proj_in_channels = proj_in_channels
        self.dtype = dtype

        self.t_embedder = TimestepEmbedder(model_channels, device=device, dtype=dtype, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(model_channels, 6 * model_channels, bias=True, device=device, dtype=dtype)
            )

        self.input_layer = SparseLinear(in_channels, model_channels, device=device, dtype=dtype, operations=operations)

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                image_attn_mode=image_attn_mode,
                proj_in_channels=proj_in_channels,
                device=device, dtype=dtype, operations=operations
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = SparseLinear(model_channels, out_channels, device=device, dtype=dtype, operations=operations)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        x: SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[SparseTensor] = None,
        transformer_options=None,
        **kwargs,
    ) -> SparseTensor:
        if concat_cond is not None:
            x = sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = VarLenTensor.from_tensor_list(cond)

        h = self.input_layer(x)
        t_emb = self.t_embedder(t, out_dtype=t.dtype)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        for block in self.blocks:
            h = block(h, t_emb, cond, transformer_options=transformer_options)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h

class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, device=None, dtype=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(channels, int(channels * mlp_ratio), device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operations.Linear(int(channels * mlp_ratio), channels, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = operations.Linear(channels, channels * 3, bias=qkv_bias, dtype=dtype, device=device)
        else:
            self.to_q = operations.Linear(channels, channels, bias=qkv_bias, device=device, dtype=dtype)
            self.to_kv = operations.Linear(self.ctx_channels, channels * 2, bias=qkv_bias, device=device, dtype=dtype)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, device=device, dtype=dtype)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, device=device, dtype=dtype)

        self.to_out = operations.Linear(channels, channels, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                phases: Optional[torch.Tensor] = None, transformer_options=None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            q, k, v = qkv.unbind(dim=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            assert phases is not None, "Phases must be provided for RoPE"
            # phases is [L, head_dim/2, 2, 2]; broadcast to [1, L, 1, ...]
            # to align with q/k of shape [B, L, H, head_dim].
            f_cis = comfy.model_management.cast_to(phases, device=q.device).unsqueeze(0).unsqueeze(2)
            q, k = apply_rope(q, k, f_cis)
            h = dense_attention(q, k, v, transformer_options=transformer_options)
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            k, v = kv.unbind(dim=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            h = dense_attention(q, k, v, transformer_options=transformer_options)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h

class ModulatedTransformerCrossBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        image_attn_mode: Literal["global", "proj"] = "global",
        proj_in_channels: Optional[int] = None,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.share_mod = share_mod
        self.image_attn_mode = image_attn_mode
        self.norm1 = operations.LayerNorm(channels, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(channels, elementwise_affine=True, eps=1e-6, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(channels, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm,
            device=device, dtype=dtype, operations=operations
        )
        cross_inner = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            device=device, dtype=dtype, operations=operations
        )
        if image_attn_mode == "global":
            self.cross_attn = cross_inner
        else:
            if proj_in_channels is None:
                raise ValueError("proj_in_channels must be set when image_attn_mode != 'global'")
            self.cross_attn = ProjectAttentionDense(
                cross_inner, channels, proj_in_channels,
                device=device, dtype=dtype, operations=operations,
            )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            device=device, dtype=dtype, operations=operations
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device))
        else:
            self.modulation = nn.Parameter(torch.empty(6 * channels, device=device, dtype=dtype))

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context,
                 phases: Optional[torch.Tensor] = None, transformer_options=None) -> torch.Tensor:
        if self.share_mod:
            mod = comfy.ops.cast_to_input(self.modulation, mod) + mod
        else:
            mod = self.adaLN_modulation(mod)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(6, dim=-1)

        h = torch.addcmul(shift_msa, self.norm1(x), 1 + scale_msa)
        h = self.self_attn(h, phases=phases, transformer_options=transformer_options)
        x = torch.addcmul(x, h, gate_msa)

        h = self.norm2(x)
        if self.image_attn_mode == "global":
            global_ctx, _ = _split_proj_context(context)
            h = self.cross_attn(h, global_ctx, transformer_options=transformer_options)
        else:
            h = self.cross_attn(h, context, transformer_options=transformer_options)
        x = x + h

        h = torch.addcmul(shift_mlp, self.norm3(x), 1 + scale_mlp)
        h = self.mlp(h)
        x = torch.addcmul(x, h, gate_mlp)
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context,
                phases: Optional[torch.Tensor] = None, transformer_options=None) -> torch.Tensor:
        return self._forward(x, mod, context, phases, transformer_options=transformer_options)


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        image_attn_mode: Literal["global", "proj"] = "global",
        proj_in_channels: Optional[int] = None,
        operations=None,
        device = None,
        dtype = None,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.image_attn_mode = image_attn_mode
        self.proj_in_channels = proj_in_channels
        self.dtype = dtype
        self.device = device

        self.t_embedder = TimestepEmbedder(model_channels, dtype=dtype, device=device, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(model_channels, 6 * model_channels, bias=True, device=device, dtype=dtype)
            )

        pos_embedder = RotaryPositionEmbedder(self.model_channels // self.num_heads, 3, device=device)
        coords = torch.meshgrid(*[torch.arange(res, device=self.device, dtype=dtype) for res in [resolution] * 3], indexing='ij')
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        rope_phases = pos_embedder(coords)
        self.register_buffer("rope_phases", rope_phases, persistent=False)

        self.input_layer = operations.Linear(in_channels, model_channels, device=device, dtype=dtype)

        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                image_attn_mode=image_attn_mode,
                proj_in_channels=proj_in_channels,
                device=device, dtype=dtype, operations=operations
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = operations.Linear(model_channels, out_channels, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                transformer_options=None) -> torch.Tensor:
        x = x.view(x.shape[0], self.in_channels, *[self.resolution] * 3)

        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        t_emb = self.t_embedder(t, out_dtype=t.dtype)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        for block in self.blocks:
            h = block(h, t_emb, cond, self.rope_phases, transformer_options=transformer_options)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3).contiguous()

        return h


# Pixal3D ProjGrid math
# World frame uses world Y as depth, camera looks along -Z local
# transform_matrix is camera-to-world (inverted internally). Intrinsics: fx = 16 / tan(fov/2) with sensor_width = 32mm.

_PROJ_GRID_ROTATION = torch.tensor(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]]
)

_PROJ_FRONT_VIEW_TRANSFORM = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, -2.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
)


def build_proj_transform_matrix(distance: torch.Tensor, batch_size: int,
                                 device, dtype=torch.float32) -> torch.Tensor:
    T = _PROJ_FRONT_VIEW_TRANSFORM.to(device=device, dtype=dtype)
    T = T.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if distance.ndim == 0:
        distance = distance.expand(batch_size)
    T[:, 1, 3] = -distance.to(device=device, dtype=dtype)
    return T


def _project_points_to_image(points_world: torch.Tensor, transform_matrix: torch.Tensor,
                             camera_angle_x: torch.Tensor, resolution: int):
    B, N, _ = points_world.shape
    ones = torch.ones((B, N, 1), device=points_world.device, dtype=points_world.dtype)
    homo = torch.cat([points_world, ones], dim=-1)
    world_to_camera = torch.linalg.inv(transform_matrix.float()).to(transform_matrix.dtype)
    p_cam = torch.bmm(homo, world_to_camera.transpose(-2, -1))[..., :3]
    x_cam, y_cam, z_cam = p_cam.unbind(dim=-1)
    depth = -z_cam
    sensor_width = 32.0
    focal_length = 16.0 / torch.tan(camera_angle_x / 2.0)
    focal_px = focal_length * resolution / sensor_width
    focal_px = focal_px.to(p_cam.dtype).unsqueeze(1)
    denom = (-z_cam + 1e-8)
    x_pix = focal_px * x_cam / denom + resolution / 2.0
    y_pix = -focal_px * y_cam / denom + resolution / 2.0
    valid = ((x_pix >= 0) & (x_pix < resolution) &
             (y_pix >= 0) & (y_pix < resolution) & (depth > 0))
    return torch.stack([x_pix, y_pix], dim=-1), depth, valid


def _sample_features(feature_map: torch.Tensor, uv_ndc: torch.Tensor) -> torch.Tensor:
    B, C, _, _ = feature_map.shape
    grid = uv_ndc.view(B, -1, 1, 2).to(feature_map.dtype)
    feat = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return feat.squeeze(-1)


def _coords_to_proj_world(coords: torch.Tensor, resolution: int, mesh_scale: torch.Tensor):
    if resolution < 1:
        raise ValueError(f"resolution must be positive, got {resolution}")
    batch_ids = coords[:, 0].long()
    if resolution == 1:
        norm = coords[:, 1:].to(torch.float32) * 0.0
    else:
        norm = coords[:, 1:].to(torch.float32) / (resolution - 1) * 2.0 - 1.0
    R = _PROJ_GRID_ROTATION.to(device=coords.device, dtype=torch.float32)
    rotated = norm @ R.T
    scale_per_voxel = mesh_scale.to(coords.device).reshape(-1)[batch_ids % mesh_scale.numel()]
    world = rotated / scale_per_voxel.unsqueeze(-1) / 2.0
    return world, batch_ids


def _dense_grid_proj_world(resolution: int, mesh_scale: torch.Tensor,
                           batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
    one = torch.linspace(-1.0, 1.0, resolution, device=device, dtype=dtype)
    x, y, z = torch.meshgrid(one, one, one, indexing="ij")
    grid = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    R_rot = _PROJ_GRID_ROTATION.to(device=device, dtype=dtype)
    grid = grid @ R_rot.T
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if mesh_scale.ndim == 0:
        mesh_scale = mesh_scale.expand(batch_size)
    grid = grid / mesh_scale.to(device=device, dtype=dtype).view(-1, 1, 1) / 2.0
    return grid


def _back_project_to_tokens(
    coords_world: torch.Tensor,
    feature_map: torch.Tensor,
    transform_matrix: torch.Tensor,
    camera_angle_x: torch.Tensor,
    image_resolution: int,
    batch_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if coords_world.dim() == 2:
        assert batch_ids is not None
        n_cond = transform_matrix.shape[0]
        out = torch.zeros((coords_world.shape[0], feature_map.shape[1]),
                          device=feature_map.device, dtype=feature_map.dtype)
        # multi-seed: latent sample b uses conditioning b % n_cond
        for b in range(int(batch_ids.max().item()) + 1):
            mask = batch_ids == b
            if not mask.any():
                continue
            c = b % n_cond
            p = coords_world[mask].unsqueeze(0)
            uv, _, _ = _project_points_to_image(
                p, transform_matrix[c:c+1], camera_angle_x[c:c+1], image_resolution)
            uv_ndc = (uv + 0.5) / image_resolution * 2.0 - 1.0
            # padding_mode='border' is load-bearing: masking out-of-frame voxels confuses
            # the SS DiT (~half the voxels go to zero, producing low poly + rotation drift).
            sampled = _sample_features(feature_map[c:c+1], uv_ndc)
            sampled = sampled.squeeze(0).transpose(0, 1)
            out[mask] = sampled
        return out
    else:
        uv, _, _ = _project_points_to_image(
            coords_world, transform_matrix, camera_angle_x, image_resolution)
        uv_ndc = (uv + 0.5) / image_resolution * 2.0 - 1.0
        sampled = _sample_features(feature_map, uv_ndc)
        out = sampled.transpose(1, 2)
        return out


def _select_stage_entry(proj_pack: dict, stage: Optional[str]):
    """Returns (feature_map_lr, feature_map_hr_or_None, image_resolution)."""
    stages = proj_pack.get("stages")
    if stages is not None and stage is not None and stage in stages:
        entry = stages[stage]
        return entry["feature_map"], entry.get("feature_map_hr"), int(entry.get("image_resolution", 1024))
    if "feature_map" in proj_pack:
        return proj_pack["feature_map"], proj_pack.get("feature_map_hr"), int(proj_pack.get("image_resolution", 1024))
    raise ValueError(f"proj_feat_pack has no usable feature_map (stage={stage!r})")


def compute_stage_proj_feats(
    proj_pack: dict,
    stage: str,
    coords: Optional[torch.Tensor] = None,
    coord_resolution: Optional[int] = None,
    dense_grid_resolution: Optional[int] = None,
    batch_size: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    """Back-project a stage's feature maps onto sparse coords [N, C] or the dense SS grid [B, R^3, C]; views at stride num_views are averaged."""
    if device is None:
        device = coords.device if coords is not None else proj_pack["mesh_scale"].device
    mesh_scale = proj_pack["mesh_scale"].to(device)
    T = proj_pack["transform_matrix"].to(device)
    cam_angle = proj_pack["camera_angle_x"].to(device)
    num_views = int(proj_pack.get("num_views", 1))
    feat_map_lr, feat_map_hr, image_resolution = _select_stage_entry(proj_pack, stage)

    if coords is not None:
        if coord_resolution is None:
            raise ValueError("compute_stage_proj_feats: coord_resolution required when coords is given")
        coords_world, batch_ids = _coords_to_proj_world(coords, coord_resolution, mesh_scale)
    else:
        if dense_grid_resolution is None or batch_size is None:
            raise ValueError("compute_stage_proj_feats: dense_grid_resolution + batch_size required for dense path")
        coords_world = _dense_grid_proj_world(dense_grid_resolution, mesh_scale, batch_size,
                                              device=device, dtype=torch.float32)
        batch_ids = None

    out = None
    for v in range(num_views):
        sel = slice(v, None, num_views)
        proj = _back_project_to_tokens(coords_world, feat_map_lr[sel].to(device), T[sel], cam_angle[sel],
                                       image_resolution=image_resolution, batch_ids=batch_ids)
        if feat_map_hr is not None:
            proj_hr = _back_project_to_tokens(coords_world, feat_map_hr[sel].to(device), T[sel], cam_angle[sel],
                                              image_resolution=image_resolution, batch_ids=batch_ids)
            proj = torch.cat([proj, proj_hr], dim=-1)
        # average views in fp32 like upstream
        out = proj.float() if out is None else out.add_(proj)
    return out.div_(num_views).to(proj.dtype)


def _shape_proj_cond(global_cond: torch.Tensor, image_attn_mode: str,
                     proj_feats: Optional[torch.Tensor],
                     batch_ids: Optional[torch.Tensor] = None,
                     eval_batch: Optional[int] = None,
                     logical_batch: Optional[int] = None,
                     proj_in_channels: Optional[int] = None,
                     stage: Optional[str] = None,
                     cond_or_uncond: Optional[list] = None):
    """Take pre-computed per-token proj features (from compute_stage_proj_feats),
    apply CFG-batch duplication + uncond-slot zeroing, and wrap into the
    ``{"global", "proj"}`` context dict consumed by ProjectAttention.

    proj_feats shape:
      sparse (shape/texture): [N_voxels, C]   (batch_ids gives per-voxel batch)
      dense  (SS):            [B, N, C]
    """
    if image_attn_mode == "global":
        return global_cond
    if proj_feats is None:
        raise ValueError(f"image_attn_mode={image_attn_mode!r} but trellis2_proj_feats is missing — "
                         f"the stage setup node (or Pixal3DConditioning for SS) should have computed it.")
    if proj_in_channels is not None and proj_feats.shape[-1] != proj_in_channels:
        hint = ""
        if proj_feats.shape[-1] < proj_in_channels:
            hint = (" — the shape/texture stages need the NAF-upsampled feature map from "
                    "a Pixal3D DINOv3 checkpoint containing naf. weights.")
        raise ValueError(
            f"proj_feats for stage {stage!r} has {proj_feats.shape[-1]} channels, "
            f"sub-model expects {proj_in_channels}.{hint}"
        )

    # multi-seed: latent sample i uses conditioning i % B
    if batch_ids is None and logical_batch is not None and proj_feats.shape[0] != logical_batch:
        reps = -(-logical_batch // proj_feats.shape[0])
        proj_feats = proj_feats.repeat((reps,) + (1,) * (proj_feats.ndim - 1))[:logical_batch]

    # CFG-duplicate proj_feats to match the model's eval batch.
    if eval_batch is not None and logical_batch is not None and eval_batch > logical_batch:
        repeats = eval_batch // logical_batch
        if batch_ids is None:
            proj_feats = proj_feats.repeat((repeats,) + (1,) * (proj_feats.ndim - 1))
        else:
            proj_feats = proj_feats.repeat((repeats, 1))

    # zero proj for any uncond batch slot
    if cond_or_uncond is not None and eval_batch is not None:
        uncond_slots = [i for i, v in enumerate(cond_or_uncond) if v == 1]
        if uncond_slots:
            uncond_batches = [batch_idx
                              for slot in uncond_slots
                              for batch_idx in range(slot * logical_batch,
                                                     min((slot + 1) * logical_batch, eval_batch))]
            uncond_idx = torch.tensor(uncond_batches, device=proj_feats.device, dtype=torch.long)
            if batch_ids is None:
                proj_feats = proj_feats.clone()
                proj_feats[uncond_idx] = 0
            else:
                neg_mask = torch.isin(batch_ids, uncond_idx).unsqueeze(-1).to(proj_feats.dtype)
                proj_feats = proj_feats * (1.0 - neg_mask)
    return {"global": global_cond, "proj": proj_feats}

class Trellis2(nn.Module):
    def __init__(self, resolution,
                 in_channels = 32,
                 out_channels = 32,
                 model_channels = 1536,
                 cond_channels = 1024,
                 num_blocks = 30,
                 num_heads = 12,
                 mlp_ratio = 5.3334,
                 share_mod = True,
                 qk_rms_norm = True,
                 qk_rms_norm_cross = True,
                 init_txt_model=False, # for now
                 image_attn_mode_structure: str = "global",
                 proj_in_channels_structure: Optional[int] = None,
                 image_attn_mode_shape: str = "global",
                 proj_in_channels_shape: Optional[int] = None,
                 image_attn_mode_texture: str = "global",
                 proj_in_channels_texture: Optional[int] = None,
                 dtype=None, device=None, operations=None, **kwargs):

        super().__init__()
        self.dtype = dtype
        args = {
            "out_channels":out_channels, "num_blocks":num_blocks, "cond_channels" :cond_channels,
            "model_channels":model_channels, "num_heads":num_heads, "mlp_ratio": mlp_ratio, "share_mod": share_mod,
            "qk_rms_norm": qk_rms_norm, "qk_rms_norm_cross": qk_rms_norm_cross, "device": device, "dtype": dtype, "operations": operations
        }
        self.image_attn_mode_structure = image_attn_mode_structure
        self.image_attn_mode_shape = image_attn_mode_shape
        self.image_attn_mode_texture = image_attn_mode_texture
        shape_proj_kwargs = {"image_attn_mode": image_attn_mode_shape, "proj_in_channels": proj_in_channels_shape}
        tex_proj_kwargs = {"image_attn_mode": image_attn_mode_texture, "proj_in_channels": proj_in_channels_texture}
        struct_proj_kwargs = {"image_attn_mode": image_attn_mode_structure, "proj_in_channels": proj_in_channels_structure}
        txt_only = kwargs.get("txt_only", False)
        if not txt_only:
            self.img2shape = SLatFlowModel(resolution=resolution, in_channels=in_channels, **shape_proj_kwargs, **args)
            self.shape2txt = None
            if init_txt_model:
                self.shape2txt = SLatFlowModel(resolution=resolution, in_channels=in_channels*2, **tex_proj_kwargs, **args)
            self.img2shape_512 = SLatFlowModel(resolution=32, in_channels=in_channels, **shape_proj_kwargs, **args)
            args.pop("out_channels")
            self.structure_model = SparseStructureFlowModel(resolution=16, in_channels=8, out_channels=8, **struct_proj_kwargs, **args)
        else:
            self.shape2txt = SLatFlowModel(resolution=resolution, in_channels=in_channels*2, **tex_proj_kwargs, **args)

    def forward(self, x, timestep, context, **kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        timestep = timestep.to(x.dtype)
        embeds = kwargs.get("embeds")
        if embeds is None:
            raise ValueError("Trellis2.forward requires 'embeds' in kwargs")

        # Per-stage cascade metadata
        coords = kwargs.get("trellis2_coords")
        coord_counts = kwargs.get("trellis2_coord_counts")
        mode = kwargs.get("trellis2_generation_mode", "structure_generation")
        # Pre-computed per-stage back-projected features
        proj_feats = kwargs.get("trellis2_proj_feats")

        is_first_shape_pass = False
        if mode == "shape_generation_512":
            is_first_shape_pass = True
            mode = "shape_generation"

        if coords is not None:
            x = x.squeeze(-1).transpose(1, 2)
            is_sparse_mode = True
        else:
            mode = "structure_generation"
            is_sparse_mode = False

        if x.size(-1) == 16 and x.size(-2) == 16:
            mode = "structure_generation"
            is_sparse_mode = False

        if not is_sparse_mode:
            bsz = x.size(0)
            x = x[:, :8]
            x = x.view(bsz, 8, 16, 16, 16)

        if is_sparse_mode and not is_first_shape_pass:
            context = embeds

        if is_sparse_mode:
            t_eval = timestep
            c_eval = context

            B, N, C = x.shape

            # Vectorized SparseTensor Construction
            if mode in ["shape_generation", "texture_generation"]:
                if coord_counts is not None:
                    logical_batch = coord_counts.shape[0]
                    # Duplicate sparse coords when the sampler asks for >1 cond
                    # (CFG or otherwise). Each duplicate is offset along col 0
                    # so SparseTensor sees a fresh logical batch.
                    if B > logical_batch:
                        reps = B // logical_batch
                        c_copies = []
                        for i in range(reps):
                            c = coords.clone()
                            c[:, 0] += i * logical_batch
                            c_copies.append(c)
                        batched_coords = torch.cat(c_copies, dim=0)
                        counts_eval = coord_counts.repeat(reps)
                    else:
                        batched_coords = coords
                        counts_eval = coord_counts

                    # Boolean mask [B, N] to drop the padded zeros instantly
                    mask = torch.arange(N, device=x.device).unsqueeze(0) < counts_eval.unsqueeze(1)
                    feats_flat = x[mask]
                else:
                    feats_flat = x.reshape(-1, C)
                    coords_list = []
                    for i in range(B):
                        c = coords.clone()
                        c[:, 0] = i
                        coords_list.append(c)
                    batched_coords = torch.cat(coords_list, dim=0)
                    mask = None
            else:
                batched_coords = coords
                feats_flat = x
                mask = None

            x_st = SparseTensor(
                feats=feats_flat,
                coords=batched_coords.to(device=x.device, dtype=torch.int32),
                shape=torch.Size([B] + list(feats_flat.shape[1:])),
            )

        if mode == "shape_generation":
            shape_attn = self.image_attn_mode_shape
            if shape_attn != "global":
                sub_model = self.img2shape_512 if is_first_shape_pass else self.img2shape
                stage_name = "shape_512" if is_first_shape_pass else "shape_1024"
                # batched_coords carries CFG-doubled batch ids in col 0; per-voxel
                # batch_ids drive uncond-slot masking inside _shape_proj_cond.
                batch_ids = batched_coords[:, 0].long()
                logical_batch = coord_counts.shape[0] if coord_counts is not None else B
                c_eval = _shape_proj_cond(c_eval, shape_attn, proj_feats,
                                          batch_ids=batch_ids,
                                          eval_batch=B, logical_batch=logical_batch,
                                          proj_in_channels=sub_model.proj_in_channels,
                                          stage=stage_name,
                                          cond_or_uncond=cond_or_uncond)
            if is_first_shape_pass:
                out = self.img2shape_512(x_st, t_eval, c_eval, transformer_options=transformer_options)
            else:
                out = self.img2shape(x_st, t_eval, c_eval, transformer_options=transformer_options)

        elif mode == "texture_generation":
            if self.shape2txt is None:
                raise ValueError("Checkpoint for Trellis2 doesn't include texture generation!")
            slat = kwargs.get("trellis2_shape_slat")
            if slat is None:
                raise ValueError("shape_slat can't be None")

            slat_feats = slat
            # Duplicate shape context if CFG is active
            if coord_counts is not None and B > coord_counts.shape[0]:
                slat_feats = slat_feats.repeat(B // coord_counts.shape[0], 1)
            elif coord_counts is None:
                slat_feats = slat_feats[:N].repeat(B, 1)

            x_st = x_st.replace(feats=torch.cat([x_st.feats, slat_feats.to(x_st.feats.device)], dim=-1))
            tex_attn = self.image_attn_mode_texture
            if tex_attn != "global":
                batch_ids = batched_coords[:, 0].long()
                logical_batch = coord_counts.shape[0] if coord_counts is not None else B
                c_eval = _shape_proj_cond(c_eval, tex_attn, proj_feats,
                                          batch_ids=batch_ids,
                                          eval_batch=B, logical_batch=logical_batch,
                                          proj_in_channels=self.shape2txt.proj_in_channels,
                                          stage="tex_1024",
                                          cond_or_uncond=cond_or_uncond)
            out = self.shape2txt(x_st, t_eval, c_eval, transformer_options=transformer_options)

        else: # structure
            struct_attn = self.image_attn_mode_structure
            logical_batch_ss = x.shape[0] // len(cond_or_uncond) if cond_or_uncond else x.shape[0]
            struct_cond = context
            if struct_attn != "global":
                struct_cond = _shape_proj_cond(context, struct_attn, proj_feats,
                                               batch_ids=None,
                                               eval_batch=x.shape[0], logical_batch=logical_batch_ss,
                                               proj_in_channels=self.structure_model.proj_in_channels,
                                               stage="ss",
                                               cond_or_uncond=cond_or_uncond)
            out = self.structure_model(x, timestep, struct_cond, transformer_options=transformer_options)

        if is_sparse_mode:
            if mask is not None:
                # Instantly scatter the valid tokens back into a padded rectangular tensor
                padded_out = torch.zeros((B, N, out.feats.shape[-1]), device=x.device, dtype=out.feats.dtype)
                padded_out[mask] = out.feats
                out_tensor = padded_out.transpose(1, 2).unsqueeze(-1)
            else:
                out_tensor = out.feats.view(B, N, -1).transpose(1, 2).unsqueeze(-1)
            return out_tensor
        else:
            out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, 0, 0, 24))

        return out
