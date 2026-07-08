from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention_for_device

COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush',
]

# ---------------------------------------------------------------------------
# HGNetv2 backbone
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """Conv→BN→ReLU.  padding='same' adds asymmetric zero-pad (stem)."""
    def __init__(self, ic, oc, k=3, s=1, groups=1, use_act=True, device=None, dtype=None, operations=None):
        super().__init__()

        self.conv = operations.Conv2d(ic, oc, k, s, (k - 1) // 2, groups=groups, bias=False, device=device, dtype=dtype)
        self.bn   = nn.BatchNorm2d(oc, device=device, dtype=dtype)
        self.act  = nn.ReLU() if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class LightConvBNAct(nn.Module):
    def __init__(self, ic, oc, k, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv1 = ConvBNAct(ic, oc, 1, use_act=False, device=device, dtype=dtype, operations=operations)
        self.conv2 = ConvBNAct(oc, oc, k, groups=oc, use_act=True, device=device, dtype=dtype, operations=operations)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class _StemBlock(nn.Module):
    def __init__(self, ic, mc, oc, device=None, dtype=None, operations=None):
        super().__init__()
        self.stem1  = ConvBNAct(ic,    mc,    3, 2, device=device, dtype=dtype, operations=operations)
        # stem2a/stem2b use kernel=2, stride=1, no internal padding;
        # padding is applied manually in forward (matching PaddlePaddle original)
        self.stem2a = ConvBNAct(mc,    mc//2, 2, 1, device=device, dtype=dtype, operations=operations)
        self.stem2b = ConvBNAct(mc//2, mc,    2, 1, device=device, dtype=dtype, operations=operations)
        self.stem3  = ConvBNAct(mc*2,  mc,    3, 2, device=device, dtype=dtype, operations=operations)
        self.stem4  = ConvBNAct(mc,    oc,    1, device=device, dtype=dtype, operations=operations)
        self.pool   = nn.MaxPool2d(2, 1, ceil_mode=True)

    def forward(self, x):
        x  = self.stem1(x)
        x  = F.pad(x, (0, 1, 0, 1))   # pad before pool and stem2a
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))  # pad before stem2b
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        return self.stem4(self.stem3(torch.cat([x1, x2], 1)))


class _HG_Block(nn.Module):
    def __init__(self, ic, mc, oc, layer_num, k=3, residual=False, light=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.residual = residual
        if light:
            self.layers = nn.ModuleList(
                [LightConvBNAct(ic if i == 0 else mc, mc, k, device=device, dtype=dtype, operations=operations) for i in range(layer_num)])
        else:
            self.layers = nn.ModuleList(
                [ConvBNAct(ic if i == 0 else mc, mc, k, device=device, dtype=dtype, operations=operations) for i in range(layer_num)])
        total = ic + layer_num * mc

        self.aggregation = nn.Sequential(
            ConvBNAct(total,   oc // 2, 1, device=device, dtype=dtype, operations=operations),
            ConvBNAct(oc // 2, oc,      1, device=device, dtype=dtype, operations=operations))

    def forward(self, x):
        identity = x
        outs = [x]
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        x = self.aggregation(torch.cat(outs, 1))
        return x + identity if self.residual else x


class _HG_Stage(nn.Module):
    # config order: ic, mc, oc, num_blocks, downsample, light, k, layer_num
    def __init__(self, ic, mc, oc, num_blocks, downsample=True, light=False, k=3, layer_num=6, device=None, dtype=None, operations=None):
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(ic, ic, 3, 2, groups=ic, use_act=False, device=device, dtype=dtype, operations=operations)
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.Sequential(*[
            _HG_Block(ic if i == 0 else oc, mc, oc, layer_num,
                      k=k, residual=(i != 0), light=light, device=device, dtype=dtype, operations=operations)
            for i in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class HGNetv2(nn.Module):
    # B5 config: stem=[3,32,64], stages=[ic, mc, oc, blocks, down, light, k, layers]
    _STAGE_CFGS = [[64,  64,  128,  1, False, False, 3, 6],
                   [128, 128, 512,  2, True,  False, 3, 6],
                   [512, 256, 1024, 5, True,  True,  5, 6],
                   [1024,512, 2048, 2, True,  True,  5, 6]]

    def __init__(self, return_idx=(1, 2, 3), device=None, dtype=None, operations=None):
        super().__init__()
        self.stem   = _StemBlock(3, 32, 64, device=device, dtype=dtype, operations=operations)
        self.stages = nn.ModuleList([_HG_Stage(*cfg, device=device, dtype=dtype, operations=operations) for cfg in self._STAGE_CFGS])
        self.return_idx  = list(return_idx)
        self.out_channels = [self._STAGE_CFGS[i][2] for i in return_idx]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.return_idx:
                outs.append(x)
        return outs


# ---------------------------------------------------------------------------
# Encoder — HybridEncoder  (dfine version: RepNCSPELAN4 + SCDown PAN)
# ---------------------------------------------------------------------------

class ConvNormLayer(nn.Module):
    """Conv→act (expects pre-fused BN weights)."""
    def __init__(self, ic, oc, k, s, g=1, padding=None, act=None, device=None, dtype=None, operations=None):
        super().__init__()
        p = (k - 1) // 2 if padding is None else padding
        self.conv = operations.Conv2d(ic, oc, k, s, p, groups=g, bias=True, device=device, dtype=dtype)
        self.act  = nn.SiLU() if act == 'silu' else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class VGGBlock(nn.Module):
    """Rep-VGG block (expects pre-fused weights)."""
    def __init__(self, ic, oc, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = operations.Conv2d(ic, oc, 3, 1, padding=1, bias=True, device=device, dtype=dtype)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x))


class CSPLayer(nn.Module):
    def __init__(self, ic, oc, num_blocks=3, expansion=1.0, act='silu', device=None, dtype=None, operations=None):
        super().__init__()
        h = int(oc * expansion)
        self.conv1 = ConvNormLayer(ic, h, 1, 1, act=act, device=device, dtype=dtype, operations=operations)
        self.conv2 = ConvNormLayer(ic, h, 1, 1, act=act, device=device, dtype=dtype, operations=operations)
        self.bottlenecks = nn.Sequential(*[VGGBlock(h, h, device=device, dtype=dtype, operations=operations) for _ in range(num_blocks)])
        self.conv3 = ConvNormLayer(h, oc, 1, 1, act=act, device=device, dtype=dtype, operations=operations) if h != oc else nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN block — the FPN/PAN block in RTv4's HybridEncoder."""
    def __init__(self, c1, c2, c3, c4, n=3, act='silu', device=None, dtype=None, operations=None):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer(c1, c3, 1, 1, act=act, device=device, dtype=dtype, operations=operations)
        self.cv2 = nn.Sequential(CSPLayer(c3 // 2, c4, n, 1.0, act=act, device=device, dtype=dtype, operations=operations), ConvNormLayer(c4, c4, 3, 1, act=act, device=device, dtype=dtype, operations=operations))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1.0, act=act, device=device, dtype=dtype, operations=operations), ConvNormLayer(c4, c4, 3, 1, act=act, device=device, dtype=dtype, operations=operations))
        self.cv4 = ConvNormLayer(c3 + 2 * c4, c2, 1, 1, act=act, device=device, dtype=dtype, operations=operations)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class SCDown(nn.Module):
    """Separable conv downsampling used in HybridEncoder PAN bottom-up path."""
    def __init__(self, ic, oc, k, s, device=None, dtype=None, operations=None):
        super().__init__()
        self.cv1 = ConvNormLayer(ic, oc, 1, 1, device=device, dtype=dtype, operations=operations)
        self.cv2 = ConvNormLayer(oc, oc, k, s, g=oc, device=device, dtype=dtype, operations=operations)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.q_proj   = operations.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.k_proj   = operations.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.v_proj   = operations.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.out_proj = operations.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    def forward(self, query, key, value, attn_mask=None):
        optimized_attention = optimized_attention_for_device(query.device, False, small_input=True)
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        out = optimized_attention(q, k, v, heads=self.num_heads, mask=attn_mask)
        return self.out_proj(out)


class _TransformerEncoderLayer(nn.Module):
    """Single AIFI encoder layer (pre- or post-norm, GELU by default)."""
    def __init__(self, d_model, nhead, dim_feedforward, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn  = SelfAttention(d_model, nhead, device=device, dtype=dtype, operations=operations)
        self.linear1    = operations.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2    = operations.Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.norm1      = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2      = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, pos_embed=None):
        q = k = src if pos_embed is None else src + pos_embed
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = self.norm1(src + src2)
        src2 = self.linear2(self.activation(self.linear1(src)))
        return self.norm2(src + src2)


class _TransformerEncoder(nn.Module):
    """Thin wrapper so state-dict keys are  encoder.0.layers.N.*"""
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(d_model, nhead, dim_feedforward, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, pos_embed=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, pos_embed=pos_embed)
        return src


class HybridEncoder(nn.Module):
    def __init__(self, in_channels=(512, 1024, 2048), feat_strides=(8, 16, 32), hidden_dim=256, nhead=8, dim_feedforward=2048, use_encoder_idx=(2,), num_encoder_layers=1,
                 pe_temperature=10000, expansion=1.0, depth_mult=1.0, act='silu', eval_spatial_size=(640, 640), device=None, dtype=None, operations=None):
        super().__init__()
        self.in_channels       = list(in_channels)
        self.feat_strides      = list(feat_strides)
        self.hidden_dim        = hidden_dim
        self.use_encoder_idx   = list(use_encoder_idx)
        self.pe_temperature    = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels      = [hidden_dim] * len(in_channels)
        self.out_strides       = list(feat_strides)

        # channel projection (expects pre-fused weights)
        self.input_proj = nn.ModuleList([
            nn.Sequential(OrderedDict([('conv', operations.Conv2d(ch, hidden_dim, 1, bias=True, device=device, dtype=dtype))]))
            for ch in in_channels
        ])

        # AIFI transformer — use _TransformerEncoder so keys are  encoder.0.layers.N.*
        self.encoder = nn.ModuleList([
            _TransformerEncoder(num_encoder_layers, hidden_dim, nhead, dim_feedforward, device=device, dtype=dtype, operations=operations)
            for _ in range(len(use_encoder_idx))
        ])

        nb  = round(3 * depth_mult)
        exp = expansion

        # top-down FPN  (dfine: lateral conv has no act)
        self.lateral_convs = nn.ModuleList(
            [ConvNormLayer(hidden_dim, hidden_dim, 1, 1, device=device, dtype=dtype, operations=operations)
             for _ in range(len(in_channels) - 1)])
        self.fpn_blocks = nn.ModuleList(
            [RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(exp * hidden_dim // 2), nb, act=act, device=device, dtype=dtype, operations=operations)
             for _ in range(len(in_channels) - 1)])

        # bottom-up PAN  (dfine: nn.Sequential(SCDown) — keeps checkpoint key  .0.cv1/.0.cv2)
        self.downsample_convs = nn.ModuleList(
            [nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, device=device, dtype=dtype, operations=operations))
             for _ in range(len(in_channels) - 1)])
        self.pan_blocks = nn.ModuleList(
            [RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(exp * hidden_dim // 2), nb, act=act, device=device, dtype=dtype, operations=operations)
             for _ in range(len(in_channels) - 1)])

        # cache positional embeddings for fixed spatial size
        if eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pe = self._build_pe(eval_spatial_size[1] // stride,
                                    eval_spatial_size[0] // stride,
                                    hidden_dim, pe_temperature)
                setattr(self, f'pos_embed{idx}', pe)

    @staticmethod
    def _build_pe(w, h, dim=256, temp=10000.):
        assert dim % 4 == 0
        gw = torch.arange(w, dtype=torch.float32)
        gh = torch.arange(h, dtype=torch.float32)
        gw, gh = torch.meshgrid(gw, gh, indexing='ij')
        pdim  = dim // 4
        omega = 1. / (temp ** (torch.arange(pdim, dtype=torch.float32) / pdim))
        ow = gw.flatten()[:, None] @ omega[None]
        oh = gh.flatten()[:, None] @ omega[None]
        return torch.cat([ow.sin(), ow.cos(), oh.sin(), oh.cos()], 1)[None]

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]

        for i, enc_idx in enumerate(self.use_encoder_idx):
            h, w = proj[enc_idx].shape[2:]
            src  = proj[enc_idx].flatten(2).permute(0, 2, 1)
            pe = getattr(self, f'pos_embed{enc_idx}').to(device=src.device, dtype=src.dtype)
            for layer in self.encoder[i].layers:
                src = layer(src, pos_embed=pe)
            proj[enc_idx] = src.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        n = len(self.in_channels)
        inner = [proj[-1]]
        for k in range(n - 1, 0, -1):
            j = n - 1 - k
            top = self.lateral_convs[j](inner[0])
            inner[0] = top
            up = F.interpolate(top, scale_factor=2., mode='nearest')
            inner.insert(0, self.fpn_blocks[j](torch.cat([up, proj[k - 1]], 1)))

        outs = [inner[0]]
        for k in range(n - 1):
            outs.append(self.pan_blocks[k](
                torch.cat([self.downsample_convs[k](outs[-1]), inner[k + 1]], 1)))
        return outs


# ---------------------------------------------------------------------------
# Decoder — DFINETransformer
# ---------------------------------------------------------------------------

def _deformable_attn_v2(value: list, spatial_shapes, sampling_locations: torch.Tensor, attention_weights: torch.Tensor, num_points_list: List[int]) -> torch.Tensor:
    """
    value            : list of per-level tensors  [bs*n_head, c, h_l, w_l]
    sampling_locations: [bs, Lq, n_head, sum(pts), 2]  in [0,1]
    attention_weights : [bs, Lq, n_head, sum(pts)]
    """
    _, c = value[0].shape[:2]      # bs*n_head, c
    _, Lq, n_head, _, _ = sampling_locations.shape
    bs = sampling_locations.shape[0]
    n_h = n_head

    grids = (2 * sampling_locations - 1)          # [bs, Lq, n_head, sum_pts, 2]
    grids = grids.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [bs*n_head, Lq, sum_pts, 2]
    grids_per_lvl = grids.split(num_points_list, dim=2)  # list of [bs*n_head, Lq, pts_l, 2]

    sampled = []
    for lvl, (h, w) in enumerate(spatial_shapes):
        val_l = value[lvl].reshape(bs * n_h, c, h, w)
        sv = F.grid_sample(val_l, grids_per_lvl[lvl], mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled.append(sv) # sv: [bs*n_head, c, Lq, pts_l]

    attn = attention_weights.permute(0, 2, 1, 3)  # [bs, n_head, Lq, sum_pts]
    attn = attn.flatten(0, 1).unsqueeze(1)         # [bs*n_head, 1, Lq, sum_pts]
    out  = (torch.cat(sampled, -1) * attn).sum(-1) # [bs*n_head, c, Lq]
    out  = out.reshape(bs, n_h * c, Lq)
    return out.permute(0, 2, 1)                    # [bs, Lq, hidden]


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4, offset_scale=0.5, device=None, dtype=None, operations=None):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim  = embed_dim // num_heads
        pts = num_points if isinstance(num_points, list) else [num_points] * num_levels
        self.num_points_list = pts
        self.offset_scale    = offset_scale
        total = num_heads * sum(pts)
        self.register_buffer('num_points_scale', torch.tensor([1. / n for n in pts for _ in range(n)], dtype=torch.float32))
        self.sampling_offsets  = operations.Linear(embed_dim, total * 2, device=device, dtype=dtype)
        self.attention_weights = operations.Linear(embed_dim, total, device=device, dtype=dtype)

    def forward(self, query, ref_pts, value, spatial_shapes):
        bs, Lq = query.shape[:2]
        offsets = self.sampling_offsets(query).reshape(
            bs, Lq, self.num_heads, sum(self.num_points_list), 2)
        attn_w  = F.softmax(
            self.attention_weights(query).reshape(
                bs, Lq, self.num_heads, sum(self.num_points_list)), -1)
        scale   = self.num_points_scale.to(query).unsqueeze(-1)
        offset  = offsets * scale * ref_pts[:, :, None, :, 2:] * self.offset_scale
        locs    = ref_pts[:, :, None, :, :2] + offset  # [bs, Lq, n_head, sum_pts, 2]
        return _deformable_attn_v2(value, spatial_shapes, locs, attn_w, self.num_points_list)


class Gate(nn.Module):
    def __init__(self, d_model, device=None, dtype=None, operations=None):
        super().__init__()
        self.gate = operations.Linear(2 * d_model, 2 * d_model, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, x1, x2):
        g1, g2 = torch.sigmoid(self.gate(torch.cat([x1, x2], -1))).chunk(2, -1)
        return self.norm(g1 * x1 + g2 * x2)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, device=None, dtype=None, operations=None):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(operations.Linear(dims[i], dims[i + 1], device=device, dtype=dtype) for i in range(num_layers))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.SiLU()(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, num_levels=3, num_points=4, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn  = SelfAttention(d_model, nhead, device=device, dtype=dtype, operations=operations)
        self.norm1      = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.cross_attn = MSDeformableAttention(d_model, nhead, num_levels, num_points, device=device, dtype=dtype, operations=operations)
        self.gateway    = Gate(d_model, device=device, dtype=dtype, operations=operations)
        self.linear1    = operations.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.activation = nn.ReLU()
        self.linear2    = operations.Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.norm3      = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, target, ref_pts, value, spatial_shapes, attn_mask=None, query_pos=None):
        q = k = target if query_pos is None else target + query_pos
        t2 = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = self.norm1(target + t2)
        t2 = self.cross_attn(
            target if query_pos is None else target + query_pos,
            ref_pts, value, spatial_shapes)
        target = self.gateway(target, t2)
        t2 = self.linear2(self.activation(self.linear1(target)))
        target = self.norm3((target + t2).clamp(-65504, 65504))
        return target


# ---------------------------------------------------------------------------
# FDR utilities
# ---------------------------------------------------------------------------

def weighting_function(reg_max, up, reg_scale):
    """Non-uniform weighting function W(n) for FDR box regression."""
    ub1 = (abs(up[0]) * abs(reg_scale)).item()
    ub2 = ub1 * 2
    step = (ub1 + 1) ** (2 / (reg_max - 2))
    left  = [-(step ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [ (step ** i) - 1 for i in range(1, reg_max // 2)]
    vals  = [-ub2] + left + [0] + right + [ub2]
    return torch.tensor(vals, dtype=up.dtype, device=up.device)


def distance2bbox(points, distance, reg_scale):
    """Decode edge-distances → cxcywh boxes."""
    rs = abs(reg_scale).to(dtype=points.dtype)
    x1 = points[..., 0] - (0.5 * rs + distance[..., 0]) * (points[..., 2] / rs)
    y1 = points[..., 1] - (0.5 * rs + distance[..., 1]) * (points[..., 3] / rs)
    x2 = points[..., 0] + (0.5 * rs + distance[..., 2]) * (points[..., 2] / rs)
    y2 = points[..., 1] + (0.5 * rs + distance[..., 3]) * (points[..., 3] / rs)
    x0, y0, x1_, y1_ = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return torch.stack([x0, y0, x1_, y1_], -1)


class Integral(nn.Module):
    """Sum Pr(n)·W(n) over the distribution bins."""
    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), 1)
        x = F.linear(x, project.to(device=x.device, dtype=x.dtype)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    """Location Quality Estimator — refines class scores using corner distribution."""
    def __init__(self, k=4, hidden_dim=64, num_layers=2, reg_max=32, device=None, dtype=None, operations=None):
        super().__init__()
        self.k, self.reg_max = k, reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, device=device, dtype=dtype, operations=operations)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.shape
        prob     = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), -1)
        topk, _  = prob.topk(self.k, -1)
        stat     = torch.cat([topk, topk.mean(-1, keepdim=True)], -1)
        return scores + self.reg_conf(stat.reshape(B, L, -1))


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward, num_levels, num_points, num_layers, reg_max, reg_scale, up, eval_idx=-1, device=None, dtype=None, operations=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead      = nhead
        self.eval_idx   = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, num_levels, num_points, device=device, dtype=dtype, operations=operations)
            for _ in range(self.eval_idx + 1)
        ])
        self.lqe_layers = nn.ModuleList([LQE(4, 64, 2, reg_max, device=device, dtype=dtype, operations=operations) for _ in range(self.eval_idx + 1)])
        self.register_buffer('project', weighting_function(reg_max, up, reg_scale))

    def _value_op(self, memory, spatial_shapes):
        """Reshape memory to per-level value tensors for deformable attention."""
        c = self.hidden_dim // self.nhead
        split = [h * w for h, w in spatial_shapes]
        val = memory.reshape(memory.shape[0], memory.shape[1], self.nhead, c) # memory: [bs, sum(h*w), hidden_dim]
        # → [bs, n_head, c, sum_hw]
        val = val.permute(0, 2, 3, 1).flatten(0, 1)  # [bs*n_head, c, sum_hw]
        return val.split(split, dim=-1)  # list of [bs*n_head, c, h_l*w_l]

    def forward(self, target, ref_pts_unact, memory, spatial_shapes, bbox_head, score_head, query_pos_head, pre_bbox_head, integral):
        val_split_flat = self._value_op(memory, spatial_shapes) # pre-split value for deformable attention

        # reshape to [bs*n_head, c, h_l, w_l]
        value = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            v = val_split_flat[lvl]   # [bs*n_head, c, h*w]
            value.append(v.reshape(v.shape[0], v.shape[1], h, w))

        ref_pts  = F.sigmoid(ref_pts_unact)
        output   = target
        output_detach = pred_corners_undetach = 0

        dec_bboxes, dec_logits = [], []

        for i, layer in enumerate(self.layers):
            ref_input    = ref_pts.unsqueeze(2)           # [bs, Lq, 1, 4]
            query_pos    = query_pos_head(ref_pts).clamp(-10, 10)
            output       = layer(output, ref_input, value, spatial_shapes, query_pos=query_pos)

            if i == 0:
                ref_unact = ref_pts.clamp(1e-5, 1 - 1e-5)
                ref_unact = torch.log(ref_unact / (1 - ref_unact))
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + ref_unact)
                ref_pts_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_pts_initial, integral(pred_corners, self.project), self.reg_scale)

            if i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_bboxes.append(inter_ref_bbox)
                dec_logits.append(scores)
                break

            pred_corners_undetach = pred_corners
            ref_pts        = inter_ref_bbox.detach()
            output_detach  = output.detach()

        return torch.stack(dec_bboxes), torch.stack(dec_logits)


class DFINETransformer(nn.Module):
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300, feat_channels=[256, 256, 256], feat_strides=[8, 16, 32],
                 num_levels=3, num_points=[3, 6, 3], nhead=8, num_layers=6, dim_feedforward=1024, eval_idx=-1, eps=1e-2, reg_max=32,
                 reg_scale=8.0, eval_spatial_size=(640, 640), device=None, dtype=None, operations=None):
        super().__init__()
        assert len(feat_strides) == len(feat_channels)
        self.hidden_dim  = hidden_dim
        self.num_queries = num_queries
        self.num_levels  = num_levels
        self.eps         = eps
        self.eval_spatial_size = eval_spatial_size

        self.feat_strides = list(feat_strides)
        for i in range(num_levels - len(feat_strides)):
            self.feat_strides.append(feat_strides[-1] * 2 ** (i + 1))

        # input projection (expects pre-fused weights)
        self.input_proj = nn.ModuleList()
        for ch in feat_channels:
            if ch == hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(nn.Sequential(OrderedDict([
                    ('conv', operations.Conv2d(ch, hidden_dim, 1, bias=True, device=device, dtype=dtype))])))
        in_ch = feat_channels[-1]
        for i in range(num_levels - len(feat_channels)):
            self.input_proj.append(nn.Sequential(OrderedDict([
                ('conv', operations.Conv2d(in_ch if i == 0 else hidden_dim,
                                           hidden_dim, 3, 2, 1, bias=True, device=device, dtype=dtype))])))
            in_ch = hidden_dim

        # FDR parameters (non-trainable placeholders, set from config)
        self.up        = nn.Parameter(torch.tensor([0.5]),      requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        pts = num_points if isinstance(num_points, (list, tuple)) else [num_points] * num_levels
        self.decoder = TransformerDecoder(hidden_dim, nhead, dim_feedforward, num_levels, pts,
                                          num_layers, reg_max, self.reg_scale, self.up, eval_idx, device=device, dtype=dtype, operations=operations)

        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, device=device, dtype=dtype, operations=operations)
        self.enc_output     = nn.Sequential(OrderedDict([
            ('proj', operations.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)),
            ('norm', operations.LayerNorm(hidden_dim, device=device, dtype=dtype))]))
        self.enc_score_head = operations.Linear(hidden_dim, num_classes, device=device, dtype=dtype)
        self.enc_bbox_head  = MLP(hidden_dim, hidden_dim, 4, 3, device=device, dtype=dtype, operations=operations)

        self.eval_idx_ = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [operations.Linear(hidden_dim, num_classes, device=device, dtype=dtype) for _ in range(self.eval_idx_ + 1)])
        self.pre_bbox_head  = MLP(hidden_dim, hidden_dim, 4, 3, device=device, dtype=dtype, operations=operations)
        self.dec_bbox_head  = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), 3, device=device, dtype=dtype, operations=operations)
             for _ in range(self.eval_idx_ + 1)])
        self.integral = Integral(reg_max)

        if eval_spatial_size:
            # Register as buffers so checkpoint values override the freshly-computed defaults
            anchors, valid_mask = self._gen_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

    def _gen_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        if spatial_shapes is None:
            h0, w0 = self.eval_spatial_size
            spatial_shapes = [[int(h0 / s), int(w0 / s)] for s in self.feat_strides]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            gxy = (torch.stack([gx, gy], -1).float() + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh  = torch.ones_like(gxy) * grid_size * (2. ** lvl)
            anchors.append(torch.cat([gxy, wh], -1).reshape(-1, h * w, 4))
        anchors    = torch.cat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) & (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors    = torch.log(anchors / (1 - anchors))
        anchors    = torch.where(valid_mask, anchors, torch.full_like(anchors, float('inf')))
        return anchors, valid_mask

    def _encoder_input(self, feats: List[torch.Tensor]):
        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]
        for i in range(len(feats), self.num_levels):
            proj.append(self.input_proj[i](feats[-1] if i == len(feats) else proj[-1]))
        flat, shapes = [], []
        for f in proj:
            _, _, h, w = f.shape
            flat.append(f.flatten(2).permute(0, 2, 1))
            shapes.append([h, w])
        return torch.cat(flat, 1), shapes

    def _decoder_input(self, memory: torch.Tensor):
        anchors, valid_mask = self.anchors.to(memory), self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        mem      = valid_mask.to(memory) * memory
        out_mem  = self.enc_output(mem)
        logits   = self.enc_score_head(out_mem)
        _, idx   = torch.topk(logits.max(-1).values, self.num_queries, dim=-1)
        idx_e    = idx.unsqueeze(-1)
        topk_mem = out_mem.gather(1, idx_e.expand(-1, -1, out_mem.shape[-1]))
        topk_anc = anchors.gather(1, idx_e.expand(-1, -1, anchors.shape[-1]))
        topk_ref = self.enc_bbox_head(topk_mem) + topk_anc
        return topk_mem.detach(), topk_ref.detach()

    def forward(self, feats: List[torch.Tensor]):
        memory, shapes = self._encoder_input(feats)
        content, ref   = self._decoder_input(memory)
        out_bboxes, out_logits = self.decoder(
            content, ref, memory, shapes,
            self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, self.pre_bbox_head, self.integral)
        return {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RTv4(nn.Module):
    def __init__(self, num_classes=80, num_queries=300, enc_h=256, dec_h=256, enc_ff=2048, dec_ff=1024, feat_strides=[8, 16, 32], device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.operations = operations

        self.backbone = HGNetv2(device=device, dtype=dtype, operations=operations)
        self.encoder  = HybridEncoder(hidden_dim=enc_h, dim_feedforward=enc_ff, device=device, dtype=dtype, operations=operations)
        self.decoder  = DFINETransformer(num_classes=num_classes, hidden_dim=dec_h, num_queries=num_queries,
            feat_channels=[enc_h] * len(feat_strides), feat_strides=feat_strides, dim_feedforward=dec_ff, device=device, dtype=dtype, operations=operations)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.load_device = comfy.model_management.get_torch_device()

    def _forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(self.backbone(x)))

    def postprocess(self, outputs, orig_size: tuple = (640, 640)) -> List[dict]:
        logits = outputs['pred_logits']
        boxes  = torchvision.ops.box_convert(outputs['pred_boxes'], 'cxcywh', 'xyxy')
        boxes  = boxes * torch.tensor(orig_size, device=boxes.device, dtype=boxes.dtype).repeat(1, 2).unsqueeze(1)
        scores = F.sigmoid(logits)
        scores, idx = torch.topk(scores.flatten(1), self.num_queries, dim=-1)
        labels = idx % self.num_classes
        boxes  = boxes.gather(1, (idx // self.num_classes).unsqueeze(-1).expand(-1, -1, 4))
        return [{'labels': lbl, 'boxes': b, 'scores': s} for lbl, b, s in zip(labels, boxes, scores)]

    def forward(self, x: torch.Tensor, orig_size: tuple = (640, 640), **kwargs):
        outputs = self._forward(x.to(device=self.load_device, dtype=self.dtype))
        return self.postprocess(outputs, orig_size)
