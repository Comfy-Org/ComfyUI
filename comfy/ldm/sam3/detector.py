# SAM3 detector: transformer encoder-decoder, segmentation head, geometry encoder, scoring.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.sam3.tracker import SAM3Tracker, SAM31Tracker
from comfy.ldm.sam3.sam import SAM3VisionBackbone  # noqa: used in __init__
from comfy.ldm.sam3.sam import MLP, PositionEmbeddingSine

TRACKER_CLASSES = {"SAM3": SAM3Tracker, "SAM31": SAM31Tracker}
from comfy.ops import cast_to_input


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def gen_sineembed_for_position(pos_tensor, num_feats=256):
    """Per-coordinate sinusoidal embedding: (..., N) -> (..., N * num_feats)."""
    assert num_feats % 2 == 0
    hdim = num_feats // 2
    freqs = 10000.0 ** (2 * (torch.arange(hdim, dtype=torch.float32, device=pos_tensor.device) // 2) / hdim)
    embeds = []
    for c in range(pos_tensor.shape[-1]):
        raw = (pos_tensor[..., c].float() * 2 * math.pi).unsqueeze(-1) / freqs
        embeds.append(torch.stack([raw[..., 0::2].sin(), raw[..., 1::2].cos()], dim=-1).flatten(-2))
    return torch.cat(embeds, dim=-1).to(pos_tensor.dtype)


class SplitMHA(nn.Module):
    """Multi-head attention with separate Q/K/V projections (split from fused in_proj_weight)."""
    def __init__(self, d_model, num_heads=8, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, q_input, k_input=None, v_input=None, mask=None):
        q = self.q_proj(q_input)
        if k_input is None:
            k = self.k_proj(q_input)
            v = self.v_proj(q_input)
        else:
            k = self.k_proj(k_input)
            v = self.v_proj(v_input if v_input is not None else k_input)
        if mask is not None and mask.ndim == 2:
            mask = mask[:, None, None, :]  # [B, T] -> [B, 1, 1, T] for SDPA broadcast
        dtype = q.dtype  # manual_cast may produce mixed dtypes
        out = optimized_attention(q, k.to(dtype), v.to(dtype), self.num_heads, mask=mask, low_precision_attention=False)
        return self.out_proj(out)


class MLPWithNorm(nn.Module):
    """MLP with residual connection and output LayerNorm."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, residual=True, device=None, dtype=None, operations=None):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([
            operations.Linear(dims[i], dims[i + 1], device=device, dtype=dtype)
            for i in range(num_layers)
        ])
        self.out_norm = operations.LayerNorm(output_dim, device=device, dtype=dtype)
        self.residual = residual and (input_dim == output_dim)

    def forward(self, x):
        orig = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        if self.residual:
            x = x + orig
        return self.out_norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dim_ff=2048, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = SplitMHA(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.cross_attn_image = SplitMHA(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.linear1 = operations.Linear(d_model, dim_ff, device=device, dtype=dtype)
        self.linear2 = operations.Linear(dim_ff, d_model, device=device, dtype=dtype)
        self.norm1 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, x, pos, text_memory=None, text_mask=None):
        normed = self.norm1(x)
        q_k = normed + pos
        x = x + self.self_attn(q_k, q_k, normed)
        if text_memory is not None:
            normed = self.norm2(x)
            x = x + self.cross_attn_image(normed, text_memory, text_memory, mask=text_mask)
        normed = self.norm3(x)
        x = x + self.linear2(F.relu(self.linear1(normed)))
        return x


class TransformerEncoder(nn.Module):
    """Checkpoint: transformer.encoder.layers.N.*"""
    def __init__(self, d_model=256, num_heads=8, dim_ff=2048, num_layers=6, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dim_ff, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])

    def forward(self, x, pos, text_memory=None, text_mask=None):
        for layer in self.layers:
            x = layer(x, pos, text_memory, text_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dim_ff=2048, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = SplitMHA(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.cross_attn = SplitMHA(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.ca_text = SplitMHA(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.norm1 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.catext_norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.linear1 = operations.Linear(d_model, dim_ff, device=device, dtype=dtype)
        self.linear2 = operations.Linear(dim_ff, d_model, device=device, dtype=dtype)

    def forward(self, x, memory, x_pos, memory_pos, text_memory=None, text_mask=None, cross_attn_bias=None):
        q_k = x + x_pos
        x = self.norm2(x + self.self_attn(q_k, q_k, x))
        if text_memory is not None:
            x = self.catext_norm(x + self.ca_text(x + x_pos, text_memory, text_memory, mask=text_mask))
        x = self.norm1(x + self.cross_attn(x + x_pos, memory + memory_pos, memory, mask=cross_attn_bias))
        x = self.norm3(x + self.linear2(F.relu(self.linear1(x))))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dim_ff=2048, num_layers=6,
                 num_queries=200, device=None, dtype=None, operations=None):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dim_ff, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])
        self.norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.query_embed = operations.Embedding(num_queries, d_model, device=device, dtype=dtype)
        self.reference_points = operations.Embedding(num_queries, 4, device=device, dtype=dtype) # Reference points: Embedding(num_queries, 4) — learned anchor boxes
        self.ref_point_head = MLP(d_model * 2, d_model, d_model, 2, device=device, dtype=dtype, operations=operations) # ref_point_head input: 512 (4 coords * 128 sine features each)
        self.bbox_embed = MLP(d_model, d_model, 4, 3, device=device, dtype=dtype, operations=operations)

        self.boxRPB_embed_x = MLP(2, d_model, num_heads, 2, device=device, dtype=dtype, operations=operations)
        self.boxRPB_embed_y = MLP(2, d_model, num_heads, 2, device=device, dtype=dtype, operations=operations)

        self.presence_token = operations.Embedding(1, d_model, device=device, dtype=dtype)
        self.presence_token_head = MLP(d_model, d_model, 1, 3, device=device, dtype=dtype, operations=operations)
        self.presence_token_out_norm = operations.LayerNorm(d_model, device=device, dtype=dtype)

    @staticmethod
    def _inverse_sigmoid(x):
        return torch.log(x / (1 - x + 1e-6) + 1e-6)

    def _compute_box_rpb(self, ref_points, H, W):
        """Box rotary position bias: (B, Q, 4) cxcywh -> (B, n_heads, Q+1, H*W) bias."""
        boxes_xyxy = box_cxcywh_to_xyxy(ref_points)
        B, Q, _ = boxes_xyxy.shape
        coords_h = torch.arange(H, device=ref_points.device, dtype=torch.float32) / H
        coords_w = torch.arange(W, device=ref_points.device, dtype=torch.float32) / W
        deltas_x = coords_w.view(1, 1, -1, 1) - boxes_xyxy[:, :, None, 0:3:2]
        deltas_y = coords_h.view(1, 1, -1, 1) - boxes_xyxy[:, :, None, 1:4:2]

        log2_8 = float(math.log2(8))
        def log_scale(d):
            return torch.sign(d * 8) * torch.log2(torch.abs(d * 8) + 1.0) / log2_8

        rpb_x = self.boxRPB_embed_x(log_scale(deltas_x).to(ref_points.dtype))
        rpb_y = self.boxRPB_embed_y(log_scale(deltas_y).to(ref_points.dtype))

        bias = (rpb_y.unsqueeze(3) + rpb_x.unsqueeze(2)).flatten(2, 3).permute(0, 3, 1, 2)
        pres_bias = torch.zeros(B, bias.shape[1], 1, bias.shape[3], device=bias.device, dtype=bias.dtype)
        return torch.cat([pres_bias, bias], dim=2)

    def forward(self, memory, memory_pos, text_memory=None, text_mask=None, H=72, W=72):
        B = memory.shape[0]
        tgt = cast_to_input(self.query_embed.weight, memory).unsqueeze(0).expand(B, -1, -1)
        presence_out = cast_to_input(self.presence_token.weight, memory)[None].expand(B, -1, -1)
        ref_points = cast_to_input(self.reference_points.weight, memory).unsqueeze(0).expand(B, -1, -1).sigmoid()

        for layer_idx, layer in enumerate(self.layers):
            query_pos = self.ref_point_head(gen_sineembed_for_position(ref_points, self.d_model))
            tgt_with_pres = torch.cat([presence_out, tgt], dim=1)
            pos_with_pres = torch.cat([torch.zeros_like(presence_out), query_pos], dim=1)
            tgt_with_pres = layer(tgt_with_pres, memory, pos_with_pres, memory_pos,
                                  text_memory, text_mask, self._compute_box_rpb(ref_points, H, W))
            presence_out, tgt = tgt_with_pres[:, :1], tgt_with_pres[:, 1:]
            if layer_idx < len(self.layers) - 1:
                ref_inv = self._inverse_sigmoid(ref_points)
                ref_points = (ref_inv + self.bbox_embed(self.norm(tgt))).sigmoid().detach()

        query_out = self.norm(tgt)
        ref_inv = self._inverse_sigmoid(ref_points)
        boxes = (ref_inv + self.bbox_embed(query_out)).sigmoid()
        presence = self.presence_token_head(self.presence_token_out_norm(presence_out)).squeeze(-1)
        return {"decoder_output": query_out, "pred_boxes": boxes, "presence": presence}


class Transformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dim_ff=2048, enc_layers=6, dec_layers=6,
                 num_queries=200, device=None, dtype=None, operations=None):
        super().__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, dim_ff, enc_layers, device=device, dtype=dtype, operations=operations)
        self.decoder = TransformerDecoder(d_model, num_heads, dim_ff, dec_layers, num_queries, device=device, dtype=dtype, operations=operations)


class GeometryEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=3, roi_size=7, device=None, dtype=None, operations=None):
        super().__init__()
        self.d_model = d_model
        self.roi_size = roi_size
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model, normalize=True)
        self.points_direct_project = operations.Linear(2, d_model, device=device, dtype=dtype)
        self.points_pool_project = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.points_pos_enc_project = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.boxes_direct_project = operations.Linear(4, d_model, device=device, dtype=dtype)
        self.boxes_pool_project = operations.Conv2d(d_model, d_model, kernel_size=roi_size, device=device, dtype=dtype)
        self.boxes_pos_enc_project = operations.Linear(d_model + 2, d_model, device=device, dtype=dtype)
        self.label_embed = operations.Embedding(2, d_model, device=device, dtype=dtype)
        self.cls_embed = operations.Embedding(1, d_model, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.img_pre_norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.encode = nn.ModuleList([
            EncoderLayer(d_model, num_heads, 2048, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])
        self.encode_norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.final_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)

    def _encode_points(self, coords, labels, img_feat_2d):
        """Encode point prompts: direct + pool + pos_enc + label. coords: [B, N, 2] normalized."""
        B, N, _ = coords.shape
        embed = self.points_direct_project(coords)
        # Pool features from backbone at point locations via grid_sample
        grid = (coords * 2 - 1).unsqueeze(2)  # [B, N, 1, 2] in [-1, 1]
        sampled = F.grid_sample(img_feat_2d, grid, align_corners=False)  # [B, C, N, 1]
        embed = embed + self.points_pool_project(sampled.squeeze(-1).permute(0, 2, 1))  # [B, N, C]
        # Positional encoding of coordinates
        x, y = coords[:, :, 0], coords[:, :, 1]  # [B, N]
        pos_x, pos_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
        enc = torch.cat([pos_x, pos_y], dim=-1).view(B, N, -1)
        embed = embed + self.points_pos_enc_project(cast_to_input(enc, embed))
        embed = embed + cast_to_input(self.label_embed(labels.long()), embed)
        return embed

    def _encode_boxes(self, boxes, labels, img_feat_2d):
        """Encode box prompts: direct + pool + pos_enc + label. boxes: [B, N, 4] normalized cxcywh."""
        B, N, _ = boxes.shape
        embed = self.boxes_direct_project(boxes)
        # ROI align from backbone at box regions
        H, W = img_feat_2d.shape[-2:]
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
        boxes_scaled = boxes_xyxy * scale
        sampled = roi_align(img_feat_2d, boxes_scaled.view(-1, 4).split(N), self.roi_size)
        proj = self.boxes_pool_project(sampled).view(B, N, -1)  # Conv2d(roi_size) -> [B*N, C, 1, 1] -> [B, N, C]
        embed = embed + proj
        # Positional encoding of box center + size
        cx, cy, w, h = boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2], boxes[:, :, 3]
        enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
        enc = enc.view(B, N, -1)
        embed = embed + self.boxes_pos_enc_project(cast_to_input(enc, embed))
        embed = embed + cast_to_input(self.label_embed(labels.long()), embed)
        return embed

    def forward(self, points=None, boxes=None, image_features=None):
        """Encode geometry prompts. image_features: [B, HW, C] flattened backbone features."""
        # Prepare 2D image features for pooling
        img_feat_2d = None
        if image_features is not None:
            B = image_features.shape[0]
            HW, C = image_features.shape[1], image_features.shape[2]
            hw = int(math.sqrt(HW))
            img_normed = self.img_pre_norm(image_features)
            img_feat_2d = img_normed.permute(0, 2, 1).view(B, C, hw, hw)

        embeddings = []
        if points is not None:
            coords, labels = points
            embeddings.append(self._encode_points(coords, labels, img_feat_2d))
        if boxes is not None:
            B = boxes.shape[0]
            box_labels = torch.ones(B, boxes.shape[1], dtype=torch.long, device=boxes.device)
            embeddings.append(self._encode_boxes(boxes, box_labels, img_feat_2d))
        if not embeddings:
            return None
        geo = torch.cat(embeddings, dim=1)
        geo = self.norm(geo)
        if image_features is not None:
            for layer in self.encode:
                geo = layer(geo, torch.zeros_like(geo), image_features)
        geo = self.encode_norm(geo)
        return self.final_proj(geo)


class PixelDecoder(nn.Module):
    """Top-down FPN pixel decoder with GroupNorm + ReLU + nearest interpolation."""
    def __init__(self, d_model=256, num_stages=3, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv_layers = nn.ModuleList([operations.Conv2d(d_model, d_model, kernel_size=3, padding=1, device=device, dtype=dtype) for _ in range(num_stages)])
        self.norms = nn.ModuleList([operations.GroupNorm(8, d_model, device=device, dtype=dtype) for _ in range(num_stages)])

    def forward(self, backbone_features):
        prev = backbone_features[-1]
        for i, feat in enumerate(backbone_features[:-1][::-1]):
            prev = F.relu(self.norms[i](self.conv_layers[i](feat + F.interpolate(prev, size=feat.shape[-2:], mode="nearest"))))
        return prev


class MaskPredictor(nn.Module):
    def __init__(self, d_model=256, device=None, dtype=None, operations=None):
        super().__init__()
        self.mask_embed = MLP(d_model, d_model, d_model, 3, device=device, dtype=dtype, operations=operations)

    def forward(self, query_embeddings, pixel_features):
        mask_embed = self.mask_embed(query_embeddings)
        return torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_features)


class SegmentationHead(nn.Module):
    def __init__(self, d_model=256, num_heads=8, device=None, dtype=None, operations=None):
        super().__init__()
        self.d_model = d_model
        self.pixel_decoder = PixelDecoder(d_model, 3, device=device, dtype=dtype, operations=operations)
        self.mask_predictor = MaskPredictor(d_model, device=device, dtype=dtype, operations=operations)
        self.cross_attend_prompt = SplitMHA(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.cross_attn_norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.instance_seg_head = operations.Conv2d(d_model, d_model, kernel_size=1, device=device, dtype=dtype)
        self.semantic_seg_head = operations.Conv2d(d_model, 1, kernel_size=1, device=device, dtype=dtype)

    def forward(self, query_embeddings, backbone_features, encoder_hidden_states=None, prompt=None, prompt_mask=None):
        if encoder_hidden_states is not None and prompt is not None:
            enc_normed = self.cross_attn_norm(encoder_hidden_states)
            enc_cross = self.cross_attend_prompt(enc_normed, prompt, prompt, mask=prompt_mask)
            encoder_hidden_states = enc_cross + encoder_hidden_states

        if encoder_hidden_states is not None:
            B, H, W = encoder_hidden_states.shape[0], backbone_features[-1].shape[-2], backbone_features[-1].shape[-1]
            encoder_visual = encoder_hidden_states[:, :H * W].permute(0, 2, 1).view(B, self.d_model, H, W)
            backbone_features = list(backbone_features)
            backbone_features[-1] = encoder_visual

        pixel_features = self.pixel_decoder(backbone_features)
        instance_features = self.instance_seg_head(pixel_features)
        masks = self.mask_predictor(query_embeddings, instance_features)
        return masks


class DotProductScoring(nn.Module):
    def __init__(self, d_model=256, device=None, dtype=None, operations=None):
        super().__init__()
        self.hs_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.prompt_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.prompt_mlp = MLPWithNorm(d_model, 2048, d_model, 2, device=device, dtype=dtype, operations=operations)
        self.scale = 1.0 / (d_model ** 0.5)

    def forward(self, query_embeddings, prompt_embeddings, prompt_mask=None):
        prompt = self.prompt_mlp(prompt_embeddings)
        if prompt_mask is not None:
            weight = prompt_mask.unsqueeze(-1).to(dtype=prompt.dtype)
            pooled = (prompt * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)
        else:
            pooled = prompt.mean(dim=1)
        hs = self.hs_proj(query_embeddings)
        pp = self.prompt_proj(pooled).unsqueeze(-1).to(hs.dtype)
        scores = torch.matmul(hs, pp)
        return (scores * self.scale).clamp(-12.0, 12.0).squeeze(-1)


class SAM3Detector(nn.Module):
    def __init__(self, d_model=256, embed_dim=1024, num_queries=200, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        image_model = kwargs.pop("image_model", "SAM3")
        for k in ("num_heads", "num_head_channels"):
            kwargs.pop(k, None)
        multiplex = image_model == "SAM31"
        # SAM3: 4 FPN levels, drop last (scalp=1); SAM3.1: 3 levels, use all (scalp=0)
        self.scalp = 0 if multiplex else 1
        self.backbone = nn.ModuleDict({
            "vision_backbone": SAM3VisionBackbone(embed_dim=embed_dim, d_model=d_model, multiplex=multiplex, device=device, dtype=dtype, operations=operations, **kwargs),
            "language_backbone": nn.ModuleDict({"resizer": operations.Linear(embed_dim, d_model, device=device, dtype=dtype)}),
        })
        self.transformer = Transformer(d_model=d_model, num_queries=num_queries, device=device, dtype=dtype, operations=operations)
        self.segmentation_head = SegmentationHead(d_model=d_model, device=device, dtype=dtype, operations=operations)
        self.geometry_encoder = GeometryEncoder(d_model=d_model, device=device, dtype=dtype, operations=operations)
        self.dot_prod_scoring = DotProductScoring(d_model=d_model, device=device, dtype=dtype, operations=operations)

    def _get_backbone_features(self, images):
        """Run backbone and return (detector_features, detector_positions, tracker_features, tracker_positions)."""
        bb = self.backbone["vision_backbone"]
        if bb.multiplex:
            all_f, all_p, tf, tp = bb(images, tracker_mode="propagation")
        else:
            all_f, all_p, tf, tp = bb(images, need_tracker=True)
        return all_f, all_p, tf, tp

    @staticmethod
    def _run_geo_layer(layer, x, memory, memory_pos):
        x = x + layer.self_attn(layer.norm1(x))
        x = x + layer.cross_attn_image(layer.norm2(x), memory + memory_pos, memory)
        x = x + layer.linear2(F.relu(layer.linear1(layer.norm3(x))))
        return x

    def _detect(self, features, positions, text_embeddings=None, text_mask=None,
                points=None, boxes=None):
        """Shared detection: geometry encoding, transformer, scoring, segmentation."""
        B = features[0].shape[0]
        # Scalp for encoder (use top-level feature), but keep all levels for segmentation head
        seg_features = features
        if self.scalp > 0:
            features = features[:-self.scalp]
            positions = positions[:-self.scalp]
        enc_feat, enc_pos = features[-1], positions[-1]
        _, _, H, W = enc_feat.shape
        img_flat = enc_feat.flatten(2).permute(0, 2, 1)
        pos_flat = enc_pos.flatten(2).permute(0, 2, 1)

        has_prompts = text_embeddings is not None or points is not None or boxes is not None
        if has_prompts:
            geo_enc = self.geometry_encoder
            geo_prompts = geo_enc(points=points, boxes=boxes, image_features=img_flat)
            geo_cls = geo_enc.norm(geo_enc.final_proj(cast_to_input(geo_enc.cls_embed.weight, img_flat).view(1, 1, -1).expand(B, -1, -1)))
            for layer in geo_enc.encode:
                geo_cls = self._run_geo_layer(layer, geo_cls, img_flat, pos_flat)
            geo_cls = geo_enc.encode_norm(geo_cls)
            if text_embeddings is not None and text_embeddings.shape[0] != B:
                text_embeddings = text_embeddings.expand(B, -1, -1)
            if text_mask is not None and text_mask.shape[0] != B:
                text_mask = text_mask.expand(B, -1)
            parts = [t for t in [text_embeddings, geo_prompts, geo_cls] if t is not None]
            text_embeddings = torch.cat(parts, dim=1)
            n_new = text_embeddings.shape[1] - (text_mask.shape[1] if text_mask is not None else 0)
            if text_mask is not None:
                text_mask = torch.cat([text_mask, torch.ones(B, n_new, dtype=torch.bool, device=text_mask.device)], dim=1)
            else:
                text_mask = torch.ones(B, text_embeddings.shape[1], dtype=torch.bool, device=text_embeddings.device)

        memory = self.transformer.encoder(img_flat, pos_flat, text_embeddings, text_mask)
        dec_out = self.transformer.decoder(memory, pos_flat, text_embeddings, text_mask, H, W)
        query_out, pred_boxes = dec_out["decoder_output"], dec_out["pred_boxes"]

        if text_embeddings is not None:
            scores = self.dot_prod_scoring(query_out, text_embeddings, text_mask)
        else:
            scores = torch.zeros(B, query_out.shape[1], device=query_out.device)

        masks = self.segmentation_head(query_out, seg_features, encoder_hidden_states=memory, prompt=text_embeddings, prompt_mask=text_mask)
        return box_cxcywh_to_xyxy(pred_boxes), scores, masks, dec_out

    def forward(self, images, text_embeddings=None, text_mask=None, points=None, boxes=None, threshold=0.3, orig_size=None):
        features, positions, _, _ = self._get_backbone_features(images)

        if text_embeddings is not None:
            text_embeddings = self.backbone["language_backbone"]["resizer"](text_embeddings)
            if text_mask is not None:
                text_mask = text_mask.bool()

        boxes_xyxy, scores, masks, dec_out = self._detect(
            features, positions, text_embeddings, text_mask, points, boxes)

        if orig_size is not None:
            oh, ow = orig_size
            boxes_xyxy = boxes_xyxy * torch.tensor([ow, oh, ow, oh], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
            masks = F.interpolate(masks, size=orig_size, mode="bilinear", align_corners=False)

        return {
            "boxes": boxes_xyxy,
            "scores": scores,
            "masks": masks,
            "presence": dec_out.get("presence"),
        }

    def forward_from_trunk(self, trunk_out, text_embeddings, text_mask):
        """Run detection using a pre-computed ViTDet trunk output.

        text_embeddings must already be resized through language_backbone.resizer.
        Returns dict with boxes (normalized xyxy), scores, masks at detector resolution.
        """
        bb = self.backbone["vision_backbone"]
        features = [conv(trunk_out) for conv in bb.convs]
        positions = [cast_to_input(bb.position_encoding(f), f) for f in features]

        if text_mask is not None:
            text_mask = text_mask.bool()

        boxes_xyxy, scores, masks, _ = self._detect(features, positions, text_embeddings, text_mask)
        return {"boxes": boxes_xyxy, "scores": scores, "masks": masks}


class SAM3Model(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        image_model = kwargs.get("image_model", "SAM3")
        tracker_cls = TRACKER_CLASSES[image_model]
        self.detector = SAM3Detector(device=device, dtype=dtype, operations=operations, **kwargs)
        self.tracker = tracker_cls(device=device, dtype=dtype, operations=operations, **kwargs)

    def forward(self, images, **kwargs):
        return self.detector(images, **kwargs)

    def forward_segment(self, images, point_inputs=None, box_inputs=None, mask_inputs=None):
        """Interactive segmentation using SAM decoder with point/box/mask prompts.

        Args:
            images: [B, 3, 1008, 1008] preprocessed images
            point_inputs: {"point_coords": [B, N, 2], "point_labels": [B, N]} in 1008x1008 pixel space
            box_inputs: [B, 2, 2] box corners (top-left, bottom-right) in 1008x1008 pixel space
            mask_inputs: [B, 1, H, W] coarse mask logits to refine
        Returns:
            [B, 1, image_size, image_size] high-res mask logits
        """
        bb = self.detector.backbone["vision_backbone"]
        if bb.multiplex:
            _, _, tracker_features, tracker_positions = bb(images, tracker_mode="interactive")
        else:
            _, _, tracker_features, tracker_positions = bb(images, need_tracker=True)
            if self.detector.scalp > 0:
                tracker_features = tracker_features[:-self.detector.scalp]
                tracker_positions = tracker_positions[:-self.detector.scalp]

        high_res = list(tracker_features[:-1])
        backbone_feat = tracker_features[-1]
        B, C, H, W = backbone_feat.shape
        # Add no-memory embedding (init frame path)
        no_mem = getattr(self.tracker, 'interactivity_no_mem_embed', None)
        if no_mem is None:
            no_mem = getattr(self.tracker, 'no_mem_embed', None)
        if no_mem is not None:
            feat_flat = backbone_feat.flatten(2).permute(0, 2, 1)
            feat_flat = feat_flat + cast_to_input(no_mem, feat_flat)
            backbone_feat = feat_flat.view(B, H, W, C).permute(0, 3, 1, 2)

        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        _, high_res_masks, _, _ = self.tracker._forward_sam_heads(
            backbone_features=backbone_feat,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            box_inputs=box_inputs,
            high_res_features=high_res,
            multimask_output=(0 < num_pts <= 1),
        )
        return high_res_masks

    def forward_video(self, images, initial_masks, pbar=None, text_prompts=None,
                       new_det_thresh=0.5, max_objects=0, detect_interval=1,
                       target_device=None, target_dtype=None):
        """Track video with optional per-frame text-prompted detection."""
        bb = self.detector.backbone["vision_backbone"]

        def backbone_fn(frame, frame_idx=None):
            trunk_out = bb.trunk(frame)
            if bb.multiplex:
                _, _, tf, tp = bb(frame, tracker_mode="propagation", cached_trunk=trunk_out, tracker_only=True)
            else:
                _, _, tf, tp = bb(frame, need_tracker=True, cached_trunk=trunk_out, tracker_only=True)
            return tf, tp, trunk_out

        detect_fn = None
        if text_prompts:
            resizer = self.detector.backbone["language_backbone"]["resizer"]
            resized = [(resizer(emb), m.bool() if m is not None else None) for emb, m in text_prompts]
            def detect_fn(trunk_out):
                all_scores, all_masks = [], []
                for emb, mask in resized:
                    det = self.detector.forward_from_trunk(trunk_out, emb, mask)
                    all_scores.append(det["scores"])
                    all_masks.append(det["masks"])
                return {"scores": torch.cat(all_scores, dim=1), "masks": torch.cat(all_masks, dim=1)}

        if hasattr(self.tracker, 'track_video_with_detection'):
            return self.tracker.track_video_with_detection(
                backbone_fn, images, initial_masks, detect_fn,
                new_det_thresh=new_det_thresh, max_objects=max_objects,
                detect_interval=detect_interval, backbone_obj=bb, pbar=pbar,
                target_device=target_device, target_dtype=target_dtype)
        # SAM3 (non-multiplex) — no detection support, requires initial masks
        if initial_masks is None:
            raise ValueError("SAM3 (non-multiplex) requires initial_mask for video tracking")
        return self.tracker.track_video(backbone_fn, images, initial_masks, pbar=pbar, backbone_obj=bb,
                                         target_device=target_device, target_dtype=target_dtype)
