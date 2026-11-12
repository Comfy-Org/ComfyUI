"""WeVisOne-VL: Qwen3-VL answering in coordinate tokens, with an alpha-matte decoder read from its <OBJ> hidden states.

Mask decoder ported from Tencent/WeVisOne-VL (Apache-2.0).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.text_encoders.qwen_vl
from comfy.ldm.rt_detr.rtdetr_v4 import _deformable_attn_v2
from comfy.ldm.sam3.detector import SplitMHA
from .llama import generate_dtype
from .qwen3vl import Qwen3VL, Qwen3VLClipModel, Qwen3VLTEModel, Qwen3VLTokenizer

ADDED_TOKENS = ["<{}>".format(i) for i in range(1001)] + ["<ref>", "</ref>", "<box>", "</box>", "<point>", "</point>", "<polygon>", "</polygon>", "<OBJ>"]
VOCAB_SIZE = 152680
OBJ_TOKEN = 152678

TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
MIN_PIXELS = 16 * 32 * 32
MAX_PIXELS = 2025 * 32 * 32

FEATURE_LAYERS = (5, 11, 17, 23)  # ViT blocks whose pre-merge outputs feed the mask decoder
CANVAS_SIZE = 1024
HIDDEN = 192
MASK_DIM = 64
LEVELS = 4
POINTS = 4
HEADS = 8
FFN = 768
ALPHA_DIM = 48


def to_grid(image):
    # the processor's 32 pixel grid within its pixel limits, so process_qwen2vl_images' own resize is a no-op
    image = image[..., :3]
    return comfy.text_encoders.qwen_vl.resize_8bit(image, *comfy.text_encoders.qwen_vl.smart_resize(image.shape[1], image.shape[2], 32, MIN_PIXELS, MAX_PIXELS))


def group_norm(channels, device, dtype, operations):
    groups = 32
    while channels % groups:
        groups -= 1
    return operations.GroupNorm(groups, channels, device=device, dtype=dtype)


def fourier(values):
    phase = 2.0 * math.pi * values[..., None] * 2.0 ** torch.arange(32, device=values.device, dtype=torch.float32)
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


def position_embedding(h, w, device):
    # DETR sine encoding of the token centres
    y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * (2.0 * math.pi)
    x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * (2.0 * math.pi)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dim_t = 10000.0 ** (2.0 * torch.div(torch.arange(HIDDEN // 2, device=device, dtype=torch.float32), 2, rounding_mode="floor") / (HIDDEN // 2))
    pos_x = xx[..., None] / dim_t
    pos_y = yy[..., None] / dim_t
    pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
    pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
    return torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1).unsqueeze(0)


def resize(value, size):
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


class BoxPosition(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(operations.Linear(4 * 64, HIDDEN, device=device, dtype=dtype), nn.GELU(), operations.Linear(HIDDEN, HIDDEN, device=device, dtype=dtype))

    def forward(self, boxes):
        # x1, y1, log(w), log(h)
        x1, y1, x2, y2 = boxes.unbind(-1)
        width = 0.2 * (x2 - x1).log() + 0.5
        height = 0.2 * (y2 - y1).log() + 0.5
        return self.mlp(torch.cat([fourier(x1), fourier(y1), fourier(width), fourier(height)], dim=-1))


class QueryBuilder(nn.Module):
    def __init__(self, llm_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.content_proj = operations.Linear(llm_dim, HIDDEN, device=device, dtype=dtype)
        self.box_position = BoxPosition(device=device, dtype=dtype, operations=operations)
        self.content_norm = operations.LayerNorm(HIDDEN, device=device, dtype=dtype)
        self.position_norm = operations.LayerNorm(HIDDEN, device=device, dtype=dtype)

    def forward(self, t_seg, boxes):
        return self.content_norm(self.content_proj(t_seg)), self.position_norm(self.box_position(boxes))


class SpatialConvEncoderLayer(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.depthwise = operations.Conv2d(HIDDEN, HIDDEN, 3, padding=1, groups=HIDDEN, bias=False, device=device, dtype=dtype)
        self.pointwise = operations.Conv2d(HIDDEN, HIDDEN, 1, bias=False, device=device, dtype=dtype)
        self.norm1 = operations.LayerNorm(HIDDEN, device=device, dtype=dtype)
        self.linear1 = operations.Linear(HIDDEN, FFN, device=device, dtype=dtype)
        self.linear2 = operations.Linear(FFN, HIDDEN, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(HIDDEN, device=device, dtype=dtype)

    def forward(self, src, pos, shapes):
        mixed = []
        for tokens, (h, w) in zip((src + pos).split([h * w for h, w in shapes], dim=1), shapes):
            feature = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], h, w)
            mixed.append(self.pointwise(F.gelu(self.depthwise(feature))).flatten(2).transpose(1, 2))
        src = self.norm1(src + torch.cat(mixed, dim=1))
        return self.norm2(src + self.linear2(F.gelu(self.linear1(src))))


class SpatialConvEncoder(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([SpatialConvEncoderLayer(device=device, dtype=dtype, operations=operations) for _ in range(3)])

    def forward(self, src, pos, shapes):
        for layer in self.layers:
            src = layer(src, pos, shapes)
        return src


class VisualEncoder(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.input_proj = nn.ModuleList([nn.Sequential(operations.Conv2d(1024, HIDDEN, 1, bias=False, device=device, dtype=dtype), group_norm(HIDDEN, device, dtype, operations)) for _ in range(LEVELS)])
        self.level_embed = nn.Parameter(torch.empty(LEVELS, HIDDEN, device=device, dtype=dtype))
        self.encoder = SpatialConvEncoder(device=device, dtype=dtype, operations=operations)

    def forward(self, features):
        projected = [proj(feature) for proj, feature in zip(self.input_proj, features)]
        shapes = [tuple(feature.shape[-2:]) for feature in projected]
        src = torch.cat([feature.flatten(2).transpose(1, 2) for feature in projected], dim=1)
        level_embed = comfy.ops.cast_to_input(self.level_embed, src)
        pos = torch.cat([(position_embedding(h, w, src.device) + level_embed[level][None, :, None, None]).flatten(2).transpose(1, 2) for level, (h, w) in enumerate(shapes)], dim=1)
        memory = self.encoder(src, pos, shapes)
        maps = [chunk.transpose(1, 2).reshape(chunk.shape[0], chunk.shape[-1], h, w) for chunk, (h, w) in zip(memory.split([h * w for h, w in shapes], dim=1), shapes)]
        return maps, memory, shapes


class MSDeformAttn(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.sampling_offsets = operations.Linear(HIDDEN, HEADS * LEVELS * POINTS * 2, device=device, dtype=dtype)
        self.attention_weights = operations.Linear(HIDDEN, HEADS * LEVELS * POINTS, device=device, dtype=dtype)
        self.value_proj = operations.Linear(HIDDEN, HIDDEN, device=device, dtype=dtype)
        self.output_proj = operations.Linear(HIDDEN, HIDDEN, device=device, dtype=dtype)

    def forward(self, query, references, memory, shapes):
        b, q, _ = query.shape
        value = self.value_proj(memory).view(b, memory.shape[1], HEADS, HIDDEN // HEADS)
        values = [v.permute(0, 2, 3, 1).reshape(b * HEADS, HIDDEN // HEADS, h, w) for v, (h, w) in zip(value.split([h * w for h, w in shapes], dim=1), shapes)]
        offsets = self.sampling_offsets(query).view(b, q, HEADS, LEVELS, POINTS, 2)
        weights = self.attention_weights(query).view(b, q, HEADS, LEVELS * POINTS).softmax(-1)
        # offsets scaled by the reference box size
        locations = references[:, :, None, :, None, :2] + offsets / POINTS * references[:, :, None, :, None, 2:] * 0.5
        return self.output_proj(_deformable_attn_v2(values, shapes, locations.flatten(3, 4), weights, [POINTS] * LEVELS))


class DecoderLayer(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = SplitMHA(HIDDEN, HEADS, device=device, dtype=dtype, operations=operations)
        self.cross_attn = MSDeformAttn(device=device, dtype=dtype, operations=operations)
        self.norm1 = operations.LayerNorm(HIDDEN, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(HIDDEN, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(HIDDEN, device=device, dtype=dtype)
        self.linear1 = operations.Linear(HIDDEN, FFN, device=device, dtype=dtype)
        self.linear2 = operations.Linear(FFN, HIDDEN, device=device, dtype=dtype)

    def forward(self, target, query_pos, references, memory, shapes):
        qk = target + query_pos
        target = self.norm1(target + self.self_attn(qk, qk, target))
        target = self.norm2(target + self.cross_attn(target + query_pos, references, memory, shapes))
        return self.norm3(target + self.linear2(F.gelu(self.linear1(target))))


class Decoder(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(device=device, dtype=dtype, operations=operations) for _ in range(6)])

    def forward(self, target, query_pos, boxes, memory, shapes):
        x1, y1, x2, y2 = boxes.unbind(-1)
        references = torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1)
        references = references[None, :, None].expand(target.shape[0], -1, LEVELS, -1)
        for layer in self.layers:
            target = layer(target, query_pos, references, memory, shapes)
        return target


class FPN(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.lateral = nn.ModuleList([operations.Conv2d(HIDDEN, HIDDEN, 1, bias=False, device=device, dtype=dtype) for _ in range(LEVELS)])
        self.output = nn.ModuleList([nn.Sequential(operations.Conv2d(HIDDEN, HIDDEN, 3, padding=1, bias=False, device=device, dtype=dtype), group_norm(HIDDEN, device, dtype, operations), nn.GELU()) for _ in range(LEVELS - 1)])

    def forward(self, features):
        fused = self.lateral[-1](features[-1])
        for i, level in enumerate(range(len(features) - 2, -1, -1)):
            fused = self.output[i](fused + self.lateral[level](features[level]))
        return fused


class PixelShuffleStage(nn.Module):
    def __init__(self, in_dim, out_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = operations.Conv2d(in_dim, out_dim * 4, 3, padding=1, device=device, dtype=dtype)

    def forward(self, x):
        return F.gelu(F.pixel_shuffle(self.conv(x), 2))


class ResidualPixelShuffleStage(nn.Module):
    def __init__(self, channels, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = operations.Conv2d(channels, channels * 4, 3, padding=1, device=device, dtype=dtype)

    def forward(self, x):
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False) + F.pixel_shuffle(self.conv(x), 2)


class ImageStem(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.net = nn.Sequential(
            operations.Conv2d(3, 128, 3, stride=2, padding=1, bias=False, device=device, dtype=dtype), group_norm(128, device, dtype, operations), nn.GELU(),
            operations.Conv2d(128, MASK_DIM, 3, padding=1, bias=False, device=device, dtype=dtype), group_norm(MASK_DIM, device, dtype, operations), nn.GELU(),
        )

    def forward(self, image):
        return self.net(image)


class PixelFusion(nn.Module):
    # semantic features from the ViT levels plus an RGB stem, at half the canvas resolution
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.fpn = FPN(device=device, dtype=dtype, operations=operations)
        self.upsample = nn.Sequential(PixelShuffleStage(HIDDEN, HIDDEN, device=device, dtype=dtype, operations=operations), PixelShuffleStage(HIDDEN, MASK_DIM, device=device, dtype=dtype, operations=operations))
        self.s2_upsample = ResidualPixelShuffleStage(MASK_DIM, device=device, dtype=dtype, operations=operations)
        self.cnn = ImageStem(device=device, dtype=dtype, operations=operations)
        self.fpn_proj = operations.Conv2d(MASK_DIM, MASK_DIM, 1, bias=False, device=device, dtype=dtype)
        self.cnn_proj = operations.Conv2d(MASK_DIM, MASK_DIM, 1, bias=False, device=device, dtype=dtype)
        self.smooth = nn.Sequential(group_norm(MASK_DIM, device, dtype, operations), nn.GELU(), operations.Conv2d(MASK_DIM, MASK_DIM, 3, padding=1, bias=False, device=device, dtype=dtype))

    def forward(self, image, levels):
        fpn = self.upsample(self.fpn(levels))
        cnn = self.cnn(image)
        dst_h, dst_w = cnn.shape[-2] // 2, cnn.shape[-1] // 2
        pool = (min(fpn.shape[-2], dst_h), min(fpn.shape[-1], dst_w))
        if pool != tuple(fpn.shape[-2:]):
            fpn = F.adaptive_avg_pool2d(fpn, pool)
        fpn = resize(fpn, (dst_h, dst_w))
        return self.s2_upsample(self.smooth(self.fpn_proj(fpn))) + self.cnn_proj(cnn)


class DynamicMaskPredictor(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.kernel = nn.Sequential(operations.Linear(HIDDEN, HIDDEN, device=device, dtype=dtype), nn.GELU(), operations.Linear(HIDDEN, MASK_DIM + 1, device=device, dtype=dtype))

    def forward(self, queries, feature):
        params = self.kernel(queries)
        return torch.einsum("qc,chw->qhw", params[:, :-1], feature[0]) + params[:, -1, None, None]


class AlphaResidualBlock(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.net = nn.Sequential(
            group_norm(ALPHA_DIM, device, dtype, operations), nn.GELU(), operations.Conv2d(ALPHA_DIM, ALPHA_DIM, 3, padding=1, bias=False, device=device, dtype=dtype),
            group_norm(ALPHA_DIM, device, dtype, operations), nn.GELU(), operations.Conv2d(ALPHA_DIM, ALPHA_DIM, 3, padding=1, bias=False, device=device, dtype=dtype),
        )

    def forward(self, x):
        return x + self.net(x)


class AlphaCascade(nn.Module):
    # refines the S/2 coarse logits at S/4, S/2 and full canvas resolution from RGB, semantics and the previous estimate
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        def project(in_channels):
            return nn.Sequential(operations.Conv2d(in_channels, ALPHA_DIM, 3, padding=1, bias=False, device=device, dtype=dtype), group_norm(ALPHA_DIM, device, dtype, operations), nn.GELU())

        def blocks():
            return nn.Sequential(*[AlphaResidualBlock(device=device, dtype=dtype, operations=operations) for _ in range(2)])

        self.semantic = project(MASK_DIM)
        self.rgb_s4 = project(3)
        self.rgb_s2 = project(3)
        self.rgb_s1 = project(3)
        self.coarse_s4 = project(1)
        self.coarse_s2 = project(1)
        self.previous_s2 = project(1)
        self.previous_s1 = project(1)
        self.fuse_s4 = operations.Conv2d(3 * ALPHA_DIM, ALPHA_DIM, 1, bias=False, device=device, dtype=dtype)
        self.fuse_s2 = operations.Conv2d(5 * ALPHA_DIM, ALPHA_DIM, 1, bias=False, device=device, dtype=dtype)
        self.fuse_s1 = operations.Conv2d(4 * ALPHA_DIM, ALPHA_DIM, 1, bias=False, device=device, dtype=dtype)
        self.blocks_s4 = blocks()
        self.blocks_s2 = blocks()
        self.blocks_s1 = blocks()
        self.up_s4_to_s2 = operations.ConvTranspose2d(ALPHA_DIM, ALPHA_DIM, kernel_size=4, stride=2, padding=1, bias=False, device=device, dtype=dtype)
        self.up_s2_to_s1 = operations.ConvTranspose2d(ALPHA_DIM, ALPHA_DIM, kernel_size=4, stride=2, padding=1, bias=False, device=device, dtype=dtype)
        self.query_s4 = operations.Linear(HIDDEN, 2 * ALPHA_DIM, device=device, dtype=dtype)
        self.query_s2 = operations.Linear(HIDDEN, 2 * ALPHA_DIM, device=device, dtype=dtype)
        self.query_s1 = operations.Linear(HIDDEN, 2 * ALPHA_DIM, device=device, dtype=dtype)
        self.output_s4 = operations.Conv2d(ALPHA_DIM, 1, 1, device=device, dtype=dtype)
        self.output_s2 = operations.Conv2d(ALPHA_DIM, 1, 1, device=device, dtype=dtype)
        self.output_s1 = operations.Conv2d(ALPHA_DIM, 1, 1, device=device, dtype=dtype)

    @staticmethod
    def condition(feature, queries, affine):
        scale, bias = affine(queries).chunk(2, dim=-1)
        return feature * (1.0 + 0.1 * scale.tanh()[:, :, None, None]) + 0.1 * bias[:, :, None, None]

    def forward(self, image, pixel_feature, queries, coarse):
        s2 = tuple(coarse.shape[-2:])
        s1 = tuple(image.shape[-2:])
        s4 = (s2[0] // 2, s2[1] // 2)
        semantic_s2 = self.semantic(pixel_feature)
        semantic_s4 = resize(semantic_s2, s4)
        semantic_s1 = resize(semantic_s2, s1)
        rgb_s4 = self.rgb_s4(resize(image, s4))
        rgb_s2 = self.rgb_s2(resize(image, s2))
        rgb_s1 = self.rgb_s1(image)

        logits = []
        for query, coarse_s2 in zip(queries.split(1), coarse.split(1)):  # one object at a time, the full canvas stages are large
            coarse_s4 = resize(coarse_s2[:, None], s4)[:, 0]
            feature_s4 = self.fuse_s4(torch.cat([semantic_s4, rgb_s4, self.coarse_s4(coarse_s4[:, None])], dim=1))
            feature_s4 = self.blocks_s4(self.condition(feature_s4, query, self.query_s4))
            logits_s4 = coarse_s4 + self.output_s4(feature_s4)[:, 0]

            previous_s2 = resize(logits_s4[:, None], s2)
            feature_s2 = self.fuse_s2(torch.cat([self.up_s4_to_s2(feature_s4), semantic_s2, rgb_s2, self.coarse_s2(coarse_s2[:, None]), self.previous_s2(previous_s2.sigmoid())], dim=1))
            feature_s2 = self.blocks_s2(self.condition(feature_s2, query, self.query_s2))
            logits_s2 = coarse_s2 + self.output_s2(feature_s2)[:, 0]

            previous_s1 = resize(logits_s2[:, None], s1)
            feature_s1 = self.fuse_s1(torch.cat([self.up_s2_to_s1(feature_s2), semantic_s1, rgb_s1, self.previous_s1(previous_s1.sigmoid())], dim=1))
            feature_s1 = self.blocks_s1(self.condition(feature_s1, query, self.query_s1))
            logits.append(previous_s1[:, 0] + self.output_s1(feature_s1)[:, 0])
        return torch.cat(logits)


class MaskDecoder(nn.Module):
    def __init__(self, llm_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.visual_encoder = VisualEncoder(device=device, dtype=dtype, operations=operations)
        self.query_builder = QueryBuilder(llm_dim, device=device, dtype=dtype, operations=operations)
        self.transformer = Decoder(device=device, dtype=dtype, operations=operations)
        self.pixel_fusion = PixelFusion(device=device, dtype=dtype, operations=operations)
        self.mask_predictor = DynamicMaskPredictor(device=device, dtype=dtype, operations=operations)
        self.highres_refine = AlphaCascade(device=device, dtype=dtype, operations=operations)

    def forward(self, image, features, t_seg, boxes):
        # image [1, 3, S, S] in 0..1, features the pre-merge ViT maps, t_seg the <OBJ> hidden states, boxes normalized xyxy
        encoded, memory, shapes = self.visual_encoder(features)
        pixel_feature = self.pixel_fusion(image, encoded)
        content, position = self.query_builder(t_seg, boxes)
        # the reference decodes 16 objects at a time, which attend to each other
        queries = torch.cat([self.transformer(content[None, i:i + 16], position[None, i:i + 16], boxes[i:i + 16], memory, shapes)[0] for i in range(0, boxes.shape[0], 16)])
        return self.highres_refine(image, pixel_feature, queries, self.mask_predictor(queries, pixel_feature))


class WeVisOne(Qwen3VL):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__({**config_dict, "vocab_size": VOCAB_SIZE, "lm_head": True}, dtype, device, operations)
        # computes in fp32, the dtype of all its inputs
        self.mask_decoder = MaskDecoder(self.model.config.hidden_size, device=device, dtype=dtype, operations=operations)

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            # the mask pass teacher-forces the same prompt dicts generation saw, so the vision tower runs once per image
            if "features" not in embed:
                image, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(to_grid(embed["data"]), patch_size=16, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
                premerge = dict.fromkeys(FEATURE_LAYERS)
                merged, deepstack = self.visual(image.to(device, dtype=generate_dtype(device)), grid, premerge=premerge)
                embed["features"] = merged, {"grid": grid, "deepstack": deepstack, "premerge": premerge}
            return embed["features"]
        return None, None


class WeVisOneClipModel(Qwen3VLClipModel):
    model_class = WeVisOne
    penalize_prompt = True  # the reference decodes greedily with repetition_penalty 1.05 over prompt and answer

    def decode_masks(self, tokens, ids, image, device, objects):
        # teacher-forces the generated answer ids after the prompt; full canvas mask logits for each object (ordinal of its <OBJ> in the answer, normalized xyxy box)
        tokens = [[t[0] for t in b] + list(ids) for b in tokens]
        embeds, _, _, embeds_info = self.process_tokens(tokens, device)
        _, hidden = self.transformer(None, embeds=embeds.to(generate_dtype(device)), intermediate_output=-1, final_layer_norm_intermediate=False, embeds_info=embeds_info)
        positions = [hidden.shape[1] - len(ids) + i for i, t in enumerate(ids) if t == OBJ_TOKEN]

        extra = embeds_info[0]["extra"]
        _, h, w = extra["grid"][0].tolist()
        # undo the processor's 2x2 merge-window token order; the decoder runs in fp32 like the reference's
        features = [f.reshape(h // 2, w // 2, 2, 2, -1).permute(0, 2, 1, 3, 4).reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0).float() for f in extra["premerge"].values()]
        canvas = comfy.text_encoders.qwen_vl.resize_8bit(image[..., :3], CANVAS_SIZE, CANVAS_SIZE).movedim(-1, 1).to(device)
        t_seg = hidden[0, [positions[ordinal] for ordinal, _ in objects]].float()
        return self.transformer.mask_decoder(canvas, features, t_seg, torch.tensor([box for _, box in objects], device=device, dtype=torch.float32))


class WeVisOneTEModel(Qwen3VLTEModel):
    clip_model_class = WeVisOneClipModel

    def decode_masks(self, tokens, ids, image, device, objects):
        return getattr(self, self.clip).decode_masks(tokens[self.clip_name], ids, image, device, objects)


class WeVisOneTokenizer(Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, model_type="qwen3vl_4b"):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type=model_type)
        # the coordinate and tag tokens follow the stock Qwen3 vocab in order, ids 151669-152678
        getattr(self, self.clip).tokenizer.add_tokens(ADDED_TOKENS)
        self.llama_template = TEMPLATE
        self.llama_template_images = TEMPLATE.format("<|vision_start|><|image_pad|><|vision_end|>{}")
