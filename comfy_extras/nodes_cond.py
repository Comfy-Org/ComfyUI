import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.text_encoders.qwen_vl
from comfy_api.latest import ComfyExtension, io


def _spatial_fusion_mask(height, width, num_sources, method, block_size, dither_ratio, device, seed=0):
    rows = torch.arange(height).unsqueeze(1)
    columns = torch.arange(width).unsqueeze(0)

    if method == "spatial-checkerboard":
        mask = (rows + columns) % num_sources
    elif method == "spatial-block-interleave":
        mask = (rows // block_size + columns // block_size) % num_sources
    elif method == "spatial-dither-random":
        generator = torch.Generator().manual_seed(seed)
        random = torch.rand((height, width), generator=generator)
        other_sources = 1 + ((rows + columns) % (num_sources - 1))
        mask = torch.where(random < dither_ratio, 0, other_sources)
    else:
        raise ValueError(f"Unsupported visual fusion method: {method}")
    return mask.flatten().to(device)


def _visual_token_span(tokens, cond_length, visual_tokens):
    if len(tokens) != 1:
        raise ValueError("Image fusion requires a compatible multimodal CLIP encoder with one token stream.")

    token_pairs = next(iter(tokens.values()))[0]
    image_positions = [i for i, pair in enumerate(token_pairs) if isinstance(pair[0], dict) and pair[0].get("type") == "image"]
    if len(image_positions) != 1:
        raise ValueError("Image fusion requires exactly one visual token block per encoding pass.")

    image_position = image_positions[0]
    if any(not isinstance(pair[0], (int, float)) for pair in token_pairs[image_position + 1:]):
        raise ValueError("Image fusion does not support embeddings after the image token block.")

    end = cond_length - (len(token_pairs) - image_position - 1)
    start = end - visual_tokens
    if start < 0 or end > cond_length:
        raise ValueError("Could not locate the visual token block in the encoded conditioning.")
    return start, end


def _visual_grid(image):
    height, width = image.shape[1:3]
    height, width = comfy.text_encoders.qwen_vl.qwen2vl_image_size(height, width, patch_size=16, merge_size=2)
    return height // 32, width // 32


def _resize_visual_tokens(visual, source_grid, target_grid):
    if source_grid == target_grid:
        return visual

    batch, _, dimensions = visual.shape
    height, width = source_grid
    target_height, target_width = target_grid
    visual = visual.reshape(batch, height, width, dimensions).permute(0, 3, 1, 2)
    visual = F.interpolate(visual, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return visual.permute(0, 2, 3, 1).reshape(batch, target_height * target_width, dimensions)


def _fuse_conditionings(conditionings, tokens, visual_grids, method, block_size, dither_ratio, seed=0):
    schedule_count = len(conditionings[0])
    if any(len(source) != schedule_count for source in conditionings):
        raise ValueError("All image fusion sources must use the same CLIP schedule.")

    target_grid = visual_grids[0]
    fused = []
    for schedule in range(schedule_count):
        source_conds = [source[schedule][0] for source in conditionings]
        spans = [_visual_token_span(source_tokens, cond.shape[1], height * width) for source_tokens, cond, (height, width) in zip(tokens, source_conds, visual_grids)]
        prefix_length = spans[0][0]
        suffix_length = source_conds[0].shape[1] - spans[0][1]
        if any(start != prefix_length or cond.shape[1] - end != suffix_length for cond, (start, end) in zip(source_conds[1:], spans[1:])):
            raise ValueError("Image fusion sources produced different text token layouts.")

        visuals = []
        for cond, (start, end), grid in zip(source_conds, spans, visual_grids):
            visual = _resize_visual_tokens(cond[:, start:end], grid, target_grid)
            visuals.append(visual.to(dtype=source_conds[0].dtype, device=source_conds[0].device))

        visuals = torch.stack(visuals, dim=2)
        target_height, target_width = target_grid
        mask = _spatial_fusion_mask(target_height, target_width, len(source_conds), method, block_size, dither_ratio, visuals.device, seed)
        blended_visual = torch.take_along_dim(visuals, mask[None, :, None, None], dim=2).squeeze(2)

        start, end = spans[0]
        blended = source_conds[0].clone()
        blended[:, start:end] = blended_visual
        fused.append([blended, conditionings[0][schedule][1].copy()])
    return fused


def _flatten_images(images):
    sources = []
    for name in sorted(images, key=lambda value: int(value.rsplit("_", 1)[-1])):
        image = images[name]
        if image is None:
            continue
        if image.ndim == 3:
            image = image.unsqueeze(0)
        sources.extend(image[i:i + 1].clone() for i in range(image.shape[0]))
    return sources


class CLIPTextEncodeImageFusion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        images = io.Autogrow.TemplateNames(
            io.Image.Input("image"),
            names=[f"image_{i}" for i in range(1, 17)],
            min=2,
        )
        return io.Schema(
            node_id="CLIPTextEncodeImageFusion",
            display_name="CLIP Text Encode (Image Fusion)",
            category="model/conditioning",
            description="Encodes images separately and spatially interleaves their visual conditioning tokens.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Autogrow.Input("images", template=images),
                io.Combo.Input(
                    "fusion_method",
                    options=["spatial-checkerboard", "spatial-block-interleave", "spatial-dither-random"],
                    default="spatial-checkerboard",
                ),
                io.Int.Input("block_size", default=2, min=1, max=8, step=1, advanced=True),
                io.Float.Input(
                    "dither_ratio",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Probability of selecting the first source. Remaining sources are selected with a checkerboard pattern.",
                ),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True, advanced=True, tooltip="Seed for the spatial-dither-random pattern."),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, text, images: io.Autogrow.Type, fusion_method, block_size=2, dither_ratio=0.5, seed=0) -> io.NodeOutput:
        sources = _flatten_images(images)
        if len(sources) < 2:
            raise ValueError("Image fusion requires at least two images.")

        sources = [source[:, :, :, :3] for source in sources]
        visual_grids = [_visual_grid(source) for source in sources]
        tokens = [clip.tokenize(text, images=[source]) for source in sources]
        conditionings = [clip.encode_from_tokens_scheduled(source_tokens) for source_tokens in tokens]
        conditioning = _fuse_conditionings(conditionings, tokens, visual_grids, fusion_method, block_size, dither_ratio, seed)
        return io.NodeOutput(conditioning)


class CLIPTextEncodeControlnet(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPTextEncodeControlnet",
            display_name="CLIP Text Encode (Controlnet)",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
            ],
            outputs=[io.Conditioning.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, conditioning, text) -> io.NodeOutput:
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['cross_attn_controlnet'] = cond
            n[1]['pooled_output_controlnet'] = pooled
            c.append(n)
        return io.NodeOutput(c)

class T5TokenizerOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T5TokenizerOptions",
            display_name="T5 Tokenizer Options",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("min_padding", default=0, min=0, max=10000, step=1),
                io.Int.Input("min_length", default=0, min=0, max=10000, step=1),
            ],
            outputs=[io.Clip.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, min_padding, min_length) -> io.NodeOutput:
        clip = clip.clone()
        for t5_type in ["t5xxl", "pile_t5xl", "t5base", "mt5xl", "umt5xxl"]:
            clip.set_tokenizer_option("{}_min_padding".format(t5_type), min_padding)
            clip.set_tokenizer_option("{}_min_length".format(t5_type), min_length)

        return io.NodeOutput(clip)


class CondExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeImageFusion,
            CLIPTextEncodeControlnet,
            T5TokenizerOptions,
        ]


async def comfy_entrypoint() -> CondExtension:
    return CondExtension()
