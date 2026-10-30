"""Fibo 2.0 text encoder: Qwen3-VL-4B, 15 hidden layers at unit RMS in six 7680-dim bundles (Perceiver, then each injection block)."""

import json
import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.rmsnorm
import comfy.text_encoders.qwen3vl
from comfy import sd1_clip

VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

# the null caption training used for classifier-free guidance, a whole chat turn so the tokenizer takes it as it is, uncleaned
NULL_CAPTION = TEMPLATE.format('{"short_description":"","objects":[],"background_setting":"","lighting":{},"aesthetics":{},"photographic_characteristics":{},"style_medium":"","context":"","artistic_style":""}')

BUNDLE_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
TAP_LAYERS = sorted({layer for bundle in BUNDLE_LAYERS for layer in bundle})
BUNDLE_INDEX = [TAP_LAYERS.index(layer) for bundle in BUNDLE_LAYERS for layer in bundle]

SCORES = ("aesthetic_score", "preference_score")


def dumps(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def empty(value):
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def clean(value):
    if isinstance(value, dict):
        value = {k: clean(v) for k, v in value.items()}
        return {k: v for k, v in value.items() if not empty(v)}
    if isinstance(value, list):
        return [v for v in map(clean, value) if not empty(v)]
    return value


def parse_prompt(prompt):
    # the caption as training saw it, and the edit instruction; a prompt that isn't a JSON object comes back as it is
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction

    keys = set(caption)
    caption = clean({key: {k: v for k, v in value.items() if k not in SCORES} if isinstance(value, dict) else value
                     for key, value in caption.items() if key not in SCORES})
    # exactly a short_description and a text_render list keeps both, the list even when empty: the image has no text
    if keys == {"short_description", "text_render"}:
        caption = {"short_description": caption.get("short_description") or "", "text_render": caption.get("text_render") or []}
    return caption or None, instruction


def has_glyph(font, notdef, character):
    # a font without the glyph draws its notdef box, the one it draws for U+FFFE, which no font has
    try:
        mask = font.getmask(character)
    except (OSError, ValueError):  # as in training, a font that can't say is taken to have the glyph
        return True
    return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))


def draw_text_image(text, font):
    # Pillow's built-in font, black on white, whitespace collapsed, split every 256 characters (training's non-ASCII fallback fonts aren't shipped)
    line = " ".join(text.split())
    lines = [line[i:i + 256] for i in range(0, len(line), 256)] or [""]
    notdef = font.getmask(chr(0xFFFE))
    if all(ord(c) < 128 or has_glyph(font, notdef, c) for c in set(line)):
        # each line placed by its bounding box, at least as tall as "Ag"
        min_box = font.getbbox("Ag")
        boxes = [font.getbbox(line or " ") for line in lines]
        heights = [max(1, min_box[3] - min_box[1], box[3] - box[1]) for box in boxes]
        width = max([1] + [box[2] - box[0] for box in boxes])
        origins, anchor = [(2 - box[0], -box[1]) for box in boxes], None
    else:
        # a character the font lacks (drawn as its notdef box) puts every line on the baseline, as tall as the font
        ascent, descent = font.getmetrics()
        lines = [line or " " for line in lines]
        heights = [max(1, ascent + descent)] * len(lines)
        width = max(max(1, round(font.getlength(line))) for line in lines)
        origins, anchor = [(2.0, ascent)] * len(lines), "ls"

    image = Image.new("RGB", (width + 4, sum(heights) + 4 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    y = 2
    for line, (x, dy), height in zip(lines, origins, heights):
        draw.text((x, y + dy), line, font=font, fill=(0, 0, 0), anchor=anchor)
        y += height + 4
    return torch.from_numpy(np.array(image)).float().div(255.0).unsqueeze(0)


def resize_8bit(image, height, width):
    # antialiased bicubic rounded to 8 bit, as PIL and Qwen3-VL's processor resize
    image = torch.nn.functional.interpolate(image.movedim(-1, 1), size=(height, width), mode="bicubic", antialias=True)
    return image.mul(255.0).round().clamp(0, 255).div(255.0).movedim(1, -1)


def thumbnail(image, size=768):
    # Qwen3-VL sees each image to edit at most 768 pixels a side, as in training
    scale = size / max(image.shape[1], image.shape[2])
    if scale >= 1:
        return image
    return resize_8bit(image, max(1, round(image.shape[1] * scale)), max(1, round(image.shape[2] * scale)))


def to_qwen_grid(image):
    # the processor's 32 pixel grid, so ComfyUI's own bilinear resize is a no-op
    h, w = image.shape[1], image.shape[2]
    height, width = round(h / 32) * 32, round(w / 32) * 32
    if height * width > 1605632:
        beta = math.sqrt(h * w / 1605632)
        height, width = max(32, math.floor(h / beta / 32) * 32), max(32, math.floor(w / beta / 32) * 32)
    elif height * width < 3136:
        beta = math.sqrt(3136 / (h * w))
        height, width = math.ceil(h * beta / 32) * 32, math.ceil(w * beta / 32) * 32
    return resize_8bit(image, height, width)


class Fibo2Tokenizer(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_4b")
        self.llama_template_images = TEMPLATE  # the images sit inside the JSON, not before it
        self.font = ImageFont.load_default(20)

    def tokenize_with_weights(self, text, return_word_ids=False, images=[], text_images=True, **kwargs):
        # an empty prompt is the null caption, without the images: the negative of a generation or an edit
        if text.strip() == "":
            return super().tokenize_with_weights(NULL_CAPTION, return_word_ids, **kwargs)
        # text_images=False gives the caption a generation starts from: each text entry's marker left empty, no images
        caption, instruction = parse_prompt(text)
        if len(images) > 0 and instruction is None:
            caption, instruction = None, text
        entries = [e for e in caption.get("text_render") or [] if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"]] if isinstance(caption, dict) else []
        for entry in entries:
            entry["image_tokens"] = VISION_MARKER if text_images else ""

        if len(images) > 0:
            edit = {"image": VISION_MARKER} if len(images) == 1 else {f"image_{k}": VISION_MARKER for k in range(1, len(images) + 1)}
            if caption:
                edit["structured_caption"] = caption
            edit["edit_instruction"] = instruction
            text = dumps(edit)
        elif not isinstance(caption, str):
            text = dumps(caption or {})

        images = [thumbnail(image[..., :3]) for image in images]
        if text_images:
            images += [draw_text_image(entry["text"], self.font) for entry in entries]
        return super().tokenize_with_weights(text, return_word_ids, images=[to_qwen_grid(image) for image in images], **kwargs)


class Fibo2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        clip_model = lambda **kw: comfy.text_encoders.qwen3vl.Qwen3VLClipModel(**kw, layer=TAP_LAYERS, layer_idx=None, model_type="qwen3vl_4b")  # noqa: E731
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b", clip_model=clip_model, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)  # (B, 15, seq, 2560) float32
        # bf16 like the reference's text encoder output, which also halves every cond
        out = comfy.rmsnorm.rms_norm(out).to(torch.bfloat16)
        return out[:, BUNDLE_INDEX].permute(0, 2, 1, 3).flatten(2), pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class Fibo2TEModel_(Fibo2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Fibo2TEModel_
