"""MiniMax H3 text/vision conditioning: Qwen3-VL-32B (truncated to 50 layers).

The H3 presentation is NOT chat-templated: token ids are raw prompt/label text
(no special tokens) with explicit vision blocks spliced in:

  t2va:   <prompt>
  fl2va:  "<Picture 1>: " <vision block> ["<Picture 2>: " <vision block>] <prompt>
  ref2va: per condition in request order (1-based ordinals per type):
            image -> "<Picture i>: " <vision block>
            audio -> "<Audio j>: "              (audio never enters Qwen)
            video -> "<Video k>: " then per 2-frame temporal block
                     "<T.T seconds>" <vision block(2 frames)>
          then <prompt>

The conditioning is the unnormalized hidden state after LM layer 50 (the
converted checkpoint is truncated there, so this is simply the last-layer
output with no final norm). Vision-pad positions carry adaLN token tag 0
(video modality) in the DiT; text positions carry tag 1 — the tags are
returned alongside the embeddings as "minimax_token_tags".
"""

import math

import torch
import torch.nn.functional as F
import comfy.sd1_clip
from .qwen3vl import Qwen3VL, Qwen3VLSDTokenizer

VISION_START = 151652
VISION_END = 151653
# FL2VA/Ref2VA tokenizer_config extends Qwen with these, ids fixed by the released tokenizer
MINIMAX_EXTRA_TOKENS = {"<d>": 151669, "</d>": 151670, "<|cutoff|>": 151671,
                        "<|lyrics_start|>": 151672, "<|lyrics_end|>": 151673,
                        "<|caption_start|>": 151674, "<|caption_end|>": 151675}
QWEN_IMAGE_MEAN = [0.5, 0.5, 0.5]
QWEN_IMAGE_STD = [0.5, 0.5, 0.5]


def process_video_block(frames, patch_size=16, temporal_patch_size=2, merge_size=2,
                        min_pixels=3136, max_pixels=12845056):
    """[2, H, W, C] frame pair -> (flatten_patches, grid_thw) with grid_t=1.

    Same resize/normalize policy as process_qwen2vl_images, but the two frames
    fill the temporal patch instead of repeating a single frame.
    """
    t, height, width, _ = frames.shape
    imgs = frames.permute(0, 3, 1, 2)
    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    imgs = F.interpolate(imgs, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(QWEN_IMAGE_MEAN, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(QWEN_IMAGE_STD, device=imgs.device).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    patches = imgs.reshape(1, temporal_patch_size, 3, grid_h // merge_size, merge_size,
                           patch_size, grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.stack([torch.tensor([1, grid_h, grid_w], device=frames.device, dtype=torch.long)])
    return flatten, grid_thw


def token_tags_from_embeds_info(seq_len, embeds_info):
    # whole vision block VIDEO(0), including the flanking <|vision_start|>/<|vision_end|> tokens
    # embeds_info spans cover only the expanded embeddings, so widen by one on each side.
    tags = torch.ones(seq_len, dtype=torch.long)
    for e in embeds_info:
        if e.get("type") == "image":
            tags[max(0, e["index"] - 1):e["index"] + e["size"] + 1] = 0
    return tags


class MiniMaxQwen3VL(Qwen3VL):
    model_type = "qwen3vl_32b"

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image" and embed.get("minimax_video_block", False):
            flatten, grid = process_video_block(embed["data"])
            merged, deepstack = self.visual(flatten.to(device, dtype=torch.float32), grid)
            return merged, {"grid": grid, "deepstack": deepstack}
        return super().preprocess_embed(embed, device)

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, embeds_info=[], **kwargs):
        seq = embeds.shape[1] if embeds is not None else input_ids.shape[1]
        self.last_token_tags = token_tags_from_embeds_info(seq, embeds_info)
        return super().forward(input_ids, attention_mask=attention_mask, embeds=embeds,
                               num_tokens=num_tokens, intermediate_output=intermediate_output,
                               final_layer_norm_intermediate=final_layer_norm_intermediate,
                               dtype=dtype, embeds_info=embeds_info, **kwargs)


class MiniMaxH3ClipModel(comfy.sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        super().__init__(device=device, layer="last", layer_idx=None, textmodel_json_config={},
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=MiniMaxQwen3VL, enable_attention_masks=False,
                         return_attention_masks=False, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out = super().encode_token_weights(token_weight_pairs)
        tags = getattr(self.transformer, "last_token_tags", None)
        if tags is not None:
            extra = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
            extra["minimax_token_tags"] = tags
            out = (out[0], out[1], extra)
        return out


class MiniMaxH3TEModel(comfy.sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_32b",
                         clip_model=MiniMaxH3ClipModel, model_options=model_options)

    def memory_estimation_function(self, token_weight_pairs, device=None):
        text = 2.0 * 1024 * 1024 * 1024  # ~2 GB

        constant = 0.14  # ~400 MB per RGB Megapixel
        image_dim = 0

        token_weight_pairs: list[list[tuple[int | dict, float]]] = token_weight_pairs.get("qwen3vl_32b", [])
        for batch in token_weight_pairs:
            for token, weight in batch:
                if not isinstance(token, dict):
                    continue
                if token["type"] == "image":
                    tensor: torch.Tensor = token["data"]
                    image_dim += tensor.numel()  # B, H, W, C

        return image_dim * constant * 1024 + text


class MiniMaxQwenSDTokenizer(Qwen3VLSDTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.add_special_tokens({"additional_special_tokens": list(MINIMAX_EXTRA_TOKENS)})
        self.inv_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}


class MiniMaxH3Tokenizer(comfy.sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = lambda *a, **kw: MiniMaxQwenSDTokenizer(*a, **kw, embedding_size=5120, embedding_key="qwen3vl_32b")
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen3vl_32b", tokenizer=tokenizer)

    @staticmethod
    def _vision_entry(data, video_block=False):
        emb = {"type": "image", "data": data, "original_type": "image"}
        if video_block:
            emb["minimax_video_block"] = True
        return emb

    def tokenize_with_weights(self, text, return_word_ids=False, images=[],
                              minimax_ref_items=None, **kwargs):
        entries = []

        def add_text(s):
            if not s:
                return
            token_batches = self.qwen3vl_32b.tokenize_with_weights(
                s,
                return_word_ids=False,
                disable_weights=True,
            )
            if len(token_batches) != 1:
                raise ValueError("MiniMax H3 text segment exceeds the supported prompt length.")
            entries.extend(token_batches[0])

        def add_vision(data, video_block=False):
            entries.append((VISION_START, 1.0))
            entries.append((self._vision_entry(data, video_block), 1.0))
            entries.append((VISION_END, 1.0))

        if minimax_ref_items:
            counters = {"image": 0, "audio": 0, "video": 0}
            for item in minimax_ref_items:
                kind = item["type"]
                counters[kind] += 1
                if kind == "image":
                    add_text("<Picture %d>: " % counters["image"])
                    add_vision(item["data"])
                elif kind == "audio":
                    add_text("<Audio %d>: " % counters["audio"])
                elif kind == "video":
                    frames = item["data"]  # [T, H, W, C], sampled at 2 fps
                    timestamps = item.get("timestamps")
                    if timestamps is None:
                        timestamps = [i / 2.0 for i in range(frames.shape[0])]
                    if frames.shape[0] % 2 == 1:  # repeat-pad to temporal patch of 2
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        timestamps = list(timestamps) + [timestamps[-1]]
                    add_text("<Video %d>: " % counters["video"])
                    for i in range(0, frames.shape[0], 2):
                        block_ts = (timestamps[i] + timestamps[i + 1]) / 2.0
                        add_text("<%.1f seconds>" % block_ts)
                        add_vision(frames[i:i + 2], video_block=True)
        else:
            for i, img in enumerate(images):
                add_text("<Picture %d>: " % (i + 1))
                add_vision(img)

        add_text(text)
        if len(entries) == 0:
            entries.append((151643, 1.0))
        if return_word_ids:
            entries = [t + (0,) for t in entries]
        return {"qwen3vl_32b": [entries]}

    def untokenize(self, token_weight_pair):
        return self.qwen3vl_32b.untokenize(token_weight_pair)


def te(dtype_llama=None, llama_quantization_metadata=None, **kwargs):
    class MiniMaxH3TEModel_(MiniMaxH3TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return MiniMaxH3TEModel_
