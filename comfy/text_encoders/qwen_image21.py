import numbers

import torch

import comfy.text_encoders.qwen3vl
from comfy import sd1_clip

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
T2I_TEMPLATE = SYSTEM_PROMPT + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class QwenImage21Tokenizer(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_8b")
        self.llama_template = T2I_TEMPLATE

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=True, keep_vision=False, **kwargs):
        image = kwargs.get("image", None)
        if image is not None and len(images) == 0:
            images = [image[i:i + 1] for i in range(image.shape[0])]
        if llama_template is None and len(images) > 0:
            refs = " ".join("<image{}>{}".format(i + 1, VISION_BLOCK) for i in range(len(images)))
            llama_template = T2I_TEMPLATE.replace("{}", refs + "{}", 1)
        out = super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, images=images, prevent_empty_text=prevent_empty_text, thinking=thinking, **kwargs)
        out["keep_vision"] = keep_vision
        return out


class QwenImage21Qwen3VLClipModel(comfy.text_encoders.qwen3vl.Qwen3VLClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, dtype=dtype, attention_mask=attention_mask, model_options=model_options, model_type="qwen3vl_8b")
        # last layer without the final RMSNorm: transformers 4.57 hidden_states[-1], which Qwen's results are tuned to (5.x norms it)
        self.layer_norm_hidden_state = False
        self.image_spans = []

    def process_tokens(self, tokens, device):
        embeds, attention_mask, num_tokens, embeds_info = super().process_tokens(tokens, device)
        self.image_spans = [(e["index"], e["size"]) for e in embeds_info if e["type"] == "image"]
        return embeds, attention_mask, num_tokens, embeds_info


class QwenImage21TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_8b", clip_model=QwenImage21Qwen3VLClipModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        tokens = [t[0] for t in token_weight_pairs["qwen3vl_8b"][0]]
        image_spans = getattr(self, self.clip).image_spans

        # drop the system turn, everything before the second <|im_start|>; positions shift by each image expanded before it
        im_starts, offset, spans = [], 0, iter(image_spans)
        for i, t in enumerate(tokens):
            if isinstance(t, numbers.Integral):
                if t == 151644:
                    im_starts.append(i + offset)
            elif isinstance(t, dict) and t.get("type") == "image":  # a textual embedding is a bare tensor and has no span
                offset += next(spans, (0, 1))[1] - 1
        keep = torch.ones(out.shape[1], dtype=torch.bool)
        keep[:im_starts[1] if len(im_starts) > 1 else 0] = False

        # vision tokens are replaced by reference latents in the DiT: drop them and record where each image goes
        # with no latents coming (no vae) they stay, and the image conditions through the text encoder alone
        slots = []
        if not token_weight_pairs.get("keep_vision", False):
            for start, size in image_spans:
                keep[start:start + size] = False
                slots.append(int(keep[:start].sum()))

        out = out[:, keep.to(out.device)]
        extra["attention_mask"] = extra["attention_mask"][:, keep.to(extra["attention_mask"].device)]
        if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
            extra.pop("attention_mask")
        if len(slots) > 0:
            extra["image_slots"] = slots
        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class QwenImage21TEModel_(QwenImage21TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return QwenImage21TEModel_
