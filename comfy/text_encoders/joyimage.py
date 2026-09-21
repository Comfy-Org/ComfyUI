import torch

from comfy import sd1_clip
import comfy.text_encoders.qwen_vl
from comfy.text_encoders.qwen3vl import Qwen3VL, Qwen3VLTokenizer

JOYIMAGE_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
JOYIMAGE_TEMPLATE_TEXT = (
    "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
JOYIMAGE_TEMPLATE_IMAGE = (
    "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    f"<|im_start|>user\n{JOYIMAGE_VISION_BLOCK}{{}}<|im_end|>\n<|im_start|>assistant\n"
)
# The DiT was trained without the leading system-prompt tokens.
JOYIMAGE_DROP_IDX = 34
PAD_TOKEN = 151643


class Qwen3VL8B_JoyImage(Qwen3VL):
    model_type = "qwen3vl_8b"

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(
                embed["data"], min_pixels=65536, max_pixels=16777216, patch_size=16,
                image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
                interpolation="bicubic",
            )
            merged, deepstack = self.visual(image.to(device, dtype=torch.float32), grid)
            return merged, {"grid": grid, "deepstack": deepstack}
        return None, None


class JoyImageTokenizer(Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
            model_type="qwen3vl_8b",
        )
        self.llama_template = JOYIMAGE_TEMPLATE_TEXT
        self.llama_template_images = JOYIMAGE_TEMPLATE_IMAGE

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=None, **kwargs):
        kwargs.pop("thinking", None)
        return super().tokenize_with_weights(
            text, return_word_ids=return_word_ids, llama_template=llama_template,
            images=images or [], thinking=True, **kwargs,
        )


class _JoyImageClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-1, dtype=None,
                 attention_mask=True, model_options={}):
        super().__init__(
            device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={},
            # JoyImage conditions on the pre-final-norm output of the last decoder layer.
            dtype=dtype, special_tokens={"pad": PAD_TOKEN}, layer_norm_hidden_state=False,
            model_class=Qwen3VL8B_JoyImage, enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask, model_options=model_options,
        )


class JoyImageTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(
            device=device, dtype=dtype, name="qwen3vl_8b",
            clip_model=_JoyImageClipModel, model_options=model_options,
        )

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        if out.shape[1] <= JOYIMAGE_DROP_IDX:
            raise ValueError(
                f"JoyImageTEModel: encoded sequence length {out.shape[1]} is shorter "
                f"than drop_idx={JOYIMAGE_DROP_IDX}; the prompt did not include the "
                f"template prefix."
            )
        out = out[:, JOYIMAGE_DROP_IDX:]
        if "attention_mask" in extra:
            extra["attention_mask"] = extra["attention_mask"][:, JOYIMAGE_DROP_IDX:]
        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class JoyImageTEModel_(JoyImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return JoyImageTEModel_
