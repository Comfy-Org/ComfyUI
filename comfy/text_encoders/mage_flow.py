"""Mage-Flow text encoder: Qwen3-VL-4B, last hidden state (2560-dim).

Mage-Flow conditions on the final hidden state of Qwen3-VL-4B with the leading
system + user-opening template tokens stripped (reference start_idx 34 for t2i,
64 for edit). The t2i template is identical to Qwen-Image's; the edit template
uses the same system prompt as Qwen-Image-Edit with "Image N: " reference
prefixes and no <think> block.
"""

import numbers

import torch

import comfy.text_encoders.qwen3vl
from comfy import sd1_clip

MAGE_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

MAGE_T2I_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
MAGE_EDIT_TEMPLATE = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class MageFlowTokenizer(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_4b")
        self.llama_template = MAGE_T2I_TEMPLATE
        self.llama_template_images = MAGE_EDIT_TEMPLATE

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=True, **kwargs):
        image = kwargs.get("image", None)
        if image is not None and len(images) == 0:
            images = [image[i:i + 1] for i in range(image.shape[0])]
        if llama_template is None:
            if len(images) > 0:
                # Training-time multi-reference body: "Image 1: <ph>Image 2: <ph>...{instruction}"
                prefix = "".join("Image {}: {}".format(j + 1, MAGE_VISION_BLOCK) for j in range(len(images)))
                llama_template = self.llama_template_images.replace("{}", prefix + "{}", 1)
            else:
                llama_template = self.llama_template
        # thinking=True: Mage templates end at "<|im_start|>assistant\n" with no <think> block.
        return super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, images=images, prevent_empty_text=prevent_empty_text, thinking=thinking, **kwargs)


class MageFlowQwen3VLClipModel(comfy.text_encoders.qwen3vl.Qwen3VLClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}, model_type="qwen3vl_4b"):
        super().__init__(device=device, dtype=dtype, attention_mask=attention_mask, model_options=model_options, model_type=model_type)
        # apply the final RMSNorm to the tapped last layer (HF last_hidden_state)
        self.layer_norm_hidden_state = True


class MageFlowTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        clip_model = lambda **kw: MageFlowQwen3VLClipModel(**kw, model_type="qwen3vl_4b")  # noqa: E731
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b", clip_model=clip_model, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs, template_end=-1):
        # Strip the system + user-opening prefix (reference drop_idx: 34 t2i / 64 edit).
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        tok_pairs = token_weight_pairs["qwen3vl_4b"][0]
        count_im_start = 0
        if template_end == -1:
            for i, v in enumerate(tok_pairs):
                elem = v[0]
                if not torch.is_tensor(elem):
                    if isinstance(elem, numbers.Integral):
                        if elem == 151644 and count_im_start < 2:  # <|im_start|>
                            template_end = i
                            count_im_start += 1

            if out.shape[1] > (template_end + 3):
                if tok_pairs[template_end + 1][0] == 872:      # "user"
                    if tok_pairs[template_end + 2][0] == 198:  # "\n"
                        template_end += 3

        out = out[:, template_end:]

        if "attention_mask" in extra:
            extra["attention_mask"] = extra["attention_mask"][:, template_end:]
            if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
                extra.pop("attention_mask")  # attention mask is useless if no masked elements

        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class MageFlowTEModel_(MageFlowTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return MageFlowTEModel_
