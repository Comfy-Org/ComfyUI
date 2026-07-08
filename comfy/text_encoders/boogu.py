"""Boogu-Image text encoder: full Qwen3-VL-8B, last hidden state (4096-dim).

Boogu uses the final hidden state of Qwen3-VL as the per-token instruction feature
(num_instruction_feature_layers=1, reduce_type=mean -> just the last layer).
The model itself is the standard Qwen3-VL TE, only the chat template differs
(a fixed system prompt and no <think> block).
"""

import comfy.text_encoders.qwen3vl
from comfy import sd1_clip


# System prompts from the reference pipeline (pipeline_boogu.py).
# T2I (non-empty instruction, no image) uses the helpful-assistant prompt
# everything else (the CFG negative / "drop" condition, and any image case) uses the TI2I "describe" prompt.
BOOGU_T2I_SYSTEM = "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows."
BOOGU_DROP_SYSTEM = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."


class BooguTokenizer(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_8b")
        # apply_chat_template without add_generation_prompt
        self.llama_template = "<|im_start|>system\n" + BOOGU_T2I_SYSTEM + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
        self.llama_template_images = "<|im_start|>system\n" + BOOGU_DROP_SYSTEM + "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
        # Reference SYSTEM_PROMPT_DROP: used for the empty negative/uncond instruction.
        self.llama_template_drop = "<|im_start|>system\n" + BOOGU_DROP_SYSTEM + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=True, **kwargs):
        if llama_template is None and len(images) == 0 and text.strip() == "":
            llama_template = self.llama_template_drop
        # Boogu conditions on the no-think template; thinking=True drops the empty <think> block qwen3vl adds by default.
        return super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, images=images, prevent_empty_text=prevent_empty_text, thinking=thinking, **kwargs)


class BooguQwen3VLClipModel(comfy.text_encoders.qwen3vl.Qwen3VLClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}, model_type="qwen3vl_8b"):
        super().__init__(device=device, dtype=dtype, attention_mask=attention_mask, model_options=model_options, model_type=model_type)
        # apply the final RMSNorm to the tapped last layer
        self.layer_norm_hidden_state = True


class BooguTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        clip_model = lambda **kw: BooguQwen3VLClipModel(**kw, model_type="qwen3vl_8b")
        super().__init__(device=device, dtype=dtype, name="qwen3vl_8b", clip_model=clip_model, model_options=model_options)


def te(dtype_llama=None, llama_quantization_metadata=None):
    class BooguTEModel_(BooguTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return BooguTEModel_
