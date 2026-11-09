import torch

from comfy import sd1_clip
from .lumina2 import Gemma2BTokenizer, LuminaModel
import comfy.text_encoders.llama


class PixelDiTGemma2_2BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata

        super().__init__(
            device=device, layer=layer, layer_idx=layer_idx,
            textmodel_json_config={}, dtype=dtype,
            special_tokens={"start": 2, "pad": 0},
            layer_norm_hidden_state=False,
            model_class=comfy.text_encoders.llama.Gemma2_2B,
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options,
        )


_PIXELDIT_CHI_PROMPT = (
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions '
    "suitable for image generation. Evaluate the level of detail in the user prompt:\n"
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, "
    "and spatial relationships to create vivid and concrete scenes.\n"
    "- If the prompt is already detailed, refine and enhance the existing details slightly without "
    "overcomplicating.\n"
    "Here are examples of how to transform or refine prompts:\n"
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, "
    "sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring "
    "glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus "
    "passing by towering glass skyscrapers.\n"
    "Please generate only the enhanced description for the prompt below and avoid including any "
    "additional commentary or evaluations:\n"
    "User Prompt: "
)

_PIXELDIT_MAX_LENGTH = 300
_PIXELDIT_CHI_PROMPT_DETECT_PREFIX = 'Given a user prompt, generate an "Enhanced prompt"'


class PixelDiTGemma2Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
                         name="gemma2_2b", tokenizer=Gemma2BTokenizer)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        if not text.strip():
            return super().tokenize_with_weights("", return_word_ids=return_word_ids, disable_weights=True, min_length=_PIXELDIT_MAX_LENGTH)

        chi_token_count = len(self.gemma2_2b.tokenizer(_PIXELDIT_CHI_PROMPT)["input_ids"])
        combined = text if text.startswith(_PIXELDIT_CHI_PROMPT_DETECT_PREFIX) else _PIXELDIT_CHI_PROMPT + text
        max_length_all = chi_token_count + _PIXELDIT_MAX_LENGTH - 2
        out = super().tokenize_with_weights(combined, return_word_ids=return_word_ids,
                                            disable_weights=True, min_length=max_length_all)
        out["gemma2_2b"] = [out["gemma2_2b"][0][:max_length_all]]
        return out

    def untokenize(self, token_weight_pair):
        return self.gemma2_2b.untokenize(token_weight_pair)

    def state_dict(self):
        return self.gemma2_2b.state_dict()


class PixelDiTGemma2TE(LuminaModel):
    # PixelDiT's select_index: keep BOS + last 299 embeddings of the padded sequence.
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="gemma2_2b",
                         clip_model=PixelDiTGemma2_2BModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        result = super().encode_token_weights(token_weight_pairs)
        cond, pooled = result[0], result[1]
        extra = result[2] if len(result) > 2 else None
        if cond.shape[1] > _PIXELDIT_MAX_LENGTH:
            cond = torch.cat([cond[:, :1], cond[:, -(_PIXELDIT_MAX_LENGTH - 1):]], dim=1)
            if extra is not None and "attention_mask" in extra:
                am = extra["attention_mask"]
                extra["attention_mask"] = torch.cat([am[..., :1], am[..., -(_PIXELDIT_MAX_LENGTH - 1):]], dim=-1)
        if extra is not None:
            return cond, pooled, extra
        return cond, pooled


def pixeldit_te(dtype_llama=None, llama_quantization_metadata=None):
    class PixelDiTTE_(PixelDiTGemma2TE):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["llama_quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return PixelDiTTE_
