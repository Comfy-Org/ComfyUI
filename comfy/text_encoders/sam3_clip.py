import re
from comfy import sd1_clip

SAM3_CLIP_CONFIG = {
    "architectures": ["CLIPTextModel"],
    "hidden_act": "quick_gelu",
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "max_position_embeddings": 32,
    "projection_dim": 512,
    "vocab_size": 49408,
    "layer_norm_eps": 1e-5,
    "eos_token_id": 49407,
}


class SAM3ClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, max_length=32, layer="last", textmodel_json_config=SAM3_CLIP_CONFIG, special_tokens={"start": 49406, "end": 49407, "pad": 0}, return_projected_pooled=False, return_attention_masks=True, enable_attention_masks=True, model_options=model_options)


class SAM3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(max_length=32, pad_with_end=False, pad_token=0, embedding_directory=embedding_directory, embedding_size=1024, embedding_key="sam3_clip", tokenizer_data=tokenizer_data)
        self.disable_weights = True


def _parse_prompts(text):
    """Split comma-separated prompts with optional :N max detections per category"""
    text = text.replace("(", "").replace(")", "")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    result = []
    for part in parts:
        m = re.match(r'^(.+?)\s*:\s*([\d.]+)\s*$', part)
        if m:
            text_part = m.group(1).strip()
            val = m.group(2)
            max_det = max(1, round(float(val)))
            result.append((text_part, max_det))
        else:
            result.append((part, 1))
    return result


class SAM3TokenizerWrapper(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="l", tokenizer=SAM3Tokenizer, name="sam3_clip")

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        parsed = _parse_prompts(text)
        if len(parsed) <= 1 and (not parsed or parsed[0][1] == 1):
            return super().tokenize_with_weights(text, return_word_ids, **kwargs)
        # Tokenize each prompt part separately, store per-part batches and metadata
        inner = getattr(self, self.clip)
        per_prompt = []
        for prompt_text, max_det in parsed:
            batches = inner.tokenize_with_weights(prompt_text, return_word_ids, **kwargs)
            per_prompt.append((batches, max_det))
        # Main output uses first prompt's tokens (for compatibility)
        out = {self.clip_name: per_prompt[0][0], "sam3_per_prompt": per_prompt}
        return out


class SAM3ClipModelWrapper(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(device=device, dtype=dtype, model_options=model_options, clip_name="l", clip_model=SAM3ClipModel, name="sam3_clip")

    def encode_token_weights(self, token_weight_pairs):
        per_prompt = token_weight_pairs.pop("sam3_per_prompt", None)
        if per_prompt is None:
            return super().encode_token_weights(token_weight_pairs)

        # Encode each prompt separately, pack into extra dict
        inner = getattr(self, self.clip)
        multi_cond = []
        first_pooled = None
        for batches, max_det in per_prompt:
            out = inner.encode_token_weights(batches)
            cond, pooled = out[0], out[1]
            extra = out[2] if len(out) > 2 else {}
            if first_pooled is None:
                first_pooled = pooled
            multi_cond.append({
                "cond": cond,
                "attention_mask": extra.get("attention_mask"),
                "max_detections": max_det,
            })

        # Return first prompt as main (for non-SAM3 consumers), all prompts in metadata
        main = multi_cond[0]
        main_extra = {}
        if main["attention_mask"] is not None:
            main_extra["attention_mask"] = main["attention_mask"]
        main_extra["sam3_multi_cond"] = multi_cond
        return (main["cond"], first_pooled, main_extra)
