from comfy import sd1_clip
from .kandinsky5 import Qwen25_7BVLIModel
from .qwen_image import QwenImageTokenizer, QwenImageTEModel
from comfy.ldm.kandinsky6.core_contract import PROMPT_TEMPLATE, PROMPT_TOKENIZER_MAX_LENGTH


class Kandinsky6Tokenizer(QwenImageTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.llama_template = PROMPT_TEMPLATE
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        out = super().tokenize_with_weights(text, return_word_ids, **kwargs)
        # Canonical K6 admits at most PROMPT_TOKENIZER_MAX_LENGTH caption tokens
        # after the prompt prefix; the shared Qwen tokenizer is unbounded.
        out["qwen25_7b"] = [row[:PROMPT_TOKENIZER_MAX_LENGTH] for row in out["qwen25_7b"]]
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out


class Kandinsky6TEModel(QwenImageTEModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super(QwenImageTEModel, self).__init__(device=device, dtype=dtype, name="qwen25_7b", clip_model=Qwen25_7BVLIModel, model_options=model_options)
        self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        # Qwen state is the pre-final-norm last layer (hidden -1), matching the
        # reference Kandinsky 6 text embedder; CLIP-L supplies the pooled `y`.
        cond, p, extra = super().encode_token_weights(token_weight_pairs, template_end=-1)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs["l"])
        return cond, l_pooled, extra

    def set_clip_options(self, options):
        super().set_clip_options(options)
        self.clip_l.set_clip_options(options)

    def reset_clip_options(self):
        super().reset_clip_options()
        self.clip_l.reset_clip_options()

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        return super().load_sd(sd)


def te(dtype_llama=None, llama_quantization_metadata=None):
    class Kandinsky6TEModel_(Kandinsky6TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["llama_quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Kandinsky6TEModel_
