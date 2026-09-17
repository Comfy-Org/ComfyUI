import os

from transformers import Qwen2Tokenizer

from comfy import sd1_clip
import comfy.text_encoders.llama


_FAMILIES = {
    "qwen3_0_6b": {
        "embedding_size": 1024,
        "model": comfy.text_encoders.llama.Qwen3_06BGeneration,
    },
    "qwen3_4b": {
        "embedding_size": 2560,
        "model": comfy.text_encoders.llama.Qwen3_4BGeneration,
    },
}


class QwenGenerationSDTokenizer(sd1_clip.SDTokenizer):
    def __init__(
        self, embedding_directory=None, tokenizer_data={},
        model_type="qwen3_4b",
    ):
        details = _FAMILIES[model_type]
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_directory=embedding_directory,
            embedding_size=details["embedding_size"],
            embedding_key=model_type,
            tokenizer_class=Qwen2Tokenizer,
            has_start_token=False,
            has_end_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=1,
            pad_token=151643,
            tokenizer_data=tokenizer_data,
        )


class QwenGenerationTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(
        self, embedding_directory=None, tokenizer_data={},
        model_type="qwen3_4b",
    ):
        tokenizer = lambda *args, **kwargs: QwenGenerationSDTokenizer(
            *args, **kwargs, model_type=model_type)
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            name=model_type,
            tokenizer=tokenizer,
        )

    def tokenize_with_weights(
        self, text, return_word_ids=False, skip_template=False,
        thinking=False, **kwargs,
    ):
        if not skip_template:
            text = (
                "<|im_start|>user\n" + text
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            if thinking:
                text += "<think>\n"
        return super().tokenize_with_weights(
            text,
            return_word_ids=return_word_ids,
            disable_weights=True,
            **kwargs,
        )


class QwenGenerationClipModel(sd1_clip.SDClipModel):
    def __init__(
        self, device="cpu", layer="last", layer_idx=None, dtype=None,
        attention_mask=True, model_options={}, model_type="qwen3_4b",
    ):
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config={},
            dtype=dtype,
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=_FAMILIES[model_type]["model"],
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options,
        )

    def generate(
        self, tokens, do_sample, max_length, temperature, top_k, top_p,
        min_p, repetition_penalty, seed, presence_penalty=0.0,
        num_beams=1,
    ):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[item[0] for item in row] for row in tokens]
        embeds = self.process_tokens(tokens_only, self.execution_device)[0]
        return self.transformer.generate(
            embeds,
            do_sample,
            max_length,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            seed,
            presence_penalty=presence_penalty,
            num_beams=num_beams,
        )


class QwenGenerationTEModel(sd1_clip.SD1ClipModel):
    def __init__(
        self, device="cpu", dtype=None, model_options={},
        model_type="qwen3_4b",
    ):
        clip_model = lambda **kwargs: QwenGenerationClipModel(
            **kwargs, model_type=model_type)
        super().__init__(
            device=device,
            dtype=dtype,
            name=model_type,
            clip_model=clip_model,
            model_options=model_options,
        )


def tokenizer(model_type="qwen3_4b"):
    class QwenGenerationTokenizer_(QwenGenerationTokenizer):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(
                embedding_directory=embedding_directory,
                tokenizer_data=tokenizer_data,
                model_type=model_type,
            )

    return QwenGenerationTokenizer_


def te(
    dtype_llama=None, llama_quantization_metadata=None,
    model_type="qwen3_4b",
):
    class QwenGenerationTEModel_(QwenGenerationTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = (
                    llama_quantization_metadata)
            super().__init__(
                device=device,
                dtype=dtype,
                model_options=model_options,
                model_type=model_type,
            )

    return QwenGenerationTEModel_
