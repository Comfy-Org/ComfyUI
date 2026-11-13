from transformers import Qwen2Tokenizer
from comfy import sd1_clip
import comfy.text_encoders.llama
import os
import torch
import numbers

class Qwen25_7BVLITokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=3584, embedding_key='qwen25_7b', tokenizer_class=Qwen2Tokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)


class QwenImageTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen25_7b", tokenizer=Qwen25_7BVLITokenizer)
        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.llama_template_images = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_image_placeholders(self, text, image_count):
        vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
        placeholder_count = text.count("<|image_pad|>")
        if placeholder_count > image_count:
            raise ValueError(
                "Qwen prompt has more image placeholders than supplied images"
            )

        missing = image_count - placeholder_count
        if missing <= 0:
            return text

        image_prompt = vision_block * missing
        if vision_block in text:
            index = text.index(vision_block) + len(vision_block)
        else:
            user_turn = "<|im_start|>user\n"
            if user_turn not in text:
                return f"{image_prompt}{text}"
            index = text.index(user_turn) + len(user_turn)
        return f"{text[:index]}{image_prompt}{text[index:]}"

    def tokenize_with_weights(
        self,
        text,
        return_word_ids=False,
        llama_template=None,
        images=[],
        prevent_empty_text=False,
        thinking=None,
        skip_template=False,
        system_prompt="",
        image=None,
        **kwargs,
    ):
        if kwargs.get("video") is not None or kwargs.get("audio") is not None:
            raise NotImplementedError(
                "Qwen 2.5-VL TextGenerate does not support video or audio inputs"
            )

        if image is not None and not images:
            images = [image[i : i + 1] for i in range(image.shape[0])]

        skip_template = (
            skip_template
            or text.startswith("<|im_start|>")
            or text.startswith("<|start_header_id|>")
        )
        if prevent_empty_text and text == "":
            text = " "

        if skip_template:
            llama_text = self._ensure_image_placeholders(text, len(images))
        elif llama_template is not None:
            llama_text = llama_template.format(text)
            llama_text = self._ensure_image_placeholders(llama_text, len(images))
        elif thinking is not None:
            system = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                if system_prompt
                else ""
            )
            image_prompt = "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
            llama_text = f"{system}<|im_start|>user\n{image_prompt}{text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            if len(images) > 0:
                llama_text = self.llama_template_images.format(text)
                llama_text = self._ensure_image_placeholders(llama_text, len(images))
            else:
                llama_text = self.llama_template.format(text)
        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        key_name = next(iter(tokens))
        embed_count = 0
        qwen_tokens = tokens[key_name]
        for r in qwen_tokens:
            for i in range(len(r)):
                if r[i][0] == 151655:
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1
        return tokens


class Qwen25_7BVLIModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen25_7BVLI, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

    def generate(
        self,
        tokens,
        do_sample=True,
        max_length=256,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        min_p=0.0,
        repetition_penalty=1.0,
        seed=None,
        presence_penalty=0.0,
        mtp=True,
    ):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in batch] for batch in tokens]
        embeds, attention_mask, _, embeds_info = self.process_tokens(
            tokens_only, self.execution_device
        )
        if not torch.all(attention_mask):
            raise ValueError(
                "Qwen TextGenerate does not support padded prompt embeddings"
            )
        position_ids = self.transformer.build_position_ids(embeds, embeds_info)
        return self.transformer.generate(
            embeds,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=42 if seed is None else seed,
            presence_penalty=presence_penalty,
            position_ids=position_ids,
            embeds_info=embeds_info,
        )


class QwenImageTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen25_7b", clip_model=Qwen25_7BVLIModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs, template_end=-1):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        tok_pairs = token_weight_pairs["qwen25_7b"][0]
        count_im_start = 0
        if template_end == -1:
            for i, v in enumerate(tok_pairs):
                elem = v[0]
                if not torch.is_tensor(elem):
                    if isinstance(elem, numbers.Integral):
                        if elem == 151644 and count_im_start < 2:
                            template_end = i
                            count_im_start += 1

            if out.shape[1] > (template_end + 3):
                if tok_pairs[template_end + 1][0] == 872:
                    if tok_pairs[template_end + 2][0] == 198:
                        template_end += 3

        out = out[:, template_end:]

        extra["attention_mask"] = extra["attention_mask"][:, template_end:]
        if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
            extra.pop("attention_mask")  # attention mask is useless if no masked elements

        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class QwenImageTEModel_(QwenImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return QwenImageTEModel_
