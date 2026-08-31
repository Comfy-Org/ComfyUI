from transformers import Qwen2Tokenizer
from comfy import sd1_clip
import comfy.text_encoders.llama
import comfy.text_encoders.qwen_vl
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

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, **kwargs):
        skip_template = False
        if text.startswith('<|im_start|>'):
            skip_template = True
        if text.startswith('<|start_header_id|>'):
            skip_template = True
        if prevent_empty_text and text == '':
            text = ' '

        if skip_template:
            llama_text = text
        else:
            if llama_template is None:
                if len(images) > 0:
                    llama_text = self.llama_template_images.format(text)
                else:
                    llama_text = self.llama_template.format(text)
            else:
                llama_text = llama_template.format(text)
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


class Qwen25VLTokenizer(sd1_clip.SD1Tokenizer):
    """Static official Qwen2.5-VL tokenizer/template for generation."""

    def __init__(
        self, embedding_directory=None, tokenizer_data={},
        model_type="qwen2_5_vl_7b",
    ):
        embedding_size = 2048 if model_type == "qwen2_5_vl_3b" else 3584

        class Tokenizer(Qwen25_7BVLITokenizer):
            def __init__(inner_self, embedding_directory=None, tokenizer_data={}):
                tokenizer_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "qwen25_tokenizer",
                )
                sd1_clip.SDTokenizer.__init__(
                    inner_self,
                    tokenizer_path,
                    pad_with_end=False,
                    embedding_directory=embedding_directory,
                    embedding_size=embedding_size,
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

        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            name=model_type,
            tokenizer=Tokenizer,
        )
        self.model_type = model_type

    def tokenize_with_weights(
        self, text, return_word_ids=False, image=None, video=None,
        skip_template=False, **kwargs,
    ):
        descriptors = []
        if skip_template:
            llama_text = text
        else:
            content = ""
            if image is not None:
                content += "<|vision_start|><|image_pad|><|vision_end|>"
                descriptors.append({
                    "type": "image",
                    "data": image,
                    "original_type": "image",
                })
            if video is not None:
                content += "<|vision_start|><|video_pad|><|vision_end|>"
                descriptors.append({
                    "type": "video",
                    "data": video,
                    "original_type": "video",
                })
            content += text
            llama_text = (
                "<|im_start|>system\nYou are a helpful assistant."
                "<|im_end|>\n<|im_start|>user\n"
                + content
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
        tokens = super().tokenize_with_weights(
            llama_text,
            return_word_ids=return_word_ids,
            disable_weights=True,
            **kwargs,
        )
        rows = tokens[self.clip_name]
        descriptor_index = 0
        for row in rows:
            for index, item in enumerate(row):
                if item[0] in {151655, 151656} and descriptor_index < len(descriptors):
                    row[index] = (descriptors[descriptor_index],) + item[1:]
                    descriptor_index += 1
        if descriptor_index != len(descriptors):
            raise ValueError("Qwen2.5-VL template/media placeholder mismatch")
        return tokens


class Qwen25VLClipModel(sd1_clip.SDClipModel):
    def __init__(
        self, device="cpu", layer="last", layer_idx=None, dtype=None,
        attention_mask=True, model_options={}, model_type="qwen2_5_vl_7b",
    ):
        model_class = comfy.text_encoders.llama.make_qwen25_vl_model(model_type)
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config={},
            dtype=dtype,
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=model_class,
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
        embeds, _, _, embeds_info = self.process_tokens(
            tokens_only, self.execution_device)
        position_ids = comfy.text_encoders.qwen_vl.qwen2vl_mrope_position_ids(
            embeds_info, embeds.shape[1], embeds.device)
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
            position_ids=position_ids,
            num_beams=num_beams,
        )


class Qwen25VLTEModel(sd1_clip.SD1ClipModel):
    def __init__(
        self, device="cpu", dtype=None, model_options={},
        model_type="qwen2_5_vl_7b",
    ):
        clip_model = lambda **kwargs: Qwen25VLClipModel(
            **kwargs, model_type=model_type)
        super().__init__(
            device=device,
            dtype=dtype,
            name=model_type,
            clip_model=clip_model,
            model_options=model_options,
        )


def vl_tokenizer(model_type="qwen2_5_vl_7b"):
    class Qwen25VLTokenizer_(Qwen25VLTokenizer):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(
                embedding_directory=embedding_directory,
                tokenizer_data=tokenizer_data,
                model_type=model_type,
            )

    return Qwen25VLTokenizer_


def vl_te(
    dtype_llama=None, llama_quantization_metadata=None,
    model_type="qwen2_5_vl_7b",
):
    class Qwen25VLTEModel_(Qwen25VLTEModel):
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

    return Qwen25VLTEModel_
