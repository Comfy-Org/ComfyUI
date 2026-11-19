import re
import numbers
import torch
from comfy import sd1_clip
from comfy.text_encoders.qwen_image import Qwen25_7BVLITokenizer, Qwen25_7BVLIModel
import logging

logger = logging.getLogger(__name__)

QUOTE_PAIRS = [("'", "'"), ('"', '"'), ("\u2018", "\u2019"), ("\u201c", "\u201d")]
QUOTE_PATTERN = "|".join(
    [
        re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2)
        for q1, q2 in QUOTE_PAIRS
    ]
)
WORD_INTERNAL_QUOTE_RE = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")


def split_quotation(prompt):
    matches = WORD_INTERNAL_QUOTE_RE.findall(prompt)
    mapping = []
    for i, word_src in enumerate(set(matches)):
        word_tgt = "longcat_$##$_longcat" * (i + 1)
        prompt = prompt.replace(word_src, word_tgt)
        mapping.append((word_src, word_tgt))

    parts = re.split(f"({QUOTE_PATTERN})", prompt)
    result = []
    for part in parts:
        for word_src, word_tgt in mapping:
            part = part.replace(word_tgt, word_src)
        if not part:
            continue
        is_quoted = bool(re.match(QUOTE_PATTERN, part))
        result.append((part, is_quoted))
    return result


class LongCatImageBaseTokenizer(Qwen25_7BVLITokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = 512

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        parts = split_quotation(text)
        all_tokens = []
        for part_text, is_quoted in parts:
            if is_quoted:
                for char in part_text:
                    ids = self.tokenizer(char, add_special_tokens=False)["input_ids"]
                    all_tokens.extend(ids)
            else:
                ids = self.tokenizer(part_text, add_special_tokens=False)["input_ids"]
                all_tokens.extend(ids)

        if len(all_tokens) > self.max_length:
            all_tokens = all_tokens[: self.max_length]
            logger.warning(f"Truncated prompt to {self.max_length} tokens")

        output = [(t, 1.0) for t in all_tokens]
        # Pad to max length
        self.pad_tokens(output, self.max_length - len(output))
        return [output]


IMAGE_PAD_TOKEN_ID = 151655

class LongCatImageTokenizer(sd1_clip.SD1Tokenizer):
    T2I_PREFIX = "<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n"
    EDIT_PREFIX = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            name="qwen25_7b",
            tokenizer=LongCatImageBaseTokenizer,
        )

    def tokenize_with_weights(self, text, return_word_ids=False, images=None, **kwargs):
        skip_template = False
        if text.startswith("<|im_start|>"):
            skip_template = True
        if text.startswith("<|start_header_id|>"):
            skip_template = True
        if text == "":
            text = " "

        base_tok = getattr(self, "qwen25_7b")
        if skip_template:
            tokens = super().tokenize_with_weights(
                text, return_word_ids=return_word_ids, disable_weights=True, **kwargs
            )
        else:
            has_images = images is not None and len(images) > 0
            template_prefix = self.EDIT_PREFIX if has_images else self.T2I_PREFIX

            prefix_ids = base_tok.tokenizer(
                template_prefix, add_special_tokens=False
            )["input_ids"]
            suffix_ids = base_tok.tokenizer(
                self.SUFFIX, add_special_tokens=False
            )["input_ids"]

            prompt_tokens = base_tok.tokenize_with_weights(
                text, return_word_ids=return_word_ids, **kwargs
            )
            prompt_pairs = prompt_tokens[0]

            prefix_pairs = [(t, 1.0) for t in prefix_ids]
            suffix_pairs = [(t, 1.0) for t in suffix_ids]

            combined = prefix_pairs + prompt_pairs + suffix_pairs

            if has_images:
                embed_count = 0
                for i in range(len(combined)):
                    if combined[i][0] == IMAGE_PAD_TOKEN_ID and embed_count < len(images):
                        combined[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"}, combined[i][1])
                        embed_count += 1

            tokens = {"qwen25_7b": [combined]}

        return tokens


class LongCatImageTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(
            device=device,
            dtype=dtype,
            name="qwen25_7b",
            clip_model=Qwen25_7BVLIModel,
            model_options=model_options,
        )

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

        if template_end == -1:
            template_end = 0

        suffix_start = None
        for i in range(len(tok_pairs) - 1, -1, -1):
            elem = tok_pairs[i][0]
            if not torch.is_tensor(elem) and isinstance(elem, numbers.Integral):
                if elem == 151645:
                    suffix_start = i
                    break

        out = out[:, template_end:]

        if "attention_mask" in extra:
            extra["attention_mask"] = extra["attention_mask"][:, template_end:]
            if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
                extra.pop("attention_mask")

        if suffix_start is not None:
            suffix_len = len(tok_pairs) - suffix_start
            if suffix_len > 0 and out.shape[1] > suffix_len:
                out = out[:, :-suffix_len]
                if "attention_mask" in extra:
                    extra["attention_mask"] = extra["attention_mask"][:, :-suffix_len]
                    if extra["attention_mask"].sum() == torch.numel(
                        extra["attention_mask"]
                    ):
                        extra.pop("attention_mask")

        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class LongCatImageTEModel_(LongCatImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)

    return LongCatImageTEModel_
