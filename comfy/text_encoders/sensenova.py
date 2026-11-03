"""Tokenizer-only conditioning for SenseNova U1.5.

The language model is part of the diffusion checkpoint, so CLIP only needs to
produce token ids. SenseNova extends the Qwen vocabulary with image-control
tokens; their order is significant because the checkpoint embeds them by id.
"""

import os
import re

import torch
from transformers import Qwen2Tokenizer

from comfy import sd1_clip


SYSTEM_MESSAGE = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)

INTERLEAVE_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and images. "
    "You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a <think></think> block "
    "and place all reasoning inside it. You MUST interleave text with generated images "
    "using tags like <image1>, <image2>. Images can ONLY be generated between <think> and "
    "</think>, and may be referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. "
    "Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. "
    "The answer may include text, images, or both. Match the user's language in both reasoning "
    "and the final answer."
)


def _generation_prompt_parts(text, thinking, mode):
    if mode == "interleave":
        system = INTERLEAVE_SYSTEM_MESSAGE
        assistant = "" if thinking else "<think>\n\n</think>\n\n"
    else:
        system = SYSTEM_MESSAGE
        assistant = "<think>\n" if thinking else "<think>\n\n</think>\n\n<img>"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user",
        "\n" + text,
        f"<|im_end|>\n<|im_start|>assistant\n{assistant}",
    )


def build_generation_prompt(text, thinking=False):
    """Build a SenseNova image-generation chat prompt."""

    return "".join(_generation_prompt_parts(text, thinking, "image"))


def build_unconditional_prompt():
    """Build the negative image-generation prompt."""

    return "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>"


def build_interleave_unconditional_prompt():
    """Build the negative interleave prompt."""

    return "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"


class SenseNovaQwen2Tokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tokenizer = Qwen2Tokenizer.from_pretrained(*args, **kwargs)
        existing_special_tokens = [
            token
            for _, token in sorted(tokenizer.added_tokens_decoder.items())
            if token.special
        ]
        extra_tokens = [
            "<IMG_CONTEXT>",
            "<img>",
            "</img>",
            "<quad>",
            "</quad>",
            "<ref>",
            "</ref>",
            "<box>",
            "</box>",
            "<|action_start|>",
            "<|action_end|>",
            "<|plugin|>",
            "<|interpreter|>",
        ]
        extra_tokens.extend(f"<FAKE_PAD_{index}>" for index in range(254))
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing_special_tokens + extra_tokens}
        )
        return tokenizer


class SenseNovaQwenTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer"
        )
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_size=4096,
            embedding_key="sensenova_u15",
            tokenizer_class=SenseNovaQwen2Tokenizer,
            has_start_token=False,
            has_end_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=1,
            pad_token=151643,
            tokenizer_data=tokenizer_data,
        )


class SenseNovaTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            name="sensenova_u15",
            tokenizer=SenseNovaQwenTokenizer,
        )

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        thinking = kwargs.pop("thinking", False)
        mode = kwargs.pop("mode", "image")
        if text or thinking:
            parts = _generation_prompt_parts(text, thinking, mode)
        else:
            prompt = (
                build_interleave_unconditional_prompt()
                if mode == "interleave"
                else build_unconditional_prompt()
            )
            parts = (prompt,)
        values = []
        word_id = 1
        for index, part in enumerate(parts):
            part = sd1_clip.unescape_important(sd1_clip.escape_important(part))
            # Preserve SDTokenizer's segmentation even without textual-inversion embeddings.
            segments = re.split(r"(?<=\s)(?={})".format(re.escape(self.sensenova_u15.embedding_identifier)), part)
            for segment_index, segment in enumerate(segments):
                word_id += int(segment_index > 0)
                # Only the template may introduce protocol IDs; user markers remain text.
                ids = self.sensenova_u15.tokenizer.encode(
                    segment, add_special_tokens=False, split_special_tokens=index == 1,
                )
                values.extend((token, 1.0, word_id) if return_word_ids else (token, 1.0) for token in ids)
        return {"sensenova_u15": [values]}


class SenseNovaTextEncoder(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.dtypes = {torch.float32}
        self.disable_offload = True
        self.device = torch.device("cpu") if device is None else torch.device(device)

    def encode_token_weights(self, token_weight_pairs):
        pairs = token_weight_pairs["sensenova_u15"][0]
        input_ids = torch.tensor([[int(value[0]) for value in pairs]], dtype=torch.long)
        return (
            input_ids.unsqueeze(-1).to(torch.float32),
            None,
            {"text_input_ids": input_ids},
        )

    def load_sd(self, sd):
        return []

    def get_sd(self):
        return {}

    def reset_clip_options(self):
        pass

    def set_clip_options(self, options):
        pass
