"""HiDream-O1-Image tokenizer-only text encoder.

The real Qwen3-VL backbone runs inside diffusion_model.* every step, so this
module just tokenizes the prompt into text_input_ids and emits them as
conditioning. Position ids / token_types / vinput_mask depend on target H/W
and are built later in model_base.HiDreamO1.extra_conds.
"""

import os

import torch
from transformers import Qwen2Tokenizer

from comfy import sd1_clip


# Qwen3-VL special tokens
IM_START_ID = 151644
IM_END_ID = 151645
ASSISTANT_ID = 77091
USER_ID = 872
NEWLINE_ID = 198
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
# HiDream-O1-specific tokens
BOI_TOKEN_ID = 151669
BOR_TOKEN_ID = 151670
EOR_TOKEN_ID = 151671
BOT_TOKEN_ID = 151672
TMS_TOKEN_ID = 151673


class HiDreamO1QwenTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer"
        )
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_size=4096,
            embedding_key="hidream_o1",
            tokenizer_class=Qwen2Tokenizer,
            has_start_token=False,
            has_end_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=1,
            pad_token=151643,
            tokenizer_data=tokenizer_data,
        )


class HiDreamO1Tokenizer(sd1_clip.SD1Tokenizer):
    """Wraps prompt in the upstream chat template ending with boi/tms markers.
    Image tokens get spliced in at sample time once target H/W is known.
    """

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            name="hidream_o1",
            tokenizer=HiDreamO1QwenTokenizer,
        )

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        text_tokens_dict = super().tokenize_with_weights(
            text, return_word_ids=return_word_ids, disable_weights=True, **kwargs
        )
        text_tuples = text_tokens_dict["hidream_o1"][0]
        text_tuples = [t for t in text_tuples if int(t[0]) != 151643]  # strip pad

        # <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|boi|><|tms|>
        def tok(tid):
            return (tid, 1.0) if not return_word_ids else (tid, 1.0, 0)

        prefix = [tok(IM_START_ID), tok(USER_ID), tok(NEWLINE_ID)]
        suffix = [
            tok(IM_END_ID), tok(NEWLINE_ID),
            tok(IM_START_ID), tok(ASSISTANT_ID), tok(NEWLINE_ID),
            tok(BOI_TOKEN_ID), tok(TMS_TOKEN_ID),
        ]
        full = prefix + list(text_tuples) + suffix
        return {"hidream_o1": [full]}


class HiDreamO1TE(torch.nn.Module):
    """Passthrough TE: emits int token ids; the Qwen3-VL backbone in diffusion_model does the actual encoding."""

    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.dtypes = {torch.float32}
        self.disable_offload = True # skips dynamic VRAM management for this zero-parameter module
        self.device = torch.device("cpu") if device is None else torch.device(device)

    def encode_token_weights(self, token_weight_pairs):
        tok_pairs = token_weight_pairs["hidream_o1"][0]
        ids = [int(t[0]) for t in tok_pairs]
        input_ids = torch.tensor([ids], dtype=torch.long)
        # Surrogate keeps the cross_attn slot non-empty for CONDITIONING
        # plumbing; the model reads text_input_ids out of `extra` instead.
        cross_attn = input_ids.unsqueeze(-1).to(torch.float32)
        extra = {"text_input_ids": input_ids}
        return cross_attn, None, extra

    def load_sd(self, sd):
        return []

    def get_sd(self):
        return {}

    def reset_clip_options(self):
        pass

    def set_clip_options(self, options):
        pass
