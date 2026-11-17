"""The Qwen3-2B (Ovis) text encoder: conditioning keeps the Ovis template, TextGenerate gets the Qwen3 chat template."""

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops  # noqa: E402
from comfy.text_encoders.llama import Ovis25_2B  # noqa: E402
from comfy.text_encoders.ovis import OvisTokenizer  # noqa: E402

PROMPT = "a red fox in the snow"
OVIS = "<|im_start|>user\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background: {}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CHAT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


@pytest.fixture(scope="module")
def tokenizer():
    return OvisTokenizer()


def tokenize(tokenizer, text, **kwargs):
    ids = [t[0] for t in tokenizer.tokenize_with_weights(text, **kwargs)["qwen3_2b"][0]]
    return ids, tokenizer.decode(ids, skip_special_tokens=False)


def test_conditioning_keeps_the_ovis_template(tokenizer):
    ids, text = tokenize(tokenizer, PROMPT)
    assert len(ids) == 284  # padded to the encoder's min_length
    assert text.removesuffix("<|endoftext|>" * text.count("<|endoftext|>")) == OVIS.format(PROMPT)


@pytest.mark.parametrize("system_prompt", ["", "You write image prompts."])
def test_textgenerate_thinking_off(tokenizer, system_prompt):
    ids, text = tokenize(tokenizer, PROMPT, min_length=1, thinking=False, system_prompt=system_prompt, skip_template=False)
    system = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
    assert text == system + CHAT.format(PROMPT) + "<think>\n\n</think>\n\n"
    assert ids[-4:] == [151667, 271, 151668, 271]  # <think> \n\n </think> \n\n as single tokens


def test_textgenerate_thinking_on_leaves_the_turn_open(tokenizer):
    _, text = tokenize(tokenizer, PROMPT, min_length=1, thinking=True, system_prompt="", skip_template=False)
    assert text == CHAT.format(PROMPT)


def test_textgenerate_without_default_template_passes_text_through(tokenizer):
    raw = CHAT.format(PROMPT)
    _, text = tokenize(tokenizer, raw, min_length=1, thinking=False, system_prompt="", skip_template=True)
    assert text == raw


def tiny_model(vocab_size):
    torch.manual_seed(0)
    model = Ovis25_2B(dict(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1),
                      torch.float32, "cpu", comfy.ops.manual_cast)
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        p.data = torch.randn_like(p) * (0.05 if "norm" in name else 0.3)
    return model


def test_greedy_generate_matches_full_forward_with_tied_head():
    model = tiny_model(97)
    embeds = torch.randn(1, 5, 64)
    generated = model.generate(embeds.clone(), do_sample=False, max_length=12, stop_tokens=[], execution_dtype=torch.float32)

    expected, x = [], embeds
    for _ in range(12):
        hidden = model.model(None, embeds=x.clone())[0]  # the blocks write their output into their input
        token = int((hidden[:, -1] @ model.model.embed_tokens.weight.T).argmax())
        expected.append(token)
        x = torch.cat([x, model.model.embed_tokens.weight[token].view(1, 1, -1)], dim=1)
    assert generated == expected


def test_generate_stops_at_im_end():
    model = tiny_model(151936)
    embeds = torch.randn(1, 5, 64)
    free = model.generate(embeds.clone(), do_sample=False, max_length=3, stop_tokens=[], execution_dtype=torch.float32)
    assert len(set(free)) == 3 and 151645 not in free
    weight = model.model.embed_tokens.weight.data
    weight[[free[2], 151645]] = weight[[151645, free[2]]]  # <|im_end|> takes the place of the third token
    assert model.generate(embeds.clone(), do_sample=False, max_length=8, execution_dtype=torch.float32) == free[:2] + [151645]
