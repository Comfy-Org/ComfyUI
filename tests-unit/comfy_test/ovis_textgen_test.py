"""The Qwen3-2B (Ovis) text encoder: conditioning keeps the Ovis template, TextGenerate gets the Qwen3 chat template."""

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops  # noqa: E402
from comfy.text_encoders.flux import KleinTokenizer8B  # noqa: E402
from comfy.text_encoders.llama import Ovis25_2B  # noqa: E402
from comfy.text_encoders.ovis import OvisTokenizer  # noqa: E402
from comfy.text_encoders.z_image import ZImageTokenizer  # noqa: E402
from comfy_extras.nodes_textgen import TextGenerateV2  # noqa: E402

PROMPT = "a red fox in the snow"
OVIS = "<|im_start|>user\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background: {}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CHAT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
NO_THINK = "<think>\n\n</think>\n\n"


@pytest.fixture(scope="module")
def tokenizer():
    return OvisTokenizer()


@pytest.fixture(scope="module")
def other_qwen3_sizes():
    return ZImageTokenizer(), KleinTokenizer8B()  # Qwen3-4B, Qwen3-8B


def ids_of(tokens):
    return [t[0] for t in next(iter(tokens.values()))[0]]


def tokenize(tokenizer, text, **kwargs):
    ids = ids_of(tokenizer.tokenize_with_weights(text, **kwargs))
    return ids, tokenizer.decode(ids, skip_special_tokens=False)


def test_conditioning_keeps_the_ovis_template(tokenizer):
    ids, text = tokenize(tokenizer, PROMPT)
    assert len(ids) == 284  # padded to the encoder's min_length
    assert text.removesuffix("<|endoftext|>" * text.count("<|endoftext|>")) == OVIS.format(PROMPT)
    assert tokenize(tokenizer, PROMPT, llama_template="{}", thinking=False, min_length=1)[1] == PROMPT  # an explicit template still wins


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("system_prompt", ["", "You write image prompts."])
def test_textgenerate_chat_template_matches_the_other_qwen3_sizes(tokenizer, other_qwen3_sizes, thinking, system_prompt):
    kwargs = dict(min_length=1, thinking=thinking, system_prompt=system_prompt, skip_template=False)
    ids, text = tokenize(tokenizer, PROMPT, **kwargs)
    system = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
    assert text == system + CHAT.format(PROMPT) + ("" if thinking else NO_THINK)
    if not thinking:
        assert ids[-4:] == [151667, 271, 151668, 271]  # <think> \n\n </think> \n\n as single tokens
    for other in other_qwen3_sizes:
        assert ids_of(other.tokenize_with_weights(PROMPT, **kwargs)) == ids


def test_textgenerate_without_default_template_passes_text_through(tokenizer):
    raw = CHAT.format(PROMPT)
    _, text = tokenize(tokenizer, raw, min_length=1, thinking=False, system_prompt="", skip_template=True)
    assert text == raw


class RecordingClip:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize_with_weights(text, **kwargs)

    def generate(self, tokens, **kwargs):
        self.prompt_ids = ids_of(tokens)
        return [151645]

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def test_generate_text_node_gets_the_chat_template(tokenizer):
    clip = RecordingClip(tokenizer)
    TextGenerateV2.execute(clip, PROMPT, 8, {"sampling_mode": "off"})  # thinking and system prompt left at their defaults
    assert tokenizer.decode(clip.prompt_ids, skip_special_tokens=False) == CHAT.format(PROMPT) + NO_THINK


def tiny_model(vocab_size):
    torch.manual_seed(0)
    model = Ovis25_2B(dict(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1),
                      torch.float32, "cpu", comfy.ops.manual_cast)
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        p.data = 1 + torch.randn_like(p) * 0.05 if "norm" in name else torch.randn_like(p) * 0.3
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


@pytest.mark.parametrize("stop", [151643, 151645])  # <|endoftext|>, <|im_end|>
def test_generate_stops_at_end_of_turn(stop):
    model = tiny_model(151936)
    embeds = torch.randn(1, 5, 64)
    free = model.generate(embeds.clone(), do_sample=False, max_length=8, stop_tokens=[], execution_dtype=torch.float32)
    assert stop not in free
    k = next(i for i in range(1, len(free)) if free[i] not in free[:i])  # a decode step whose token is new
    weight = model.model.embed_tokens.weight.data
    weight[[free[k], stop]] = weight[[stop, free[k]]]  # the stop token takes its place, earlier steps are unchanged
    assert model.generate(embeds.clone(), do_sample=False, max_length=8, execution_dtype=torch.float32) == free[:k] + [stop]
