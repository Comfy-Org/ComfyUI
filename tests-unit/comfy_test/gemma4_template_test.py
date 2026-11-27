"""Gemma4 chat template regression tests."""

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.sd1_clip as sd1_clip  # noqa: E402
import comfy.text_encoders.gemma4 as gemma4  # noqa: E402
import comfy.text_encoders.lt as lt  # noqa: E402

PROMPT = "describe a cute anime girl with fennec ears"
THOUGHT_BLOCK = "<|channel>thought\n<channel|>"

# E2B/E4B and 12B/31B ship different canonical chat templates: only the latter prime a
# closed thought block when thinking is off.
NO_PRIMING = [gemma4.Gemma4_E2B, gemma4.Gemma4_E4B]
PRIMING = [gemma4.Gemma4_31B, gemma4.Gemma4_12B]


class _CaptureTemplate:
    """Stands in for SDTokenizer.tokenize_with_weights so the built template is checked without model files."""
    llama_text = ""

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        self.llama_text = text
        return {}


def build_template(variant, **kwargs):
    prime = variant.tokenizer.tokenizer_class.prime_empty_thought
    probe = type("Probe", (gemma4.Gemma4_Tokenizer, _CaptureTemplate), {"prime_empty_thought": prime})()
    probe.tokenize_with_weights(PROMPT, **kwargs)
    return probe.llama_text


@pytest.mark.parametrize("variant", NO_PRIMING + PRIMING)
def test_thinking_enabled_only_asks_via_the_system_turn(variant):
    template = build_template(variant, skip_template=False, thinking=True)
    assert template == f"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n"


@pytest.mark.parametrize("variant", NO_PRIMING)
def test_thinking_disabled_does_not_prime_a_thought_channel(variant):
    template = build_template(variant, skip_template=False, thinking=False)
    assert template == f"<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n"
    assert "channel" not in template
    assert "<|think|>" not in template


@pytest.mark.parametrize("variant", PRIMING)
def test_thinking_disabled_primes_a_thought_channel(variant):
    template = build_template(variant, skip_template=False, thinking=False)
    assert template == f"<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n{THOUGHT_BLOCK}"


@pytest.mark.parametrize("variant", NO_PRIMING + PRIMING)
@pytest.mark.parametrize("thinking", [False, True])
def test_skip_template_passes_text_through_unchanged(variant, thinking):
    assert build_template(variant, skip_template=True, thinking=thinking) == PROMPT


class _StubBackendTokenizer:
    def __call__(self, text):
        return {"input_ids": [10, 11] if text else []}

    def get_vocab(self):
        return {"a": 10, "b": 11}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


@pytest.mark.parametrize(
    ("mixin", "embedding_key"),
    [
        (lt.Gemma3_Tokenizer, "gemma3_12b"),
        (gemma4.Gemma4_Tokenizer, "gemma4"),
    ],
)
def test_gemma_tokenizers_forward_kwargs_for_min_length_and_options(mixin, embedding_key):
    tok_cls = type(f"Test{mixin.__name__}", (mixin, sd1_clip.SDTokenizer), {})
    tokenizer = tok_cls(
        None,
        pad_with_end=False,
        embedding_size=3840,
        embedding_key=embedding_key,
        tokenizer_class=_StubBackendTokenizer,
        has_start_token=True,
        has_end_token=False,
        pad_to_max_length=False,
        max_length=99999999,
        min_length=1024,
        pad_left=True,
        disable_weights=True,
        start_token=2,
    )

    default_tokens = tokenizer.tokenize_with_weights("hi", skip_template=True)[0]
    assert len(default_tokens) == 1024
    assert [t for t, _ in default_tokens[-3:]] == [2, 10, 11]

    unpadded = tokenizer.tokenize_with_weights("hi", skip_template=True, min_length=1)[0]
    assert [t for t, _ in unpadded] == [2, 10, 11]

    custom = tokenizer.tokenize_with_weights(
        "hi", skip_template=True, tokenizer_options={f"{embedding_key}_min_length": 16}
    )[0]
    assert len(custom) == 16
    assert [t for t, _ in custom[-3:]] == [2, 10, 11]
