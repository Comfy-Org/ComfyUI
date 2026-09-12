"""Gemma4 chat template regression tests."""

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.text_encoders.gemma4 as gemma4  # noqa: E402

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
