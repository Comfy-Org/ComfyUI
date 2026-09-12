"""Gemma4 channel-marker decode regression tests."""

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.text_encoders.gemma4 import Gemma4SDTokenizer  # noqa: E402


class _CannedTokenizer:
    """Stands in for the wrapped tokenizer so decode is checked without model files."""
    def __init__(self, text):
        self.text = text

    def decode(self, token_ids, skip_special_tokens=False):
        return self.text


def decode(text):
    tokenizer = Gemma4SDTokenizer.__new__(Gemma4SDTokenizer)
    tokenizer.tokenizer = _CannedTokenizer(text)
    return tokenizer.decode([])


def test_thought_channel_becomes_think_tags():
    assert decode("<|channel>thought\nreasoning<channel|>the answer") == "<think>\nreasoning</think>the answer"


def test_primed_empty_thought_channel_closes_immediately():
    assert decode("<|channel>thought\n<channel|>the answer") == "<think>\n</think>the answer"


def test_thought_channel_left_unclosed_still_opens():
    assert decode("<|channel>thought\nreasoning") == "<think>\nreasoning"


def test_close_of_a_non_thought_channel_is_not_reasoning():
    # Non-thinking LTX2 prompt enhancement primes a "final" channel in the prompt, so only its
    # close is generated. Reading that close as </think> made the whole answer look like
    # reasoning and the node returned an empty string.
    assert decode("the answer<channel|>") == "the answer"


def test_thought_close_does_not_consume_a_later_channel():
    assert decode("<|channel>thought\nreasoning<channel|><|channel>final\nthe answer<channel|>") == "<think>\nreasoning</think>the answer"


def test_turn_and_eos_markers_are_stripped():
    assert decode("<|turn>model\nthe answer<turn|><eos>") == "the answer"
