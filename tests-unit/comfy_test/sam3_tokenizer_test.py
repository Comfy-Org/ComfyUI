"""Regression test for the SAM3 tokenizer ":N" suffix leak.

comfy/text_encoders/sam3_clip.py's SAM3TokenizerWrapper.tokenize_with_weights()
takes a fast path when there's a single prompt with max_detections == 1 (i.e.
plain "person" or "person:1"). _parse_prompts() correctly strips the ":1"
suffix and returns [("person", 1)] in both cases, but the fast path forwarded
the raw, unparsed `text` to the inner tokenizer instead of the parsed phrase --
so "person:1" was encoded as the literal string "person:1" rather than
"person", producing a badly wrong embedding for some wordings (see issue for a
measured example where this drops SAM3's detected mask coverage on a person
from ~32% to <1%).

This mirrors tests-unit/comfy_test/gemma4_template_test.py's approach of
subclassing the real tokenizer wrapper with a small capture stand-in for the
inner SDTokenizer, so the fix is verified against the exact code path
(SAM3TokenizerWrapper.tokenize_with_weights) without needing real model/vocab
files.
"""

import pytest
import torch  # noqa: F401  (forces CPU args like gemma4_template_test.py, see below)

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.text_encoders.sam3_clip import SAM3TokenizerWrapper, _parse_prompts  # noqa: E402


class _CaptureInnerTokenizer:
    """Stands in for the real SDTokenizer so no vocab/model files are needed.

    Records exactly the text tokenize_with_weights() was called with.
    """

    def __init__(self, *args, **kwargs):
        self.calls = []

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        self.calls.append(text)
        return {"tokens": text}


def _make_wrapper():
    wrapper = SAM3TokenizerWrapper.__new__(SAM3TokenizerWrapper)
    wrapper.clip_name = "l"
    wrapper.clip = "l"
    setattr(wrapper, wrapper.clip, _CaptureInnerTokenizer())
    return wrapper


@pytest.mark.parametrize(
    "prompt,expected_encoded",
    [
        ("person", "person"),
        ("person:1", "person"),  # the exact regression from the issue
        ("  person  :  1  ", "person"),
    ],
)
def test_single_prompt_max_detections_one_encodes_stripped_phrase(prompt, expected_encoded):
    """Fast path (single prompt, max_detections==1) must forward the parsed
    phrase, not the raw text still carrying the ":N" suffix."""
    wrapper = _make_wrapper()
    inner = getattr(wrapper, wrapper.clip)

    wrapper.tokenize_with_weights(prompt)

    assert inner.calls == [expected_encoded]


def test_bare_prompt_without_suffix_is_unaffected():
    """Sanity check: a bare prompt with nothing to strip must still work,
    guarding against the fix accidentally requiring a ":N" suffix to be present."""
    wrapper = _make_wrapper()
    inner = getattr(wrapper, wrapper.clip)

    wrapper.tokenize_with_weights("girl")

    assert inner.calls == ["girl"]


def test_multi_prompt_path_is_unaffected_by_the_fix():
    """person:2 and person:1,person:1 already took the (correct) multi-prompt
    path before this fix; confirm it still tokenizes each parsed phrase
    separately and is untouched by the single-prompt fast-path change."""
    wrapper = _make_wrapper()
    inner = getattr(wrapper, wrapper.clip)

    out = wrapper.tokenize_with_weights("person:2")

    assert inner.calls == ["person"]
    assert out["sam3_per_prompt"][0][1] == 2  # max_detections preserved

    wrapper2 = _make_wrapper()
    inner2 = getattr(wrapper2, wrapper2.clip)
    wrapper2.tokenize_with_weights("person:1,person:1")
    assert inner2.calls == ["person", "person"]


def test_empty_prompt_falls_back_to_raw_text():
    """_parse_prompts('') -> [] (nothing to strip), so the fast path must fall
    back to the original (empty) text rather than indexing into an empty list."""
    wrapper = _make_wrapper()
    inner = getattr(wrapper, wrapper.clip)

    wrapper.tokenize_with_weights("")

    assert inner.calls == [""]


def test_parse_prompts_matches_the_issues_repro_table():
    """Pin the exact _parse_prompts() outputs from the issue's repro script,
    so a future change to the parser can't silently reintroduce the leak by a
    different route."""
    assert _parse_prompts("person") == [("person", 1)]
    assert _parse_prompts("person:1") == [("person", 1)]
    assert _parse_prompts("person:2") == [("person", 2)]
    assert _parse_prompts("person:1,person:1") == [("person", 1), ("person", 1)]
