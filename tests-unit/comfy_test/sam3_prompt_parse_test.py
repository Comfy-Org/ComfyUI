"""#15811 — SAM3's single-prompt fast path forwarded the raw prompt text.

`:N` sets max_detections per category and defaults to 1, so `person:1` is by
definition the same detection prompt as `person`. `_parse_prompts()` strips the
suffix correctly, but `SAM3TokenizerWrapper.tokenize_with_weights()` took the
fast path for exactly the `foo:1` shape (one prompt, max_detections == 1) and
handed `super()` the unparsed string — so the encoder grounded on the literal
`"person:1"`.

The module is loaded here with `comfy.sd1_clip` stubbed, so the forwarding
contract is asserted without torch, transformers or any model weights. The
stub is scoped to the load and never enters `sys.modules` under the real
module name.
"""

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "comfy" / "text_encoders" / "sam3_clip.py"


class _RecordingTokenizer:
    """Stands in for `sd1_clip.SD1Tokenizer`, recording what it is asked to tokenize."""

    def __init__(self, *args, **kwargs):
        self.seen = []
        self.clip = "clip_l"
        self.clip_name = "l"
        setattr(self, self.clip, self)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        self.seen.append(text)
        return [[("token", text)]]


@contextmanager
def _stubbed_sd1_clip():
    """Load sam3_clip with a stub base module, restoring sys.modules afterwards."""
    stub = types.ModuleType("comfy.sd1_clip")
    stub.SDClipModel = type("SDClipModel", (), {"__init__": lambda self, **kwargs: None})
    stub.SDTokenizer = type("SDTokenizer", (), {"__init__": lambda self, **kwargs: None})
    stub.SD1ClipModel = type("SD1ClipModel", (), {"__init__": lambda self, **kwargs: None})
    stub.SD1Tokenizer = _RecordingTokenizer

    comfy_pkg = sys.modules.get("comfy") or types.ModuleType("comfy")
    saved = {
        "comfy": sys.modules.get("comfy"),
        "comfy.sd1_clip": sys.modules.get("comfy.sd1_clip"),
    }
    saved_attr = getattr(comfy_pkg, "sd1_clip", None)
    try:
        sys.modules["comfy"] = comfy_pkg
        sys.modules["comfy.sd1_clip"] = stub
        comfy_pkg.sd1_clip = stub

        spec = importlib.util.spec_from_file_location("sam3_clip_under_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        if saved_attr is None:
            if hasattr(comfy_pkg, "sd1_clip"):
                delattr(comfy_pkg, "sd1_clip")
        else:
            comfy_pkg.sd1_clip = saved_attr


@pytest.fixture
def sam3():
    with _stubbed_sd1_clip() as module:
        yield module


def _tokenized(sam3_module, prompt):
    """The strings the inner tokenizer was actually handed for `prompt`."""
    wrapper = sam3_module.SAM3TokenizerWrapper()
    wrapper.tokenize_with_weights(prompt)
    return getattr(wrapper, wrapper.clip).seen


def test_max_detections_suffix_does_not_reach_the_encoder(sam3):
    assert _tokenized(sam3, "person:1") == ["person"]


def test_explicit_one_matches_the_bare_prompt(sam3):
    """The equivalence the prompt syntax promises: `:1` is the default."""
    assert _tokenized(sam3, "person:1") == _tokenized(sam3, "person")


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("person", ["person"]),
        ("person:1", ["person"]),
        ("  person  :  1  ", ["person"]),
        ("a person on a bike:1", ["a person on a bike"]),
        # `_parse_prompts` strips parentheses (SAM3 has weights disabled), and
        # the multi-prompt path already tokenized the stripped phrase; the fast
        # path now agrees with it instead of encoding the literal brackets.
        ("(person)", ["person"]),
    ],
)
def test_single_prompt_shapes(sam3, prompt, expected):
    assert _tokenized(sam3, prompt) == expected


def test_empty_prompt_still_reaches_the_tokenizer(sam3):
    # Nothing parses out of these, so the original text is forwarded unchanged
    # rather than being turned into None.
    for prompt in ["", "   ", ",", " , "]:
        assert _tokenized(sam3, prompt) == [prompt]


def test_multi_prompt_path_is_untouched(sam3):
    wrapper = sam3.SAM3TokenizerWrapper()
    out = wrapper.tokenize_with_weights("person:2, car")

    assert getattr(wrapper, wrapper.clip).seen == ["person", "car"]
    assert [max_det for _batches, max_det in out["sam3_per_prompt"]] == [2, 1]


def test_parse_prompts_itself_is_unchanged(sam3):
    assert sam3._parse_prompts("person") == [("person", 1)]
    assert sam3._parse_prompts("person:1") == [("person", 1)]
    assert sam3._parse_prompts("person:3") == [("person", 3)]
    assert sam3._parse_prompts("person:1.4") == [("person", 1)]
    assert sam3._parse_prompts("person, car:2") == [("person", 1), ("car", 2)]
    assert sam3._parse_prompts("") == []
