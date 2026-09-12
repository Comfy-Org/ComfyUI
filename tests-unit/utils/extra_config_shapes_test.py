"""A YAML list of paths in extra_model_paths.yaml aborted startup.

`load_extra_path_config()` called `.split("\\n")` on every value, so the shape
people naturally write —

    comfyui:
      base_path: /models
      checkpoints:
        - ckpt_a
        - ckpt_b

— raised `AttributeError: 'list' object has no attribute 'split'`, a traceback
naming neither the file nor the key. Measured on master before the change:

    list of paths        -> AttributeError: 'list' object has no attribute 'split'
    numeric value        -> AttributeError: 'int' object has no attribute 'split'
    nested mapping       -> AttributeError: 'dict' object has no attribute 'split'
    section is a string  -> TypeError: string indices must be integers
    newline string       -> ok  (the documented form, unaffected)

A malformed entry now warns and is skipped, so one bad key no longer takes the
whole config — and the rest of the search paths still load.
"""

import itertools
import os

import pytest
from unittest.mock import patch

import folder_paths
from utils.extra_config import load_extra_path_config


@pytest.fixture
def load_yaml(tmp_path):
    """Write a config, load it, and return the paths that were registered."""
    counter = itertools.count()

    def _load(text: str) -> list[tuple]:
        # `tmp_path` is cleaned up by pytest; a fresh name per call keeps two
        # loads in the same test from overwriting each other.
        path = tmp_path / f"extra_model_paths_{next(counter)}.yaml"
        path.write_text(text, encoding="utf-8")
        path = str(path)

        added: list[tuple] = []
        with patch.object(
            folder_paths, "add_model_folder_path", lambda *args, **kwargs: added.append(args)
        ):
            load_extra_path_config(path)
        return added

    return _load


def _basenames(added: list[tuple]) -> list[str]:
    return [os.path.basename(entry[1]) for entry in added]


def test_list_of_paths_is_accepted(load_yaml):
    added = load_yaml(
        "comfyui:\n"
        "  base_path: /models\n"
        "  checkpoints:\n"
        "    - ckpt_a\n"
        "    - ckpt_b\n"
    )

    assert _basenames(added) == ["ckpt_a", "ckpt_b"]
    assert [entry[0] for entry in added] == ["checkpoints", "checkpoints"]


def test_newline_string_is_unchanged(load_yaml):
    """The documented form must keep behaving exactly as before."""
    added = load_yaml(
        "comfyui:\n"
        "  base_path: /models\n"
        "  checkpoints: |\n"
        "    ckpt_a\n"
        "    ckpt_b\n"
    )

    assert _basenames(added) == ["ckpt_a", "ckpt_b"]


def test_list_and_string_agree(load_yaml):
    as_list = load_yaml("comfyui:\n  base_path: /models\n  checkpoints:\n    - a\n    - b\n")
    as_string = load_yaml("comfyui:\n  base_path: /models\n  checkpoints: |\n    a\n    b\n")

    assert as_list == as_string


@pytest.mark.parametrize(
    "bad_value",
    [
        "  checkpoints: 42\n",
        "  checkpoints: true\n",
        "  checkpoints:\n    a: b\n",
    ],
)
def test_a_malformed_value_is_skipped_without_losing_the_rest(load_yaml, bad_value):
    added = load_yaml("comfyui:\n  base_path: /models\n" + bad_value + "  loras: keep_me\n")

    assert _basenames(added) == ["keep_me"]


def test_a_malformed_entry_inside_a_list_is_skipped(load_yaml):
    added = load_yaml(
        "comfyui:\n  base_path: /models\n  checkpoints:\n    - ok_a\n    - 7\n"
    )

    assert _basenames(added) == ["ok_a"]


def test_a_section_that_is_not_a_mapping_is_skipped(load_yaml):
    added = load_yaml("comfyui: /models\nother:\n  base_path: /m2\n  loras: keep_me\n")

    assert _basenames(added) == ["keep_me"]


def test_an_empty_section_is_still_tolerated(load_yaml):
    added = load_yaml("comfyui:\nother:\n  base_path: /m2\n  loras: keep_me\n")

    assert _basenames(added) == ["keep_me"]
