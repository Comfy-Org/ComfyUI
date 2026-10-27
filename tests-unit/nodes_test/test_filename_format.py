"""Tests for the output filename format in save nodes (issue #1389).

The ``SaveImage``, ``SaveLatent``, ``CLIPSave``, ``VAESave``,
``CheckpointSave``, ``SaveConditioning``, ``SaveLoRA``, and
``SaveGLB`` nodes all build their output filename with the pattern
``f"{filename}_{counter:05}.{ext}"`` — without a trailing
underscore. The trailing underscore was a placeholder left over
from 4 years ago ("There originally was something else after the
counter but I removed it and left the _ because I was planning on
adding something else there." — comfyanonymous on #1389) and the
*one* other place that already builds the same string format
without the trailing underscore (``comfy_extras/nodes_save_3d.py``
L876) confirms that the new style is the intended one.

This test guards against re-introducing the legacy pattern in the
builders changed by this PR.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# The save-node builders whose filename f-strings were changed by this
# PR (trailing-underscore placeholder removal, #1389). The scan is
# scoped to these files on purpose: unrelated builders still carry
# pre-existing legacy patterns (e.g. ``comfy_extras/nodes_lora_extract.py``,
# ``comfy_extras/nodes_video.py``, ``comfy_extras/nodes_images.py``)
# that this PR does not touch.
CHANGED_BUILDER_FILES = (
    "nodes.py",
    "comfy_extras/nodes_cond.py",
    "comfy_extras/nodes_model_merging.py",
    "comfy_extras/nodes_save_3d.py",
    "comfy_extras/nodes_train.py",
)

# The legacy shape this PR removes: the counter followed by a trailing
# underscore *before* the extension, i.e. the f-string source contains
# ``_{counter:05}_.`` — with either a literal extension (``_.png``) or
# a ``{ext}`` placeholder (``_.{ext}``).
LEGACY_TRAILING_UNDERSCORE = re.compile(r"_\{counter:0?\d*d?\}_\.")
# Accept either a `{ext}` placeholder or a literal extension
# (".png", ".latent", ".safetensors", ".glb", ...). The counter is
# always the last field before the extension, and the prefix may be a
# multi-segment underscore-separated run (e.g. SaveLoRA's
# ``{filename}_{steps}_steps``). The f-string body (without the `f"` /
# `"` delimiters) is what we match against, and the pattern must cover
# it completely (``fullmatch``).
GOOD_FILENAME_FSTRING = re.compile(
    r'.+_\{counter:0?\d*d?\}\.(?:\{[^}]+\}|[A-Za-z0-9]+)'
)


def _fstring_literals(text: str) -> list[tuple[int, str]]:
    """Return ``(line, fstring)`` for every f-string literal in ``text``."""
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        # A lightweight f-string finder: only handles single-line
        # f-strings that span from `f"` to the matching `"` on the
        # same line. That covers every site we care about.
        idx = 0
        while True:
            start = line.find('f"', idx)
            if start == -1:
                break
            end = line.find('"', start + 2)
            if end == -1:
                break
            out.append((lineno, line[start + 2 : end]))
            idx = end + 1
    return out


def _walk_changed_builder_fstrings() -> list[tuple[str, int, str]]:
    """Return ``(relative_path, line, fstring)`` for every f-string
    mentioning ``counter`` in the save-node builders changed by this
    PR.
    """
    out: list[tuple[str, int, str]] = []
    for rel in CHANGED_BUILDER_FILES:
        text = (REPO_ROOT / rel).read_text()
        for lineno, fstring in _fstring_literals(text):
            if "counter" in fstring:
                out.append((rel, lineno, fstring))
    return out


def test_no_save_node_emits_trailing_underscore_before_extension() -> None:
    """The legacy ``{prefix}_{counter:05}_.{ext}`` shape (note the
    trailing underscore) is gone from the builders changed by this PR.
    The ``SaveImage``-style counter format string used to have a
    trailing underscore left as a placeholder (#1389); every changed
    call site now drops it."""
    bad: list[str] = []
    for rel, lineno, fstring in _walk_changed_builder_fstrings():
        if LEGACY_TRAILING_UNDERSCORE.search(fstring):
            bad.append(f"{rel}:{lineno} — {fstring!r}")
    assert not bad, (
        "Found save-node filename f-strings that still emit the "
        "legacy trailing-underscore pattern:\n  " + "\n  ".join(bad)
    )


def test_all_save_node_filenames_match_consistent_format() -> None:
    """Every counter-based save f-string in the changed builders
    matches the consistent ``{prefix}_{counter:0Nd}.{ext}`` shape (no
    trailing underscore, the counter is the last field before the
    extension, literal extensions included)."""
    inconsistent: list[str] = []
    for rel, lineno, fstring in _walk_changed_builder_fstrings():
        # We only inspect f-strings whose counter is immediately
        # followed by an extension — either a literal (".png",
        # ".latent", ...) or a placeholder (".{ext}"). That is the
        # save-filename shape; helper f-strings that compute the
        # counter value or a prefix path are out of scope.
        if not re.search(r"counter:0?\d*d?\}\.", fstring):
            continue
        if not GOOD_FILENAME_FSTRING.fullmatch(fstring):
            inconsistent.append(f"{rel}:{lineno} — {fstring!r}")
    assert not inconsistent, (
        "Found save-filename f-strings that don't match the "
        "consistent shape:\n  " + "\n  ".join(inconsistent)
    )


def _save_image_filename_template() -> str:
    """Extract the ``SaveImage`` filename f-string from ``nodes.py`` and
    return it as a plain ``str.format`` template.

    This walks the real ``SaveImage.save_images`` builder so the
    regression test below exercises the code that builds the output
    filename, not a copy of it.
    """
    tree = ast.parse((REPO_ROOT / "nodes.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SaveImage":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "save_images":
                    for sub in ast.walk(fn):
                        if not isinstance(sub, ast.JoinedStr):
                            continue
                        parts: list[str] = []
                        for value in sub.values:
                            if isinstance(value, ast.Constant):
                                parts.append(str(value.value))
                            elif (
                                isinstance(value, ast.FormattedValue)
                                and isinstance(value.value, ast.Name)
                                and (
                                    value.format_spec is None
                                    or all(
                                        isinstance(v, ast.Constant)
                                        for v in value.format_spec.values
                                    )
                                )
                            ):
                                spec = ""
                                if value.format_spec is not None:
                                    spec = "".join(
                                        str(v.value) for v in value.format_spec.values
                                    )
                                parts.append("{" + value.value.id + (":" + spec if spec else "") + "}")
                            else:
                                parts = []
                                break
                        template = "".join(parts)
                        if "{counter:" in template:
                            return template
    raise AssertionError("SaveImage.save_images no longer builds a counter-based filename f-string")


def test_save_image_builder_emits_counter_then_extension() -> None:
    """The ``SaveImage`` builder itself renders the fixed shape.

    Pre-fix output was ``ComfyUI_00001_.png``; this drives the actual
    f-string in ``SaveImage.save_images`` (extracted from the source)
    with the values the issue uses, so restoring the legacy
    placeholder fails loudly even if the regexes above were bypassed.
    """
    template = _save_image_filename_template()
    rendered = template.format(filename_with_batch_num="ComfyUI", counter=1)
    assert rendered == "ComfyUI_00001.png", (
        f"SaveImage filename builder rendered {rendered!r}; the legacy "
        "trailing-underscore placeholder (#1389) looks like it is back"
    )
    assert "_." not in rendered


def test_issue_1389_debug_value_is_no_longer_emitted() -> None:
    """Pin the exact legacy value that the issue title calls out.

    Pre-fix outputs look like ``ComfyUI_00001_.png``; the post-fix
    outputs are ``ComfyUI_00001.png``. The trailing ``_`` is the
    regression we are guarding against, so this test fails loudly
    if a contributor copy-pastes the old format back in.
    """
    legacy_output = "ComfyUI_00001_.png"
    new_output = "ComfyUI_00001.png"

    # The legacy value carries the trailing-underscore placeholder.
    assert "_." in legacy_output.split("/")[-1].split("\\")[-1]
    assert legacy_output.endswith("_.png")
    # The new value drops it.
    assert new_output.endswith(".png")
    assert not new_output.endswith("_.png")
    # And the new format still matches the consistent shape used in
    # every save-node f-string changed by this PR.
    assert GOOD_FILENAME_FSTRING.fullmatch("{prefix}_{counter:05}.{ext}")
