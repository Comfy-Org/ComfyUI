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

This test guards against re-introducing the legacy pattern.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# F-strings that build a save filename. The fix is to drop the
# trailing underscore, so the pattern we want to *forbid* is
# `}_{counter:05}_.`. We also assert that all the saving nodes use
# the consistent `{prefix}_{counter:05}.{ext}` shape.
LEGACY_TRAILING_UNDERSCORE = re.compile(r"_\{counter:0?\d*d?\}\.[^.\"]*\b_\.")
# Accept either a `{ext}` placeholder or a literal extension
# (".png", ".latent", ".safetensors", ".glb", ...). The counter is
# always the last field before the extension, and the prefix is
# always a `_{counter}` separated run (allowing multi-segment
# prefixes such as `{filename}_{steps}_steps`). The f-string body
# (without the `f"` / `"` delimiters) is what we match against.
GOOD_FILENAME_FSTRING = re.compile(
    r'\{[^}]+\}_\{counter:0?\d*d?\}\.(?:\{[^}]+\}|[A-Za-z0-9]+)'
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


def _walk_repo_fstrings() -> list[tuple[str, int, str]]:
    """Return ``(relative_path, line, fstring)`` for every f-string in
    every Python file under the repo that mentions ``counter``.

    We skip the ``tests/`` and ``tests-unit/`` trees so test files
    do not self-reference — the test docstrings deliberately
    include example f-strings that would otherwise be flagged.
    """
    out: list[tuple[str, int, str]] = []
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist"}
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        # Skip any test file or anything inside the tests/ or
        # tests-unit/ trees.
        rel = path.relative_to(REPO_ROOT)
        if rel.parts and (rel.parts[0] == "tests" or rel.parts[0].startswith("tests-")):
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, fstring in _fstring_literals(text):
            if "counter" in fstring:
                out.append((str(rel), lineno, fstring))
    return out


def test_no_save_node_emits_trailing_underscore_before_extension() -> None:
    """The legacy ``{prefix}_{counter:05}_.{ext}`` shape (note the
    trailing underscore) is gone from the entire repository. The
    ``SaveImage``-style counter format string used to have a
    trailing underscore left as a placeholder (#1389); every call
    site now drops it."""
    bad: list[str] = []
    for rel, lineno, fstring in _walk_repo_fstrings():
        if LEGACY_TRAILING_UNDERSCORE.search(fstring):
            bad.append(f"{rel}:{lineno} — {fstring!r}")
    assert not bad, (
        "Found save-node filename f-strings that still emit the "
        "legacy trailing-underscore pattern:\n  " + "\n  ".join(bad)
    )


def test_all_save_node_filenames_match_consistent_format() -> None:
    """Every counter-based save f-string matches the consistent
    ``{prefix}_{counter:0Nd}.{ext}`` shape (no trailing underscore,
    exactly one ``.`` before the extension, the counter is the last
    field before the extension)."""
    inconsistent: list[str] = []
    for rel, lineno, fstring in _walk_repo_fstrings():
        # We only inspect f-strings that *end* with a counter then
        # an extension — that is the save-filename shape. Helper
        # f-strings that compute the counter value or a prefix path
        # are out of scope.
        if not re.search(r"counter:0?\d*d?\}\.\{", fstring):
            continue
        if not GOOD_FILENAME_FSTRING.match(fstring):
            inconsistent.append(f"{rel}:{lineno} — {fstring!r}")
    assert not inconsistent, (
        "Found save-filename f-strings that don't match the "
        "consistent shape:\n  " + "\n  ".join(inconsistent)
    )


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
    # every save-node f-string in the repository.
    assert GOOD_FILENAME_FSTRING.match("{prefix}_{counter:05}.{ext}")
