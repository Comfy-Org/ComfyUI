"""Tests for the output filename format in save nodes (issue #1389).

Every save node that writes a file builds its name with the pattern
``f"{prefix}_{counter:05}.{ext}"`` — without a trailing underscore.
The trailing underscore was a placeholder left over from 4 years ago
("There originally was something else after the counter but I removed
it and left the _ because I was planning on adding something else
there." — comfyanonymous on #1389).  Several save nodes already
built the same format without the trailing underscore (e.g.
``comfy_extras/nodes_save_3d.py`` L876, ``comfy_extras/nodes_text.py``
L48, ``comfy_extras/nodes_images.py`` L2063), confirming that the
new style is the intended one.

These tests guard the whole tree against re-introducing the legacy
pattern, and they pin the counter behaviour that the rename must not
change: the counter is derived from the files already on disk, so
renaming the *new* outputs never overwrites an existing one.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import folder_paths


REPO_ROOT = Path(__file__).resolve().parents[2]

# Runtime code that can build a saved output file name.  The test trees
# are deliberately not scanned: they hold sample payloads that use the
# legacy shape on purpose.
SCAN_ROOTS = ("comfy", "comfy_api", "comfy_api_nodes", "comfy_extras", "app")

# A counter field is always spelled ``{counter`` in an f-string, so a
# file without that text cannot hold a counter f-string.  Filtering on
# it first keeps the scan cheap and avoids parsing unrelated files.
COUNTER_TEXT = re.compile(r"\{\s*counter\b")

# The legacy shape this PR removes: the counter followed by a trailing
# underscore *before* the extension, i.e. the f-string source contains
# ``_{counter:05}_.`` — with either a literal extension (``_.png``) or
# a ``{ext}`` placeholder (``_.{ext}``).
LEGACY_TRAILING_UNDERSCORE = re.compile(r"_\{counter[^{}]*?\}_\.")

# A counter that is immediately followed by an extension — either a
# literal (".png", ".latent", ...) or a placeholder (".{ext}").
COUNTER_BEFORE_EXTENSION = re.compile(r"\{counter[^{}]*?\}\.")

# The consistent shape: a (possibly multi-segment) prefix, the counter
# as the last field before the extension, and the extension last.
# ``{filename}_{steps}_steps_{counter:05}.safetensors`` (SaveLoRA) and
# ``{filename_with_batch_num}_{counter:05}.png`` (SaveImage) both
# satisfy it.
GOOD_FILENAME_FSTRING = re.compile(
    r'.+_\{counter[^{}]*?\}\.(?:\{[^}]+\}|[A-Za-z0-9]+)'
)


def _joined_str_body(node: ast.JoinedStr) -> str:
    """Return the literal shape of an f-string.

    ``f"{prefix}_{counter:05}.png"`` becomes
    ``{prefix}_{counter:05}.png`` — the literal text is kept verbatim
    and every placeholder is rendered back into ``{name}`` / ``{name:spec}``
    form, so the result can be matched against the patterns above
    without depending on the quoting of the original source.
    """
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
            spec = ""
            if isinstance(value.format_spec, ast.JoinedStr):
                spec = "".join(
                    part.value
                    for part in value.format_spec.values
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                )
            parts.append("{" + value.value.id + (":" + spec if spec else "") + "}")
        else:
            # A placeholder that is not a plain name (an attribute, a
            # call, ...) — keep it as an opaque field.
            parts.append("{...}")
    return "".join(parts)


def _candidate_files() -> list[Path]:
    """Return every runtime file that could contain a counter f-string."""
    files = sorted(REPO_ROOT.glob("*.py"))
    for root in SCAN_ROOTS:
        files.extend(sorted((REPO_ROOT / root).rglob("*.py")))
    return [path for path in files if COUNTER_TEXT.search(path.read_text(encoding="utf-8", errors="replace"))]


def _counter_fstrings() -> list[tuple[str, int, str]]:
    """Return ``(relative_path, line, fstring_body)`` for every f-string
    that holds a counter field, over the whole runtime tree.
    """
    found: list[tuple[str, int, str]] = []
    for path in _candidate_files():
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        # A format spec is an f-string of its own, but it shares the
        # position of the f-string it belongs to; skip it so a counter
        # is not reported twice.
        format_specs = {
            id(node.format_spec)
            for node in ast.walk(tree)
            if isinstance(node, ast.FormattedValue) and node.format_spec is not None
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr) or id(node) in format_specs:
                continue
            body = _joined_str_body(node)
            if not COUNTER_TEXT.search(body):
                continue
            found.append((str(path.relative_to(REPO_ROOT)), node.lineno, body))
    return found


def test_no_save_node_emits_trailing_underscore_before_extension() -> None:
    """The legacy ``{prefix}_{counter:05}_.{ext}`` shape (note the
    trailing underscore) is gone from every save node in the tree.
    The ``SaveImage``-style counter format string used to have a
    trailing underscore left as a placeholder (#1389); every call site
    now drops it."""
    bad: list[str] = []
    for rel, lineno, fstring in _counter_fstrings():
        if LEGACY_TRAILING_UNDERSCORE.search(fstring):
            bad.append(f"{rel}:{lineno} — {fstring!r}")
    assert not bad, (
        "Found save-node filename f-strings that still emit the "
        "legacy trailing-underscore pattern:\n  " + "\n  ".join(bad)
    )


def test_all_save_node_filenames_match_consistent_format() -> None:
    """Every counter-based save f-string matches the consistent
    ``{prefix}_{counter:0Nd}.{ext}`` shape (no trailing underscore, the
    counter is the last field before the extension, literal extensions
    included)."""
    inconsistent: list[str] = []
    for rel, lineno, fstring in _counter_fstrings():
        # We only inspect f-strings whose counter is immediately
        # followed by an extension — either a literal (".png",
        # ".latent", ...) or a placeholder (".{ext}"). That is the
        # save-filename shape; helper f-strings that compute the
        # counter value or a prefix path are out of scope.
        if not COUNTER_BEFORE_EXTENSION.search(fstring):
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
    # The builder really produces the new value, not the legacy one.
    assert _save_image_filename_template().format(filename_with_batch_num="ComfyUI", counter=1) == new_output


def test_counter_still_counts_files_written_before_the_rename(tmp_path) -> None:
    """Renaming the new outputs must not change how the counter is
    derived from the files already on disk.

    ``folder_paths.get_save_image_path`` reads the digits of the
    existing file names, so a directory that still holds the legacy
    ``ComfyUI_00001_.png`` has to keep producing counter 2.  The fix
    renames what is written from now on; it never overwrites a file
    that is already there.
    """
    legacy_only = tmp_path / "legacy"
    legacy_only.mkdir()
    (legacy_only / "ComfyUI_00001_.png").write_bytes(b"")

    _, _, counter, _, _ = folder_paths.get_save_image_path("ComfyUI", str(legacy_only))
    assert counter == 2, (
        "the legacy ComfyUI_00001_.png no longer advances the counter; "
        "the next save would overwrite it"
    )

    # A directory that holds both shapes keeps counting the highest
    # counter it has seen, whichever shape the file has.
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "ComfyUI_00001_.png").write_bytes(b"")
    (mixed / "ComfyUI_00002.png").write_bytes(b"")
    _, _, counter, _, _ = folder_paths.get_save_image_path("ComfyUI", str(mixed))
    assert counter == 3
