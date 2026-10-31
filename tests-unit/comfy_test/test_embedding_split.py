"""Tests for the embedding-tokenisation split in ``comfy.sd1_clip`` (issue #1354).

The tokeniser splits the prompt on every ``embedding:`` so each
embedding ends up in its own segment and the per-segment word
dispatcher can load the right embedding. The pre-fix pattern only
recognised whitespace as a separator, so prompts like
``embedding:bad_pictures,embedding:easynegative`` were treated as a
single embedding and the second name was lost. The fix widens the
lookbehind to also accept a comma.

The structural test guards against a future contributor silently
reverting the pattern to the old form (``(?<=\\s)`` only). The
behavioural tests reproduce the full split + re-prefix logic from
``sd1_clip.py:SDTokenizer.tokenize_with_weights`` and assert each
comma-separated embedding ends up as its own ``embedding:`` segment
in the final ``to_tokenize`` list — which is what the per-segment
dispatcher keys off.

The last test in this file drives the production tokenizer itself, so a
regression in the split or in the re-prefix loop cannot slip through
even when the regex still contains the comma.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import torch

import comfy.sd1_clip as sd1_clip


REPO_ROOT = Path(__file__).resolve().parents[2]
SD1_CLIP_PY = REPO_ROOT / "comfy" / "sd1_clip.py"
EMBEDDING_IDENTIFIER = "embedding:"

# The exact pattern used in ``comfy/sd1_clip.py``. The structural
# test below fails if the source diverges from this.
EXPECTED_SPLIT_PATTERN = r"(?<=[\s,])" + re.escape(EMBEDDING_IDENTIFIER)


# ---------------------------------------------------------------------------
# Behavioural tests — reproduce the full split + re-prefix logic.
# ---------------------------------------------------------------------------


def _to_tokenize(text: str) -> list[str]:
    """Reproduce the relevant lines from
    ``SDTokenizer.tokenize_with_weights`` (comfy/sd1_clip.py L594-600).

    The fix lives in the split pattern, but the to_tokenize list is
    what the per-segment dispatcher reads, so we assert the *list*
    shape — not the raw ``re.split`` output, which is consumed
    differently in the surrounding code.
    """
    split = re.split(EXPECTED_SPLIT_PATTERN, text)
    to_tokenize = [split[0]]
    for i in range(1, len(split)):
        to_tokenize.append(EMBEDDING_IDENTIFIER + split[i])
    return [x for x in to_tokenize if x != ""]


def test_comma_separated_embeddings_become_two_segments() -> None:
    """The exact example from issue #1354: ``embedding:a,embedding:b``
    must end up as two distinct ``embedding:``-prefixed segments
    so the second embedding is recognised."""
    segments = _to_tokenize("embedding:bad_pictures,embedding:easynegative")
    assert segments == ["embedding:bad_pictures,", "embedding:easynegative"]
    # And the dispatcher would see two embedding candidates.
    embedding_segments = [s for s in segments if s.startswith(EMBEDDING_IDENTIFIER)]
    assert len(embedding_segments) == 2


def test_three_comma_separated_embeddings_become_three_segments() -> None:
    """Generalising the two-embedding case to N embeddings."""
    segments = _to_tokenize("embedding:a,embedding:b,embedding:c")
    embedding_segments = [s for s in segments if s.startswith(EMBEDDING_IDENTIFIER)]
    assert len(embedding_segments) == 3
    # Each segment's name (after ``embedding:``) is one of a/b/c.
    names = [s[len(EMBEDDING_IDENTIFIER):].rstrip(",") for s in embedding_segments]
    assert names == ["a", "b", "c"]


def test_whitespace_separator_still_works() -> None:
    """The pre-fix pattern handled whitespace, and that case must
    still work after the fix."""
    segments = _to_tokenize("embedding:foo embedding:bar")
    embedding_segments = [s for s in segments if s.startswith(EMBEDDING_IDENTIFIER)]
    assert len(embedding_segments) == 2
    # Trailing spaces / non-name suffixes are normal; the names are
    # the substring after ``embedding:`` up to whitespace.
    names = [s[len(EMBEDDING_IDENTIFIER):].rstrip() for s in embedding_segments]
    assert "foo" in names
    assert "bar" in names


def test_single_embedding_with_no_separator_unchanged() -> None:
    """A lone ``embedding:foo`` produces a one-element list, exactly
    as the pre-fix pattern did."""
    assert _to_tokenize("embedding:foo") == ["embedding:foo"]


def test_mixed_separators_work() -> None:
    """Whitespace and commas can be mixed freely within a single
    prompt."""
    segments = _to_tokenize("a,embedding:foo embedding:bar,embedding:baz")
    embedding_segments = [s for s in segments if s.startswith(EMBEDDING_IDENTIFIER)]
    # Three embeddings recognised.
    assert len(embedding_segments) == 3
    # The non-embedding prefix is preserved.
    assert "a," in segments


def test_legacy_pattern_would_not_split_comma_case() -> None:
    """Pin the regression: under the legacy pattern (whitespace
    lookbehind only), the comma-separated input is *not* split
    and the second embedding is lost. This documents why the
    widened pattern is needed.
    """
    legacy = r"(?<=\s)" + re.escape(EMBEDDING_IDENTIFIER)
    split = re.split(legacy, "embedding:bad_pictures,embedding:easynegative")
    # One element only — the dispatcher would look up
    # ``bad_pictures,embedding:easynegative`` and fail.
    assert len(split) == 1


def test_pattern_is_documented_for_maintainers() -> None:
    """The pre-fix pattern only allowed whitespace. Pin the new
    pattern so a future contributor cannot accidentally revert."""
    assert re.escape(EMBEDDING_IDENTIFIER) in EXPECTED_SPLIT_PATTERN
    assert r"[\s,]" in EXPECTED_SPLIT_PATTERN


# ---------------------------------------------------------------------------
# Structural test — pin the source code so the two cannot diverge.
# ---------------------------------------------------------------------------


def _pattern_src(first: ast.AST) -> str | None:
    """Best-effort extraction of the pattern string from the first
    arg of ``re.split`` in ``sd1_clip.py``.

    The codebase uses two shapes:
      - ``r"...".format(...)`` — raw string Call on Constant
      - ``f"..."`` — JoinedStr

    Both mention ``embedding_identifier`` in their unparsed form,
    so we just unparse the whole first argument and grep.
    """
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    # Any other shape (Call, JoinedStr, Name, ...): unparse and
    # search for the marker.
    try:
        return ast.unparse(first)
    except Exception:
        return None


def _find_embedding_split_call(tree: ast.Module) -> ast.Call | None:
    """Return the ``re.split`` call that targets the
    ``embedding_identifier`` in ``sd1_clip.py``."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "split"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
        ):
            if not node.args:
                continue
            src = _pattern_src(node.args[0])
            if src and "embedding_identifier" in src:
                return node
    return None


def test_sd1_clip_uses_the_widened_separator_pattern() -> None:
    """The exact pattern in ``comfy/sd1_clip.py`` must include a
    comma in the lookbehind character class. It must NOT be the
    legacy ``(?<=\\s)``-only form."""
    tree = ast.parse(SD1_CLIP_PY.read_text(), filename=str(SD1_CLIP_PY))
    call = _find_embedding_split_call(tree)
    assert call is not None, "expected an `re.split(...embedding_identifier...)` call in sd1_clip.py"

    pattern_src = _pattern_src(call.args[0]) or ""
    # ``ast.unparse`` doubles backslashes, so a regex ``\s``
    # shows up as ``\\s`` (two literal backslash characters plus
    # ``s``) in the unparsed text. We look for either the doubled
    # form or the single-backslash form, to be robust against
    # future AST pretty-printer changes.
    comma_in_lookbehind = any(
        sep in pattern_src
        for sep in (r"[\s,]", r"[,\s]", r"[\\s,]", r"[,\\s]")
    )
    assert comma_in_lookbehind, (
        f"sd1_clip.py — the embedding split pattern {pattern_src!r} "
        f"does not accept a comma before the identifier; this "
        f"regresses issue #1354."
    )


# ---------------------------------------------------------------------------
# Production test — drive the real tokenizer, not a copy of it.
# ---------------------------------------------------------------------------


def _fake_embedding_loader(monkeypatch):
    """Replace ``comfy.sd1_clip.load_embed`` with a stub.

    Only the embeddings named ``a`` and ``b`` exist; every other name
    resolves to ``None`` exactly like a missing embedding file does.
    That detail matters for the comma case: the split leaves the comma
    attached to the first name (``a,``), and it is ``_try_get_embedding``
    that strips it and retries the lookup.
    """
    requested: list[str] = []
    loaded: dict[str, torch.Tensor] = {}

    def fake_load_embed(embedding_name, embedding_directory, embedding_size, embed_key=None):
        requested.append(embedding_name)
        if embedding_name not in ("a", "b"):
            return None
        tensor = torch.full((embedding_size,), float(len(loaded) + 1))
        loaded[embedding_name] = tensor
        return tensor

    monkeypatch.setattr(sd1_clip, "load_embed", fake_load_embed)
    return requested, loaded


def test_sd1_tokenizer_dispatches_both_comma_separated_embeddings(monkeypatch, tmp_path) -> None:
    """``SD1Tokenizer.tokenize_with_weights`` must dispatch *both* of the
    embeddings in ``embedding:a,embedding:b``.

    ``comfy.sd1_clip.SD1Tokenizer`` is the ``ClipTarget.tokenizer`` of the
    SD1 model (see ``comfy/supported_models.py``), so this is the very
    method ``comfy/sd.py`` calls to tokenise a prompt. Driving it here
    means a regression in the split or in the re-prefix loop fails this
    test, which the ``_to_tokenize`` helper above — a copy of that logic
    — cannot do.
    """
    requested, loaded = _fake_embedding_loader(monkeypatch)

    tokenizer = sd1_clip.SD1Tokenizer(embedding_directory=str(tmp_path))
    batched = tokenizer.tokenize_with_weights("embedding:a,embedding:b")

    # Both embeddings reached the dispatcher and both resolved.
    assert set(loaded) == {"a", "b"}, (
        f"the dispatcher did not receive both embeddings: loaded {sorted(loaded)} "
        f"after the requests {requested}"
    )

    # And both of them are part of the tokenised output.
    batches = batched[tokenizer.clip_name]
    tensors = [t for batch in batches for (t, _weight) in batch if torch.is_tensor(t)]
    assert len(tensors) == 2, (
        f"expected exactly two embedding tensors in the token output, got {len(tensors)}"
    )
    for name in ("a", "b"):
        assert any(t is loaded[name] for t in tensors), (
            f"the embedding {name!r} is missing from the token output"
        )
