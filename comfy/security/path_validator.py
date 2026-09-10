"""Centralized path-traversal validation for HTTP-exposed file paths.

Replaces ad-hoc checks (`'..' in filename`, `commonpath` comparisons, etc.)
that previously lived inline across server.py upload/view endpoints.
Issue: https://github.com/Comfy-Org/ComfyUI/issues/13742
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, os.PathLike]


def resolve_safe_path(base_dir: PathLike, *user_parts: PathLike) -> Path | None:
    """Resolve user-supplied path parts against a trusted base directory.

    Joins ``user_parts`` to ``base_dir``, fully resolves the result (following
    symlinks), and returns it only if it stays inside ``base_dir``. Returns
    ``None`` for any unsafe input — null bytes, absolute user parts that escape
    the base, ``..`` segments that walk above the base, symlinks that point
    outside the base, or any OS-level resolution failure.

    Callers must use the returned ``Path`` for filesystem operations and must
    not re-join the original user input afterwards, or the validation is moot.
    """
    for part in user_parts:
        if _has_unsafe_chars(part):
            return None

    try:
        base = Path(base_dir).resolve(strict=False)
    except (OSError, ValueError):
        return None

    try:
        joined = base.joinpath(*(os.fspath(p) for p in user_parts))
        candidate = joined.resolve(strict=False)
    except (OSError, ValueError):
        return None

    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def _has_unsafe_chars(part: PathLike) -> bool:
    s = os.fspath(part)
    # NUL byte: some platforms truncate paths at \x00, which can defeat
    # subsequent containment checks performed on the truncated string.
    return "\x00" in s
