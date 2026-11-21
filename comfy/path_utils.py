"""Small stdlib-only path helpers shared across ComfyUI.

Kept separate from :mod:`comfy.utils` so it can be imported in unit tests
that run without a torch install.
"""

import os


def is_subpath(path: str, parent: str) -> bool:
    """Return True if ``path`` is the same as or located under ``parent``.

    ``os.path.commonpath`` raises ``ValueError`` when the two paths live on
    different drives (e.g. ``C:\\`` vs ``D:\\`` on Windows). In that case
    the answer is unambiguously "no" -- they cannot be in the same
    directory tree -- so we return False instead of letting the exception
    bubble up and turn the request into a 500.
    """
    abs_path = os.path.abspath(path)
    abs_parent = os.path.abspath(parent)
    try:
        # Parent goes first: on Windows, commonpath is case-insensitive but
        # returns the first argument's casing, so a contained path whose
        # case differs from the parent would otherwise compare unequal.
        return os.path.commonpath((abs_parent, abs_path)) == abs_parent
    except ValueError:
        return False
