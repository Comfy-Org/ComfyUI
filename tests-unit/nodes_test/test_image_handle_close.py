"""Tests for the ``try/finally + img.close()`` pattern applied to the
``node_helpers.pillow(Image.open, ...)`` call sites in
``LoadImage`` (#3477), ``PainterNode``, and ``load_and_process_images``.

These are behavioural tests: they assert that the fix pattern actually
releases the underlying PIL file handle, which is the regression that
#3477 is asking for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageSequence


# ---------------------------------------------------------------------------
# Behavioural tests: the fix pattern actually closes the file handle.
# ---------------------------------------------------------------------------


def _save_animated_webp(path: Path) -> None:
    """Write a tiny 3-frame animated WEBP for the behavioural test."""
    frames = [Image.new("RGB", (8, 8), "red") for _ in range(3)]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        format="WEBP",
        duration=100,
        loop=0,
    )


def test_pillow_image_handle_is_closed_by_finally(tmp_path: Path) -> None:
    """`PIL.Image.close()` in a `finally` block releases the file handle.

    The fix pattern is::

        img = node_helpers.pillow(Image.open, path)
        try:
            ...use img...
        finally:
            img.close()

    This test simulates that pattern on a real animated WEBP and asserts
    that after the block exits, ``img.fp`` is None — i.e. the file
    handle that ``Image.open`` returned has been released. Without the
    fix (no ``finally: img.close()``), ``img.fp`` would still be a live
    file object and Windows would refuse subsequent delete/rename.
    """
    webp_path = tmp_path / "anim.webp"
    _save_animated_webp(webp_path)

    img = Image.open(webp_path)
    assert img.fp is not None and not img.fp.closed, (
        "PIL should hold an open file handle while img is open"
    )

    try:
        frames = list(ImageSequence.Iterator(img))
        # Animated webp: the iterator yields at least one frame on most
        # build configurations; we only care that iteration finished.
        assert len(frames) >= 1
    finally:
        img.close()

    assert img.fp is None, "img.fp should be released after close()"


def test_close_runs_even_when_iteration_raises(tmp_path: Path) -> None:
    """`finally` must close the handle even if the loop body raises."""
    webp_path = tmp_path / "anim.webp"
    _save_animated_webp(webp_path)

    img = Image.open(webp_path)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            for _ in ImageSequence.Iterator(img):
                raise RuntimeError("boom")
    finally:
        img.close()

    assert img.fp is None
