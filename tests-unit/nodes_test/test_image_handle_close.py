"""Regression tests for the PIL file-handle cleanup added to the image
loaders (``LoadImage.load_image`` in ``nodes.py``, ``PainterNode`` in
``comfy_extras/nodes_painter.py`` and ``load_and_process_images`` in
``comfy_extras/nodes_dataset.py``).

Every one of those loaders opens a file with
``node_helpers.pillow(Image.open, ...)``.  Pillow keeps the underlying
file handle open for multi-frame inputs so that ``seek()`` can reach the
later frames, so the handle has to be released explicitly -- both on the
success path and when the processing raises.

These tests drive the production loaders themselves (instead of
re-implementing the ``try/finally`` pattern inside the test) and assert
that the image object the loader opened no longer holds a file handle.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import node_helpers
import nodes
from comfy_extras import nodes_dataset, nodes_painter
from comfy_extras.nodes_dataset import load_and_process_images


class OpenedImage:
    """An image opened by a production loader, plus how it was closed."""

    def __init__(self, image, close_calls):
        self.image = image
        self.close_calls = close_calls

    @property
    def was_closed(self) -> bool:
        """True when the loader called ``close()`` on this very image."""
        return bool(self.close_calls)

    @property
    def handle_released(self) -> bool:
        """True when the file handle behind the image is gone."""
        return self.image.fp is None


def _save_animated_webp(path: Path, frames: int = 3) -> None:
    """Write a tiny multi-frame animated WEBP.

    The frames have to differ: the WEBP encoder folds consecutive
    identical frames into one, which would make the file single-frame.
    """
    images = [Image.new("RGB", (8, 8), color) for color in ("red", "green", "blue")[:frames]]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        format="WEBP",
        duration=100,
        loop=0,
    )


@pytest.fixture
def opened_images(monkeypatch):
    """Record the images the production loaders open.

    ``node_helpers.pillow`` is the single choke point every loader uses
    to open a file, so wrapping it captures the exact ``Image`` object
    whose handle has to be released.  ``close()`` is instrumented on
    that object so the test can assert that the loader closed *the
    image it opened* -- closing a converted or resized copy of it does
    not release the handle that ``Image.open`` took.
    """
    recorded = []
    original_pillow = node_helpers.pillow

    def recording_pillow(fn, arg):
        image = original_pillow(fn, arg)
        if fn is Image.open:
            close_calls = []
            original_close = image.close

            def close(*args, **kwargs):
                close_calls.append(True)
                return original_close(*args, **kwargs)

            image.close = close
            recorded.append(OpenedImage(image=image, close_calls=close_calls))
        return image

    monkeypatch.setattr(node_helpers, "pillow", recording_pillow)
    return recorded


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# comfy_extras/nodes_dataset.py -- load_and_process_images
# ---------------------------------------------------------------------------


def test_dataset_loader_closes_the_image_it_opened(tmp_path: Path, opened_images) -> None:
    """``load_and_process_images`` must release the file handle of the
    image it opened, even though the working image is replaced by a
    converted copy."""
    _save_animated_webp(tmp_path / "anim.webp")

    tensors = load_and_process_images(["anim.webp"], str(tmp_path))

    assert len(tensors) == 1
    assert len(opened_images) == 1
    assert opened_images[0].was_closed, (
        "load_and_process_images did not close the image it opened"
    )
    assert opened_images[0].handle_released, (
        "load_and_process_images left the file handle of the image it opened open"
    )


def test_dataset_loader_closes_the_image_it_opened_on_failure(
    tmp_path: Path, opened_images, monkeypatch
) -> None:
    """The ``finally`` block must also run when the processing raises."""
    _save_animated_webp(tmp_path / "anim.webp")
    monkeypatch.setattr(nodes_dataset.np, "array", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        nodes_dataset.load_and_process_images(["anim.webp"], str(tmp_path))

    assert len(opened_images) == 1
    assert opened_images[0].was_closed, (
        "load_and_process_images did not close the image it opened after a failure"
    )
    assert opened_images[0].handle_released, (
        "load_and_process_images left the file handle open after a failure"
    )


# ---------------------------------------------------------------------------
# comfy_extras/nodes_painter.py -- PainterNode
# ---------------------------------------------------------------------------


def test_painter_closes_the_image_it_opened(tmp_path: Path, opened_images, monkeypatch) -> None:
    """``PainterNode`` converts and resizes the mask, so the image it
    opened has to be tracked separately from the working image."""
    mask_path = tmp_path / "mask.webp"
    _save_animated_webp(mask_path)
    monkeypatch.setattr(
        nodes_painter.folder_paths,
        "get_annotated_filepath",
        lambda mask: str(mask_path),
    )

    output = nodes_painter.PainterNode.execute(mask="mask.webp", width=8, height=8)

    assert output is not None
    assert len(opened_images) == 1
    assert opened_images[0].was_closed, (
        "PainterNode did not close the image it opened"
    )
    assert opened_images[0].handle_released, (
        "PainterNode left the file handle of the image it opened open"
    )


def test_painter_closes_the_image_it_opened_on_failure(
    tmp_path: Path, opened_images, monkeypatch
) -> None:
    """The ``finally`` block must also run when the processing raises."""
    mask_path = tmp_path / "mask.webp"
    _save_animated_webp(mask_path)
    monkeypatch.setattr(
        nodes_painter.folder_paths,
        "get_annotated_filepath",
        lambda mask: str(mask_path),
    )
    monkeypatch.setattr(nodes_painter.np, "array", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        nodes_painter.PainterNode.execute(mask="mask.webp", width=8, height=8)

    assert len(opened_images) == 1
    assert opened_images[0].was_closed, (
        "PainterNode did not close the image it opened after a failure"
    )
    assert opened_images[0].handle_released, (
        "PainterNode left the file handle open after a failure"
    )


# ---------------------------------------------------------------------------
# nodes.py -- LoadImage
# ---------------------------------------------------------------------------


class _EmptyVideoComponents:
    """Stands in for the pyav decode result of ``InputImpl.VideoFromFile``.

    pyav decodes animated WEBP on its own, so the ``Image.open`` branch
    of ``LoadImage.load_image`` is only reached when the video reader
    comes back empty.
    """

    images = torch.zeros((0, 8, 8, 3))
    alpha = None


class _NoVideo:
    def __init__(self, path):
        self.path = path

    def get_components(self):
        return _EmptyVideoComponents()


@pytest.fixture
def load_image_node(monkeypatch, tmp_path: Path):
    """``nodes.LoadImage`` wired to a temporary multi-frame image."""
    image_path = tmp_path / "anim.webp"
    _save_animated_webp(image_path)
    monkeypatch.setattr(
        nodes.folder_paths,
        "get_annotated_filepath",
        lambda image: str(image_path),
    )
    monkeypatch.setattr(
        nodes,
        "InputImpl",
        type("InputImpl", (), {"VideoFromFile": _NoVideo}),
    )
    return nodes.LoadImage()


def test_load_image_closes_the_image_it_opened(load_image_node, opened_images) -> None:
    """``LoadImage.load_image`` must release the file handle of the image
    it opened after iterating over all of its frames."""
    image, mask = load_image_node.load_image("anim.webp")

    assert image.shape[0] == 3
    assert mask.shape[0] == 3
    assert len(opened_images) == 1
    assert opened_images[0].was_closed, (
        "LoadImage.load_image did not close the image it opened"
    )
    assert opened_images[0].handle_released, (
        "LoadImage.load_image left the file handle of the image it opened open"
    )


def test_load_image_closes_the_image_it_opened_on_failure(
    load_image_node, opened_images, monkeypatch
) -> None:
    """The ``finally`` block must also run when the frame loop raises."""
    monkeypatch.setattr(nodes.ImageSequence, "Iterator", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        load_image_node.load_image("anim.webp")

    assert len(opened_images) == 1
    assert opened_images[0].was_closed, (
        "LoadImage.load_image did not close the image it opened after a failure"
    )
    assert opened_images[0].handle_released, (
        "LoadImage.load_image left the file handle open after a failure"
    )
