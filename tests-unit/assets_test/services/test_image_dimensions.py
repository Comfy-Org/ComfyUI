"""Tests for the image_dimensions service."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.assets.services.image_dimensions import extract_image_dimensions


def _make_png(path: Path, size: tuple[int, int]) -> Path:
    img = Image.new("RGB", size, color=(123, 45, 67))
    img.save(path, format="PNG")
    return path


def _make_jpeg(path: Path, size: tuple[int, int]) -> Path:
    img = Image.new("RGB", size, color=(10, 20, 30))
    img.save(path, format="JPEG", quality=80)
    return path


class TestExtractImageDimensions:
    def test_extracts_png_dimensions(self, tmp_path: Path):
        f = _make_png(tmp_path / "rect.png", (320, 240))

        result = extract_image_dimensions(str(f), mime_type="image/png")

        assert result == {"kind": "image", "width": 320, "height": 240}

    def test_extracts_jpeg_dimensions(self, tmp_path: Path):
        f = _make_jpeg(tmp_path / "shot.jpg", (1920, 1080))

        result = extract_image_dimensions(str(f), mime_type="image/jpeg")

        assert result == {"kind": "image", "width": 1920, "height": 1080}

    def test_works_when_mime_type_is_none(self, tmp_path: Path):
        f = _make_png(tmp_path / "no_mime.png", (50, 100))

        result = extract_image_dimensions(str(f), mime_type=None)

        assert result == {"kind": "image", "width": 50, "height": 100}

    def test_skips_non_image_mime_without_touching_file(self, tmp_path: Path):
        # Path doesn't need to exist — non-image MIME short-circuits.
        result = extract_image_dimensions(
            str(tmp_path / "model.safetensors"),
            mime_type="application/octet-stream",
        )

        assert result is None

    @pytest.mark.parametrize(
        "mime",
        ["application/json", "text/plain", "video/mp4", "audio/mpeg"],
    )
    def test_skips_all_non_image_mime_types(self, tmp_path: Path, mime: str):
        f = tmp_path / "file.bin"
        f.write_bytes(b"\x00\x01\x02")

        assert extract_image_dimensions(str(f), mime_type=mime) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        result = extract_image_dimensions(
            str(tmp_path / "does_not_exist.png"), mime_type="image/png"
        )

        assert result is None

    def test_returns_none_for_corrupt_image(self, tmp_path: Path):
        f = tmp_path / "corrupt.png"
        f.write_bytes(b"not actually a png file")

        result = extract_image_dimensions(str(f), mime_type="image/png")

        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.png"
        f.write_bytes(b"")

        result = extract_image_dimensions(str(f), mime_type="image/png")

        assert result is None
