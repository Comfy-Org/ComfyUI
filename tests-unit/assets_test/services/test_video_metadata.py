"""Tests for the video_metadata service."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.assets.services.media_metadata import extract_media_metadata
from app.assets.services.video_metadata import extract_video_metadata


def _make_video(
    path: Path,
    width: int = 64,
    height: int = 48,
    frames: int = 12,
    fps: int = 8,
    codec: str = "libx264",
) -> Path:
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((height, width, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _make_mp4(
    path: Path, width: int = 64, height: int = 48, frames: int = 12, fps: int = 8
) -> Path:
    return _make_video(path, width=width, height=height, frames=frames, fps=fps)


class TestExtractVideoMetadata:
    def test_extracts_stream_metadata(self, tmp_path: Path):
        f = _make_mp4(tmp_path / "clip.mp4", width=64, height=48, frames=12, fps=8)

        result = extract_video_metadata(str(f), mime_type="video/mp4")

        assert result is not None
        assert result["kind"] == "video"
        assert result["width"] == 64
        assert result["height"] == 48
        assert result["frame_count"] == 12
        assert result["fps"] == pytest.approx(8.0)
        assert result["duration"] == pytest.approx(12 / 8, abs=0.1)

    def test_works_when_mime_type_is_none(self, tmp_path: Path):
        f = _make_mp4(tmp_path / "no_mime.mp4")

        result = extract_video_metadata(str(f), mime_type=None)

        assert result is not None
        assert result["kind"] == "video"

    @pytest.mark.parametrize(
        "mime",
        ["application/json", "text/plain", "image/png", "audio/mpeg"],
    )
    def test_skips_non_video_mime_types(self, tmp_path: Path, mime: str):
        result = extract_video_metadata(
            str(tmp_path / "untouched.mp4"), mime_type=mime
        )

        assert result is None

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        result = extract_video_metadata(
            str(tmp_path / "does_not_exist.mp4"), mime_type="video/mp4"
        )

        assert result is None

    def test_returns_none_for_corrupt_video(self, tmp_path: Path):
        f = tmp_path / "corrupt.mp4"
        f.write_bytes(b"not actually an mp4 file")

        result = extract_video_metadata(str(f), mime_type="video/mp4")

        assert result is None

    def test_webm_duration_from_container_without_frame_count(self, tmp_path: Path):
        f = _make_video(tmp_path / "clip.webm", frames=12, fps=8, codec="libvpx-vp9")

        result = extract_video_metadata(str(f), mime_type="video/webm")

        assert result is not None
        assert result["duration"] == pytest.approx(12 / 8, abs=0.2)
        assert "frame_count" not in result


class TestExtractMediaMetadata:
    def test_dispatches_video_mime_to_video_extractor(self, tmp_path: Path):
        f = _make_mp4(tmp_path / "clip.mp4")

        result = extract_media_metadata(str(f), mime_type="video/mp4")

        assert result is not None
        assert result["kind"] == "video"

    def test_dispatches_image_mime_to_image_extractor(self, tmp_path: Path):
        Image = pytest.importorskip("PIL.Image")

        f = tmp_path / "img.png"
        Image.new("RGB", (32, 16)).save(f, format="PNG")

        result = extract_media_metadata(str(f), mime_type="image/png")

        assert result == {"kind": "image", "width": 32, "height": 16}

    def test_returns_none_without_mime_type(self, tmp_path: Path):
        f = _make_mp4(tmp_path / "clip.mp4")

        assert extract_media_metadata(str(f), mime_type=None) is None

    def test_returns_none_for_non_media_mime(self, tmp_path: Path):
        f = tmp_path / "file.bin"
        f.write_bytes(b"\x00")

        assert extract_media_metadata(str(f), mime_type="text/plain") is None


class TestIngestStoresVideoMetadata:
    def test_register_file_in_place_stores_video_metadata(
        self, mock_create_session, temp_dir: Path, session
    ):
        from app.assets.database.models import AssetReference
        from app.assets.services.ingest import _ingest_file_from_path

        f = _make_mp4(temp_dir / "clip.mp4", width=64, height=48)

        result = _ingest_file_from_path(
            abs_path=str(f),
            asset_hash="blake3:video123",
            size_bytes=f.stat().st_size,
            mtime_ns=1234567890000000000,
            mime_type="video/mp4",
        )

        assert result.reference_id is not None
        ref = session.query(AssetReference).one()
        meta = ref.system_metadata or {}
        assert meta["kind"] == "video"
        assert meta["width"] == 64
        assert meta["height"] == 48
        assert meta["frame_count"] == 12
        assert meta["fps"] == pytest.approx(8.0)

    def test_register_existing_asset_backfills_metadata_from_sibling(
        self, mock_create_session, temp_dir: Path, session
    ):
        from app.assets.database.models import AssetReference
        from app.assets.services.ingest import (
            _ingest_file_from_path,
            _register_existing_asset,
        )

        f = _make_mp4(temp_dir / "clip.mp4", width=64, height=48)
        _ingest_file_from_path(
            abs_path=str(f),
            asset_hash="blake3:videosibling",
            size_bytes=f.stat().st_size,
            mtime_ns=1234567890000000000,
            mime_type="video/mp4",
        )

        result = _register_existing_asset(
            asset_hash="blake3:videosibling",
            name="copy.mp4",
            mime_type="video/mp4",
        )

        assert result.created is True
        session.expire_all()
        ref = session.query(AssetReference).filter_by(name="copy.mp4").one()
        meta = ref.system_metadata or {}
        assert meta["kind"] == "video"
        assert meta["width"] == 64
        assert meta["height"] == 48
        assert meta["frame_count"] == 12
        assert meta["fps"] == pytest.approx(8.0)
        assert "duration" in meta
