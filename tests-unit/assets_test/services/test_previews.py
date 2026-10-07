import asyncio
import os
import struct
import threading
import time
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pytest
from PIL import Image
from sqlalchemy import select

import folder_paths
from app.assets import previews
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record, fetch_record_tags
from app.assets.services.image_dimensions import extract_image_dimensions, read_exr_windows
from comfy_execution import preview_generators


def write_exr(path: Path, width: int, height: int, value=(1.0, 0.5, 0.25), display_window=None) -> Path:
    """An EXR the way SaveImageAdvanced writes one (PyAV, uncompressed half)."""
    rgb = np.empty((height, width, 3), np.float32)
    rgb[...] = value
    codec = av.CodecContext.create("exr", "w")
    codec.width, codec.height, codec.pix_fmt = width, height, "gbrpf32le"
    codec.time_base = Fraction(1, 1)
    codec.options = {"format": "half"}
    frame = av.VideoFrame.from_ndarray(rgb, format="gbrpf32le")
    frame.pts = 0
    frame.time_base = codec.time_base
    data = bytearray(b"".join(bytes(p) for p in list(codec.encode(frame)) + list(codec.encode(None))))
    if display_window is not None:
        key = b"displayWindow\x00box2i\x00"
        at = data.find(key) + len(key) + 4
        struct.pack_into("<4i", data, at, *display_window)
    path.write_bytes(bytes(data))
    return path


@pytest.fixture
def previews_dir(tmp_path: Path):
    before = folder_paths.get_previews_directory()
    folder_paths.set_previews_directory(str(tmp_path / "previews"))
    yield tmp_path / "previews"
    folder_paths.set_previews_directory(before)


@pytest.fixture
def exr_mime():
    with patch.object(previews, "preview_mime_type", lambda path: "image/x-exr" if path.lower().endswith(".exr") else None):
        yield


def _parent(session, path: Path, mime_type: str | None = "image/x-exr") -> Asset:
    content = create_content(session, str(path))
    record = create_record(session, content.id, path.name, mime_type=mime_type)
    session.commit()
    return record


def _events(caplog, name: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if f"[assets-event] {name}" in r.getMessage()]


# --- header reader and dimensions ---


def test_header_reader_reports_display_and_data_windows(tmp_path):
    path = write_exr(tmp_path / "a.exr", 64, 48, display_window=(8, 8, 55, 39))

    assert read_exr_windows(str(path)) == ((48, 32), (64, 48))


def test_header_reader_rejects_non_exr(tmp_path):
    path = tmp_path / "a.exr"
    path.write_bytes(b"not an exr at all")

    assert read_exr_windows(str(path)) is None


def test_exr_dimensions_are_the_display_window(tmp_path):
    path = write_exr(tmp_path / "a.exr", 64, 48, display_window=(-8, -8, 71, 55))

    assert extract_image_dimensions(str(path), mime_type="image/x-exr") == {"kind": "image", "width": 80, "height": 64}


# --- Core's EXR generator ---


@pytest.mark.parametrize(("width", "height"), [(64, 48), (1600, 1000)])
def test_exr_preview_is_at_most_one_megapixel_and_keeps_aspect(tmp_path, width, height):
    path = write_exr(tmp_path / "a.exr", width, height)

    image = previews.ExrPreviewGenerator().generate(str(path), previews.PREVIEW_MAX_PIXELS)

    assert image.width * image.height <= previews.PREVIEW_MAX_PIXELS
    assert abs(image.width / image.height - width / height) < 0.01
    if width * height <= previews.PREVIEW_MAX_PIXELS:
        assert image.size == (width, height), "never upscaled, never needlessly shrunk"


def test_exr_preview_tonemaps_linear_to_srgb(tmp_path):
    path = write_exr(tmp_path / "a.exr", 8, 8, value=(4.0, 0.5, 0.0))

    image = previews.ExrPreviewGenerator().generate(str(path), previews.PREVIEW_MAX_PIXELS)

    r, g, b = image.getpixel((4, 4))
    assert r == 255, "over-range clamps to white"
    assert 185 <= g <= 190, "linear 0.5 is sRGB ~188, not 128"
    assert b == 0


def test_exr_preview_shows_the_display_window(tmp_path):
    path = write_exr(tmp_path / "a.exr", 64, 48, display_window=(8, 8, 55, 39))

    image = previews.ExrPreviewGenerator().generate(str(path), previews.PREVIEW_MAX_PIXELS)

    assert image.size == (48, 32)


def test_oversized_exr_is_skipped_without_decoding(tmp_path):
    path = write_exr(tmp_path / "a.exr", 64, 48)

    with (
        patch.object(previews, "PREVIEW_MAX_SOURCE_PIXELS", 1000),
        patch.object(previews, "_decode_for_preview") as decode,
    ):
        with pytest.raises(previews.PreviewSkipped) as skipped:
            previews.ExrPreviewGenerator().generate(str(path), previews.PREVIEW_MAX_PIXELS)

    assert skipped.value.reason == "too_large"
    decode.assert_not_called()


def test_size_guard_counts_a_data_window_bigger_than_the_display_window(tmp_path):
    path = write_exr(tmp_path / "a.exr", 64, 48, display_window=(0, 0, 7, 7))

    with patch.object(previews, "PREVIEW_MAX_SOURCE_PIXELS", 1000), pytest.raises(previews.PreviewSkipped):
        previews.ExrPreviewGenerator().generate(str(path), previews.PREVIEW_MAX_PIXELS)


# --- encoding a generator's image ---


@pytest.mark.parametrize("mode", ["I;16", "F", "I"])
def test_wide_images_from_a_generator_are_refused_not_clipped(mode):
    with pytest.raises(TypeError):
        previews._encode(Image.new(mode, (4, 4)))


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_eight_bit_images_encode_to_webp(mode):
    webp, width, height = previews._encode(Image.new(mode, (2000, 1000)))

    assert webp[8:12] == b"WEBP"
    assert width * height <= previews.PREVIEW_MAX_PIXELS


# --- make, store and link ---


def test_a_generated_preview_is_stored_tagged_and_linked(session, mock_create_session, previews_dir, exr_mime, tmp_path, caplog):
    caplog.set_level("INFO")
    parent = _parent(session, write_exr(tmp_path / "frame.exr", 64, 48))

    linked = asyncio.run(previews.generate_previews([(parent.id, str(tmp_path / "frame.exr"))], "output"))

    session.expire_all()
    preview_id = session.get(Asset, parent.id).preview_id
    assert linked == {parent.id: preview_id}
    preview = session.get(Asset, preview_id)
    assert fetch_record_tags(session, preview.id) == ["preview"]
    assert preview.mime_type == "image/webp"
    assert session.get(AssetContent, preview.content_id).path == str(previews_dir / f"{preview.name}")
    assert len(_events(caplog, "previews.generated")) == 1


def test_a_parent_deleted_during_generation_leaves_nothing_behind(session, mock_create_session, previews_dir, exr_mime, tmp_path):
    parent = _parent(session, write_exr(tmp_path / "frame.exr", 64, 48))
    session.delete(parent)
    session.commit()

    linked = asyncio.run(previews.generate_previews([(parent.id, str(tmp_path / "frame.exr"))], "upload"))

    assert linked == {}
    assert not any(previews_dir.glob("*")) if previews_dir.exists() else True
    assert session.scalars(select(Asset).where(Asset.mime_type == "image/webp")).first() is None


class _Sleeper:
    mime_types = ("image/x-test-slow",)

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.finished = threading.Event()

    def generate(self, source_path, max_pixels):
        time.sleep(self.seconds)
        self.finished.set()
        return Image.new("RGB", (4, 4))


def test_a_slow_generator_does_not_hold_the_caller_past_the_deadline(session, mock_create_session, previews_dir, tmp_path, caplog):
    caplog.set_level("INFO")
    source = tmp_path / "slow.tst"
    source.write_bytes(b"x")
    parent = _parent(session, source, mime_type="image/x-test-slow")
    sleeper = _Sleeper(1.5)
    preview_generators.register_preview_generator(sleeper)
    try:
        with (
            patch.object(previews, "preview_mime_type", lambda path: "image/x-test-slow"),
            patch.object(previews, "preview_deadline_seconds", lambda count: 0.3),
        ):
            started = time.monotonic()
            linked = asyncio.run(previews.generate_previews([(parent.id, str(source))], "output"))
            waited = time.monotonic() - started
            assert sleeper.finished.wait(5)
            time.sleep(0.1)
    finally:
        preview_generators.unregister_preview_generator(sleeper)

    assert linked == {}
    assert waited < 1.0, "the deadline, not the generator, decides when the caller moves on"
    assert not previews_dir.exists() or not any(previews_dir.iterdir()), "a late result is never stored"
    session.expire_all()
    assert session.get(Asset, parent.id).preview_id is None
    assert any("reason=timeout" in e for e in _events(caplog, "previews.generation_failed"))


def test_storing_stops_when_the_deadline_passes(session, mock_create_session, previews_dir, exr_mime, tmp_path, caplog):
    caplog.set_level("INFO")
    frames = [write_exr(tmp_path / f"f{i}.exr", 8, 8) for i in range(2)]
    parents = [_parent(session, f) for f in frames]
    store = previews._store_and_link

    def slow_store(*args):
        time.sleep(0.4)
        return store(*args)

    with (
        patch.object(previews, "preview_deadline_seconds", lambda count: 0.3),
        patch.object(previews, "_store_and_link", slow_store),
    ):
        linked = asyncio.run(previews.generate_previews([(p.id, str(f)) for p, f in zip(parents, frames)], "output"))

    assert len(linked) <= 1, "nothing is stored once the deadline has passed"
    assert any("reason=timeout" in e for e in _events(caplog, "previews.generation_failed"))


def test_the_event_loop_keeps_running_while_previews_generate(session, mock_create_session, previews_dir, tmp_path):
    source = tmp_path / "slow.tst"
    source.write_bytes(b"x")
    parent = _parent(session, source, mime_type="image/x-test-slow")
    sleeper = _Sleeper(0.5)
    preview_generators.register_preview_generator(sleeper)

    async def scenario():
        ticks = 0

        async def tick():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        await previews.generate_previews([(parent.id, str(source))], "output")
        ticker.cancel()
        return ticks

    try:
        with patch.object(previews, "preview_mime_type", lambda path: "image/x-test-slow"):
            ticks = asyncio.run(scenario())
    finally:
        preview_generators.unregister_preview_generator(sleeper)

    assert ticks >= 20, "an async node pending in the same loop must not stall while previews generate"


class _Raises:
    mime_types = ("image/x-test-raises",)

    class AVeryLongThirdPartyExceptionNameThatGoesOnAndOnWellPastTheEventLogLimit(Exception):
        pass

    def generate(self, source_path, max_pixels):
        raise self.AVeryLongThirdPartyExceptionNameThatGoesOnAndOnWellPastTheEventLogLimit()


class _Declines:
    mime_types = ("image/x-test-declines",)

    def generate(self, source_path, max_pixels):
        return None


def test_a_raising_generator_is_logged_and_a_declining_one_is_not(session, mock_create_session, previews_dir, tmp_path, caplog):
    caplog.set_level("INFO")
    raising, declining = tmp_path / "a.raise", tmp_path / "b.decline"
    raising.write_bytes(b"x")
    declining.write_bytes(b"x")
    parents = [_parent(session, raising, None), _parent(session, declining, None)]
    mimes = {str(raising): "image/x-test-raises", str(declining): "image/x-test-declines"}
    generators = [_Raises(), _Declines()]
    for g in generators:
        preview_generators.register_preview_generator(g)
    try:
        with patch.object(previews, "preview_mime_type", lambda path: mimes.get(path)):
            linked = asyncio.run(
                previews.generate_previews([(parents[0].id, str(raising)), (parents[1].id, str(declining))], "upload")
            )
    finally:
        for g in generators:
            preview_generators.unregister_preview_generator(g)

    assert linked == {}
    failed = _events(caplog, "previews.generation_failed")
    assert len(failed) == 1, "declining is a normal answer, not a failure"
    assert "reason=decode_failed" in failed[0] and "format=other" in failed[0]
    error_type = failed[0].split("error_type=")[1].split()[0]
    assert len(error_type) <= 64, "an arbitrary exception name must still make a valid event"


# --- registry ---


class _ExrOverride:
    mime_types = ("image/x-exr",)

    def generate(self, source_path, max_pixels):
        return Image.new("RGB", (2, 2))


def test_a_registered_generator_replaces_cores_and_unregistering_restores_it():
    core = preview_generators.get_preview_generator("image/x-exr")
    override = _ExrOverride()

    asyncio.run(_register(override))
    try:
        assert preview_generators.get_preview_generator("image/x-exr") is override
    finally:
        asyncio.run(_unregister(override))

    assert preview_generators.get_preview_generator("image/x-exr") is core


def test_registration_order_against_core_does_not_matter():
    override = _ExrOverride()
    asyncio.run(_register(override))
    try:
        # Core registering its own again (e.g. assets starting after custom nodes load) must not win.
        preview_generators.set_core_preview_generator(previews.ExrPreviewGenerator())
        assert preview_generators.get_preview_generator("image/x-exr") is override
    finally:
        asyncio.run(_unregister(override))


async def _register(generator):
    from comfy_api.latest import ComfyAPI

    await ComfyAPI().previews.register_generator(generator)


async def _unregister(generator):
    from comfy_api.latest import ComfyAPI

    await ComfyAPI().previews.unregister_generator(generator)


def test_lookup_uses_the_path_not_an_uploader_supplied_type(tmp_path):
    from utils.mime_types import init_mime_types

    init_mime_types()
    assert previews.preview_mime_type(str(tmp_path / "frame.exr")) == "image/x-exr"
    assert previews.has_preview_generator(str(tmp_path / "frame.exr"))
    assert not previews.has_preview_generator(str(tmp_path / "frame.png"))


def test_workers_are_daemon_threads():
    preview_generators.submit_preview_job(lambda: None).result(timeout=5)

    workers = [t for t in threading.enumerate() if t.name.startswith("preview-worker-")]
    assert workers and all(t.daemon for t in workers), "a hung generator must not block shutdown"
    assert len(workers) == preview_generators.PREVIEW_WORKERS


def test_previews_live_beside_the_other_roots_by_default():
    assert folder_paths.get_previews_directory() == os.path.join(folder_paths.base_path, "previews")
