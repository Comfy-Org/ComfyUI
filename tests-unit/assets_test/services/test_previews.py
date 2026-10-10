import asyncio
import io
import os
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

from .preview_helpers import write_exr


@pytest.fixture
def previews_dir(tmp_path: Path):
    before = folder_paths.get_previews_directory()
    folder_paths.set_previews_directory(str(tmp_path / "previews"))
    yield tmp_path / "previews"
    folder_paths.set_previews_directory(before)


@pytest.fixture(autouse=True)
def exr_mime():
    from utils.mime_types import init_mime_types

    init_mime_types()


def _parent(session, path: Path, mime_type: str | None = "image/x-exr") -> Asset:
    content = create_content(session, str(path))
    record = create_record(session, content.id, path.name, mime_type=mime_type)
    session.commit()
    return record


def _events(caplog, name: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if f"[assets-event] {name}" in r.getMessage()]


def _generate(asset_id: str, path) -> str | None:
    return asyncio.run(previews.generate_upload_preview(asset_id, str(path), None))


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


def _duplicate_data_window(data: bytes) -> bytes:
    key = b"dataWindow\x00box2i\x00"
    at = data.find(key)
    end = at + len(key) + 4 + 16
    return data[:end] + data[at:end] + data[end:]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b[:20],  # truncated mid-attribute
        lambda b: b.replace(b"displayWindow\x00box2i\x00\x10\x00\x00\x00", b"displayWindow\x00box2i\x00\xff\xff\xff\xff"),  # negative size
        lambda b: b.replace(b"displayWindow", b"displayWindoX"),  # missing window
        lambda b: b.replace(b"\x00" * 8 + b"\x3f\x00\x00\x00", b"\x00" * 8 + b"\xf0\xff\xff\xff", 1),  # inverted window
        lambda b: _duplicate_data_window(b),  # decoders differ on which copy applies
    ],
)
def test_a_malformed_exr_header_reads_as_none(tmp_path, mutate):
    path = write_exr(tmp_path / "a.exr", 64, 48)
    path.write_bytes(mutate(path.read_bytes()))

    assert read_exr_windows(str(path)) is None


# --- decoding an uploaded EXR ---


@pytest.mark.parametrize(("width", "height"), [(64, 48), (1600, 1000)])
def test_exr_preview_is_at_most_one_megapixel_and_keeps_aspect(tmp_path, width, height):
    image = previews._decode_for_preview(str(write_exr(tmp_path / "a.exr", width, height)))

    assert image.width * image.height <= 1_000_000
    assert abs(image.width / image.height - width / height) < 0.01
    if width * height <= 1_000_000:
        assert image.size == (width, height), "never upscaled, never needlessly shrunk"


def test_exr_preview_tonemaps_linear_to_srgb(tmp_path):
    image = previews._decode_for_preview(str(write_exr(tmp_path / "a.exr", 8, 8, value=(4.0, 0.5, 0.0))))

    r, g, b = image.getpixel((4, 4))
    assert r == 255, "over-range clamps to white"
    assert 185 <= g <= 190, "linear 0.5 is sRGB ~188, not 128"
    assert b == 0


def test_an_exr_with_alpha_keeps_it(tmp_path):
    webp, _, _ = previews._make_preview(str(write_exr(tmp_path / "a.exr", 8, 8, value=(0.5, 0.5, 0.5, 0.25))))

    image = Image.open(io.BytesIO(webp))
    assert image.mode == "RGBA"
    assert abs(image.getpixel((4, 4))[3] - 64) <= 3


def test_exr_preview_shows_the_display_window(tmp_path):
    image = previews._decode_for_preview(str(write_exr(tmp_path / "a.exr", 64, 48, display_window=(8, 8, 55, 39))))

    assert image.size == (48, 32)


def test_a_half_float_luminance_exr_keeps_its_darks(tmp_path):
    path = tmp_path / "y.exr"
    gray = np.full((8, 8), 0.0627, np.float16)  # sRGB ~70/255
    codec = av.CodecContext.create("exr", "w")
    codec.width, codec.height, codec.pix_fmt = 8, 8, "grayf32le"
    codec.time_base = Fraction(1, 1)
    codec.options = {"format": "half"}
    frame = av.VideoFrame.from_ndarray(gray.astype(np.float32), format="grayf32le")
    frame.pts, frame.time_base = 0, codec.time_base
    path.write_bytes(b"".join(bytes(p) for p in list(codec.encode(frame)) + list(codec.encode(None))))

    image = previews._decode_for_preview(str(path))

    assert 66 <= image.getpixel((4, 4))[0] <= 74, "linear 0.0627 is sRGB ~70, not near black"


@pytest.mark.parametrize(("height", "skipped"), [(4000, False), (4001, True)])
def test_uploads_over_17_megapixels_are_skipped_without_decoding(height, skipped):
    with (
        patch.object(previews, "read_exr_windows", return_value=[(4250, height)]),
        patch.object(previews.av, "open", side_effect=RuntimeError("decoded")) as decode,
    ):
        with pytest.raises(previews.PreviewSkipped if skipped else RuntimeError):
            previews._decode_for_preview("a.exr")

    assert decode.called is not skipped


def test_size_guard_counts_a_data_window_bigger_than_the_display_window(tmp_path):
    path = write_exr(tmp_path / "a.exr", 64, 48, display_window=(0, 0, 7, 7))

    with patch.object(previews, "PREVIEW_MAX_SOURCE_PIXELS", 1000), pytest.raises(previews.PreviewSkipped):
        previews._decode_for_preview(str(path))


# --- storing and linking ---


def test_a_generated_preview_is_stored_tagged_and_linked(session, mock_create_session, previews_dir, tmp_path, caplog):
    caplog.set_level("INFO")
    parent = _parent(session, write_exr(tmp_path / "frame.exr", 64, 48))

    linked = _generate(parent.id, tmp_path / "frame.exr")

    session.expire_all()
    preview_id = session.get(Asset, parent.id).preview_id
    assert linked == preview_id is not None
    preview = session.get(Asset, preview_id)
    assert fetch_record_tags(session, preview.id) == ["preview"]
    assert preview.mime_type == "image/webp"
    assert session.get(AssetContent, preview.content_id).path == str(previews_dir / f"{preview.name}")
    assert len(_events(caplog, "previews.generated")) == 1


def test_a_parent_deleted_before_storing_leaves_nothing_behind(session, mock_create_session, previews_dir, tmp_path, caplog):
    caplog.set_level("INFO")
    parent = _parent(session, write_exr(tmp_path / "frame.exr", 64, 48))
    session.delete(parent)
    session.commit()

    assert _generate(parent.id, tmp_path / "frame.exr") is None

    assert not previews_dir.exists() or not any(previews_dir.iterdir())
    assert session.scalars(select(Asset).where(Asset.mime_type == "image/webp")).first() is None
    assert not _events(caplog, "previews.generation_failed"), "a gone parent is not a failure"


def test_a_preview_set_while_generating_is_kept(session, mock_create_session, previews_dir, tmp_path):
    parent = _parent(session, write_exr(tmp_path / "frame.exr", 64, 48))
    chosen = _parent(session, tmp_path / "frame.exr", mime_type="image/png")
    parent.preview_id = chosen.id  # a client's PUT landed first
    session.commit()

    assert _generate(parent.id, tmp_path / "frame.exr") is None

    session.expire_all()
    assert session.get(Asset, parent.id).preview_id == chosen.id
    assert not previews_dir.exists() or not any(previews_dir.iterdir())


def test_only_exrs_get_a_preview(tmp_path):
    with patch.object(previews, "_make_preview") as make:
        assert _generate("id", tmp_path / "still.tiff") is None

    make.assert_not_called()


def test_a_preview_the_client_set_is_kept_without_decoding(tmp_path):
    with patch.object(previews, "_make_preview") as make:
        assert asyncio.run(previews.generate_upload_preview("id", str(tmp_path / "a.exr"), "client-preview")) == "client-preview"

    make.assert_not_called()


@pytest.mark.parametrize(
    ("make_error", "store_error", "reason"),
    [
        (previews.PreviewSkipped("too_large"), None, "too_large"),
        (ValueError("bad file"), None, "decode_failed"),
        (None, OSError("disk"), "write_failed"),
    ],
)
def test_failures_are_logged_and_leave_no_preview(session, mock_create_session, previews_dir, tmp_path, caplog, make_error, store_error, reason):
    caplog.set_level("INFO")
    parent = _parent(session, write_exr(tmp_path / "frame.exr", 8, 8))
    with (
        patch.object(previews, "_make_preview", side_effect=make_error, wraps=None if make_error else previews._make_preview),
        patch.object(previews, "_store_and_link", side_effect=store_error, wraps=None if store_error else previews._store_and_link),
    ):
        assert _generate(parent.id, tmp_path / "frame.exr") is None

    failed = _events(caplog, "previews.generation_failed")
    assert len(failed) == 1 and f"reason={reason}" in failed[0]
    session.expire_all()
    assert session.get(Asset, parent.id).preview_id is None
    assert not previews_dir.exists() or not any(previews_dir.iterdir())


def test_the_event_loop_keeps_running_while_an_upload_decodes(session, mock_create_session, previews_dir, tmp_path):
    parent = _parent(session, write_exr(tmp_path / "frame.exr", 8, 8))
    real = previews._make_preview

    def slow(path):
        time.sleep(0.5)
        return real(path)

    async def scenario():
        ticks = []

        async def tick():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        await previews.generate_upload_preview(parent.id, str(tmp_path / "frame.exr"), None)
        await asyncio.sleep(0.05)  # one more tick, so a stall after the decode shows as a gap
        ticker.cancel()
        return ticks

    with patch.object(previews, "_make_preview", slow):
        ticks = asyncio.run(scenario())

    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert ticks[-1] - ticks[0] >= 0.4 and max(gaps) < 0.4, "the loop keeps running while a preview decodes"


def test_previews_live_beside_the_other_roots_by_default():
    assert folder_paths.get_previews_directory() == os.path.join(folder_paths.base_path, "previews")


def test_a_failed_registration_leaves_no_preview_file(session, mock_create_session, previews_dir, tmp_path):
    parent = _parent(session, write_exr(tmp_path / "frame.exr", 8, 8))

    with patch.object(previews, "create_record", side_effect=RuntimeError("db")):
        assert _generate(parent.id, tmp_path / "frame.exr") is None

    assert not any(previews_dir.iterdir())
    session.expire_all()
    assert session.get(Asset, parent.id).preview_id is None



def test_at_most_two_uploads_decode_at_once(tmp_path):
    import threading

    running, peak, lock = 0, 0, threading.Lock()

    def make(path):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.1)
        with lock:
            running -= 1
        return b"webp", 2, 2

    async def uploads():
        await asyncio.gather(*(previews.generate_upload_preview(f"id{i}", str(tmp_path / f"{i}.exr"), None) for i in range(6)))

    async def uploads_with_fresh_slots():
        # A semaphore made inside the loop, so the module's stays unbound to this test's loop.
        with patch.object(previews, "_DECODE_SLOTS", asyncio.Semaphore(previews._DECODE_SLOTS._value)):
            await uploads()

    with patch.object(previews, "_make_preview", make), patch.object(previews, "_store_and_link", return_value="linked") as store:
        asyncio.run(uploads_with_fresh_slots())

    assert peak == 2, "each decode can hold ~0.5 GB, and waiting uploads must not hold executor threads"
    assert store.call_count == 6, "every upload still gets its preview"


def test_the_event_loop_keeps_running_while_an_upload_preview_is_stored(tmp_path):
    def slow_store(*args):
        time.sleep(0.5)

    async def scenario():
        ticks = []

        async def tick():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        await previews.generate_upload_preview("id", str(tmp_path / "a.exr"), None)
        await asyncio.sleep(0.05)
        ticker.cancel()
        return ticks

    with patch.object(previews, "_make_preview", return_value=(b"webp", 2, 2)), patch.object(previews, "_store_and_link", slow_store):
        ticks = asyncio.run(scenario())

    assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.4, "storing a preview doesn't block the server"

