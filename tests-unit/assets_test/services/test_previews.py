import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from app.assets import previews


@pytest.mark.parametrize(("height", "skipped"), [(4000, False), (4001, True)])
def test_upload_decodes_are_bounded(tmp_path, height, skipped):
    """A decode costs ~30 MB per megapixel, and running out of memory can't be caught."""
    # Over 17 megapixels is skipped from the header, before decoding.
    with (
        patch.object(previews, "read_exr_windows", return_value=[(4250, height)]),
        patch.object(previews.av, "open", side_effect=RuntimeError("decoded")) as decode,
    ):
        with pytest.raises(previews.PreviewSkipped if skipped else RuntimeError):
            previews._decode_for_preview("a.exr")
    assert decode.called is not skipped

    # At most two decode at once, and the rest still get their preview.
    running, peak, lock = 0, 0, threading.Lock()

    def make(path):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return b"webp", 2, 2

    async def uploads():
        # A semaphore made inside this loop, at the module's limit, so the module's stays unbound.
        with patch.object(previews, "_DECODE_SLOTS", asyncio.Semaphore(previews._DECODE_SLOTS._value)):
            await asyncio.gather(*(previews.generate_upload_preview(f"id{i}", str(tmp_path / f"{i}.exr"), None) for i in range(6)))

    with patch.object(previews, "_make_preview", make), patch.object(previews, "_store_and_link", return_value="linked") as store:
        asyncio.run(uploads())
    assert peak == 2 and store.call_count == 6
