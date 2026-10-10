from unittest.mock import patch

import pytest

from app.assets import previews


@pytest.mark.parametrize(("height", "skipped"), [(4000, False), (4001, True)])
def test_uploads_over_17_megapixels_are_skipped_without_decoding(height, skipped):
    """A decode costs ~30 MB per megapixel, and running out of memory can't be caught."""
    with (
        patch.object(previews, "read_exr_windows", return_value=[(4250, height)]),
        patch.object(previews.av, "open", side_effect=RuntimeError("decoded")) as decode,
    ):
        with pytest.raises(previews.PreviewSkipped if skipped else RuntimeError):
            previews._decode_for_preview("a.exr")

    assert decode.called is not skipped
