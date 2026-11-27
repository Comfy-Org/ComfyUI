import numpy as np
import pytest

from comfy.audio_encoders.sheetsage2_abc import MelodyVoiceError, _notes_to_arr


TIMES = np.arange(0.0, 2.0, 0.25)


def test_note_shorter_than_the_grid_is_skipped():
    # Audio cut mid-note leaves a sliver like this right before the end.
    arr = _notes_to_arr([[0.25, 0.5, 60], [1.26, 1.27, 67]], TIMES, "Ins")
    assert arr[1] == 60 * 2 + 3
    assert np.count_nonzero(arr) == 1


def test_overlapping_notes_still_fail():
    with pytest.raises(MelodyVoiceError):
        _notes_to_arr([[0.25, 1.0, 60], [0.5, 0.75, 62]], TIMES, "Ins")
