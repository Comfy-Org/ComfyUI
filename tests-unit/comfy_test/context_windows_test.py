from types import SimpleNamespace

from comfy.context_windows import create_windows_uniform_standard, shift_window_to_end


def test_shift_window_to_end_uses_window_max():
    # A dilated (strided) window wraps around, so it is not monotonic and its last
    # element is not its largest. shift_window_to_end must slide the window so its
    # maximum lands on num_frames - 1 and every index stays in range -- shifting
    # based on window[-1] instead over-shifts and produces out-of-bounds indices.
    num_frames = 48
    window = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 0, 4, 8, 12]
    shift_window_to_end(window, num_frames)
    assert max(window) == num_frames - 1
    assert all(0 <= i < num_frames for i in window)


def test_create_windows_uniform_standard_strided_indices_in_range():
    # WAN/LTXV-style manual context windows with context_stride >= 2 produce dilated
    # windows; every generated index must be a valid frame index. create_windows_
    # uniform_standard only reads these handler attributes.
    num_frames = 48
    handler = SimpleNamespace(context_length=16, context_stride=3, context_overlap=0, _step=0)
    windows = create_windows_uniform_standard(num_frames, handler, {})
    assert windows
    for window in windows:
        for idx in window:
            assert 0 <= idx < num_frames, f"index {idx} out of range in window {window}"
