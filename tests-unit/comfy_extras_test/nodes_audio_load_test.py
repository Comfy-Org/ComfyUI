"""Tests for audio file loading in comfy_extras/nodes_audio.py."""
import av
import numpy as np
import pytest

from comfy_extras import nodes_audio

SAMPLE_RATE = 44100


def _make_mp3(path, seconds=2.0):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pcm = np.stack([tone, tone])
    with av.open(str(path), "w") as container:
        try:
            stream = container.add_stream("mp3", rate=SAMPLE_RATE, layout="stereo")
        except (av.error.FFmpegError, ValueError):
            pytest.skip("PyAV build has no MP3 encoder")
        for i in range(0, pcm.shape[1], 1152):
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm[:, i:i + 1152]), format="fltp", layout="stereo")
            frame.sample_rate = SAMPLE_RATE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    with av.open(str(path)) as container:
        return [p.pos for p in container.demux(container.streams.audio[0]) if p.size]


def test_load_skips_corrupt_mp3_frame(tmp_path):
    path = tmp_path / "song.mp3"
    positions = _make_mp3(path)
    clean, sr = nodes_audio.load(str(path))

    # Zero one frame header mid-file, like a damaged frame in a ripped MP3.
    data = bytearray(path.read_bytes())
    pos = positions[len(positions) // 2]
    data[pos:pos + 4] = bytes(4)
    path.write_bytes(bytes(data))
    with av.open(str(path)) as container, pytest.raises(av.error.InvalidDataError):
        for _ in container.decode(audio=0):
            pass

    wav, sr2 = nodes_audio.load(str(path))
    assert sr2 == sr == SAMPLE_RATE
    assert wav.shape[0] == 2
    assert clean.shape[1] - 4 * 1152 <= wav.shape[1] < clean.shape[1]
