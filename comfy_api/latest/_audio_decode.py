"""Closed first-stream, native-rate, planar float32 audio decoding."""
from __future__ import annotations

import io
import time

import av
import torch

from ._deferred_audio import AudioDecodeError, AudioWindow


def decode_audio(source: bytes, window: AudioWindow) -> dict[str, object]:
    deadline = time.monotonic() + window.timeout_seconds
    try:
        with av.open(io.BytesIO(source), mode="r", options={
            "protocol_whitelist": "", "enable_drefs": "0", "use_absolute_path": "0",
        }) as container:
            return _decode_stream(container, window, deadline)
    except (av.error.InvalidDataError, av.error.EOFError) as error:
        raise AudioDecodeError(str(error)) from error


def _decode_stream(container: av.container.InputContainer, window: AudioWindow, deadline: float) -> dict[str, object]:
    if not container.streams.audio:
        raise AudioDecodeError("audio source has no audio stream")
    stream = container.streams.audio[0]
    rate = stream.codec_context.sample_rate
    channels = stream.codec_context.channels
    if not 8000 <= rate <= 192000 or not 1 <= channels <= 8:
        raise ValueError("audio stream rate or channel count exceeds its limit")
    start = int(window.start_seconds * rate)
    end = None if window.duration_seconds is None else start + int(window.duration_seconds * rate)
    resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout.name, rate=rate)
    frames: list[torch.Tensor] = []
    position = 0
    size = 0

    def append(frame: av.AudioFrame) -> bool:
        nonlocal position, size
        if time.monotonic() > deadline:
            raise TimeoutError("audio decode exceeded its time budget")
        count = frame.samples
        left = max(0, start - position)
        right = count if end is None else min(count, end - position)
        if right > left:
            length = right - left
            if channels * length * 4 > window.max_decoded_bytes - size:
                raise ValueError("audio decode exceeded its decoded byte budget")
            samples = frame.to_ndarray()[:, left:right].copy()
            frames.append(torch.from_numpy(samples))
            size += channels * length * 4
        position += count
        return end is not None and position >= end

    finished = False
    for frame in container.decode(stream):
        for converted in resampler.resample(frame):
            if append(converted):
                finished = True
                break
        if finished:
            break
    if not finished:
        for converted in resampler.resample(None):
            if append(converted):
                break
    if time.monotonic() > deadline:
        raise TimeoutError("audio decode exceeded its time budget")
    waveform = torch.cat(frames, dim=1) if frames else torch.empty((channels, 0), dtype=torch.float32)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": rate}
