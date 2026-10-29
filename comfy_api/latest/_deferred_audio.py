"""Owned AUDIO values whose transport does not consume their waveform."""
from __future__ import annotations

import math
import asyncio
import os
import stat
import threading
import weakref
from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from dataclasses import dataclass


class AudioDecodeError(RuntimeError):
    """The source has invalid encoded data, unexpected EOF or no audio stream."""


@dataclass(frozen=True)
class AudioWindow:
    start_seconds: float
    duration_seconds: float | None
    max_decoded_bytes: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        for name, value in (
            ("start_seconds", self.start_seconds),
            ("duration_seconds", self.duration_seconds),
            ("timeout_seconds", self.timeout_seconds),
        ):
            if value is None and name == "duration_seconds":
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name != "start_seconds" and value == 0):
                raise ValueError(f"{name} is outside its admitted range")
        if self.start_seconds > 86400 or (
            self.duration_seconds is not None and self.duration_seconds > 86400
        ):
            raise ValueError("audio windows are limited to one day")
        if type(self.max_decoded_bytes) is not int or not 4 <= self.max_decoded_bytes <= 2**30:
            raise ValueError("audio decoded byte limit must be in [4, 1 GiB]")
        if self.timeout_seconds > 120:
            raise ValueError("audio decode timeout must not exceed 120 seconds")


class DeferredAudio(Mapping[str, object]):
    """The engine/cache owns this value, independently of producer ref tables.

    Every ordinary mapping read is consumption. Internal SDK recognition uses
    the concrete type without reading the mapping. Only successful decodes are
    cached; a failed access can be attempted again by a later consumer.
    """

    def __init__(self, window: AudioWindow, reserved_bytes: int) -> None:
        self.window = window
        self.reserved_bytes = reserved_bytes
        self._source = b""
        self._value: dict[str, object] | None = None
        self._lock = threading.Lock()
        self._attempt: Future[dict[str, object]] | None = None
        self._waiters = 0

    def _join(self) -> tuple[Future[dict[str, object]], bool]:
        with self._lock:
            if self._waiters >= 16:
                raise ValueError("audio consumption waiter limit exceeded")
            self._waiters += 1
            if self._value is not None:
                completed: Future[dict[str, object]] = Future()
                completed.set_result(self._value)
                return completed, False
            if self._attempt is None:
                self._attempt = Future()
                return self._attempt, True
            return self._attempt, False

    def _leave(self) -> None:
        with self._lock:
            self._waiters -= 1

    def _decode(self, attempt: Future[dict[str, object]]) -> None:
        try:
            from ._audio_decode import decode_audio

            value = decode_audio(self._source, self.window)
            with self._lock:
                self._value = value
                self._source = b""
            attempt.set_result(value)
        except BaseException as error:
            attempt.set_exception(error)
        finally:
            with self._lock:
                if self._attempt is attempt:
                    self._attempt = None

    def materialize(self) -> dict[str, object]:
        attempt, leader = self._join()
        try:
            if leader:
                self._decode(attempt)
            return attempt.result(timeout=self.window.timeout_seconds)
        finally:
            self._leave()

    async def materialize_async(self) -> dict[str, object]:
        attempt, leader = self._join()
        try:
            if leader:
                try:
                    asyncio.get_running_loop().run_in_executor(None, self._decode, attempt)
                except BaseException as error:
                    attempt.set_exception(error)
                    with self._lock:
                        if self._attempt is attempt:
                            self._attempt = None
            return await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(attempt)), self.window.timeout_seconds,
            )
        finally:
            self._leave()

    def __getitem__(self, key: str) -> object:
        return self.materialize()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.materialize())

    def __len__(self) -> int:
        return len(self.materialize())

    def _comfy_cache_tensors(self) -> tuple[object, ...]:
        return () if self._value is None else (self._value["waveform"],)

    def _comfy_cache_extra_bytes(self) -> int:
        return len(self._source)

    _comfy_external_cacheable = False


class DeferredAudioPool:
    """Weak bookkeeping bounds values still owned by outputs and caches.

    Encoded sources are immutable bytes, not files or producer sessions. There
    is no external lease to strand when the last engine/cache owner releases
    a value. The pool never owns source bytes or decoded tensors itself.
    """

    def __init__(self, *, max_values: int = 16, max_bytes: int = 2**30) -> None:
        self._values: list[weakref.ReferenceType[DeferredAudio]] = []
        self._lock = threading.Lock()
        self._max_values = max_values
        self._max_bytes = max_bytes

    def capture(
        self, path: str, window: AudioWindow, max_source_bytes: int,
    ) -> DeferredAudio:
        if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_NONBLOCK")):
            raise NotImplementedError("managed audio capture requires no-follow regular-file support")
        if type(max_source_bytes) is not int or not 1 <= max_source_bytes <= 2**29:
            raise ValueError("audio source byte limit must be in [1, 512 MiB]")
        value = DeferredAudio(window, max_source_bytes + 2 * window.max_decoded_bytes)
        with self._lock:
            live = [item for ref in self._values if (item := ref()) is not None]
            self._values = [weakref.ref(item) for item in live]
            if (len(live) >= self._max_values or value.reserved_bytes >
                    self._max_bytes - sum(item.reserved_bytes for item in live)):
                raise ValueError("managed audio live-value budget exceeded")
            self._values.append(weakref.ref(value))
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size > max_source_bytes:
                    raise ValueError("audio source must be a bounded regular file")
                encoded = source.read(max_source_bytes + 1)
                after = os.fstat(source.fileno())
            if (len(encoded) > max_source_bytes or
                    (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise ValueError("audio source changed during capture or exceeds its limit")
            value._source = encoded
            return value
        except BaseException:
            with self._lock:
                self._values = [ref for ref in self._values if ref() is not value]
            raise
