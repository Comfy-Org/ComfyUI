"""Dispatch-owned encoded output groups with incremental publication."""
from __future__ import annotations

import asyncio
import errno
import logging
import math
import os
import re
import secrets
import time
from dataclasses import dataclass
from typing import AsyncIterable, Literal, TypedDict


MediaRole = Literal["utility", "intermediate", "final"]
FailedEntryPolicy = Literal["retain", "discard"]


class EncodedMediaEntry(TypedDict):
    entry_id: str
    name: str
    folder: Literal["output", "temp"]
    role: MediaRole
    state: Literal["reserved", "publishing", "committed", "failed", "deleted"]
    bytes_written: int
    materialized: bool
    retention_error: str | None
    cleanup_error: str | None


def refuse_group_namespace(name: str) -> None:
    normalized = os.path.normpath(str(name)).replace("\\", "/")
    if normalized.split("/")[0].casefold() == ".comfy-media":
        raise PermissionError("managed media group names require their owning lease")


def _directory(parent: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent)
    except FileExistsError:
        pass
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


def _media_header(suffix: str, header: bytes) -> bool:
    extension = os.path.splitext(suffix)[1].lower()
    if extension == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if extension == ".gif":
        return header.startswith((b"GIF87a", b"GIF89a"))
    if extension == ".webp":
        return header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    if extension in {".mp4", ".mov"}:
        return header[4:8] == b"ftyp"
    if extension in {".webm", ".mkv"}:
        return header.startswith(b"\x1aE\xdf\xa3")
    raise ValueError("encoded media extension is not admitted")


@dataclass
class _Entry:
    descriptor: EncodedMediaEntry
    suffix: str
    identity: tuple[int, int] | None = None


class EncodedMediaGroup:
    def __init__(
        self, root: str, prefix: str, folder: Literal["output", "temp"], *,
        max_bytes: int, max_entries: int, timeout_seconds: float,
        failed_entry_policy: FailedEntryPolicy,
    ) -> None:
        if (not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY"))
                or not {os.open, os.mkdir, os.stat, os.link, os.unlink, os.rmdir}.issubset(os.supports_dir_fd)):
            raise NotImplementedError("managed media groups require descriptor-relative filesystem support")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 2**30:
            raise ValueError("media group byte budget must be in [1, 1 GiB]")
        if type(max_entries) is not int or not 1 <= max_entries <= 256:
            raise ValueError("media group entry limit must be in [1, 256]")
        if (type(timeout_seconds) not in (int, float) or
                not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120):
            raise ValueError("media group timeout must be in (0, 120] seconds")
        if failed_entry_policy not in {"retain", "discard"}:
            raise ValueError("failed media entry policy must be explicit")
        if (type(prefix) is not str or not prefix or len(prefix.encode("utf-8")) > 1024
                or any(char in prefix for char in "\\:\x00")):
            raise ValueError("media prefix must be a bounded relative name")
        parts = prefix.split("/")
        if any(part in {"", ".", ".."} or len(part.encode("utf-8")) > 128 for part in parts):
            raise ValueError("media prefix contains an invalid path component")
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for component in [".comfy-media", *parts[:-1]]:
                child = _directory(parent, component)
                os.close(parent)
                parent = child
            for counter in range(1, 1_000_001):
                basename = f"{parts[-1]}_{counter:05}"
                try:
                    os.mkdir(basename, mode=0o700, dir_fd=parent)
                    break
                except FileExistsError:
                    continue
            else:
                raise ValueError("media group naming range is exhausted")
            directory = os.open(basename, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        except BaseException:
            os.close(parent)
            raise
        self._parent = parent
        self._directory = directory
        self._basename = basename
        self._relative = "/".join([".comfy-media", *parts[:-1], basename])
        self._folder = folder
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._deadline = time.monotonic() + timeout_seconds
        self._failure_policy = failed_entry_policy
        self._entries: dict[str, _Entry] = {}
        self._active: set[asyncio.Task] = set()
        self._reserved = 0
        self._used = 0
        self._closed = False
        self._released = False

    def _live(self, *, allow_expired: bool = False) -> None:
        if self._closed:
            raise RuntimeError("media group dispatch has ended")
        if not allow_expired and time.monotonic() >= self._deadline:
            raise TimeoutError("media group lease expired")

    def reserve(self, entry_id: str, suffix: str, role: MediaRole) -> EncodedMediaEntry:
        self._live()
        if type(entry_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", entry_id):
            raise ValueError("media entry ID must be a bounded logical identifier")
        if type(suffix) is not str or not re.fullmatch(r"[._][A-Za-z0-9_.-]{0,127}", suffix):
            raise ValueError("media suffix must not contain path components")
        if role not in {"utility", "intermediate", "final"}:
            raise ValueError("unknown media entry role")
        existing = self._entries.get(entry_id)
        if existing is not None:
            if existing.suffix != suffix or existing.descriptor["role"] != role:
                raise ValueError("media entry ID already has a different declaration")
            return dict(existing.descriptor)
        if len(self._entries) >= self._max_entries:
            raise ValueError("media group entry limit exceeded")
        if any(entry.suffix.casefold() == suffix.casefold() for entry in self._entries.values()):
            raise ValueError("media entry filename is already reserved")
        component = self._basename + suffix
        logical_name = f"{self._relative}/{component}"
        if len(component.encode("utf-8")) > 255 or len(logical_name.encode("utf-8")) > 1024:
            raise ValueError("generated media filename exceeds its byte limit")
        descriptor: EncodedMediaEntry = {
            "entry_id": entry_id, "name": logical_name,
            "folder": self._folder, "role": role, "state": "reserved",
            "bytes_written": 0, "materialized": False,
            "retention_error": None,
            "cleanup_error": None,
        }
        self._entries[entry_id] = _Entry(descriptor, suffix)
        return dict(descriptor)

    def entries(self) -> list[EncodedMediaEntry]:
        self._live(allow_expired=True)
        return [dict(entry.descriptor) for entry in self._entries.values()]

    async def publish(
        self, entry_id: str, chunks: AsyncIterable[bytes], max_bytes: int,
    ) -> EncodedMediaEntry:
        self._live()
        entry = self._entries[entry_id]
        if entry.descriptor["state"] == "committed":
            return dict(entry.descriptor)
        if entry.descriptor["state"] != "reserved":
            raise ValueError("media entry is already finished")
        if (type(max_bytes) is not int or not 1 <= max_bytes <=
                self._max_bytes - self._used - self._reserved):
            raise ValueError("media entry exceeds the remaining group byte budget")
        _media_header(entry.suffix, b"")
        task = asyncio.current_task()
        if task in self._active:
            raise RuntimeError("media group publication cannot reenter itself")
        iterator = aiter(chunks)
        temporary = ".partial-" + secrets.token_hex(16)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._directory)
        entry.descriptor["state"] = "publishing"
        self._reserved += max_bytes
        self._active.add(task)
        count = 0
        uncertain_size = False
        header = b""
        primary: BaseException | None = None
        try:
            while True:
                self._live()
                try:
                    chunk = await asyncio.wait_for(anext(iterator), self._deadline - time.monotonic())
                except StopAsyncIteration:
                    break
                self._live()
                if type(chunk) is not bytes or not 1 <= len(chunk) <= 1024 * 1024:
                    raise ValueError("media chunks must be 1..1 MiB bytes")
                if len(chunk) > max_bytes - count:
                    raise ValueError("media entry exceeded its byte budget")
                remaining = memoryview(chunk)
                while remaining:
                    self._live()
                    written = os.write(fd, remaining)
                    if written <= 0:
                        raise OSError("media writer made no progress")
                    count += written
                    remaining = remaining[written:]
                header = (header + chunk)[:16]
            if not _media_header(entry.suffix, header):
                raise ValueError("encoded media header does not match its extension")
            os.fsync(fd)
            completed_fd, fd = fd, -1
            os.close(completed_fd)
            self._live()
            self._commit(temporary, entry)
            entry.descriptor["state"] = "committed"
            entry.descriptor["bytes_written"] = count
            return dict(entry.descriptor)
        except BaseException as error:
            primary = error
            entry.descriptor["state"] = "failed"
            if fd >= 0:
                failed_fd, fd = fd, -1
                try:
                    os.close(failed_fd)
                except OSError as cleanup:
                    entry.descriptor["cleanup_error"] = type(cleanup).__name__
            try:
                count = os.stat(temporary, dir_fd=self._directory, follow_symlinks=False).st_size
            except OSError as cleanup:
                uncertain_size = True
                entry.descriptor["cleanup_error"] = type(cleanup).__name__
            if self._failure_policy == "retain" and (count or uncertain_size):
                try:
                    self._commit(temporary, entry)
                except OSError as error:
                    entry.descriptor["retention_error"] = type(error).__name__
            raise
        finally:
            entry.descriptor["bytes_written"] = count
            self._used += max_bytes if uncertain_size else count
            self._reserved -= max_bytes
            cleanup: BaseException | None = None
            try:
                try:
                    os.unlink(temporary, dir_fd=self._directory)
                except OSError as error:
                    cleanup = error
                try:
                    close = getattr(iterator, "aclose", None)
                    if close is not None:
                        await asyncio.wait_for(close(), 2)
                except BaseException as error:
                    cleanup = cleanup or error
            finally:
                self._active.discard(task)
                if self._closed and not self._active:
                    try:
                        self._release()
                    except OSError as error:
                        cleanup = cleanup or error
            if cleanup is not None:
                entry.descriptor["cleanup_error"] = type(cleanup).__name__
            if entry.descriptor["cleanup_error"] or entry.descriptor["retention_error"]:
                logging.getLogger(__name__).warning(
                    "media entry %s cleanup=%s retention=%s",
                    entry_id, entry.descriptor["cleanup_error"], entry.descriptor["retention_error"],
                )
            if cleanup is not None:
                if primary is None:
                    raise cleanup

    def _commit(self, temporary: str, entry: _Entry) -> None:
        name = self._basename + entry.suffix
        metadata = os.stat(temporary, dir_fd=self._directory, follow_symlinks=False)
        os.link(temporary, name, src_dir_fd=self._directory, dst_dir_fd=self._directory, follow_symlinks=False)
        entry.identity = (metadata.st_dev, metadata.st_ino)
        entry.descriptor["materialized"] = True

    def remove(self, entry_id: str) -> EncodedMediaEntry:
        self._live()
        entry = self._entries[entry_id]
        if entry.descriptor["state"] == "publishing":
            raise RuntimeError("cannot remove an entry while it is publishing")
        if entry.descriptor["materialized"]:
            name = self._basename + entry.suffix
            metadata = os.stat(name, dir_fd=self._directory, follow_symlinks=False)
            if entry.identity != (metadata.st_dev, metadata.st_ino):
                raise PermissionError("media entry was replaced by another writer")
            os.unlink(name, dir_fd=self._directory)
        entry.descriptor["state"] = "deleted"
        entry.descriptor["materialized"] = False
        return dict(entry.descriptor)

    def close(self) -> None:
        self._closed = True
        for task in self._active:
            task.cancel()
        if not self._active:
            self._release()

    def _release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            os.rmdir(self._basename, dir_fd=self._parent)
        except OSError as error:
            if error.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                raise
        finally:
            try:
                os.close(self._directory)
            finally:
                os.close(self._parent)
