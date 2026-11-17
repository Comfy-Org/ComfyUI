import errno
import json
import logging
import os
import stat
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import folder_paths
from comfy_execution.utils import get_executing_context

_PRODUCER_ENV = "COMFYUI_OUTPUT_PRODUCER_ID"
_JOURNAL_ENV = "COMFYUI_OUTPUT_JOURNAL_DIR"
_LINK_UNSUPPORTED = {
    errno.ENOSYS, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    errno.EPERM, errno.EXDEV, errno.EINVAL,
}
_UINT64 = (1 << 64) - 1
_DEVICE_MASK = (1 << 32) - 1 if os.name == "nt" else _UINT64
_journal_lock = threading.Lock()
_journal_id = uuid.uuid4().hex


@dataclass
class OutputFile:
    path: str
    filename: str


def _append_record(directory: Path, producer_id: str, job_id: str, target: Path, revision: os.stat_result):
    global _journal_id
    if not 0 < revision.st_ino <= _UINT64:
        return
    event = {
        "version": 1,
        "producer_id": producer_id,
        "job_id": job_id,
        "path": str(target),
        "device": str(revision.st_dev & _DEVICE_MASK),
        "inode": str(revision.st_ino),
        "size": str(revision.st_size),
        "mtime_ns": str(revision.st_mtime_ns),
    }
    line = (json.dumps(event, separators=(",", ":")) + "\n").encode("ascii")
    with _journal_lock:
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            journal = directory / f"{os.getpid()}-{_journal_id}.jsonl"
            with journal.open("ab") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            _journal_id = uuid.uuid4().hex
            logging.warning("Could not record output provenance for %s: %s", target.name, error)


def _publish(staged: Path, target: Path, record, overwrite: bool) -> Path:
    if overwrite:
        record(target)
        os.replace(staged, target)
        return target

    candidate = target
    link_supported = True
    while True:
        record(candidate)
        if link_supported:
            try:
                os.link(staged, candidate)
                return candidate
            except FileExistsError:
                candidate = target.with_name(f"{target.stem}_{uuid.uuid4().hex[:12]}{target.suffix}")
                continue
            except OSError as error:
                if error.errno not in _LINK_UNSUPPORTED:
                    raise
                link_supported = False
        try:
            with candidate.open("xb") as reservation:
                reserved = os.fstat(reservation.fileno())
        except FileExistsError:
            candidate = target.with_name(f"{target.stem}_{uuid.uuid4().hex[:12]}{target.suffix}")
            continue
        try:
            os.replace(staged, candidate)
        except OSError:
            try:
                if os.path.samestat(reserved, candidate.lstat()):
                    candidate.unlink()
            except FileNotFoundError:
                pass
            raise
        return candidate


@contextmanager
def output_file(path: str, *, track: bool = True, overwrite: bool = False):
    """Yield a path for one save. After exit, filename is the published basename."""
    path = Path(path)
    directory = os.environ.get(_JOURNAL_ENV)
    producer_id = os.environ.get(_PRODUCER_ENV)
    execution = get_executing_context()
    journal = Path(directory) if directory else None
    if not track or execution is None or not producer_id or journal is None or not journal.is_absolute():
        yield OutputFile(str(path), path.name)
        return

    target = path.parent.resolve() / path.name
    if not target.parent.is_relative_to(Path(folder_paths.get_output_directory()).resolve()):
        yield OutputFile(str(path), path.name)
        return

    with tempfile.TemporaryDirectory(prefix=".comfy-output-", dir=target.parent, ignore_cleanup_errors=True) as staging:
        staged = Path(staging) / target.name
        output = OutputFile(str(staged), target.name)
        yield output
        revision = staged.lstat()
        if not stat.S_ISREG(revision.st_mode):
            raise ValueError(f"Output encoder did not write a regular file: {target.name}")

        def record(destination: Path):
            _append_record(journal, producer_id, execution.prompt_id, destination, revision)

        output.filename = _publish(staged, target, record, overwrite).name
