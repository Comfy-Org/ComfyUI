"""Journals the output files a prompt emitted, for a launcher that runs several
ComfyUI builds over one output folder. A launcher opts in with a producer ID and
a journal directory. A record names the file, the prompt, and the file revision
seen when the node returned. See "Local output journal" in app/assets/docs.
"""

import json
import logging
import os
import stat
import threading
import uuid
from pathlib import Path

_PRODUCER_ENV = "COMFYUI_OUTPUT_PRODUCER_ID"
_JOURNAL_ENV = "COMFYUI_OUTPUT_JOURNAL_DIR"
_UINT64 = (1 << 64) - 1
# Windows reports a 64-bit volume serial. Readers that see the 32-bit one compare the low half.
_DEVICE_MASK = (1 << 32) - 1 if os.name == "nt" else _UINT64
_journal_lock = threading.Lock()
_journal_id = uuid.uuid4().hex


def _launcher_journal() -> tuple[Path, str] | None:
    directory = os.environ.get(_JOURNAL_ENV)
    producer_id = os.environ.get(_PRODUCER_ENV)
    if not directory or not producer_id:
        return None
    journal = Path(directory)
    return (journal, producer_id) if journal.is_absolute() else None


def enabled() -> bool:
    return _launcher_journal() is not None


def _append_record(directory: Path, producer_id: str, job_id: str | None, target: Path, revision: os.stat_result) -> None:
    global _journal_id
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
            # A failed write may have left a partial line. Start a new file so
            # only the last line of any journal can be incomplete.
            _journal_id = uuid.uuid4().hex
            logging.warning("Could not record output provenance for %s: %s", target.name, error)


def record_output(path: str, job_id: str | None) -> None:
    """Journal the file at ``path`` as an output of ``job_id``. Never raises."""
    import folder_paths

    try:
        launcher = _launcher_journal()
        if launcher is None:
            return
        directory, producer_id = launcher
        target = Path(path).parent.resolve() / os.path.basename(path)
        if not target.parent.is_relative_to(Path(folder_paths.get_output_directory()).resolve()):
            return
        revision = target.lstat()
        # A file ID the journal cannot carry leaves the output unattributed.
        if not stat.S_ISREG(revision.st_mode) or not 0 < revision.st_ino <= _UINT64:
            return
        _append_record(directory, producer_id, job_id, target, revision)
    except Exception:
        logging.warning("Could not record output provenance for %s", os.path.basename(path), exc_info=True)
