"""Publishes a saved output file under a name no other writer holds.

Save nodes pick the next numbered name by listing the folder, so two writers
that share an output folder can pick the same name and one file replaces the
other. ``output_file`` has the encoder write to a hidden sibling path and then
publishes it at the requested name, or at a suffixed name when that one is
taken. A failed encode leaves nothing under the output name.
"""

import errno
import os
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Volumes that cannot hard-link report it in several ways. Any of these falls back to reserve-then-rename.
_LINK_UNSUPPORTED = {
    errno.ENOSYS, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    errno.EPERM, errno.EXDEV, errno.EINVAL,
}


@dataclass
class OutputFile:
    path: str
    filename: str


def _alternate(target: Path) -> Path:
    return target.with_name(f"{target.stem}_{uuid.uuid4().hex[:12]}{target.suffix}")


def _publish(staged: Path, target: Path, overwrite: bool) -> Path:
    if overwrite:
        os.replace(staged, target)
        return target

    candidate = target
    link_supported = True
    while True:
        if link_supported:
            try:
                # A hard link fails when the name exists, so it never replaces another writer's file.
                os.link(staged, candidate)
                return candidate
            except FileExistsError:
                candidate = _alternate(target)
                continue
            except OSError as error:
                if error.errno not in _LINK_UNSUPPORTED:
                    raise
                link_supported = False
        try:
            with candidate.open("xb") as reservation:
                reserved = os.fstat(reservation.fileno())
        except FileExistsError:
            candidate = _alternate(target)
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
def output_file(path: str, *, atomic: bool = True, overwrite: bool = False):
    """Yield where to write one saved file, and publish it when the block exits.

    Write and close the file at ``output.path`` inside the block. After the
    block, ``output.filename`` is the published basename, which carries a suffix
    when another writer took the requested name first. ``overwrite=True``
    replaces an existing file instead. ``atomic=False`` writes straight to
    ``path``, for previews and other files nothing else competes for.
    """
    target = Path(path)
    if not atomic:
        yield OutputFile(str(target), target.name)
        return

    # A sibling file inherits the folder's permissions and stays on its volume.
    # The leading dot keeps it out of folder listings and numbered-name scans.
    staged = target.with_name(f".{uuid.uuid4().hex[:12]}-{target.name}")
    output = OutputFile(str(staged), target.name)
    try:
        yield output
        if not stat.S_ISREG(staged.lstat().st_mode):
            raise ValueError(f"Output encoder did not write a regular file: {target.name}")
        output.filename = _publish(staged, target, overwrite).name
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
