import os
from typing import Callable, NamedTuple

from app.assets.services.gil import yield_gil

# Longer run window for output rescans: they repeat after prompts, so pausing every
# 2ms would add up to a much slower rescan.
RESCAN_YIELD_RUN = 0.010


def get_mtime_ns(stat_result: os.stat_result) -> int:
    """Extract mtime in nanoseconds from a stat result."""
    return getattr(
        stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)
    )


def get_size_and_mtime_ns(path: str, follow_symlinks: bool = True) -> tuple[int, int]:
    """Get file size in bytes and mtime in nanoseconds."""
    st = os.stat(path, follow_symlinks=follow_symlinks)
    return st.st_size, get_mtime_ns(st)


def verify_file_unchanged(
    mtime_db: int | None,
    size_db: int | None,
    stat_result: os.stat_result,
) -> bool:
    """Check if a file is unchanged based on mtime and size.

    Returns True if the file's mtime and size match the database values.
    Returns False if mtime_db is None or values don't match.

    size_db=None means don't check size; 0 is a valid recorded size.
    """
    if mtime_db is None:
        return False
    actual_mtime_ns = get_mtime_ns(stat_result)
    if int(mtime_db) != int(actual_mtime_ns):
        return False
    if size_db is not None:
        return int(stat_result.st_size) == int(size_db)
    return True


def is_visible(name: str) -> bool:
    """Return True if a file or directory name is visible (not hidden)."""
    return not name.startswith(".")


# dir path -> (visible file names, visible subdir names)
DirListings = dict[str, tuple[list[str], list[str]]]


class ListingWalk(NamedTuple):
    files: list[str]
    listings: DirListings
    dirs_listed: int


def _list_visible_entries(
    dirpath: str, should_stop: Callable[[], bool] | None = None
) -> tuple[list[str], list[str]] | None:
    """One directory's visible (file names, subdir names), classified as os.walk does:
    anything whose is_dir() is false or raises is a file. The exception is a symlink
    whose target is gone: it is left out, so a row for it reads as vanished, as it did
    when the rescan stat'ed every row through the link.

    ``should_stop`` is called before each entry, so a directory of 100k entries on a slow
    share can pause part way; True abandons the directory and returns None."""
    files: list[str] = []
    subdirs: list[str] = []
    with os.scandir(dirpath) as entries:
        for entry in entries:
            if should_stop is not None and should_stop():
                return None
            yield_gil(run=RESCAN_YIELD_RUN)
            if not is_visible(entry.name):
                continue
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            if is_dir:
                subdirs.append(entry.name)
            elif not (entry.is_symlink() and not os.path.exists(entry.path)):
                files.append(entry.name)
    return files, subdirs


def walk_listings(base_dir: str, should_stop: Callable[[], bool] | None = None) -> ListingWalk:
    """Every visible file under ``base_dir``, following symlinks, and every directory
    listing read on the way.

    The traversal is os.walk's (top-down, depth-first in listing order, following
    symlinked directories), with a device/inode guard against symlink cycles and hidden
    names left out. ``listings`` holds exactly the directories this walk listed, keyed by
    normalized absolute path, so it doubles as the record of which directories the walk
    can vouch for.

    ``should_stop`` is called before each directory and each entry; it may block. Once it
    returns True the walk ends: the result is partial, and the directory being read is in
    neither ``files`` nor ``listings``.
    """
    files: list[str] = []
    listings: DirListings = {}
    # No isdir() precheck, so each directory costs exactly one stat: a root that is
    # missing or not a directory fails its stat or scandir below and yields nothing.
    seen_dirs: set[tuple[int, int]] = set()
    stack = [os.path.abspath(base_dir)]
    while stack:
        if should_stop is not None and should_stop():
            break
        yield_gil(run=RESCAN_YIELD_RUN)
        dirpath = stack.pop()
        try:
            st = os.stat(dirpath)
        except OSError:
            continue
        dir_id = (st.st_dev, st.st_ino)
        if dir_id in seen_dirs:
            continue
        try:
            listing = _list_visible_entries(dirpath, should_stop)
        except OSError:
            continue
        if listing is None:
            break
        names, subdirs = listing
        seen_dirs.add(dir_id)
        listings[dirpath] = (names, subdirs)
        for name in names:
            yield_gil(run=RESCAN_YIELD_RUN)
            files.append(os.path.abspath(os.path.join(dirpath, name)))
        stack.extend(os.path.join(dirpath, name) for name in reversed(subdirs))
    return ListingWalk(files, listings, len(listings))
