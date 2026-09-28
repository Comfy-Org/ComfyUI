import os
from typing import NamedTuple


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


def list_files_recursively(base_dir: str) -> list[str]:
    """Recursively list all files in a directory, following symlinks."""
    out: list[str] = []
    base_abs = os.path.abspath(base_dir)
    if not os.path.isdir(base_abs):
        return out
    # Track seen real directory identities to prevent circular symlink loops
    seen_dirs: set[tuple[int, int]] = set()
    for dirpath, subdirs, filenames in os.walk(
        base_abs, topdown=True, followlinks=True
    ):
        try:
            st = os.stat(dirpath)
            dir_id = (st.st_dev, st.st_ino)
        except OSError:
            subdirs.clear()
            continue
        if dir_id in seen_dirs:
            subdirs.clear()
            continue
        seen_dirs.add(dir_id)
        subdirs[:] = [d for d in subdirs if is_visible(d)]
        for name in filenames:
            if not is_visible(name):
                continue
            out.append(os.path.abspath(os.path.join(dirpath, name)))
    return out


# dir path -> (st_mtime_ns, visible file names, visible subdir names)
DirListings = dict[str, tuple[int, list[str], list[str]]]


class ListingWalk(NamedTuple):
    files: list[str]
    listings: DirListings
    dirs_listed: int


def _list_visible_entries(dirpath: str) -> tuple[list[str], list[str]]:
    """One directory's visible (file names, subdir names), classified as os.walk does:
    anything whose is_dir() is false or raises is a file. The exception is a symlink
    whose target is gone: it is left out, so a row for it reads as vanished, as it did
    when the rescan stat'ed every row through the link."""
    files: list[str] = []
    subdirs: list[str] = []
    with os.scandir(dirpath) as entries:
        for entry in entries:
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


def walk_listings(base_dir: str) -> ListingWalk:
    """list_files_recursively, also returning every directory listing it read.

    Same traversal as the os.walk version (visit order, symlink following, device/inode
    cycle guard, hidden filtering), except each directory is stat'ed before it is listed.
    ``listings`` holds exactly the directories this walk listed, keyed by normalized
    absolute path, so it doubles as the record of which directories the walk can vouch for.
    """
    files: list[str] = []
    listings: DirListings = {}
    # No isdir() precheck, so each directory costs exactly one stat: a root that is
    # missing or not a directory fails its stat or scandir below and yields nothing.
    seen_dirs: set[tuple[int, int]] = set()
    stack = [os.path.abspath(base_dir)]
    while stack:
        dirpath = stack.pop()
        try:
            st = os.stat(dirpath)
        except OSError:
            continue
        dir_id = (st.st_dev, st.st_ino)
        if dir_id in seen_dirs:
            continue
        try:
            names, subdirs = _list_visible_entries(dirpath)
        except OSError:
            continue
        seen_dirs.add(dir_id)
        listings[dirpath] = (st.st_mtime_ns, names, subdirs)
        files.extend(os.path.abspath(os.path.join(dirpath, name)) for name in names)
        stack.extend(os.path.join(dirpath, name) for name in reversed(subdirs))
    return ListingWalk(files, listings, len(listings))
