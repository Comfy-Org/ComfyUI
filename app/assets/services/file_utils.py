import os
from collections.abc import Callable


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


def list_files_recursively(
    base_dir: str, on_error: Callable[[str, OSError], None] | None = None
) -> list[str]:
    """Recursively list all files in a directory, following symlinks.

    What cannot be read is left out of the listing. ``on_error`` hears about it as
    ``("walk_root", exc)`` when ``base_dir`` itself cannot be stat'ed or listed, such
    as an unmounted share, and ``("walk_dir", exc)`` for a directory below it.
    """
    out: list[str] = []
    base_abs = os.path.abspath(base_dir)
    if not os.path.isdir(base_abs):
        if on_error is not None:
            try:
                os.stat(base_abs)
            except OSError as exc:
                on_error("walk_root", exc)
        return out

    def report_dir_error(exc: OSError) -> None:
        if on_error is not None:
            on_error("walk_root" if exc.filename == base_abs else "walk_dir", exc)

    # Track seen real directory identities to prevent circular symlink loops
    seen_dirs: set[tuple[int, int]] = set()
    for dirpath, subdirs, filenames in os.walk(
        base_abs, topdown=True, onerror=report_dir_error, followlinks=True
    ):
        try:
            st = os.stat(dirpath)
            dir_id = (st.st_dev, st.st_ino)
        except OSError as exc:
            report_dir_error(exc)
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
