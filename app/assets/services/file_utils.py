import os


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
