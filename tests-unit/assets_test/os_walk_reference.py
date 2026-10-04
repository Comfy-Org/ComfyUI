"""The os.walk walker the scan used before walk_listings replaced it, frozen here as the
reference the parity tests compare walk_listings against."""

import os

from app.assets.services.file_utils import is_visible


def list_files_with_os_walk(base_dir: str) -> list[str]:
    out: list[str] = []
    base_abs = os.path.abspath(base_dir)
    if not os.path.isdir(base_abs):
        return out
    seen_dirs: set[tuple[int, int]] = set()
    for dirpath, subdirs, filenames in os.walk(base_abs, topdown=True, followlinks=True):
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
