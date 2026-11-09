"""Core engine extension: reading ComfyUI's input, output and temp folders.

A ref is a file name as ComfyUI's loaders take it: ``"photo.png"`` (input folder),
``"sub/photo.png [input]"``, ``"result_00001_.png [output]"`` or ``"x.png [temp]"``.
Refs stay inside those folders.
"""
import os

FOLDERS = ("input", "output", "temp")


def _root(folder):
    import folder_paths

    if folder not in FOLDERS:
        raise ValueError("folder must be 'input', 'output' or 'temp'")
    return os.path.abspath(folder_paths.get_directory_by_type(folder))


def _resolve(ref):
    import folder_paths

    if not isinstance(ref, str) or not ref or "\x00" in ref:
        raise ValueError("a file ref is a non-empty string")
    path = os.path.abspath(folder_paths.get_annotated_filepath(ref))
    if not any(path.startswith(_root(f) + os.sep) for f in FOLDERS):
        raise PermissionError(f"{ref!r} is outside the input, output and temp folders")
    return path


class EngineOps:
    def read(self, ref):
        with open(_resolve(ref), "rb") as file:
            return file.read()

    def exists(self, ref):
        try:
            return os.path.isfile(_resolve(ref))
        except (ValueError, PermissionError):
            return False

    def list(self, folder="input", subfolder="", recursive=False, extensions=None):
        root = _root(folder)
        start = os.path.abspath(os.path.join(root, subfolder))
        if start != root and not start.startswith(root + os.sep):
            raise PermissionError("subfolder is outside the folder")
        wanted = {("." + e.lower().lstrip(".")) for e in extensions} if extensions else None
        found = []
        for directory, dirs, names in os.walk(start):
            dirs.sort()
            for name in sorted(names):
                if wanted is not None and os.path.splitext(name)[1].lower() not in wanted:
                    continue
                path = os.path.join(directory, name)
                stat = os.stat(path)
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                found.append({"ref": f"{relative} [{folder}]", "name": relative, "size": stat.st_size,
                              "mtime": stat.st_mtime})
            if not recursive:
                break
        return found
