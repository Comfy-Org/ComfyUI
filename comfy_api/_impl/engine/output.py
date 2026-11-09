"""Core engine extension: writing files to ComfyUI's output and temp folders."""
import os

FOLDERS = ("output", "temp")


def _check(data, suffix):
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes the node encoded")
    suffix = str(suffix).lstrip(".")
    if not suffix or os.sep in suffix or "/" in suffix or len(suffix) > 16:
        raise ValueError("suffix must be a short file extension such as 'png'")
    return suffix


class EngineOps:
    def write(self, data, filename_prefix, suffix, folder="output", width=0, height=0, trailing_underscore=True):
        """One file (``data`` is bytes) or a batch (``data`` is a list of bytes), named as ComfyUI's
        save nodes name them: ``<prefix>_<counter>_.<suffix>``. A batch shares one counter run and
        replaces ``%batch_num%`` in the prefix per file, as ``SaveImage`` does."""
        import folder_paths

        if folder not in FOLDERS:
            raise ValueError("folder must be 'output' or 'temp'")
        batch = isinstance(data, (list, tuple))
        files = list(data) if batch else [data]
        if not files:
            return []
        suffix = _check(files[0], suffix)
        for item in files[1:]:
            _check(item, suffix)
        root = folder_paths.get_output_directory() if folder == "output" else folder_paths.get_temp_directory()
        # Raises if the prefix leaves the folder.
        full_folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, root, width, height)
        end = "_" if trailing_underscore else ""
        # ComfyUI's counter can't see past %batch_num% names, so those overwrite as SaveImage does;
        # every other name is new, and an existing file is never replaced.
        mode = "wb" if "%batch_num%" in name else "xb"
        results = []
        for batch_number, item in enumerate(files):
            filename = f"{name.replace('%batch_num%', str(batch_number))}_{counter:05}{end}.{suffix}"
            with open(os.path.join(full_folder, filename), mode) as file:
                file.write(bytes(item))
            results.append({"filename": filename, "subfolder": subfolder, "type": folder})
            counter += 1
        return results if batch else results[0]
