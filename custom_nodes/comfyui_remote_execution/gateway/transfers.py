import hashlib
import json
import mimetypes
import os
import tempfile

import aiohttp

REMOTE_SUBFOLDER = "remote_execution"


class FileAccessError(Exception):
    pass


def file_inputs(node_def):
    # Inputs the node declares as uploaded files (LoadImage, LoadImageMask, ...).
    inputs = node_def.get("input", {})
    specs = {**(inputs.get("required") or {}), **(inputs.get("optional") or {})}
    return {name for name, spec in specs.items() if len(spec) > 1 and isinstance(spec[1], dict) and spec[1].get("image_upload") is True}


def parse_file_reference(value):
    # Same annotation format as folder_paths.get_annotated_filepath: "sub/name.png [input]".
    folder = "input"
    for t in ("input", "output", "temp"):
        suffix = f" [{t}]"
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            folder = t
            break
    subfolder, _, filename = value.rpartition("/")
    parts = subfolder.split("/") if subfolder else []
    if "\\" in value or ":" in value or value.startswith("/") or filename in ("", ".", "..") or any(p in ("", ".", "..") for p in parts):
        raise FileAccessError(f"Rejected file reference {value!r}: only plain names under the input, output or temp folders are supported.")
    return folder, subfolder, filename


class UploadCache:
    """Remote upload names keyed by remote target and file content."""

    def __init__(self, path):
        self.path = path
        self.entries = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.entries = json.load(f)

    def get(self, target_id, digest):
        return self.entries.get(f"{target_id}:{digest}")

    def put(self, target_id, digest, remote_value):
        self.entries[f"{target_id}:{digest}"] = remote_value
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f)

    def drop(self, target_id, digest):
        self.entries.pop(f"{target_id}:{digest}", None)


async def remote_file_exists(session, target, value):
    folder, subfolder, filename = parse_file_reference(value)
    async with session.head(f"{target.url}/view", params={"filename": filename, "subfolder": subfolder, "type": folder}) as resp:
        return resp.status == 200


async def transfer_input(session, local_url, target, cache, tmp_dir, value):
    folder, subfolder, filename = parse_file_reference(value)
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as f:
            # The local /view route owns path containment for the local folders.
            async with session.get(f"{local_url}/view", params={"filename": filename, "subfolder": subfolder, "type": folder}) as resp:
                if resp.status == 404:
                    raise FileNotFoundError(f"Local file {value!r} does not exist.")
                if resp.status != 200:
                    raise FileAccessError(f"Local file {value!r} could not be read (HTTP {resp.status}).")
                async for chunk in resp.content.iter_chunked(1 << 20):
                    digest.update(chunk)
                    f.write(chunk)
        digest = digest.hexdigest()

        cached = cache.get(target.id, digest)
        if cached is not None:
            if await remote_file_exists(session, target, cached):
                return cached
            cache.drop(target.id, digest)

        # Content-addressed names cannot collide with different remote files.
        remote_name = f"{digest[:16]}-{filename}"
        with open(tmp_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("image", f, filename=remote_name, content_type=mimetypes.guess_type(filename)[0] or "application/octet-stream")
            form.add_field("type", "input")
            form.add_field("subfolder", REMOTE_SUBFOLDER)
            form.add_field("overwrite", "true")
            async with session.post(f"{target.url}/upload/image", data=form) as resp:
                if resp.status != 200:
                    raise FileAccessError(f"Remote upload of {value!r} failed (HTTP {resp.status}): {(await resp.text())[:200]}")
                result = await resp.json()
        remote_value = f"{result['subfolder']}/{result['name']} [input]" if result.get("subfolder") else f"{result['name']} [input]"
        cache.put(target.id, digest, remote_value)
        return remote_value
    finally:
        os.unlink(tmp_path)
