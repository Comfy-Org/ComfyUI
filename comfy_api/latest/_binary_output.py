"""Bounded binary output with no partially published file or open writer ref."""
import os
import tempfile

CHUNK_BYTES = 1024 * 1024


async def save_binary(chunks, prefix, extension, max_bytes):
    import folder_paths

    if extension not in {'.gguf', '.bin'}:
        raise ValueError('binary output extension must be .gguf or .bin')
    if type(max_bytes) is not int or not 1 <= max_bytes <= 2**63 - 1:
        raise ValueError('binary output requires a positive finite max_bytes')
    root = os.path.realpath(folder_paths.get_output_directory())
    proposed = os.path.realpath(os.path.join(root, prefix))
    if os.path.commonpath((root, proposed)) != root:
        raise ValueError('binary target escapes the output directory')
    folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, root)
    folder = os.path.realpath(folder)
    if os.path.commonpath((root, folder)) != root:
        raise ValueError('binary target escapes the output directory')
    fd, temporary = tempfile.mkstemp(prefix='.comfy-binary-', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as output:
            size = 0
            async for chunk in chunks:
                if type(chunk) is not bytes or not 1 <= len(chunk) <= CHUNK_BYTES:
                    raise ValueError('binary chunks must be 1..1 MiB bytes')
                if len(chunk) > max_bytes - size:
                    raise ValueError('binary output exceeds max_bytes')
                output.write(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        while True:
            name = f'{filename}_{counter:05}_.{extension[1:]}'
            target = os.path.join(folder, name)
            try:
                # A complete file appears atomically. An occupied name, even a
                # symlink, is never opened or replaced by this operation.
                os.link(temporary, target)
                return os.path.relpath(target, root).replace(os.sep, '/')
            except FileExistsError:
                counter += 1
    finally:
        os.unlink(temporary)
