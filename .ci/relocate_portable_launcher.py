from pathlib import Path
import struct
import sys


RELATIVE_INTERPRETER = b"#!<launcher_dir>\\..\\python.exe"
ZIP_EOCD = b"PK\x05\x06"


def relocate_launcher(path: Path) -> None:
    data = path.read_bytes()
    eocd = data.rfind(ZIP_EOCD)
    if eocd < 0 or eocd + 20 > len(data):
        raise ValueError(f"{path} is not an appended Python launcher")

    central_directory_size = struct.unpack_from("<I", data, eocd + 12)[0]
    central_directory_offset = struct.unpack_from("<I", data, eocd + 16)[0]
    archive_start = eocd - central_directory_size - central_directory_offset
    if archive_start < 0:
        raise ValueError(f"{path} has an invalid appended archive")

    shebang_start = archive_start
    if data[shebang_start : shebang_start + 2] != b"#!":
        shebang_start = data.rfind(b"#!", max(0, archive_start - 1024), archive_start)
    if shebang_start < 0:
        raise ValueError(f"{path} does not contain a launcher shebang")

    line_end = min(
        (offset for marker in (b"\r", b"\n") if (offset := data.find(marker, shebang_start)) >= 0),
        default=-1,
    )
    if line_end < shebang_start or line_end > archive_start + 2:
        raise ValueError(f"{path} has an invalid launcher shebang")

    shebang = data[shebang_start:line_end]
    if b"python.exe" not in shebang.lower():
        raise ValueError(f"{path} does not launch python.exe")
    if len(RELATIVE_INTERPRETER) > len(shebang):
        raise ValueError(f"{path} cannot fit the relative interpreter path")

    updated = RELATIVE_INTERPRETER.ljust(len(shebang), b" ")
    path.write_bytes(data[:shebang_start] + updated + data[line_end:])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: relocate_portable_launcher.py <launcher.exe>")
    relocate_launcher(Path(sys.argv[1]))
