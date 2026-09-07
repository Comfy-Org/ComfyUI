import ipaddress
import os
import subprocess
import sys
import threading


def is_loopback_address(address: str | None) -> bool:
    if not address:
        return False

    try:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False

    if parsed.is_loopback:
        return True
    return bool(
        parsed.version == 6 and parsed.ipv4_mapped and parsed.ipv4_mapped.is_loopback
    )


def build_file_manager_command(
    file_path: str, platform: str | None = None
) -> list[str]:
    normalized_path = os.path.abspath(file_path)
    current_platform = platform or sys.platform

    if current_platform == "win32":
        return ["explorer.exe", f"/select,{normalized_path}"]
    if current_platform == "darwin":
        return ["open", "-R", normalized_path]
    return ["xdg-open", os.path.dirname(normalized_path)]


def reveal_file_in_file_manager(file_path: str) -> None:
    normalized_path = os.path.abspath(file_path)
    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(normalized_path)

    process = subprocess.Popen(
        build_file_manager_command(normalized_path),
        close_fds=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    threading.Thread(target=process.wait, daemon=True).start()
