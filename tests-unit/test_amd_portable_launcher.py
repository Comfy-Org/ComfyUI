import importlib.util
import io
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / ".ci" / "relocate_portable_launcher.py"
SPEC = importlib.util.spec_from_file_location("relocate_portable_launcher", HELPER_PATH)
relocate_portable_launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relocate_portable_launcher)


def make_launcher(path: Path, shebang: bytes) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("__main__.py", "print('launcher')")
    data = b"MZ #!launcher resource" + shebang + b"\r\n" + archive.getvalue()
    path.write_bytes(data)
    return data


class PortableLauncherTests(unittest.TestCase):
    def test_relocates_interpreter_relative_to_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "offload-arch.exe"
            original = make_launcher(
                launcher,
                b"#!D:\\a\\ComfyUI\\python_embeded\\python.exe",
            )

            relocate_portable_launcher.relocate_launcher(launcher)

            updated = launcher.read_bytes()
            eocd = updated.rfind(b"PK\x05\x06")
            central_directory_size, central_directory_offset = struct.unpack_from("<II", updated, eocd + 12)
            archive_start = eocd - central_directory_size - central_directory_offset
            shebang_start = updated.rfind(b"#!", max(0, archive_start - 1024), archive_start)
            shebang_end = updated.index(b"\r\n", shebang_start)
            shebang = updated[shebang_start:shebang_end]
            self.assertTrue(shebang.startswith(b"#!<launcher_dir>\\..\\python.exe"))
            self.assertEqual(updated[archive_start:], original[archive_start:])

    def test_rejects_launcher_without_python_shebang(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "offload-arch.exe"
            original = make_launcher(launcher, b"#!D:\\tools\\wrong.exe")

            with self.assertRaisesRegex(ValueError, "does not launch python.exe"):
                relocate_portable_launcher.relocate_launcher(launcher)

            self.assertEqual(launcher.read_bytes(), original)

    def test_rejects_launcher_when_relative_path_does_not_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "offload-arch.exe"
            original = make_launcher(launcher, b"#!python.exe")

            with self.assertRaisesRegex(ValueError, "cannot fit"):
                relocate_portable_launcher.relocate_launcher(launcher)

            self.assertEqual(launcher.read_bytes(), original)

    def test_rejects_file_without_appended_python_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "offload-arch.exe"
            original = b"MZ not a Python launcher"
            launcher.write_bytes(original)

            with self.assertRaisesRegex(ValueError, "not an appended Python launcher"):
                relocate_portable_launcher.relocate_launcher(launcher)

            self.assertEqual(launcher.read_bytes(), original)

    def test_amd_release_relocates_offload_arch_launcher(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "stable-release.yml").read_text()

        self.assertIn('inputs.rel_name }}" = "amd"', workflow)
        self.assertIn("relocate_portable_launcher.py ./Scripts/offload-arch.exe", workflow)
