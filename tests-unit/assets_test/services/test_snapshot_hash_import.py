import subprocess
import sys
from pathlib import Path


def test_snapshot_hash_defers_and_chains_blake3_import_failure() -> None:
    script = """
import builtins
import importlib.util
import sys
import tempfile
from pathlib import Path

real_import = builtins.__import__
blake3_import_error = ImportError("simulated blake3 ABI failure")

def import_without_blake3(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "blake3" or name.startswith("blake3."):
        raise blake3_import_error
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_blake3

module_path = Path("app/assets/services/snapshot_hash.py")
spec = importlib.util.spec_from_file_location("isolated_snapshot_hash", module_path)
assert spec is not None
assert spec.loader is not None
snapshot_hash_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = snapshot_hash_module
spec.loader.exec_module(snapshot_hash_module)

snapshot_hash = snapshot_hash_module.snapshot_hash

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "asset.bin"
    path.write_bytes(b"hash me")
    try:
        snapshot_hash(str(path))
    except ModuleNotFoundError as error:
        assert error.__cause__ is blake3_import_error
    else:
        raise AssertionError("snapshot_hash should require blake3 at use time")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
