import builtins
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_snapshot_hash_defers_and_chains_blake3_import_failure(tmp_path: Path) -> None:
    real_import = builtins.__import__
    blake3_import_error = ImportError("simulated blake3 ABI failure")

    def import_without_blake3(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "blake3" or name.startswith("blake3."):
            raise blake3_import_error
        return real_import(name, globals, locals, fromlist, level)

    module_path = Path(__file__).parents[3] / "app/assets/services/snapshot_hash.py"
    spec = importlib.util.spec_from_file_location(
        "isolated_snapshot_hash_import_guard", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    snapshot_hash_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = snapshot_hash_module
    try:
        with patch("builtins.__import__", side_effect=import_without_blake3):
            spec.loader.exec_module(snapshot_hash_module)

            path = tmp_path / "asset.bin"
            path.write_bytes(b"hash me")
            with pytest.raises(ModuleNotFoundError) as error_info:
                snapshot_hash_module.snapshot_hash(str(path))

            assert error_info.value.__cause__ is blake3_import_error
    finally:
        sys.modules.pop(spec.name, None)
