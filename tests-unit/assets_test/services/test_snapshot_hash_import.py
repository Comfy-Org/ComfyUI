import subprocess
import sys
from pathlib import Path


def test_disabled_assets_start_when_blake3_is_unavailable() -> None:
    script = """
import builtins
import sys

real_import = builtins.__import__

def import_without_blake3(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "blake3" or name.startswith("blake3."):
        raise ModuleNotFoundError("No module named 'blake3'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_blake3
sys.argv = ["main.py", "--cpu"]

import main
from app.assets.manager import NoAssets, default_asset_manager
from comfy.cli_args import args

args.enable_assets = False
assert isinstance(default_asset_manager(), NoAssets)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
