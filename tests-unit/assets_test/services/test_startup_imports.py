import subprocess
import sys
from pathlib import Path


def test_the_asset_manager_does_not_import_torch():
    """main.py imports it before cuda_malloc configures the allocator, which must precede torch."""
    code = "import sys, app.assets.manager; assert 'torch' not in sys.modules, 'torch imported'"

    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[3])
