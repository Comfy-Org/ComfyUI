"""The converter's tests need the table of ComfyUI's own nodes (core_nodes.json), made once per session."""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CONVERSION = os.path.join(ROOT, "comfy_api", "conversion")


def pytest_configure(config):
    if os.environ.get("CONVERTER_CORE_NODES"):
        return
    path = os.path.join(tempfile.gettempdir(), "comfy_conversion", "core_nodes.json")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run([sys.executable, os.path.join(CONVERSION, "core_nodes.py"), path], cwd=ROOT,
                       env={**os.environ, "COMFY_ROOT": ROOT}, check=True, capture_output=True, text=True, timeout=900)
    os.environ["CONVERTER_CORE_NODES"] = path
