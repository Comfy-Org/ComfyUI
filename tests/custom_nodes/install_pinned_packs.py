#!/usr/bin/env python3
"""Clone the packs in manifest.json into custom_nodes/ at their pinned commits.

Used by .github/workflows/test-custom-nodes.yml, and the same way to reproduce
a CI failure locally:

    python tests/custom_nodes/install_pinned_packs.py
    for r in custom_nodes/*/requirements.txt; do pip install -r "$r"; done
    python main.py --cpu 2>&1 | tee console_output.log
    COMFYUI_URL=http://127.0.0.1:8188 COMFYUI_STARTUP_LOG=console_output.log \
        pytest tests/custom_nodes

It refuses to touch a directory that is already there, so it cannot quietly
discard local edits to a pack you are bisecting.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def main() -> int:
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    for pack in manifest["packs"]:
        target = REPO_ROOT / "custom_nodes" / pack["pack"]
        if target.exists():
            sys.stderr.write(f"{pack['pack']}: already at {target}, left alone\n")
            continue
        subprocess.run(
            ["git", "clone", "--quiet", pack["repo"], str(target)], check=True
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--quiet", pack["pin"]], check=True
        )
        sys.stderr.write(f"{pack['pack']} @ {pack['pin']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
