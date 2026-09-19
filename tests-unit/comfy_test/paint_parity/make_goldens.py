"""Author-side: regenerate the committed tiny golden bundle.

    python tests-unit/comfy_test/paint_parity/make_goldens.py

Rerun only when the tiny model definition legitimately changes (parameter
renames/additions shift the per-parameter seeded init); commit the new bundle
together with that change and call it out in the PR.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from comfy.cli_args import args as comfy_args  # noqa: E402
if not torch.cuda.is_available():
    comfy_args.cpu = True

from paint_parity import harness  # noqa: E402


def main():
    out = os.path.join(_HERE, "goldens", "tiny_golden.safetensors")
    harness.make_tiny_golden(out)
    sys.stdout.write(f"wrote {out} ({os.path.getsize(out)} bytes)\n")


if __name__ == "__main__":
    main()
