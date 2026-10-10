"""--windows-standalone-build turns the assets system on; --disable-assets still wins."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PARSE = (
    "import sys, comfy.options; comfy.options.enable_args_parsing(); "
    "sys.argv = ['main.py', *sys.argv[1:]]; "
    "from comfy.cli_args import args; print(args.disable_assets)"
)


def _assets_disabled(*flags: str) -> bool:
    # Args are resolved at import time, so parse in a fresh interpreter.
    out = subprocess.run(
        [sys.executable, "-c", PARSE, *flags], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return out.strip().splitlines()[-1] == "True"


@pytest.mark.parametrize(
    ("flags", "disabled"),
    [
        ((), True),
        (("--windows-standalone-build",), False),
        (("--windows-standalone-build", "--disable-assets"), True),
        (("--windows-standalone-build", "--enable-assets"), False),
    ],
)
def test_standalone_build_turns_assets_on_unless_disabled(flags, disabled):
    assert _assets_disabled(*flags) is disabled
