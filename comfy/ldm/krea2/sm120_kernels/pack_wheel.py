"""Package the precompiled Windows SM120 adapters as a wheel."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    source = Path(__file__).resolve().parent
    name = "comfyui_krea2_sm120"
    version = "0.1.0"
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp:
        stage = Path(temp)
        package = stage / "krea2_sm120"
        package.mkdir()
        for dll in ("half.dll", "mx.dll", "mxlt.dll", "rms.dll"):
            shutil.copyfile(args.kernel_directory / dll, package / dll)
        src = package / "source"
        src.mkdir()
        for filename in ("half.cpp", "mx.cpp", "mxlt.cu", "rms.cu", "build.ps1", "pack_wheel.py"):
            shutil.copyfile(source / filename, src / filename)

        metadata = stage / f"{name}-{version}.dist-info"
        licenses = metadata / "licenses"
        licenses.mkdir(parents=True)
        shutil.copyfile(source.parents[3] / "LICENSE", licenses / "LICENSE")
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.4\n"
            "Name: comfyui-krea2-sm120\n"
            f"Version: {version}\n"
            "Summary: Prebuilt Krea 2 Turbo SM120 adapters for ComfyUI on Windows\n"
            "Requires-Python: >=3.10\n"
            "License-Expression: GPL-3.0-only\n"
            "License-File: LICENSE\n"
            "Requires-Dist: nvidia-cublas==13.8.0.4\n\n",
            encoding="utf-8",
        )
        (metadata / "WHEEL").write_text(
            "Wheel-Version: 1.0\n"
            "Root-Is-Purelib: false\n"
            "Tag: py3-none-win_amd64\n",
            encoding="utf-8",
        )
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(stage), "--dest-dir", str(args.output_directory)],
            check=True,
        )


if __name__ == "__main__":
    main()
