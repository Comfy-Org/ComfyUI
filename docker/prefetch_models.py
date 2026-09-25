"""Download the checkpoints the wrapper workflows need, before the first render.

The wrapper already fetches a workflow's models on first use
(``--auto-download-models``), but that first use is a customer's render: on an
empty RunPod network volume the first pod would spend its boot downloading
~40 GB while voxmin-backend waits for it. Run this once against the volume
instead (a throwaway CPU pod is enough) and every later pod starts warm:

    docker run --rm -v /path/to/models:/opt/ComfyUI/models -e HF_TOKEN=... \\
        shivanshtalwar0/comfyui prefetch

Specs name a workflow and, for MiniMax H3, a quantization:

    minimaxh3[:quant]      text/image-to-video (fl2va) UNET + text encoder + VAEs
    minimaxh3-ref[:quant]  reference-to-video (ref2va) UNET + the same encoder/VAEs
    flux2klein9b           FLUX.2 [klein] 9B stills (gated on Hugging Face: HF_TOKEN)

Quantizations are the wrapper's: nvfp4 (fp8 UNET), int8, fp8, bf16; they pick
the UNET. The text encoder is always the one the wrapper loads at render time,
the smallest on disk, so the 16 GB nvfp4 one: fetching the 34 GB int8 or 66 GB
bf16 encoder for an int8/bf16 UNET would only fill the volume. With no specs,
PREFETCH_MODELS (space or comma separated) is used, else the FloStudio default
set: int8 UNETs, the weights the RTX 5090 rig runs, so a shot looks the same
whichever GPU renders it.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_SPECS = ["minimaxh3:int8", "minimaxh3-ref:int8", "flux2klein9b"]


def say(message: str, *, error: bool = False) -> None:
    """Print for a CLI script (the repo's ruff config bans print())."""
    stream = sys.stderr if error else sys.stdout
    stream.write(message + "\n")
    stream.flush()


def models_for(spec: str) -> list[dict]:
    from api_wrapper import workflows as wf

    name, _, quant = spec.partition(":")
    name = name.strip().lower()
    quant = (quant.strip().lower() or wf.MINIMAX_H3_DEFAULT_QUANT)
    if name in ("minimaxh3", "minimaxh3-ref"):
        if quant not in wf.MINIMAX_H3_QUANT_MODELS:
            raise SystemExit(f"prefetch: unknown quantization {quant!r} (one of {', '.join(wf.MINIMAX_H3_QUANT_MODELS)})")
        # The encoder the wrapper's picker loads (smallest first), not the
        # quantization's own: see the module docstring.
        return wf.minimax_h3_models(quant, ref2va=name == "minimaxh3-ref",
                                    clip_name=wf.MINIMAX_H3_CLIP_ORDER[0])
    if name == "flux2klein9b":
        return list(wf.FLUX2_KLEIN_9B_MODELS)
    raise SystemExit(f"prefetch: unknown spec {spec!r} (minimaxh3[:quant], minimaxh3-ref[:quant], flux2klein9b)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="*", help="what to download (see above)")
    parser.add_argument("--dry-run", action="store_true", help="list the files and where they would go")
    opts = parser.parse_args(argv)

    specs = opts.specs or [s for s in os.environ.get("PREFETCH_MODELS", "").replace(",", " ").split() if s] or DEFAULT_SPECS

    # ComfyUI modules read nothing from our argv (comfy.options keeps CLI
    # parsing off unless main.py enables it), so importing them here is safe.
    import folder_paths
    from comfy import model_downloader

    model_downloader.set_enabled(True)

    wanted: dict[tuple[str, str], str] = {}
    for spec in specs:
        for model in models_for(spec):
            wanted.setdefault((model["folder"], model["filename"]), model["url"])

    failures = 0
    missing = 0
    for (folder, filename), url in wanted.items():
        present = folder_paths.get_full_path(folder, filename)
        target_dir = (folder_paths.get_folder_paths(folder) or ["?"])[0]
        if present:
            say(f"ok       {folder}/{filename}")
            continue
        if opts.dry_run:
            missing += 1
            say(f"missing  {folder}/{filename} -> {target_dir}")
            continue
        say(f"download {folder}/{filename} -> {target_dir}")
        path, error = model_downloader.download_model(folder, filename, url)
        if path is None:
            failures += 1
            say(f"FAILED   {folder}/{filename}: {error}", error=True)
        else:
            say(f"ok       {folder}/{filename} ({os.path.getsize(path) / 1e9:.1f} GB)")

    if opts.dry_run:
        say(f"prefetch (dry run): {len(wanted) - missing} of {len(wanted)} file(s) present, {missing} to download")
        return 0
    if failures:
        say(f"prefetch: {failures} of {len(wanted)} file(s) failed", error=True)
        return 1
    say(f"prefetch: {len(wanted)} file(s) ready")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
