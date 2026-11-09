import json
import math
from pathlib import Path
import tempfile

import numpy as np
from safetensors.torch import save_file
import torch

from . import moss_case, moss_checkpoint, process


def _timing(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"TensorFold returned an invalid MOSS timing: {name}")


def read_receipt(path, tensors):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1 << 20:
        raise ValueError("TensorFold returned an invalid MOSS receipt file")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("TensorFold returned a malformed MOSS receipt") from error
    if not isinstance(data, dict):
        raise ValueError("TensorFold returned an invalid MOSS receipt")
    sizes = {
        "image_rows": tensors["latents"].shape[0] * tensors["latents"].shape[1],
        "prompt_rows": tensors["prompt"].shape[0],
        "negative_rows": tensors["negative"].shape[0],
    }
    for name, expected in sizes.items():
        if type(data.get(name)) is not int or data[name] != expected:
            raise ValueError(f"TensorFold's MOSS receipt does not match {name}")
    if not isinstance(data.get("device"), str) or not data["device"]:
        raise ValueError("TensorFold's MOSS receipt does not identify its GPU")
    if "decode_s" in data or "native_encoder" in data:
        raise ValueError("MOSS sampling requires TensorFold's CUDA denoiser executable")
    run = data.get("run")
    if not isinstance(run, list) or len(run) != len(tensors["timesteps"]):
        raise ValueError("TensorFold's MOSS receipt has the wrong number of denoising steps")
    for index, step in enumerate(run):
        if not isinstance(step, dict):
            raise ValueError("TensorFold returned an invalid MOSS step receipt")
        for name in ("wall_s", "gpu_s", "host_s"):
            _timing(step.get(name), f"step {index + 1} {name}")
    for name in ("load_s", "refine_s", "denoise_s"):
        _timing(data.get(name), name)
    return data


def sample_moss(checkpoint_path, executable, positive, negative, width, height, steps, seed,
                work_root, interrupt, progress, environment=None):
    interrupt()
    checkpoint = moss_checkpoint.read_checkpoint(checkpoint_path)
    tensors = moss_case.make_case(positive, negative, width, height, steps, seed)
    interrupt()
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    executable = Path(executable).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="tensorfold-moss-", dir=work_root) as scratch:
        work = Path(scratch)
        transformer = work / "transformer"
        checkpoint.stage(transformer)
        case = work / "case.safetensors"
        save_file(tensors, str(case))
        interrupt()
        process.run_native([str(executable), str(transformer), str(case), str(work)],
                           cwd=work, interrupt=interrupt, progress=progress, environment=environment)
        checkpoint.verify()
        receipt = read_receipt(work / "run.json", tensors)
        output = work / "latents.f32"
        if not output.is_file() or output.is_symlink() or output.stat().st_size != tensors["latents"].numel() * 4:
            raise ValueError("TensorFold returned a truncated or oversized MOSS latent buffer")
        latents = torch.from_numpy(np.fromfile(output, dtype="<f4"))
        result = moss_case.unpack_latents(latents, width, height)
        interrupt()
    receipt.update({
        "backend": "tensorfold-moss-cuda", "width": width, "height": height, "seed": seed,
        "requested_steps": steps, "effective_steps": len(tensors["timesteps"]),
        "guidance": 3.0, "negative_skip_layer": 22,
        "schedule": "moss-v2-shift5-training1000", "sigmas_sha256": moss_case.digest(tensors["sigmas"]),
        "noise_sha256": moss_case.digest(tensors["latents"]),
        "checkpoint_header_sha256": checkpoint.header_sha256,
    })
    return result, receipt
