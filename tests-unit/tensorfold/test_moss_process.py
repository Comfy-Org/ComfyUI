import os
import sys
import textwrap

import pytest
import torch

from comfy.tensorfold import process
from comfy.tensorfold.moss import read_receipt, sample_moss
from test_moss_checkpoint import write_checkpoint


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="The MOSS worker and sparse checkpoint fixtures target Linux")


@pytest.fixture
def setup(tmp_path):
    executable = tmp_path / "tf-moss-image"
    executable.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''
        import json
        import math
        import os
        from pathlib import Path
        import struct
        import sys
        import time

        transformer, case, output = map(Path, sys.argv[1:])
        assert (transformer / "config.json").is_file()
        assert (transformer / "diffusion_pytorch_model.safetensors").is_file()
        with case.open("rb") as source:
            length = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(length))
        assert set(header) == {"prompt", "negative", "latents", "sigmas", "timesteps"}
        assert header["prompt"]["dtype"] == "BF16"
        assert header["latents"]["dtype"] == "F32"
        mode = os.environ.get("MOSS_FIXTURE_MODE", "good")
        if mode == "exit":
            print("fixture GPU load failure", file=sys.stderr)
            sys.exit(7)
        steps = header["timesteps"]["shape"][0]
        count = math.prod(header["latents"]["shape"])
        values = list(range(count))
        if mode == "nan":
            values[0] = float("nan")
        if mode == "short":
            values.pop()
        (output / "latents.f32").write_bytes(struct.pack("<" + "f" * len(values), *values))
        receipt = {
            "device": "CPU protocol fixture, no GPU inference",
            "image_rows": count // 128,
            "prompt_rows": header["prompt"]["shape"][0],
            "negative_rows": header["negative"]["shape"][0],
            "load_s": 0.01, "refine_s": 0.02, "denoise_s": steps * 0.03,
            "run": [{"wall_s": 0.03, "gpu_s": 0.02, "host_s": 0.01} for _ in range(steps)],
        }
        if mode == "rows":
            receipt["image_rows"] += 1
        if mode == "steps":
            receipt["run"].pop()
        if mode == "time":
            receipt["denoise_s"] = float("inf")
        if mode == "metal":
            receipt["decode_s"] = 1.0
        (output / "run.json").write_text(json.dumps(receipt))
        print(f"step 1/{steps}: 0.030 s", file=sys.stderr, flush=True)
        if mode == "cancel":
            time.sleep(60)
    '''), encoding="utf-8")
    executable.chmod(0o700)
    checkpoint = write_checkpoint(tmp_path / "moss.safetensors")
    work = tmp_path / "work"
    return checkpoint, executable, work


def generate(setup, *, mode="good", progress=lambda current, total: None):
    checkpoint, executable, work = setup
    positive = [[torch.zeros(1, 3, 4096), {"pooled_output": None}]]
    negative = [[torch.zeros(1, 1, 4096), {}]]
    return sample_moss(checkpoint, executable, positive, negative, 48, 32, 4, 42, work,
                       interrupt=lambda: None, progress=progress, environment={"MOSS_FIXTURE_MODE": mode})


def test_real_subprocess_case_output_layout_and_receipt(setup):
    updates = []
    before = os.environ.get("MOSS_FIXTURE_MODE")
    latent, receipt = generate(setup, progress=lambda step, total: updates.append((step, total)))
    samples = latent["samples"]
    assert samples.shape == (1, 128, 2, 3)
    assert samples[0, 7, 1, 2] == 647
    assert updates == [(1, 4)]
    assert isinstance(receipt, dict)
    assert receipt["seed"] == 42
    assert receipt["guidance"] == 3.0
    assert receipt["negative_skip_layer"] == 22
    assert receipt["backend"] == "tensorfold-moss-cuda"
    assert receipt["requested_steps"] == receipt["effective_steps"] == 4
    assert len(receipt["noise_sha256"]) == 64
    assert len(receipt["checkpoint_header_sha256"]) == 64
    assert os.environ.get("MOSS_FIXTURE_MODE") == before
    assert list(setup[2].iterdir()) == []
    assert setup[0].exists()


@pytest.mark.parametrize("mode,match", [
    ("nan", "non-finite"), ("short", "truncated"), ("rows", "image_rows"),
    ("steps", "number of denoising steps"), ("time", "timing"),
    ("metal", "CUDA denoiser"), ("exit", "status 7"),
])
def test_native_failures_cannot_return_success(setup, mode, match):
    with pytest.raises((ValueError, RuntimeError), match=match):
        generate(setup, mode=mode)
    assert list(setup[2].iterdir()) == []
    assert setup[0].exists()


def test_cancellation_stops_owned_process_and_removes_case(setup):
    class Cancelled(Exception):
        pass

    def cancel(current, total):
        raise Cancelled

    with pytest.raises(Cancelled):
        generate(setup, mode="cancel", progress=cancel)
    assert list(setup[2].iterdir()) == []
    assert setup[0].exists()


def test_invalid_conditioning_never_launches_native(setup, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid input launched the native executable")

    monkeypatch.setattr(process, "run_native", unexpected)
    with pytest.raises(ValueError, match="MOSS text encoder"):
        sample_moss(setup[0], setup[1], [[torch.zeros(1, 2, 768), {}]], [], 32, 32, 4, 0,
                    setup[2], interrupt=lambda: None, progress=lambda current, total: None)


def test_missing_or_malformed_receipt_is_refused(tmp_path):
    path = tmp_path / "run.json"
    with pytest.raises(ValueError, match="receipt file"):
        read_receipt(path, {})
    path.write_text("not JSON")
    with pytest.raises(ValueError, match="malformed"):
        read_receipt(path, {})
