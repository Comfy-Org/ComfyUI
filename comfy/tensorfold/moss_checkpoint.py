from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct


CONFIG = {
    "_class_name": "MOSSImageTransformer", "dim": 3072, "ffn_dim": 8192,
    "num_heads": 24, "num_layers": 42, "text_dim": 4096, "freq_dim": 256,
    "latent_channels": 128, "rope_base": 10000.0,
}


def tensor_schema():
    h, f, d = 3072, 8192, 128
    joint = {
        "proj.q.weight": [h, h], "proj.k.weight": [h, h],
        "proj.v.weight": [h, h], "proj.o.weight": [h, h],
        "proj.ffn_up_gate.weight": [2 * f, h], "proj.ffn_down.weight": [h, f],
        "norm1.weight": [h], "norm2.weight": [h],
        "q_norm.weight": [d], "k_norm.weight": [d],
    }
    text = {
        "norm1.weight": [h], "norm2.weight": [h],
        "q_norm.weight": [d], "k_norm.weight": [d],
        "qkv.weight": [3 * h, h], "qkv.bias": [3 * h],
        "proj.weight": [h, h], "proj.bias": [h],
        "ffn_gate.weight": [f, h], "ffn_gate.bias": [f],
        "ffn_up.weight": [f, h], "ffn_up.bias": [f],
        "ffn_down.weight": [h, f], "ffn_down.bias": [h],
    }
    shapes = {f"layers.{i}.{key}": shape for i in range(42) for key, shape in joint.items()}
    shapes.update({f"text_embedder.blocks.{i}.{key}": shape for i in range(2) for key, shape in text.items()})
    shapes.update({
        "text_embedder.proj.weight": [h, 4096], "text_embedder.proj.bias": [h],
        "video_embedder.weight": [h, 128], "video_embedder.bias": [h],
        "final_norm_video.weight": [h], "final_linear_video.weight": [128, h],
        "time_embedding.0.weight": [h, 256], "time_embedding.0.bias": [h],
        "time_embedding.2.weight": [h, h], "time_embedding.2.bias": [h],
        "shared_block_modulation.weight": [6 * h, h], "shared_block_modulation.bias": [6 * h],
        "video_head_modulation.weight": [2 * h, h], "video_head_modulation.bias": [2 * h],
    })
    result = {key: ("BF16", shape) for key, shape in shapes.items()}
    result["rope.bands"] = ("F32", [16])
    return result


def _identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate MOSS checkpoint field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    identity: tuple
    header_sha256: str

    def verify(self):
        if _identity(self.path.stat()) != self.identity:
            raise RuntimeError("The MOSS checkpoint changed during generation; reload the workflow")

    def stage(self, directory):
        self.verify()
        directory.mkdir(mode=0o700)
        destination = directory / "diffusion_pytorch_model.safetensors"
        try:
            os.link(self.path, destination)
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK):
                raise
            destination.symlink_to(self.path)
        (directory / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        self.verify()


def read_checkpoint(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("MOSS requires a regular, single-file safetensors checkpoint")
    with path.open("rb") as source:
        if _identity(os.fstat(source.fileno())) != _identity(before):
            raise RuntimeError("The MOSS checkpoint changed while opening it")
        raw_size = source.read(8)
        if len(raw_size) != 8:
            raise ValueError("The MOSS safetensors header is truncated")
        size = struct.unpack("<Q", raw_size)[0]
        if not 2 <= size <= min(8 << 20, before.st_size - 8):
            raise ValueError("The MOSS safetensors header has an invalid size")
        raw_header = source.read(size)
    header = json.loads(raw_header, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError("The MOSS safetensors header must be an object")
    if size % 2:
        raise ValueError("MOSS checkpoint data must align to two bytes for native BF16 host reads")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("MOSS safetensors metadata must contain string values")
    if "layers.0.proj.qkv.weight" in header:
        raise ValueError("Prepare fused MOSS BF16 checkpoints with TensorFold prepare_moss.py before generation")
    expected = tensor_schema()
    if header.keys() != expected.keys():
        missing = sorted(expected.keys() - header.keys())
        extra = sorted(header.keys() - expected.keys())
        detail = f"missing {missing[:2]}" if missing else f"unexpected {extra[:2]}"
        raise ValueError(f"TensorFold needs the original MOSS-Image-Base BF16 transformer: {detail}")
    spans = []
    for name, (dtype, shape) in expected.items():
        tensor = header[name]
        if not isinstance(tensor, dict) or tensor.get("dtype") != dtype or tensor.get("shape") != shape:
            raise ValueError(f"MOSS tensor {name} must be {dtype} with shape {shape}")
        if any(type(n) is not int for n in tensor["shape"]):
            raise ValueError(f"Invalid MOSS tensor dimensions: {name}")
        span = tensor.get("data_offsets")
        count = math.prod(shape) * (4 if dtype == "F32" else 2)
        if not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span):
            raise ValueError(f"Invalid MOSS tensor offsets: {name}")
        if span[0] < 0 or span[1] - span[0] != count:
            raise ValueError(f"Invalid MOSS tensor extent: {name}")
        spans.append(tuple(span))
    end = 0
    for start, stop in sorted(spans):
        if start != end:
            raise ValueError("MOSS tensor storage overlaps or contains a gap")
        end = stop
    if 8 + size + end != before.st_size:
        raise ValueError("MOSS tensor data is truncated or contains trailing bytes")
    result = Checkpoint(path, _identity(before), hashlib.sha256(raw_header).hexdigest())
    result.verify()
    return result
