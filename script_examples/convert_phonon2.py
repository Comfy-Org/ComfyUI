"""Convert the Phonon-2 `model.fermion` container into a dense fp16 safetensors file for ComfyUI.

    tar --zstd -xf phonon-2.bps.tar.zst
    python convert_phonon2.py model.fermion phonon-2.safetensors

Put the result in models/audio_encoders and load it with Load Audio Encoder.
"""
import json
import sys

import numpy as np
import torch
from safetensors.torch import save_file

FORMAT = "fermion-five-value-parakeet-v1"


def unpack_five_value(blob, shape):
    out_dim, in_dim = shape
    row_bytes = (in_dim + 4) // 5
    packed = np.frombuffer(blob, dtype=np.uint8, count=out_dim * row_bytes).reshape(out_dim, row_bytes).astype(np.uint16)
    codes = np.stack([(packed // 3 ** k) % 3 for k in range(5)], axis=-1).reshape(out_dim, -1)[:, :in_dim]
    nonzero = codes != 1
    offset = out_dim * row_bytes
    bit_bytes = (int(nonzero.sum()) + 7) // 8
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8, count=bit_bytes, offset=offset), bitorder="little")[:int(nonzero.sum())].astype(bool)
    offset += bit_bytes
    lo = np.frombuffer(blob, dtype=np.float16, count=out_dim, offset=offset)
    hi = np.frombuffer(blob, dtype=np.float16, count=out_dim, offset=offset + 2 * out_dim)
    assert offset + 4 * out_dim == len(blob)
    is_hi = np.zeros(codes.shape, dtype=bool)
    is_hi[nonzero] = bits
    magnitude = np.where(is_hi, hi[:, None], lo[:, None])
    return (codes.astype(np.int8) - 1).astype(np.float16) * magnitude


def unpack_int(blob, shape, bits):
    out_dim = shape[0]
    total = int(np.prod(shape))
    body, scales = blob[:-2 * out_dim], np.frombuffer(blob[-2 * out_dim:], dtype=np.float16)
    if bits == 8:
        q = np.frombuffer(body, dtype=np.int8).astype(np.int32)[:total]
    elif bits == 6:
        b = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        word = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        q = np.stack([(word >> s) & 0x3F for s in (0, 6, 12, 18)], axis=1).ravel()[:total].astype(np.int32) - 32
    else:
        raise ValueError(f"unsupported int{bits} record")
    return (q.reshape(out_dim, -1).astype(np.float32) * scales.astype(np.float32)[:, None]).reshape(shape)


def main(src, dst):
    sd = {}
    with open(src, "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
        assert header["format"] == FORMAT, header["format"]
        for e in header["index"]:
            blob = f.read(e["b"])
            name, kind, shape = e["n"], e["k"], tuple(e["shape"])
            if kind == "five_value":
                name, arr = name + ".weight", unpack_five_value(blob, shape)
            elif kind.startswith("int"):
                arr = unpack_int(blob, shape, int(kind[3:]))
            else:
                arr = np.frombuffer(blob, dtype=np.float16).reshape(shape)
            if name.endswith("num_batches_tracked"):
                sd[name] = torch.zeros((), dtype=torch.int64)
                continue
            if arr.ndim == 2 and ".conv.pointwise_conv" in name:
                arr = arr[:, :, None]
            sd[name] = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float16)))
        assert f.read(1) == b""
    save_file(sd, dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
