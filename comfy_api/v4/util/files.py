"""Weight files: reading and writing `{name: tensor}`."""

import torch

import comfy.utils as _comfy_utils

# Arguments go to ComfyUI by name, so a parameter renamed there fails here instead of shifting.


def load_torch_file(
    ckpt: str,
    safe_load: bool = False,
    device: torch.device | None = None,
    return_metadata: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    """Read a `.safetensors` or `.sft` file, or a torch file, into `{name: tensor}`.

    Torch files are always read with `weights_only=True`, so `safe_load` changes nothing. A
    torch file's `"state_dict"` entry, or its only entry if that is a dict, is what is
    returned. With `return_metadata`, returns `(weights, metadata)`; metadata is None for torch
    files, and None or `{}` for a safetensors file without metadata.
    """
    return _comfy_utils.load_torch_file(
        ckpt=ckpt, safe_load=True, device=device, return_metadata=return_metadata
    )


def save_torch_file(
    sd: dict[str, torch.Tensor], ckpt: str, metadata: dict[str, str] | None = None
) -> None:
    """Write `{name: tensor}` to `ckpt` as safetensors, whatever its extension, with `metadata`.

    Raises `RuntimeError` if tensors share memory, such as tied weights, `ValueError` for a tensor
    that is not contiguous, and `TypeError` for a `metadata` value that is not a string.
    `load_torch_file` reads the file back only under a `.safetensors` or `.sft` name.
    """
    _comfy_utils.save_torch_file(sd=sd, ckpt=ckpt, metadata=metadata)


__all__ = ["load_torch_file", "save_torch_file"]
