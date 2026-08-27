from __future__ import annotations

import logging


def _place(model, device, dtype):
    import torch

    model = model.eval().to(device=device, dtype=dtype)
    if torch.device(device).type == "cuda":
        model.to(memory_format=torch.channels_last)
    return model


def _build_flat_decoder(state_dict):
    import torch.nn as nn
    from comfy.taesd.taesd import Block, Clamp, conv

    by_index = {}
    for key, value in state_dict.items():
        head, _, rest = key.partition(".")
        if not head.isdigit():
            raise ValueError(
                f"unexpected tiny-VAE decoder key {key!r}")
        by_index.setdefault(int(head), {})[rest] = value
    if not by_index:
        raise ValueError("tiny-VAE decoder state dict is empty")

    modules = []
    for index in range(max(by_index) + 1):
        entry = by_index.get(index)
        if entry is None:
            modules.append(
                Clamp() if index == 0 else
                nn.ReLU() if index == 2 else
                nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in entry:
            weight = entry["conv.0.weight"]
            kwargs = {"use_midblock_gn": True} \
                if "pool.0.weight" in entry else {}
            modules.append(Block(weight.shape[1], weight.shape[0], **kwargs))
        elif "weight" in entry:
            weight = entry["weight"]
            modules.append(conv(
                weight.shape[1], weight.shape[0], bias="bias" in entry))
        else:
            raise ValueError(
                f"unrecognized tiny-VAE decoder module {index}: "
                f"{sorted(entry)}")
    return nn.Sequential(*modules)


class _FlatDecoder:
    def __init__(self, state_dict, device, dtype):
        first = next(iter(state_dict), "")
        if not first:
            raise ValueError("tiny-VAE decoder state dict is empty")
        if not first.split(".", 1)[0].isdigit():
            prefix = first.split(".", 1)[0] + "."
            state_dict = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        self.device = device
        self.dtype = dtype
        self.model = _build_flat_decoder(state_dict)
        self.model.load_state_dict(state_dict)
        self.model = _place(self.model, device, dtype)
        self.latent_channels = self.model[1].weight.shape[1]

    def decode(self, latent):
        import torch

        result = self.model(latent.to(
            device=self.device, dtype=self.dtype))
        return result.to(device=latent.device, dtype=torch.float32)

    def decode_video(self, latent, frame_indices=None):
        import torch

        values = latent[0]
        indices = (range(values.shape[1]) if frame_indices is None
                   else frame_indices)
        frames = [
            self.decode(values[:, index].unsqueeze(0))[0].movedim(0, -1)
            for index in indices
        ]
        return torch.stack(frames, dim=0)


class _TemporalDecoder:
    def __init__(self, state_dict, device, dtype):
        from comfy.taesd.taehv import TAEHV, conv

        latent_channels = state_dict["decoder.1.weight"].shape[1]
        patch_size = max(1, int(round(
            (state_dict["decoder.22.bias"].shape[0] / 3) ** 0.5)))
        model = TAEHV(latent_channels=latent_channels)
        if model.patch_size != patch_size:
            model.patch_size = patch_size
            model.encoder[0] = conv(
                3 * patch_size ** 2, model.encoder[0].out_channels)
            model.decoder[-1] = conv(
                model.decoder[-1].in_channels, 3 * patch_size ** 2)
        model.load_state_dict(state_dict)
        del model.encoder

        self.device = device
        self.dtype = dtype
        self.model = _place(model, device, dtype)
        self.latent_channels = latent_channels
        self.is_h3 = latent_channels == 24 and patch_size == 2

    def _decode(self, latent):
        import torch

        result = self.model.decode(latent.to(
            device=self.device, dtype=self.dtype))
        return result.to(device=latent.device, dtype=torch.float32)

    def decode(self, latent):
        return self._decode(latent.unsqueeze(2))[:, :, 0]

    def _decode_h3_full(self, latent):
        import torch
        import torch.nn.functional as functional
        import comfy.model_management
        from comfy.taesd.taehv import apply_model_with_memblocks

        model = self.model
        value = model.process_in(latent.to(
            device=self.device, dtype=self.dtype)).movedim(2, 1)
        value = apply_model_with_memblocks(
            model.decoder, value, model.parallel, False,
            output_device=comfy.model_management.intermediate_device(),
            patch_size=model.patch_size, decode=True)
        chunk = 5 * model.t_upscale
        value = functional.pad(
            value, (0, 0, 0, 0, 0, 0, 0, -value.shape[1] % chunk))
        value = value.unflatten(1, (-1, chunk))[
            :, :, model.frames_to_trim:
        ].flatten(1, 2)
        value = value[:, :-3 * model.t_upscale]
        return value.movedim(2, 1).to(
            device=latent.device, dtype=torch.float32)

    def decode_video(self, latent, frame_indices=None):
        import torch

        total = latent.shape[2]
        count = total if frame_indices is None else max(
            1, min(len(frame_indices), total))
        if count == total:
            result = (self._decode_h3_full(latent[:1])
                      if self.is_h3 else self._decode(latent[:1]))
            return result[0].movedim(0, -1).contiguous()
        result = self._decode(latent[:1, :, :count])[0].movedim(0, -1)
        if result.shape[0] > count:
            indices = torch.linspace(
                0, result.shape[0] - 1, count).round().long()
            result = result[indices]
        return result.contiguous()


def load(name):
    import torch
    import comfy.model_management
    import comfy.utils
    import folder_paths

    path = folder_paths.get_full_path("vae_approx", name)
    if path is None:
        raise FileNotFoundError(
            f"tiny VAE {name!r} is not in the vae_approx catalogue")
    state_dict = comfy.utils.load_torch_file(path, safe_load=True)
    device = comfy.model_management.vae_device()
    dtype = comfy.model_management.vae_dtype(
        device, [torch.float16, torch.bfloat16])
    try:
        if ("decoder.1.weight" in state_dict
                and "decoder.22.bias" in state_dict):
            return _TemporalDecoder(state_dict, device, dtype)
        return _FlatDecoder(state_dict, device, dtype)
    except Exception:
        logging.exception("could not load tiny VAE %r", name)
        raise
