import hashlib

import torch


def _linspace(end, count):
    if count == 1:
        return torch.ones(1, device="cpu", dtype=torch.float32)
    end = torch.tensor(end, device="cpu", dtype=torch.float32)
    step = (end - 1) / (count - 1)
    index = torch.arange(count, device="cpu", dtype=torch.float32)
    # Separate float32 operations preserve the scalar publisher grid on CPUs that fuse linspace's multiply-add.
    return torch.where(index < count // 2, 1 + step * index, end - step * (count - 1 - index))


def schedule(steps):
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("MOSS steps must be an integer from 1 to 1000")
    # The publisher's V2 schedule snaps shifted sigmas onto its 1000-point training grid.
    grid = _linspace(5 / 1004, steps)
    proposed = 5 * grid / (1 + 4 * grid)
    training = _linspace(1 / 1000, 1000).flip(0)
    upper = torch.searchsorted(training, proposed).clamp(max=999)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((proposed - training[lower]).abs() <= (training[upper] - proposed).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return sigmas * 1000, torch.cat((sigmas, torch.zeros(1, device="cpu", dtype=torch.float32)))


def dimensions(width, height):
    if any(type(n) is not int or n <= 0 or n % 16 for n in (width, height)):
        raise ValueError("MOSS width and height must be positive multiples of 16")
    rows, columns = height // 16, width // 16
    if rows * columns > (2**31 - 1) // (2 * 8192) - 4096:
        raise ValueError("MOSS image dimensions exceed the native kernel's address limits")
    return rows, columns


def conditioning(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError(f"TensorFold MOSS needs one unscheduled {name} conditioning entry")
    entry = value[0]
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise ValueError(f"Invalid MOSS {name} conditioning entry")
    tensor, metadata = entry
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid MOSS {name} conditioning metadata")
    unsupported = {k for k, v in metadata.items() if k not in {"pooled_output", "attention_mask"} and v is not None}
    if unsupported:
        raise ValueError(f"TensorFold MOSS does not support conditioning options: {', '.join(sorted(unsupported))}")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != 4096:
        raise ValueError(f"MOSS {name} conditioning must have shape [1, tokens, 4096]; use the MOSS text encoder")
    if not tensor.is_floating_point():
        raise ValueError(f"MOSS {name} conditioning must contain floating-point rows")
    tensor = tensor.detach().to(device="cpu", dtype=torch.bfloat16)[0]
    mask = metadata.get("attention_mask")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (1, tensor.shape[0]):
            raise ValueError(f"MOSS {name} attention mask must have shape [1, tokens]")
        mask = mask.detach().to(device="cpu")[0]
        if not torch.all((mask == 0) | (mask == 1)).item():
            raise ValueError(f"MOSS {name} attention mask must contain only zero or one")
        tensor = tensor[mask.bool()]
    if not 1 <= tensor.shape[0] <= 4096:
        raise ValueError(f"MOSS {name} conditioning needs 1 to 4096 unmasked tokens")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"MOSS {name} conditioning contains non-finite values")
    return tensor.contiguous()


def make_case(positive, negative, width, height, steps, seed):
    rows, columns = dimensions(width, height)
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
        raise ValueError("MOSS seed must be an unsigned 64-bit integer")
    timesteps, sigmas = schedule(steps)
    prompt = conditioning(positive, "positive")
    negative_prompt = conditioning(negative, "negative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, 128, rows, columns), generator=generator, device="cpu", dtype=torch.float32)
    return {
        "prompt": prompt,
        "negative": negative_prompt,
        "latents": noise[0].permute(1, 2, 0).contiguous(),
        "timesteps": timesteps,
        "sigmas": sigmas,
    }


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def unpack_latents(tensor, width, height):
    rows, columns = dimensions(width, height)
    if tensor.numel() != rows * columns * 128 or tensor.dtype != torch.float32:
        raise ValueError("TensorFold returned an invalid MOSS latent buffer")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("TensorFold returned non-finite MOSS latents")
    return {"samples": tensor.reshape(rows, columns, 128).permute(2, 0, 1).unsqueeze(0).contiguous()}
