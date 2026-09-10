import torch
from einops import rearrange, repeat
import comfy.rmsnorm


def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"

    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad

    return torch.nn.functional.pad(img, pad, mode=padding_mode)


rms_norm = comfy.rmsnorm.rms_norm

def process_img(x, index=0, h_offset=0, w_offset=0, patch_size=2, transformer_options={}, num_axes=3):
    bs, c, h, w = x.shape
    x = pad_to_patch_size(x, (patch_size, patch_size))

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
    h_len = ((h + (patch_size // 2)) // patch_size)
    w_len = ((w + (patch_size // 2)) // patch_size)

    h_offset = ((h_offset + (patch_size // 2)) // patch_size)
    w_offset = ((w_offset + (patch_size // 2)) // patch_size)

    steps_h = h_len
    steps_w = w_len

    rope_options = transformer_options.get("rope_options", None)
    if rope_options is not None:
        h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
        w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0

        index += rope_options.get("shift_t", 0.0)
        h_offset += rope_options.get("shift_y", 0.0)
        w_offset += rope_options.get("shift_x", 0.0)

    img_ids = torch.zeros((steps_h, steps_w, num_axes), device=x.device, dtype=x.dtype)
    img_ids[:, :, 0] = img_ids[:, :, 1] + index
    img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=steps_h, device=x.device, dtype=x.dtype).unsqueeze(1)
    img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=steps_w, device=x.device, dtype=x.dtype).unsqueeze(0)
    return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)
