"""comfyui-vslinx-nodes' MultiDiffusion tiling (nodes/multidiffusion_tiled_hires.py and the helpers in
nodes/anima_lllite_tiled_sampler.py), ported to the V2 API: the model wrapper on its own, as a node
that returns a MODEL.

The original installs its wrapper with ``model.set_model_unet_function_wrapper`` and chains a wrapper
already on the model by reading ``model.model_options``. The port uses ``eng.model.set_unet_wrapper``,
which chains for it. The tiling code is the original's.
"""
import torch

from comfy_api.index import eng, io


def _md_spans(total, n, ext):
    base = max(1, total // n)
    spans = []
    for i in range(n):
        s = 0 if i == 0 else i * base - ext
        e = total if i == n - 1 else (i + 1) * base + ext
        spans.append((max(0, s), min(total, e)))
    return spans


def _md_weight_1d(length, taper_left, taper_right, device, dtype):
    w = torch.ones(length, device=device, dtype=dtype)
    tl = min(int(taper_left), length // 2)
    tr = min(int(taper_right), length // 2)
    if tl > 0:
        w[:tl] = torch.linspace(0, 1, tl + 2, device=device, dtype=dtype)[1:-1]
    if tr > 0:
        w[length - tr:] = torch.linspace(1, 0, tr + 2, device=device, dtype=dtype)[1:-1]
    return w


def multidiffusion_wrapper(latent_h, latent_w, rows, columns, overlap, overlap_x=0, overlap_y=0):
    tile_hl = max(1, latent_h // rows)
    tile_wl = max(1, latent_w // columns)
    ov_hl = 0 if rows == 1 else min(int(tile_hl * overlap) + overlap_y // 8, tile_hl // 2)
    ov_wl = 0 if columns == 1 else min(int(tile_wl * overlap) + overlap_x // 8, tile_wl // 2)
    ys = _md_spans(latent_h, rows, ov_hl)
    xs = _md_spans(latent_w, columns, ov_wl)
    state = {"tag": None, "tiles": None}

    def build_tiles(device, dtype):
        tiles = []
        for i, (y0, y1) in enumerate(ys):
            ty_l = (ys[i - 1][1] - y0) if i > 0 else 0
            ty_r = (y1 - ys[i + 1][0]) if i < rows - 1 else 0
            wy = _md_weight_1d(y1 - y0, ty_l, ty_r, device, dtype)
            for j, (x0, x1) in enumerate(xs):
                tx_l = (xs[j - 1][1] - x0) if j > 0 else 0
                tx_r = (x1 - xs[j + 1][0]) if j < columns - 1 else 0
                wx = _md_weight_1d(x1 - x0, tx_l, tx_r, device, dtype)
                weight = wy[:, None] * wx[None, :]
                tiles.append({"y0": y0, "y1": y1, "x0": x0, "x1": x1, "weight": weight})
        return tiles

    def wrapper(apply_model, args):
        x_in = args["input"]
        t = args["timestep"]
        c = args["c"]
        device, dtype = x_in.device, x_in.dtype
        tag = (device, dtype)
        if state["tag"] != tag:
            state["tiles"] = build_tiles(device, dtype)
            state["tag"] = tag
        acc = torch.zeros_like(x_in)
        lead = (1,) * (x_in.ndim - 2)
        wsum = torch.zeros(lead + (x_in.shape[-2], x_in.shape[-1]), device=device, dtype=dtype)
        for td in state["tiles"]:
            y0, y1, x0, x1 = td["y0"], td["y1"], td["x0"], td["x1"]
            x_tile = x_in[..., y0:y1, x0:x1]
            eps = apply_model(x_tile, t, **c)
            wb = td["weight"].view(lead + td["weight"].shape)
            acc[..., y0:y1, x0:x1] += eps * wb
            wsum[..., y0:y1, x0:x1] += wb
        return acc / wsum.clamp(min=1e-6)

    return wrapper


class MultiDiffusionModel(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="VSLinxMultiDiffusionModel", category="sampling",
                         inputs=[io.Model.Input("model"), io.Latent.Input("latent"),
                                 io.Int.Input("rows", default=2, min=1, max=8),
                                 io.Int.Input("columns", default=2, min=1, max=8),
                                 io.Float.Input("overlap", default=0.25, min=0.0, max=0.5)],
                         outputs=[io.Model.Output()])

    @classmethod
    async def execute(cls, model, latent, rows, columns, overlap):
        h, w = int(latent["samples"].shape[-2]), int(latent["samples"].shape[-1])
        return io.NodeOutput(await eng.model.set_unet_wrapper(model, multidiffusion_wrapper(h, w, rows, columns, overlap)))
