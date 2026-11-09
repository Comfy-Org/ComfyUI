"""Y7-SBS-2Dto3D's coordinate-grid cache (nodes/sbs.py, ``get_grid_gs``), ported to the V2 API.

The original keeps grids in a module-level dict keyed by ``(h, w, dtype)``, which leaves out the
device, so a CPU run and a GPU run share an entry. The port keeps them in ``eng.cache``, keyed by the
device too. The grid is built exactly as the original builds it.
"""
import torch

from comfy_api.index import eng, io


def build_grid_gs(h, w, dtype, device):
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                          torch.arange(w, device=device, dtype=dtype), indexing="ij")
    return (y.unsqueeze(0).unsqueeze(0), x.unsqueeze(0).unsqueeze(0))


async def get_grid_gs(h, w, dtype, device):
    key = f"grid_gs:{h}x{w}:{dtype}:{device}"
    grid = await eng.cache.get(key)
    if grid is None:
        grid = build_grid_gs(h, w, dtype, device)
        await eng.cache.set(key, list(grid))
    return tuple(grid)


class Y7Grid(io.ComfyNode):
    """The cached grid for an image's size: the step Y7's side-by-side nodes start from."""
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Y7Grid", inputs=[io.Image.Input("image")],
                         outputs=[io.Image.Output(), io.Image.Output()])

    @classmethod
    async def execute(cls, image):
        y, x = await get_grid_gs(image.shape[1], image.shape[2], torch.float16, image.device)
        return io.NodeOutput(y, x)
