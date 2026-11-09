#Taken from: https://github.com/tfernd/HyperTile/

import math
from typing_extensions import override
from einops import rearrange
# Use torch rng for consistency across generations
from torch import randint
from comfy_api.latest import ComfyExtension, io

def random_divisor(value: int, min_value: int, /, max_options: int = 1) -> int:
    min_value = min(min_value, value)

    # All big divisors of value (inclusive)
    divisors = [i for i in range(min_value, value + 1) if value % i == 0]

    ns = [value // i for i in divisors[:max_options]]  # has at least 1 element

    if len(ns) - 1 > 0:
        idx = randint(low=0, high=len(ns) - 1, size=(1,)).item()
    else:
        idx = 0

    return ns[idx]

class HyperTile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HyperTile",
            category="model/patch/unet",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("tile_size", default=256, min=1, max=2048, advanced=True),
                io.Int.Input("swap_size", default=2, min=1, max=128, advanced=True),
                io.Int.Input("max_depth", default=0, min=0, max=10, advanced=True),
                io.Boolean.Input("scale_depth", default=False, advanced=True),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, tile_size, swap_size, max_depth, scale_depth) -> io.NodeOutput:
        latent_tile_size = max(32, tile_size) // 8
        temp = None

        def hypertile_in(q, k, v, extra_options):
            nonlocal temp
            model_chans = q.shape[-2]
            orig_shape = extra_options['original_shape']
            apply_to = []
            for i in range(max_depth + 1):
                apply_to.append((orig_shape[-2] / (2 ** i)) * (orig_shape[-1] / (2 ** i)))

            if model_chans in apply_to:
                shape = extra_options["original_shape"]
                aspect_ratio = shape[-1] / shape[-2]

                hw = q.size(1)
                h, w = round(math.sqrt(hw * aspect_ratio)), round(math.sqrt(hw / aspect_ratio))

                factor = (2 ** apply_to.index(model_chans)) if scale_depth else 1
                nh = random_divisor(h, latent_tile_size * factor, swap_size)
                nw = random_divisor(w, latent_tile_size * factor, swap_size)

                if nh * nw > 1:
                    q = rearrange(q, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)
                    temp = (nh, nw, h, w)
                return q, k, v

            return q, k, v
        def hypertile_out(out, extra_options):
            nonlocal temp
            if temp is not None:
                nh, nw, h, w = temp
                temp = None
                out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)
            return out


        m = model.clone()
        m.set_model_attn1_patch(hypertile_in)
        m.set_model_attn1_output_patch(hypertile_out)
        return (m, )


class HyperTileExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HyperTile,
        ]


async def comfy_entrypoint() -> HyperTileExtension:
    return HyperTileExtension()
