import torch
from typing_extensions import override

import node_helpers
from comfy.ldm.flux.model import views_to_grid
from comfy_api.latest import ComfyExtension, io

# Tile (row, col) in UltraTex's 2x3 atlas of each Render Mesh canonical view (views 4 with top_bottom: front, right,
# back, left, top, bottom). Row 0 holds front, left, right; row 1 back, top, bottom.
VIEW_GRID = [(0, 0), (0, 2), (1, 0), (0, 1), (1, 1), (1, 2)]


def _on_white(image, mask):
    return image[..., :3] * mask[..., None] + (1.0 - mask[..., None])


def _resize_mask(mask, h, w):
    if mask.shape[-2:] == (h, w):
        return mask
    return torch.nn.functional.interpolate(mask[:, None], size=(h, w), mode="bilinear")[:, 0]


class UltraTexConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UltraTexConditioning",
            display_name="UltraTex Conditioning",
            category="3d/texturing",
            description=(
                "Conditioning for texturing a mesh with FLUX.2 Klein base 9B and the UltraTex LoRA, from Render Mesh's six "
                "canonical views (world normal mode, views 4, top_bottom) and a reference image of the textured object. "
                "An orbit camera at yaw 0, pitch 0, fov 35 and distance 1.6 matches the training framing of a unit-sized "
                "mesh. Sample six latents at the view size, then bake the decoded views with Bake Texture From Views."),
            inputs=[
                io.Conditioning.Input("conditioning", tooltip="Text conditioning; UltraTex was trained with an empty prompt."),
                io.Vae.Input("vae"),
                io.Image.Input("normals", tooltip="Render Mesh world normal images of the six canonical views, in its order."),
                io.Mask.Input("masks", tooltip="Render Mesh object masks of the same six views."),
                io.Image.Input("reference", tooltip="Image of the object with the look to texture it with, e.g. an edited render."),
                io.Mask.Input("reference_mask", optional=True,
                              tooltip="Object mask of the reference (1 = object; Load Image's mask is the inverse, "
                                      "so pass it through Invert Mask). Without it every reference token is attended."),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning, vae, normals, masks, reference, reference_mask=None) -> io.NodeOutput:
        if normals.shape[0] != len(VIEW_GRID) or masks.shape[0] != len(VIEW_GRID):
            raise ValueError(f"UltraTex needs the {len(VIEW_GRID)} canonical views (Render Mesh views 4 with top_bottom), "
                             f"got {normals.shape[0]} normal images and {masks.shape[0]} masks.")
        h, w = normals.shape[1:3]
        masks = _resize_mask(masks.to(normals), h, w)
        # UltraTex was trained on world normals with the up axis negated.
        normals = normals[..., :3].clone()
        normals[..., 1] = 1.0 - normals[..., 1]
        normal_atlas = views_to_grid(vae.encode(_on_white(normals, masks)), VIEW_GRID)

        reference = reference[:1]
        if reference_mask is None:
            reference_mask = torch.ones(reference.shape[:3])
        reference_mask = _resize_mask(reference_mask[:1].to(reference), reference.shape[1], reference.shape[2])
        ref_latent = vae.encode(_on_white(reference, reference_mask))

        # Dropped background tokens land on this latent: black is the fine-tuned decoder's background.
        background = vae.encode(torch.zeros((1, h, w, 3)))
        conditioning = node_helpers.conditioning_set_values(conditioning, {
            "reference_latents": [normal_atlas, ref_latent],
            "reference_latents_method": "index",
            "view_grid": VIEW_GRID,
            "token_masks": {"target": masks, "refs": [views_to_grid(masks[:, None], VIEW_GRID)[:, 0], reference_mask]},
            "background_latent": background,
        })
        return io.NodeOutput(conditioning)


class UltraTexExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [UltraTexConditioning]


async def comfy_entrypoint() -> UltraTexExtension:
    return UltraTexExtension()
