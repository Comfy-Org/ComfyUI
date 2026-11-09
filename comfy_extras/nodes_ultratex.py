import torch
from typing_extensions import override

import comfy.utils
import node_helpers
from comfy.ldm.flux.model import views_to_grid
from comfy_api.latest import ComfyExtension, io

VIEW_GRID = [(0, 0), (0, 2), (1, 0), (0, 1), (1, 1), (1, 2)]
VIEW_NAMES = ["front", "right", "back", "left", "top", "bottom"]


def _on_white(image, mask):
    return image[..., :3] * mask[..., None] + (1.0 - mask[..., None])


class UltraTexConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UltraTexConditioning",
            display_name="UltraTex Conditioning",
            category="3d/texturing",
            description=(
                "Conditioning for FLUX.2 Klein 9B with the UltraTex LoRA: paints a reference image's look onto a mesh, as one "
                "image per view. Views come from Render Mesh (world normal) with a Create Camera Orbit of 4 plus top and "
                "bottom; bake the result with Bake Texture From Views."),
            inputs=[
                io.Conditioning.Input("conditioning", tooltip="Text conditioning; UltraTex was trained with an empty prompt."),
                io.Vae.Input("vae"),
                io.Image.Input("normals", tooltip="Render Mesh 'world normal' renders: the 6 canonical views, or the 4 side views."),
                io.Mask.Input("masks", tooltip="Render Mesh masks of the same views."),
                io.Image.Input("reference", tooltip="The object as it should look, seen from the front camera's side."),
                io.Mask.Input("reference_mask", optional=True,
                              tooltip="Object mask of the reference (white = object)"),
                io.DynamicCombo.Input("extra_views", options=[
                    io.DynamicCombo.Option("off", []),
                    io.DynamicCombo.Option("on", [
                        io.Image.Input(name, optional=True, tooltip=f"The object seen from the {name} view")
                        for name in VIEW_NAMES[1:]
                    ]),
                ], tooltip="More references: the object seen from the other views, each tied to its view as the reference is to the front."),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning, vae, normals, masks, reference, extra_views, reference_mask=None) -> io.NodeOutput:
        if normals.shape[0] not in (4, len(VIEW_GRID)) or masks.shape[0] != normals.shape[0]:
            raise ValueError(f"UltraTex needs the {len(VIEW_GRID)} canonical views (Create Camera Orbit count 4, top and bottom) or the "
                             f"4 side views, got {normals.shape[0]} normal images and {masks.shape[0]} masks.")
        # 4 side views: the top and bottom tiles stay empty and drop out with the background
        grid = VIEW_GRID[:normals.shape[0]]
        h, w = normals.shape[1:3]
        masks = comfy.utils.common_upscale(masks.to(normals)[:, None], w, h, "bilinear", "disabled")[:, 0]
        # UltraTex was trained on world normals with the up axis negated.
        normals = normals[..., :3].clone()
        normals[..., 1] = 1.0 - normals[..., 1]
        normal_atlas = views_to_grid(vae.encode(_on_white(normals, masks)), grid)

        reference = reference[:1]
        if reference_mask is None:
            reference_mask = torch.ones_like(reference[..., 0])
        else:
            reference_mask = comfy.utils.common_upscale(reference_mask[:1].to(reference)[:, None], reference.shape[2], reference.shape[1], "bilinear", "disabled")[:, 0]
        ref_latents = [normal_atlas, vae.encode(_on_white(reference, reference_mask))]
        ref_masks = [views_to_grid(masks[:, None], grid)[:, 0], reference_mask]
        # the reference shares the front tile's positions; the normal atlas has none
        ref_views = [None, 0]
        for view_index, name in enumerate(VIEW_NAMES[1:len(grid)], 1):
            image = extra_views.get(name)
            if image is not None:
                image = image[:1]
                view_mask = comfy.utils.common_upscale(masks[view_index:view_index + 1].to(image)[:, None], image.shape[2], image.shape[1], "bilinear", "disabled")[:, 0]
                ref_latents.append(vae.encode(_on_white(image, view_mask)))
                ref_masks.append(view_mask)
                ref_views.append(view_index)

        # Dropped background tokens land on this latent: black is the fine-tuned decoder's background.
        background = vae.encode(torch.zeros((1, h, w, 3)))
        conditioning = node_helpers.conditioning_set_values(conditioning, {
            "reference_latents": ref_latents,
            "reference_latents_method": "index",
            "view_grid": grid,
            "token_masks": {"target": masks, "refs": ref_masks},
            "reference_views": ref_views,
            "background_latent": background,
        })
        return io.NodeOutput(conditioning)


class UltraTexExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [UltraTexConditioning]


async def comfy_entrypoint() -> UltraTexExtension:
    return UltraTexExtension()
