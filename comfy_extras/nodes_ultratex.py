import torch
from typing_extensions import override

import node_helpers
from comfy.ldm.flux.model import views_to_grid
from comfy_api.latest import ComfyExtension, io

VIEW_GRID = [(0, 0), (0, 2), (1, 0), (0, 1), (1, 1), (1, 2)]
VIEW_NAMES = ["front", "right", "back", "left", "top", "bottom"]


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
                "Conditioning for texturing a mesh with FLUX.2 Klein base 9B and the UltraTex LoRA, from Render Mesh's six views and a reference image of the textured object. "
                "Extra views add images of the object from the other canonical views as further references."),
            inputs=[
                io.Conditioning.Input("conditioning", tooltip="Text conditioning; UltraTex was trained with an empty prompt."),
                io.Vae.Input("vae"),
                io.Image.Input("normals", tooltip="Render Mesh world normal images"),
                io.Mask.Input("masks", tooltip="Render Mesh object masks of the same views."),
                io.Image.Input("reference", tooltip="Image of the object with the look to texture it with"),
                io.Mask.Input("reference_mask", optional=True,
                              tooltip="Object mask of the reference (white = object)"),
                io.DynamicCombo.Input("extra_views", options=[
                    io.DynamicCombo.Option("off", []),
                    io.DynamicCombo.Option("on", [
                        io.Image.Input(name, optional=True, tooltip=f"The object seen from the {name} view")
                        for name in VIEW_NAMES[1:]
                    ]),
                ], tooltip="Further references, each masked by its view's silhouette and placed over its tile, like the reference over the front."),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning, vae, normals, masks, reference, extra_views, reference_mask=None) -> io.NodeOutput:
        if normals.shape[0] not in (4, len(VIEW_GRID)) or masks.shape[0] != normals.shape[0]:
            raise ValueError(f"UltraTex needs the {len(VIEW_GRID)} canonical views (Create Camera Orbit count 4, top and bottom) or the "
                             f"4 side views, got {normals.shape[0]} normal images and {masks.shape[0]} masks.")
        # without poles their tiles stay empty and are dropped like background
        grid = VIEW_GRID[:normals.shape[0]]
        h, w = normals.shape[1:3]
        masks = _resize_mask(masks.to(normals), h, w)
        # UltraTex was trained on world normals with the up axis negated.
        normals = normals[..., :3].clone()
        normals[..., 1] = 1.0 - normals[..., 1]
        normal_atlas = views_to_grid(vae.encode(_on_white(normals, masks)), grid)

        reference = reference[:1]
        if reference_mask is None:
            reference_mask = torch.ones(reference.shape[:3])
        reference_mask = _resize_mask(reference_mask[:1].to(reference), reference.shape[1], reference.shape[2])
        ref_latents = [normal_atlas, vae.encode(_on_white(reference, reference_mask))]
        ref_masks = [views_to_grid(masks[:, None], grid)[:, 0], reference_mask]
        # the reference is the front view's: it shares positions with the front tile
        ref_views = [None, 0]
        for view_index, name in enumerate(VIEW_NAMES[:len(grid)]):
            image = extra_views.get(name)
            if image is not None:
                image = image[:1]
                view_mask = _resize_mask(masks[view_index:view_index + 1].to(image), image.shape[1], image.shape[2])
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
