import math

import node_helpers
import comfy.utils
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class TextEncodeBooguEdit(io.ComfyNode):
    """Boogu-Image Edit conditioning.

    The edit image is used twice, matching the reference pipeline:
      - Qwen3-VL vision tokens (instruction understanding) -> positive only
      - VAE reference latent (image identity)              -> positive and negative
    The ref latent is in both conds so it cancels under CFG (identity preserved);
    the vision tokens are only in the positive so CFG amplifies the instruction.
    The tokenizer selects the right system prompt automatically (image -> TI2I,
    empty negative -> DROP), so no template plumbing is needed here.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeBooguEdit",
            category="model/conditioning/boogu",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True, advanced=True),
                io.Vae.Input("vae"),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="Reference image(s) to edit. Boogu focuses on one reference per sample; more are allowed.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, vae=None, images: io.Autogrow.Type = None) -> io.NodeOutput:
        ref_latents = []
        images_vl = []

        images = images or {}
        for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])):
            image = images[name]
            if image is None:
                continue
            samples = image.movedim(-1, 1)

            # Vision tower input: the reference caps the VLM image at 384x384
            # (max_vlm_input_pil_pixels in pipeline_boogu.py).
            total = int(384 * 384)
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            images_vl.append(s.movedim(1, -1)[:, :, :, :3])

            # Reference latent: align to 16 px (VAE /8 * patch_size 2).
            if vae is not None:
                total = int(1024 * 1024)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by / 16.0) * 16
                height = round(samples.shape[2] * scale_by / 16.0) * 16
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

        # positive: instruction + vision tokens; negative: empty (no vision). Ref latent on both.
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images_vl))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        return io.NodeOutput(positive, negative)


class BooguExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeBooguEdit,
        ]


async def comfy_entrypoint() -> BooguExtension:
    return BooguExtension()
