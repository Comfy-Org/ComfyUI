import math

import torch
from typing_extensions import override

import comfy.model_management
import comfy.text_encoders.fibo2
import comfy.utils
import node_helpers
import nodes
from comfy_api.latest import ComfyExtension, io

# sigma 0.9 at the default shift 1.15: a generation shows Qwen3-VL the images of its caption's text from there on
TEXT_IMAGES_START = math.exp(1.15) / (9 + math.exp(1.15))


def has_images(tokens):
    return any(isinstance(t[0], dict) for batches in tokens.values() for batch in batches for t in batch)


def trained_size(aspect_ratio):
    # Bria's (width, height) for an aspect ratio: of the multiples of 16 within 10% of 1 MP, the least cropping, then the closest to 1 MP
    area = 1024 * 1024
    sizes = [(w, h) for w in range(64, 2 * area // 64, 16) for h in range(max(64, 16 * math.ceil(0.9 * area / w / 16)), 16 * math.floor(1.1 * area / w / 16) + 1, 16)]
    return min(sizes, key=lambda s: (round(1 - min(s[0] / s[1] / aspect_ratio, aspect_ratio * s[1] / s[0]), 9), abs(s[0] * s[1] - area), -s[0]))


class TextEncodeFibo2(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeFibo2",
            category="model/conditioning/fibo",
            description="Encode a Fibo 2.0 prompt, a structured JSON caption or plain text. With images, the prompt is the edit instruction, or a JSON caption of the result with an edit_instruction key, and the images become reference latents of both outputs when a vae is connected. Sample the latent output, the result's size the images are cropped for.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True),
                io.String.Input("negative_prompt", multiline=True, advanced=True, tooltip="Empty is the null caption Fibo 2.0 was trained with for CFG with edit workflows."),
                io.Vae.Input("vae", optional=True),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="Images to edit, referred to as <image_1>, <image_2> and so on in the instruction. A single image is scaled and center cropped to the result's size, since the edit lines up with it pixel for pixel; several images each to their own trained size.",
                ),
                io.Int.Input("width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=16, tooltip="Width of the result. 0 = the size Fibo 2.0 was trained at for the first image's aspect ratio, about 1 megapixel, or 1024 without images."),
                io.Int.Input("height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=16, tooltip="Height of the result. 0 = the size Fibo 2.0 was trained at for the first image's aspect ratio, about 1 megapixel, or 1024 without images."),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt="", vae=None, images: io.Autogrow.Type = None, width=0, height=0, batch_size=1) -> io.NodeOutput:
        images = images or {}
        images = [images[name] for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])) if images[name] is not None]
        images = [image[i:i + 1] for image in images for i in range(image.shape[0])]

        # as Bria's pipeline: a single image is cropped to the result's size, several images each to their own trained size
        default = trained_size(images[0].shape[2] / images[0].shape[1]) if len(images) > 0 else (1024, 1024)
        width, height = max(16, (width or default[0]) // 16 * 16), max(16, (height or default[1]) // 16 * 16)
        sizes = [(width, height)] if len(images) == 1 else [trained_size(image.shape[2] / image.shape[1]) for image in images]
        images = [image if (image.shape[2], image.shape[1]) == size else comfy.utils.common_upscale(image.movedim(-1, 1), *size, "lanczos", "center").movedim(1, -1)
                  for image, size in zip(images, sizes)]

        tokens = clip.tokenize(prompt, images=images)
        positive = clip.encode_from_tokens_scheduled(tokens)
        if len(images) == 0 and has_images(tokens):
            base = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, text_images=False))
            # ranges include both ends: the base stops just above sigma 0.9, which already has the text images
            positive = (node_helpers.conditioning_set_values(base, {"end_percent": TEXT_IMAGES_START - 1e-6})
                        + node_helpers.conditioning_set_values(positive, {"start_percent": TEXT_IMAGES_START}))
        # the negative is text only, encoded as written in the chat template, not cleaned like a caption, an edit's reference latents condition both
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(comfy.text_encoders.fibo2.TEMPLATE.format(negative_prompt) if negative_prompt.strip() else ""))

        if vae is not None and len(images) > 0:
            ref_latents = [vae.encode(image) for image in images]
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
        latent = torch.zeros([batch_size, 128, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, negative, {"samples": latent})


class Fibo2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeFibo2,
        ]


async def comfy_entrypoint() -> Fibo2Extension:
    return Fibo2Extension()
