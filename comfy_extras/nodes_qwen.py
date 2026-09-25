import node_helpers
import comfy.utils
import math
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.model_management
import torch
import nodes

class TextEncodeQwenImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEdit",
            display_name="Text Encode Qwen Image Edit",
            category="model/conditioning/qwen image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image=None) -> io.NodeOutput:
        ref_latent = None
        if image is None:
            images = []
        else:
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
        return io.NodeOutput(conditioning)


class TextEncodeQwenImageEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEditPlus",
            display_name="Text Encode Qwen Image Edit Plus",
            category="model/conditioning/qwen image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning)


class TextEncodeQwenImage21(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImage21",
            display_name="Text Encode Qwen Image 2.1",
            category="model/conditioning/qwen image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Int.Input("resolution", default=1024, min=0, max=4096, step=32,
                             tooltip="Reference images are resized to about resolution x resolution pixels, at multiples of 32, preserving aspect ratio. 0 keeps each reference at its own size, rounded to a multiple of 32. "),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="Reference images, seen by the text encoder and spliced into the sequence as VAE latents.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent",
                                 tooltip="Empty latent on the first reference image's size, to match with sampling as any other size shifts the edit."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, vae=None, resolution=1024, images: io.Autogrow.Type = None) -> io.NodeOutput:
        ref_latents = []
        images_vl = []
        images = images or {}
        latent_w = latent_h = resolution or 1024
        for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])):
            image = images[name]
            if image is None:
                continue
            # same resize for the text encoder and the VAE, so every vision slot covers 2x2 latents; one image per input
            samples = image[:1].movedim(-1, 1)
            if resolution > 0:
                ratio = samples.shape[3] / samples.shape[2]
                width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
                height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
            else:
                width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
            width, height = max(32, width), max(32, height)
            if (width, height) == (samples.shape[3], samples.shape[2]):
                s = image[:1]
            else:
                s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
            if not images_vl:
                latent_w, latent_h = width, height
            rgb = s[:, :, :, :3]
            if s.shape[-1] > 3:
                rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])  # the vision tower sees alpha over white, the vae keeps all four
            images_vl.append(rgb)
            if vae is not None:
                ref_latents.append(vae.encode(s))

        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True))
        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
        latent = torch.zeros([1, 64, latent_h // 16, latent_w // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, negative, {"samples": latent})


class QwenImage21Cache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21Cache",
            display_name="Qwen Image 2.1 Cache",
            category="model/conditioning/qwen image",
            description=(
                "Allows setting the KV cache device and quantization, by default the model uses the auto -option. Quantization potentially speeds up edit workflows when memory starved "
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("device", options=["auto", "gpu", "cpu", "off"], default="auto",
                               tooltip="auto uses spare VRAM, then RAM. cpu (RAM) is prefetched behind compute and costs little speed. off recomputes the prefix every step, which is slower but is the one way to rule the cache out."),
                io.Combo.Input("dtype", options=["default", "int8", "int4"], default="default",
                               tooltip="Storage precision. default is lossless. int8 halves the cache at about bf16 accuracy, int4 quarters it but roughly doubles the per-step error."),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, device, dtype) -> io.NodeOutput:
        m = model.clone()
        m.model_options["transformer_options"]["qwen_image21_cache"] = {"device": device, "dtype": dtype}
        return io.NodeOutput(m)


class EmptyQwenImageLayeredLatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyQwenImageLayeredLatentImage",
            display_name="Empty Qwen Image Layered Latent",
            category="model/latent/qwen",
            inputs=[
                io.Int.Input("width", default=640, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=640, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("layers", default=3, min=0, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, layers, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, layers + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent})


class QwenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeQwenImageEdit,
            TextEncodeQwenImageEditPlus,
            TextEncodeQwenImage21,
            QwenImage21Cache,
            EmptyQwenImageLayeredLatentImage,
        ]


async def comfy_entrypoint() -> QwenExtension:
    return QwenExtension()
