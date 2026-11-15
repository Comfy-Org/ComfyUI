import node_helpers
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import math
import comfy.utils


class TextEncodeZImageOmni(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeZImageOmni",
            category="model/conditioning/z-image",
            is_experimental=True,
            inputs=[
                io.Clip.Input("clip"),
                io.ClipVision.Input("image_encoder", optional=True),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Boolean.Input("auto_resize_images", default=True, advanced=True),
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
    def execute(cls, clip, prompt, image_encoder=None, auto_resize_images=True, vae=None, image1=None, image2=None, image3=None) -> io.NodeOutput:
        ref_latents = []
        images = list(filter(lambda a: a is not None, [image1, image2, image3]))

        prompt_list = []
        template = None
        if len(images) > 0:
            prompt_list = ["<|im_start|>user\n<|vision_start|>"]
            prompt_list += ["<|vision_end|><|vision_start|>"] * (len(images) - 1)
            prompt_list += ["<|vision_end|><|im_end|>"]
            template = "<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"

        encoded_images = []

        for i, image in enumerate(images):
            if image_encoder is not None:
                encoded_images.append(image_encoder.encode_image(image))

            if vae is not None:
                if auto_resize_images:
                    samples = image.movedim(-1, 1)
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    image = comfy.utils.common_upscale(samples, width, height, "area", "disabled").movedim(1, -1)
                ref_latents.append(vae.encode(image))

        tokens = clip.tokenize(prompt, llama_template=template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        extra_text_embeds = []
        for p in prompt_list:
            tokens = clip.tokenize(p, llama_template="{}")
            text_embeds = clip.encode_from_tokens_scheduled(tokens)
            extra_text_embeds.append(text_embeds[0][0])

        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        if len(encoded_images) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"clip_vision_outputs": encoded_images}, append=True)
        if len(extra_text_embeds) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents_text_embeds": extra_text_embeds}, append=True)

        return io.NodeOutput(conditioning)


class ZImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeZImageOmni,
        ]


async def comfy_entrypoint() -> ZImageExtension:
    return ZImageExtension()
