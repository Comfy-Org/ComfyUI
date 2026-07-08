import nodes
import node_helpers
import torch
import comfy.model_management
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
import folder_paths
import json

class CLIPTextEncodeHunyuanDiT(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeHunyuanDiT",
            display_name="CLIP Text Encode (Hunyuan Image)",
            category="model/conditioning/hunyuan image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("bert", multiline=True, dynamic_prompts=True),
                io.String.Input("mt5xl", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, bert, mt5xl) -> io.NodeOutput:
        tokens = clip.tokenize(bert)
        tokens["mt5xl"] = clip.tokenize(mt5xl)["mt5xl"]

        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))

    encode = execute  # TODO: remove


class EmptyHunyuanLatentVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyHunyuanLatentVideo",
            display_name="Empty HunyuanVideo 1.0 Latent",
            category="model/latent/hunyuan video",
            inputs=[
                io.Int.Input("width", default=848, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=25, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, length, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent, "downscale_ratio_spacial": 8})

    generate = execute  # TODO: remove


class EmptyHunyuanVideo15Latent(EmptyHunyuanLatentVideo):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "EmptyHunyuanVideo15Latent"
        schema.display_name = "Empty HunyuanVideo 1.5 Latent"
        schema.category = "model/latent/hunyuan video"
        return schema

    @classmethod
    def execute(cls, width, height, length, batch_size=1) -> io.NodeOutput:
        # Using scale factor of 16 instead of 8
        latent = torch.zeros([batch_size, 32, ((length - 1) // 4) + 1, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent, "downscale_ratio_spacial": 16})


class HunyuanVideo15ImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanVideo15ImageToVideo",
            category="model/conditioning/hunyuan video",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=848, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=33, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 32, ((length - 1) // 4) + 1, height // 16, width // 16], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

            encoded = vae.encode(start_image[:, :, :, :3])
            concat_latent_image = torch.zeros((latent.shape[0], 32, latent.shape[2], latent.shape[3], latent.shape[4]), device=comfy.model_management.intermediate_device())
            concat_latent_image[:, :, :encoded.shape[2], :, :] = encoded

            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)


class HunyuanVideo15SuperResolution(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanVideo15SuperResolution",
            display_name="Hunyuan Video 1.5 Super Resolution",
            category="model/conditioning/hunyuan video",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("start_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Latent.Input("latent"),
                io.Float.Input("noise_augmentation", default=0.70, min=0.0, max=1.0, step=0.01, advanced=True),

            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, latent, noise_augmentation, vae=None, start_image=None, clip_vision_output=None) -> io.NodeOutput:
        in_latent = latent["samples"]
        in_channels = in_latent.shape[1]
        cond_latent = torch.zeros([in_latent.shape[0], in_channels * 2 + 2, in_latent.shape[-3], in_latent.shape[-2], in_latent.shape[-1]], device=comfy.model_management.intermediate_device())
        cond_latent[:, in_channels + 1 : 2 * in_channels + 1] = in_latent
        cond_latent[:, 2 * in_channels + 1] = 1
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image.movedim(-1, 1), in_latent.shape[-1] * 16, in_latent.shape[-2] * 16, "bilinear", "center").movedim(1, -1)
            encoded = vae.encode(start_image[:, :, :, :3])
            cond_latent[:, :in_channels, :encoded.shape[2], :, :] = encoded
            cond_latent[:, in_channels + 1, 0] = 1

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": cond_latent, "noise_augmentation": noise_augmentation})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": cond_latent, "noise_augmentation": noise_augmentation})
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        return io.NodeOutput(positive, negative, latent)


class LatentUpscaleModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentUpscaleModelLoader",
            display_name="Load Latent Upscale Model",
            category="model/loaders",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("latent_upscale_models")),
            ],
            outputs=[
                io.LatentUpscaleModel.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("latent_upscale_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)

        if "blocks.0.block.0.conv.weight" in sd:
            config = {
                "in_channels": sd["in_conv.conv.weight"].shape[1],
                "out_channels": sd["out_conv.conv.weight"].shape[0],
                "hidden_channels": sd["in_conv.conv.weight"].shape[0],
                "num_blocks": len([k for k in sd.keys() if k.startswith("blocks.") and k.endswith(".block.0.conv.weight")]),
                "global_residual": False,
            }
            model_type = "720p"
            model = HunyuanVideo15SRModel(model_type, config)
            model.load_sd(sd)
        elif "up.0.block.0.conv1.conv.weight" in sd:
            sd = {key.replace("nin_shortcut", "nin_shortcut.conv", 1): value for key, value in sd.items()}
            config = {
                "z_channels": sd["conv_in.conv.weight"].shape[1],
                "out_channels": sd["conv_out.conv.weight"].shape[0],
                "block_out_channels": tuple(sd[f"up.{i}.block.0.conv1.conv.weight"].shape[0] for i in range(len([k for k in sd.keys() if k.startswith("up.") and k.endswith(".block.0.conv1.conv.weight")]))),
            }
            model_type = "1080p"
            model = HunyuanVideo15SRModel(model_type, config)
            model.load_sd(sd)
        elif "post_upsample_res_blocks.0.conv2.bias" in sd:
            config = json.loads(metadata["config"])
            model = LatentUpsampler.from_config(config).to(dtype=comfy.model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]))
            model.load_state_dict(sd)

        return io.NodeOutput(model)


class HunyuanVideo15LatentUpscaleWithModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanVideo15LatentUpscaleWithModel",
            display_name="Hunyuan Video 15 Latent Upscale With Model",
            category="model/latent/hunyhuan video",
            inputs=[
                io.LatentUpscaleModel.Input("model"),
                io.Latent.Input("samples"),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], default="bilinear"),
                io.Int.Input("width", default=1280, min=0, max=16384, step=8),
                io.Int.Input("height", default=720, min=0, max=16384, step=8),
                io.Combo.Input("crop", options=["disabled", "center"]),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, samples, upscale_method, width, height, crop) -> io.NodeOutput:
        if width == 0 and height == 0:
            return io.NodeOutput(samples)
        else:
            if width == 0:
                height = max(64, height)
                width = max(64, round(samples["samples"].shape[-1] * height / samples["samples"].shape[-2]))
            elif height == 0:
                width = max(64, width)
                height = max(64, round(samples["samples"].shape[-2] * width / samples["samples"].shape[-1]))
            else:
                width = max(64, width)
                height = max(64, height)
            s = comfy.utils.common_upscale(samples["samples"], width // 16, height // 16, upscale_method, crop)
            s = model.resample_latent(s)
            return io.NodeOutput({"samples": s.cpu().float()})


PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

class TextEncodeHunyuanVideo_ImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeHunyuanVideo_ImageToVideo",
            category="model/conditioning/hunyuan video",
            inputs=[
                io.Clip.Input("clip"),
                io.ClipVisionOutput.Input("clip_vision_output"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input(
                    "image_interleave",
                    default=2,
                    min=1,
                    max=512,
                    tooltip="How much the image influences things vs the text prompt. Higher number means more influence from the text prompt.",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, clip_vision_output, prompt, image_interleave) -> io.NodeOutput:
        tokens = clip.tokenize(prompt, llama_template=PROMPT_TEMPLATE_ENCODE_VIDEO_I2V, image_embeds=clip_vision_output.mm_projected, image_interleave=image_interleave)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))

    encode = execute  # TODO: remove


class HunyuanImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanImageToVideo",
            category="model/conditioning/hunyuan video",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=848, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=53, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Combo.Input("guidance_type", options=["v1 (concat)", "v2 (replace)", "custom"], advanced=True),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, vae, width, height, length, batch_size, guidance_type, start_image=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        out_latent = {}

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length, :, :, :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

            concat_latent_image = vae.encode(start_image)
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            if guidance_type == "v1 (concat)":
                cond = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            elif guidance_type == "v2 (replace)":
                cond = {'guiding_frame_index': 0}
                latent[:, :, :concat_latent_image.shape[2]] = concat_latent_image
                out_latent["noise_mask"] = mask
            elif guidance_type == "custom":
                cond = {"ref_latent": concat_latent_image}

            positive = node_helpers.conditioning_set_values(positive, cond)

        out_latent["samples"] = latent
        return io.NodeOutput(positive, out_latent)

    encode = execute  # TODO: remove


class EmptyHunyuanImageLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyHunyuanImageLatent",
            category="model/latent/hunyuan image",
            inputs=[
                io.Int.Input("width", default=2048, min=64, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=2048, min=64, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 64, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples":latent})

    generate = execute  # TODO: remove


class HunyuanRefinerLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanRefinerLatent",
            display_name="Hunyuan Latent Refiner",
            category="model/conditioning/hunyuan video",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Float.Input("noise_augmentation", default=0.10, min=0.0, max=1.0, step=0.01, advanced=True),

            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, latent, noise_augmentation) -> io.NodeOutput:
        latent = latent["samples"]
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": latent, "noise_augmentation": noise_augmentation})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": latent, "noise_augmentation": noise_augmentation})
        out_latent = {}
        out_latent["samples"] = torch.zeros([latent.shape[0], 32, latent.shape[-3], latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, negative, out_latent)


class HunyuanExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeHunyuanDiT,
            TextEncodeHunyuanVideo_ImageToVideo,
            EmptyHunyuanLatentVideo,
            EmptyHunyuanVideo15Latent,
            HunyuanVideo15ImageToVideo,
            HunyuanVideo15SuperResolution,
            HunyuanVideo15LatentUpscaleWithModel,
            LatentUpscaleModelLoader,
            HunyuanImageToVideo,
            EmptyHunyuanImageLatent,
            HunyuanRefinerLatent,
        ]


async def comfy_entrypoint() -> HunyuanExtension:
    return HunyuanExtension()
