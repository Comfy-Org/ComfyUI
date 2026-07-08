import math
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.clip_vision
import json
import numpy as np
from typing import Tuple, TypedDict
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import logging

class WanImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanImageToVideo",
            category="model/conditioning/wan",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])
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


class WanFunControlToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanFunControlToVideo",
            category="model/conditioning/wan/fun control",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("control_video", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None, control_video=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
        concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(start_image[:, :, :, :3])
            concat_latent[:,16:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(control_video[:, :, :, :3])
            concat_latent[:,:16,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)

class Wan22FunControlToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Wan22FunControlToVideo",
            category="model/conditioning/wan/fun control",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("ref_image", optional=True),
                io.Image.Input("control_video", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, ref_image=None, start_image=None, control_video=None) -> io.NodeOutput:
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent = torch.zeros([batch_size, latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale], device=comfy.model_management.intermediate_device())
        if latent_channels == 48:
            concat_latent = comfy.latent_formats.Wan22().process_out(concat_latent)
        else:
            concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
        concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(start_image[:, :, :, :3])
            concat_latent[:,latent_channels:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]
            mask[:, :, :start_image.shape[0] + 3] = 0.0

        ref_latent = None
        if ref_image is not None:
            ref_image = comfy.utils.common_upscale(ref_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            ref_latent = vae.encode(ref_image[:, :, :, :3])

        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(control_video[:, :, :, :3])
            concat_latent[:,:latent_channels,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent, "concat_mask": mask, "concat_mask_index": latent_channels})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent, "concat_mask": mask, "concat_mask_index": latent_channels})

        if ref_latent is not None:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref_latent]}, append=True)

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)

class WanFirstLastFrameToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanFirstLastFrameToVideo",
            category="model/conditioning/wan",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None) -> io.NodeOutput:
        spacial_scale = vae.spacial_compression_encode()
        latent = torch.zeros([batch_size, vae.latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0

        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        clip_vision_output = None
        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image

        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat([clip_vision_output.penultimate_hidden_states, clip_vision_end_image.penultimate_hidden_states], dim=-2)
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)


class WanFunInpaintToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanFunInpaintToVideo",
            category="model/conditioning/wan/fun inpaint",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_output=None) -> io.NodeOutput:
        flfv = WanFirstLastFrameToVideo()
        return flfv.execute(positive, negative, vae, width, height, length, batch_size, start_image=start_image, end_image=end_image, clip_vision_start_image=clip_vision_output)


class WanVaceToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanVaceToVideo",
            search_aliases=["video conditioning", "video control"],
            category="model/conditioning/wan/vace",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input("strength", default=1.0, min=0.0, max=1000.0, step=0.01),
                io.Image.Input("control_video", optional=True),
                io.Mask.Input("control_masks", optional=True),
                io.Image.Input("reference_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, strength, control_video=None, control_masks=None, reference_image=None) -> io.NodeOutput:
        latent_length = ((length - 1) // 4) + 1
        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5)
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat([reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))], dim=1)

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)

        latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent, trim_latent)

class TrimVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TrimVideoLatent",
            display_name="Trim Video Latent",
            category="model/latent",
            inputs=[
                io.Latent.Input("samples"),
                io.Int.Input("trim_amount", default=0, min=0, max=99999),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, trim_amount) -> io.NodeOutput:
        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = s1[:, :, trim_amount:]
        return io.NodeOutput(samples_out)

class WanCameraImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanCameraImageToVideo",
            category="model/conditioning/wan/camera",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("start_image", optional=True),
                io.WanCameraEmbedding.Input("camera_conditions", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None, camera_conditions=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(start_image[:, :, :, :3])
            concat_latent[:,:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]
            mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
            mask[:, :, :start_image.shape[0] + 3] = 0.0
            mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent, "concat_mask": mask})

        if camera_conditions is not None:
            positive = node_helpers.conditioning_set_values(positive, {'camera_conditions': camera_conditions})
            negative = node_helpers.conditioning_set_values(negative, {'camera_conditions': camera_conditions})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)

class WanPhantomSubjectToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanPhantomSubjectToVideo",
            category="model/conditioning/wan/phantom subject",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("images", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative_text"),
                io.Conditioning.Output(display_name="negative_img_text"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, images) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        cond2 = negative
        if images is not None:
            images = comfy.utils.common_upscale(images[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            latent_images = []
            for i in images:
                latent_images += [vae.encode(i.unsqueeze(0)[:, :, :, :3])]
            concat_latent_image = torch.cat(latent_images, dim=2)

            positive = node_helpers.conditioning_set_values(positive, {"time_dim_concat": concat_latent_image})
            cond2 = node_helpers.conditioning_set_values(negative, {"time_dim_concat": concat_latent_image})
            negative = node_helpers.conditioning_set_values(negative, {"time_dim_concat": comfy.latent_formats.Wan21().process_out(torch.zeros_like(concat_latent_image))})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, cond2, negative, out_latent)

def parse_json_tracks(tracks):
    """Parse JSON track data into a standardized format"""
    tracks_data = []
    try:
        # If tracks is a string, try to parse it as JSON
        if isinstance(tracks, str):
            parsed = json.loads(tracks.replace("'", '"'))
            tracks_data.extend(parsed)
        else:
            # If tracks is a list of strings, parse each one
            for track_str in tracks:
                parsed = json.loads(track_str.replace("'", '"'))
                tracks_data.append(parsed)

        # Check if we have a single track (dict with x,y) or a list of tracks
        if tracks_data and isinstance(tracks_data[0], dict) and 'x' in tracks_data[0]:
            # Single track detected, wrap it in a list
            tracks_data = [tracks_data]
        elif tracks_data and isinstance(tracks_data[0], list) and tracks_data[0] and isinstance(tracks_data[0][0], dict) and 'x' in tracks_data[0][0]:
            # Already a list of tracks, nothing to do
            pass
        else:
            # Unexpected format
            pass

    except json.JSONDecodeError:
        tracks_data = []
    return tracks_data

def process_tracks(tracks_np: np.ndarray, frame_size: Tuple[int, int], num_frames, quant_multi: int = 8, **kwargs):
    # tracks: shape [t, h, w, 3] => samples align with 24 fps, model trained with 16 fps.
    # frame_size: tuple (W, H)
    tracks = torch.from_numpy(tracks_np).float()

    if tracks.shape[1] == 121:
        tracks = torch.permute(tracks, (1, 0, 2, 3))

    tracks, visibles = tracks[..., :2], tracks[..., 2:3]

    short_edge = min(*frame_size)

    frame_center = torch.tensor([*frame_size]).type_as(tracks) / 2
    tracks = tracks - frame_center

    tracks = tracks / short_edge * 2

    visibles = visibles * 2 - 1

    trange = torch.linspace(-1, 1, tracks.shape[0]).view(-1, 1, 1, 1).expand(*visibles.shape)

    out_ = torch.cat([trange, tracks, visibles], dim=-1).view(121, -1, 4)

    out_0 = out_[:1]

    out_l = out_[1:] # 121 => 120 | 1
    a = 120 // math.gcd(120, num_frames)
    b = num_frames // math.gcd(120, num_frames)
    out_l = torch.repeat_interleave(out_l, b, dim=0)[1::a]  # 120 => 120 * b => 120 * b / a == F

    final_result = torch.cat([out_0, out_l], dim=0)

    return final_result

FIXED_LENGTH = 121
def pad_pts(tr):
    """Convert list of {x,y} to (FIXED_LENGTH,1,3) array, padding/truncating."""
    pts = np.array([[p['x'], p['y'], 1] for p in tr], dtype=np.float32)
    n = pts.shape[0]
    if n < FIXED_LENGTH:
        pad = np.zeros((FIXED_LENGTH - n, 3), dtype=np.float32)
        pts = np.vstack((pts, pad))
    else:
        pts = pts[:FIXED_LENGTH]
    return pts.reshape(FIXED_LENGTH, 1, 3)

def ind_sel(target: torch.Tensor, ind: torch.Tensor, dim: int = 1):
    """Index selection utility function"""
    assert (
        len(ind.shape) > dim
    ), "Index must have the target dim, but get dim: %d, ind shape: %s" % (dim, str(ind.shape))

    target = target.expand(
        *tuple(
            [ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)]
            + [
                -1,
            ]
            * (len(target.shape) - dim)
        )
    )

    ind_pad = ind

    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1,) * (dim + 1), *target.shape[(dim + 1) : :])

    return torch.gather(target, dim=dim, index=ind_pad)

def merge_final(vert_attr: torch.Tensor, weight: torch.Tensor, vert_assign: torch.Tensor):
    """Merge vertex attributes with weights"""
    target_dim = len(vert_assign.shape) - 1
    if len(vert_attr.shape) == 2:
        assert vert_attr.shape[0] > vert_assign.max()
        new_shape = [1] * target_dim + list(vert_attr.shape)
        tensor = vert_attr.reshape(new_shape)
        sel_attr = ind_sel(tensor, vert_assign.type(torch.long), dim=target_dim)
    else:
        assert vert_attr.shape[1] > vert_assign.max()
        new_shape = [vert_attr.shape[0]] + [1] * (target_dim - 1) + list(vert_attr.shape[1:])
        tensor = vert_attr.reshape(new_shape)
        sel_attr = ind_sel(tensor, vert_assign.type(torch.long), dim=target_dim)

    final_attr = torch.sum(sel_attr * weight.unsqueeze(-1), dim=-2)
    return final_attr


def _patch_motion_single(
    tracks: torch.FloatTensor,  # (B, T, N, 4)
    vid: torch.FloatTensor,     # (C, T, H, W)
    temperature: float,
    vae_divide: tuple,
    topk: int,
):
    """Apply motion patching based on tracks"""
    _, T, H, W = vid.shape
    N = tracks.shape[2]
    _, tracks_xy, visible = torch.split(
        tracks, [1, 2, 1], dim=-1
    )  # (B, T, N, 2) | (B, T, N, 1)
    tracks_n = tracks_xy / torch.tensor([W / min(H, W), H / min(H, W)], device=tracks_xy.device)
    tracks_n = tracks_n.clamp(-1, 1)
    visible = visible.clamp(0, 1)

    xx = torch.linspace(-W / min(H, W), W / min(H, W), W)
    yy = torch.linspace(-H / min(H, W), H / min(H, W), H)

    grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1).to(
        tracks_xy.device
    )

    tracks_pad = tracks_xy[:, 1:]
    visible_pad = visible[:, 1:]

    visible_align = visible_pad.view(T - 1, 4, *visible_pad.shape[2:]).sum(1)
    tracks_align = (tracks_pad * visible_pad).view(T - 1, 4, *tracks_pad.shape[2:]).sum(
        1
    ) / (visible_align + 1e-5)
    dist_ = (
        (tracks_align[:, None, None] - grid[None, :, :, None]).pow(2).sum(-1)
    )  # T, H, W, N
    weight = torch.exp(-dist_ * temperature) * visible_align.clamp(0, 1).view(
        T - 1, 1, 1, N
    )
    vert_weight, vert_index = torch.topk(
        weight, k=min(topk, weight.shape[-1]), dim=-1
    )

    grid_mode = "bilinear"
    point_feature = torch.nn.functional.grid_sample(
        vid.permute(1, 0, 2, 3)[:1],
        tracks_n[:, :1].type(vid.dtype),
        mode=grid_mode,
        padding_mode="zeros",
        align_corners=False,
    )
    point_feature = point_feature.squeeze(0).squeeze(1).permute(1, 0) # N, C=16

    out_feature = merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2) # T - 1, H, W, C => C, T - 1, H, W
    out_weight = vert_weight.sum(-1) # T - 1, H, W

    # out feature -> already soft weighted
    mix_feature = out_feature + vid[:, 1:] * (1 - out_weight.clamp(0, 1))

    out_feature_full = torch.cat([vid[:, :1], mix_feature], dim=1) # C, T, H, W
    out_mask_full = torch.cat([torch.ones_like(out_weight[:1]), out_weight], dim=0)  # T, H, W

    return out_mask_full[None].expand(vae_divide[0], -1, -1, -1), out_feature_full


def patch_motion(
    tracks: torch.FloatTensor,  # (B, TB, T, N, 4)
    vid: torch.FloatTensor,     # (C, T, H, W)
    temperature: float = 220.0,
    vae_divide: tuple = (4, 16),
    topk: int = 2,
):
    B = len(tracks)

    # Process each batch separately
    out_masks = []
    out_features = []

    for b in range(B):
        mask, feature = _patch_motion_single(
            tracks[b],  # (T, N, 4)
            vid[b],        # (C, T, H, W)
            temperature,
            vae_divide,
            topk
        )
        out_masks.append(mask)
        out_features.append(feature)

    # Stack results: (B, C, T, H, W)
    out_mask_full = torch.stack(out_masks, dim=0)
    out_feature_full = torch.stack(out_features, dim=0)

    return out_mask_full, out_feature_full

class WanTrackToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanTrackToVideo",
            search_aliases=["motion tracking", "trajectory video", "point tracking", "keypoint animation"],
            category="model/conditioning/wan/move",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.String.Input("tracks", multiline=True, default="[]"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input("temperature", default=220.0, min=1.0, max=1000.0, step=0.1, advanced=True),
                io.Int.Input("topk", default=2, min=1, max=10, advanced=True),
                io.Image.Input("start_image"),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, tracks, width, height, length, batch_size,
               temperature, topk, start_image=None, clip_vision_output=None) -> io.NodeOutput:

        tracks_data = parse_json_tracks(tracks)

        if not tracks_data:
            return WanImageToVideo().execute(positive, negative, vae, width, height, length, batch_size, start_image=start_image, clip_vision_output=clip_vision_output)

        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                           device=comfy.model_management.intermediate_device())

        if isinstance(tracks_data[0][0], dict):
            tracks_data = [tracks_data]

        processed_tracks = []
        for batch in tracks_data:
            arrs = []
            for track in batch:
                pts = pad_pts(track)
                arrs.append(pts)

            tracks_np = np.stack(arrs, axis=0)
            processed_tracks.append(process_tracks(tracks_np, (width, height), length - 1).unsqueeze(0))

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:batch_size].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            videos = torch.ones((start_image.shape[0], length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            for i in range(start_image.shape[0]):
                videos[i, 0] = start_image[i]

            latent_videos = []
            videos = comfy.utils.resize_to_batch_size(videos, batch_size)
            for i in range(batch_size):
                latent_videos += [vae.encode(videos[i, :, :, :, :3])]
            y = torch.cat(latent_videos, dim=0)

            # Scale latent since patch_motion is non-linear
            y = comfy.latent_formats.Wan21().process_in(y)

            processed_tracks = comfy.utils.resize_list_to_batch_size(processed_tracks, batch_size)
            res = patch_motion(
                processed_tracks, y, temperature=temperature, topk=topk, vae_divide=(4, 16)
            )

            mask, concat_latent_image = res
            concat_latent_image = comfy.latent_formats.Wan21().process_out(concat_latent_image)
            mask = -mask + 1.0  # Invert mask to match expected format
            positive = node_helpers.conditioning_set_values(positive,
                                                            {"concat_mask": mask,
                                                            "concat_latent_image": concat_latent_image})
            negative = node_helpers.conditioning_set_values(negative,
                                                            {"concat_mask": mask,
                                                            "concat_latent_image": concat_latent_image})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    features: shape=[1, T, 512]
    input_fps: fps for audio, f_a
    output_fps: fps for video, f_m
    output_len: video length
    """
    features = features.transpose(1, 2)  # [1, 512, T]
    seq_len = features.shape[2] / float(input_fps)  # T/f_a
    if output_len is None:
        output_len = int(seq_len * output_fps)  # f_m*T/f_a
    output_features = torch.nn.functional.interpolate(
        features, size=output_len, align_corners=True,
        mode='linear')  # [1, 512, output_len]
    return output_features.transpose(1, 2)  # [1, output_len, 512]


def get_sample_indices(original_fps,
                       total_frames,
                       target_fps,
                       num_sample,
                       fixed_start=None):
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    if fixed_start is not None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    start_time = start_frame / original_fps

    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def get_audio_embed_bucket_fps(audio_embed, fps=16, batch_frames=81, m=0, video_rate=30):
    num_layers, audio_frame_num, audio_dim = audio_embed.shape

    if num_layers > 1:
        return_all_layers = True
    else:
        return_all_layers = False

    scale = video_rate / fps

    min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1

    bucket_num = min_batch_num * batch_frames
    padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * video_rate) - audio_frame_num
    batch_idx = get_sample_indices(
        original_fps=video_rate,
        total_frames=audio_frame_num + padd_audio_num,
        target_fps=fps,
        num_sample=bucket_num,
        fixed_start=0)
    batch_audio_eb = []
    audio_sample_stride = int(video_rate / fps)
    for bi in batch_idx:
        if bi < audio_frame_num:

            chosen_idx = list(
                range(bi - m * audio_sample_stride, bi + (m + 1) * audio_sample_stride, audio_sample_stride))
            chosen_idx = [0 if c < 0 else c for c in chosen_idx]
            chosen_idx = [
                audio_frame_num - 1 if c >= audio_frame_num else c
                for c in chosen_idx
            ]

            if return_all_layers:
                frame_audio_embed = audio_embed[:, chosen_idx].flatten(
                    start_dim=-2, end_dim=-1)
            else:
                frame_audio_embed = audio_embed[0][chosen_idx].flatten()
        else:
            frame_audio_embed = torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device) if not return_all_layers \
                else torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
        batch_audio_eb.append(frame_audio_embed)
    batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)

    return batch_audio_eb, min_batch_num


def wan_sound_to_video(positive, negative, vae, width, height, length, batch_size, frame_offset=0, ref_image=None, audio_encoder_output=None, control_video=None, ref_motion=None, ref_motion_latent=None):
    latent_t = ((length - 1) // 4) + 1
    if audio_encoder_output is not None:
        feat = torch.cat(audio_encoder_output["encoded_audio_all_layers"])
        video_rate = 30
        fps = 16
        feat = linear_interpolation(feat, input_fps=50, output_fps=video_rate)
        batch_frames = latent_t * 4
        audio_embed_bucket, num_repeat = get_audio_embed_bucket_fps(feat, fps=fps, batch_frames=batch_frames, m=0, video_rate=video_rate)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)

        audio_embed_bucket = audio_embed_bucket[:, :, :, frame_offset:frame_offset + batch_frames]
        if audio_embed_bucket.shape[3] > 0:
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": audio_embed_bucket})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": audio_embed_bucket * 0.0})
            frame_offset += batch_frames

    if ref_image is not None:
        ref_image = comfy.utils.common_upscale(ref_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        ref_latent = vae.encode(ref_image[:, :, :, :3])
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref_latent]}, append=True)

    if ref_motion is not None:
        if ref_motion.shape[0] > 73:
            ref_motion = ref_motion[-73:]

        ref_motion = comfy.utils.common_upscale(ref_motion.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        if ref_motion.shape[0] < 73:
            r = torch.ones([73, height, width, 3]) * 0.5
            r[-ref_motion.shape[0]:] = ref_motion
            ref_motion = r

        ref_motion_latent = vae.encode(ref_motion[:, :, :, :3])

    if ref_motion_latent is not None:
        ref_motion_latent = ref_motion_latent[:, :, -19:]
        positive = node_helpers.conditioning_set_values(positive, {"reference_motion": ref_motion_latent})
        negative = node_helpers.conditioning_set_values(negative, {"reference_motion": ref_motion_latent})

    latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=comfy.model_management.intermediate_device())

    control_video_out = comfy.latent_formats.Wan21().process_out(torch.zeros_like(latent))
    if control_video is not None:
        control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        control_video = vae.encode(control_video[:, :, :, :3])
        control_video_out[:, :, :control_video.shape[2]] = control_video

    # TODO: check if zero is better than none if none provided
    positive = node_helpers.conditioning_set_values(positive, {"control_video": control_video_out})
    negative = node_helpers.conditioning_set_values(negative, {"control_video": control_video_out})

    out_latent = {}
    out_latent["samples"] = latent
    return positive, negative, out_latent, frame_offset


class WanSoundImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSoundImageToVideo",
            category="model/conditioning/wan/sound",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=77, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.AudioEncoderOutput.Input("audio_encoder_output", optional=True),
                io.Image.Input("ref_image", optional=True),
                io.Image.Input("control_video", optional=True),
                io.Image.Input("ref_motion", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, ref_image=None, audio_encoder_output=None, control_video=None, ref_motion=None) -> io.NodeOutput:
        positive, negative, out_latent, frame_offset = wan_sound_to_video(positive, negative, vae, width, height, length, batch_size, ref_image=ref_image, audio_encoder_output=audio_encoder_output,
                                                                          control_video=control_video, ref_motion=ref_motion)
        return io.NodeOutput(positive, negative, out_latent)


class WanSoundImageToVideoExtend(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSoundImageToVideoExtend",
            category="model/conditioning/wan/sound",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("length", default=77, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Latent.Input("video_latent"),
                io.AudioEncoderOutput.Input("audio_encoder_output", optional=True),
                io.Image.Input("ref_image", optional=True),
                io.Image.Input("control_video", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, length, video_latent, ref_image=None, audio_encoder_output=None, control_video=None) -> io.NodeOutput:
        video_latent = video_latent["samples"]
        width = video_latent.shape[-1] * 8
        height = video_latent.shape[-2] * 8
        batch_size = video_latent.shape[0]
        frame_offset = video_latent.shape[-3] * 4
        positive, negative, out_latent, frame_offset = wan_sound_to_video(positive, negative, vae, width, height, length, batch_size, frame_offset=frame_offset, ref_image=ref_image, audio_encoder_output=audio_encoder_output,
                                                                          control_video=control_video, ref_motion=None, ref_motion_latent=video_latent)
        return io.NodeOutput(positive, negative, out_latent)


def get_audio_emb_window(audio_emb, frame_num, frame0_idx, audio_shift=2):
    zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)  # device=audio_emb.device
    iter_ = 1 + (frame_num - 1) // 4
    audio_emb_wind = []
    for lt_i in range(iter_):
        if lt_i == 0:
            st = frame0_idx + lt_i - 2
            ed = frame0_idx + lt_i + 3
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
            wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)
        else:
            st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
            ed = frame0_idx + 1 + 4 * lt_i + audio_shift
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
        audio_emb_wind.append(wind_feat)
    audio_emb_wind = torch.stack(audio_emb_wind, dim=0)

    return audio_emb_wind, ed - audio_shift


class WanHuMoImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanHuMoImageToVideo",
            category="model/conditioning/wan/humo",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=97, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.AudioEncoderOutput.Input("audio_encoder_output", optional=True),
                io.Image.Input("ref_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, ref_image=None, audio_encoder_output=None) -> io.NodeOutput:
        latent_t = ((length - 1) // 4) + 1
        latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=comfy.model_management.intermediate_device())

        if ref_image is not None:
            ref_image = comfy.utils.common_upscale(ref_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            ref_latent = vae.encode(ref_image[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
        else:
            zero_latent = torch.zeros([batch_size, 16, 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [zero_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [zero_latent]}, append=True)

        if audio_encoder_output is not None:
            audio_emb = torch.stack(audio_encoder_output["encoded_audio_all_layers"], dim=2)
            audio_len = audio_encoder_output["audio_samples"] // 640
            audio_emb = audio_emb[:, :audio_len * 2]

            feat0 = linear_interpolation(audio_emb[:, :, 0: 8].mean(dim=2), 50, 25)
            feat1 = linear_interpolation(audio_emb[:, :, 8: 16].mean(dim=2), 50, 25)
            feat2 = linear_interpolation(audio_emb[:, :, 16: 24].mean(dim=2), 50, 25)
            feat3 = linear_interpolation(audio_emb[:, :, 24: 32].mean(dim=2), 50, 25)
            feat4 = linear_interpolation(audio_emb[:, :, 32], 50, 25)
            audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]  # [T, 5, 1280]
            audio_emb, _ = get_audio_emb_window(audio_emb, length, frame0_idx=0)

            audio_emb = audio_emb.unsqueeze(0)
            audio_emb_neg = torch.zeros_like(audio_emb)
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": audio_emb})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": audio_emb_neg})
        else:
            zero_audio = torch.zeros([batch_size, latent_t + 1, 8, 5, 1280], device=comfy.model_management.intermediate_device())
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": zero_audio})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": zero_audio})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)

class WanAnimateToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanAnimateToVideo",
            category="model/conditioning/wan/animate",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=77, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("reference_image", optional=True),
                io.Image.Input("face_video", optional=True),
                io.Image.Input("pose_video", optional=True),
                io.Int.Input("continue_motion_max_frames", default=5, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Image.Input("background_video", optional=True),
                io.Mask.Input("character_mask", optional=True),
                io.Image.Input("continue_motion", optional=True),
                io.Int.Input("video_frame_offset", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="The amount of frames to seek in all the input videos. Used for generating longer videos by chunk. Connect to the video_frame_offset output of the previous node for extending a video."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_latent"),
                io.Int.Output(display_name="trim_image"),
                io.Int.Output(display_name="video_frame_offset"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, continue_motion_max_frames, video_frame_offset, reference_image=None, clip_vision_output=None, face_video=None, pose_video=None, continue_motion=None, background_video=None, character_mask=None) -> io.NodeOutput:
        trim_to_pose_video = False
        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        trim_latent = 0

        if reference_image is None:
            reference_image = torch.zeros((1, height, width, 3))

        image = comfy.utils.common_upscale(reference_image[:length].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = torch.zeros((1, 4, concat_latent_image.shape[-3], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=concat_latent_image.device, dtype=concat_latent_image.dtype)
        trim_latent += concat_latent_image.shape[2]
        ref_motion_latent_length = 0

        if continue_motion is None:
            image = torch.ones((length, height, width, 3)) * 0.5
        else:
            continue_motion = continue_motion[-continue_motion_max_frames:]
            video_frame_offset -= continue_motion.shape[0]
            video_frame_offset = max(0, video_frame_offset)
            continue_motion = comfy.utils.common_upscale(continue_motion[-length:].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
            image = torch.ones((length, height, width, continue_motion.shape[-1]), device=continue_motion.device, dtype=continue_motion.dtype) * 0.5
            image[:continue_motion.shape[0]] = continue_motion
            ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[video_frame_offset:]

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(pose_video[:length].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
            if not trim_to_pose_video:
                if pose_video.shape[0] < length:
                    pose_video = torch.cat((pose_video,) + (pose_video[-1:],) * (length - pose_video.shape[0]), dim=0)

            pose_video_latent = vae.encode(pose_video[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(positive, {"pose_video_latent": pose_video_latent})
            negative = node_helpers.conditioning_set_values(negative, {"pose_video_latent": pose_video_latent})

            if trim_to_pose_video:
                latent_length = pose_video_latent.shape[2]
                length = latent_length * 4 - 3
                image = image[:length]

        if face_video is not None:
            if face_video.shape[0] <= video_frame_offset:
                face_video = None
            else:
                face_video = face_video[video_frame_offset:]

        if face_video is not None:
            face_video = comfy.utils.common_upscale(face_video[:length].movedim(-1, 1), 512, 512, "area", "center") * 2.0 - 1.0
            face_video = face_video.movedim(0, 1).unsqueeze(0)
            positive = node_helpers.conditioning_set_values(positive, {"face_video_pixels": face_video})
            negative = node_helpers.conditioning_set_values(negative, {"face_video_pixels": face_video * 0.0 - 1.0})

        ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
        if background_video is not None:
            if background_video.shape[0] > video_frame_offset:
                background_video = background_video[video_frame_offset:]
                background_video = comfy.utils.common_upscale(background_video[:length].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
                if background_video.shape[0] > ref_images_num:
                    image[ref_images_num:background_video.shape[0]] = background_video[ref_images_num:]

        mask_refmotion = torch.ones((1, 1, latent_length * 4, concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=mask.device, dtype=mask.dtype)
        if continue_motion is not None:
            mask_refmotion[:, :, :ref_motion_latent_length * 4] = 0.0

        if character_mask is not None:
            if character_mask.shape[0] > video_frame_offset or character_mask.shape[0] == 1:
                if character_mask.shape[0] == 1:
                    character_mask = character_mask.repeat((length,) + (1,) * (character_mask.ndim - 1))
                else:
                    character_mask = character_mask[video_frame_offset:]
                if character_mask.ndim == 3:
                    character_mask = character_mask.unsqueeze(1)
                    character_mask = character_mask.movedim(0, 1)
                if character_mask.ndim == 4:
                    character_mask = character_mask.unsqueeze(1)
                character_mask = comfy.utils.common_upscale(character_mask[:, :, :length], concat_latent_image.shape[-1], concat_latent_image.shape[-2], "nearest-exact", "center")
                if character_mask.shape[2] > ref_images_num:
                    mask_refmotion[:, :, ref_images_num:character_mask.shape[2]] = character_mask[:, :, ref_images_num:]

        concat_latent_image = torch.cat((concat_latent_image, vae.encode(image[:, :, :, :3])), dim=2)


        mask_refmotion = mask_refmotion.view(1, mask_refmotion.shape[2] // 4, 4, mask_refmotion.shape[3], mask_refmotion.shape[4]).transpose(1, 2)
        mask = torch.cat((mask, mask_refmotion), dim=2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        latent = torch.zeros([batch_size, 16, latent_length + trim_latent, latent_height, latent_width], device=comfy.model_management.intermediate_device())
        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent, trim_latent, max(0, ref_motion_latent_length * 4 - 3), video_frame_offset + length)

class Wan22ImageToVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Wan22ImageToVideoLatent",
            category="model/conditioning/wan",
            inputs=[
                io.Vae.Input("vae"),
                io.Int.Input("width", default=1280, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=704, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=49, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, vae, width, height, length, batch_size, start_image=None) -> io.NodeOutput:
        latent = torch.zeros([1, 48, ((length - 1) // 4) + 1, height // 16, width // 16], device=comfy.model_management.intermediate_device())

        if start_image is None:
            out_latent = {}
            out_latent["samples"] = latent
            return io.NodeOutput(out_latent)

        mask = torch.ones([latent.shape[0], 1, ((length - 1) // 4) + 1, latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            latent_temp = vae.encode(start_image)
            latent[:, :, :latent_temp.shape[-3]] = latent_temp
            mask[:, :, :latent_temp.shape[-3]] *= 0.0

        out_latent = {}
        latent_format = comfy.latent_formats.Wan22()
        latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)
        out_latent["samples"] = latent.repeat((batch_size, ) + (1,) * (latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size, ) + (1,) * (mask.ndim - 1))
        return io.NodeOutput(out_latent)


from comfy.ldm.wan.model_multitalk import InfiniteTalkOuterSampleWrapper, MultiTalkCrossAttnPatch, MultiTalkGetAttnMapPatch, project_audio_features
class WanInfiniteTalkToVideo(io.ComfyNode):
    class DCValues(TypedDict):
        mode: str
        audio_encoder_output_2: io.AudioEncoderOutput.Type
        mask: io.Mask.Type

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanInfiniteTalkToVideo",
            category="model/conditioning/wan/infinite talk",
            inputs=[
                io.DynamicCombo.Input("mode", options=[
                io.DynamicCombo.Option("single_speaker", []),
                io.DynamicCombo.Option("two_speakers", [
                    io.AudioEncoderOutput.Input("audio_encoder_output_2", optional=True),
                    io.Mask.Input("mask_1", optional=True, tooltip="Mask for the first speaker, required if using two audio inputs."),
                    io.Mask.Input("mask_2", optional=True, tooltip="Mask for the second speaker, required if using two audio inputs."),
                    ]),
                ]),
                io.Model.Input("model"),
                io.ModelPatch.Input("model_patch"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("start_image", optional=True),
                io.AudioEncoderOutput.Input("audio_encoder_output_1"),
                io.Int.Input("motion_frame_count", default=9, min=1, max=33, step=1, tooltip="Number of previous frames to use as motion context.", advanced=True),
                io.Float.Input("audio_scale", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Image.Input("previous_frames", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_image"),
            ],
        )

    @classmethod
    def execute(cls, mode: DCValues, model, model_patch, positive, negative, vae, width, height, length, audio_encoder_output_1, motion_frame_count,
                start_image=None, previous_frames=None, audio_scale=None, clip_vision_output=None, audio_encoder_output_2=None, mask_1=None, mask_2=None) -> io.NodeOutput:

        if previous_frames is not None and previous_frames.shape[0] < motion_frame_count:
            raise ValueError("Not enough previous frames provided.")

        if mode["mode"] == "two_speakers":
            audio_encoder_output_2 = mode["audio_encoder_output_2"]
            mask_1 = mode["mask_1"]
            mask_2 = mode["mask_2"]

        if audio_encoder_output_2 is not None:
            if mask_1 is None or mask_2 is None:
                raise ValueError("Masks must be provided if two audio encoder outputs are used.")

        ref_masks = None
        if mask_1 is not None and mask_2 is not None:
            if audio_encoder_output_2 is None:
                raise ValueError("Second audio encoder output must be provided if two masks are used.")
            ref_masks = torch.cat([mask_1, mask_2])

        latent = torch.zeros([1, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])
            concat_mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            concat_mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        model_patched = model.clone()

        encoded_audio_list = []
        seq_lengths = []

        for audio_encoder_output in [audio_encoder_output_1, audio_encoder_output_2]:
            if audio_encoder_output is None:
                continue
            all_layers = audio_encoder_output["encoded_audio_all_layers"]
            encoded_audio = torch.stack(all_layers, dim=0).squeeze(1)[1:]  # shape: [num_layers, T, 512]
            encoded_audio = linear_interpolation(encoded_audio, input_fps=50, output_fps=25).movedim(0, 1) # shape: [T, num_layers, 512]
            encoded_audio_list.append(encoded_audio)
            seq_lengths.append(encoded_audio.shape[0])

        # Pad / combine depending on multi_audio_type
        multi_audio_type = "add"
        if len(encoded_audio_list) > 1:
            if multi_audio_type == "para":
                max_len = max(seq_lengths)
                padded = []
                for emb in encoded_audio_list:
                    if emb.shape[0] < max_len:
                        pad = torch.zeros(max_len - emb.shape[0], *emb.shape[1:], dtype=emb.dtype)
                        emb = torch.cat([emb, pad], dim=0)
                    padded.append(emb)
                encoded_audio_list = padded
            elif multi_audio_type == "add":
                total_len = sum(seq_lengths)
                full_list = []
                offset = 0
                for emb, seq_len in zip(encoded_audio_list, seq_lengths):
                    full = torch.zeros(total_len, *emb.shape[1:], dtype=emb.dtype)
                    full[offset:offset+seq_len] = emb
                    full_list.append(full)
                    offset += seq_len
                encoded_audio_list = full_list

        token_ref_target_masks = None
        if ref_masks is not None:
            token_ref_target_masks = torch.nn.functional.interpolate(
                ref_masks.unsqueeze(0), size=(latent.shape[-2] // 2, latent.shape[-1] // 2), mode='nearest')[0]
            token_ref_target_masks = (token_ref_target_masks > 0).view(token_ref_target_masks.shape[0], -1)

        # when extending from previous frames
        if previous_frames is not None:
            motion_frames = comfy.utils.common_upscale(previous_frames[-motion_frame_count:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            frame_offset = previous_frames.shape[0] - motion_frame_count

            audio_start = frame_offset
            audio_end = audio_start + length
            logging.info(f"InfiniteTalk: Processing audio frames {audio_start} - {audio_end}")

            motion_frames_latent = vae.encode(motion_frames[:, :, :, :3])
            trim_image = motion_frame_count
        else:
            audio_start = trim_image = 0
            audio_end = length
            motion_frames_latent = concat_latent_image[:, :, :1]

        audio_embed = project_audio_features(model_patch.model.audio_proj, encoded_audio_list, audio_start, audio_end).to(model_patched.model_dtype())
        model_patched.model_options["transformer_options"]["audio_embeds"] = audio_embed

        # add outer sample wrapper
        model_patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "infinite_talk_outer_sample",
            InfiniteTalkOuterSampleWrapper(
                motion_frames_latent,
                model_patch,
                is_extend=previous_frames is not None,
            ))
        # add cross-attention patch
        model_patched.set_model_patch(MultiTalkCrossAttnPatch(model_patch, audio_scale), "attn2_patch")
        if token_ref_target_masks is not None:
            model_patched.set_model_patch(MultiTalkGetAttnMapPatch(token_ref_target_masks), "attn1_patch")

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(model_patched, positive, negative, out_latent, trim_image)


class WanExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanTrackToVideo,
            WanImageToVideo,
            WanFunControlToVideo,
            Wan22FunControlToVideo,
            WanFunInpaintToVideo,
            WanFirstLastFrameToVideo,
            WanVaceToVideo,
            TrimVideoLatent,
            WanCameraImageToVideo,
            WanPhantomSubjectToVideo,
            WanSoundImageToVideo,
            WanSoundImageToVideoExtend,
            WanHuMoImageToVideo,
            WanAnimateToVideo,
            Wan22ImageToVideoLatent,
            WanInfiniteTalkToVideo,
        ]

async def comfy_entrypoint() -> WanExtension:
    return WanExtension()
