import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class Kandinsky5ImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Kandinsky5ImageToVideo",
            category="model/conditioning/kandinsky",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=768, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=512, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=121, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Empty video latent"),
                io.Latent.Output(display_name="cond_latent", tooltip="Clean encoded start images, used to replace the noisy start of the model output latents"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, start_image=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        cond_latent_out = {}
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            encoded = vae.encode(start_image[:, :, :, :3])
            cond_latent_out["samples"] = encoded

            mask = torch.ones((1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"time_dim_replace": encoded, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"time_dim_replace": encoded, "concat_mask": mask})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent, cond_latent_out)


def adaptive_mean_std_normalization(source, reference, clump_mean_low=0.3, clump_mean_high=0.35, clump_std_low=0.35, clump_std_high=0.5):
    source_mean = source.mean(dim=(1, 3, 4), keepdim=True)  # mean over C, H, W
    source_std = source.std(dim=(1, 3, 4), keepdim=True)    # std over C, H, W

    reference_mean = torch.clamp(reference.mean(), source_mean - clump_mean_low, source_mean + clump_mean_high)
    reference_std = torch.clamp(reference.std(), source_std - clump_std_low, source_std + clump_std_high)

    # normalization
    normalized = (source - source_mean) / (source_std + 1e-8)
    normalized = normalized * reference_std + reference_mean

    return normalized


class NormalizeVideoLatentStart(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NormalizeVideoLatentStart",
            category="model/conditioning",
            description="Normalizes the initial frames of a video latent to match the mean and standard deviation of subsequent reference frames. Helps reduce differences between the starting frames and the rest of the video.",
            inputs=[
                io.Latent.Input("latent"),
                io.Int.Input("start_frame_count", default=4, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="Number of latent frames to normalize, counted from the start"),
                io.Int.Input("reference_frame_count", default=5, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="Number of latent frames after the start frames to use as reference"),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, latent, start_frame_count, reference_frame_count) -> io.NodeOutput:
        if latent["samples"].shape[2] <= 1:
            return io.NodeOutput(latent)
        s = latent.copy()
        samples = latent["samples"].clone()

        first_frames = samples[:, :, :start_frame_count]
        reference_frames_data = samples[:, :, start_frame_count:start_frame_count+min(reference_frame_count, samples.shape[2]-1)]
        normalized_first_frames = adaptive_mean_std_normalization(first_frames, reference_frames_data)

        samples[:, :, :start_frame_count] = normalized_first_frames
        s["samples"] = samples
        return io.NodeOutput(s)


class CLIPTextEncodeKandinsky5(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeKandinsky5",
            display_name="CLIP Text Encode (Kandinsky 5)",
            search_aliases=["kandinsky prompt"],
            category="model/conditioning/kandinsky",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("clip_l", multiline=True, dynamic_prompts=True),
                io.String.Input("qwen25_7b", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, clip_l, qwen25_7b) -> io.NodeOutput:
        tokens = clip.tokenize(clip_l)
        tokens["qwen25_7b"] = clip.tokenize(qwen25_7b)["qwen25_7b"]

        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))


class Kandinsky5Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Kandinsky5ImageToVideo,
            NormalizeVideoLatentStart,
            CLIPTextEncodeKandinsky5,
        ]

async def comfy_entrypoint() -> Kandinsky5Extension:
    return Kandinsky5Extension()
