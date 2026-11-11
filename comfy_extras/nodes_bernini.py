import torch
from typing_extensions import override

import comfy.model_management
import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, io


def _resize_long_edge(image, max_size, stride=16):
    """Resize (preserve aspect) so the long edge <= max_size, then snap each side to `stride`"""
    h, w = image.shape[1], image.shape[2]
    scale = min(max_size / max(h, w), 1.0)
    nh = max(stride, round(h * scale / stride) * stride)
    nw = max(stride, round(w * scale / stride) * stride)
    return comfy.utils.common_upscale(image[:, :, :, :3].movedim(-1, 1), nw, nh, "area", "disabled").movedim(1, -1)


class BerniniConditioning(io.ComfyNode):
    """Bernini in-context conditioning for a Wan2.2-A14B model.

    Attaches the VAE-encoded source video / reference images to the conditioning
    source video first, then each reference image

    The task is inferred from which inputs are connected:
      (nothing)                  -> t2v (text-to-video)
      source_video               -> v2v (video-to-video)
      source_video + ref_images  -> rv2v (reference-guided video editing)
      ref_images only            -> r2v (reference-to-video)
      source_video + ref_video   -> ads2v (insert image/video into video)

    source_video is the edit base / canvas (resized to width x height).
    reference_video is moving content to composite in.
    Streams are ordered source_video, reference_video, then reference_images -> source_id (1, 2, 3, ...).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniConditioning",
            display_name="Bernini Conditioning",
            category="model/conditioning/bernini",
            description="Conditioning node for Bernini in-context video/image conditioning. It can be used for the following tasks: t2v (text-to-video), v2v (video-to-video), rv2v (reference-guided video editing), r2v (reference-to-video), ads2v (insert image/video into video)."
                "Reference images injected as in-context tokens (r2v, rv2v) are encoded independently at their own native aspect ratio (long edge capped at ref_max_size)",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=8192, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("source_video", optional=True, tooltip=("Source video to edit or restyle (v2v, rv2v). Resized to width/height and trimmed to length.")),
                io.Image.Input("reference_video", optional=True, tooltip=("Video to insert into the source video (ads2v).")),
                io.Autogrow.Input("reference_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_image", tooltip=("Reference image injected as an in-context token (r2v, rv2v).")),
                        prefix="reference_image_", min=0, max=8)),
                io.Int.Input("ref_max_size", default=848, min=16, max=8192, step=16, optional=True, tooltip=(
                    "Max size for the long edge of reference_video and reference_images. Resized with preserved aspect ratio and snapped to 16px.")),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, source_video=None, reference_video=None, reference_images=None, ref_max_size=848) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())

        # source_video (1), reference_video (2), reference_images (3, 4, ...).
        context = []
        if source_video is not None:
            vid = comfy.utils.common_upscale(source_video[:length, :, :, :3].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
            context.append(vae.encode(vid[:, :, :, :3]))

        if reference_video is not None:
            ref_vid = _resize_long_edge(reference_video[:length], ref_max_size)  # moving content, native aspect
            context.append(vae.encode(ref_vid[:, :, :, :3]))

        # reference_images is an autogrow dict {reference_image_0: IMAGE, ...}; each slot is a
        # separate stream at its own native aspect (a multi-image batch in one slot -> one stream per frame).
        if reference_images:
            for name in sorted(reference_images):
                imgs = reference_images[name]
                if imgs is None:
                    continue
                for i in range(imgs.shape[0]):
                    img = _resize_long_edge(imgs[i:i + 1], ref_max_size)  # native aspect per ref
                    context.append(vae.encode(img[:, :, :, :3]))

        if context:
            positive = node_helpers.conditioning_set_values(positive, {"context_latents": context})
            negative = node_helpers.conditioning_set_values(negative, {"context_latents": context})

        return io.NodeOutput(positive, negative, {"samples": latent})


class BerniniExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [BerniniConditioning,]


async def comfy_entrypoint() -> BerniniExtension:
    return BerniniExtension()
