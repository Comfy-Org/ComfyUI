import av
import torch
from av.codec import CodecContext
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.bria import (
    BriaEditImageRequest,
    BriaImageEditResponse,
    BriaRemoveBackgroundRequest,
    BriaRemoveBackgroundResponse,
    BriaRemoveVideoBackgroundRequest,
    BriaRemoveVideoBackgroundResponse,
    BriaStatusResponse,
    BriaVideoGreenScreenRequest,
    BriaVideoReplaceBackgroundRequest,
    InputModerationSettings,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    convert_mask_to_image,
    download_url_to_image_tensor,
    download_url_to_video_output,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_video_duration,
)


class BriaImageEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaImageEditNode",
            display_name="Bria FIBO Image Edit",
            category="partner/image/Bria",
            description="Edit images using Bria latest model",
            inputs=[
                IO.Combo.Input("model", options=["FIBO"]),
                IO.Image.Input("image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Instruction to edit image",
                ),
                IO.String.Input("negative_prompt", multiline=True, default=""),
                IO.String.Input(
                    "structured_prompt",
                    multiline=True,
                    default="",
                    tooltip="A string containing the structured edit prompt in JSON format. "
                    "Use this instead of usual prompt for precise, programmatic control.",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=1,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Float.Input(
                    "guidance_scale",
                    default=3,
                    min=3,
                    max=5,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely.",
                ),
                IO.Int.Input(
                    "steps",
                    default=50,
                    min=20,
                    max=50,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                ),
                IO.DynamicCombo.Input(
                    "moderation",
                    options=[
                        IO.DynamicCombo.Option("false", []),
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input("prompt_content_moderation", default=False),
                                IO.Boolean.Input("visual_input_moderation", default=False),
                                IO.Boolean.Input("visual_output_moderation", default=True),
                            ],
                        ),
                    ],
                    tooltip="Moderation settings",
                ),
                IO.Mask.Input(
                    "mask",
                    tooltip="If omitted, the edit applies to the entire image.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(display_name="structured_prompt"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.04}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        negative_prompt: str,
        structured_prompt: str,
        seed: int,
        guidance_scale: float,
        steps: int,
        moderation: InputModerationSettings,
        mask: Input.Image | None = None,
    ) -> IO.NodeOutput:
        if not prompt and not structured_prompt:
            raise ValueError("One of prompt or structured_prompt is required to be non-empty.")
        mask_url = None
        if mask is not None:
            mask_url = await upload_image_to_comfyapi(cls, convert_mask_to_image(mask), wait_label="Uploading mask")
        response = await sync_op(
            cls,
            ApiEndpoint(path="proxy/bria/v2/image/edit", method="POST"),
            data=BriaEditImageRequest(
                instruction=prompt if prompt else None,
                structured_instruction=structured_prompt if structured_prompt else None,
                images=[await upload_image_to_comfyapi(cls, image, wait_label="Uploading image")],
                mask=mask_url,
                negative_prompt=negative_prompt if negative_prompt else None,
                guidance_scale=guidance_scale,
                seed=seed,
                model_version=model,
                steps_num=steps,
                prompt_content_moderation=moderation.get("prompt_content_moderation", False),
                visual_input_content_moderation=moderation.get("visual_input_moderation", False),
                visual_output_content_moderation=moderation.get("visual_output_moderation", False),
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaImageEditResponse,
        )
        return IO.NodeOutput(
            await download_url_to_image_tensor(response.result.image_url),
            response.result.structured_prompt,
        )


class BriaRemoveImageBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaRemoveImageBackground",
            display_name="Bria Remove Image Background",
            category="partner/image/Bria",
            description="Remove the background from an image using Bria RMBG 2.0.",
            inputs=[
                IO.Image.Input("image"),
                IO.DynamicCombo.Input(
                    "moderation",
                    options=[
                        IO.DynamicCombo.Option("false", []),
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input("visual_input_moderation", default=False),
                                IO.Boolean.Input("visual_output_moderation", default=True),
                            ],
                        ),
                    ],
                    tooltip="Moderation settings",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.018}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        moderation: dict,
        seed: int,
    ) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/image/edit/remove_background", method="POST"),
            data=BriaRemoveBackgroundRequest(
                image=await upload_image_to_comfyapi(cls, image, wait_label="Uploading image"),
                sync=False,
                visual_input_content_moderation=moderation.get("visual_input_moderation", False),
                visual_output_content_moderation=moderation.get("visual_output_moderation", False),
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveBackgroundResponse,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result.image_url))


class BriaRemoveVideoBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaRemoveVideoBackground",
            display_name="Bria Remove Video Background",
            category="partner/video/Bria",
            description="Remove the background from a video using Bria. ",
            inputs=[
                IO.Video.Input("video"),
                IO.Combo.Input(
                    "background_color",
                    options=[
                        "Black",
                        "White",
                        "Gray",
                        "Red",
                        "Green",
                        "Blue",
                        "Yellow",
                        "Cyan",
                        "Magenta",
                        "Orange",
                    ],
                    tooltip="Background color for the output video.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.0042,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        background_color: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=60.0)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/video/edit/remove_background", method="POST"),
            data=BriaRemoveVideoBackgroundRequest(
                video=await upload_video_to_comfyapi(cls, video),
                background_color=background_color,
                output_container_and_codec="mp4_h264",
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveVideoBackgroundResponse,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.result.video_url))


class BriaVideoGreenScreen(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaVideoGreenScreen",
            display_name="Bria Video Green Screen",
            category="partner/video/Bria",
            description="Replace a video's background with a solid chroma-key screen using Bria.",
            inputs=[
                IO.Video.Input("video"),
                IO.Combo.Input(
                    "green_shade",
                    options=["broadcast_green", "chroma_green", "blue_screen"],
                    tooltip="Solid chroma-key shade applied behind the foreground: "
                    "broadcast_green (#00B140), chroma_green (#00FF00), or blue_screen (#0000FF).",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.0042,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        green_shade: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=60.0)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/video/edit/green_screen", method="POST"),
            data=BriaVideoGreenScreenRequest(
                video=await upload_video_to_comfyapi(cls, video),
                green_shade=green_shade,
                output_container_and_codec="mp4_h264",
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveVideoBackgroundResponse,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.result.video_url))


class BriaVideoReplaceBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaVideoReplaceBackground",
            display_name="Bria Video Replace Background",
            category="partner/video/Bria",
            description="Replace a video's background with a supplied image or video using Bria. "
            "The output keeps the foreground's resolution and frame rate; a background with a "
            "different aspect ratio is stretched to fit, so match it for undistorted results.",
            inputs=[
                IO.Video.Input("video", tooltip="Foreground video whose background is replaced."),
                IO.Image.Input(
                    "background_image",
                    optional=True,
                    tooltip="Background image to composite behind the foreground. "
                    "Provide either a background image or a background video, not both.",
                ),
                IO.Video.Input(
                    "background_video",
                    optional=True,
                    tooltip="Background video to composite behind the foreground. "
                    "Provide either a background image or a background video, not both.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.0042,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        seed: int,
        background_image: Input.Image | None = None,
        background_video: Input.Video | None = None,
    ) -> IO.NodeOutput:
        if (background_image is None) == (background_video is None):
            raise ValueError("Provide either a background image or a background video, not both.")
        validate_video_duration(video, max_duration=60.0)
        if background_video is not None:
            validate_video_duration(background_video, max_duration=60.0)
            background_url = await upload_video_to_comfyapi(cls, background_video, wait_label="Uploading background")
        else:
            # Bria's replace_background 500s on RGBA, so drop the alpha channel before upload.
            background_url = await upload_image_to_comfyapi(
                cls, background_image[:, :, :, :3], wait_label="Uploading background"
            )
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/video/edit/replace_background", method="POST"),
            data=BriaVideoReplaceBackgroundRequest(
                video=await upload_video_to_comfyapi(cls, video),
                background_url=background_url,
                output_container_and_codec="mp4_h264",
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveVideoBackgroundResponse,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.result.video_url))


def _video_to_images_and_mask(video: Input.Video) -> tuple[Input.Image, Input.Mask]:
    """Decode a transparent webm (VP9 + alpha) into image frames and an alpha mask.

    VP9 keeps its alpha in a side layer that PyAV's default vp9 decoder drops, so the frames
    are decoded with libvpx-vp9. Returns RGB images [B,H,W,3] in 0..1 and a mask [B,H,W]
    following the Load Image convention (1 = transparent) for compositing or Save WEBM.
    """
    rgb_frames: list[torch.Tensor] = []
    alpha_frames: list[torch.Tensor] = []
    with av.open(video.get_stream_source(), mode="r") as container:
        stream = container.streams.video[0]
        decoder = CodecContext.create("libvpx-vp9", "r") if stream.codec_context.name == "vp9" else None
        for packet in container.demux(stream):
            for frame in (decoder.decode(packet) if decoder is not None else packet.decode()):
                rgba = torch.from_numpy(frame.to_ndarray(format="rgba")).float() / 255.0
                rgb_frames.append(rgba[..., :3])
                alpha_frames.append(rgba[..., 3])
    images = torch.stack(rgb_frames) if rgb_frames else torch.zeros(0, 0, 0, 3)
    mask = (1.0 - torch.stack(alpha_frames)) if alpha_frames else torch.zeros((images.shape[0], 64, 64))
    return images, mask


class BriaTransparentVideoBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaTransparentVideoBackground",
            display_name="Bria Remove Video Background (Transparent)",
            category="partner/video/Bria",
            description="Remove the background from a video using Bria and return the cut-out frames "
            "plus an alpha mask. Connect both to a compositing node, or feed them to Save WEBM to "
            "write a transparent video.",
            inputs=[
                IO.Video.Input("video"),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="images"),
                IO.Mask.Output(display_name="mask"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.0042,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        seed: int,
    ) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=60.0)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/video/edit/remove_background", method="POST"),
            data=BriaRemoveVideoBackgroundRequest(
                video=await upload_video_to_comfyapi(cls, video),
                background_color="Transparent",
                output_container_and_codec="webm_vp9",
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveVideoBackgroundResponse,
        )
        video_out = await download_url_to_video_output(response.result.video_url)
        images, mask = _video_to_images_and_mask(video_out)
        return IO.NodeOutput(images, mask)


class BriaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            BriaImageEditNode,
            BriaRemoveImageBackground,
            BriaRemoveVideoBackground,
            BriaVideoGreenScreen,
            BriaVideoReplaceBackground,
            BriaTransparentVideoBackground,
        ]


async def comfy_entrypoint() -> BriaExtension:
    return BriaExtension()
