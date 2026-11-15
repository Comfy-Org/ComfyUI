from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.sync_so import (
    SyncActiveSpeakerDetection,
    SyncGeneration,
    SyncGenerationOptions,
    SyncGenerationRequest,
    SyncInputItem,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    downscale_image_tensor,
    downscale_image_tensor_by_max_side,
    get_image_dimensions,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_audio_duration,
)


class SyncLipSyncNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SyncLipSyncNode",
            display_name="sync.so Lip Sync",
            category="partner/video/sync.so",
            description=(
                "Re-sync mouth movement in a video to new speech audio using sync.so. "
                "Handles close-ups, profiles and obstructions automatically while preserving "
                "the speaker's expression. Cost scales with output duration."
            ),
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="Footage of the speaker to re-sync. Up to 4K (4096x2160); "
                    "a constant frame rate of 24/25/30 fps works best.",
                ),
                IO.Audio.Input(
                    "audio",
                    tooltip="Speech audio to sync the mouth to.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "sync-3",
                            [
                                IO.Combo.Input(
                                    "sync_mode",
                                    options=["bounce", "cut_off", "loop", "silence", "remap"],
                                    default="bounce",
                                    tooltip=(
                                        "How to handle a duration mismatch between video and audio; "
                                        "this also sets the output length. "
                                        "bounce: video plays forward then backward until the audio ends "
                                        "(output = audio length). "
                                        "loop: video restarts until the audio ends (output = audio length). "
                                        "remap: video is time-stretched to match the audio (output = audio length). "
                                        "cut_off: the longer track is trimmed (output = shorter length). "
                                        "silence: nothing is trimmed; the shorter track is padded "
                                        "(output = longer length)."
                                    ),
                                ),
                                IO.Combo.Input(
                                    "speaker_selection",
                                    options=["default", "auto-detect", "coordinates"],
                                    default="default",
                                    tooltip=(
                                        "Which face to lipsync when several people are visible. "
                                        "default: let the model decide. "
                                        "auto-detect: detect and follow the active speaker. "
                                        "coordinates: target the face at pixel (speaker_x, speaker_y) "
                                        "in the frame chosen by speaker_frame."
                                    ),
                                ),
                                IO.Int.Input(
                                    "speaker_frame",
                                    default=0,
                                    min=0,
                                    max=1_000_000,
                                    advanced=True,
                                    tooltip="Video frame used to locate the speaker. "
                                    "Only used when speaker_selection is 'coordinates'.",
                                ),
                                IO.Int.Input(
                                    "speaker_x",
                                    default=0,
                                    min=0,
                                    max=4096,
                                    advanced=True,
                                    tooltip="X pixel coordinate of the speaker's face. "
                                    "Only used when speaker_selection is 'coordinates'.",
                                ),
                                IO.Int.Input(
                                    "speaker_y",
                                    default=0,
                                    min=0,
                                    max=4096,
                                    advanced=True,
                                    tooltip="Y pixel coordinate of the speaker's face. "
                                    "Only used when speaker_selection is 'coordinates'.",
                                ),
                            ],
                        )
                    ],
                    tooltip="sync.so generation model.",
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
                expr="""{"type":"usd","usd":0.19019,"format":{"approximate":true,"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        audio: Input.Audio,
        seed: int,
        model: dict,
    ) -> IO.NodeOutput:
        try:
            width, height = video.get_dimensions()
        except Exception:
            width = height = None
        if width and height and (max(width, height) > 4096 or width * height > 4096 * 2160):
            raise ValueError(
                f"sync.so rejects videos above 4K (4096x2160); got {width}x{height}. Downscale the video first."
            )
        validate_audio_duration(audio, max_duration=600)

        if model["speaker_selection"] == "auto-detect":
            speaker_detection = SyncActiveSpeakerDetection(auto_detect=True)
        elif model["speaker_selection"] == "coordinates":
            speaker_detection = SyncActiveSpeakerDetection(
                frame_number=model["speaker_frame"],
                coordinates=[model["speaker_x"], model["speaker_y"]],
            )
        else:
            speaker_detection = None

        video_url = await upload_video_to_comfyapi(cls, video, max_duration=600)
        audio_url = await upload_audio_to_comfyapi(cls, audio)

        generation = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/synclabs/v2/generate", method="POST"),
            response_model=SyncGeneration,
            data=SyncGenerationRequest(
                model=model["model"],
                input=[
                    SyncInputItem(type="video", url=video_url),
                    SyncInputItem(type="audio", url=audio_url),
                ],
                options=SyncGenerationOptions(
                    sync_mode=model["sync_mode"],
                    active_speaker_detection=speaker_detection,
                ),
            ),
        )
        generation = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/synclabs/v2/generate/{generation.id}"),
            response_model=SyncGeneration,
            status_extractor=lambda g: g.status,
            completed_statuses=["COMPLETED", "FAILED", "REJECTED"],
            failed_statuses=[],
            queued_statuses=["PENDING"],
            poll_interval=10.0,
        )
        if generation.status != "COMPLETED":
            code = f" [{generation.errorCode}]" if generation.errorCode else ""
            raise ValueError(
                f"sync.so generation {generation.status.lower()}{code}: "
                f"{generation.error or 'no error details provided'}"
            )
        if not generation.outputUrl:
            raise ValueError("sync.so generation completed but no output URL was returned.")
        return IO.NodeOutput(await download_url_to_video_output(generation.outputUrl))


class SyncTalkingImageNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SyncTalkingImageNode",
            display_name="sync.so Talking Image",
            category="partner/video/sync.so",
            description=(
                "Animate a still portrait into a talking video driven by speech audio, "
                "using sync.so's sync-3 model. The output duration matches the audio. "
                "Cost scales with output duration."
            ),
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="A single image with a clearly visible face, up to 4K (4096x2160).",
                ),
                IO.Audio.Input(
                    "audio",
                    tooltip="Speech audio driving the talking video; the output duration matches it. "
                    "Chain any TTS node here to drive the animation from text.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Optional guidance for how the portrait comes to life, e.g. "
                    "'make the subject smile and look at the camera'. "
                    "Leave empty for natural talking motion.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "sync-3",
                            [
                                IO.Combo.Input(
                                    "speaker_selection",
                                    options=["default", "coordinates"],
                                    default="default",
                                    tooltip=(
                                        "Which face to animate when several people are visible. "
                                        "default: let the model decide. "
                                        "coordinates: target the face at pixel (speaker_x, speaker_y) "
                                        "in the image. Auto-detection is not supported for images."
                                    ),
                                ),
                                IO.Int.Input(
                                    "speaker_x",
                                    default=0,
                                    min=0,
                                    max=4096,
                                    advanced=True,
                                    tooltip="X pixel coordinate of the speaker's face. "
                                    "Only used when speaker_selection is 'coordinates'.",
                                ),
                                IO.Int.Input(
                                    "speaker_y",
                                    default=0,
                                    min=0,
                                    max=4096,
                                    advanced=True,
                                    tooltip="Y pixel coordinate of the speaker's face. "
                                    "Only used when speaker_selection is 'coordinates'.",
                                ),
                                IO.Boolean.Input(
                                    "auto_downscale",
                                    default=True,
                                    advanced=True,
                                    tooltip="Automatically downscale the image if it exceeds the 4K "
                                    "(4096x2160) input limit; speaker coordinates are scaled to match. "
                                    "When disabled, an oversized image raises an error instead.",
                                ),
                            ],
                        )
                    ],
                    tooltip="sync.so generation model. Image input is exclusive to sync-3.",
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
                expr="""{"type":"usd","usd":0.19019,"format":{"approximate":true,"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        audio: Input.Audio,
        prompt: str,
        seed: int,
        model: dict,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one image is required; got a batch. Pick one frame first.")
        validate_audio_duration(audio, max_duration=600)

        height, width = get_image_dimensions(image)
        speaker_x, speaker_y = model["speaker_x"], model["speaker_y"]
        if max(width, height) > 4096 or width * height > 4096 * 2160:
            if not model["auto_downscale"]:
                raise ValueError(
                    f"sync.so rejects images above 4K (4096x2160); got {width}x{height}. "
                    "Downscale the image first or enable auto_downscale."
                )
            image = downscale_image_tensor(image, total_pixels=4096 * 2160)
            image = downscale_image_tensor_by_max_side(image, max_side=4096)
            new_height, new_width = get_image_dimensions(image)
            # speaker coordinates are given in the original image's pixel space
            speaker_x = min(new_width - 1, round(speaker_x * new_width / width))
            speaker_y = min(new_height - 1, round(speaker_y * new_height / height))

        if model["speaker_selection"] == "coordinates":
            speaker_detection = SyncActiveSpeakerDetection(
                frame_number=0,  # images have a single frame; auto_detect is rejected by the API
                coordinates=[speaker_x, speaker_y],
            )
        else:
            speaker_detection = None

        image_url = await upload_image_to_comfyapi(cls, image, mime_type="image/png", total_pixels=None)
        audio_url = await upload_audio_to_comfyapi(cls, audio)

        generation = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/synclabs/v2/generate", method="POST"),
            response_model=SyncGeneration,
            data=SyncGenerationRequest(
                model=model["model"],
                input=[
                    SyncInputItem(type="image", url=image_url),
                    SyncInputItem(type="audio", url=audio_url),
                ],
                options=SyncGenerationOptions(
                    i2v_prompt=prompt.strip() or None,
                    active_speaker_detection=speaker_detection,
                ),
            ),
        )
        generation = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/synclabs/v2/generate/{generation.id}"),
            response_model=SyncGeneration,
            status_extractor=lambda g: g.status,
            completed_statuses=["COMPLETED", "FAILED", "REJECTED"],
            failed_statuses=[],
            queued_statuses=["PENDING"],
            poll_interval=10.0,
        )
        if generation.status != "COMPLETED":
            code = f" [{generation.errorCode}]" if generation.errorCode else ""
            raise ValueError(
                f"sync.so generation {generation.status.lower()}{code}: "
                f"{generation.error or 'no error details provided'}"
            )
        if not generation.outputUrl:
            raise ValueError("sync.so generation completed but no output URL was returned.")
        return IO.NodeOutput(await download_url_to_video_output(generation.outputUrl))


class SyncExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            SyncLipSyncNode,
            SyncTalkingImageNode,
        ]


async def comfy_entrypoint() -> SyncExtension:
    return SyncExtension()
