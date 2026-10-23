import av
from pydantic import BaseModel
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.veed import (
    VeedBackgroundRemovalRequest,
    VeedCleanAudioRequest,
    VeedCleanAudioVideoRequest,
    VeedFabricRequest,
    VeedGreenScreenRequest,
    VeedJob,
    VeedJobResponse,
    VeedLipsyncRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_bytes_to_audio_input,
    download_url_as_bytesio,
    download_url_to_video_output,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_audio_duration,
    validate_video_duration,
    video_to_images_and_mask,
)

VEED_PATH = "/proxy/veed/v1"
MAX_FABRIC_AUDIO_SECONDS = 300
MAX_LIPSYNC_AUDIO_SECONDS = 600
MAX_BACKGROUND_REMOVAL_SECONDS = 60
MAX_CLEAN_AUDIO_SECONDS = 1799


async def _run_job(cls: type[IO.ComfyNode], route: str, request: BaseModel) -> VeedJob:
    submitted = await sync_op(
        cls,
        ApiEndpoint(path=f"{VEED_PATH}/{route}", method="POST"),
        response_model=VeedJobResponse,
        data=request,
    )
    job = (
        await poll_op(
            cls,
            ApiEndpoint(path=f"{VEED_PATH}/{route}/{submitted.data.job_id}"),
            response_model=VeedJobResponse,
            status_extractor=lambda r: r.data.status,
            completed_statuses=["COMPLETED", "FAILED", "CANCELLED"],
            failed_statuses=[],
            poll_interval=10.0,
        )
    ).data
    if job.status != "COMPLETED":
        if job.error and job.error.code == "insufficient_credits":
            raise ValueError("VEED is temporarily unavailable. Please try again later.")
        if job.error:
            raise ValueError(f"VEED job {job.status.lower()} [{job.error.code}]: {job.error.message}")
        raise ValueError(f"VEED job {job.status.lower()}.")
    return job


async def _upload_audio(cls: type[IO.ComfyNode], audio: Input.Audio) -> str:
    return await upload_audio_to_comfyapi(
        cls, audio, container_format="wav", codec_name="pcm_s16le", mime_type="audio/wav"
    )


class VeedFabricNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VeedFabricNode",
            display_name="VEED Fabric 1.0",
            category="partner/video/VEED",
            description="Turn a still image and a speech track into a talking, lip-synced video with VEED Fabric. "
            "Works with photos, illustrations and mascots; the video is as long as the audio.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "fabric-1.0",
                            [
                                IO.Image.Input(
                                    "image",
                                    tooltip="Image with a clearly visible face. The video keeps its aspect ratio.",
                                ),
                                IO.Audio.Input(
                                    "audio",
                                    tooltip=f"Speech to lip-sync to, up to {MAX_FABRIC_AUDIO_SECONDS} seconds. "
                                    "Sets the video length.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["480p", "720p"],
                                    default="720p",
                                    tooltip="Output size class: about 0.4 megapixels at 480p (864x480 at 16:9) and "
                                    "0.9 megapixels at 720p (1312x736 at 16:9).",
                                ),
                            ],
                        ),
                    ],
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
                depends_on=IO.PriceBadgeDepends(widgets=["model.resolution"]),
                expr="""
                {
                  "type":"usd",
                  "usd": $lookup(widgets, "model.resolution") = "480p" ? 0.1144 : 0.2145,
                  "format":{"suffix":"/second"}
                }
                """,
            ),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        validate_audio_duration(model["audio"], max_duration=MAX_FABRIC_AUDIO_SECONDS)
        job = await _run_job(
            cls,
            model["model"],
            VeedFabricRequest(
                image_url=await upload_image_to_comfyapi(cls, model["image"], mime_type="image/png"),
                audio_url=await _upload_audio(cls, model["audio"]),
                resolution=model["resolution"],
            ),
        )
        return IO.NodeOutput(await download_url_to_video_output(job.result.video.url))


class VeedLipsyncNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VeedLipsyncNode",
            display_name="VEED Lipsync 2.0",
            category="partner/video/VEED",
            description="Re-sync a speaker's mouth in a video to a new speech track with VEED Lipsync. "
            "One speaker per video; the output keeps the video's resolution at 25 fps and is as long as the audio.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "lipsync-2.0",
                            [
                                IO.Video.Input(
                                    "video",
                                    tooltip="Footage of one front-facing speaker. If it is shorter than the audio, "
                                    "it plays forward and backward until the audio ends.",
                                ),
                                IO.Audio.Input(
                                    "audio",
                                    tooltip=f"New speech, up to {MAX_LIPSYNC_AUDIO_SECONDS} seconds. "
                                    "Sets the video length.",
                                ),
                                IO.Int.Input(
                                    "seed",
                                    default=42,
                                    min=0,
                                    max=2147483647,
                                    display_mode=IO.NumberDisplay.number,
                                    control_after_generate=True,
                                    tooltip="Seed controls whether the node should re-run; "
                                    "results are non-deterministic regardless of seed.",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr="""{"type":"usd","usd":0.1001,"format":{"suffix":"/second"}}"""),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        validate_audio_duration(model["audio"], max_duration=MAX_LIPSYNC_AUDIO_SECONDS)
        job = await _run_job(
            cls,
            model["model"],
            VeedLipsyncRequest(
                video_url=await upload_video_to_comfyapi(cls, model["video"]),
                audio_url=await _upload_audio(cls, model["audio"]),
            ),
        )
        return IO.NodeOutput(await download_url_to_video_output(job.result.video.url))


def _segmentation_option(key: str) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        key,
        [
            IO.Video.Input("video"),
            IO.Boolean.Input(
                "refine_foreground_edges",
                default=True,
                tooltip="Cleaner edges on hair, fur and fine detail. Turning it off lowers the price.",
            ),
            IO.Boolean.Input(
                "subject_is_person",
                default=True,
                tooltip="Turn off when the subject is an object or animal.",
            ),
        ],
    )


class VeedRemoveVideoBackgroundNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VeedRemoveVideoBackgroundNode",
            display_name="VEED Remove Video Background",
            category="partner/video/VEED",
            description="Cut the subject out of a video with VEED and return the frames plus an alpha mask. "
            "Use them for compositing, or combine them with Join Image with Alpha and Save WEBM to write a "
            "transparent video.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        _segmentation_option("standard"),
                        _segmentation_option("fast"),
                        IO.DynamicCombo.Option(
                            "green screen",
                            [
                                IO.Video.Input("video", tooltip="Footage shot against a green screen."),
                                IO.Float.Input(
                                    "spill_suppression_strength",
                                    default=0.8,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    tooltip="How strongly green cast is removed from the subject. Raise it when "
                                    "green spots remain; lower it when the subject's colors shift.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="standard: any background, best edges, output up to 3840x2160. fast: quicker, output up to "
                    "1920x1080. green screen: keys out a green backdrop, output up to 3840x2160. Larger videos are "
                    "downscaled.",
                ),
            ],
            outputs=[
                IO.Image.Output("images"),
                IO.Mask.Output(
                    "mask",
                    tooltip="1 where the background was removed. Use Join Image with Alpha and Save WEBM for a "
                    "transparent video; add Invert Mask before Image Composite Masked.",
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.refine_foreground_edges"]),
                expr="""
                (
                  $unrefined := $lookup(widgets, "model.refine_foreground_edges") = false;
                  $rate := widgets.model = "fast"
                    ? ($unrefined ? 0.01144 : 0.01716)
                    : widgets.model = "green screen"
                      ? 0.03575
                      : ($unrefined ? 0.02145 : 0.032175);
                  {"type":"usd","usd": $rate, "format":{"suffix":"/30 frames"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        validate_video_duration(model["video"], max_duration=MAX_BACKGROUND_REMOVAL_SECONDS)
        video_url = await upload_video_to_comfyapi(cls, model["video"])
        if model["model"] == "green screen":
            route = "video-background-removal-green-screen"
            request = VeedGreenScreenRequest(
                video_url=video_url,
                output_codec="vp9",
                spill_suppression_strength=model["spill_suppression_strength"],
            )
        else:
            route = "video-background-removal-fast" if model["model"] == "fast" else "video-background-removal"
            request = VeedBackgroundRemovalRequest(
                video_url=video_url,
                output_codec="vp9",
                refine_foreground_edges=model["refine_foreground_edges"],
                subject_is_person=model["subject_is_person"],
            )
        job = await _run_job(cls, route, request)
        video = await download_url_to_video_output(job.result.files[0].url)
        return IO.NodeOutput(*video_to_images_and_mask(video))


def _clean_audio_inputs() -> list:
    return [
        IO.Float.Input(
            "strength",
            default=0.874,
            min=0.0,
            max=1.0,
            step=0.001,
            tooltip="How much noise is removed under speech. Lower keeps more room tone behind the voice; "
            "pauses between words are always fully cleaned, and values above the default change little.",
        ),
        IO.DynamicCombo.Input(
            "loudness",
            options=[
                IO.DynamicCombo.Option(
                    "normalize",
                    [
                        IO.Float.Input(
                            "target_lufs",
                            default=-19.0,
                            min=-40.0,
                            max=-8.0,
                            step=0.1,
                            tooltip="Integrated loudness of the output in LUFS. Peaks are capped at -1.1 dBTP, so loud "
                            "targets stop early (about -15 LUFS for typical speech).",
                        ),
                    ],
                ),
                IO.DynamicCombo.Option("keep input level", []),
            ],
        ),
    ]


class VeedCleanAudioNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VeedCleanAudioNode",
            display_name="VEED Clean Audio",
            category="partner/audio/VEED",
            description="Remove background noise from speech with VEED. Built for speech in any language: "
            "music and sound effects are removed too. Returns 48 kHz mono audio.",
            inputs=[
                IO.Audio.Input("audio", tooltip="Speech recording, shorter than 30 minutes."),
                *_clean_audio_inputs(),
            ],
            outputs=[IO.Audio.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr="""{"type":"usd","usd":0.017875,"format":{"suffix":"/minute"}}"""),
        )

    @classmethod
    async def execute(cls, audio: Input.Audio, strength: float, loudness: dict) -> IO.NodeOutput:
        validate_audio_duration(audio, max_duration=MAX_CLEAN_AUDIO_SECONDS)
        job = await _run_job(
            cls,
            "clean-audio",
            VeedCleanAudioRequest(
                audio_url=await _upload_audio(cls, audio),
                strength=strength,
                normalize_loudness=loudness["loudness"] == "normalize",
                target_lufs=loudness.get("target_lufs"),
                output_format="flac",
            ),
        )
        cleaned = await download_url_as_bytesio(job.result.audio.url)
        return IO.NodeOutput(audio_bytes_to_audio_input(cleaned.getvalue()))


class VeedCleanVideoAudioNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VeedCleanVideoAudioNode",
            display_name="VEED Clean Audio (in Video)",
            category="partner/video/VEED",
            description="Remove background noise from the speech in a video with VEED. Only the soundtrack changes: "
            "it is replaced with the cleaned 48 kHz mono speech.",
            inputs=[
                IO.Video.Input("video", tooltip="Video with a speech soundtrack, shorter than 30 minutes."),
                *_clean_audio_inputs(),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr="""{"type":"usd","usd":0.017875,"format":{"suffix":"/minute"}}"""),
        )

    @classmethod
    async def execute(cls, video: Input.Video, strength: float, loudness: dict) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=MAX_CLEAN_AUDIO_SECONDS)
        with av.open(video.get_stream_source()) as container:
            if not container.streams.audio:
                raise ValueError("The video has no audio track to clean.")
        job = await _run_job(
            cls,
            "clean-audio-video",
            VeedCleanAudioVideoRequest(
                video_url=await upload_video_to_comfyapi(cls, video),
                strength=strength,
                normalize_loudness=loudness["loudness"] == "normalize",
                target_lufs=loudness.get("target_lufs"),
            ),
        )
        return IO.NodeOutput(await download_url_to_video_output(job.result.video.url))


class VeedExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            VeedFabricNode,
            VeedLipsyncNode,
            VeedRemoveVideoBackgroundNode,
            VeedCleanAudioNode,
            VeedCleanVideoAudioNode,
        ]


async def comfy_entrypoint() -> VeedExtension:
    return VeedExtension()
