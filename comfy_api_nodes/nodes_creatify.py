import asyncio
import math
import re

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.creatify import (
    CreatifyAuroraRequest,
    CreatifyBorealRequest,
    CreatifyTask,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    get_image_dimensions,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_image_dimensions,
    validate_string,
    validate_video_duration,
)

BOREAL_ASPECT_RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4}
BOREAL_REF_TAG_RE = re.compile(r"@(?P<kind>image|video|audio)(?P<idx>\d*)(?!\w)", re.IGNORECASE | re.ASCII)

BOREAL_PRICE_PER_HALF_CREDIT = 0.0572
BOREAL_RATES_EXPR = '{"768p": 13, "1088p": 27, "2k": 53}'


def _seed_input(tooltip: str) -> IO.Int.Input:
    return IO.Int.Input(
        "seed",
        default=42,
        min=0,
        max=2147483647,
        control_after_generate=True,
        tooltip=tooltip,
    )


def _boreal_settings_inputs(aspect_ratio_tooltip: str) -> list[IO.Input]:
    return [
        IO.Combo.Input("resolution", options=["768p", "1088p", "2k"], default="768p"),
        IO.Combo.Input(
            "aspect_ratio",
            options=["auto", *BOREAL_ASPECT_RATIOS],
            default="auto",
            tooltip=aspect_ratio_tooltip,
        ),
        IO.Int.Input("duration", default=5, min=1, max=15, tooltip="At 2k the maximum is 14 seconds."),
        IO.Combo.Input(
            "prompt_enhancement",
            options=["auto", "ads", "general", "none"],
            default="auto",
            tooltip="auto and ads rewrite the prompt for a video ad and can add spoken lines; general rewrites it "
            "for any other video; none renders the prompt as written.",
        ),
        _seed_input(
            "Seed controls whether the node should re-run. Creatify picks its own randomness: "
            "with prompt_enhancement none the same inputs give the same video."
        ),
    ]


def _validate_boreal_duration(resolution: str, duration: int) -> None:
    if resolution == "2k" and duration > 14:
        raise ValueError("At 2k the maximum duration is 14 seconds.")


def _resolve_ref_tags(prompt: str, counts: dict[str, int]) -> str:
    parts = []
    pos = 0
    prev_end = -1
    for match in BOREAL_REF_TAG_RE.finditer(prompt):
        start = match.start()
        if (
            start > 0
            and start != prev_end
            and prompt[start - 1].isascii()
            and (prompt[start - 1].isalnum() or prompt[start - 1] == "_")
        ):
            continue
        kind = match.group("kind").lower()
        idx = int(match.group("idx") or 1)
        if not 1 <= idx <= counts[kind]:
            raise ValueError(
                f"The prompt references @{kind.capitalize()}{idx}, but {counts[kind]} {kind} input(s) are connected."
            )
        parts.append(prompt[pos:start])
        parts.append(f"{kind.capitalize()} {idx}")
        pos = match.end()
        prev_end = match.end()
    parts.append(prompt[pos:])
    return "".join(parts)


async def _creatify_video(
    cls: type[IO.ComfyNode], endpoint: str, request: CreatifyBorealRequest | CreatifyAuroraRequest
) -> IO.NodeOutput:
    task = await sync_op(
        cls,
        ApiEndpoint(path=f"/proxy/creatify/api/{endpoint}", method="POST"),
        response_model=CreatifyTask,
        data=request,
    )
    task = await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/creatify/api/{endpoint}/{task.id}"),
        response_model=CreatifyTask,
        status_extractor=lambda t: t.status,
        completed_statuses=["done", "failed", "rejected"],
        failed_statuses=[],
        queued_statuses=["pending", "in_queue"],
    )
    if task.status != "done":
        raise ValueError(f"Creatify task {task.status}: {task.failed_reason or 'no reason given'}")
    if not task.video_output:
        raise ValueError(f"Creatify task {task.id} finished without a video.")
    return IO.NodeOutput(await download_url_to_video_output(task.video_output, cls=cls))


class CreatifyBorealH3VideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CreatifyBorealH3VideoNode",
            display_name="Creatify Boreal-H3 Video",
            category="partner/video/Creatify",
            description="Creatify Boreal-H3, Creatify's post-trained H3 model built for video ads. "
            "Generate video with sound from text, optionally starting from a first frame and ending on a last frame.",
            search_aliases=["boreal", "h3", "text to video", "image to video", "first last frame"],
            inputs=[
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Image.Input("first_frame", optional=True),
                IO.Image.Input("last_frame", optional=True, tooltip="Needs a first frame."),
                *_boreal_settings_inputs(
                    "Used without a first frame. With a first frame keep auto: the video follows the frame's ratio."
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution", "duration"]),
                expr=f"""
                (
                  $rate := $lookup({BOREAL_RATES_EXPR}, widgets.resolution);
                  {{"type":"usd","usd": $ceil($rate * widgets.duration / 50) * {BOREAL_PRICE_PER_HALF_CREDIT}}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        prompt_enhancement: str,
        seed: int,
        first_frame: Input.Image | None = None,
        last_frame: Input.Image | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)
        if last_frame is not None and first_frame is None:
            raise ValueError("A last frame needs a first frame.")
        if first_frame is not None and aspect_ratio != "auto":
            raise ValueError("Set aspect_ratio to auto when a first frame is connected; the video follows its ratio.")
        _validate_boreal_duration(resolution, duration)
        return await _creatify_video(
            cls,
            "boreal",
            CreatifyBorealRequest(
                model="boreal-h3",
                prompt=prompt,
                image_url=None
                if first_frame is None
                else await upload_image_to_comfyapi(cls, first_frame, wait_label="Uploading first frame"),
                end_image_url=None
                if last_frame is None
                else await upload_image_to_comfyapi(cls, last_frame, wait_label="Uploading last frame"),
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                prompt_enhancement=prompt_enhancement,
            ),
        )


class CreatifyBorealH3ReferenceVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CreatifyBorealH3ReferenceVideoNode",
            display_name="Creatify Boreal-H3 Reference to Video",
            category="partner/video/Creatify",
            description="Creatify Boreal-H3, Creatify's post-trained H3 model built for video ads. "
            "Generate video with sound guided by reference images, videos and audio.",
            search_aliases=["boreal", "h3", "reference to video", "reference video", "motion transfer"],
            inputs=[
                IO.Autogrow.Input(
                    "reference_images",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("reference_image"),
                        names=[f"image_{i}" for i in range(1, 10)],
                        min=0,
                    ),
                    optional=True,
                    tooltip="Up to 9 images, referred to in the prompt as @Image1..@Image9 in connection order. "
                    "At least one image or video is required, 12 references at most in total.",
                ),
                IO.Autogrow.Input(
                    "reference_videos",
                    template=IO.Autogrow.TemplateNames(
                        IO.Video.Input("reference_video"),
                        names=[f"video_{i}" for i in range(1, 4)],
                        min=0,
                    ),
                    optional=True,
                    tooltip="Up to 3 videos of at least 1 second, referred to as @Video1..@Video3. Billed for at most "
                    "15 seconds of video in total. Their sound is reused when the prompt asks to continue it; "
                    "a new line in the prompt replaces it.",
                ),
                IO.Autogrow.Input(
                    "reference_audios",
                    template=IO.Autogrow.TemplateNames(
                        IO.Audio.Input("reference_audio"),
                        names=[f"audio_{i}" for i in range(1, 4)],
                        min=0,
                    ),
                    optional=True,
                    tooltip="Up to 3 clips, referred to as @Audio1..@Audio3. Speech guides the voice, not the words; "
                    "instrumental music can become the soundtrack.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Refer to references as @Image1, @Video1, @Audio1.",
                ),
                *_boreal_settings_inputs("auto follows the first reference image, otherwise the first reference video."),
            ],
            outputs=[IO.Video.Output()],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["resolution", "duration"],
                    input_groups=["reference_images", "reference_videos"],
                ),
                expr=f"""
                (
                  $res := widgets.resolution;
                  $imgsRaw := $lookup(inputGroups, "reference_images");
                  $vidsRaw := $lookup(inputGroups, "reference_videos");
                  $imgs := $imgsRaw ? $imgsRaw : 0;
                  $vids := $vidsRaw ? $vidsRaw : 0;
                  $vrate := $res = "768p" ? 40 : 60;
                  $base := $lookup({BOREAL_RATES_EXPR}, $res) * widgets.duration
                    + 20 * $max([0, $imgs - 5]);
                  $vids > 0
                    ? {{"type":"range_usd",
                       "min_usd": $ceil(($base + $vrate * $vids) / 50) * {BOREAL_PRICE_PER_HALF_CREDIT},
                       "max_usd": $ceil(($base + $vrate * 15) / 50) * {BOREAL_PRICE_PER_HALF_CREDIT},
                       "format":{{"approximate": true}}}}
                    : {{"type":"usd","usd": $ceil($base / 50) * {BOREAL_PRICE_PER_HALF_CREDIT}}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        reference_images: dict,
        reference_videos: dict,
        reference_audios: dict,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        prompt_enhancement: str,
        seed: int,
    ) -> IO.NodeOutput:
        if not reference_images and not reference_videos:
            raise ValueError("Connect at least one reference image or video.")
        if len(reference_images) + len(reference_videos) + len(reference_audios) > 12:
            raise ValueError("At most 12 references are supported in total.")
        for video in reference_videos.values():
            validate_video_duration(video, min_duration=1)
        _validate_boreal_duration(resolution, duration)
        prompt = _resolve_ref_tags(
            prompt,
            {"image": len(reference_images), "video": len(reference_videos), "audio": len(reference_audios)},
        )
        validate_string(prompt, min_length=1)
        if resolution == "1088p" and aspect_ratio == "auto" and not reference_images:
            width, height = next(iter(reference_videos.values())).get_dimensions()
            aspect_ratio = min(BOREAL_ASPECT_RATIOS, key=lambda r: abs(math.log(width / height / BOREAL_ASPECT_RATIOS[r])))
        image_urls, video_urls, audio_urls = await asyncio.gather(
            asyncio.gather(
                *(
                    upload_image_to_comfyapi(cls, image, wait_label=f"Uploading image {i}")
                    for i, image in enumerate(reference_images.values(), 1)
                )
            ),
            asyncio.gather(
                *(
                    upload_video_to_comfyapi(cls, video, wait_label=f"Uploading video {i}")
                    for i, video in enumerate(reference_videos.values(), 1)
                )
            ),
            asyncio.gather(
                *(
                    upload_audio_to_comfyapi(
                        cls, audio, container_format="wav", codec_name="pcm_s16le", mime_type="audio/wav"
                    )
                    for audio in reference_audios.values()
                )
            ),
        )
        return await _creatify_video(
            cls,
            "boreal",
            CreatifyBorealRequest(
                model="boreal-h3",
                prompt=prompt,
                reference_image_urls=image_urls or None,
                reference_video_urls=video_urls or None,
                reference_audio_urls=audio_urls or None,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                prompt_enhancement=prompt_enhancement,
            ),
        )


class CreatifyAurora2AvatarNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CreatifyAurora2AvatarNode",
            display_name="Creatify Aurora 2.0 Avatar",
            category="partner/video/Creatify",
            description="Creatify Aurora 2.0: turn one photo and a speech or song clip into a talking or singing video. "
            "The video is as long as the audio and keeps it unchanged.",
            search_aliases=["aurora", "avatar", "talking head", "lip sync", "photo to video", "singing"],
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="At least 256 pixels on each side, aspect ratio between 1:4 and 4:1. "
                    "The video keeps the photo's aspect ratio.",
                ),
                IO.Audio.Input(
                    "audio",
                    tooltip="Speech or song, up to 60 seconds (14 at 2k). Billed per started second.",
                ),
                IO.Combo.Input("resolution", options=["480p", "720p", "1080p", "2k"], default="720p"),
                IO.String.Input(
                    "text_prompt",
                    multiline=True,
                    default="",
                    tooltip="Optional: scene, camera, lighting, expression or motion.",
                ),
                _seed_input("Seed controls whether the node should re-run; results are non-deterministic regardless of seed."),
            ],
            outputs=[IO.Video.Output()],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution"]),
                expr="""
                {
                  "type":"usd",
                  "usd": $lookup({"480p": 0.04576, "720p": 0.08008, "1080p": 0.16016, "2k": 0.32032}, widgets.resolution),
                  "format":{"suffix":"/second"}
                }
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        audio: Input.Audio,
        resolution: str,
        text_prompt: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_image_dimensions(image, min_width=256, min_height=256)
        height, width = get_image_dimensions(image)
        if not 0.25 <= width / height <= 4:
            raise ValueError(f"The photo's aspect ratio must be between 1:4 and 4:1, got {width}x{height}.")
        limit = 14 if resolution == "2k" else 60
        seconds = audio["waveform"].shape[-1] / audio["sample_rate"]
        if seconds > limit:
            raise ValueError(f"The audio is {seconds:.1f} seconds; Aurora 2.0 takes up to {limit} seconds at {resolution}.")
        image_url, audio_url = await asyncio.gather(
            upload_image_to_comfyapi(cls, image, wait_label="Uploading photo"),
            upload_audio_to_comfyapi(cls, audio, container_format="wav", codec_name="pcm_s16le", mime_type="audio/wav"),
        )
        return await _creatify_video(
            cls,
            "aurora",
            CreatifyAuroraRequest(
                image=image_url,
                audio=audio_url,
                model_version="aurora_v2",
                resolution=resolution,
                text_prompt=text_prompt.strip() or None,
            ),
        )


class CreatifyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            CreatifyBorealH3VideoNode,
            CreatifyBorealH3ReferenceVideoNode,
            CreatifyAurora2AvatarNode,
        ]


async def comfy_entrypoint() -> CreatifyExtension:
    return CreatifyExtension()
