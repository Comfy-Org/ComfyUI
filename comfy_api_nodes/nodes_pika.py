import re
import uuid
from io import BytesIO

import av
from pydantic import BaseModel
from typing_extensions import override

from comfy_api.v3 import IO, ComfyExtension, Input, Types
from comfy_api_nodes.apis.pika import (
    PikaH3FlexImageRequest,
    PikaH3FlexReference,
    PikaH3FlexReferenceRequest,
    PikaH3FlexRequest,
    PikaJob,
    PikaJobOutput,
    PikaMusicRequest,
    PikaSFXRequest,
    PikaSoundtrackRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_bytes_to_audio_input,
    download_url_as_bytesio,
    download_url_to_video_output,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_file_to_comfyapi,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_audio_duration,
    validate_string,
    validate_video_duration,
)

PIKA_H3_FLEX_ASPECT_RATIOS = ["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"]
PIKA_H3_FLEX_REF_TAG_RE = re.compile(r"@(?P<kind>image|video|audio)(?P<idx>\d*)(?!\w)", re.IGNORECASE | re.ASCII)
PIKA_H3_FLEX_REF_LABELS = {"image": "Picture", "video": "Video", "audio": "Audio"}


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input(
        "seed",
        default=42,
        min=0,
        max=2147483647,
        control_after_generate=True,
        tooltip="With rewrite_prompt off, the same seed repeats the video exactly. "
        "A local run with the same seed gives a different result.",
    )


def _audio_seed_input() -> IO.Int.Input:
    return IO.Int.Input("seed", default=42, min=0, max=2147483647, control_after_generate=True)


def _h3_flex_settings_inputs(aspect_ratio_tooltip: str) -> list[IO.Input]:
    return [
        IO.Combo.Input(
            "resolution",
            options=["480p", "768p", "1080p", "2k", "4k"],
            default="768p",
            tooltip="1080p and above render at 768p and upscale once. The same seed at 480p gives a different video.",
        ),
        IO.Combo.Input(
            "aspect_ratio",
            options=PIKA_H3_FLEX_ASPECT_RATIOS,
            default="auto",
            tooltip=aspect_ratio_tooltip,
        ),
        IO.Int.Input(
            "duration",
            default=5,
            min=5,
            max=15,
            tooltip="The output length follows a frame grid: 5 seconds gives 124 frames, about 5.17 seconds.",
        ),
        IO.Boolean.Input(
            "rewrite_prompt",
            default=False,
            tooltip="Let Pika expand the prompt first. The same seed then gives a different video on each run.",
        ),
        _seed_input(),
    ]


def _resolve_ref_tags(prompt: str, counts: dict[str, int]) -> str:
    parts = []
    pos = 0
    prev_end = -1
    for match in PIKA_H3_FLEX_REF_TAG_RE.finditer(prompt):
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
                f"The prompt references @{kind.capitalize()}{idx}, but {counts[kind]} {kind} input(s) are available."
            )
        parts.append(prompt[pos:start])
        parts.append(f"<{PIKA_H3_FLEX_REF_LABELS[kind]} {idx}>")
        pos = match.end()
        prev_end = match.end()
    parts.append(prompt[pos:])
    return "".join(parts)


async def _upload_reference_video(cls: type[IO.ComfyNode], video: Input.Video, keep_audio: bool, label: str) -> str:
    if keep_audio:
        return await upload_video_to_comfyapi(cls, video, wait_label=label)
    source = BytesIO()
    video.save_to(source, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
    source.seek(0)
    silent = BytesIO()
    with av.open(source) as src, av.open(silent, mode="w", format="mp4") as dst:
        src_stream = src.streams.video[0]
        dst_stream = dst.add_stream_from_template(src_stream)
        for packet in src.demux(src_stream):
            if packet.dts is not None:
                packet.stream = dst_stream
                dst.mux(packet)
    silent.seek(0)
    return await upload_file_to_comfyapi(cls, silent, f"{uuid.uuid4()}.mp4", "video/mp4", label)


async def _pika_job(cls: type[IO.ComfyNode], path: str, request: BaseModel) -> PikaJobOutput:
    job = await sync_op(
        cls,
        ApiEndpoint(path=f"/proxy/pika/v1/media/pika/{path}", method="POST"),
        response_model=PikaJob,
        data=request,
    )
    job = await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/pika/v1/media/jobs/{job.id}"),
        response_model=PikaJob,
        status_extractor=lambda r: r.status,
        completed_statuses=["completed"],
        failed_statuses=["failed"],
        queued_statuses=["queued"],
    )
    if job.output is None:
        raise Exception(f"Pika returned no output for job {job.id}.")
    return job.output


async def _pika_video(cls: type[IO.ComfyNode], path: str, request: BaseModel) -> IO.NodeOutput:
    output = await _pika_job(cls, path, request)
    return IO.NodeOutput(await download_url_to_video_output(output.video.url, cls=cls))


async def _pika_audio(cls: type[IO.ComfyNode], path: str, request: BaseModel) -> IO.NodeOutput:
    output = await _pika_job(cls, path, request)
    audio = await download_url_as_bytesio(output.audio.url, cls=cls)
    return IO.NodeOutput(audio_bytes_to_audio_input(audio.getvalue()))


class PikaH3FlexVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PikaH3FlexVideoNode",
            display_name="Pika H3 Flex Video",
            category="partner/video/Pika",
            description="Pika H3 Flex, Pika's post-trained MiniMax H3. "
            "Generate video with synchronized audio from text and optional first and last frames.",
            search_aliases=["minimax h3", "h3 flex", "pika h3", "text to video", "image to video", "first last frame"],
            inputs=[
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Image.Input("first_frame", optional=True),
                IO.Image.Input("last_frame", optional=True),
                *_h3_flex_settings_inputs("auto follows the first connected frame, otherwise the prompt."),
            ],
            outputs=[IO.Video.Output()],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution", "duration"]),
                expr="""
                (
                  $res := widgets.resolution;
                  $d := widgets.duration;
                  $rate := $lookup({"480p": 0.0143, "768p": 0.0286, "1080p": 0.05005, "2k": 0.0572, "4k": 0.1144}, $res);
                  $type($d) = "number"
                    ? {"type":"usd","usd": $rate * (($res = "480p" or $res = "768p")
                        ? $d : ($ceil((24 * $d - 5) / 17) * 17 + 5) / 24)}
                    : {"type":"usd","usd": $rate,"format":{"suffix":"/second"}}
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
        rewrite_prompt: bool,
        seed: int,
        first_frame: Input.Image | None = None,
        last_frame: Input.Image | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=8000)
        settings = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "seed": seed,
            "rewrite": rewrite_prompt,
        }
        if first_frame is None and last_frame is None:
            return await _pika_video(cls, "pika-h3-flex/text-to-video", PikaH3FlexRequest(**settings))
        return await _pika_video(
            cls,
            "pika-h3-flex/image-to-video",
            PikaH3FlexImageRequest(
                **settings,
                image=None
                if first_frame is None
                else await upload_image_to_comfyapi(cls, first_frame, wait_label="Uploading first frame"),
                last_frame_image=None
                if last_frame is None
                else await upload_image_to_comfyapi(cls, last_frame, wait_label="Uploading last frame"),
            ),
        )


class PikaH3FlexReferenceVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PikaH3FlexReferenceVideoNode",
            display_name="Pika H3 Flex Reference to Video",
            category="partner/video/Pika",
            description="Pika H3 Flex, Pika's post-trained MiniMax H3. "
            "Generate or edit video with synchronized audio guided by reference images, videos and audio.",
            search_aliases=["minimax h3", "h3 flex", "pika h3", "video editing", "motion transfer", "reference video"],
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
                    "At least one image or video is required.",
                ),
                IO.Autogrow.Input(
                    "reference_videos",
                    template=IO.Autogrow.TemplateNames(
                        IO.Video.Input("reference_video"),
                        names=[f"video_{i}" for i in range(1, 4)],
                        min=0,
                    ),
                    optional=True,
                    tooltip="Up to 3 videos, referred to in the prompt as @Video1..@Video3 in connection order. "
                    "2-15 seconds each, 15 seconds in total. Billed per second.",
                ),
                IO.Autogrow.Input(
                    "reference_audios",
                    template=IO.Autogrow.TemplateNames(
                        IO.Audio.Input("reference_audio"),
                        names=[f"audio_{i}" for i in range(1, 4)],
                        min=0,
                    ),
                    optional=True,
                    tooltip="Up to 3 clips, referred to in the prompt as @Audio1..@Audio3. "
                    "2-15 seconds each, 15 seconds in total. When copied to the output, clips play in connection order, "
                    "ahead of the sound of connected videos, whichever number the prompt names.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Refer to references as @Image1, @Video1, @Audio1, or as <Picture 1>, <Video 1>, <Audio 1>.",
                ),
                IO.Boolean.Input(
                    "reference_video_audio",
                    default=True,
                    tooltip="The sound of connected videos is copied to the output. Turn this off for new ambience or "
                    "motion transfer with new music. A talking head with no dialogue in the prompt then gets invented "
                    "speech.",
                ),
                *_h3_flex_settings_inputs(
                    "auto follows a reference video, otherwise the prompt. Reference images do not affect it."
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution", "duration"], input_groups=["reference_videos"]),
                expr="""
                (
                  $res := widgets.resolution;
                  $d := widgets.duration;
                  $rate := $lookup({"480p": 0.0429, "768p": 0.0858, "1080p": 0.10725, "2k": 0.1144, "4k": 0.143}, $res);
                  $vidsRaw := $lookup(inputGroups, "reference_videos");
                  $vids := $vidsRaw ? $vidsRaw : 0;
                  $type($d) != "number"
                    ? {"type":"usd","usd": $rate,"format":{"suffix":"/second"}}
                    : (
                        $base := $rate * (($res = "480p" or $res = "768p")
                          ? $d : ($ceil((24 * $d - 5) / 17) * 17 + 5) / 24);
                        $vids > 0
                          ? {"type":"range_usd","min_usd": $base + $vids * 2 * $rate,"max_usd": $base + 15 * $rate,
                             "format":{"approximate": true}}
                          : {"type":"usd","usd": $base}
                      )
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
        reference_video_audio: bool,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        rewrite_prompt: bool,
        seed: int,
    ) -> IO.NodeOutput:
        if not reference_images and not reference_videos:
            raise ValueError("At least one reference image or video is required.")
        if len(reference_images) + len(reference_videos) + len(reference_audios) > 12:
            raise ValueError("At most 12 references are supported in total.")
        for video in reference_videos.values():
            validate_video_duration(video, min_duration=2, max_duration=15)
        if sum(video.get_duration() for video in reference_videos.values()) > 15:
            raise ValueError("Reference videos must be at most 15 seconds in total.")
        for audio in reference_audios.values():
            validate_audio_duration(audio, min_duration=2, max_duration=15)
        if sum(audio["waveform"].shape[-1] / audio["sample_rate"] for audio in reference_audios.values()) > 15:
            raise ValueError("Reference audio must be at most 15 seconds in total.")
        prompt = _resolve_ref_tags(
            prompt,
            {
                "image": len(reference_images),
                "video": len(reference_videos),
                "audio": len(reference_audios) + (len(reference_videos) if reference_video_audio else 0),
            },
        )
        validate_string(prompt, min_length=1, max_length=8000)
        references = []
        for i, image in enumerate(reference_images.values(), 1):
            url = await upload_image_to_comfyapi(cls, image, wait_label=f"Uploading image {i}")
            references.append(PikaH3FlexReference(type="image", url=url))
        for audio in reference_audios.values():
            url = await upload_audio_to_comfyapi(
                cls, audio, container_format="wav", codec_name="pcm_s16le", mime_type="audio/wav"
            )
            references.append(PikaH3FlexReference(type="audio", url=url))
        for i, video in enumerate(reference_videos.values(), 1):
            url = await _upload_reference_video(cls, video, reference_video_audio, f"Uploading video {i}")
            references.append(PikaH3FlexReference(type="video", url=url))
        return await _pika_video(
            cls,
            "pika-h3-flex/reference-to-video",
            PikaH3FlexReferenceRequest(
                prompt=prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                seed=seed,
                rewrite=rewrite_prompt,
                references=references,
            ),
        )


class PikaSoundtrackNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PikaSoundtrackNode",
            display_name="Pika Soundtrack",
            category="partner/video/Pika",
            description="Generate a synchronized soundtrack for a video with Pika. "
            "The picture and length are kept; only the audio is replaced.",
            search_aliases=["video to audio", "foley", "sound design"],
            inputs=[
                IO.Video.Input("video", tooltip="Up to 10 minutes."),
                IO.String.Input(
                    "instruction",
                    multiline=True,
                    default="",
                    tooltip="What the audio should be: sound effects, ambience, music or style. "
                    "Leave empty to derive it from the picture.",
                ),
                _audio_seed_input(),
            ],
            outputs=[IO.Video.Output()],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr="""{"type":"usd","usd":0.00715,"format":{"suffix":"/second"}}"""),
        )

    @classmethod
    async def execute(cls, video: Input.Video, instruction: str, seed: int) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=600)
        validate_string(instruction, field_name="instruction", max_length=10000)
        return await _pika_video(
            cls,
            "pika-audio/pika-soundtrack",
            PikaSoundtrackRequest(
                video=await upload_video_to_comfyapi(cls, video),
                instruction=instruction or None,
                seed=seed,
            ),
        )


class PikaSFXNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PikaSFXNode",
            display_name="Pika SFX",
            category="partner/audio/Pika",
            description="Generate a sound effect from a text description with Pika.",
            search_aliases=["sound effects", "text to audio", "foley"],
            inputs=[
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Float.Input("duration", default=5.0, min=1.0, max=20.0, step=0.1),
                _audio_seed_input(),
            ],
            outputs=[IO.Audio.Output()],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration"]),
                expr="""
                (
                  $d := widgets.duration;
                  $type($d) = "number"
                    ? {"type":"usd","usd": 0.000286 * $d}
                    : {"type":"usd","usd": 0.000286,"format":{"suffix":"/second"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(cls, prompt: str, duration: float, seed: int) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=4000)
        return await _pika_audio(
            cls, "pika-audio/pika-sfx", PikaSFXRequest(prompt=prompt, duration=duration, seed=seed)
        )


class PikaMusicNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PikaMusicNode",
            display_name="Pika Music",
            category="partner/audio/Pika",
            description="Generate music from a text description and optional lyrics, or cover a song, with Pika.",
            search_aliases=["text to music", "song", "music cover"],
            inputs=[
                IO.DynamicCombo.Input(
                    "mode",
                    options=[
                        IO.DynamicCombo.Option(
                            "text to music",
                            [
                                IO.String.Input("prompt", multiline=True, default=""),
                                IO.String.Input(
                                    "lyrics",
                                    multiline=True,
                                    default="",
                                    tooltip="Optional. Section tags like [Verse] and [Chorus] are allowed.",
                                ),
                                IO.Float.Input("duration", default=30.0, min=10.0, max=360.0, step=0.1),
                                IO.Boolean.Input(
                                    "rewrite_prompt",
                                    default=False,
                                    tooltip="Expand a short or vague prompt with a model before generating.",
                                ),
                                _audio_seed_input(),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "cover",
                            [
                                IO.Audio.Input(
                                    "source_audio",
                                    tooltip="The song to cover. The result has the same length, clamped to 5-360 seconds.",
                                ),
                                IO.String.Input("prompt", multiline=True, default=""),
                                IO.String.Input(
                                    "lyrics",
                                    multiline=True,
                                    default="",
                                    tooltip="Optional. Section tags like [Verse] and [Chorus] are allowed.",
                                ),
                                IO.Boolean.Input(
                                    "rewrite_prompt",
                                    default=False,
                                    tooltip="Expand a short or vague prompt with a model before generating.",
                                ),
                                _audio_seed_input(),
                            ],
                        ),
                    ],
                ),
            ],
            outputs=[IO.Audio.Output()],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["mode", "mode.duration"]),
                expr="""
                (
                  $d := $lookup(widgets, "mode.duration");
                  widgets.mode = "text to music" and $type($d) = "number"
                    ? {"type":"usd","usd": 0.0003575 * $d}
                    : {"type":"usd","usd": 0.0003575,"format":{"suffix":"/second"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(cls, mode: dict) -> IO.NodeOutput:
        validate_string(mode["prompt"], min_length=1, max_length=4000)
        validate_string(mode["lyrics"], field_name="lyrics", max_length=8000)
        is_cover = mode["mode"] == "cover"
        return await _pika_audio(
            cls,
            "pika-audio/pika-music",
            PikaMusicRequest(
                mode="music_cover" if is_cover else "text_to_music",
                prompt=mode["prompt"],
                lyrics=mode["lyrics"] or None,
                source_audio=await upload_audio_to_comfyapi(cls, mode["source_audio"]) if is_cover else None,
                rewrite_prompt=mode["rewrite_prompt"],
                duration=None if is_cover else mode["duration"],
                seed=mode["seed"],
            ),
        )


class PikaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            PikaH3FlexVideoNode,
            PikaH3FlexReferenceVideoNode,
            PikaSoundtrackNode,
            PikaSFXNode,
            PikaMusicNode,
        ]


async def comfy_entrypoint() -> PikaExtension:
    return PikaExtension()
