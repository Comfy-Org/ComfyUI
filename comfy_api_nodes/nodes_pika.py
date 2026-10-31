from pydantic import BaseModel
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.pika import (
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
    upload_video_to_comfyapi,
    validate_string,
    validate_video_duration,
)


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input("seed", default=42, min=0, max=2147483647, control_after_generate=True)


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
                _seed_input(),
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
                _seed_input(),
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
                                _seed_input(),
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
                                _seed_input(),
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
            PikaSoundtrackNode,
            PikaSFXNode,
            PikaMusicNode,
        ]


async def comfy_entrypoint() -> PikaExtension:
    return PikaExtension()
