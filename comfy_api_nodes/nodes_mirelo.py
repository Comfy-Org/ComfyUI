import torch
from pydantic import BaseModel
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.mirelo import (
    MireloJob,
    MireloMediaURL,
    MireloOutput,
    MireloTextInput,
    MireloTextToSFXRequest,
    MireloVideoControls,
    MireloVideoInput,
    MireloVideoToSFXRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_bytes_to_audio_input,
    download_url_as_bytesio,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
    validate_video_duration,
)

MIRELO_MODELS = ["sfx-2.0"]


async def _mirelo_job(cls: type[IO.ComfyNode], family: str, request: BaseModel) -> MireloJob:
    response = await sync_op(
        cls,
        ApiEndpoint(path=f"/proxy/mirelo/v3/{family}/generations", method="POST"),
        response_model=MireloJob,
        data=request,
    )
    return await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/mirelo/v3/{family}/generations/{response.id}"),
        response_model=MireloJob,
        status_extractor=lambda r: r.status,
        completed_statuses=["succeeded", "partially_succeeded"],
        failed_statuses=["failed", "canceled", "expired"],
        queued_statuses=["queued"],
    )


async def _mirelo_audio(cls: type[IO.ComfyNode], output: MireloOutput | None) -> Input.Audio | None:
    if output is None:
        return None
    for variant in output.variants:
        if variant.status == "succeeded" and variant.files and variant.files.audio:
            audio = await download_url_as_bytesio(variant.files.audio.url, cls=cls)
            return audio_bytes_to_audio_input(audio.getvalue())
    return None


def _no_audio_error(job: MireloJob) -> Exception:
    return Exception(f"Mirelo returned no audio (status: {job.status}, errors: {[e.model_dump() for e in job.errors]}).")


class MireloVideoToSFX(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MireloVideoToSFX",
            display_name="Mirelo Video to SFX",
            category="partner/audio/Mirelo",
            description="Generate sound effects synced to a video with Mirelo SFX.",
            search_aliases=["sound effects", "foley", "video to audio"],
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            model_id,
                            [
                                IO.Video.Input(
                                    "video",
                                    tooltip="Sound is generated for the whole video, 1 to 600 seconds.",
                                ),
                                IO.Audio.Input(
                                    "audio",
                                    optional=True,
                                    tooltip="Dialogue track to use instead of the video's own soundtrack "
                                    "for preserve_speech and the speech output.",
                                ),
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Optional. The model describes the video itself "
                                    "and does not always follow the prompt.",
                                ),
                                IO.String.Input(
                                    "negative_prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Sounds to avoid, as a short comma-separated list.",
                                ),
                                IO.Boolean.Input(
                                    "preserve_speech",
                                    default=True,
                                    tooltip="Keep the dialogue mixed under the generated sound. "
                                    "Off returns sound effects only.",
                                ),
                                IO.Int.Input(
                                    "seed",
                                    default=42,
                                    min=0,
                                    max=4294967295,
                                    control_after_generate=True,
                                ),
                            ],
                        )
                        for model_id in MIRELO_MODELS
                    ],
                    tooltip="Model to use.",
                ),
            ],
            outputs=[
                IO.Audio.Output(display_name="audio"),
                IO.Audio.Output(
                    display_name="speech",
                    tooltip="The dialogue separated from the soundtrack; silent when no speech is found.",
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(),
                expr="""{"type":"usd","usd": 0.0132533115,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(cls, model: dict):
        validate_video_duration(model["video"], min_duration=1, max_duration=600)
        validate_string(model["prompt"], max_length=2000)
        validate_string(model["negative_prompt"], field_name="negative_prompt", max_length=1000)
        audio_url = None
        if model.get("audio") is not None:
            audio_url = MireloMediaURL(url=await upload_audio_to_comfyapi(cls, model["audio"]))
        job = await _mirelo_job(
            cls,
            "video-to-sfx",
            MireloVideoToSFXRequest(
                model=model["model"],
                seed=model["seed"],
                input=MireloVideoInput(
                    video=MireloMediaURL(url=await upload_video_to_comfyapi(cls, model["video"])),
                    audio=audio_url,
                    prompt=model["prompt"] or None,
                    negative_prompt=model["negative_prompt"] or None,
                ),
                controls=MireloVideoControls(preserve_speech=model["preserve_speech"], separate_speech=True),
            ),
        )
        outputs = {output.type: output for output in job.result.outputs}
        mix = await _mirelo_audio(cls, outputs.get("mix"))
        if mix is None:
            raise _no_audio_error(job)
        speech = await _mirelo_audio(cls, outputs.get("speech"))
        if speech is None:
            speech = {"waveform": torch.zeros_like(mix["waveform"]), "sample_rate": mix["sample_rate"]}
        return IO.NodeOutput(mix, speech)


class MireloTextToSFX(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MireloTextToSFX",
            display_name="Mirelo Text to SFX",
            category="partner/audio/Mirelo",
            description="Generate sound effects from a text description with Mirelo SFX.",
            search_aliases=["sound effects", "foley", "text to audio"],
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            model_id,
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Describe the sound, not the picture.",
                                ),
                                IO.String.Input(
                                    "negative_prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Sounds to avoid, as a short comma-separated list.",
                                ),
                                IO.Float.Input(
                                    "duration",
                                    default=5.0,
                                    min=0.1,
                                    max=600.0,
                                    step=0.1,
                                    tooltip="Length of the sound in seconds.",
                                ),
                                IO.Int.Input(
                                    "seed",
                                    default=42,
                                    min=0,
                                    max=4294967295,
                                    control_after_generate=True,
                                ),
                            ],
                        )
                        for model_id in MIRELO_MODELS
                    ],
                    tooltip="Model to use.",
                ),
            ],
            outputs=[
                IO.Audio.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model.duration"]),
                expr="""
                (
                  $d := $lookup(widgets, "model.duration");
                  $type($d) = "number"
                    ? {"type":"usd","usd": 0.0132533115 * $d}
                    : {"type":"usd","usd": 0.0132533115,"format":{"suffix":"/second"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(cls, model: dict):
        validate_string(model["prompt"], min_length=1, max_length=2000)
        validate_string(model["negative_prompt"], field_name="negative_prompt", max_length=1000)
        job = await _mirelo_job(
            cls,
            "text-to-sfx",
            MireloTextToSFXRequest(
                model=model["model"],
                duration_ms=round(model["duration"] * 1000),
                seed=model["seed"],
                input=MireloTextInput(prompt=model["prompt"], negative_prompt=model["negative_prompt"] or None),
            ),
        )
        audio = await _mirelo_audio(cls, job.result.output)
        if audio is None:
            raise _no_audio_error(job)
        return IO.NodeOutput(audio)


class MireloExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            MireloVideoToSFX,
            MireloTextToSFX,
        ]


async def comfy_entrypoint() -> MireloExtension:
    return MireloExtension()
