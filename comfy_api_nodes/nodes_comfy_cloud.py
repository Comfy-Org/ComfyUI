import contextlib
import math
import posixpath
from io import BytesIO
from typing import ClassVar
from urllib.parse import quote, unquote, urlsplit

import torch

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.comfy_cloud import (
    ComfyCloudAssetInput,
    ComfyCloudGenerateRequest,
    ComfyCloudGenerateResponse,
    ComfyCloudStatusResponse,
    ComfyCloudWorkflow,
    ComfyCloudWorkflowInputs,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_bytes_to_audio_input,
    download_url_to_bytesio,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    sync_op_raw,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
)


# Must stay in step with comfyCloudOutputBuckets in cloud's services/comfy-api/config/config.go.
_OUTPUT_BUCKETS = frozenset(
    {
        "comfy-cloud-assets",
        "comfy-cloud-assets-stg",
        "comfy-cloud-assets-test",
        "partner-nodes-assets",
        "partner-nodes-assets-staging",
    }
)
_GENERATE_ENDPOINT = ApiEndpoint(path="/proxy/comfy-cloud/workflow/generate", method="POST")
_RUN_TIMEOUT_SECONDS = 2100
_POLL_INTERVAL_SECONDS = 5.0
_POLL_MAX_ATTEMPTS = int(_RUN_TIMEOUT_SECONDS / _POLL_INTERVAL_SECONDS) + 24  # +2 min of slack
_OUTPUT_DOWNLOAD_TIMEOUT = 30 * 60
_MAX_UPLOAD_IMAGE_PIXELS = 32_000_000
_MAX_UPLOAD_IMAGE_DIMENSION = 8192
_MAX_DECODED_AUDIO_BYTES = 256 * 1024 * 1024
# Hardcoded mirror of Metronome's GPU rate. Nothing links the two, so it must be
# changed by hand when the rate card moves.
#
# VERIFY AGAINST A REAL CHARGE, not the rate card. This was briefly set to
# 0.00185, read off rtx_pro_6000's card entry, which over-quoted every run by
# ~43%. Dividing an actual billed event gives the truth:
#   credits_used 1.98 / gpu_seconds 7.244173 / 211 credits-per-USD = 0.0012954
#
# THIS IS THE LIST PRICE. Never put a promotional or time-boxed rate here.
# ComfyUI is pulled, not pushed, so a user keeps whatever value they last pulled
# for as long as they like. A promo rate shipped here does not expire when the
# promotion does: the rate card returns to list, that user's node still quotes
# the discount, and they are charged MORE than they were shown, indefinitely and
# invisibly to us. A promo belongs in Metronome alone, where ending it actually
# ends it, and in the launch copy. Quoting above the charged rate is the safe
# direction; quoting below it is not.
#
# The same asymmetry applies to any list-price INCREASE, which is why a
# server-supplied per-run estimate (BE-9841) is the only way to change this
# number safely rather than a nicety.
COMFY_CLOUD_GPU_SECOND_USD = 0.001295
COMFY_CLOUD_CREDITS_PER_USD = 211
COMFY_CLOUD_GPU_SECOND_CREDITS = COMFY_CLOUD_GPU_SECOND_USD * COMFY_CLOUD_CREDITS_PER_USD
_COMFY_CLOUD_PRICE_BADGE = IO.PriceBadge(
    expr=(
        f'{{"type":"usd","usd":{COMFY_CLOUD_GPU_SECOND_USD:.6f},'
        '"format":{"suffix":"/GPU-second","approximate":true}}'
    )
)
_COMFY_CLOUD_RATE_DESCRIPTION = (
    f" Runs on a Comfy Cloud GPU, billed by how long it runs at "
    f"${COMFY_CLOUD_GPU_SECOND_USD:.6f}/GPU-second "
    f"({COMFY_CLOUD_GPU_SECOND_CREDITS:.2f} credits). Paid in credits, no Cloud "
    "subscription required."
)


# Marked beta on purpose: the workflow set is curated by hand and expected to
# change, so a node can gain or lose options, and a workflow can be retired.
# display_name is not stored in a saved graph, so this is free to remove later;
# node_id and the input names are the parts that are permanent.
_COMFY_CLOUD_BETA_SUFFIX = " [BETA]"
_COMFY_CLOUD_BETA_DESCRIPTION = (
    "BETA. The Comfy Cloud node set is still changing: options may be added or "
    "removed, and a workflow may be retired. "
)


def _comfy_cloud_display_name(display_name: str) -> str:
    """A SUFFIX, not a prefix: the menu and node search sort alphabetically, so a
    leading marker would file all of these under "[" instead of their model."""
    return display_name + _COMFY_CLOUD_BETA_SUFFIX


def _comfy_cloud_description(summary: str) -> str:
    """Node descriptions carry the beta notice and the rate. The beta notice leads
    because it is the caveat to read first; the rate is here because the price
    badge only renders on Nodes 2.0, and a plain local install still defaults to
    the classic canvas."""
    return _COMFY_CLOUD_BETA_DESCRIPTION + summary + _COMFY_CLOUD_RATE_DESCRIPTION


_TEXT_LIMITS = {
    "prompt": (1, 4096),
    "instruction": (1, 4096),
    "negative_prompt": (0, 2048),
    "lyrics": (0, 4096),
}


def _task_endpoints(task_id: str) -> tuple[ApiEndpoint, ApiEndpoint]:
    if not task_id.strip():
        raise ValueError("Comfy Cloud returned an empty task ID.")
    task_path = f"/proxy/comfy-cloud/workflow/tasks/{quote(task_id, safe='')}"
    return ApiEndpoint(path=task_path), ApiEndpoint(path=f"{task_path}/cancel", method="POST")


def _with_input_sockets(inputs: list[IO.Input]) -> list[IO.Input]:
    for input_spec in inputs:
        if isinstance(input_spec, IO.WidgetInput):
            input_spec.socketless = False
    return inputs


def _cloud_schema(
    node_id: str,
    display_name: str,
    summary: str,
    category: str,
    inputs: list[IO.Input],
    output: IO.Output,
) -> IO.Schema:
    """Every Comfy Cloud node is the same schema but for its id, name, blurb, category,
    inputs and output type, so they are all built here."""
    return IO.Schema(
        node_id=node_id,
        display_name=_comfy_cloud_display_name(display_name),
        category=category,
        description=_comfy_cloud_description(summary),
        inputs=_with_input_sockets(inputs),
        outputs=[output],
        hidden=[
            IO.Hidden.auth_token_comfy_org,
            IO.Hidden.api_key_comfy_org,
            IO.Hidden.unique_id,
        ],
        is_api_node=True,
        price_badge=_COMFY_CLOUD_PRICE_BADGE,
    )


def _validated_output_url(url: str) -> str:
    parsed = urlsplit(url)
    decoded_path = unquote(parsed.path)
    is_proxy_path = (
        not parsed.scheme
        and not parsed.netloc
        and decoded_path.startswith("/proxy/comfy-cloud/")
        and posixpath.normpath(decoded_path) == decoded_path
    )
    # normpath first: a client resolves dot segments before sending, so
    # ".../comfy-cloud-assets/../other/x.png" would advertise an allowed bucket here
    # and fetch from another one on the wire.
    bucket = decoded_path.lstrip("/").split("/", 1)[0]
    is_signed_https_url = (
        parsed.scheme == "https"
        and parsed.hostname == "storage.googleapis.com"
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
        and posixpath.normpath(decoded_path) == decoded_path
        and bucket in _OUTPUT_BUCKETS
    )
    if not is_proxy_path and not is_signed_https_url:
        raise RuntimeError("Comfy Cloud returned an invalid output URL.")
    return url


def _validate_image_upload(image: Input.Image) -> None:
    if not isinstance(image, torch.Tensor):
        return
    if image.ndim not in (3, 4):
        raise ValueError("Invalid input image shape.")
    height, width = image.shape[-3:-1]
    if max(height, width) > _MAX_UPLOAD_IMAGE_DIMENSION or height * width > _MAX_UPLOAD_IMAGE_PIXELS:
        raise ValueError("Input image exceeds the 8192px or 32-megapixel Comfy Cloud limit.")


def _progress(response: ComfyCloudStatusResponse) -> float | None:
    if response.progress is None or not math.isfinite(response.progress):
        return None
    return min(100.0, max(0.0, response.progress))


async def _poll_task(cls: type[IO.ComfyNode], task_id: str) -> ComfyCloudStatusResponse:
    polling_endpoint, cancel_endpoint = _task_endpoints(task_id)
    try:
        return await poll_op(
            cls,
            polling_endpoint,
            response_model=ComfyCloudStatusResponse,
            status_extractor=lambda response: response.status,
            progress_extractor=_progress,
            cancel_endpoint=cancel_endpoint,
            poll_interval=_POLL_INTERVAL_SECONDS,
            max_poll_attempts=_POLL_MAX_ATTEMPTS,
        )
    except Exception:
        with contextlib.suppress(Exception):
            await sync_op_raw(cls, cancel_endpoint, max_retries=0)
        raise


def _validate_node_inputs(cls: type[IO.ComfyNode], values: dict) -> dict:
    validated = dict(values)
    for input_spec in cls.define_schema().inputs:
        if not isinstance(input_spec, IO.WidgetInput) or input_spec.id not in values:
            continue
        value = values[input_spec.id]
        io_type = input_spec.get_io_type()
        if io_type == "STRING":
            if not isinstance(value, str):
                raise ValueError(f"{input_spec.id} must be a string.")
            value = value.strip()
            minimum, maximum = _TEXT_LIMITS.get(input_spec.id, (0, None))
            validate_string(
                value,
                min_length=minimum,
                max_length=maximum,
                field_name=input_spec.id,
            )
            validated[input_spec.id] = value
        elif io_type == "COMBO" and value not in input_spec.options:
            raise ValueError(f"Invalid {input_spec.id}: {value!r}.")
        elif io_type == "BOOLEAN" and not isinstance(value, bool):
            raise ValueError(f"{input_spec.id} must be a boolean.")
        elif io_type == "INT":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{input_spec.id} must be an integer.")
            if input_spec.min is not None and value < input_spec.min:
                raise ValueError(f"{input_spec.id} must be at least {input_spec.min}.")
            if input_spec.max is not None and value > input_spec.max:
                raise ValueError(f"{input_spec.id} must be at most {input_spec.max}.")
        elif io_type == "FLOAT":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{input_spec.id} must be a finite number.")
            if input_spec.min is not None and value < input_spec.min:
                raise ValueError(f"{input_spec.id} must be at least {input_spec.min}.")
            if input_spec.max is not None and value > input_spec.max:
                raise ValueError(f"{input_spec.id} must be at most {input_spec.max}.")
            if input_spec.step:
                origin = input_spec.min or 0
                steps = (value - origin) / input_spec.step
                if not math.isclose(steps, round(steps), abs_tol=1e-7):
                    raise ValueError(f"{input_spec.id} must use increments of {input_spec.step}.")
    return validated


_ASPECT_RATIOS = ["1:1", "3:4", "2:3", "3:2", "4:3", "16:9", "9:16", "21:9"]
_VIDEO_RESOLUTIONS = ["480p", "720p"]
_MINIMAX_MUSIC3_QUALITIES = ["V0", "128k", "320k"]
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF
_NEGATIVE_PROMPT_TOOLTIP = "Leave empty to keep the negative prompt this pipeline was tuned with."


def _prompt_input(name: str = "prompt") -> IO.String.Input:
    return IO.String.Input(name, multiline=True, default="")


def _negative_prompt_input() -> IO.String.Input:
    return IO.String.Input(
        "negative_prompt", multiline=True, default="", tooltip=_NEGATIVE_PROMPT_TOOLTIP
    )


def _aspect_ratio_input(default: str = "1:1") -> IO.Combo.Input:
    return IO.Combo.Input("aspect_ratio", options=_ASPECT_RATIOS, default=default)


def _megapixels_input() -> IO.Float.Input:
    # Resolution Selector graphs take a ratio and pixel budget rather than raw dimensions.
    return IO.Float.Input(
        "megapixels", default=1.0, min=0.1, max=16.0, step=0.1,
        tooltip="Total pixel budget. 1.0 is about 1024x1024 at a square ratio.",
    )


def _seed_input(maximum: int = _UINT64_MAX) -> IO.Int.Input:
    return IO.Int.Input("seed", default=42, min=0, max=maximum, control_after_generate=True)


def _video_resolution_input(advanced: bool = True) -> IO.Combo.Input:
    return IO.Combo.Input(
        "resolution",
        options=_VIDEO_RESOLUTIONS,
        default="480p",
        advanced=advanced,
        tooltip="Frame size budget. 720p costs roughly twice the GPU-seconds of 480p.",
    )


async def _submit_workflow(
    cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs
) -> str:
    task = await sync_op(
        cls,
        _GENERATE_ENDPOINT,
        response_model=ComfyCloudGenerateResponse,
        data=ComfyCloudGenerateRequest(workflow=workflow, inputs=inputs),
    )
    result = await _poll_task(cls, task.task_id)
    if not result.output_url:
        raise RuntimeError("Comfy Cloud task completed without an output URL.")
    return _validated_output_url(result.output_url)


async def _run_image_workflow(
    cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs
) -> IO.NodeOutput:
    url = await _submit_workflow(cls, workflow, inputs)
    return IO.NodeOutput(
        await download_url_to_image_tensor(
            url, timeout=_OUTPUT_DOWNLOAD_TIMEOUT, cls=cls, allow_redirects=False
        )
    )


async def _run_video_workflow(
    cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs
) -> IO.NodeOutput:
    url = await _submit_workflow(cls, workflow, inputs)
    return IO.NodeOutput(
        await download_url_to_video_output(
            url, timeout=_OUTPUT_DOWNLOAD_TIMEOUT, cls=cls, allow_redirects=False
        )
    )


async def _run_audio_workflow(
    cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs
) -> IO.NodeOutput:
    url = await _submit_workflow(cls, workflow, inputs)
    buffer = BytesIO()
    await download_url_to_bytesio(
        url, buffer, timeout=_OUTPUT_DOWNLOAD_TIMEOUT, cls=cls, allow_redirects=False
    )
    return IO.NodeOutput(audio_bytes_to_audio_input(buffer.getvalue()))


async def _upload_workflow_image(cls: type[IO.ComfyNode], image: Input.Image, **options) -> str:
    if get_number_of_images(image) != 1:
        raise ValueError("Exactly one input image is required.")
    _validate_image_upload(image)
    return await upload_image_to_comfyapi(cls, image, **options)


def _image_schema(cls: type[IO.ComfyNode], inputs: list[IO.Input]) -> IO.Schema:
    return _cloud_schema(
        cls.node_id, cls.display_name, cls.summary, "comfy cloud/image", inputs, IO.Image.Output()
    )


class ComfyCloudFlux2TextToImageNode(IO.ComfyNode):
    node_id = "ComfyCloudFlux2TextToImageNode"
    display_name = "Comfy Cloud Flux 2 Text to Image"
    summary = (
        "Generates an image from a text prompt with Flux 2 dev. Turbo swaps in the chosen LoRA "
        "and a short schedule, trading a little fidelity for a much quicker run; switch it off "
        "for the full-length dev pass with no LoRA."
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                _prompt_input(),
                _seed_input(),
                _aspect_ratio_input(),
                _megapixels_input(),
                IO.Boolean.Input(
                    "turbo",
                    default=True,
                    tooltip="Run the Turbo LoRA on a short schedule, trading a little fidelity "
                            "for a much quicker run. Off runs the full dev pass with no LoRA.",
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        seed: int = 42,
        aspect_ratio: str = "1:1",
        megapixels: float = 1.0,
        turbo: bool = True,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        return await _run_image_workflow(
            cls,
            "flux-2/text-to-image",
            ComfyCloudWorkflowInputs(
                prompt=prompt, aspect_ratio=aspect_ratio, megapixels=megapixels, turbo=turbo, seed=seed,
            ),
        )


class ComfyCloudZImageTurboNode(IO.ComfyNode):
    node_id = "ComfyCloudZImageTurboNode"
    display_name = "Comfy Cloud Z-Image Turbo Text to Image"
    summary = (
        "Generates an image from a text prompt with Z-Image Turbo in 8 steps. One of the "
        "quickest and cheapest nodes here, which makes it the one to iterate on."
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                _prompt_input(),
                _seed_input(),
                _aspect_ratio_input(),
                _megapixels_input(),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        seed: int = 42,
        aspect_ratio: str = "1:1",
        megapixels: float = 1.0,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        return await _run_image_workflow(
            cls,
            "z-image-turbo/text-to-image",
            ComfyCloudWorkflowInputs(
                prompt=prompt, aspect_ratio=aspect_ratio, megapixels=megapixels, seed=seed,
            ),
        )


class _ComfyCloudMageFlowNode(IO.ComfyNode):
    """Mage-Flow text to image. The base and turbo graphs are the same pipeline at
    two schedule lengths, so they differ only in their step and cfg defaults and
    which checkpoints load."""

    workflow: ClassVar[ComfyCloudWorkflow]
    node_id: ClassVar[str]
    display_name: ClassVar[str]
    summary: ClassVar[str]

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                _prompt_input(),
                _negative_prompt_input(),
                _seed_input(),
                _aspect_ratio_input(),
                _megapixels_input(),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 42,
        aspect_ratio: str = "1:1",
        megapixels: float = 1.0,
    ) -> IO.NodeOutput:
        validated = _validate_node_inputs(cls, locals())
        return await _run_image_workflow(
            cls,
            cls.workflow,
            ComfyCloudWorkflowInputs(
                prompt=validated["prompt"], negative_prompt=validated["negative_prompt"],
                aspect_ratio=aspect_ratio, megapixels=megapixels, seed=seed,
            ),
        )


class ComfyCloudMageFlowTextToImageNode(_ComfyCloudMageFlowNode):
    workflow = "mage-flow/text-to-image"
    node_id = "ComfyCloudMageFlowTextToImageNode"
    display_name = "Comfy Cloud Mage Flow Text to Image"
    summary = (
        "Generates an image from a text prompt with Mage-Flow over a full 30-step pass. "
        "It takes a negative prompt, which the distilled turbo variant cannot use well."
    )


class ComfyCloudMageFlowTurboTextToImageNode(_ComfyCloudMageFlowNode):
    workflow = "mage-flow-turbo/text-to-image"
    node_id = "ComfyCloudMageFlowTurboTextToImageNode"
    display_name = "Comfy Cloud Mage Flow Turbo Text to Image"
    summary = (
        "Generates an image from a text prompt with distilled Mage-Flow in 4 steps at cfg 1. "
        "Roughly a seventh of the GPU time of the full pass, which makes it the one to iterate on."
    )


class ComfyCloudMiniMaxMusic3TextToAudioNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _cloud_schema(
            "ComfyCloudMiniMaxMusic3TextToAudioNode",
            "Comfy Cloud MiniMax Music 3 Text to Audio",
            (
                "Generates a full song from a description with MiniMax Music 3. The prompt "
                "carries the style, instrumentation and mood; lyrics are sung rather than "
                "described, and an empty lyric leaves the track instrumental."
            ),
            "comfy cloud/audio",
            [
                _prompt_input(),
                IO.String.Input(
                    "lyrics", multiline=True, default="",
                    tooltip="Words to sing. Leave empty for an instrumental.",
                ),
                # SeedNode caps at int64, below the uint64 the other graphs take.
                _seed_input(0x7FFFFFFFFFFFFFFF),
                IO.Float.Input(
                    "max_duration", default=120.0, min=0.04, max=360.0, step=0.04,
                    tooltip="Longest the track may run. The model can end the song earlier.",
                ),
                IO.Combo.Input(
                    "audio_quality",
                    options=_MINIMAX_MUSIC3_QUALITIES,
                    default=_MINIMAX_MUSIC3_QUALITIES[0],
                    tooltip="mp3 bitrate. V0 is variable and the highest quality of the three.",
                ),
            ],
            IO.Audio.Output(),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        lyrics: str = "",
        seed: int = 42,
        max_duration: float = 120.0,
        audio_quality: str = "V0",
    ) -> IO.NodeOutput:
        validated = _validate_node_inputs(cls, locals())
        return await _run_audio_workflow(
            cls,
            "minimax-music-3/text-to-audio",
            ComfyCloudWorkflowInputs(
                prompt=validated["prompt"], lyrics=validated["lyrics"], seed=seed,
                max_duration=max_duration, caption_cfg=1.5,
                audio_quality=audio_quality,
            ),
        )


def _video_schema(node_id: str, display_name: str, summary: str, inputs: list[IO.Input]) -> IO.Schema:
    return _cloud_schema(
        node_id, display_name, summary, "comfy cloud/video", inputs, IO.Video.Output()
    )


_MINIMAX_H3_MAX_REFERENCES = 4
_MINIMAX_H3_MAX_AUDIO_REFERENCES = 3


def _minimax_h3_inputs(default_ratio: str, plain: list[IO.Input] | None = None) -> list[IO.Input]:
    """Everything the three fl2va/ref2va graphs expose past their media inputs.

    The turbo LoRA branch those templates carry is not here: its weights have
    catalog entries but no bytes in the mirror, so cloud's frozen graphs leave
    the branch out until they are uploaded.
    """
    return [
        _prompt_input(),
        _seed_input(),
        _aspect_ratio_input(default_ratio),
        _video_resolution_input(advanced=False),
        IO.Int.Input(
            "duration_seconds",
            default=5,
            min=5,
            max=15,
            display_mode=IO.NumberDisplay.slider,
            tooltip=(
                "Length in seconds. The pipeline quantises to 17-frame steps at 24fps, "
                "so the clip lands within about two thirds of a second of this."
            ),
        ),
        *(plain or []),
    ]


async def _minimax_h3_asset(cls: type[IO.ComfyNode], image: Input.Image) -> ComfyCloudAssetInput:
    return ComfyCloudAssetInput(
        type="IMAGE", url=await _upload_workflow_image(cls, image, total_pixels=2048 * 2048)
    )


async def _minimax_h3_video_asset(cls: type[IO.ComfyNode], video: Input.Video) -> ComfyCloudAssetInput:
    return ComfyCloudAssetInput(type="VIDEO", url=await upload_video_to_comfyapi(cls, video))


async def _minimax_h3_audio_asset(cls: type[IO.ComfyNode], audio: Input.Audio) -> ComfyCloudAssetInput:
    return ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, audio))


class ComfyCloudMiniMaxH3TextToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudMiniMaxH3TextToVideoNode",
            "Comfy Cloud MiniMax H3 Text to Video",
            (
                "Generates a video with a matching soundtrack from a text prompt, using MiniMax "
                "H3. Picture and audio come out of the same pass rather than being dubbed on "
                "afterwards."
            ),
            _minimax_h3_inputs("16:9"),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        seed: int = 42,
        aspect_ratio: str = "16:9",
        resolution: str = "480p",
        duration_seconds: int = 5
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        inputs = ComfyCloudWorkflowInputs(
            prompt=prompt, aspect_ratio=aspect_ratio, duration_seconds=duration_seconds,
            seed=seed, resolution=resolution
        )
        return await _run_video_workflow(cls, "minimax-h3/text-to-video", inputs)


# MiniMax H3's two Qwen3-VL encoder precisions. Keys, not filenames: cloud holds
# the file each one maps to.
class ComfyCloudMiniMaxH3FirstLastFrameToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudMiniMaxH3FirstLastFrameToVideoNode",
            "Comfy Cloud MiniMax H3 First-Last Frame to Video",
            (
                "Generates the motion between two keyframes, with a matching soundtrack, using "
                "MiniMax H3. Give it the opening and closing frames and the model fills in the "
                "shot between them. The last frame is optional: leave it out and the motion runs "
                "away from the first frame instead."
            ),
            [
                IO.Image.Input("first_frame"),
                IO.Image.Input("last_frame", optional=True),
                *_minimax_h3_inputs("1:1"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        first_frame: Input.Image,
        prompt: str,
        seed: int = 42,
        aspect_ratio: str = "1:1",
        resolution: str = "480p",
        duration_seconds: int = 5,
        last_frame: Input.Image | None = None,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        assets = {"first_frame": await _minimax_h3_asset(cls, first_frame)}
        if last_frame is not None:
            assets["last_frame"] = await _minimax_h3_asset(cls, last_frame)
        return await _run_video_workflow(
            cls,
            "minimax-h3/first-last-frame-to-video",
            ComfyCloudWorkflowInputs(
                prompt=prompt, aspect_ratio=aspect_ratio, resolution=resolution,
                duration_seconds=duration_seconds, seed=seed, assets=assets
            ),
        )


class ComfyCloudMiniMaxH3ImageToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudMiniMaxH3ImageToVideoNode",
            "Comfy Cloud MiniMax H3 Image to Video",
            (
                "Animates a still into a video with a matching soundtrack, using MiniMax H3. "
                "Feed it a clip's closing frame and it continues the sequence, so chaining "
                "several builds a longer shot."
            ),
            [IO.Image.Input("first_frame"), *_minimax_h3_inputs("1:1")],
        )

    @classmethod
    async def execute(
        cls,
        first_frame: Input.Image,
        prompt: str,
        seed: int = 42,
        aspect_ratio: str = "1:1",
        resolution: str = "480p",
        duration_seconds: int = 5
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        return await _run_video_workflow(
            cls,
            "minimax-h3/image-to-video",
            ComfyCloudWorkflowInputs(
                prompt=prompt, aspect_ratio=aspect_ratio, resolution=resolution,
                duration_seconds=duration_seconds, seed=seed,
                assets={"first_frame": await _minimax_h3_asset(cls, first_frame)},
            ),
        )


class ComfyCloudMiniMaxH3ReferenceToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudMiniMaxH3ReferenceToVideoNode",
            "Comfy Cloud MiniMax H3 Reference to Video",
            (
                "Generates a video with a matching soundtrack from optional image, video, and "
                "audio references, using MiniMax H3. The references carry subject and style "
                "across the shot, and the prompt addresses images by connection order."
            ),
            [
                IO.Autogrow.Input(
                    "reference_images",
                    template=IO.Autogrow.TemplatePrefix(
                        input=IO.Image.Input("reference_image"),
                        prefix="reference_image_",
                        min=0,
                        max=_MINIMAX_H3_MAX_REFERENCES,
                    ),
                    tooltip=(
                        "Up to four references, addressed in the prompt as <Picture 1> upwards "
                        "in connection order."
                    ),
                ),
                IO.Video.Input("ref_video", optional=True),
                IO.Autogrow.Input(
                    "ref_audio",
                    template=IO.Autogrow.TemplatePrefix(
                        input=IO.Audio.Input("ref_audio"),
                        prefix="ref_audio_",
                        min=0,
                        max=_MINIMAX_H3_MAX_AUDIO_REFERENCES,
                    ),
                ),
                *_minimax_h3_inputs(
                    "16:9",
                    plain=[
                        IO.Combo.Input(
                            "ref_image_size",
                            options=["match", "max"],
                            default="match",
                            tooltip=(
                                "'match' scales each reference to the output's pixel area; 'max' "
                                "sends it at the 2048px short edge for the closest likeness. "
                                "Reference tokens ride through every sampling step, so 'max' "
                                "costs several times the GPU-seconds."
                            ),
                        )
                    ],
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        reference_images: dict[str, Input.Image] | None = None,
        ref_video: Input.Video | None = None,
        ref_audio: dict[str, Input.Audio] | None = None,
        seed: int = 42,
        aspect_ratio: str = "16:9",
        resolution: str = "480p",
        duration_seconds: int = 5,
        ref_image_size: str = "match"
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        images = [image for image in (reference_images or {}).values() if image is not None]
        if len(images) > _MINIMAX_H3_MAX_REFERENCES:
            raise ValueError(f"At most {_MINIMAX_H3_MAX_REFERENCES} reference images are supported.")
        audios = [audio for audio in (ref_audio or {}).values() if audio is not None]
        if len(audios) > _MINIMAX_H3_MAX_AUDIO_REFERENCES:
            raise ValueError(f"At most {_MINIMAX_H3_MAX_AUDIO_REFERENCES} reference audio inputs are supported.")
        # Numbered by connection order, which is the order the prompt's
        # <Picture i> tags refer to them in.
        assets = {
            f"reference_image_{index}": await _minimax_h3_asset(cls, image)
            for index, image in enumerate(images, 1)
        }
        if ref_video is not None:
            assets["ref_video"] = await _minimax_h3_video_asset(cls, ref_video)
        assets.update({
            f"ref_audio_{index}": await _minimax_h3_audio_asset(cls, audio)
            for index, audio in enumerate(audios, 1)
        })
        return await _run_video_workflow(
            cls,
            "minimax-h3/reference-to-video",
            ComfyCloudWorkflowInputs(
                prompt=prompt, aspect_ratio=aspect_ratio, resolution=resolution,
                duration_seconds=duration_seconds, seed=seed, ref_image_size=ref_image_size,
                assets=assets,
            ),
        )


class ComfyCloudExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ComfyCloudMiniMaxH3TextToVideoNode,
            ComfyCloudMiniMaxH3FirstLastFrameToVideoNode,
            # ComfyCloudMiniMaxH3ReferenceToVideoNode,  # Disabled until the server-side issue is fixed.
            ComfyCloudMiniMaxH3ImageToVideoNode,
            ComfyCloudMiniMaxMusic3TextToAudioNode,
            ComfyCloudFlux2TextToImageNode,
            ComfyCloudZImageTurboNode,
            ComfyCloudMageFlowTextToImageNode,
            ComfyCloudMageFlowTurboTextToImageNode,
        ]


async def comfy_entrypoint() -> ComfyCloudExtension:
    return ComfyCloudExtension()
