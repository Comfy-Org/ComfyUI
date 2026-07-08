from fractions import Fraction

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, InputImpl, Types
from comfy_api_nodes.apis.beeble import (
    CreateSwitchXRequest,
    SwitchXStatusResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    convert_mask_to_image,
    download_url_as_bytesio,
    download_url_to_image_tensor,
    download_url_to_video_output,
    downscale_image_tensor,
    downscale_video_to_max_pixels,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
    validate_video_frame_count,
)

_MAX_PIXELS = 2_770_000
_MAX_FRAMES = 240
_MAX_PROMPT_LEN = 2000


def _validate_inputs(prompt: str | None, reference_image: Input.Image | None) -> str | None:
    """Beeble requires at least one of prompt or reference_image. Returns the cleaned prompt."""
    cleaned = prompt.strip() if prompt else ""
    if not cleaned and reference_image is None:
        raise ValueError("At least one of 'prompt' or 'reference_image' must be provided.")
    if cleaned:
        validate_string(cleaned, strip_whitespace=False, max_length=_MAX_PROMPT_LEN)
    return cleaned or None


async def _upload_mask_as_image(
    cls: type[IO.ComfyNode],
    mask: Input.Image,
    *,
    wait_label: str,
) -> str:
    """Encode a single-frame MASK (H, W) or (1, H, W) as a PNG and upload."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    image = convert_mask_to_image(mask[:1])
    return await upload_image_to_comfyapi(
        cls,
        image,
        mime_type="image/png",
        wait_label=wait_label,
        total_pixels=_MAX_PIXELS,
    )


async def _upload_mask_batch_as_video(
    cls: type[IO.ComfyNode],
    mask: Input.Image,
    *,
    frame_rate: Fraction,
    source_frame_count: int,
    wait_label: str,
) -> str:
    """Encode a MASK batch (N, H, W) as a grayscale H.264 MP4 at frame_rate and upload.

    The matte is always downscaled to the pixel budget so it stays within Beeble's limit and
    keeps the same dimensions as the (similarly downscaled) source — both use the same algorithm
    from the same starting dimensions, and downscaling is a no-op when already within budget.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] != source_frame_count:
        raise ValueError(
            f"Custom alpha video frame count ({mask.shape[0]}) does not match the "
            f"source video frame count ({source_frame_count}). The Beeble API requires "
            "one mask per source frame."
        )
    images = downscale_image_tensor(convert_mask_to_image(mask), _MAX_PIXELS)
    alpha_video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=None, frame_rate=frame_rate))
    return await upload_video_to_comfyapi(cls, alpha_video, wait_label=wait_label)


def _alpha_mode_input(*, video: bool) -> IO.DynamicCombo.Input:
    """Build the alpha_mode DynamicCombo with mode-specific extra inputs."""
    select_keyframe_tooltip = (
        "First-frame keyframe mask. Beeble propagates this across the video." if video else "Grayscale keyframe mask."
    )
    custom_tooltip = (
        "Per-frame grayscale mask covering the entire video. "
        "Must have the same frame count as the source. "
        "Connect a MASK output from SAM3_TrackToMask or similar."
        if video
        else "Grayscale mask to apply."
    )
    return IO.DynamicCombo.Input(
        "alpha_mode",
        tooltip=(
            "Controls how SwitchX decides what to keep vs. regenerate. "
            "'auto' isolates the main subject automatically. "
            "'fill' regenerates the entire frame while preserving geometry. "
            "'select' propagates a first-frame keyframe across the clip. "
            "'custom' uses a per-frame alpha matte you provide."
        ),
        options=[
            IO.DynamicCombo.Option("auto", []),
            IO.DynamicCombo.Option("fill", []),
            IO.DynamicCombo.Option(
                "select",
                [IO.Mask.Input("alpha_keyframe", tooltip=select_keyframe_tooltip)],
            ),
            IO.DynamicCombo.Option(
                "custom",
                [IO.Mask.Input("alpha_mask", tooltip=custom_tooltip)],
            ),
        ],
    )


def _common_inputs(*, source: IO.Input, video: bool) -> list[IO.Input]:
    return [
        source,
        IO.String.Input(
            "prompt",
            multiline=True,
            default="",
            tooltip=(
                "Text description of the desired output (max 2000 chars). "
                "At least one of 'prompt' or 'reference_image' is required."
            ),
        ),
        IO.Image.Input(
            "reference_image",
            optional=True,
            tooltip=(
                "Reference image whose look (background, lighting, costume) the result "
                "should adopt. At least one of 'reference_image' or 'prompt' is required."
            ),
        ),
        _alpha_mode_input(video=video),
        IO.Combo.Input(
            "max_resolution",
            options=["1080p", "720p"],
            default="1080p",
            tooltip="Maximum output resolution.",
        ),
        IO.Int.Input(
            "seed",
            default=0,
            min=0,
            max=2147483647,
            control_after_generate=True,
            tooltip=(
                "Seed controls whether the node should re-run; " "results are non-deterministic regardless of seed."
            ),
        ),
    ]


async def _submit_and_poll(
    cls: type[IO.ComfyNode],
    request: CreateSwitchXRequest,
) -> SwitchXStatusResponse:
    initial = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/beeble/v1/switchx/generations", method="POST"),
        response_model=SwitchXStatusResponse,
        data=request,
    )
    return await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/beeble/v1/switchx/generations/{initial.id}"),
        response_model=SwitchXStatusResponse,
        status_extractor=lambda r: r.status,
        progress_extractor=lambda r: r.progress,
    )


def _require_output_url(response: SwitchXStatusResponse, name: str) -> str:
    if response.output is None or getattr(response.output, name) is None:
        raise RuntimeError(f"Beeble job {response.id} completed without a {name!r} output URL.")
    return getattr(response.output, name)


def _alpha_url(response: SwitchXStatusResponse, mode: str) -> str | None:
    """URL of the alpha matte, or None when the mode produces no separate matte.

    'fill' selects the whole frame, so Beeble writes no alpha asset even though the status
    response still returns a (dangling) signed URL for it — fetching it 403s with S3
    AccessDenied. The other three modes ('auto', 'custom', 'select') all produce a real,
    downloadable matte.
    """
    if mode == "fill" or response.output is None:
        return None
    return response.output.alpha


class BeebleSwitchXVideoEdit(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="BeebleSwitchXVideoEdit",
            display_name="Beeble SwitchX Video Edit",
            category="partner/video/Beeble",
            description=(
                "Edit a video with Beeble SwitchX. Switches anything in the scene (background, "
                "lighting, costume) while preserving the original subject's pixels and motion. "
                "Provide a reference image and/or text prompt to describe the new look. "
                "Max 240 frames, max ~2.77MP per frame."
            ),
            inputs=_common_inputs(source=IO.Video.Input("video"), video=True),
            outputs=[
                IO.Video.Output(display_name="video"),
                IO.Video.Output(
                    display_name="alpha",
                    tooltip="The alpha matte Beeble used. Empty for 'fill' mode, which has no separate matte.",
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["max_resolution"]),
                expr="""
                (
                  $rate := widgets.max_resolution = "1080p" ? 0.429 : 0.143;
                  {"type":"usd","usd": $rate, "format":{"suffix":"/30 frames"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        prompt: str,
        alpha_mode: dict,
        max_resolution: str,
        seed: int,
        reference_image: Input.Image | None = None,
    ) -> IO.NodeOutput:
        cleaned_prompt = _validate_inputs(prompt, reference_image)

        validate_video_frame_count(video, max_frame_count=_MAX_FRAMES)
        video = downscale_video_to_max_pixels(video, _MAX_PIXELS)

        mode = alpha_mode["alpha_mode"]
        alpha_uri: str | None = None
        if mode == "select":
            alpha_uri = await _upload_mask_as_image(cls, alpha_mode["alpha_keyframe"], wait_label="Uploading keyframe")
        elif mode == "custom":
            alpha_uri = await _upload_mask_batch_as_video(
                cls,
                alpha_mode["alpha_mask"],
                frame_rate=video.get_frame_rate(),
                source_frame_count=video.get_frame_count(),
                wait_label="Uploading alpha video",
            )

        source_uri = await upload_video_to_comfyapi(cls, video, wait_label="Uploading source")
        reference_uri: str | None = None
        if reference_image is not None:
            reference_uri = await upload_image_to_comfyapi(
                cls,
                reference_image,
                mime_type="image/png",
                wait_label="Uploading reference",
                total_pixels=_MAX_PIXELS,
            )

        request = CreateSwitchXRequest(
            generation_type="video",
            source_uri=source_uri,
            alpha_mode=mode,
            prompt=cleaned_prompt,
            reference_image_uri=reference_uri,
            alpha_uri=alpha_uri,
            max_resolution=1080 if max_resolution == "1080p" else 720,
        )
        response = await _submit_and_poll(cls, request)

        render = await download_url_to_video_output(_require_output_url(response, "render"))
        alpha = None
        if (alpha_url := _alpha_url(response, mode)) is not None:
            alpha = await download_url_to_video_output(alpha_url)
        return IO.NodeOutput(render, alpha)


class BeebleSwitchXImageEdit(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="BeebleSwitchXImageEdit",
            display_name="Beeble SwitchX Image Edit",
            category="partner/image/Beeble",
            description=(
                "Edit a single image with Beeble SwitchX. Switches anything in the scene "
                "(background, lighting, costume) while preserving the original subject's pixels. "
                "Provide a reference image and/or text prompt to describe the new look. "
                "Max ~2.77MP."
            ),
            inputs=_common_inputs(source=IO.Image.Input("image"), video=False),
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(
                    display_name="alpha",
                    tooltip="The alpha matte Beeble used. Empty for 'fill' mode, which has no separate matte.",
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["max_resolution"]),
                expr="""
                (
                  $rate := widgets.max_resolution = "1080p" ? 0.429 : 0.143;
                  {"type":"usd","usd": $rate}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        prompt: str,
        alpha_mode: dict,
        max_resolution: str,
        seed: int,
        reference_image: Input.Image | None = None,
    ) -> IO.NodeOutput:
        cleaned_prompt = _validate_inputs(prompt, reference_image)

        image = downscale_image_tensor(image, _MAX_PIXELS)

        mode = alpha_mode["alpha_mode"]
        alpha_uri: str | None = None
        if mode == "select":
            alpha_uri = await _upload_mask_as_image(cls, alpha_mode["alpha_keyframe"], wait_label="Uploading keyframe")
        elif mode == "custom":
            alpha_uri = await _upload_mask_as_image(cls, alpha_mode["alpha_mask"], wait_label="Uploading alpha")

        source_uri = await upload_image_to_comfyapi(
            cls,
            image,
            mime_type="image/png",
            wait_label="Uploading source",
            total_pixels=None,
        )
        reference_uri: str | None = None
        if reference_image is not None:
            reference_uri = await upload_image_to_comfyapi(
                cls,
                reference_image,
                mime_type="image/png",
                wait_label="Uploading reference",
                total_pixels=_MAX_PIXELS,
            )

        request = CreateSwitchXRequest(
            generation_type="image",
            source_uri=source_uri,
            alpha_mode=mode,
            prompt=cleaned_prompt,
            reference_image_uri=reference_uri,
            alpha_uri=alpha_uri,
            max_resolution=1080 if max_resolution == "1080p" else 720,
        )
        response = await _submit_and_poll(cls, request)

        render = await download_url_to_image_tensor(_require_output_url(response, "render"))
        alpha_mask = None
        if (alpha_url := _alpha_url(response, mode)) is not None:
            alpha_image = bytesio_to_image_tensor(await download_url_as_bytesio(alpha_url), mode="L")
            alpha_mask = alpha_image.squeeze(-1) if alpha_image.dim() == 4 else alpha_image
        return IO.NodeOutput(render, alpha_mask)


class BeebleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            BeebleSwitchXVideoEdit,
            BeebleSwitchXImageEdit,
        ]


async def comfy_entrypoint() -> BeebleExtension:
    return BeebleExtension()
