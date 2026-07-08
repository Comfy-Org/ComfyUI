import hashlib
import logging
import math
import re
from io import BytesIO

import torch
from typing_extensions import override

from comfy.utils import common_upscale
from comfy_api.latest import IO, ComfyExtension, Input, Types
from comfy_api_nodes.apis.bytedance import (
    RECOMMENDED_PRESETS,
    RECOMMENDED_PRESETS_SEEDREAM_4,
    RECOMMENDED_PRESETS_SEEDREAM_4_0,
    RECOMMENDED_PRESETS_SEEDREAM_4_5,
    RECOMMENDED_PRESETS_SEEDREAM_5_LITE,
    SEEDANCE2_REF_VIDEO_PIXEL_LIMITS,
    VIDEO_TASKS_EXECUTION_TIME,
    GetAssetResponse,
    Image2VideoTaskCreationRequest,
    ImageTaskCreationResponse,
    Seedance2TaskCreationRequest,
    SeedanceCreateAssetRequest,
    SeedanceCreateAssetResponse,
    SeedanceCreateVisualValidateSessionResponse,
    SeedanceGetVisualValidateSessionResponse,
    SeedanceVirtualLibraryCreateAssetRequest,
    Seedream4Options,
    Seedream4TaskCreationRequest,
    TaskAudioContent,
    TaskAudioContentUrl,
    TaskCreationResponse,
    TaskImageContent,
    TaskImageContentUrl,
    TaskStatusResponse,
    TaskTextContent,
    TaskVideoContent,
    TaskVideoContentUrl,
    Text2ImageTaskCreationRequest,
    Text2VideoTaskCreationRequest,
    seedance2_price_per_1k_tokens,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    downscale_image_tensor_by_max_side,
    downscale_video_to_max_pixels,
    get_number_of_images,
    image_tensor_pair_to_batch,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    upscale_video_to_min_pixels,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_string,
    validate_video_dimensions,
    validate_video_duration,
)
from server import PromptServer

BYTEPLUS_IMAGE_ENDPOINT = "/proxy/byteplus/api/v3/images/generations"

_VERIFICATION_POLL_TIMEOUT_SEC = 120
_VERIFICATION_POLL_INTERVAL_SEC = 3

SEEDREAM_MODELS = {
    "seedream 5.0 lite": "seedream-5-0-260128",
    "seedream-4-5-251128": "seedream-4-5-251128",
    "seedream-4-0-250828": "seedream-4-0-250828",
}

SEEDREAM_PRESETS = {
    "seedream-5-0-260128": RECOMMENDED_PRESETS_SEEDREAM_5_LITE,
    "seedream-4-5-251128": RECOMMENDED_PRESETS_SEEDREAM_4_5,
    "seedream-4-0-250828": RECOMMENDED_PRESETS_SEEDREAM_4_0,
}

# Long-running tasks endpoints(e.g., video)
BYTEPLUS_TASK_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"
BYTEPLUS_TASK_STATUS_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"  # + /{task_id}
BYTEPLUS_SEEDANCE2_TASK_STATUS_ENDPOINT = "/proxy/byteplus-seedance2/api/v3/contents/generations/tasks"  # + /{task_id}

SEEDANCE_MODELS = {
    "Seedance 2.0": "dreamina-seedance-2-0-260128",
    "Seedance 2.0 Fast": "dreamina-seedance-2-0-fast-260128",
    "Seedance 2.0 Mini": "dreamina-seedance-2-0-mini",
}

DEPRECATED_MODELS = {"seedance-1-0-lite-t2v-250428", "seedance-1-0-lite-i2v-250428"}


logger = logging.getLogger(__name__)


def _validate_ref_video_pixels(video: Input.Video, model_id: str, resolution: str, index: int) -> None:
    """Validate reference video pixel count against Seedance 2.0 model limits for the selected resolution."""
    model_limits = SEEDANCE2_REF_VIDEO_PIXEL_LIMITS.get(model_id)
    if not model_limits:
        return
    limits = model_limits.get(resolution)
    if not limits:
        return
    try:
        w, h = video.get_dimensions()
    except Exception:
        return
    pixels = w * h
    min_px = limits.get("min")
    max_px = limits.get("max")
    if min_px and pixels < min_px:
        raise ValueError(
            f"Reference video {index} is too small: {w}x{h} = {pixels:,} total pixels. "
            f"Minimum for this model is {min_px:,} total pixels."
        )
    if max_px and pixels > max_px:
        raise ValueError(
            f"Reference video {index} is too large: {w}x{h} = {pixels:,} total pixels. "
            f"Maximum for this model is {max_px:,} total pixels. Try downscaling the video."
        )


def _prepare_seedance_image(image: Input.Image) -> Input.Image:
    """Auto-downscale a Seedance image input to the per-side limits, then validate it."""
    validate_image_aspect_ratio(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5
    image = downscale_image_tensor_by_max_side(image, max_side=6000)
    validate_image_dimensions(image, min_width=300, min_height=300, max_width=6000, max_height=6000)
    return image


# Supported output aspect ratios, used to pre-size FLF frames to matching pixel pair to avoid the 1080p stretch jump.
SEEDANCE2_RATIO_WH = {
    "16:9": (16, 9),
    "4:3": (4, 3),
    "1:1": (1, 1),
    "3:4": (3, 4),
    "9:16": (9, 16),
    "21:9": (21, 9),
}
SEEDANCE2_RES_SHORT_SIDE = {"480p": 480, "720p": 720, "1080p": 1080, "4k": 2160}


def _seedance2_target_dims(resolution: str, ratio: str, image: torch.Tensor) -> tuple[int, int]:
    """Exact supported output (width, height) for (resolution, ratio).

    The shorter side equals the resolution number (e.g. 1080p 16:9 -> 1920x1080). For ratio
    "adaptive" (or any unexpected value) the ratio is derived from the image's own aspect, snapped
    to the nearest supported ratio, so the output keeps the frame's orientation.
    """
    short = SEEDANCE2_RES_SHORT_SIDE[resolution]
    if ratio not in SEEDANCE2_RATIO_WH:
        aspect = image.shape[-2] / image.shape[-3]  # W / H; tensor is (B, H, W, C)
        ratio = min(SEEDANCE2_RATIO_WH, key=lambda k: abs(SEEDANCE2_RATIO_WH[k][0] / SEEDANCE2_RATIO_WH[k][1] - aspect))
    rw, rh = SEEDANCE2_RATIO_WH[ratio]
    if rw >= rh:  # landscape or square: shorter side is the height
        out_w, out_h = round(short * rw / rh), short
    else:  # portrait: shorter side is the width
        out_w, out_h = short, round(short * rh / rw)
    return out_w - out_w % 2, out_h - out_h % 2


def _resize_to_exact(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Center-crop to the target aspect and resize to exactly width x height (lanczos)."""
    samples = image.movedim(-1, 1)  # (B, H, W, C) -> (B, C, H, W)
    resized = common_upscale(samples, width, height, "lanczos", "center")
    return resized.movedim(1, -1)


async def _resolve_reference_assets(
    cls: type[IO.ComfyNode],
    asset_ids: list[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Look up each asset, validate Active status, group by asset_type.

    Returns (image_assets, video_assets, audio_assets), each mapping asset_id -> "asset://<asset_id>".
    """
    image_assets: dict[str, str] = {}
    video_assets: dict[str, str] = {}
    audio_assets: dict[str, str] = {}
    for i, raw_id in enumerate(asset_ids, 1):
        asset_id = (raw_id or "").strip()
        if not asset_id:
            continue
        result = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/seedance/assets/{asset_id}"),
            response_model=GetAssetResponse,
        )
        if result.status != "Active":
            extra = f" {result.error.code}: {result.error.message}" if result.error else ""
            raise ValueError(f"Reference asset {i} (Id={asset_id}) is not Active (Status={result.status}).{extra}")
        asset_uri = f"asset://{asset_id}"
        if result.asset_type == "Image":
            image_assets[asset_id] = asset_uri
        elif result.asset_type == "Video":
            video_assets[asset_id] = asset_uri
        elif result.asset_type == "Audio":
            audio_assets[asset_id] = asset_uri
    return image_assets, video_assets, audio_assets


_ASSET_REF_RE = re.compile(r"\basset ?(\d{1,2})\b", re.IGNORECASE)


def _build_asset_labels(
    reference_assets: dict[str, str],
    image_asset_uris: dict[str, str],
    video_asset_uris: dict[str, str],
    audio_asset_uris: dict[str, str],
    n_reference_images: int,
    n_reference_videos: int,
    n_reference_audios: int,
) -> dict[int, str]:
    """Map asset slot number (from 'asset_N' keys) to its positional label.

    Asset entries are appended to `content` after the reference_images/videos/audios,
    so their 1-indexed labels continue from the count of existing same-type refs:
    one reference_images entry + one Image-type asset -> asset labelled "Image 2".
    """
    image_n = n_reference_images
    video_n = n_reference_videos
    audio_n = n_reference_audios
    labels: dict[int, str] = {}
    for slot_key, raw_id in reference_assets.items():
        asset_id = (raw_id or "").strip()
        if not asset_id:
            continue
        try:
            slot_num = int(slot_key.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if asset_id in image_asset_uris:
            image_n += 1
            labels[slot_num] = f"Image {image_n}"
        elif asset_id in video_asset_uris:
            video_n += 1
            labels[slot_num] = f"Video {video_n}"
        elif asset_id in audio_asset_uris:
            audio_n += 1
            labels[slot_num] = f"Audio {audio_n}"
    return labels


def _rewrite_asset_refs(prompt: str, labels: dict[int, str]) -> str:
    """Case-insensitively replace 'assetNN' (1-2 digit) tokens with their labels."""
    if not labels:
        return prompt

    def _sub(m: "re.Match[str]") -> str:
        return labels.get(int(m.group(1)), m.group(0))

    return _ASSET_REF_RE.sub(_sub, prompt)


async def _obtain_group_id_via_h5_auth(cls: type[IO.ComfyNode]) -> str:
    session = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/seedance/visual-validate/sessions", method="POST"),
        response_model=SeedanceCreateVisualValidateSessionResponse,
    )
    logger.warning("Seedance authentication required. Open link: %s", session.h5_link)

    h5_text = f"Open this link in your browser and complete face verification:\n\n{session.h5_link}"

    result = await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/seedance/visual-validate/sessions/{session.session_id}"),
        response_model=SeedanceGetVisualValidateSessionResponse,
        status_extractor=lambda r: r.status,
        completed_statuses=["completed"],
        failed_statuses=["failed"],
        poll_interval=_VERIFICATION_POLL_INTERVAL_SEC,
        max_poll_attempts=(_VERIFICATION_POLL_TIMEOUT_SEC // _VERIFICATION_POLL_INTERVAL_SEC) - 1,
        estimated_duration=_VERIFICATION_POLL_TIMEOUT_SEC - 1,
        extra_text=h5_text,
    )

    if not result.group_id:
        raise RuntimeError(f"Seedance session {session.session_id} completed without a group_id")

    logger.warning("Seedance authentication complete. New GroupId: %s", result.group_id)
    PromptServer.instance.send_progress_text(
        f"Authentication complete. New GroupId: {result.group_id}", cls.hidden.unique_id
    )
    return result.group_id


async def _resolve_group_id(cls: type[IO.ComfyNode], group_id: str) -> str:
    if group_id and group_id.strip():
        return group_id.strip()
    return await _obtain_group_id_via_h5_auth(cls)


async def _create_seedance_asset(
    cls: type[IO.ComfyNode],
    *,
    group_id: str,
    url: str,
    name: str,
    asset_type: str,
) -> str:
    req = SeedanceCreateAssetRequest(
        group_id=group_id,
        url=url,
        asset_type=asset_type,
        name=name or None,
    )
    result = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/seedance/assets", method="POST"),
        response_model=SeedanceCreateAssetResponse,
        data=req,
    )
    return result.asset_id


async def _wait_for_asset_active(cls: type[IO.ComfyNode], asset_id: str, group_id: str) -> GetAssetResponse:
    """Poll the newly created asset until its status becomes Active."""
    return await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/seedance/assets/{asset_id}"),
        response_model=GetAssetResponse,
        status_extractor=lambda r: r.status,
        completed_statuses=["Active"],
        failed_statuses=["Failed"],
        poll_interval=5,
        max_poll_attempts=1200,
        extra_text=f"Waiting for asset pre-processing...\n\nasset_id: {asset_id}\n\ngroup_id: {group_id}",
    )


async def _seedance_virtual_library_upload_image_asset(
    cls: type[IO.ComfyNode],
    image: torch.Tensor,
    *,
    wait_label: str = "Uploading image",
) -> str:
    """Upload an image into the caller's per-customer Seedance virtual library."""
    public_url = await upload_image_to_comfyapi(cls, image, wait_label=wait_label)
    normalized = image.detach().cpu().contiguous().to(torch.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(normalized.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(normalized.numpy().tobytes())
    image_hash = digest.hexdigest()
    create_resp = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/seedance/virtual-library/assets", method="POST"),
        response_model=SeedanceCreateAssetResponse,
        data=SeedanceVirtualLibraryCreateAssetRequest(url=public_url, hash=image_hash),
    )
    await _wait_for_asset_active(cls, create_resp.asset_id, group_id="virtual-library")
    return f"asset://{create_resp.asset_id}"


async def _seedance_virtual_library_upload_video_asset(
    cls: type[IO.ComfyNode],
    video: Input.Video,
    *,
    wait_label: str = "Uploading video",
) -> str:
    buf = BytesIO()
    video.save_to(buf, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
    video_hash = hashlib.sha256(buf.getbuffer()).hexdigest()
    public_url = await upload_video_to_comfyapi(cls, video, wait_label=wait_label)
    create_resp = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/seedance/virtual-library/assets", method="POST"),
        response_model=SeedanceCreateAssetResponse,
        data=SeedanceVirtualLibraryCreateAssetRequest(url=public_url, hash=video_hash, asset_type="Video"),
    )
    await _wait_for_asset_active(cls, create_resp.asset_id, group_id="virtual-library")
    return f"asset://{create_resp.asset_id}"


def _seedance2_price_extractor(model_id: str, has_video_input: bool, resolution: str):
    """Returns a price_extractor closure for Seedance 2.0 poll_op."""
    rate = seedance2_price_per_1k_tokens(model_id, has_video_input, resolution)
    if rate is None:
        return None

    def extractor(response: TaskStatusResponse) -> float | None:
        if response.usage is None:
            return None
        return response.usage.total_tokens * 1.43 * rate / 1_000.0

    return extractor


def get_image_url_from_response(response: ImageTaskCreationResponse) -> str:
    if response.error:
        error_msg = f"ByteDance request failed. Code: {response.error['code']}, message: {response.error['message']}"
        logging.info(error_msg)
        raise RuntimeError(error_msg)
    logging.info("ByteDance task succeeded, image URL: %s", response.data[0]["url"])
    return response.data[0]["url"]


class ByteDanceImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceImageNode",
            display_name="ByteDance Image",
            category="partner/image/ByteDance",
            description="Generate images using ByteDance models via api based on prompt",
            inputs=[
                IO.Combo.Input("model", options=["seedream-3-0-t2i-250415"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the image",
                ),
                IO.Combo.Input(
                    "size_preset",
                    options=[label for label, _, _ in RECOMMENDED_PRESETS],
                    tooltip="Pick a recommended size. Select Custom to use the width and height below",
                ),
                IO.Int.Input(
                    "width",
                    default=1024,
                    min=512,
                    max=2048,
                    step=64,
                    tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
                ),
                IO.Int.Input(
                    "height",
                    default=1024,
                    min=512,
                    max=2048,
                    step=64,
                    tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation",
                    optional=True,
                ),
                IO.Float.Input(
                    "guidance_scale",
                    default=2.5,
                    min=1.0,
                    max=10.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the image',
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.03}""",
            ),
            is_deprecated=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        size_preset: str,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        w = h = None
        for label, tw, th in RECOMMENDED_PRESETS:
            if label == size_preset:
                w, h = tw, th
                break

        if w is None or h is None:
            w, h = width, height
            if not (512 <= w <= 2048) or not (512 <= h <= 2048):
                raise ValueError(
                    f"Custom size out of range: {w}x{h}. " "Both width and height must be between 512 and 2048 pixels."
                )

        payload = Text2ImageTaskCreationRequest(
            model=model,
            prompt=prompt,
            size=f"{w}x{h}",
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
        )
        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_IMAGE_ENDPOINT, method="POST"),
            data=payload,
            response_model=ImageTaskCreationResponse,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))


class ByteDanceSeedreamNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceSeedreamNode",
            display_name="ByteDance Seedream 4.5 & 5.0",
            category="partner/image/ByteDance",
            description="Unified text-to-image generation and precise single-sentence editing at up to 4K resolution.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=list(SEEDREAM_MODELS.keys()),
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for creating or editing an image.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Input image(s) for image-to-image generation. "
                    "Reference image(s) for single or multi-reference generation.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size_preset",
                    options=[label for label, _, _ in RECOMMENDED_PRESETS_SEEDREAM_4],
                    tooltip="Pick a recommended size. Select Custom to use the width and height below.",
                ),
                IO.Int.Input(
                    "width",
                    default=2048,
                    min=1024,
                    max=6240,
                    step=2,
                    tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
                    optional=True,
                ),
                IO.Int.Input(
                    "height",
                    default=2048,
                    min=1024,
                    max=4992,
                    step=2,
                    tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
                    optional=True,
                ),
                IO.Combo.Input(
                    "sequential_image_generation",
                    options=["disabled", "auto"],
                    tooltip="Group image generation mode. "
                    "'disabled' generates a single image. "
                    "'auto' lets the model decide whether to generate multiple related images "
                    "(e.g., story scenes, character variations).",
                    optional=True,
                ),
                IO.Int.Input(
                    "max_images",
                    default=1,
                    min=1,
                    max=15,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Maximum number of images to generate when sequential_image_generation='auto'. "
                    "Total images (input + generated) cannot exceed 15.",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the image.',
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "fail_on_partial",
                    default=True,
                    tooltip="If enabled, abort execution if any requested images are missing or return an error.",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model"]),
                expr="""
                (
                  $price := $contains(widgets.model, "5.0 lite") ? 0.035 :
                            $contains(widgets.model, "4-5") ? 0.04 : 0.03;
                  {
                    "type":"usd",
                    "usd": $price,
                    "format": { "suffix":" x images/Run", "approximate": true }
                  }
                )
                """,
            ),
            is_deprecated=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        image: Input.Image | None = None,
        size_preset: str = RECOMMENDED_PRESETS_SEEDREAM_4[0][0],
        width: int = 2048,
        height: int = 2048,
        sequential_image_generation: str = "disabled",
        max_images: int = 1,
        seed: int = 0,
        watermark: bool = False,
        fail_on_partial: bool = True,
    ) -> IO.NodeOutput:
        model = SEEDREAM_MODELS[model]
        validate_string(prompt, strip_whitespace=True, min_length=1)
        w = h = None
        for label, tw, th in RECOMMENDED_PRESETS_SEEDREAM_4:
            if label == size_preset:
                w, h = tw, th
                break

        if w is None or h is None:
            w, h = width, height

        out_num_pixels = w * h
        mp_provided = out_num_pixels / 1_000_000.0
        if ("seedream-4-5" in model or "seedream-5-0" in model) and out_num_pixels < 3686400:
            raise ValueError(
                f"Minimum image resolution for the selected model is 3.68MP, " f"but {mp_provided:.2f}MP provided."
            )
        if "seedream-4-0" in model and out_num_pixels < 921600:
            raise ValueError(
                f"Minimum image resolution that the selected model can generate is 0.92MP, "
                f"but {mp_provided:.2f}MP provided."
            )
        max_pixels = 10_404_496 if "seedream-5-0" in model else 16_777_216
        if out_num_pixels > max_pixels:
            raise ValueError(
                f"Maximum image resolution for the selected model is {max_pixels / 1_000_000:.2f}MP, "
                f"but {mp_provided:.2f}MP provided."
            )
        n_input_images = get_number_of_images(image) if image is not None else 0
        max_num_of_images = 14 if model == "seedream-5-0-260128" else 10
        if n_input_images > max_num_of_images:
            raise ValueError(
                f"Maximum of {max_num_of_images} reference images are supported, but {n_input_images} received."
            )
        if sequential_image_generation == "auto" and n_input_images + max_images > 15:
            raise ValueError(
                "The maximum number of generated images plus the number of reference images cannot exceed 15."
            )
        reference_images_urls = []
        if n_input_images:
            for i in image:
                validate_image_aspect_ratio(i, (1, 3), (3, 1))
            reference_images_urls = await upload_images_to_comfyapi(
                cls,
                image,
                max_images=n_input_images,
                mime_type="image/png",
            )
        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_IMAGE_ENDPOINT, method="POST"),
            response_model=ImageTaskCreationResponse,
            data=Seedream4TaskCreationRequest(
                model=model,
                prompt=prompt,
                image=reference_images_urls,
                size=f"{w}x{h}",
                seed=seed,
                sequential_image_generation=sequential_image_generation,
                sequential_image_generation_options=Seedream4Options(max_images=max_images),
                watermark=watermark,
                output_format="png" if model == "seedream-5-0-260128" else None,
            ),
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))
        urls = [str(d["url"]) for d in response.data if isinstance(d, dict) and "url" in d]
        if fail_on_partial and len(urls) < len(response.data):
            raise RuntimeError(f"Only {len(urls)} of {len(response.data)} images were generated before error.")
        return IO.NodeOutput(torch.cat([await download_url_to_image_tensor(i) for i in urls]))


def _seedream_model_inputs(*, max_ref_images: int, presets: list):
    return [
        IO.Combo.Input(
            "size_preset",
            options=[label for label, _, _ in presets],
            tooltip="Pick a recommended size. Select Custom to use the width and height below.",
        ),
        IO.Int.Input(
            "width",
            default=2048,
            min=1024,
            max=6240,
            step=2,
            tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
        ),
        IO.Int.Input(
            "height",
            default=2048,
            min=1024,
            max=4992,
            step=2,
            tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
        ),
        IO.Int.Input(
            "max_images",
            default=1,
            min=1,
            max=max_ref_images,
            step=1,
            display_mode=IO.NumberDisplay.number,
            tooltip="Maximum number of images to generate. With 1, exactly one image is produced. "
            "With >1, the model generates between 1 and max_images related images "
            "(e.g., story scenes, character variations). "
            "Total images (input + generated) cannot exceed 15.",
        ),
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, max_ref_images + 1)],
                min=0,
            ),
            tooltip=f"Optional reference image(s) for image-to-image or multi-reference generation. "
            f"Up to {max_ref_images} images.",
        ),
        IO.Boolean.Input(
            "fail_on_partial",
            default=False,
            tooltip="If enabled, abort execution if any requested images are missing or return an error.",
            advanced=True,
        ),
    ]


class ByteDanceSeedreamNodeV2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceSeedreamNodeV2",
            display_name="ByteDance Seedream 4.5 & 5.0",
            category="partner/image/ByteDance",
            description="Unified text-to-image generation and precise single-sentence editing at up to 4K resolution.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for creating or editing an image.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "seedream 5.0 lite",
                            _seedream_model_inputs(max_ref_images=14, presets=RECOMMENDED_PRESETS_SEEDREAM_5_LITE),
                        ),
                        IO.DynamicCombo.Option(
                            "seedream-4-5-251128",
                            _seedream_model_inputs(max_ref_images=10, presets=RECOMMENDED_PRESETS_SEEDREAM_4_5),
                        ),
                        IO.DynamicCombo.Option(
                            "seedream-4-0-250828",
                            _seedream_model_inputs(max_ref_images=10, presets=RECOMMENDED_PRESETS_SEEDREAM_4_0),
                        ),
                    ],
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the image.',
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model"]),
                expr="""
                (
                  $price := $contains(widgets.model, "5.0 lite") ? 0.035 :
                            $contains(widgets.model, "4-5") ? 0.04 : 0.03;
                  {
                    "type":"usd",
                    "usd": $price,
                    "format": { "suffix":" x images/Run", "approximate": true }
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int = 0,
        watermark: bool = False,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        model_id = SEEDREAM_MODELS[model["model"]]
        presets = SEEDREAM_PRESETS[model_id]

        size_preset = model.get("size_preset", presets[0][0])
        width = model.get("width", 2048)
        height = model.get("height", 2048)
        max_images = model.get("max_images", 1)
        sequential_image_generation = "disabled" if max_images == 1 else "auto"
        images_dict = model.get("images") or {}
        fail_on_partial = model.get("fail_on_partial", False)

        w = h = None
        for label, tw, th in presets:
            if label == size_preset:
                w, h = tw, th
                break
        if w is None or h is None:
            w, h = width, height

        out_num_pixels = w * h
        mp_provided = out_num_pixels / 1_000_000.0
        if ("seedream-4-5" in model_id or "seedream-5-0" in model_id) and out_num_pixels < 3686400:
            raise ValueError(
                f"Minimum image resolution for the selected model is 3.68MP, but {mp_provided:.2f}MP provided."
            )
        if "seedream-4-0" in model_id and out_num_pixels < 921600:
            raise ValueError(
                f"Minimum image resolution that the selected model can generate is 0.92MP, "
                f"but {mp_provided:.2f}MP provided."
            )
        if out_num_pixels > 16_777_216:
            raise ValueError(
                f"Maximum image resolution for the selected model is 16.78MP, but {mp_provided:.2f}MP provided."
            )

        image_tensors: list[Input.Image] = [t for t in images_dict.values() if t is not None]
        n_input_images = sum(get_number_of_images(t) for t in image_tensors)
        max_num_of_images = 14 if model_id == "seedream-5-0-260128" else 10
        if n_input_images > max_num_of_images:
            raise ValueError(
                f"Maximum of {max_num_of_images} reference images are supported, but {n_input_images} received."
            )
        if sequential_image_generation == "auto" and n_input_images + max_images > 15:
            raise ValueError(
                "The maximum number of generated images plus the number of reference images cannot exceed 15."
            )

        reference_images_urls: list[str] = []
        if image_tensors:
            for tensor in image_tensors:
                validate_image_aspect_ratio(tensor, (1, 3), (3, 1))
            reference_images_urls = await upload_images_to_comfyapi(
                cls,
                image_tensors,
                max_images=n_input_images,
                mime_type="image/png",
                wait_label="Uploading reference images",
            )

        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_IMAGE_ENDPOINT, method="POST"),
            response_model=ImageTaskCreationResponse,
            data=Seedream4TaskCreationRequest(
                model=model_id,
                prompt=prompt,
                image=reference_images_urls,
                size=f"{w}x{h}",
                seed=seed,
                sequential_image_generation=sequential_image_generation,
                sequential_image_generation_options=Seedream4Options(max_images=max_images),
                watermark=watermark,
            ),
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))
        urls = [str(d["url"]) for d in response.data if isinstance(d, dict) and "url" in d]
        if fail_on_partial and len(urls) < len(response.data):
            raise RuntimeError(f"Only {len(urls)} of {len(response.data)} images were generated before error.")
        return IO.NodeOutput(torch.cat([await download_url_to_image_tensor(i) for i in urls]))


class ByteDanceTextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceTextToVideoNode",
            display_name="ByteDance Text to Video",
            category="partner/video/ByteDance",
            description="Generate video using ByteDance models via api based on prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=[
                        "seedance-1-5-pro-251215",
                        "seedance-1-0-pro-250528",
                        "seedance-1-0-lite-t2v-250428",
                        "seedance-1-0-pro-fast-251015",
                    ],
                    default="seedance-1-0-pro-fast-251015",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                    "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    tooltip="This parameter is ignored for any model except seedance-1-5-pro.",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE_VIDEO,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
        generate_audio: bool = False,
    ) -> IO.NodeOutput:
        if model == "seedance-1-5-pro-251215" and duration < 4:
            raise ValueError("Minimum supported duration for Seedance 1.5 Pro is 4 seconds.")
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])

        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--camerafixed {str(camera_fixed).lower()} "
            f"--watermark {str(watermark).lower()}"
        )
        return await process_video_task(
            cls,
            payload=Text2VideoTaskCreationRequest(
                model=model,
                content=[TaskTextContent(text=prompt)],
                generate_audio=generate_audio if model == "seedance-1-5-pro-251215" else None,
            ),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


class ByteDanceImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceImageToVideoNode",
            display_name="ByteDance Image to Video",
            category="partner/video/ByteDance",
            description="Generate video using ByteDance models via api based on image and prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=[
                        "seedance-1-5-pro-251215",
                        "seedance-1-0-pro-250528",
                        "seedance-1-0-lite-i2v-250428",
                        "seedance-1-0-pro-fast-251015",
                    ],
                    default="seedance-1-0-pro-fast-251015",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="First frame to be used for the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                    "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    tooltip="This parameter is ignored for any model except seedance-1-5-pro.",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE_VIDEO,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        image: Input.Image,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
        generate_audio: bool = False,
    ) -> IO.NodeOutput:
        if model == "seedance-1-5-pro-251215" and duration < 4:
            raise ValueError("Minimum supported duration for Seedance 1.5 Pro is 4 seconds.")
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])
        validate_image_dimensions(image, min_width=300, min_height=300, max_width=6000, max_height=6000)
        validate_image_aspect_ratio(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        image_url = (await upload_images_to_comfyapi(cls, image, max_images=1))[0]
        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--camerafixed {str(camera_fixed).lower()} "
            f"--watermark {str(watermark).lower()}"
        )

        return await process_video_task(
            cls,
            payload=Image2VideoTaskCreationRequest(
                model=model,
                content=[TaskTextContent(text=prompt), TaskImageContent(image_url=TaskImageContentUrl(url=image_url))],
                generate_audio=generate_audio if model == "seedance-1-5-pro-251215" else None,
            ),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


class ByteDanceFirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceFirstLastFrameNode",
            display_name="ByteDance First-Last-Frame to Video",
            category="partner/video/ByteDance",
            description="Generate video using prompt and first and last frames.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["seedance-1-5-pro-251215", "seedance-1-0-pro-250528", "seedance-1-0-lite-i2v-250428"],
                    default="seedance-1-0-lite-i2v-250428",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="First frame to be used for the video.",
                ),
                IO.Image.Input(
                    "last_frame",
                    tooltip="Last frame to be used for the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                    "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    tooltip="This parameter is ignored for any model except seedance-1-5-pro.",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE_VIDEO,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        first_frame: Input.Image,
        last_frame: Input.Image,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
        generate_audio: bool = False,
    ) -> IO.NodeOutput:
        if model == "seedance-1-5-pro-251215" and duration < 4:
            raise ValueError("Minimum supported duration for Seedance 1.5 Pro is 4 seconds.")
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])
        for i in (first_frame, last_frame):
            validate_image_dimensions(i, min_width=300, min_height=300, max_width=6000, max_height=6000)
            validate_image_aspect_ratio(i, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        download_urls = await upload_images_to_comfyapi(
            cls,
            image_tensor_pair_to_batch(first_frame, last_frame),
            max_images=2,
            mime_type="image/png",
        )

        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--camerafixed {str(camera_fixed).lower()} "
            f"--watermark {str(watermark).lower()}"
        )

        return await process_video_task(
            cls,
            payload=Image2VideoTaskCreationRequest(
                model=model,
                content=[
                    TaskTextContent(text=prompt),
                    TaskImageContent(image_url=TaskImageContentUrl(url=str(download_urls[0])), role="first_frame"),
                    TaskImageContent(image_url=TaskImageContentUrl(url=str(download_urls[1])), role="last_frame"),
                ],
                generate_audio=generate_audio if model == "seedance-1-5-pro-251215" else None,
            ),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


class ByteDanceImageReferenceNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceImageReferenceNode",
            display_name="ByteDance Reference Images to Video",
            category="partner/video/ByteDance",
            description="Generate video using prompt and reference images.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["seedance-1-0-pro-250528", "seedance-1-0-lite-i2v-250428"],
                    default="seedance-1-0-lite-i2v-250428",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "images",
                    tooltip="One to four images.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "duration", "resolution"]),
                expr="""
                (
                  $priceByModel := {
                    "seedance-1-0-pro": {
                      "480p":[0.23,0.24],
                      "720p":[0.51,0.56]
                    },
                    "seedance-1-0-lite": {
                      "480p":[0.17,0.18],
                      "720p":[0.37,0.41]
                    }
                  };
                  $model := widgets.model;
                  $modelKey :=
                    $contains($model, "seedance-1-0-pro")  ? "seedance-1-0-pro" :
                    "seedance-1-0-lite";
                  $resolution := widgets.resolution;
                  $resKey :=
                    $contains($resolution, "720") ? "720p" :
                    "480p";
                  $modelPrices := $lookup($priceByModel, $modelKey);
                  $baseRange := $lookup($modelPrices, $resKey);
                  $min10s := $baseRange[0];
                  $max10s := $baseRange[1];
                  $scale := widgets.duration / 10;
                  $minCost := $min10s * $scale;
                  $maxCost := $max10s * $scale;
                  ($minCost = $maxCost)
                    ? {"type":"usd","usd": $minCost}
                    : {"type":"range_usd","min_usd": $minCost, "max_usd": $maxCost}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        images: Input.Image,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "watermark"])
        for image in images:
            validate_image_dimensions(image, min_width=300, min_height=300, max_width=6000, max_height=6000)
            validate_image_aspect_ratio(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        image_urls = await upload_images_to_comfyapi(cls, images, max_images=4, mime_type="image/png")
        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--watermark {str(watermark).lower()}"
        )
        x = [
            TaskTextContent(text=prompt),
            *[TaskImageContent(image_url=TaskImageContentUrl(url=str(i)), role="reference_image") for i in image_urls],
        ]
        return await process_video_task(
            cls,
            payload=Image2VideoTaskCreationRequest(model=model, content=x, generate_audio=None),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


def raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    for i in text_params:
        if f"--{i} " in prompt:
            raise ValueError(
                f"--{i} is not allowed in the prompt, use the appropriated widget input to change this value."
            )


PRICE_BADGE_VIDEO = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["model", "duration", "resolution", "generate_audio"]),
    expr="""
    (
      $priceByModel := {
        "seedance-1-5-pro": {
          "480p":[0.12,0.12],
          "720p":[0.26,0.26],
          "1080p":[0.58,0.59]
        },
        "seedance-1-0-pro": {
          "480p":[0.23,0.24],
          "720p":[0.51,0.56],
          "1080p":[1.18,1.22]
        },
        "seedance-1-0-pro-fast": {
          "480p":[0.09,0.1],
          "720p":[0.21,0.23],
          "1080p":[0.47,0.49]
        },
        "seedance-1-0-lite": {
          "480p":[0.17,0.18],
          "720p":[0.37,0.41],
          "1080p":[0.85,0.88]
        }
      };
      $model := widgets.model;
      $modelKey :=
        $contains($model, "seedance-1-5-pro")      ? "seedance-1-5-pro" :
        $contains($model, "seedance-1-0-pro-fast") ? "seedance-1-0-pro-fast" :
        $contains($model, "seedance-1-0-pro")      ? "seedance-1-0-pro" :
        "seedance-1-0-lite";
      $resolution := widgets.resolution;
      $resKey :=
        $contains($resolution, "1080") ? "1080p" :
        $contains($resolution, "720")  ? "720p" :
        "480p";
      $modelPrices := $lookup($priceByModel, $modelKey);
      $baseRange := $lookup($modelPrices, $resKey);
      $min10s := $baseRange[0];
      $max10s := $baseRange[1];
      $scale := widgets.duration / 10;
      $audioMultiplier := ($modelKey = "seedance-1-5-pro" and widgets.generate_audio) ? 2 : 1;
      $minCost := $min10s * $scale * $audioMultiplier;
      $maxCost := $max10s * $scale * $audioMultiplier;
      ($minCost = $maxCost)
        ? {"type":"usd","usd": $minCost, "format": { "approximate": true }}
        : {"type":"range_usd","min_usd": $minCost, "max_usd": $maxCost, "format": { "approximate": true }}
    )
    """,
)


def _seedance2_text_inputs(resolutions: list[str], default_ratio: str = "16:9"):
    return [
        IO.String.Input(
            "prompt",
            multiline=True,
            default="",
            tooltip="Text prompt for video generation.",
        ),
        IO.Combo.Input(
            "resolution",
            options=resolutions,
            tooltip="Resolution of the output video.",
        ),
        IO.Combo.Input(
            "ratio",
            options=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"],
            default=default_ratio,
            tooltip="Aspect ratio of the output video.",
        ),
        IO.Int.Input(
            "duration",
            default=7,
            min=4,
            max=15,
            step=1,
            tooltip="Duration of the output video in seconds (4-15).",
            display_mode=IO.NumberDisplay.slider,
        ),
        IO.Boolean.Input(
            "generate_audio",
            default=True,
            tooltip="Enable audio generation for the output video.",
        ),
    ]


class ByteDance2TextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDance2TextToVideoNode",
            display_name="ByteDance Seedance 2.0 Text to Video",
            category="partner/video/ByteDance",
            description="Generate video using Seedance 2.0 models based on a text prompt.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("Seedance 2.0", _seedance2_text_inputs(["480p", "720p", "1080p", "4k"])),
                        IO.DynamicCombo.Option("Seedance 2.0 Fast", _seedance2_text_inputs(["480p", "720p"])),
                        IO.DynamicCombo.Option("Seedance 2.0 Mini", _seedance2_text_inputs(["480p", "720p"])),
                    ],
                    tooltip="Seedance 2.0 for maximum quality; Fast for speed optimization; "
                    "Mini for the fastest, lowest-cost generation.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add a watermark to the video.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.resolution", "model.duration"]),
                expr="""
                (
                  $rate480 := 10044;
                  $rate720 := 21600;
                  $rate1080 := 48800;
                  $rate4k := 195200;
                  $m := widgets.model;
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $pricePer1K := $res = "4k"    ? 0.00572 :
                                 $res = "1080p" ? 0.011011 :
                                 $contains($m, "mini") ? 0.005005 :
                                 $contains($m, "fast") ? 0.008008 : 0.01001;
                  $rate := $res = "4k"    ? $rate4k :
                           $res = "1080p" ? $rate1080 :
                           $res = "720p"  ? $rate720 :
                                            $rate480;
                  $cost := $dur * $rate * $pricePer1K / 1000;
                  {"type": "usd", "usd": $cost, "format": {"approximate": true}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        seed: int,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=True, min_length=1)
        model_id = SEEDANCE_MODELS[model["model"]]
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_TASK_ENDPOINT, method="POST"),
            data=Seedance2TaskCreationRequest(
                model=model_id,
                content=[TaskTextContent(text=model["prompt"])],
                generate_audio=model["generate_audio"],
                resolution=model["resolution"],
                ratio=model["ratio"],
                duration=model["duration"],
                seed=seed,
                watermark=watermark,
            ),
            response_model=TaskCreationResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"{BYTEPLUS_SEEDANCE2_TASK_STATUS_ENDPOINT}/{initial_response.id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: r.status,
            price_extractor=_seedance2_price_extractor(model_id, has_video_input=False, resolution=model["resolution"]),
            poll_interval=9,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.content.video_url))


class ByteDance2FirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDance2FirstLastFrameNode",
            display_name="ByteDance Seedance 2.0 First-Last-Frame to Video",
            category="partner/video/ByteDance",
            description="Generate video using Seedance 2.0 from a first frame image and optional last frame image.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "Seedance 2.0",
                            _seedance2_text_inputs(["480p", "720p", "1080p", "4k"], default_ratio="adaptive"),
                        ),
                        IO.DynamicCombo.Option(
                            "Seedance 2.0 Fast",
                            _seedance2_text_inputs(["480p", "720p"], default_ratio="adaptive"),
                        ),
                        IO.DynamicCombo.Option(
                            "Seedance 2.0 Mini",
                            _seedance2_text_inputs(["480p", "720p"], default_ratio="adaptive"),
                        ),
                    ],
                    tooltip="Seedance 2.0 for maximum quality; Fast for speed optimization; "
                    "Mini for the fastest, lowest-cost generation.",
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="First frame image for the video.",
                    optional=True,
                ),
                IO.Image.Input(
                    "last_frame",
                    tooltip="Last frame image for the video.",
                    optional=True,
                ),
                IO.String.Input(
                    "first_frame_asset_id",
                    default="",
                    tooltip="Seedance asset_id to use as the first frame. "
                    "Mutually exclusive with the first_frame image input.",
                    optional=True,
                ),
                IO.String.Input(
                    "last_frame_asset_id",
                    default="",
                    tooltip="Seedance asset_id to use as the last frame. "
                    "Mutually exclusive with the last_frame image input.",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add a watermark to the video.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.resolution", "model.duration"]),
                expr="""
                (
                  $rate480 := 10044;
                  $rate720 := 21600;
                  $rate1080 := 48800;
                  $rate4k := 195200;
                  $m := widgets.model;
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $pricePer1K := $res = "4k"    ? 0.00572 :
                                 $res = "1080p" ? 0.011011 :
                                 $contains($m, "mini") ? 0.005005 :
                                 $contains($m, "fast") ? 0.008008 : 0.01001;
                  $rate := $res = "4k"    ? $rate4k :
                           $res = "1080p" ? $rate1080 :
                           $res = "720p"  ? $rate720 :
                                            $rate480;
                  $cost := $dur * $rate * $pricePer1K / 1000;
                  {"type": "usd", "usd": $cost, "format": {"approximate": true}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        seed: int,
        watermark: bool,
        first_frame: Input.Image | None = None,
        last_frame: Input.Image | None = None,
        first_frame_asset_id: str = "",
        last_frame_asset_id: str = "",
    ) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=True, min_length=1)
        model_id = SEEDANCE_MODELS[model["model"]]

        first_frame_asset_id = first_frame_asset_id.strip()
        last_frame_asset_id = last_frame_asset_id.strip()

        if first_frame is not None and first_frame_asset_id:
            raise ValueError("Provide only one of first_frame or first_frame_asset_id, not both.")
        if first_frame is None and not first_frame_asset_id:
            raise ValueError("Either first_frame or first_frame_asset_id is required.")
        if last_frame is not None and last_frame_asset_id:
            raise ValueError("Provide only one of last_frame or last_frame_asset_id, not both.")

        request_ratio = model["ratio"]
        if first_frame_asset_id or last_frame_asset_id:
            if first_frame is not None:
                first_frame = _prepare_seedance_image(first_frame)
            if last_frame is not None:
                last_frame = _prepare_seedance_image(last_frame)
        else:
            # The 1080p FLF stretch fix (pre-size frames to a supported pixel pair + submit ratio="adaptive")
            # only applies to local image inputs we can resize.
            request_ratio = "adaptive"
            target_dims: tuple[int, int] | None = None
            if first_frame is not None:
                validate_image_aspect_ratio(first_frame, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5
                validate_image_dimensions(first_frame, min_width=300, min_height=300)
                target_dims = _seedance2_target_dims(model["resolution"], model["ratio"], first_frame)
                first_frame = _resize_to_exact(first_frame, *target_dims)
            if last_frame is not None:
                validate_image_aspect_ratio(last_frame, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5
                validate_image_dimensions(last_frame, min_width=300, min_height=300)
                if target_dims is None:
                    target_dims = _seedance2_target_dims(model["resolution"], model["ratio"], last_frame)
                last_frame = _resize_to_exact(last_frame, *target_dims)

        asset_ids_to_resolve = [a for a in (first_frame_asset_id, last_frame_asset_id) if a]
        image_assets: dict[str, str] = {}
        if asset_ids_to_resolve:
            image_assets, _, _ = await _resolve_reference_assets(cls, asset_ids_to_resolve)
            for aid in asset_ids_to_resolve:
                if aid not in image_assets:
                    raise ValueError(f"Asset {aid} is not an Image asset.")

        if first_frame_asset_id:
            first_frame_url = image_assets[first_frame_asset_id]
        else:
            first_frame_url = await _seedance_virtual_library_upload_image_asset(
                cls, first_frame, wait_label="Uploading first frame."
            )

        content: list[TaskTextContent | TaskImageContent] = [
            TaskTextContent(text=model["prompt"]),
            TaskImageContent(
                image_url=TaskImageContentUrl(url=first_frame_url),
                role="first_frame",
            ),
        ]
        if last_frame_asset_id:
            content.append(
                TaskImageContent(
                    image_url=TaskImageContentUrl(url=image_assets[last_frame_asset_id]),
                    role="last_frame",
                ),
            )
        elif last_frame is not None:
            content.append(
                TaskImageContent(
                    image_url=TaskImageContentUrl(
                        url=await _seedance_virtual_library_upload_image_asset(
                            cls, last_frame, wait_label="Uploading last frame."
                        )
                    ),
                    role="last_frame",
                ),
            )

        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_TASK_ENDPOINT, method="POST"),
            data=Seedance2TaskCreationRequest(
                model=model_id,
                content=content,
                generate_audio=model["generate_audio"],
                resolution=model["resolution"],
                ratio=request_ratio,
                duration=model["duration"],
                seed=seed,
                watermark=watermark,
            ),
            response_model=TaskCreationResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"{BYTEPLUS_SEEDANCE2_TASK_STATUS_ENDPOINT}/{initial_response.id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: r.status,
            price_extractor=_seedance2_price_extractor(model_id, has_video_input=False, resolution=model["resolution"]),
            poll_interval=9,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.content.video_url))


def _seedance2_reference_inputs(resolutions: list[str], default_ratio: str = "16:9"):
    return [
        *_seedance2_text_inputs(resolutions, default_ratio=default_ratio),
        IO.Autogrow.Input(
            "reference_images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("reference_image"),
                names=[
                    "image_1",
                    "image_2",
                    "image_3",
                    "image_4",
                    "image_5",
                    "image_6",
                    "image_7",
                    "image_8",
                    "image_9",
                ],
                min=0,
            ),
        ),
        IO.Autogrow.Input(
            "reference_videos",
            template=IO.Autogrow.TemplateNames(
                IO.Video.Input("reference_video"),
                names=["video_1", "video_2", "video_3"],
                min=0,
            ),
        ),
        IO.Autogrow.Input(
            "reference_audios",
            template=IO.Autogrow.TemplateNames(
                IO.Audio.Input("reference_audio"),
                names=["audio_1", "audio_2", "audio_3"],
                min=0,
            ),
        ),
        IO.Boolean.Input(
            "auto_downscale",
            default=True,
            optional=True,
            tooltip="Automatically downscale reference videos that exceed the model's pixel budget "
            "for the selected resolution. Aspect ratio is preserved; videos already within limits are untouched.",
        ),
        IO.Boolean.Input(
            "auto_upscale",
            default=False,
            advanced=True,
            optional=True,
            tooltip="Automatically upscale reference videos that are below the model's minimum pixel count "
            "for the selected resolution. Aspect ratio is preserved; videos already meeting the minimum are "
            "untouched. Note: upscaling a low-resolution source does not add real detail and may produce "
            "lower-quality generations.",
        ),
        IO.Autogrow.Input(
            "reference_assets",
            template=IO.Autogrow.TemplateNames(
                IO.String.Input("reference_asset"),
                names=[
                    "asset_1",
                    "asset_2",
                    "asset_3",
                    "asset_4",
                    "asset_5",
                    "asset_6",
                    "asset_7",
                    "asset_8",
                    "asset_9",
                ],
                min=0,
            ),
        ),
    ]


class ByteDance2ReferenceNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDance2ReferenceNode",
            display_name="ByteDance Seedance 2.0 Reference to Video",
            category="partner/video/ByteDance",
            description="Generate, edit, or extend video using Seedance 2.0 with reference images, "
            "videos, and audio. Supports multimodal reference, video editing, and video extension.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "Seedance 2.0",
                            _seedance2_reference_inputs(["480p", "720p", "1080p", "4k"], default_ratio="adaptive"),
                        ),
                        IO.DynamicCombo.Option(
                            "Seedance 2.0 Fast",
                            _seedance2_reference_inputs(["480p", "720p"], default_ratio="adaptive"),
                        ),
                        IO.DynamicCombo.Option(
                            "Seedance 2.0 Mini",
                            _seedance2_reference_inputs(["480p", "720p"], default_ratio="adaptive"),
                        ),
                    ],
                    tooltip="Seedance 2.0 for maximum quality; Fast for speed optimization; "
                    "Mini for the fastest, lowest-cost generation.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add a watermark to the video.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["model", "model.resolution", "model.duration"],
                    input_groups=["model.reference_videos"],
                ),
                expr="""
                (
                  $rate480 := 10044;
                  $rate720 := 21600;
                  $rate1080 := 48800;
                  $rate4k := 195200;
                  $m := widgets.model;
                  $hasVideo := $lookup(inputGroups, "model.reference_videos") > 0;
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $noVideoPricePer1K := $res = "4k"    ? 0.00572 :
                                        $res = "1080p" ? 0.011011 :
                                        $contains($m, "mini") ? 0.005005 :
                                        $contains($m, "fast") ? 0.008008 : 0.01001;
                  $videoPricePer1K := $res = "4k"    ? 0.003432 :
                                      $res = "1080p" ? 0.006721 :
                                      $contains($m, "mini") ? 0.003003 :
                                      $contains($m, "fast") ? 0.004719 : 0.006149;
                  $rate := $res = "4k"    ? $rate4k :
                           $res = "1080p" ? $rate1080 :
                           $res = "720p"  ? $rate720 :
                                            $rate480;
                  $noVideoCost := $dur * $rate * $noVideoPricePer1K / 1000;
                  $minVideoFactor := $ceil($dur * 5 / 3);
                  $minVideoCost := $minVideoFactor * $rate * $videoPricePer1K / 1000;
                  $maxVideoCost := (15 + $dur) * $rate * $videoPricePer1K / 1000;
                  $hasVideo
                    ? {
                        "type": "range_usd",
                        "min_usd": $minVideoCost,
                        "max_usd": $maxVideoCost,
                        "format": {"approximate": true}
                      }
                    : {
                        "type": "usd",
                        "usd": $noVideoCost,
                        "format": {"approximate": true}
                      }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        seed: int,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=True, min_length=1)

        reference_images = model.get("reference_images", {})
        reference_videos = model.get("reference_videos", {})
        reference_audios = model.get("reference_audios", {})
        reference_assets = model.get("reference_assets", {})

        reference_image_assets, reference_video_assets, reference_audio_assets = await _resolve_reference_assets(
            cls, list(reference_assets.values())
        )

        if not reference_images and not reference_videos and not reference_image_assets and not reference_video_assets:
            raise ValueError("At least one reference image or video or asset is required.")

        total_images = len(reference_images) + len(reference_image_assets)
        if total_images > 9:
            raise ValueError(
                f"Too many reference images: {total_images} "
                f"(images={len(reference_images)}, image assets={len(reference_image_assets)}). Maximum is 9."
            )
        total_videos = len(reference_videos) + len(reference_video_assets)
        if total_videos > 3:
            raise ValueError(
                f"Too many reference videos: {total_videos} "
                f"(videos={len(reference_videos)}, video assets={len(reference_video_assets)}). Maximum is 3."
            )
        total_audios = len(reference_audios) + len(reference_audio_assets)
        if total_audios > 3:
            raise ValueError(
                f"Too many reference audios: {total_audios} "
                f"(audios={len(reference_audios)}, audio assets={len(reference_audio_assets)}). Maximum is 3."
            )

        for key in reference_images:
            reference_images[key] = _prepare_seedance_image(reference_images[key])

        model_id = SEEDANCE_MODELS[model["model"]]
        has_video_input = total_videos > 0

        if model.get("auto_downscale") and reference_videos:
            max_px = SEEDANCE2_REF_VIDEO_PIXEL_LIMITS.get(model_id, {}).get(model["resolution"], {}).get("max")
            if max_px:
                for key in reference_videos:
                    reference_videos[key] = downscale_video_to_max_pixels(reference_videos[key], max_px)

        if model.get("auto_upscale") and reference_videos:
            min_px = SEEDANCE2_REF_VIDEO_PIXEL_LIMITS.get(model_id, {}).get(model["resolution"], {}).get("min")
            if min_px:
                for key in reference_videos:
                    reference_videos[key] = upscale_video_to_min_pixels(reference_videos[key], min_px)

        total_video_duration = 0.0
        for i, key in enumerate(reference_videos, 1):
            video = reference_videos[key]
            _validate_ref_video_pixels(video, model_id, model["resolution"], i)
            try:
                dur = video.get_duration()
                if dur < 1.8:
                    raise ValueError(f"Reference video {i} is too short: {dur:.1f}s. Minimum duration is 1.8 seconds.")
                total_video_duration += dur
            except ValueError:
                raise
            except Exception:
                pass
        if total_video_duration > 15.1:
            raise ValueError(f"Total reference video duration is {total_video_duration:.1f}s. Maximum is 15.1 seconds.")

        total_audio_duration = 0.0
        for i, key in enumerate(reference_audios, 1):
            audio = reference_audios[key]
            dur = int(audio["waveform"].shape[-1]) / int(audio["sample_rate"])
            if dur < 1.8:
                raise ValueError(f"Reference audio {i} is too short: {dur:.1f}s. Minimum duration is 1.8 seconds.")
            total_audio_duration += dur
        if total_audio_duration > 15.1:
            raise ValueError(f"Total reference audio duration is {total_audio_duration:.1f}s. Maximum is 15.1 seconds.")

        asset_labels = _build_asset_labels(
            reference_assets,
            reference_image_assets,
            reference_video_assets,
            reference_audio_assets,
            len(reference_images),
            len(reference_videos),
            len(reference_audios),
        )
        prompt_text = _rewrite_asset_refs(model["prompt"], asset_labels)

        content: list[TaskTextContent | TaskImageContent | TaskVideoContent | TaskAudioContent] = [
            TaskTextContent(text=prompt_text),
        ]
        for i, key in enumerate(reference_images, 1):
            content.append(
                TaskImageContent(
                    image_url=TaskImageContentUrl(
                        url=await _seedance_virtual_library_upload_image_asset(
                            cls,
                            reference_images[key],
                            wait_label=f"Uploading image {i}",
                        ),
                    ),
                    role="reference_image",
                ),
            )
        for i, key in enumerate(reference_videos, 1):
            content.append(
                TaskVideoContent(
                    video_url=TaskVideoContentUrl(
                        url=await _seedance_virtual_library_upload_video_asset(
                            cls,
                            reference_videos[key],
                            wait_label=f"Uploading video {i}",
                        ),
                    ),
                ),
            )
        for key in reference_audios:
            content.append(
                TaskAudioContent(
                    audio_url=TaskAudioContentUrl(
                        url=await upload_audio_to_comfyapi(
                            cls,
                            reference_audios[key],
                            container_format="mp3",
                            codec_name="libmp3lame",
                            mime_type="audio/mpeg",
                        ),
                    ),
                ),
            )
        for url in reference_image_assets.values():
            content.append(
                TaskImageContent(
                    image_url=TaskImageContentUrl(url=url),
                    role="reference_image",
                ),
            )
        for url in reference_video_assets.values():
            content.append(
                TaskVideoContent(video_url=TaskVideoContentUrl(url=url)),
            )
        for url in reference_audio_assets.values():
            content.append(
                TaskAudioContent(audio_url=TaskAudioContentUrl(url=url)),
            )
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_TASK_ENDPOINT, method="POST"),
            data=Seedance2TaskCreationRequest(
                model=model_id,
                content=content,
                generate_audio=model["generate_audio"],
                resolution=model["resolution"],
                ratio=model["ratio"],
                duration=model["duration"],
                seed=seed,
                watermark=watermark,
            ),
            response_model=TaskCreationResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"{BYTEPLUS_SEEDANCE2_TASK_STATUS_ENDPOINT}/{initial_response.id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: r.status,
            price_extractor=_seedance2_price_extractor(
                model_id, has_video_input=has_video_input, resolution=model["resolution"]
            ),
            poll_interval=9,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.content.video_url))


async def process_video_task(
    cls: type[IO.ComfyNode],
    payload: Text2VideoTaskCreationRequest | Image2VideoTaskCreationRequest,
    estimated_duration: int | None,
) -> IO.NodeOutput:
    if payload.model in DEPRECATED_MODELS:
        logger.warning(
            "Model '%s' is deprecated and will be deactivated on May 13, 2026. "
            "Please switch to a newer model. Recommended: seedance-1-0-pro-fast-251015.",
            payload.model,
        )
    initial_response = await sync_op(
        cls,
        ApiEndpoint(path=BYTEPLUS_TASK_ENDPOINT, method="POST"),
        data=payload,
        response_model=TaskCreationResponse,
    )
    response = await poll_op(
        cls,
        ApiEndpoint(path=f"{BYTEPLUS_TASK_STATUS_ENDPOINT}/{initial_response.id}"),
        status_extractor=lambda r: r.status,
        estimated_duration=estimated_duration,
        response_model=TaskStatusResponse,
    )
    return IO.NodeOutput(await download_url_to_video_output(response.content.video_url))


class ByteDanceCreateImageAsset(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ByteDanceCreateImageAsset",
            display_name="ByteDance Create Image Asset",
            category="partner/image/ByteDance",
            description=(
                "Create a Seedance 2.0 personal image asset. Uploads the input image and "
                "registers it in the given asset group. If group_id is empty, runs a real-person "
                "H5 authentication flow to create a new group before adding the asset."
            ),
            inputs=[
                IO.Image.Input("image", tooltip="Image to register as a personal asset."),
                IO.String.Input(
                    "group_id",
                    default="",
                    tooltip="Reuse an existing Seedance asset group ID to skip repeated human verification for the "
                    "same person. Leave empty to run real-person authentication in the browser and create a new group.",
                ),
                # IO.String.Input(
                #     "name",
                #     default="",
                #     tooltip="Asset name (up to 64 characters).",
                # ),
            ],
            outputs=[
                IO.String.Output(display_name="asset_id"),
                IO.String.Output(display_name="group_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            # is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        group_id: str = "",
        # name: str = "",
    ) -> IO.NodeOutput:
        # if len(name) > 64:
        #     raise ValueError("Name of asset can not be greater then 64 symbols")
        validate_image_dimensions(image, min_width=300, max_width=6000, min_height=300, max_height=6000)
        validate_image_aspect_ratio(image, min_ratio=(0.4, 1), max_ratio=(2.5, 1))
        resolved_group = await _resolve_group_id(cls, group_id)
        asset_id = await _create_seedance_asset(
            cls,
            group_id=resolved_group,
            url=await upload_image_to_comfyapi(cls, image),
            name="",
            asset_type="Image",
        )
        await _wait_for_asset_active(cls, asset_id, resolved_group)
        PromptServer.instance.send_progress_text(
            f"Please save the asset_id and group_id for reuse.\n\nasset_id: {asset_id}\n\n"
            f"group_id: {resolved_group}",
            cls.hidden.unique_id,
        )
        return IO.NodeOutput(asset_id, resolved_group)


class ByteDanceCreateVideoAsset(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ByteDanceCreateVideoAsset",
            display_name="ByteDance Create Video Asset",
            category="partner/video/ByteDance",
            description=(
                "Create a Seedance 2.0 personal video asset. Uploads the input video and "
                "registers it in the given asset group. If group_id is empty, runs a real-person "
                "H5 authentication flow to create a new group before adding the asset."
            ),
            inputs=[
                IO.Video.Input("video", tooltip="Video to register as a personal asset."),
                IO.String.Input(
                    "group_id",
                    default="",
                    tooltip="Reuse an existing Seedance asset group ID to skip repeated human verification for the "
                    "same person. Leave empty to run real-person authentication in the browser and create a new group.",
                ),
                # IO.String.Input(
                #     "name",
                #     default="",
                #     tooltip="Asset name (up to 64 characters).",
                # ),
            ],
            outputs=[
                IO.String.Output(display_name="asset_id"),
                IO.String.Output(display_name="group_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            # is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        group_id: str = "",
        # name: str = "",
    ) -> IO.NodeOutput:
        # if len(name) > 64:
        #     raise ValueError("Name of asset can not be greater then 64 symbols")
        validate_video_duration(video, min_duration=2, max_duration=15)
        validate_video_dimensions(video, min_width=300, max_width=6000, min_height=300, max_height=6000)

        w, h = video.get_dimensions()
        if h > 0:
            ratio = w / h
            if not (0.4 <= ratio <= 2.5):
                raise ValueError(f"Asset video aspect ratio (W/H) must be in [0.4, 2.5], got {ratio:.3f} ({w}x{h}).")
        pixels = w * h
        if not (409_600 <= pixels <= 927_408):
            raise ValueError(
                f"Asset video total pixels (W×H) must be in [409600, 927408], " f"got {pixels:,} ({w}x{h})."
            )

        fps = float(video.get_frame_rate())
        if not (24 <= fps <= 60):
            raise ValueError(f"Asset video FPS must be in [24, 60], got {fps:.2f}.")

        resolved_group = await _resolve_group_id(cls, group_id)
        asset_id = await _create_seedance_asset(
            cls,
            group_id=resolved_group,
            url=await upload_video_to_comfyapi(cls, video),
            name="",
            asset_type="Video",
        )
        await _wait_for_asset_active(cls, asset_id, resolved_group)
        PromptServer.instance.send_progress_text(
            f"Please save the asset_id and group_id for reuse.\n\nasset_id: {asset_id}\n\n"
            f"group_id: {resolved_group}",
            cls.hidden.unique_id,
        )
        return IO.NodeOutput(asset_id, resolved_group)


class ByteDanceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ByteDanceImageNode,
            ByteDanceSeedreamNode,
            ByteDanceSeedreamNodeV2,
            ByteDanceTextToVideoNode,
            ByteDanceImageToVideoNode,
            ByteDanceFirstLastFrameNode,
            ByteDanceImageReferenceNode,
            ByteDance2TextToVideoNode,
            ByteDance2FirstLastFrameNode,
            ByteDance2ReferenceNode,
            ByteDanceCreateImageAsset,
            ByteDanceCreateVideoAsset,
        ]


async def comfy_entrypoint() -> ByteDanceExtension:
    return ByteDanceExtension()
