import asyncio
import base64
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
    RECOMMENDED_PRESETS_SEEDREAM_4,
    RECOMMENDED_PRESETS_SEEDREAM_4_0,
    RECOMMENDED_PRESETS_SEEDREAM_4_5,
    RECOMMENDED_PRESETS_SEEDREAM_5_LITE,
    RECOMMENDED_PRESETS_SEEDREAM_5_PRO,
    SEEDANCE2_REF_VIDEO_PIXEL_LIMITS,
    VIDEO_TASKS_EXECUTION_TIME,
    GetAssetResponse,
    Image2VideoTaskCreationRequest,
    ImageTaskCreationResponse,
    MediaKitTaskCreateResponse,
    MediaKitTaskResponse,
    MediaKitVideoEnhanceRequest,
    SeedAudioConfig,
    SeedAudioReference,
    SeedAudioRequest,
    SeedAudioResponse,
    Seedance2TaskCreationRequest,
    SeedanceCreateAssetRequest,
    SeedanceCreateAssetResponse,
    SeedanceCreateVisualValidateSessionResponse,
    SeedanceGetVisualValidateSessionResponse,
    SeedanceVirtualLibraryCreateAssetRequest,
    Seedream4Options,
    Seedream4TaskCreationRequest,
    Seedream5LayerOptimizePromptOptions,
    Seedream5LayerSeparationRequest,
    Seedream5OptimizePromptOptions,
    TaskAudioContent,
    TaskAudioContentUrl,
    TaskCreationResponse,
    TaskImageContent,
    TaskImageContentUrl,
    TaskStatusResponse,
    TaskTextContent,
    TaskVideoContent,
    TaskVideoContentUrl,
    Text2VideoTaskCreationRequest,
    seedance2_reference_limits,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_bytes_to_audio_input,
    audio_input_to_mp3,
    bytesio_to_image_tensor,
    download_url_as_bytesio,
    download_url_to_image_tensor,
    download_url_to_video_output,
    downscale_image_tensor_by_max_side,
    downscale_video_to_max_pixels,
    get_number_of_images,
    image_tensor_pair_to_batch,
    poll_op,
    sync_op,
    tensor_to_base64_string,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    upscale_image_tensor_to_min_pixels,
    upscale_video_to_min_pixels,
    validate_audio_duration,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_string,
    validate_video_dimensions,
    validate_video_duration,
)
from comfy_api_nodes.util.common_exceptions import ProcessingInterrupted
from server import PromptServer

BYTEPLUS_IMAGE_ENDPOINT = "/proxy/byteplus/api/v3/images/generations"

_VERIFICATION_POLL_TIMEOUT_SEC = 120
_VERIFICATION_POLL_INTERVAL_SEC = 3

SEEDREAM_MODELS = {
    "seedream 5.0 pro": "seedream-5-0-pro-260628",
    "seedream 5.0 lite": "seedream-5-0-260128",
    "seedream-4-5-251128": "seedream-4-5-251128",
    "seedream-4-0-250828": "seedream-4-0-250828",
}

SEEDREAM_PRESETS = {
    "seedream-5-0-pro-260628": RECOMMENDED_PRESETS_SEEDREAM_5_PRO,
    "seedream-5-0-260128": RECOMMENDED_PRESETS_SEEDREAM_5_LITE,
    "seedream-4-5-251128": RECOMMENDED_PRESETS_SEEDREAM_4_5,
    "seedream-4-0-250828": RECOMMENDED_PRESETS_SEEDREAM_4_0,
}

SEEDREAM_LAYER_SEPARATION_MODEL = "seedream-5-0-pro-260628"

# Long-running tasks endpoints(e.g., video)
BYTEPLUS_TASK_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"
BYTEPLUS_TASK_STATUS_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"  # + /{task_id}
BYTEPLUS_SEEDANCE2_TASK_STATUS_ENDPOINT = "/proxy/byteplus-seedance2/api/v3/contents/generations/tasks"  # + /{task_id}

SEEDANCE_MODELS = {
    "Seedance 2.5": "dreamina-seedance-2-5-260628",
    "Seedance 2.0": "dreamina-seedance-2-0-260128",
    "Seedance 2.0 Fast": "dreamina-seedance-2-0-fast-260128",
    "Seedance 2.0 Mini": "dreamina-seedance-2-0-mini",
}

SEEDANCE_MODEL_TOOLTIP = (
    "Seedance 2.5 for the newest model, videos up to 30 seconds and mp4/mov output; "
    "Seedance 2.0 for maximum quality and 4k; Fast for speed optimization; "
    "Mini for the fastest, lowest-cost generation."
)

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


def get_image_url_from_response(response: ImageTaskCreationResponse) -> str:
    if response.error:
        error_msg = f"ByteDance request failed. Code: {response.error['code']}, message: {response.error['message']}"
        logging.info(error_msg)
        raise RuntimeError(error_msg)
    logging.info("ByteDance task succeeded, image URL: %s", response.data[0]["url"])
    return response.data[0]["url"]


class ByteDanceSeedreamNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceSeedreamNode",
            display_name="ByteDance Seedream 4.5 & 5.0 (legacy)",
            category="partner/image/ByteDance",
            description=(
                "Unified text-to-image generation and precise single-sentence editing at up to 4K resolution. "
                "This class name is historical and is kept only for compatibility with existing workflows; "
                "it supports the current models, "
                "so prefer the ByteDanceSeedreamNodeV2 ('ByteDance Seedream 4.5 & 5.0') node for new workflows."
            ),
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


def _seedream_model_inputs(
    *,
    max_ref_images: int,
    presets: list,
    max_width: int = 6240,
    max_height: int = 4992,
    supports_batch: bool = True,
    supports_fast: bool = False,
    include_common: bool = False,
):
    inputs = [
        IO.Combo.Input(
            "size_preset",
            options=[label for label, _, _ in presets],
            tooltip="Pick a recommended size. Select Custom to use the width and height below.",
        ),
        IO.Int.Input(
            "width",
            default=2048,
            min=1024,
            max=max_width,
            step=2,
            tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
        ),
        IO.Int.Input(
            "height",
            default=2048,
            min=1024,
            max=max_height,
            step=2,
            tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
        ),
    ]
    if supports_batch:
        inputs.append(
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
            )
        )
    inputs.append(
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, max_ref_images + 1)],
                min=0,
            ),
            tooltip=f"Optional reference image(s) for image-to-image or multi-reference generation. "
            f"Up to {max_ref_images} images.",
        )
    )
    if supports_batch:
        inputs.append(
            IO.Boolean.Input(
                "fail_on_partial",
                default=False,
                tooltip="If enabled, abort execution if any requested images are missing or return an error.",
                advanced=True,
            )
        )
    if supports_fast:
        inputs.append(
            IO.Combo.Input(
                "prompt_optimization",
                options=["standard", "fast"],
                default="standard",
                tooltip="Prompt-optimization mode when reference images are provided: "
                "'standard' gives higher quality, 'fast' shorter generation time.",
                advanced=True,
            )
        )
    if include_common:
        inputs.extend(
            [
                IO.Int.Input(
                    "seed",
                    default=42,
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
                IO.Boolean.Input(
                    "thinking",
                    default=True,
                    tooltip=(
                        "Enable the model's prompt-optimization reasoning ('thinking') for better adherence. "
                        "Can substantially increase generation time — notably on Seedream 5.0 Pro. "
                        "Can only be disabled for text-to-image (not when reference images are provided)."
                    ),
                    advanced=True,
                ),
            ]
        )
    return inputs


class ByteDanceSeedreamNodeV3(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceSeedreamNodeV3",
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
                            "seedream 5.0 pro",
                            _seedream_model_inputs(
                                max_ref_images=10,
                                presets=RECOMMENDED_PRESETS_SEEDREAM_5_PRO,
                                max_width=3136,
                                max_height=2496,
                                supports_batch=False,
                                supports_fast=True,
                                include_common=True,
                            ),
                        ),
                        IO.DynamicCombo.Option(
                            "seedream 5.0 lite",
                            _seedream_model_inputs(
                                max_ref_images=14,
                                presets=RECOMMENDED_PRESETS_SEEDREAM_5_LITE,
                                include_common=True,
                            ),
                        ),
                        IO.DynamicCombo.Option(
                            "seedream-4-5-251128",
                            _seedream_model_inputs(
                                max_ref_images=10,
                                presets=RECOMMENDED_PRESETS_SEEDREAM_4_5,
                                include_common=True,
                            ),
                        ),
                        IO.DynamicCombo.Option(
                            "seedream-4-0-250828",
                            _seedream_model_inputs(
                                max_ref_images=10,
                                presets=RECOMMENDED_PRESETS_SEEDREAM_4_0,
                                include_common=True,
                            ),
                        ),
                    ],
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
                depends_on=IO.PriceBadgeDepends(
                    widgets=["model", "model.size_preset", "model.width", "model.height"],
                    input_groups=["model.images"],
                ),
                expr="""
                (
                  $model := $string(widgets.model);
                  $sp := $string($lookup(widgets, "model.size_preset"));
                  $w := $lookup(widgets, "model.width");
                  $h := $lookup(widgets, "model.height");
                  $px := ($type($w) = "number" and $type($h) = "number") ? $w * $h : 0;
                  $refs := $lookup(inputGroups, "model.images");
                  $extra := ($type($refs) = "number" and $refs > 1) ? ($refs - 1) * 0.003 : 0;
                  $isPro := $contains($model, "5.0 pro");
                  $isCustom := $contains($sp, "custom");
                  $sizeKnown := $isCustom ? $px > 0 : ($contains($sp, "1k") or $contains($sp, "2k"));
                  $proPrice := $isCustom
                                 ? ($px < 2610000 ? 0.045 : 0.09)
                                 : ($contains($sp, "1k") ? 0.045 : 0.09);
                  ($isPro and ($sizeKnown = false))
                    ? {
                        "type": "range_usd",
                        "min_usd": 0.045 + $extra,
                        "max_usd": 0.09 + $extra,
                        "format": { "suffix": "/Image", "approximate": true }
                      }
                    : {
                        "type": "usd",
                        "usd": $isPro ? $proPrice + $extra
                               : $contains($model, "5.0 lite") ? 0.035
                               : $contains($model, "4-5") ? 0.04
                               : 0.03,
                        "format": { "suffix": $isPro ? "/Image" : " x images/Run", "approximate": true }
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
        thinking: bool = True,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        model_id = SEEDREAM_MODELS[model["model"]]
        presets = SEEDREAM_PRESETS[model_id]
        is_pro = "seedream-5-0-pro" in model_id

        size_preset = model.get("size_preset", presets[0][0])
        width = model.get("width", 2048)
        height = model.get("height", 2048)
        max_images = model.get("max_images", 1)
        sequential_image_generation = "disabled" if max_images == 1 else "auto"
        images_dict = model.get("images") or {}
        fail_on_partial = model.get("fail_on_partial", False)
        prompt_optimization = model.get("prompt_optimization", "standard")
        seed = model.get("seed", seed)
        watermark = model.get("watermark", watermark)
        thinking = model.get("thinking", thinking)

        w = h = None
        for label, tw, th in presets:
            if label == size_preset:
                w, h = tw, th
                break
        if w is None or h is None:
            w, h = width, height

        out_num_pixels = w * h
        mp_provided = out_num_pixels / 1_000_000.0
        if is_pro:
            if out_num_pixels < 921_600:
                raise ValueError(
                    f"Minimum image resolution for the selected model is 0.92MP, but {mp_provided:.2f}MP provided."
                )
            if out_num_pixels > 4_194_304:
                raise ValueError(
                    f"Maximum image resolution for the selected model is 4.19MP, but {mp_provided:.2f}MP provided."
                )
        else:
            if ("seedream-4-5" in model_id or "seedream-5-0" in model_id) and out_num_pixels < 3_686_400:
                raise ValueError(
                    f"Minimum image resolution for the selected model is 3.68MP, but {mp_provided:.2f}MP provided."
                )
            if "seedream-4-0" in model_id and out_num_pixels < 921_600:
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
        if not thinking and n_input_images > 0:
            raise ValueError(
                "'thinking' can only be disabled for text-to-image; enable it when using reference images."
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

        optimize_prompt_options = None
        if n_input_images == 0:
            optimize_prompt_options = Seedream5OptimizePromptOptions(thinking="enabled" if thinking else "disabled")
        elif prompt_optimization == "fast":
            optimize_prompt_options = Seedream5OptimizePromptOptions(mode="fast")
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
                sequential_image_generation=None if is_pro else sequential_image_generation,
                sequential_image_generation_options=None if is_pro else Seedream4Options(max_images=max_images),
                watermark=watermark,
                optimize_prompt_options=optimize_prompt_options,
            ),
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))
        urls = [str(d["url"]) for d in response.data if isinstance(d, dict) and "url" in d]
        if fail_on_partial and len(urls) < len(response.data):
            raise RuntimeError(f"Only {len(urls)} of {len(response.data)} images were generated before error.")
        return IO.NodeOutput(torch.cat([await download_url_to_image_tensor(i) for i in urls]))


class ByteDanceSeedreamNodeV2(ByteDanceSeedreamNodeV3):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceSeedreamNodeV2",
            display_name="ByteDance Seedream 4.5 & 5.0 (Legacy)",
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
                            "seedream 5.0 pro",
                            _seedream_model_inputs(
                                max_ref_images=10,
                                presets=RECOMMENDED_PRESETS_SEEDREAM_5_PRO,
                                max_width=3136,
                                max_height=2496,
                                supports_batch=False,
                            ),
                        ),
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
                IO.Boolean.Input(
                    "thinking",
                    default=True,
                    tooltip=(
                        "Enable the model's prompt-optimization reasoning ('thinking') for better adherence. "
                        "Can substantially increase generation time — notably on Seedream 5.0 Pro. "
                        "Can only be disabled for text-to-image (not when reference images are provided)."
                    ),
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
            is_deprecated=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["model", "model.size_preset", "model.width", "model.height"]
                ),
                expr="""
                (
                  $sp := $lookup(widgets, "model.size_preset");
                  $px := $lookup(widgets, "model.width") * $lookup(widgets, "model.height");
                  $isPro := $contains(widgets.model, "5.0 pro");
                  $price := $isPro
                              ? (
                                  $contains($sp, "custom")
                                    ? ($px <= 2360000 ? 0.045 : 0.09)
                                    : ($contains($sp, "1k") ? 0.045 : 0.09)
                                )
                              : $contains(widgets.model, "5.0 lite") ? 0.035
                              : $contains(widgets.model, "4-5") ? 0.04
                              : 0.03;
                  {
                    "type": "usd",
                    "usd": $price,
                    "format": { "suffix": $isPro ? "/Image" : " x images/Run", "approximate": true }
                  }
                )
                """,
            ),
        )

class ByteDanceSeedreamLayerSeparationNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceSeedreamLayerSeparationNode",
            display_name="ByteDance Seedream 5.0 Pro Layer Separation",
            category="partner/image/ByteDance",
            search_aliases=["layer separation", "split layers", "decompose", "cutout", "RGBA layers"],
            description=(
                "Decompose an image into a background plate plus up to 16 repositionable transparent layers, "
                "each with stacking order, bounding box, name and description."
            ),
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip=(
                        "The image to separate. Exactly one image, at least 512x512 pixels, aspect ratio "
                        "between 1:16 and 16:1. Inputs larger than about 4MP are downscaled before upload."
                    ),
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "How to separate the image. Leave empty to auto-detect and separate all major elements. "
                        "Describe elements in natural language to control the separation, or target exact regions "
                        "with <bbox>left top right bottom</bbox> tags (0-1000 per-mille coordinates)."
                    ),
                ),
                IO.Combo.Input(
                    "size",
                    options=["auto", "1K", "1.5K", "2K"],
                    default="auto",
                    tooltip="Output resolution level. 'auto' follows the input image size (clamped to the 1K-2K range).",
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
                IO.Combo.Input(
                    "prompt_optimization",
                    options=["standard", "fast"],
                    default="standard",
                    optional=True,
                    advanced=True,
                    tooltip="Prompt-optimization mode: 'standard' gives higher quality, 'fast' shorter generation time.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip='Whether to add an "AI generated" watermark to the images.',
                ),
                IO.Boolean.Input(
                    "crop_layers",
                    default=False,
                    optional=True,
                    label_on="minimal size",
                    label_off="full canvas",
                    tooltip=(
                        "Geometry of the layers/masks batch outputs (layer_stack is unaffected and always "
                        "tight). Full canvas: each layer on a base-sized canvas at its bounding-box position - "
                        "recompose directly with ImageCompositeMasked. Minimal size: each layer cropped to its "
                        "bounding box (padded to the largest layer for batching) - much smaller tensors; "
                        "rebuild placement with Layers From Bounding Boxes using the bboxes output."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="base_image",
                    tooltip="The base image (background plate) the layers stack onto.",
                ),
                IO.Mask.Output(
                    display_name="base_mask",
                    tooltip=(
                        "Transparency of the base image (1 = transparent, LoadImage convention); currently "
                        "always fully opaque."
                    ),
                ),
                IO.Image.Output(
                    display_name="layers",
                    tooltip=(
                        "Transparent layers ordered bottom to top. Full canvas mode: placed on a black "
                        "base-sized canvas at their bounding-box position. Minimal size mode: cropped to "
                        "their bounding box, anchored top-left, padded to the largest layer."
                    ),
                ),
                IO.Mask.Output(
                    display_name="masks",
                    tooltip=(
                        "Per-layer transparency, index-aligned with the layers batch (1 = transparent, "
                        "LoadImage convention). For ImageCompositeMasked-style compositing, add InvertMask first."
                    ),
                ),
                IO.BoundingBox.Output(
                    display_name="bboxes",
                    tooltip=(
                        "One placement box per layer, index-aligned with the layers batch (feed both, plus "
                        "masks, into Layers From Bounding Boxes to rebuild per-layer placement): {x, y, width, "
                        "height, metadata: {name, desc, z_index, native_size, content_rect, flags}}. "
                        "content_rect = [left, top, width, height] is the layer's content region within its "
                        "own frame; it lands on the canvas at the box position plus that offset."
                    ),
                ),
                IO.Layers.Output(
                    display_name="layer_stack",
                    tooltip=(
                        "Ready-to-edit layer document for Create Layered Image: the base plate plus each "
                        "element as its own named, tight-cropped layer at its true position and stacking "
                        "order. Connect directly, or extend with Add Layer."
                    ),
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["size"]),
                expr="""
                (
                  widgets.size in ["1k", "1.5k"]
                    ? {
                        "type": "usd",
                        "usd": 0.032,
                        "format": { "suffix": " x images/Run", "approximate": true }
                      }
                    : {
                        "type": "range_usd",
                        "min_usd": 0.032,
                        "max_usd": 0.064,
                        "format": { "suffix": " x images/Run", "approximate": true }
                      }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        prompt: str = "",
        size: str = "auto",
        seed: int = 0,
        prompt_optimization: str = "standard",
        watermark: bool = False,
        crop_layers: bool = False,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) != 1:
            raise ValueError("Only a single input image is supported.")
        validate_image_aspect_ratio(image, (1, 16), (16, 1), strict=False)
        validate_image_dimensions(image, min_width=512, min_height=512)

        request = Seedream5LayerSeparationRequest(
            model=SEEDREAM_LAYER_SEPARATION_MODEL,
            prompt=prompt.strip() or None,
            image=await upload_image_to_comfyapi(cls, image),
            size=size,
            seed=seed,
            watermark=watermark,
            optimize_prompt_options=Seedream5LayerOptimizePromptOptions(mode=prompt_optimization),
        )
        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_IMAGE_ENDPOINT, method="POST"),
            response_model=ImageTaskCreationResponse,
            data=request,
            wait_label="Separating layers",
        )
        if response.error:
            raise RuntimeError(
                f"ByteDance request failed. Code: {response.error['code']}, message: {response.error['message']}"
            )

        def z_index_of(d: dict) -> int:
            v = d.get("z_index")
            if isinstance(v, bool):
                return 1_000_000
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str):
                try:
                    return int(v.strip())
                except ValueError:
                    return 1_000_000
            return 1_000_000

        data = [d for d in (response.data or []) if isinstance(d, dict)]
        if not data or "url" not in data[0]:
            raise RuntimeError("Unexpected response: no base image returned.")
        base_item = data[0]
        if base_item.get("bounding_box") is not None:
            logging.warning(
                "ByteDance layer separation: base item unexpectedly carries a bounding_box; ignoring it."
            )
        if z_index_of(base_item) not in (0, 1_000_000):
            raise RuntimeError("Unexpected response: the first item is not the base image.")
        layer_items = [d for d in data[1:] if "url" in d]
        dropped = len(data) - 1 - len(layer_items)
        if dropped > 0:
            logging.warning(
                "ByteDance layer separation: %d of %d returned elements had no 'url' and were dropped.",
                dropped,
                len(data) - 1,
            )
        if not layer_items:
            raise RuntimeError("The model returned no layers. Try a different prompt or input image.")
        layer_items.sort(key=z_index_of)

        base_image = (await download_url_to_image_tensor(str(base_item["url"])))[..., :3].contiguous()
        height, width = base_image.shape[1], base_image.shape[2]

        specs = []
        for item in layer_items:
            flags = []
            bbox = item.get("bounding_box")
            absolute = bbox.get("absolute") if isinstance(bbox, dict) else None
            if (
                isinstance(absolute, (list, tuple))
                and len(absolute) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in absolute)
            ):
                left, top, right, bottom = (int(round(v)) for v in absolute)
                rect_w, rect_h = right - left, bottom - top  # exclusive right/bottom
                if rect_w > width or rect_h > height:
                    rect_w, rect_h = min(rect_w, width), min(rect_h, height)
                    flags.append("bbox_clamped")
                if rect_w <= 0 or rect_h <= 0:
                    flags.append("bbox_degenerate")
            else:
                flags.append("bbox_missing")
                left, top, rect_w, rect_h = 0, 0, width, height
            specs.append({"item": item, "flags": flags, "left": left, "top": top,
                          "rect_w": rect_w, "rect_h": rect_h, "native_size": "", "stack_item": None})

        if crop_layers:
            canvas_w = max((s["rect_w"] for s in specs if "bbox_degenerate" not in s["flags"]), default=1)
            canvas_h = max((s["rect_h"] for s in specs if "bbox_degenerate" not in s["flags"]), default=1)
        else:
            canvas_w, canvas_h = width, height
        base_mask = torch.zeros((1, height, width))
        layers = torch.zeros((len(specs), canvas_h, canvas_w, 3))
        # Create Layered Image / LoadImage mask convention: 1 = transparent
        masks = torch.ones((len(specs), canvas_h, canvas_w))

        semaphore = asyncio.Semaphore(4)

        async def fetch_and_place(i: int, spec: dict) -> None:
            item, flags = spec["item"], spec["flags"]
            left, top, rect_w, rect_h = spec["left"], spec["top"], spec["rect_w"], spec["rect_h"]
            async with semaphore:
                try:
                    # the layer math below needs the alpha channel, and ByteDance encodes
                    # alpha-less images as plain RGB (the base plate is one), so force RGBA
                    rgba = bytesio_to_image_tensor(await download_url_as_bytesio(str(item["url"])), mode="RGBA")[0]
                except ProcessingInterrupted:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to download layer {i + 1} of {len(specs)} (name={item.get('name')!r}): {exc} "
                        "The generation completed and was billed; the response with all layer URLs "
                        "is in ComfyUI/temp/api_logs/."
                    ) from exc
            spec["native_size"] = f"{rgba.shape[1]}x{rgba.shape[0]}"
            if "bbox_degenerate" in flags:
                return
            if (rgba.shape[1], rgba.shape[0]) != (rect_w, rect_h):
                # premultiply before resizing: interpolating straight alpha bleeds the undefined
                # colors of transparent pixels into the anti-aliased edges
                rgba = rgba.clone()
                rgba[..., :3] *= rgba[..., 3:4]
                rgba = (
                    torch.nn.functional.interpolate(
                        rgba.permute(2, 0, 1).unsqueeze(0),
                        size=(rect_h, rect_w),
                        mode="bilinear",
                        antialias=True,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                alpha = rgba[..., 3:4]
                rgba = torch.cat([rgba[..., :3] / alpha.clamp(min=1e-6), alpha], dim=-1).clamp(0, 1)
                flags.append("resized_to_bbox")
            # straight (unpremultiplied) RGB: downstream compositing applies the mask itself
            if crop_layers:
                layers[i, :rect_h, :rect_w] = rgba[..., :3]
                masks[i, :rect_h, :rect_w] = 1.0 - rgba[..., 3]
            else:
                x0, y0 = max(left, 0), max(top, 0)
                x1, y1 = min(left + rect_w, width), min(top + rect_h, height)
                if x0 < x1 and y0 < y1:
                    patch = rgba[y0 - top : y1 - top, x0 - left : x1 - left]
                    layers[i, y0:y1, x0:x1] = patch[..., :3]
                    masks[i, y0:y1, x0:x1] = 1.0 - patch[..., 3]
                else:
                    flags.append("bbox_out_of_canvas")
            zi = z_index_of(item)
            stack_item = {
                "image": rgba[..., :3].unsqueeze(0).contiguous(),
                "type": "raster",
                "x": left,
                "y": top,
                "z_index": zi if zi != 1_000_000 else i + 1,
                "mask": (1.0 - rgba[..., 3]).unsqueeze(0),
            }
            if isinstance(item.get("name"), str):
                stack_item["name"] = item["name"]
            spec["stack_item"] = stack_item

        await asyncio.gather(*(fetch_and_place(i, s) for i, s in enumerate(specs)))

        stack_items = [{"image": base_image, "type": "raster", "x": 0, "y": 0, "z_index": 0, "name": "background"}]
        boxes = []
        for i, s in enumerate(specs):
            abnormal = [f for f in s["flags"] if f != "resized_to_bbox"]
            if abnormal:
                logging.warning(
                    "ByteDance layer separation: layer %d (%r) flagged %s.",
                    i + 1,
                    s["item"].get("name"),
                    ", ".join(abnormal),
                )
            if s["stack_item"] is not None:
                stack_items.append(s["stack_item"])
            zi = z_index_of(s["item"])
            # placement box sized to this layer's tensor so Create Layered Image renders it 1:1;
            # the true content rect travels in metadata, frame-relative
            rect_x, rect_y = (0, 0) if crop_layers else (s["left"], s["top"])
            boxes.append(
                {
                    "x": s["left"] if crop_layers else 0,
                    "y": s["top"] if crop_layers else 0,
                    "width": canvas_w,
                    "height": canvas_h,
                    "metadata": {
                        "name": s["item"].get("name"),
                        "desc": s["item"].get("description"),
                        "z_index": zi if zi != 1_000_000 else None,
                        "native_size": s["native_size"],
                        "content_rect": [rect_x, rect_y, max(s["rect_w"], 0), max(s["rect_h"], 0)],
                        "flags": s["flags"],
                    },
                }
            )
        # a single frame holding every box: the per-frame BOUNDING_BOX shape for boxes that
        # annotate one image, as emitted and consumed by CreateBoundingBoxes
        bboxes = [boxes]
        layer_stack = {"version": 1, "canvas": (width, height), "layers": stack_items}
        return IO.NodeOutput(base_image, base_mask, layers, masks, bboxes, layer_stack)


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
                    options=["seedance-1-5-pro-251215", "seedance-1-0-pro-250528"],
                    default="seedance-1-5-pro-251215",
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
        }
      };
      $model := widgets.model;
      $modelKey :=
        $contains($model, "seedance-1-5-pro")      ? "seedance-1-5-pro" :
        $contains($model, "seedance-1-0-pro-fast") ? "seedance-1-0-pro-fast" :
        "seedance-1-0-pro";
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


def _seedance25_text_inputs(with_ratio: bool = True, with_video_editing: bool = False, with_task_type: bool = False):
    return [
        IO.String.Input(
            "prompt",
            multiline=True,
            default="",
            tooltip="Text prompt for video generation. Put spoken lines in double quotes to steer "
            "the generated dialogue.",
        ),
        IO.Combo.Input(
            "resolution",
            options=["480p", "720p", "1080p"],
            default="720p",
            tooltip="Resolution of the output video.",
        ),
        *(
            [
                IO.Combo.Input(
                    "ratio",
                    options=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"],
                    default="16:9",
                    tooltip="Aspect ratio of the output video.",
                )
            ]
            if with_ratio
            else []
        ),
        IO.Int.Input(
            "duration",
            default=5,
            min=4,
            max=30,
            step=1,
            tooltip="Duration of the output video in seconds (4-30).",
            display_mode=IO.NumberDisplay.slider,
        ),
        IO.Boolean.Input(
            "generate_audio",
            default=True,
            tooltip="Enable audio generation for the output video.",
        ),
        *(
            [
                IO.Boolean.Input(
                    "video_editing",
                    default=False,
                    tooltip="Enable when the prompt edits a connected reference video, for example "
                    "replacing an object in it. The output then keeps the source clip's own length "
                    "and aspect ratio, and the duration and ratio widgets are ignored. Leave "
                    "disabled to generate a new video, or to extend one to the duration you set.",
                )
            ]
            if with_video_editing
            else []
        ),
        *(
            [
                IO.Combo.Input(
                    "task_type",
                    options=["auto", "reference", "edit", "extend"],
                    default="auto",
                    tooltip="What to do with the reference media. Every value except auto is "
                    "validated when the task is submitted, so mismatched settings fail before "
                    "generation starts. auto: the model infers the task from the prompt and "
                    "inputs, and settings that conflict with its reading fail only after "
                    "generation has started. reference: generate a new video guided by the "
                    "reference images, videos, and audio. edit: change a connected reference "
                    "video (add, remove, replace); the output keeps the source clip's own length "
                    "and aspect ratio, and the duration and ratio widgets are ignored. extend: "
                    "continue a connected reference video forward or backward; the prompt should "
                    "say 'extend forward', 'extend backward', or 'continue', the aspect ratio "
                    "follows the source clip, and the output contains only the newly generated "
                    "segment of the duration you set, not the source clip.",
                )
            ]
            if with_task_type
            else []
        ),
        IO.Combo.Input(
            "output_format",
            options=["mp4"],
            default="mp4",
            tooltip="Container format of the output video.",
        ),
    ]


def _seedance25_reference_inputs(with_video_editing: bool = False, with_task_type: bool = False):
    return [
        *_seedance25_text_inputs(with_video_editing=with_video_editing, with_task_type=with_task_type),
        IO.Autogrow.Input(
            "reference_images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("reference_image"),
                names=[f"image_{i}" for i in range(1, 31)],
                min=0,
            ),
        ),
        IO.Autogrow.Input(
            "reference_videos",
            template=IO.Autogrow.TemplateNames(
                IO.Video.Input("reference_video"),
                names=[f"video_{i}" for i in range(1, 11)],
                min=0,
            ),
        ),
        IO.Autogrow.Input(
            "reference_audios",
            template=IO.Autogrow.TemplateNames(
                IO.Audio.Input("reference_audio"),
                names=[f"audio_{i}" for i in range(1, 11)],
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
                names=[f"asset_{i}" for i in range(1, 31)],
                min=0,
            ),
        ),
    ]


def _seedance2_build_request(
    model: dict,
    model_id: str,
    content: list,
    seed: int,
    watermark: bool,
    ratio: str,
) -> Seedance2TaskCreationRequest:
    task_type = model.get("task_type", "auto")
    duration = model["duration"]
    if model.get("video_editing") or task_type == "edit":
        ratio, duration = "adaptive", -1
    elif task_type == "extend":
        ratio = "adaptive"
    return Seedance2TaskCreationRequest(
        model=model_id,
        content=content,
        generate_audio=model["generate_audio"],
        resolution=model["resolution"],
        ratio=ratio,
        duration=duration,
        seed=seed,
        watermark=watermark,
        output_format=model.get("output_format"),
        omni_reference_task_type=None if task_type == "auto" else task_type,
    )


_SEEDANCE2_PRICE_EXPR_TEMPLATE = """
(
  $m := widgets.model;
  $res := $lookup(widgets, "model.resolution");
  $ratio := $lookup(widgets, "model.ratio");
  $dur := $lookup(widgets, "model.duration");
  $auto := __IS_EDIT__;
  $hasVideo := __HAS_VIDEO__;
  $ready := $type($m) = "string" and $type($res) = "string" and ($auto or $type($dur) = "number");
  $ready ? (
    $contains($m, "2.5") ? (
      $is480 := $res = "480p";
      $is1080 := $res = "1080p";
      $perFrame := $ratio = "1:1"  ? ($is480 ? 400      : $is1080 ? 2025      : 900) :
                   $ratio = "4:3"  ? ($is480 ? 411.25   : $is1080 ? 2028      : 905.6719) :
                   $ratio = "3:4"  ? ($is480 ? 411.25   : $is1080 ? 2028      : 905.6719) :
                   $ratio = "21:9" ? ($is480 ? 418.5    : $is1080 ? 2037.9648 : 904.3945) :
                                     ($is480 ? 400.3125 : $is1080 ? 2025      : 900);
      $price := $is1080
        ? ($hasVideo ? 0.01001 : 0.016731)
        : ($hasVideo ? 0.009152 : 0.015301);
      $costFor := function($d) { $floor($perFrame * (24 * $d + 1)) / 1000 * $price };
      $lo := $costFor($auto ? 4 : $dur);
      $hi := $costFor(($auto ? 30 : $dur) + ($hasVideo ? 30 : 0));
      $lo = $hi
        ? {"type": "usd", "usd": $lo, "format": {"approximate": true}}
        : {"type": "range_usd", "min_usd": $lo, "max_usd": $hi, "format": {"approximate": true}}
    ) : (
      $rate := $res = "4k"    ? 195200 :
               $res = "1080p" ? 48800 :
               $res = "720p"  ? 21600 : 10044;
      $noVideoPrice := $res = "4k"    ? 0.00572 :
                       $res = "1080p" ? 0.011011 :
                       $contains($m, "mini") ? 0.005005 :
                       $contains($m, "fast") ? 0.008008 : 0.01001;
      $videoPrice := $res = "4k"    ? 0.003432 :
                     $res = "1080p" ? 0.006721 :
                     $contains($m, "mini") ? 0.003003 :
                     $contains($m, "fast") ? 0.004719 : 0.006149;
      $hasVideo
        ? {"type": "range_usd",
           "min_usd": $ceil($dur * 5 / 3) * $rate * $videoPrice / 1000,
           "max_usd": (15 + $dur) * $rate * $videoPrice / 1000,
           "format": {"approximate": true}}
        : {"type": "usd", "usd": $dur * $rate * $noVideoPrice / 1000,
           "format": {"approximate": true}}
    )
  ) : undefined
)
"""


_SEEDANCE_AUDIO_POLICY_CODE = "OutputAudioSensitiveContentDetected.PolicyViolation"
_SEEDANCE_TASK_TYPE_CONSTRAINT_CODE = "InvalidParameter.TaskTypeConstraint"
_SEEDANCE_TASK_TYPE_MISMATCH_CODE = "InvalidParameter.TaskTypeMismatch"


async def _seedance2_poll_video_task(
    cls: type[IO.ComfyNode],
    task_id: str,
    task_type: str | None = None,
) -> TaskStatusResponse:
    try:
        return await poll_op(
            cls,
            ApiEndpoint(path=f"{BYTEPLUS_SEEDANCE2_TASK_STATUS_ENDPOINT}/{task_id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: r.status,
            poll_interval=9,
        )
    except Exception as exc:
        if _SEEDANCE_AUDIO_POLICY_CODE in str(exc):
            raise ValueError(
                "The provider rejected the audio track this model generated for the video "
                "(possible copyright match). The video itself was fine. Turn off generate_audio "
                "to get a silent video, or adjust the prompt and try again."
            ) from exc
        if _SEEDANCE_TASK_TYPE_CONSTRAINT_CODE in str(exc):
            if task_type is None:
                raise ValueError(
                    "Seedance read this prompt as editing the reference video, and an edit always "
                    "takes its duration and aspect ratio from that video. Enable video_editing on "
                    "this node and run again, or reword the prompt so it describes a new video "
                    "rather than a change to the reference one."
                ) from exc
            if task_type == "edit":
                raise ValueError(
                    "The request does not satisfy the 'edit' constraints: the clip being edited "
                    "must be 4 to 30 seconds long."
                ) from exc
            if task_type == "extend":
                raise ValueError(
                    "The request does not satisfy the 'extend' constraints: the clip being "
                    "extended must be 1.9 to 30 seconds long."
                ) from exc
            raise ValueError(
                "Seedance decided from the prompt that this task's duration or aspect ratio "
                "must come from the reference video, and the current settings conflict with "
                "that. Set task_type to the task you mean ('edit' or 'extend') and run again, "
                "or reword the prompt so it describes a new video rather than a change to the "
                "reference one."
            ) from exc
        if _SEEDANCE_TASK_TYPE_MISMATCH_CODE in str(exc):
            raise ValueError(
                f"Seedance read this prompt as a different task than the selected task_type "
                f"'{task_type}'. Reword the prompt so it matches: an extend prompt should say "
                "'extend forward', 'extend backward', or 'continue'; an edit prompt should use "
                "words like add, remove, replace, or change. Or set task_type to auto."
            ) from exc
        raise


def _seedance2_price_badge(with_reference_videos: bool, legacy_video_editing: bool = False) -> IO.PriceBadge:
    widgets = ["model", "model.resolution", "model.ratio", "model.duration"]
    if legacy_video_editing:
        is_edit = '$lookup(widgets, "model.video_editing") = true'
    else:
        is_edit = '$lookup(widgets, "model.task_type") = "edit"'
    if with_reference_videos:
        widgets.append("model.video_editing" if legacy_video_editing else "model.task_type")
    has_video = (
        '$exists(inputGroups) and $lookup(inputGroups, "model.reference_videos") > 0'
        if with_reference_videos
        else "false"
    )
    return IO.PriceBadge(
        depends_on=IO.PriceBadgeDepends(
            widgets=widgets,
            input_groups=["model.reference_videos"] if with_reference_videos else [],
        ),
        expr=_SEEDANCE2_PRICE_EXPR_TEMPLATE.replace("__HAS_VIDEO__", has_video).replace("__IS_EDIT__", is_edit),
    )


class ByteDance2TextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDance2TextToVideoNode",
            display_name="ByteDance Seedance 2.5 Text to Video",
            category="partner/video/ByteDance",
            description="Generate video using Seedance 2.5 or 2.0 models based on a text prompt.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("Seedance 2.5", _seedance25_text_inputs()),
                        IO.DynamicCombo.Option("Seedance 2.0", _seedance2_text_inputs(["480p", "720p", "1080p", "4k"])),
                        IO.DynamicCombo.Option("Seedance 2.0 Fast", _seedance2_text_inputs(["480p", "720p"])),
                        IO.DynamicCombo.Option("Seedance 2.0 Mini", _seedance2_text_inputs(["480p", "720p"])),
                    ],
                    tooltip=SEEDANCE_MODEL_TOOLTIP,
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
            price_badge=_seedance2_price_badge(with_reference_videos=False),
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
            data=_seedance2_build_request(
                model,
                model_id,
                [TaskTextContent(text=model["prompt"])],
                seed,
                watermark,
                ratio=model["ratio"],
            ),
            response_model=TaskCreationResponse,
        )
        response = await _seedance2_poll_video_task(cls, initial_response.id)
        return IO.NodeOutput(await download_url_to_video_output(response.content.video_url))


class ByteDance2FirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDance2FirstLastFrameNode",
            display_name="ByteDance Seedance 2.5 First-Last-Frame to Video",
            category="partner/video/ByteDance",
            description="Generate video using Seedance 2.5 or 2.0 from a first frame image "
            "and optional last frame image.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("Seedance 2.5", _seedance25_text_inputs(with_ratio=False)),
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
                    tooltip=SEEDANCE_MODEL_TOOLTIP,
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
            price_badge=_seedance2_price_badge(with_reference_videos=False),
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

        if model_id == "dreamina-seedance-2-5-260628":
            # 2.5 accepts ratio="adaptive" only here and keeps the first frame's own aspect
            # (a 1920x1088 frame yields 850x482, not a grid ratio), so pre-sizing the frames to a
            # supported pixel pair would crop framing the model would otherwise have preserved.
            request_ratio = "adaptive"
            if first_frame is not None:
                first_frame = _prepare_seedance_image(first_frame)
            if last_frame is not None:
                last_frame = _prepare_seedance_image(last_frame)
        elif first_frame_asset_id or last_frame_asset_id:
            request_ratio = model["ratio"]
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
            data=_seedance2_build_request(model, model_id, content, seed, watermark, ratio=request_ratio),
            response_model=TaskCreationResponse,
        )
        response = await _seedance2_poll_video_task(cls, initial_response.id)
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


class ByteDance2ReferenceNodeV2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDance2ReferenceNodeV2",
            display_name="ByteDance Seedance 2.5 Reference to Video",
            category="partner/video/ByteDance",
            description="Generate, edit, or extend video using Seedance 2.5 or 2.0 with reference "
            "images, videos, and audio. Supports multimodal reference, video editing, and video extension.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("Seedance 2.5", _seedance25_reference_inputs(with_task_type=True)),
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
                    tooltip=SEEDANCE_MODEL_TOOLTIP,
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
            price_badge=_seedance2_price_badge(with_reference_videos=True),
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

        model_id = SEEDANCE_MODELS[model["model"]]
        limits = seedance2_reference_limits(model_id)

        if not reference_images and not reference_videos and not reference_image_assets and not reference_video_assets:
            if model_id != "dreamina-seedance-2-5-260628" or not (reference_audios or reference_audio_assets):
                raise ValueError("At least one reference image or video or asset is required.")

        total_images = len(reference_images) + len(reference_image_assets)
        if total_images > limits["max_images"]:
            raise ValueError(
                f"Too many reference images: {total_images} "
                f"(images={len(reference_images)}, image assets={len(reference_image_assets)}). "
                f"Maximum is {limits['max_images']}."
            )
        total_videos = len(reference_videos) + len(reference_video_assets)
        if total_videos > limits["max_videos"]:
            raise ValueError(
                f"Too many reference videos: {total_videos} "
                f"(videos={len(reference_videos)}, video assets={len(reference_video_assets)}). "
                f"Maximum is {limits['max_videos']}."
            )
        task_type = model.get("task_type")
        if task_type in ("edit", "extend") and total_videos == 0:
            raise ValueError(
                f"A '{task_type}' task needs at least one reference video. Connect the video "
                f"you want to {'change' if task_type == 'edit' else 'continue'}, or set "
                "task_type to 'reference' to generate a new video from the references you have."
            )
        total_audios = len(reference_audios) + len(reference_audio_assets)
        if total_audios > limits["max_audios"]:
            raise ValueError(
                f"Too many reference audios: {total_audios} "
                f"(audios={len(reference_audios)}, audio assets={len(reference_audio_assets)}). "
                f"Maximum is {limits['max_audios']}."
            )

        for key in reference_images:
            reference_images[key] = _prepare_seedance_image(reference_images[key])

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
        if total_video_duration > limits["max_total_seconds"]:
            raise ValueError(
                f"Total reference video duration is {total_video_duration:.1f}s. "
                f"Maximum is {limits['max_total_seconds']} seconds."
            )

        total_audio_duration = 0.0
        for i, key in enumerate(reference_audios, 1):
            audio = reference_audios[key]
            dur = int(audio["waveform"].shape[-1]) / int(audio["sample_rate"])
            if dur < 1.8:
                raise ValueError(f"Reference audio {i} is too short: {dur:.1f}s. Minimum duration is 1.8 seconds.")
            total_audio_duration += dur
        if total_audio_duration > limits["max_total_seconds"]:
            raise ValueError(
                f"Total reference audio duration is {total_audio_duration:.1f}s. "
                f"Maximum is {limits['max_total_seconds']} seconds."
            )

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
            data=_seedance2_build_request(model, model_id, content, seed, watermark, ratio=model["ratio"]),
            response_model=TaskCreationResponse,
        )
        response = await _seedance2_poll_video_task(
            cls,
            initial_response.id,
            task_type=task_type,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.content.video_url))


class ByteDance2ReferenceNode(ByteDance2ReferenceNodeV2):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDance2ReferenceNode",
            display_name="ByteDance Seedance 2.5 Reference to Video (Legacy)",
            category="partner/video/ByteDance",
            description="Generate, edit, or extend video using Seedance 2.5 or 2.0 with reference "
            "images, videos, and audio. Supports multimodal reference, video editing, and video extension.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("Seedance 2.5", _seedance25_reference_inputs(with_video_editing=True)),
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
                    tooltip=SEEDANCE_MODEL_TOOLTIP,
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
            is_deprecated=True,
            price_badge=_seedance2_price_badge(with_reference_videos=True, legacy_video_editing=True),
        )


async def process_video_task(
    cls: type[IO.ComfyNode],
    payload: Text2VideoTaskCreationRequest | Image2VideoTaskCreationRequest,
    estimated_duration: int | None,
) -> IO.NodeOutput:
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


MODE_TEXT = "text only"
MODE_AUDIO = "audio reference"
MODE_IMAGE = "image reference"
MODE_SPEAKER = "preset voice"

# (speaker_id, display_label) for built-in TTS 2.0 voices; resolvable ids are account-scoped.
SEED_AUDIO_PRESET_VOICES: list[tuple[str, str]] = [
    ("zh_female_vv_uranus_bigtts", "Vivi (Female, multilingual)"),
    ("zh_female_xiaohe_uranus_bigtts", "Mindy (Female, multilingual)"),
    ("en_female_stokie_uranus_bigtts", "Stokie (Female, English)"),
    ("en_female_dacey_uranus_bigtts", "Dacey (Female, English)"),
    ("en_male_tim_uranus_bigtts", "Tim (Male, English)"),
    ("zh_male_m191_uranus_bigtts", "Kian (Male, multilingual)"),
    ("zh_male_taocheng_uranus_bigtts", "Cedric (Male, multilingual)"),
    ("zh_male_sophie_uranus_bigtts", "Sophie (Female, multilingual)"),
    ("zh_female_yingyujiaoxue_uranus_bigtts", "Jean (Female, multilingual)"),
    ("zh_male_dayi_uranus_bigtts", "Magnus (Male, multilingual)"),
    ("zh_female_mizai_uranus_bigtts", "Mabel (Female, multilingual)"),
    ("zh_female_jitangnv_uranus_bigtts", "Nadia (Female, multilingual)"),
    ("zh_female_meilinvyou_uranus_bigtts", "Opal (Female, multilingual)"),
    ("zh_female_liuchangnv_uranus_bigtts", "Pearl (Female, multilingual)"),
    ("zh_male_ruyayichen_uranus_bigtts", "Quentin (Male, multilingual)"),
    ("zh_female_vivo_uranus_bigtts", "Vienna (Female, multilingual)"),
    ("zh_female_xiaoai_uranus_bigtts", "Alina (Female, multilingual)"),
    ("zh_female_cancan_uranus_bigtts", "Corinne (Female, multilingual)"),
    ("zh_female_tianmeixiaoyuan_uranus_bigtts", "Esther (Female, multilingual)"),
    ("zh_female_tianmeitaozi_uranus_bigtts", "Freya (Female, multilingual)"),
    ("zh_female_shuangkuaisisi_uranus_bigtts", "Gigi (Female, multilingual)"),
    ("zh_female_peiqi_uranus_bigtts", "Holly (Female, multilingual)"),
    ("zh_female_xiaoxue_uranus_bigtts", "Lyla (Female, multilingual)"),
    ("zh_female_yuanqi_uranus_bigtts", "Daisy (Female, multilingual)"),
    ("zh_female_kefunvsheng_uranus_bigtts", "Tracy (Female, multilingual)"),
    ("zh_male_shaonianzixin_uranus_bigtts", "Jess (Male, multilingual)"),
    ("zh_female_linjianvhai_uranus_bigtts", "Pinky (Female, multilingual)"),
    ("zh_female_kiwi_uranus_bigtts", "Sweety (Female, multilingual)"),
    ("zh_female_sajiaoxuemei_uranus_bigtts", "Sandy (Female, multilingual)"),
    ("de_male_seven_uranus_bigtts", "Sven (Male, German)"),
    ("jp_female_minimi_uranus_bigtts", "Minimi (Female, Japanese)"),
    ("fr_male_usseau_uranus_bigtts", "Usseau (Male, French)"),
    ("es_male_felipe_uranus_bigtts", "Felipe (Male, Spanish)"),
    ("id_male_han_uranus_bigtts", "Han (Male, Indonesian)"),
    ("pt_male_martins_uranus_bigtts", "Martins (Male, Portuguese)"),
    ("it_male_enzo_uranus_bigtts", "Enzo (Male, Italian)"),
    ("kr_male_shane_uranus_bigtts", "Shane (Male, Korean)"),
    ("zh_male_liufei_uranus_bigtts", "Felix (Male, Chinese)"),
    ("zh_female_qingxinnvsheng_uranus_bigtts", "Celeste (Female, Chinese)"),
    ("zh_male_sunwukong_uranus_bigtts", "Monkey King (Male, Chinese)"),
]
SEED_AUDIO_VOICE_OPTIONS = [label for _, label in SEED_AUDIO_PRESET_VOICES]
SEED_AUDIO_VOICE_MAP = {label: speaker_id for speaker_id, label in SEED_AUDIO_PRESET_VOICES}

_AUDIO_TAG_RE = re.compile(r"@Audio(\d+)", re.IGNORECASE)


def max_audio_tag(prompt: str) -> int:
    """Highest N referenced as @AudioN in the prompt (0 if none)."""
    nums = [int(m) for m in _AUDIO_TAG_RE.findall(prompt or "")]
    return max(nums) if nums else 0


def connected_audio_indices(reference_mode: dict) -> list[int]:
    """Indices (1-based) of connected reference_audio sockets, in order."""
    return [
        i
        for i in range(1, 3 + 1)
        if reference_mode.get(f"reference_audio_{i}") is not None
    ]


def validate_seed_audio_inputs(
    text_prompt: str,
    mode: str,
    audio_indices: list[int],
    has_image: bool,
    preset_voice: str | None = None,
) -> None:
    validate_string(text_prompt, field_name="text_prompt", min_length=1, max_length=3000)
    max_tag = max_audio_tag(text_prompt)

    if mode == MODE_TEXT:
        if max_tag:
            raise ValueError(
                f"The prompt references @Audio{max_tag}, but reference mode is '{MODE_TEXT}'. "
                f"Switch to '{MODE_AUDIO}' and connect the reference clip(s)."
            )
    elif mode == MODE_AUDIO:
        if not audio_indices:
            raise ValueError(
                f"Reference mode '{MODE_AUDIO}' requires at least one reference_audio input "
                f"(or switch to '{MODE_TEXT}')."
            )
        if audio_indices != list(range(1, len(audio_indices) + 1)):
            raise ValueError(
                "Connect reference_audio inputs in order without gaps: reference_audio_1, then _2, then _3."
            )
        if max_tag > len(audio_indices):
            raise ValueError(
                f"The prompt references @Audio{max_tag}, but only {len(audio_indices)} "
                f"reference audio(s) are connected."
            )
    elif mode == MODE_IMAGE:
        if not has_image:
            raise ValueError(f"Reference mode '{MODE_IMAGE}' requires a reference_image input.")
        if max_tag:
            raise ValueError(
                f"@AudioN tags are not used in '{MODE_IMAGE}' mode; the prompt should contain "
                f"only the text to synthesize."
            )
    elif mode == MODE_SPEAKER:
        if not preset_voice or preset_voice not in SEED_AUDIO_VOICE_MAP:
            raise ValueError(f"Reference mode '{MODE_SPEAKER}' requires selecting a preset voice.")
        if max_tag > 1:
            raise ValueError(
                f"'{MODE_SPEAKER}' mode uses a single voice, so @Audio{max_tag} is out of range. "
                f"Remove the @AudioN tags — the whole prompt is read in the selected voice."
            )
    else:
        raise ValueError(f"Unknown reference mode: {mode!r}")


class ByteDanceSeedAudioNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ByteDanceSeedAudio",
            display_name="ByteDance Seed Audio 1.0",
            category="partner/audio/ByteDance",
            description=(
                "Generate speech, music, sound effects and multi-speaker dialogue from a single prompt "
                "with ByteDance Seed Audio 1.0. Describe the voice(s), emotion, ambience, background music "
                "and sound effects in the prompt, and include the lines to speak. Optionally pick a built-in "
                "preset voice, clone voices from up to 3 reference clips (tagged @Audio1-3 in the prompt), "
                "or derive a voice from a character image. Up to 2 minutes of audio per run. "
                "The multilingual model supports 20 languages and timestamp-based timing control."
            ),
            inputs=[
                IO.String.Input(
                    "text_prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Describe the voice(s), emotion, pacing, ambience, background music and sound "
                        "effects, and include the lines to speak (name characters inline for dialogue). "
                        "In 'audio reference' mode, refer to connected clips by order as @Audio1, @Audio2, "
                        "@Audio3. With the multilingual model, a quoted line can start with a timestamp "
                        'range that controls when and how long it is spoken, e.g. "[5.5s:8.0s] Wait for me!". '
                        "Write the prompt in the same language as the lines to speak. Maximum 3000 characters."
                    ),
                ),
                IO.DynamicCombo.Input(
                    "reference_mode",
                    options=[
                        IO.DynamicCombo.Option(MODE_TEXT, []),
                        IO.DynamicCombo.Option(
                            MODE_AUDIO,
                            [
                                IO.Audio.Input(
                                    "reference_audio_1",
                                    optional=True,
                                    tooltip="Reference clip for voice cloning, tagged @Audio1 in the prompt. "
                                    "Up to 30s.",
                                ),
                                IO.Audio.Input(
                                    "reference_audio_2",
                                    optional=True,
                                    tooltip="Reference clip tagged @Audio2 in the prompt. Up to 30s.",
                                ),
                                IO.Audio.Input(
                                    "reference_audio_3",
                                    optional=True,
                                    tooltip="Reference clip tagged @Audio3 in the prompt. Up to 30s.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            MODE_IMAGE,
                            [
                                IO.Image.Input(
                                    "reference_image",
                                    optional=True,
                                    tooltip="A single character image; the model derives a voice from it. "
                                    "Cannot be combined with reference audio.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            MODE_SPEAKER,
                            [
                                IO.Combo.Input(
                                    "preset_voice",
                                    options=SEED_AUDIO_VOICE_OPTIONS,
                                    default=SEED_AUDIO_VOICE_OPTIONS[0],
                                    tooltip="A built-in TTS 2.0 voice that reads the prompt. No reference "
                                    "clip needed, and @AudioN tags are not used in this mode.",
                                ),
                            ],
                        ),
                    ],
                    tooltip=(
                        "How to condition the voice: 'text only' (describe everything in the prompt), "
                        "'audio reference' (clone up to 3 voices, tagged @Audio1-3), 'image reference' "
                        "(derive a voice from one character image), or 'preset voice' (pick a built-in "
                        "named voice that reads the prompt)."
                    ),
                ),
                IO.Combo.Input(
                    "sample_rate",
                    options=["8000", "16000", "24000", "32000", "44100", "48000"],
                    default="24000",
                    tooltip="Output sample rate in Hz.",
                ),
                IO.Int.Input(
                    "speech_rate",
                    default=0,
                    min=-50,
                    max=100,
                    tooltip="Speaking speed. 0 = normal, 100 = 2.0x, -50 = 0.5x.",
                ),
                IO.Int.Input(
                    "loudness_rate",
                    default=0,
                    min=-50,
                    max=100,
                    tooltip="Loudness. 0 = normal, 100 = 2.0x, -50 = 0.5x.",
                ),
                IO.Int.Input(
                    "pitch_rate",
                    default=0,
                    min=-12,
                    max=12,
                    tooltip="Pitch shift in semitones (-12 to 12).",
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
                IO.Combo.Input(
                    "model",
                    options=["seed-audio-1.0-multilingual", "seed-audio-1.0"],
                    default="seed-audio-1.0-multilingual",
                    optional=True,
                    tooltip=(
                        "seed-audio-1.0-multilingual: 20 languages (English, Chinese, Japanese, Korean, "
                        "Mexican & Castilian Spanish, Indonesian, German, Brazilian Portuguese, French, "
                        "Thai, Vietnamese, Malay, Filipino, Italian, Russian, Dutch, Polish, Turkish, "
                        'Swedish) plus per-sentence timing control via "[5.5s:8.0s] ..." timestamps. '
                        "seed-audio-1.0: English and Chinese only, no timing control."
                    ),
                ),
            ],
            outputs=[IO.Audio.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd": 0.2145, "format":{"suffix":"/minute","approximate":true}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        text_prompt: str,
        reference_mode: dict,
        sample_rate: str,
        speech_rate: int,
        loudness_rate: int,
        pitch_rate: int,
        seed: int,
        model: str = "seed-audio-1.0-multilingual",
    ) -> IO.NodeOutput:
        mode = reference_mode["reference_mode"]
        audio_indices = connected_audio_indices(reference_mode)
        image = reference_mode.get("reference_image")
        preset_voice = reference_mode.get("preset_voice")
        validate_seed_audio_inputs(text_prompt, mode, audio_indices, image is not None, preset_voice)

        references: list[SeedAudioReference] | None = None
        if mode == MODE_AUDIO:
            references = []
            for i in audio_indices:
                clip = reference_mode[f"reference_audio_{i}"]
                validate_audio_duration(clip, max_duration=30.0)
                mp3_bytes = audio_input_to_mp3(clip).getvalue()
                references.append(SeedAudioReference(audio_data=base64.b64encode(mp3_bytes).decode("utf-8")))
        elif mode == MODE_IMAGE:
            image = upscale_image_tensor_to_min_pixels(image, 160_000)
            references = [SeedAudioReference(image_data=tensor_to_base64_string(image, mime_type="image/png"))]
        elif mode == MODE_SPEAKER:
            references = [SeedAudioReference(speaker=SEED_AUDIO_VOICE_MAP[preset_voice])]

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/byteplus/api/v3/tts/create", method="POST"),
            response_model=SeedAudioResponse,
            data=SeedAudioRequest(
                model=model,
                text_prompt=text_prompt,
                references=references,
                audio_config=SeedAudioConfig(
                    sample_rate=int(sample_rate),
                    speech_rate=speech_rate,
                    loudness_rate=loudness_rate,
                    pitch_rate=pitch_rate,
                ),
            ),
        )
        if not response.audio:
            raise Exception(
                f"Seed Audio returned no audio (code={response.code}): {response.message}"
            )
        return IO.NodeOutput(audio_bytes_to_audio_input(base64.b64decode(response.audio)))


_VCUBE_ENHANCE_VIDEO_ENDPOINT = ApiEndpoint(path="/proxy/byteplusmediakit/api/v1/tools/enhance-video", method="POST")
_VCUBE_TASK_ENDPOINT_PREFIX = "/proxy/byteplusmediakit/api/v1/tasks/"

_VCUBE_MAX_DURATION_SECONDS = 600
_VCUBE_MIN_FPS = 15.0
_VCUBE_MAX_FPS = 120.0
_VCUBE_MIN_SHORT_SIDE = 128
_VCUBE_MAX_SHORT_SIDE = 4320
_VCUBE_MAX_INPUT_SHORT_SIDE = 1440
_VCUBE_MAX_INPUT_LONG_SIDE = 2560
_VCUBE_RESOLUTION_PRESETS = ["1080p", "720p", "2k", "4k", "8k"]
_VCUBE_FPS_PRESETS = ["source", "24", "25", "30", "48", "50", "60", "120"]


class ByteDanceVideoEnhanceNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ByteDanceVideoEnhanceNode",
            display_name="ByteDance vCube Video Enhance",
            category="partner/video/ByteDance",
            description="Upscales and restores a video with ByteDance vCube: super-resolution up to 8K, "
            "compression artifact and noise removal, colour and sharpness enhancement, "
            "optional frame interpolation.",
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="Video to enhance. The source resolution must be at most 2560x1440 (2K); "
                    "the output size is set by the resolution input.",
                ),
                IO.DynamicCombo.Input(
                    "tool_version",
                    options=[
                        IO.DynamicCombo.Option(
                            "standard",
                            [
                                IO.Combo.Input(
                                    "scene",
                                    options=["aigc", "common", "ugc", "short_series", "old_film"],
                                    default="aigc",
                                    tooltip="Preset tuned to the content: 'aigc' for AI-generated footage, "
                                    "'common' for general video, 'ugc' for compressed phone clips, "
                                    "'short_series' for drama with faces, 'old_film' for scratched or "
                                    "flickering archive footage.",
                                ),
                                IO.Combo.Input(
                                    "enhance_style",
                                    options=["hd", "natural"],
                                    default="hd",
                                    tooltip="'hd' applies a sharper enhancement; 'natural' reduces the strength "
                                    "for a softer, less sharpened look.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "professional",
                            [
                                IO.Combo.Input(
                                    "enhance_style",
                                    options=["hd", "natural"],
                                    default="hd",
                                    tooltip="'hd' applies a sharper enhancement; 'natural' reduces the strength "
                                    "for a softer, less sharpened look.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="'standard' balances speed and quality with 10+ enhancement algorithms. "
                    "'professional' uses 30+ algorithms for cinema-grade restoration, takes about "
                    "3x longer and costs 10x more.",
                ),
                IO.DynamicCombo.Input(
                    "resolution",
                    options=[
                        *[IO.DynamicCombo.Option(preset, []) for preset in _VCUBE_RESOLUTION_PRESETS],
                        IO.DynamicCombo.Option("source", []),
                        IO.DynamicCombo.Option(
                            "custom",
                            [
                                IO.Int.Input(
                                    "short_side",
                                    default=1080,
                                    min=_VCUBE_MIN_SHORT_SIDE,
                                    max=_VCUBE_MAX_SHORT_SIDE,
                                    tooltip="Short side of the output in pixels; the long side follows "
                                    "the source aspect ratio.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Output resolution. The short side is set to the chosen level and the long side "
                    "follows the source aspect ratio. 'source' keeps the source size, 'custom' sets "
                    "the short side in pixels. Sources wider or taller than about 2.2:1 are billed one "
                    "resolution tier higher.",
                ),
                IO.Combo.Input(
                    "fps",
                    options=_VCUBE_FPS_PRESETS,
                    default="source",
                    tooltip="Output frame rate. A higher rate than the source enables AI frame interpolation; "
                    "a lower one drops frames. 'source' keeps the source rate, up to 120 fps. "
                    "Rates above 30 fps cost 2x, above 60 fps 4x.",
                ),
                IO.Combo.Input(
                    "bitrate_level",
                    options=["low", "medium", "high"],
                    default="medium",
                    advanced=True,
                    tooltip="Target bitrate of the delivered file, scaled to the output resolution and frame rate.",
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
                depends_on=IO.PriceBadgeDepends(
                    widgets=["tool_version", "resolution", "resolution.short_side", "fps"],
                ),
                expr="""
                (
                  $tv := $lookup(widgets, "tool_version");
                  $res := $lookup(widgets, "resolution");
                  $fps := $lookup(widgets, "fps");
                  $tiers := {"720p": 1, "1080p": 2, "2k": 4, "4k": 8, "8k": 32};
                  $tier := $res = "custom"
                    ? ($s := $number($lookup(widgets, "resolution.short_side"));
                       $s < 1080 ? 1 : $s < 1440 ? 2 : $s < 2160 ? 4 : $s < 4320 ? 8 : 32)
                    : $lookup($tiers, $res);
                  $fpsMul := $fps = "source" ? 1 : ($number($fps) <= 30 ? 1 : ($number($fps) <= 60 ? 2 : 4));
                  $base := 0.2066 * 1.43 / 60 * ($tv = "professional" ? 10 : 1);
                  $min := $base * ($res = "source" ? 1 : $tier) * ($fps = "source" ? 1 : $fpsMul);
                  $max := $base * ($res = "source" ? 4 : $tier) * ($fps = "source" ? 4 : $fpsMul);
                  $min = $max
                    ? {"type": "usd", "usd": $min, "format": {"suffix": "/second"}}
                    : {"type": "range_usd", "min_usd": $min, "max_usd": $max,
                       "format": {"approximate": true, "suffix": "/second",
                                  "note": $res = "source"
                                    ? ($fps = "source" ? "(by source size and frame rate)" : "(720p-2K, by source size)")
                                    : "(by source frame rate)"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        tool_version: dict,
        resolution: dict,
        fps: str,
        bitrate_level: str,
    ) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=_VCUBE_MAX_DURATION_SECONDS)
        width, height = video.get_dimensions()
        if min(width, height) > _VCUBE_MAX_INPUT_SHORT_SIDE or max(width, height) > _VCUBE_MAX_INPUT_LONG_SIDE:
            raise ValueError(
                f"Video resolution must be at most {_VCUBE_MAX_INPUT_LONG_SIDE}x{_VCUBE_MAX_INPUT_SHORT_SIDE} "
                f"(2K), got {width}x{height}. Scale the video down before enhancing it."
            )
        if fps == "source":
            source_fps = float(video.get_frame_rate())
            output_fps = round(min(source_fps, _VCUBE_MAX_FPS), 3) if source_fps >= _VCUBE_MIN_FPS else None
        else:
            output_fps = float(fps)
        target = resolution["resolution"]
        resolution_preset = target if target in _VCUBE_RESOLUTION_PRESETS else None
        short_side = None
        if target == "custom":
            short_side = resolution["short_side"]
        elif target == "source" and min(width, height) >= _VCUBE_MIN_SHORT_SIDE:
            short_side = min(width, height)
        url = await upload_video_to_comfyapi(cls, video, wait_label="Uploading source video")
        request = MediaKitVideoEnhanceRequest(
            video_url=url,
            tool_version=tool_version["tool_version"],
            scene=tool_version.get("scene"),
            enhance_style=tool_version.get("enhance_style"),
            resolution=resolution_preset,
            resolution_limit=short_side,
            fps=output_fps,
            bitrate_level=bitrate_level,
        )
        created = await sync_op(
            cls, _VCUBE_ENHANCE_VIDEO_ENDPOINT, response_model=MediaKitTaskCreateResponse, data=request
        )
        if not created.success or not created.task_id:
            error = created.error
            raise ValueError(
                f"{error.code}: {error.message}" if error and error.message else "Task submission failed."
            )
        task = await poll_op(
            cls,
            ApiEndpoint(path=_VCUBE_TASK_ENDPOINT_PREFIX + created.task_id, method="GET"),
            response_model=MediaKitTaskResponse,
            status_extractor=lambda r: r.status,
            completed_statuses=["completed"],
            failed_statuses=["failed"],
            queued_statuses=[],
            poll_interval=10.0,
            max_poll_attempts=2000,
        )
        return IO.NodeOutput(await download_url_to_video_output(task.result.video_url))


class ByteDanceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ByteDanceSeedreamNode,
            ByteDanceSeedreamNodeV2,
            ByteDanceSeedreamNodeV3,
            ByteDanceSeedreamLayerSeparationNode,
            ByteDanceTextToVideoNode,
            ByteDanceImageToVideoNode,
            ByteDanceFirstLastFrameNode,
            ByteDance2TextToVideoNode,
            ByteDance2FirstLastFrameNode,
            ByteDance2ReferenceNode,
            ByteDance2ReferenceNodeV2,
            ByteDanceCreateImageAsset,
            ByteDanceCreateVideoAsset,
            ByteDanceSeedAudioNode,
            ByteDanceVideoEnhanceNode,
        ]


async def comfy_entrypoint() -> ByteDanceExtension:
    return ByteDanceExtension()
