"""Krea image-generation nodes."""

import re

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.krea import (
    KreaAssetResponse,
    KreaGenerateImageRequest,
    KreaImageStyleReference,
    KreaJob,
    KreaMoodboard,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    poll_op,
    sync_op,
    tensor_to_bytesio,
    validate_string,
)


class KreaIO:
    STYLE_REF = "KREA_STYLE_REF"


async def _upload_image_to_krea_assets(cls: type[IO.ComfyNode], image: Input.Image) -> str:
    """Upload an image to Krea's /assets endpoint and return the Krea-hosted image URL."""
    img_io = tensor_to_bytesio(image, total_pixels=2048 * 2048, mime_type="image/png")
    response = await sync_op(
        cls,
        endpoint=ApiEndpoint(path="/proxy/krea/assets", method="POST"),
        response_model=KreaAssetResponse,
        files=[("file", (img_io.name, img_io, "image/png"))],
        content_type="multipart/form-data",
        max_retries=1,
        wait_label="Uploading reference",
    )
    return response.image_url


_MODEL_MEDIUM = "Krea 2 Medium"
_MODEL_MEDIUM_TURBO = "Krea 2 Medium Turbo"
_MODEL_LARGE = "Krea 2 Large"
_MODEL_ENDPOINTS: dict[str, str] = {
    _MODEL_MEDIUM: "/proxy/krea/generate/image/krea/krea-2/medium",
    _MODEL_MEDIUM_TURBO: "/proxy/krea/generate/image/krea/krea-2/medium-turbo",
    _MODEL_LARGE: "/proxy/krea/generate/image/krea/krea-2/large",
}

_ASPECT_RATIOS = ["1:1", "4:3", "3:2", "16:9", "2.35:1", "4:5", "2:3", "9:16"]
_RESOLUTIONS = ["1K"]
_CREATIVITY_LEVELS = ["raw", "low", "medium", "high"]
_KREA_QUEUED_STATUSES = ["backlogged", "queued", "scheduled"]

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _krea_model_inputs() -> list:
    """Nested inputs shared by Krea 2 Medium, Medium Turbo and Large under the DynamicCombo."""
    return [
        IO.Combo.Input(
            "aspect_ratio",
            options=_ASPECT_RATIOS,
            tooltip="Output aspect ratio.",
        ),
        IO.Combo.Input(
            "resolution",
            options=_RESOLUTIONS,
            tooltip="Resolution scale.",
        ),
        IO.Combo.Input(
            "creativity",
            options=_CREATIVITY_LEVELS,
            default="medium",
            tooltip="Prompt interpretation strength: raw stays closest to the prompt; high is most creative.",
        ),
        IO.String.Input(
            "moodboard_id",
            default="",
            tooltip="Optional Krea moodboard UUID (e.g. from the Krea website). "
            "Leave empty to disable. Only one moodboard is supported per request.",
            optional=True,
        ),
        IO.Float.Input(
            "moodboard_strength",
            default=0.35,
            min=-0.5,
            max=1.5,
            step=0.05,
            tooltip="Moodboard influence; ignored when moodboard_id is empty.",
            optional=True,
        ),
        IO.Custom(KreaIO.STYLE_REF).Input(
            "style_reference",
            optional=True,
            tooltip="Optional chain of style references (max 10) from Krea 2 Style Reference nodes.",
        ),
    ]


class Krea2ImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Krea2ImageNode",
            display_name="Krea 2 Image",
            category="partner/image/Krea",
            description=(
                "Generate images via Krea 2 — pick Medium (expressive illustrations) or "
                "Large (expressive photorealism). Supports an optional moodboard and up "
                "to 10 chained image style references."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the image.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(_MODEL_MEDIUM, _krea_model_inputs()),
                        IO.DynamicCombo.Option(_MODEL_MEDIUM_TURBO, _krea_model_inputs()),
                        IO.DynamicCombo.Option(_MODEL_LARGE, _krea_model_inputs()),
                    ],
                    tooltip="Krea 2 Medium is best for expressive illustrations; "
                    "Krea 2 Large is best for expressive photorealism.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Random seed for reproducibility.",
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["model", "model.moodboard_id"],
                    inputs=["model.style_reference"],
                ),
                expr="""
                (
                  $rates := {
                    "krea 2 medium turbo": {"text": 0.015, "style": 0.0175, "moodboard": 0.02},
                    "krea 2 medium": {"text": 0.03, "style": 0.035, "moodboard": 0.04},
                    "krea 2 large": {"text": 0.06, "style": 0.065, "moodboard": 0.07}
                  };
                  $r := $lookup($rates, widgets.model);
                  $hasMoodboard := $length($lookup(widgets, "model.moodboard_id")) > 0;
                  $hasStyle := $lookup(inputs, "model.style_reference").connected;
                  $usd := $hasMoodboard ? $r.moodboard : ($hasStyle ? $r.style : $r.text);
                  {"type":"usd","usd": $usd}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=1)

        model_choice = model["model"]
        endpoint_path = _MODEL_ENDPOINTS.get(model_choice)
        if endpoint_path is None:
            raise ValueError(f"Unknown Krea 2 model: {model_choice!r}")

        moodboards: list[KreaMoodboard] | None = None
        mb_id = (model.get("moodboard_id") or "").strip()
        if mb_id:
            if not _UUID_RE.match(mb_id):
                raise ValueError(f"moodboard_id must be a UUID (received {mb_id!r}); copy it from the Krea website.")
            mb_strength = model.get("moodboard_strength")
            moodboards = [KreaMoodboard(id=mb_id, strength=0.35 if mb_strength is None else float(mb_strength))]

        style_reference = model.get("style_reference")
        image_style_references: list[KreaImageStyleReference] | None = None
        if style_reference:
            if len(style_reference) > 10:
                raise ValueError(f"Krea 2 accepts at most 10 image_style_references; received {len(style_reference)}.")
            image_style_references = [
                KreaImageStyleReference(url=ref["url"], strength=float(ref["strength"])) for ref in style_reference
            ]
        initial = await sync_op(
            cls,
            ApiEndpoint(path=endpoint_path, method="POST"),
            response_model=KreaJob,
            data=KreaGenerateImageRequest(
                prompt=prompt,
                aspect_ratio=model["aspect_ratio"],
                resolution=model["resolution"],
                seed=seed,
                creativity=model["creativity"],
                moodboards=moodboards,
                image_style_references=image_style_references,
            ),
        )
        job = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/krea/jobs/{initial.job_id}", method="GET"),
            response_model=KreaJob,
            status_extractor=lambda r: r.status,
            queued_statuses=_KREA_QUEUED_STATUSES,
        )
        if not job.result or not job.result.urls:
            raise RuntimeError(f"Krea 2 job {job.job_id} completed without any image URLs.")
        image = await download_url_to_image_tensor(job.result.urls[0])
        return IO.NodeOutput(image)


class Krea2StyleReferenceNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Krea2StyleReferenceNode",
            display_name="Krea 2 Style Reference",
            category="partner/image/Krea",
            description=(
                "Add an image style reference to a Krea 2 generation. Chain multiple Krea 2 "
                "Style Reference nodes (max 10) and feed the final `style_reference` output "
                "into Krea 2 Image. Each image is uploaded to ComfyAPI storage and passed as URL."
            ),
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Reference image whose style influences the generation.",
                ),
                IO.Float.Input(
                    "strength",
                    default=1.0,
                    min=-2.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Reference strength; negative values invert the style influence.",
                ),
                IO.Custom(KreaIO.STYLE_REF).Input(
                    "style_reference",
                    optional=True,
                    tooltip="Optional incoming chain of style references; this node appends one more.",
                ),
            ],
            outputs=[IO.Custom(KreaIO.STYLE_REF).Output(display_name="style_reference")],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        strength: float,
        style_reference: list[dict] | None = None,
    ) -> IO.NodeOutput:
        chain: list[dict] = list(style_reference) if style_reference else []
        if len(chain) >= 10:
            raise ValueError("Krea 2 accepts at most 10 image_style_references in one generation.")
        url = await _upload_image_to_krea_assets(cls, image)
        chain.append({"url": url, "strength": float(strength)})
        return IO.NodeOutput(chain)


class KreaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Krea2ImageNode,
            Krea2StyleReferenceNode,
        ]


async def comfy_entrypoint() -> KreaExtension:
    return KreaExtension()
