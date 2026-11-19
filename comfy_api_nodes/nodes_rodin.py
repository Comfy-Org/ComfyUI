"""
ComfyUI X Rodin3D(Deemos) API Nodes

Rodin API docs: https://developer.hyper3d.ai/

"""

import logging
import math
import os
from inspect import cleandoc
from io import BytesIO
from typing import Any

import aiohttp
from PIL import Image
from typing_extensions import override

import folder_paths as comfy_paths
from comfy_api.latest import IO, ComfyExtension, Types
from comfy_api_nodes.apis.rodin import (
    JobStatus,
    Rodin3DCheckStatusRequest,
    Rodin3DCheckStatusResponse,
    Rodin3DDownloadRequest,
    Rodin3DDownloadResponse,
    Rodin3DGen25Request,
    Rodin3DGenerateRequest,
    Rodin3DGenerateResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_bytesio,
    download_url_to_file_3d,
    poll_op,
    sync_op,
    validate_string,
)

COMMON_PARAMETERS = [
    IO.Int.Input(
        "Seed",
        default=0,
        min=0,
        max=65535,
        display_mode=IO.NumberDisplay.number,
        optional=True,
    ),
    IO.Combo.Input("Material_Type", options=["PBR", "Shaded"], default="PBR", optional=True),
    IO.Combo.Input(
        "Polygon_count",
        options=["4K-Quad", "8K-Quad", "18K-Quad", "50K-Quad", "200K-Triangle"],
        default="18K-Quad",
        optional=True,
    ),
]


_QUALITY_MESH_OPTIONS: dict[str, tuple[str, int]] = {
    "4K-Quad":       ("Quad", 4000),
    "8K-Quad":       ("Quad", 8000),
    "18K-Quad":      ("Quad", 18000),
    "50K-Quad":      ("Quad", 50000),
    "200K-Quad":     ("Quad", 200000),
    "2K-Triangle":   ("Raw", 2000),
    "20K-Triangle":  ("Raw", 20000),
    "150K-Triangle": ("Raw", 150000),
    "200K-Triangle": ("Raw", 200000),
    "500K-Triangle": ("Raw", 500000),
    "1M-Triangle":   ("Raw", 1000000),
}


def get_quality_mode(poly_count: str) -> tuple[str, int]:
    """Map a polygon-count preset like '18K-Quad' to (mesh_mode, quality_override).

    Falls back to ('Quad', 18000) for unknown labels; legacy parity.
    """
    return _QUALITY_MESH_OPTIONS.get(poly_count, ("Quad", 18000))


def tensor_to_filelike(tensor, max_pixels: int = 2048 * 2048):
    """
    Converts a PyTorch tensor to a file-like object.

    Args:
    - tensor (torch.Tensor): A tensor representing an image of shape (H, W, C)
      where C is the number of channels (3 for RGB), H is height, and W is width.

    Returns:
    - io.BytesIO: A file-like object containing the image data.
    """
    array = tensor.cpu().numpy()
    array = (array * 255).astype("uint8")
    image = Image.fromarray(array, "RGB")

    original_width, original_height = image.size
    original_pixels = original_width * original_height
    if original_pixels > max_pixels:
        scale = math.sqrt(max_pixels / original_pixels)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    else:
        new_width, new_height = original_width, original_height

    if new_width != original_width or new_height != original_height:
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")  # PNG is used for lossless compression
    img_byte_arr.seek(0)
    return img_byte_arr


async def create_generate_task(
    cls: type[IO.ComfyNode],
    images=None,
    seed=1,
    material="PBR",
    quality_override=18000,
    tier="Regular",
    mesh_mode="Quad",
    ta_pose: bool = False,
):
    if images is None:
        raise Exception("Rodin 3D generate requires at least 1 image.")
    if len(images) > 5:
        raise Exception("Rodin 3D generate requires up to 5 image.")

    response = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/rodin/api/v2/rodin", method="POST"),
        response_model=Rodin3DGenerateResponse,
        data=Rodin3DGenerateRequest(
            seed=seed,
            tier=tier,
            material=material,
            quality_override=quality_override,
            mesh_mode=mesh_mode,
            TAPose=ta_pose,
        ),
        files=[
            ("images", open(image, "rb") if isinstance(image, str) else tensor_to_filelike(image))
            for image in images
            if image is not None
        ],
        content_type="multipart/form-data",
    )

    if hasattr(response, "error"):
        error_message = f"Rodin3D Create 3D generate Task Failed. Message: {response.message}, error: {response.error}"
        logging.error(error_message)
        raise Exception(error_message)

    logging.info("[ Rodin3D API - Submit Jobs ] Submit Generate Task Success!")
    subscription_key = response.jobs.subscription_key
    task_uuid = response.uuid
    logging.info("[ Rodin3D API - Submit Jobs ] UUID: %s", task_uuid)
    return task_uuid, subscription_key


def check_rodin_status(response: Rodin3DCheckStatusResponse) -> str:
    all_done = all(job.status == JobStatus.Done for job in response.jobs)
    status_list = [str(job.status) for job in response.jobs]
    logging.info("[ Rodin3D API - CheckStatus ] Generate Status: %s", status_list)
    if any(job.status == JobStatus.Failed for job in response.jobs):
        logging.error("[ Rodin3D API - CheckStatus ] Generate Failed: %s, Please try again.", status_list)
        raise Exception("[ Rodin3D API ] Generate Failed, Please Try again.")
    if all_done:
        return "DONE"
    return "Generating"


def extract_progress(response: Rodin3DCheckStatusResponse) -> int | None:
    if not response.jobs:
        return None
    completed_count = sum(1 for job in response.jobs if job.status == JobStatus.Done)
    return int((completed_count / len(response.jobs)) * 100)


async def poll_for_task_status(subscription_key: str, cls: type[IO.ComfyNode]) -> Rodin3DCheckStatusResponse:
    logging.info("[ Rodin3D API - CheckStatus ] Generate Start!")
    return await poll_op(
        cls,
        ApiEndpoint(path="/proxy/rodin/api/v2/status", method="POST"),
        response_model=Rodin3DCheckStatusResponse,
        data=Rodin3DCheckStatusRequest(subscription_key=subscription_key),
        status_extractor=check_rodin_status,
        progress_extractor=extract_progress,
    )


async def get_rodin_download_list(uuid: str, cls: type[IO.ComfyNode]) -> Rodin3DDownloadResponse:
    logging.info("[ Rodin3D API - Downloading ] Generate Successfully!")
    return await sync_op(
        cls,
        ApiEndpoint(path="/proxy/rodin/api/v2/download", method="POST"),
        response_model=Rodin3DDownloadResponse,
        data=Rodin3DDownloadRequest(task_uuid=uuid),
        monitor_progress=False,
    )


async def download_files(url_list, task_uuid: str) -> tuple[str | None, Types.File3D | None]:
    result_folder_name = f"Rodin3D_{task_uuid}"
    save_path = os.path.join(comfy_paths.get_output_directory(), result_folder_name)
    os.makedirs(save_path, exist_ok=True)
    model_file_path = None
    file_3d = None

    for i in url_list.items:
        file_path = os.path.join(save_path, i.name)
        if i.name.lower().endswith(".glb"):
            model_file_path = os.path.join(result_folder_name, i.name)
            file_3d = await download_url_to_file_3d(i.url, "glb")
            # Save to disk for backward compatibility
            with open(file_path, "wb") as f:
                f.write(file_3d.get_bytes())
        else:
            await download_url_to_bytesio(i.url, file_path)

    return model_file_path, file_3d


class Rodin3D_Regular(IO.ComfyNode):
    """Generate 3D Assets using Rodin API"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Regular",
            display_name="Rodin 3D Generate - Regular Generate",
            category="partner/3d/Rodin",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("Images"),
                *COMMON_PARAMETERS,
            ],
            outputs=[
                IO.String.Output(display_name="3D Model Path"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
    ) -> IO.NodeOutput:
        tier = "Regular"
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality_override = get_quality_mode(Polygon_count)
        task_uuid, subscription_key = await create_generate_task(
            cls,
            images=m_images,
            seed=Seed,
            material=Material_Type,
            quality_override=quality_override,
            tier=tier,
            mesh_mode=mesh_mode,
        )
        await poll_for_task_status(subscription_key, cls)
        download_list = await get_rodin_download_list(task_uuid, cls)
        model_path, file_3d = await download_files(download_list, task_uuid)

        return IO.NodeOutput(model_path, file_3d)


class Rodin3D_Detail(IO.ComfyNode):
    """Generate 3D Assets using Rodin API"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Detail",
            display_name="Rodin 3D Generate - Detail Generate",
            category="partner/3d/Rodin",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("Images"),
                *COMMON_PARAMETERS,
            ],
            outputs=[
                IO.String.Output(display_name="3D Model Path"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
    ) -> IO.NodeOutput:
        tier = "Detail"
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality_override = get_quality_mode(Polygon_count)
        task_uuid, subscription_key = await create_generate_task(
            cls,
            images=m_images,
            seed=Seed,
            material=Material_Type,
            quality_override=quality_override,
            tier=tier,
            mesh_mode=mesh_mode,
        )
        await poll_for_task_status(subscription_key, cls)
        download_list = await get_rodin_download_list(task_uuid, cls)
        model_path, file_3d = await download_files(download_list, task_uuid)

        return IO.NodeOutput(model_path, file_3d)


class Rodin3D_Smooth(IO.ComfyNode):
    """Generate 3D Assets using Rodin API"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Smooth",
            display_name="Rodin 3D Generate - Smooth Generate",
            category="partner/3d/Rodin",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("Images"),
                *COMMON_PARAMETERS,
            ],
            outputs=[
                IO.String.Output(display_name="3D Model Path"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
    ) -> IO.NodeOutput:
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality_override = get_quality_mode(Polygon_count)
        task_uuid, subscription_key = await create_generate_task(
            cls,
            images=m_images,
            seed=Seed,
            material=Material_Type,
            quality_override=quality_override,
            tier="Smooth",
            mesh_mode=mesh_mode,
        )
        await poll_for_task_status(subscription_key, cls)
        download_list = await get_rodin_download_list(task_uuid, cls)
        model_path, file_3d = await download_files(download_list, task_uuid)

        return IO.NodeOutput(model_path, file_3d)


class Rodin3D_Sketch(IO.ComfyNode):
    """Generate 3D Assets using Rodin API"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Sketch",
            display_name="Rodin 3D Generate - Sketch Generate",
            category="partner/3d/Rodin",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("Images"),
                IO.Int.Input(
                    "Seed",
                    default=0,
                    min=0,
                    max=65535,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="3D Model Path"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        Images,
        Seed,
    ) -> IO.NodeOutput:
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        task_uuid, subscription_key = await create_generate_task(
            cls,
            images=m_images,
            seed=Seed,
            material="PBR",
            quality_override=18000,
            tier="Sketch",
            mesh_mode="Quad",
        )
        await poll_for_task_status(subscription_key, cls)
        download_list = await get_rodin_download_list(task_uuid, cls)
        model_path, file_3d = await download_files(download_list, task_uuid)

        return IO.NodeOutput(model_path, file_3d)


class Rodin3D_Gen2(IO.ComfyNode):
    """Generate 3D Assets using Rodin API"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Gen2",
            display_name="Rodin 3D Generate - Gen-2 Generate",
            category="partner/3d/Rodin",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("Images"),
                IO.Int.Input(
                    "Seed",
                    default=0,
                    min=0,
                    max=65535,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                ),
                IO.Combo.Input("Material_Type", options=["PBR", "Shaded"], default="PBR", optional=True),
                IO.Combo.Input(
                    "Polygon_count",
                    options=[
                        "4K-Quad",
                        "8K-Quad",
                        "18K-Quad",
                        "50K-Quad",
                        "2K-Triangle",
                        "20K-Triangle",
                        "150K-Triangle",
                        "500K-Triangle",
                    ],
                    default="500K-Triangle",
                    optional=True,
                ),
                IO.Boolean.Input("TAPose", default=False, advanced=True),
            ],
            outputs=[
                IO.String.Output(display_name="3D Model Path"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
        TAPose,
    ) -> IO.NodeOutput:
        tier = "Gen-2"
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality_override = get_quality_mode(Polygon_count)
        task_uuid, subscription_key = await create_generate_task(
            cls,
            images=m_images,
            seed=Seed,
            material=Material_Type,
            quality_override=quality_override,
            tier=tier,
            mesh_mode=mesh_mode,
            ta_pose=TAPose,
        )
        await poll_for_task_status(subscription_key, cls)
        download_list = await get_rodin_download_list(task_uuid, cls)
        model_path, file_3d = await download_files(download_list, task_uuid)

        return IO.NodeOutput(model_path, file_3d)


def _rodin_multipart_parser(data: dict[str, Any]) -> aiohttp.FormData:
    """Convert a Rodin request dict to an aiohttp form, fixing bool/list serialization.

    Booleans --> "true"/"false". Lists --> one field per element.
    """
    form = aiohttp.FormData(default_to_multipart=True)
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, bool):
            form.add_field(key, "true" if value else "false")
        elif isinstance(value, list):
            for item in value:
                form.add_field(key, str(item))
        elif isinstance(value, (bytes, bytearray)):
            form.add_field(key, value)
        else:
            form.add_field(key, str(value))
    return form


async def _create_gen25_task(
    cls: type[IO.ComfyNode],
    request: Rodin3DGen25Request,
    images: list | None,
) -> tuple[str, str]:
    """Submit a Gen-2.5 generate job; returns (task_uuid, subscription_key)."""

    if images is not None and len(images) > 5:
        raise ValueError("Rodin Gen-2.5 supports at most 5 input images.")

    files = None
    if images:
        files = [
            (
                "images",
                open(image, "rb") if isinstance(image, str) else tensor_to_filelike(image),
            )
            for image in images
            if image is not None
        ]

    response = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/rodin/api/v2/rodin", method="POST"),
        response_model=Rodin3DGenerateResponse,
        data=request,
        files=files,
        content_type="multipart/form-data",
        multipart_parser=_rodin_multipart_parser,
    )

    if not response.uuid or not response.jobs or not response.jobs.subscription_key:
        raise RuntimeError(f"Rodin Gen-2.5 submit failed: message={response.message!r}")
    return response.uuid, response.jobs.subscription_key


_PREVIEWABLE_3D_EXTS = {".glb", ".obj", ".fbx", ".stl", ".gltf"}


async def _download_gen25_files(
    download_list: Rodin3DDownloadResponse,
    task_uuid: str,
    geometry_file_format: str,
) -> Types.File3D | None:
    """Download every file in the list; return the File3D matching the chosen format."""

    folder_name = f"Rodin3D_Gen25_{task_uuid}"
    save_dir = os.path.join(comfy_paths.get_output_directory(), folder_name)
    os.makedirs(save_dir, exist_ok=True)

    target_ext = f".{geometry_file_format.lower().lstrip('.')}"
    file_3d: Types.File3D | None = None

    for item in download_list.items:
        file_path = os.path.join(save_dir, item.name)
        ext = os.path.splitext(item.name.lower())[1]
        # Prefer the file matching the user's chosen format; fall back below.
        if file_3d is None and ext == target_ext and ext in _PREVIEWABLE_3D_EXTS:
            file_3d = await download_url_to_file_3d(item.url, target_ext.lstrip("."))
            with open(file_path, "wb") as f:
                f.write(file_3d.get_bytes())
            continue
        await download_url_to_bytesio(item.url, file_path)

    # If the chosen format wasn't found, surface any model file we did get.
    if file_3d is None:
        for item in download_list.items:
            ext = os.path.splitext(item.name.lower())[1]
            if ext in _PREVIEWABLE_3D_EXTS:
                file_3d = await download_url_to_file_3d(item.url, ext.lstrip("."))
                break
    return file_3d


_MODE_REGULAR = "Regular"
_MODE_FAST = "Fast"
_MODE_EXTREME_HIGH = "Extreme-High"

_REGULAR_POLY_OPTIONS = [
    "Default",
    "4K-Quad",
    "8K-Quad",
    "18K-Quad",
    "50K-Quad",
    "2K-Triangle",
    "20K-Triangle",
    "150K-Triangle",
    "500K-Triangle",
    "1M-Triangle",
]

_TEXTURE_MODE_OPTIONS = ["Default", "legacy", "extreme-low", "low", "medium", "high"]
_GEOMETRY_FORMAT_OPTIONS = ["glb", "fbx", "obj", "stl"]
_MATERIAL_OPTIONS = ["PBR", "Shaded", "All", "None"]


def _build_mode_input(name: str = "mode") -> IO.DynamicCombo.Input:
    return IO.DynamicCombo.Input(
        name,
        options=[
            IO.DynamicCombo.Option(
                _MODE_REGULAR,
                [
                    IO.Combo.Input(
                        "tier",
                        options=["Gen-2.5-Low", "Gen-2.5-Medium", "Gen-2.5-High"],
                        default="Gen-2.5-High",
                        tooltip="Quality tier. Higher tiers produce higher-fidelity geometry.",
                    ),
                    IO.Combo.Input(
                        "polygon_count",
                        options=_REGULAR_POLY_OPTIONS,
                        default="Default",
                        tooltip="Preset face count. 'Default' uses the server's default for the selected tier.",
                    ),
                    IO.Boolean.Input(
                        "creative",
                        default=False,
                        tooltip="Creative mode (Medium/High only). Enhances generative robustness.",
                    ),
                ],
            ),
            IO.DynamicCombo.Option(
                _MODE_FAST,
                [
                    IO.Combo.Input(
                        "tier",
                        options=[
                            "Gen-2.5-Extreme-Low",
                            "Gen-2.5-Low",
                            "Gen-2.5-Medium",
                            "Gen-2.5-High",
                        ],
                        default="Gen-2.5-Low",
                    ),
                    IO.Int.Input(
                        "mesh_faces",
                        default=20000,
                        min=1000,
                        max=20000,
                        display_mode=IO.NumberDisplay.number,
                        tooltip="Mesh face count (1K-20K in Fast mode).",
                    ),
                ],
            ),
            IO.DynamicCombo.Option(
                _MODE_EXTREME_HIGH,
                [
                    IO.Combo.Input("mesh_mode", options=["Raw", "Quad"], default="Raw"),
                    IO.Int.Input(
                        "mesh_faces",
                        default=1000000,
                        min=20000,
                        max=2000000,
                        display_mode=IO.NumberDisplay.number,
                        tooltip=(
                            "Mesh face count. Raw mode: 20K-2M. "
                            "Quad mode: keep under 200K (upstream may reject higher values)."
                        ),
                    ),
                    IO.Boolean.Input(
                        "is_micro",
                        default=False,
                        tooltip="Enable micro detail (Extreme-High only).",
                    ),
                    IO.Boolean.Input(
                        "creative",
                        default=False,
                        tooltip="Creative mode. Enhances generative robustness.",
                    ),
                ],
            ),
        ],
        tooltip=(
            "Generation mode. Regular = balanced. Fast = 1K-20K faces for rapid prototyping. "
            "Extreme-High = 20K-2M faces with optional micro details."
        ),
    )


def _build_common_inputs(*, include_image_only: bool) -> list:
    inputs: list = [
        IO.Combo.Input("material", options=_MATERIAL_OPTIONS, default="Shaded"),
        IO.Combo.Input("geometry_file_format", options=_GEOMETRY_FORMAT_OPTIONS, default="glb"),
        IO.Combo.Input(
            "texture_mode",
            options=_TEXTURE_MODE_OPTIONS,
            default="Default",
            optional=True,
            tooltip="Texture quality preset. 'Default' uses the server's default for the selected tier.",
        ),
        IO.Int.Input(
            "seed",
            default=0,
            min=0,
            max=65535,
            display_mode=IO.NumberDisplay.number,
            control_after_generate=True,
            optional=True,
        ),
        IO.Boolean.Input(
            "TAPose", default=False, optional=True, advanced=True, tooltip="T/A pose for human-like models."
        ),
        IO.Boolean.Input(
            "hd_texture", default=False, optional=True, advanced=True, tooltip="High-quality texture enhancement."
        ),
        IO.Boolean.Input(
            "texture_delight",
            default=False,
            optional=True,
            advanced=True,
            tooltip="Remove baked lighting from textures.",
        ),
    ]
    if include_image_only:
        inputs.append(
            IO.Boolean.Input(
                "use_original_alpha",
                default=False,
                optional=True,
                advanced=True,
                tooltip="Preserve image transparency.",
            )
        )
    inputs.extend(
        [
            IO.Boolean.Input(
                "addon_highpack",
                default=False,
                optional=True,
                advanced=True,
                tooltip="HighPack addon: 4K textures and ~16x faces in Quad mode.",
            ),
            IO.Int.Input(
                "bbox_width",
                default=0,
                min=0,
                max=300,
                display_mode=IO.NumberDisplay.number,
                optional=True,
                advanced=True,
                tooltip="Bounding-box width (Y axis). Set to 0 with the others to skip bbox.",
            ),
            IO.Int.Input(
                "bbox_height",
                default=0,
                min=0,
                max=300,
                display_mode=IO.NumberDisplay.number,
                optional=True,
                advanced=True,
                tooltip="Bounding-box height (Z axis).",
            ),
            IO.Int.Input(
                "bbox_length",
                default=0,
                min=0,
                max=300,
                display_mode=IO.NumberDisplay.number,
                optional=True,
                advanced=True,
                tooltip="Bounding-box length (X axis).",
            ),
            IO.Int.Input(
                "height_cm",
                default=0,
                min=0,
                max=10000,
                display_mode=IO.NumberDisplay.number,
                optional=True,
                advanced=True,
                tooltip="Approximate model height in centimeters (0 to skip).",
            ),
        ]
    )
    return inputs


_PRICE_EXPR = """
(
  $baseCredits := widgets.mode = "extreme-high" ? 1.0 : 0.5;
  $addonCredits := widgets.addon_highpack ? 1.0 : 0.0;
  $total := ($baseCredits * 1.5) + ($addonCredits * 0.8);
  {"type":"usd","usd": $total}
)
"""


def _resolve_mode_params(mode_input: dict) -> dict:
    """Translate the DynamicCombo `mode` payload into Gen-2.5 request fields.

    Returns a dict with: tier, quality_override, mesh_mode, geometry_instruct_mode, is_micro.
    Missing keys mean "do not send" (so we don't override server defaults).
    """
    selected = mode_input["mode"]
    out: dict = {}

    if selected == _MODE_REGULAR:
        out["tier"] = mode_input["tier"]
        polygon = mode_input.get("polygon_count", "Default")
        if polygon != "Default":
            mesh_mode, faces = get_quality_mode(polygon)
            out["mesh_mode"] = mesh_mode
            out["quality_override"] = faces
        if mode_input.get("creative"):
            out["geometry_instruct_mode"] = "creative"

    elif selected == _MODE_FAST:
        out["tier"] = mode_input["tier"]
        out["mesh_mode"] = "Raw"
        out["quality_override"] = int(mode_input["mesh_faces"])

    elif selected == _MODE_EXTREME_HIGH:
        out["tier"] = "Gen-2.5-Extreme-High"
        out["mesh_mode"] = mode_input["mesh_mode"]
        out["quality_override"] = int(mode_input["mesh_faces"])
        if mode_input.get("is_micro"):
            out["is_micro"] = True
        if mode_input.get("creative"):
            out["geometry_instruct_mode"] = "creative"
    return out


def _build_request(
    *,
    mode_input: dict,
    material: str,
    geometry_file_format: str,
    texture_mode: str,
    seed: int,
    TAPose: bool,
    hd_texture: bool,
    texture_delight: bool,
    addon_highpack: bool,
    bbox_width: int,
    bbox_height: int,
    bbox_length: int,
    height_cm: int,
    prompt: str | None = None,
    use_original_alpha: bool = False,
) -> Rodin3DGen25Request:
    mode_params = _resolve_mode_params(mode_input)

    bbox = None
    if bbox_width and bbox_height and bbox_length:
        bbox = [bbox_width, bbox_height, bbox_length]

    return Rodin3DGen25Request(
        tier=mode_params["tier"],
        prompt=prompt or None,
        seed=seed,
        material=material,
        geometry_file_format=geometry_file_format,
        texture_mode=None if texture_mode == "Default" else texture_mode,
        mesh_mode=mode_params.get("mesh_mode"),
        quality_override=mode_params.get("quality_override"),
        geometry_instruct_mode=mode_params.get("geometry_instruct_mode"),
        bbox_condition=bbox,
        height=height_cm or None,
        TAPose=TAPose or None,
        hd_texture=hd_texture or None,
        texture_delight=texture_delight or None,
        is_micro=mode_params.get("is_micro"),
        use_original_alpha=use_original_alpha or None,
        addons=["HighPack"] if addon_highpack else None,
    )


class Rodin3D_Gen25_Image(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Gen25_Image",
            display_name="Rodin 3D Gen-2.5 - Image to 3D",
            category="partner/3d/Rodin",
            description=(
                "Generate a 3D model from 1-5 reference images via Rodin Gen-2.5. "
                "Pick a mode (Fast / Regular / Extreme-High) to tune quality vs. cost."
            ),
            inputs=[
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplatePrefix(IO.Image.Input("image"), prefix="image", min=1, max=5),
                    tooltip="1-5 images. The first image is used for materials when multi-view.",
                ),
                _build_mode_input(),
                *_build_common_inputs(include_image_only=True),
            ],
            outputs=[IO.File3DAny.Output(display_name="model_file")],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["mode", "addon_highpack"]),
                expr=_PRICE_EXPR,
            ),
        )

    @classmethod
    async def execute(
        cls,
        images: IO.Autogrow.Type,
        mode: dict,
        material: str,
        geometry_file_format: str,
        texture_mode: str,
        seed: int,
        TAPose: bool,
        hd_texture: bool,
        texture_delight: bool,
        use_original_alpha: bool,
        addon_highpack: bool,
        bbox_width: int,
        bbox_height: int,
        bbox_length: int,
        height_cm: int,
    ) -> IO.NodeOutput:
        image_tensors = [img for img in images.values() if img is not None]
        if not image_tensors:
            raise ValueError("Rodin Gen-2.5 Image-to-3D requires at least one image.")

        # Flatten multi-image tensors into individual frames; the API accepts each as a separate part.
        flat_images: list = []
        for tensor in image_tensors:
            if hasattr(tensor, "shape") and len(tensor.shape) == 4:
                for i in range(tensor.shape[0]):
                    flat_images.append(tensor[i])
            else:
                flat_images.append(tensor)

        if len(flat_images) > 5:
            raise ValueError(f"Rodin Gen-2.5 accepts at most 5 images; received {len(flat_images)}.")

        request = _build_request(
            mode_input=mode,
            material=material,
            geometry_file_format=geometry_file_format,
            texture_mode=texture_mode,
            seed=seed,
            TAPose=TAPose,
            hd_texture=hd_texture,
            texture_delight=texture_delight,
            addon_highpack=addon_highpack,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            bbox_length=bbox_length,
            height_cm=height_cm,
            prompt=None,
            use_original_alpha=use_original_alpha,
        )

        task_uuid, subscription_key = await _create_gen25_task(cls, request, flat_images)
        await poll_for_task_status(subscription_key, cls)
        download_list = await get_rodin_download_list(task_uuid, cls)
        file_3d = await _download_gen25_files(download_list, task_uuid, geometry_file_format)
        return IO.NodeOutput(file_3d)


class Rodin3D_Gen25_Text(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Gen25_Text",
            display_name="Rodin 3D Gen-2.5 - Text to 3D",
            category="partner/3d/Rodin",
            description=(
                "Generate a 3D model from a text prompt via Rodin Gen-2.5. "
                "Pick a mode (Fast / Regular / Extreme-High) to tune quality vs. cost."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the 3D model.",
                ),
                _build_mode_input(),
                *_build_common_inputs(include_image_only=False),
            ],
            outputs=[IO.File3DAny.Output(display_name="model_file")],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["mode", "addon_highpack"]),
                expr=_PRICE_EXPR,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        mode: dict,
        material: str,
        geometry_file_format: str,
        texture_mode: str,
        seed: int,
        TAPose: bool,
        hd_texture: bool,
        texture_delight: bool,
        addon_highpack: bool,
        bbox_width: int,
        bbox_height: int,
        bbox_length: int,
        height_cm: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, field_name="prompt", min_length=1, max_length=2500)
        request = _build_request(
            mode_input=mode,
            material=material,
            geometry_file_format=geometry_file_format,
            texture_mode=texture_mode,
            seed=seed,
            TAPose=TAPose,
            hd_texture=hd_texture,
            texture_delight=texture_delight,
            addon_highpack=addon_highpack,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            bbox_length=bbox_length,
            height_cm=height_cm,
            prompt=prompt,
        )
        task_uuid, subscription_key = await _create_gen25_task(cls, request, images=None)
        await poll_for_task_status(subscription_key, cls)
        download_list = await get_rodin_download_list(task_uuid, cls)
        file_3d = await _download_gen25_files(download_list, task_uuid, geometry_file_format)
        return IO.NodeOutput(file_3d)


class Rodin3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Rodin3D_Regular,
            Rodin3D_Detail,
            Rodin3D_Smooth,
            Rodin3D_Sketch,
            Rodin3D_Gen2,
            Rodin3D_Gen25_Image,
            Rodin3D_Gen25_Text,
        ]


async def comfy_entrypoint() -> Rodin3DExtension:
    return Rodin3DExtension()
