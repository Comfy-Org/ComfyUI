import zipfile
from io import BytesIO

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, Types
from comfy_api_nodes.apis.hunyuan3d import (
    Hunyuan3DViewImage,
    InputGenerateType,
    ResultFile3D,
    SmartTopologyRequest,
    TaskFile3DInput,
    TextureEditTaskRequest,
    To3DPartTaskRequest,
    To3DProTaskCreateResponse,
    To3DProTaskQueryRequest,
    To3DProTaskRequest,
    To3DProTaskResultResponse,
    To3DUVTaskRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    download_url_to_bytesio,
    download_url_to_file_3d,
    download_url_to_image_tensor,
    downscale_image_tensor_by_max_side,
    poll_op,
    sync_op,
    upload_3d_model_to_comfyapi,
    upload_image_to_comfyapi,
    validate_image_dimensions,
    validate_string,
)


def _is_tencent_rate_limited(status: int, body: object) -> bool:
    return (
        status == 400
        and isinstance(body, dict)
        and "RequestLimitExceeded" in str(body.get("Response", {}).get("Error", {}).get("Code", ""))
    )


class ObjZipResult:
    __slots__ = ("obj", "texture", "metallic", "normal", "roughness")

    def __init__(
        self,
        obj: Types.File3D,
        texture: Input.Image | None = None,
        metallic: Input.Image | None = None,
        normal: Input.Image | None = None,
        roughness: Input.Image | None = None,
    ):
        self.obj = obj
        self.texture = texture
        self.metallic = metallic
        self.normal = normal
        self.roughness = roughness


async def download_and_extract_obj_zip(url: str) -> ObjZipResult:
    """The Tencent API returns OBJ results as ZIP archives containing the .obj mesh, and texture images.

    When PBR is enabled, the ZIP may contain additional metallic, normal, and roughness maps
    identified by their filename suffixes.
    """
    data = BytesIO()
    await download_url_to_bytesio(url, data)
    data.seek(0)
    if not zipfile.is_zipfile(data):
        data.seek(0)
        return ObjZipResult(obj=Types.File3D(source=data, file_format="obj"))
    data.seek(0)
    obj_bytes = None
    textures: dict[str, Input.Image] = {}
    with zipfile.ZipFile(data) as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith(".obj"):
                obj_bytes = zf.read(name)
            elif any(lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
                stem = lower.rsplit(".", 1)[0]
                tensor = bytesio_to_image_tensor(BytesIO(zf.read(name)), mode="RGB")
                matched_key = "texture"
                for suffix, key in {
                    "_metallic": "metallic",
                    "_normal": "normal",
                    "_roughness": "roughness",
                }.items():
                    if stem.endswith(suffix):
                        matched_key = key
                        break
                textures[matched_key] = tensor
    if obj_bytes is None:
        raise ValueError("ZIP archive does not contain an OBJ file.")
    return ObjZipResult(
        obj=Types.File3D(source=BytesIO(obj_bytes), file_format="obj"),
        texture=textures.get("texture"),
        metallic=textures.get("metallic"),
        normal=textures.get("normal"),
        roughness=textures.get("roughness"),
    )


def get_file_from_response(
    response_objs: list[ResultFile3D], file_type: str, raise_if_not_found: bool = True
) -> ResultFile3D | None:
    for i in response_objs:
        if i.Type.lower() == file_type.lower():
            return i
    if raise_if_not_found:
        raise ValueError(f"'{file_type}' file type is not found in the response.")
    return None


class TencentTextToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentTextToModelNode",
            display_name="Hunyuan3D: Text to Model",
            category="partner/3d/Tencent",
            essentials_category="3D",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["3.0", "3.1"],
                    tooltip="The LowPoly option is unavailable for the `3.1` model.",
                ),
                IO.String.Input("prompt", multiline=True, default="", tooltip="Supports up to 1024 characters."),
                IO.Int.Input("face_count", default=500000, min=3000, max=1500000),
                IO.DynamicCombo.Input(
                    "generate_type",
                    options=[
                        IO.DynamicCombo.Option("Normal", [IO.Boolean.Input("pbr", default=False)]),
                        IO.DynamicCombo.Option(
                            "LowPoly",
                            [
                                IO.Combo.Input("polygon_type", options=["triangle", "quadrilateral"]),
                                IO.Boolean.Input("pbr", default=False),
                            ],
                        ),
                        IO.DynamicCombo.Option("Geometry", []),
                    ],
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DOBJ.Output(display_name="OBJ"),
                IO.Image.Output(display_name="texture_image"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["generate_type", "generate_type.pbr", "face_count"]),
                expr="""
                (
                  $base := widgets.generate_type = "normal" ? 25 : widgets.generate_type = "lowpoly" ? 30 : 15;
                  $pbr := $lookup(widgets, "generate_type.pbr") ? 10 : 0;
                  $face := widgets.face_count != 500000 ? 10 : 0;
                  {"type":"usd","usd": ($base + $pbr + $face) * 0.02}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        face_count: int,
        generate_type: InputGenerateType,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        validate_string(prompt, field_name="prompt", min_length=1, max_length=1024)
        if model == "3.1" and generate_type["generate_type"].lower() == "lowpoly":
            raise ValueError("The LowPoly option is currently unavailable for the 3.1 model.")
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-pro", method="POST"),
            response_model=To3DProTaskCreateResponse,
            data=To3DProTaskRequest(
                Model=model,
                Prompt=prompt,
                FaceCount=face_count,
                GenerateType=generate_type["generate_type"],
                EnablePBR=generate_type.get("pbr", None),
                PolygonType=generate_type.get("polygon_type", None),
            ),
            is_rate_limited=_is_tencent_rate_limited,
        )
        if response.Error:
            raise ValueError(f"Task creation failed with code {response.Error.Code}: {response.Error.Message}")
        task_id = response.JobId
        result = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-pro/query", method="POST"),
            data=To3DProTaskQueryRequest(JobId=task_id),
            response_model=To3DProTaskResultResponse,
            status_extractor=lambda r: r.Status,
        )
        obj_file_response = get_file_from_response(result.ResultFile3Ds, "obj", raise_if_not_found=False)
        obj_result = None
        if obj_file_response:
            obj_result = await download_and_extract_obj_zip(obj_file_response.Url)
        return IO.NodeOutput(
            f"{task_id}.glb",
            await download_url_to_file_3d(
                get_file_from_response(result.ResultFile3Ds, "glb").Url, "glb", task_id=task_id
            ),
            obj_result.obj if obj_result else None,
            obj_result.texture if obj_result else None,
        )


class TencentImageToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentImageToModelNode",
            display_name="Hunyuan3D: Image(s) to Model",
            category="partner/3d/Tencent",
            essentials_category="3D",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["3.0", "3.1"],
                    tooltip="The LowPoly option is unavailable for the `3.1` model.",
                ),
                IO.Image.Input("image"),
                IO.Image.Input("image_left", optional=True),
                IO.Image.Input("image_right", optional=True),
                IO.Image.Input("image_back", optional=True),
                IO.Int.Input("face_count", default=500000, min=3000, max=1500000),
                IO.DynamicCombo.Input(
                    "generate_type",
                    options=[
                        IO.DynamicCombo.Option("Normal", [IO.Boolean.Input("pbr", default=False)]),
                        IO.DynamicCombo.Option(
                            "LowPoly",
                            [
                                IO.Combo.Input("polygon_type", options=["triangle", "quadrilateral"]),
                                IO.Boolean.Input("pbr", default=False),
                            ],
                        ),
                        IO.DynamicCombo.Option("Geometry", []),
                    ],
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DOBJ.Output(display_name="OBJ"),
                IO.Image.Output(display_name="texture_image"),
                IO.Image.Output(display_name="optional_metallic"),
                IO.Image.Output(display_name="optional_normal"),
                IO.Image.Output(display_name="optional_roughness"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["generate_type", "generate_type.pbr", "face_count"],
                    inputs=["image_left", "image_right", "image_back"],
                ),
                expr="""
                (
                  $base := widgets.generate_type = "normal" ? 25 : widgets.generate_type = "lowpoly" ? 30 : 15;
                  $multiview := (
                    inputs.image_left.connected or inputs.image_right.connected or inputs.image_back.connected
                  ) ? 10 : 0;
                  $pbr := $lookup(widgets, "generate_type.pbr") ? 10 : 0;
                  $face := widgets.face_count != 500000 ? 10 : 0;
                  {"type":"usd","usd": ($base + $multiview + $pbr + $face) * 0.02}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        face_count: int,
        generate_type: InputGenerateType,
        seed: int,
        image_left: Input.Image | None = None,
        image_right: Input.Image | None = None,
        image_back: Input.Image | None = None,
    ) -> IO.NodeOutput:
        _ = seed
        if model == "3.1" and generate_type["generate_type"].lower() == "lowpoly":
            raise ValueError("The LowPoly option is currently unavailable for the 3.1 model.")
        validate_image_dimensions(image, min_width=128, min_height=128)
        multiview_images = []
        for k, v in {
            "left": image_left,
            "right": image_right,
            "back": image_back,
        }.items():
            if v is None:
                continue
            validate_image_dimensions(v, min_width=128, min_height=128)
            multiview_images.append(
                Hunyuan3DViewImage(
                    ViewType=k,
                    ViewImageUrl=await upload_image_to_comfyapi(
                        cls,
                        downscale_image_tensor_by_max_side(v, max_side=4900),
                        mime_type="image/webp",
                        total_pixels=24_010_000,
                    ),
                )
            )
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-pro", method="POST"),
            response_model=To3DProTaskCreateResponse,
            data=To3DProTaskRequest(
                Model=model,
                FaceCount=face_count,
                GenerateType=generate_type["generate_type"],
                ImageUrl=await upload_image_to_comfyapi(
                    cls,
                    downscale_image_tensor_by_max_side(image, max_side=4900),
                    mime_type="image/webp",
                    total_pixels=24_010_000,
                ),
                MultiViewImages=multiview_images if multiview_images else None,
                EnablePBR=generate_type.get("pbr", None),
                PolygonType=generate_type.get("polygon_type", None),
            ),
            is_rate_limited=_is_tencent_rate_limited,
        )
        if response.Error:
            raise ValueError(f"Task creation failed with code {response.Error.Code}: {response.Error.Message}")
        task_id = response.JobId
        result = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-pro/query", method="POST"),
            data=To3DProTaskQueryRequest(JobId=task_id),
            response_model=To3DProTaskResultResponse,
            status_extractor=lambda r: r.Status,
        )
        obj_file_response = get_file_from_response(result.ResultFile3Ds, "obj", raise_if_not_found=False)
        if obj_file_response:
            obj_result = await download_and_extract_obj_zip(obj_file_response.Url)
            return IO.NodeOutput(
                f"{task_id}.glb",
                await download_url_to_file_3d(
                    get_file_from_response(result.ResultFile3Ds, "glb").Url, "glb", task_id=task_id
                ),
                obj_result.obj,
                obj_result.texture,
                obj_result.metallic if obj_result.metallic is not None else torch.zeros(1, 1, 1, 3),
                obj_result.normal if obj_result.normal is not None else torch.zeros(1, 1, 1, 3),
                obj_result.roughness if obj_result.roughness is not None else torch.zeros(1, 1, 1, 3),
            )
        return IO.NodeOutput(
            f"{task_id}.glb",
            await download_url_to_file_3d(
                get_file_from_response(result.ResultFile3Ds, "glb").Url, "glb", task_id=task_id
            ),
            None,
            None,
            None,
            None,
            None,
        )


class TencentModelTo3DUVNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentModelTo3DUVNode",
            display_name="Hunyuan3D: Model to UV",
            category="partner/3d/Tencent",
            description="Perform UV unfolding on a 3D model to generate UV texture. "
            "Input model must have less than 30000 faces.",
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[IO.File3DGLB, IO.File3DOBJ, IO.File3DFBX, IO.File3DAny],
                    tooltip="Input 3D model (GLB, OBJ, or FBX)",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.File3DOBJ.Output(display_name="OBJ"),
                IO.File3DFBX.Output(display_name="FBX"),
                IO.Image.Output(display_name="uv_image"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr='{"type":"usd","usd":0.2}'),
        )

    SUPPORTED_FORMATS = {"glb", "obj", "fbx"}

    @classmethod
    async def execute(
        cls,
        model_3d: Types.File3D,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        file_format = model_3d.format.lower()
        if file_format not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: '{file_format}'. "
                f"Supported formats: {', '.join(sorted(cls.SUPPORTED_FORMATS))}."
            )
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-uv", method="POST"),
            response_model=To3DProTaskCreateResponse,
            data=To3DUVTaskRequest(
                File=TaskFile3DInput(
                    Type=file_format.upper(),
                    Url=await upload_3d_model_to_comfyapi(cls, model_3d, file_format),
                )
            ),
            is_rate_limited=_is_tencent_rate_limited,
        )
        if response.Error:
            raise ValueError(f"Task creation failed with code {response.Error.Code}: {response.Error.Message}")
        result = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-uv/query", method="POST"),
            data=To3DProTaskQueryRequest(JobId=response.JobId),
            response_model=To3DProTaskResultResponse,
            status_extractor=lambda r: r.Status,
        )
        uv_image_file = get_file_from_response(result.ResultFile3Ds, "uv_image", raise_if_not_found=False)
        uv_image = (
            await download_url_to_image_tensor(uv_image_file.Url)
            if uv_image_file is not None
            else torch.zeros(1, 1, 1, 3)
        )
        return IO.NodeOutput(
            await download_url_to_file_3d(get_file_from_response(result.ResultFile3Ds, "obj").Url, "obj"),
            await download_url_to_file_3d(get_file_from_response(result.ResultFile3Ds, "fbx").Url, "fbx"),
            uv_image,
        )


class Tencent3DTextureEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Tencent3DTextureEditNode",
            display_name="Hunyuan3D: 3D Texture Edit",
            category="partner/3d/Tencent",
            description="After inputting the 3D model, perform 3D model texture redrawing.",
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[IO.File3DFBX, IO.File3DAny],
                    tooltip="3D model in FBX format. Model should have less than 100000 faces.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Describes texture editing. Supports up to 1024 UTF-8 characters.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DOBJ.Output(display_name="OBJ"),
                IO.Image.Output(display_name="texture_image"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd": 0.6}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model_3d: Types.File3D,
        prompt: str,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        file_format = model_3d.format.lower()
        if file_format != "fbx":
            raise ValueError(f"Unsupported file format: '{file_format}'. Only FBX format is supported.")
        validate_string(prompt, field_name="prompt", min_length=1, max_length=1024)
        model_url = await upload_3d_model_to_comfyapi(cls, model_3d, file_format)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-texture-edit", method="POST"),
            response_model=To3DProTaskCreateResponse,
            data=TextureEditTaskRequest(
                File3D=TaskFile3DInput(Type=file_format.upper(), Url=model_url),
                Prompt=prompt,
                EnablePBR=True,
            ),
            is_rate_limited=_is_tencent_rate_limited,
        )
        if response.Error:
            raise ValueError(f"Task creation failed with code {response.Error.Code}: {response.Error.Message}")

        result = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-texture-edit/query", method="POST"),
            data=To3DProTaskQueryRequest(JobId=response.JobId),
            response_model=To3DProTaskResultResponse,
            status_extractor=lambda r: r.Status,
        )
        return IO.NodeOutput(
            await download_url_to_file_3d(get_file_from_response(result.ResultFile3Ds, "glb").Url, "glb"),
            await download_url_to_file_3d(get_file_from_response(result.ResultFile3Ds, "obj").Url, "obj"),
            await download_url_to_image_tensor(get_file_from_response(result.ResultFile3Ds, "texture_image").Url),
        )


class Tencent3DPartNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Tencent3DPartNode",
            display_name="Hunyuan3D: 3D Part",
            category="partner/3d/Tencent",
            description="Automatically perform component identification and generation based on the model structure.",
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[IO.File3DFBX, IO.File3DAny],
                    tooltip="3D model in FBX format. Model should have less than 30000 faces.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr='{"type":"usd","usd":0.6}'),
        )

    @classmethod
    async def execute(
        cls,
        model_3d: Types.File3D,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        file_format = model_3d.format.lower()
        if file_format != "fbx":
            raise ValueError(f"Unsupported file format: '{file_format}'. Only FBX format is supported.")
        model_url = await upload_3d_model_to_comfyapi(cls, model_3d, file_format)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-part", method="POST"),
            response_model=To3DProTaskCreateResponse,
            data=To3DPartTaskRequest(
                File=TaskFile3DInput(Type=file_format.upper(), Url=model_url),
            ),
            is_rate_limited=_is_tencent_rate_limited,
        )
        if response.Error:
            raise ValueError(f"Task creation failed with code {response.Error.Code}: {response.Error.Message}")
        result = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-part/query", method="POST"),
            data=To3DProTaskQueryRequest(JobId=response.JobId),
            response_model=To3DProTaskResultResponse,
            status_extractor=lambda r: r.Status,
        )
        return IO.NodeOutput(
            await download_url_to_file_3d(get_file_from_response(result.ResultFile3Ds, "fbx").Url, "fbx"),
        )


class TencentSmartTopologyNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentSmartTopologyNode",
            display_name="Hunyuan3D: Smart Topology",
            category="partner/3d/Tencent",
            description="Perform smart retopology on a 3D model. "
            "Supports GLB/OBJ formats; max 200MB; recommended for high-poly models.",
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[IO.File3DGLB, IO.File3DOBJ, IO.File3DAny],
                    tooltip="Input 3D model (GLB or OBJ)",
                ),
                IO.Combo.Input(
                    "polygon_type",
                    options=["triangle", "quadrilateral"],
                    tooltip="Surface composition type.",
                ),
                IO.Combo.Input(
                    "face_level",
                    options=["medium", "high", "low"],
                    tooltip="Polygon reduction level.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.File3DOBJ.Output(display_name="OBJ"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr='{"type":"usd","usd":1.0}'),
        )

    SUPPORTED_FORMATS = {"glb", "obj"}

    @classmethod
    async def execute(
        cls,
        model_3d: Types.File3D,
        polygon_type: str,
        face_level: str,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        file_format = model_3d.format.lower()
        if file_format not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: '{file_format}'. " f"Supported: {', '.join(sorted(cls.SUPPORTED_FORMATS))}."
            )
        model_url = await upload_3d_model_to_comfyapi(cls, model_3d, file_format)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-smart-topology", method="POST"),
            response_model=To3DProTaskCreateResponse,
            data=SmartTopologyRequest(
                File3D=TaskFile3DInput(Type=file_format.upper(), Url=model_url),
                PolygonType=polygon_type,
                FaceLevel=face_level,
            ),
            is_rate_limited=_is_tencent_rate_limited,
        )
        if response.Error:
            raise ValueError(f"Task creation failed: [{response.Error.Code}] {response.Error.Message}")
        result = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-smart-topology/query", method="POST"),
            data=To3DProTaskQueryRequest(JobId=response.JobId),
            response_model=To3DProTaskResultResponse,
            status_extractor=lambda r: r.Status,
        )
        return IO.NodeOutput(
            await download_url_to_file_3d(get_file_from_response(result.ResultFile3Ds, "obj").Url, "obj"),
        )


class TencentHunyuan3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TencentTextToModelNode,
            TencentImageToModelNode,
            TencentModelTo3DUVNode,
            Tencent3DTextureEditNode,
            Tencent3DPartNode,
            TencentSmartTopologyNode,
        ]


async def comfy_entrypoint() -> TencentHunyuan3DExtension:
    return TencentHunyuan3DExtension()
