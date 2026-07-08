import nodes
import folder_paths
import os
import uuid

from typing_extensions import override
from comfy_api.latest import IO, UI, ComfyExtension, InputImpl, Types

from pathlib import Path


def normalize_path(path):
    return path.replace('\\', '/')

class Load3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")

        os.makedirs(input_dir, exist_ok=True)

        input_path = Path(input_dir)
        base_path = Path(folder_paths.get_input_directory())

        files = [
            normalize_path(str(file_path.relative_to(base_path)))
            for file_path in input_path.rglob("*")
            if file_path.suffix.lower() in {'.gltf', '.glb', '.obj', '.fbx', '.stl', '.spz', '.splat', '.ply', '.ksplat'}
        ]
        return IO.Schema(
            node_id="Load3D",
            display_name="Load 3D & Animation",
            category="3d",
            essentials_category="Basics",
            is_experimental=True,
            inputs=[
                IO.Combo.Input("model_file", options=["none"] + sorted(files), upload=IO.UploadType.model),
                IO.Load3D.Input("image"),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(display_name="mask"),
                IO.String.Output(display_name="mesh_path"),
                IO.Image.Output(display_name="normal"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Video.Output(display_name="recording_video"),
                IO.File3DAny.Output(display_name="model_3d"),
                IO.Load3DModelInfo.Output(display_name="model_3d_info"),
            ],
        )

    @classmethod
    def validate_inputs(cls, model_file, **kwargs) -> bool | str:
        if not model_file or model_file == "none":
            return True
        if not folder_paths.exists_annotated_filepath(model_file):
            return f"Invalid 3D model file: {model_file}"
        return True

    @classmethod
    def execute(cls, model_file, image, **kwargs) -> IO.NodeOutput:
        image_path = folder_paths.get_annotated_filepath(image['image'])
        mask_path = folder_paths.get_annotated_filepath(image['mask'])
        normal_path = folder_paths.get_annotated_filepath(image['normal'])

        load_image_node = nodes.LoadImage()
        output_image, ignore_mask = load_image_node.load_image(image=image_path)
        ignore_image, output_mask = load_image_node.load_image(image=mask_path)
        normal_image, ignore_mask2 = load_image_node.load_image(image=normal_path)

        video = None

        if image['recording'] != "":
            recording_video_path = folder_paths.get_annotated_filepath(image['recording'])

            video = InputImpl.VideoFromFile(recording_video_path)

        file_3d = None
        mesh_path = ""
        if model_file and model_file != "none":
            file_3d = Types.File3D(folder_paths.get_annotated_filepath(model_file))
            mesh_path = model_file
        model_3d_info = image.get('model_3d_info', [])
        return IO.NodeOutput(output_image, output_mask, mesh_path, normal_image, image['camera_info'], video, file_3d, model_3d_info)

    process = execute  # TODO: remove


class Preview3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Preview3D",
            search_aliases=["view mesh", "3d viewer"],
            display_name="Preview 3D & Animation",
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.MultiType.Input(
                    IO.String.Input("model_file", default="", multiline=False),
                    types=[
                        IO.File3DGLB,
                        IO.File3DGLTF,
                        IO.File3DFBX,
                        IO.File3DOBJ,
                        IO.File3DSTL,
                        IO.File3DUSDZ,
                        IO.File3DAny,
                    ],
                    tooltip="3D model file or path string",
                ),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True),
                IO.Image.Input("bg_image", optional=True, advanced=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, model_file: str | Types.File3D, **kwargs) -> IO.NodeOutput:
        if isinstance(model_file, Types.File3D):
            filename = f"preview3d_{uuid.uuid4().hex}.{model_file.format}"
            model_file.save_to(os.path.join(folder_paths.get_output_directory(), filename))
        else:
            filename = model_file
        camera_info = kwargs.get("camera_info", None)
        bg_image = kwargs.get("bg_image", None)
        return IO.NodeOutput(ui=UI.PreviewUI3D(filename, camera_info, bg_image=bg_image))

    process = execute  # TODO: remove


class Preview3DAdvanced(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Preview3DAdvanced",
            display_name="Preview 3D (Advanced)",
            search_aliases=["preview 3d", "3d viewer", "view mesh", "frame 3d", "3d camera output"],
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[
                        IO.File3DGLB,
                        IO.File3DGLTF,
                        IO.File3DFBX,
                        IO.File3DOBJ,
                        IO.File3DSTL,
                        IO.File3DUSDZ,
                        IO.File3DAny,
                    ],
                    tooltip="3D model file from an upstream 3D node.",
                ),
                IO.Load3DModelInfo.Input("model_3d_info", optional=True, advanced=True),
                IO.Load3D.Input("viewport_state"),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                IO.File3DAny.Output(display_name="model_3d"),
                IO.Load3DModelInfo.Output(display_name="model_3d_info"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, model_3d: Types.File3D, viewport_state, width: int, height: int, **kwargs) -> IO.NodeOutput:
        filename = f"preview3d_advanced_{uuid.uuid4().hex}.{model_3d.format}"
        model_3d.save_to(os.path.join(folder_paths.get_temp_directory(), filename))

        camera_info_input = kwargs.get("camera_info", None)
        camera_info = camera_info_input if camera_info_input is not None else viewport_state['camera_info']
        model_3d_info_input = kwargs.get("model_3d_info", None)
        model_3d_info = model_3d_info_input if model_3d_info_input is not None else viewport_state.get('model_3d_info', [])
        return IO.NodeOutput(
            model_3d,
            model_3d_info,
            camera_info,
            width,
            height,
            ui=UI.PreviewUI3DAdvanced(filename, camera_info, model_3d_info),
        )


class PreviewGaussianSplat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PreviewGaussianSplat",
            display_name="Preview Splat",
            category="3d",
            is_experimental=True,
            is_output_node=True,
            search_aliases=[
                "view splat",
                "view gaussian",
                "view gaussian splat",
                "preview gaussian",
                "preview gaussian splat",
                "view 3dgs",
                "preview 3dgs",
                "preview ply",
                "preview spz",
                "preview splat",
                "preview ksplat",
            ],
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[
                        IO.File3DSplatAny,
                        IO.File3DPLY,
                        IO.File3DSPLAT,
                        IO.File3DSPZ,
                        IO.File3DKSPLAT,
                    ],
                    tooltip="A gaussian splat 3D file.",
                ),
                IO.Load3DModelInfo.Input("model_3d_info", optional=True, advanced=True),
                IO.Load3D.Input("viewport_state"),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                IO.File3DSplatAny.Output(display_name="model_3d"),
                IO.Load3DModelInfo.Output(display_name="model_3d_info"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, model_3d: Types.File3D, viewport_state, width: int, height: int, **kwargs) -> IO.NodeOutput:
        filename = f"preview_splat_{uuid.uuid4().hex}.{model_3d.format}"
        model_3d.save_to(os.path.join(folder_paths.get_temp_directory(), filename))

        camera_info_input = kwargs.get("camera_info", None)
        camera_info = camera_info_input if camera_info_input is not None else viewport_state['camera_info']
        model_3d_info_input = kwargs.get("model_3d_info", None)
        model_3d_info = model_3d_info_input if model_3d_info_input is not None else viewport_state.get('model_3d_info', [])
        return IO.NodeOutput(
            model_3d,
            model_3d_info,
            camera_info,
            width,
            height,
            ui=UI.PreviewUI3DAdvanced(filename, camera_info, model_3d_info),
        )


class PreviewPointCloud(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PreviewPointCloud",
            display_name="Preview Point Cloud",
            category="3d",
            is_experimental=True,
            is_output_node=True,
            search_aliases=[
                "view point cloud",
                "view pointcloud",
                "preview point cloud",
                "preview pointcloud",
                "preview ply",
            ],
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[
                        IO.File3DPointCloudAny,
                        IO.File3DPLY,
                    ],
                    tooltip="Point cloud file (.ply)",
                ),
                IO.Load3DModelInfo.Input("model_3d_info", optional=True, advanced=True),
                IO.Load3D.Input("viewport_state"),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                IO.File3DPointCloudAny.Output(display_name="model_3d"),
                IO.Load3DModelInfo.Output(display_name="model_3d_info"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, model_3d: Types.File3D, viewport_state, width: int, height: int, **kwargs) -> IO.NodeOutput:
        filename = f"preview_pointcloud_{uuid.uuid4().hex}.{model_3d.format}"
        model_3d.save_to(os.path.join(folder_paths.get_temp_directory(), filename))

        camera_info_input = kwargs.get("camera_info", None)
        camera_info = camera_info_input if camera_info_input is not None else viewport_state['camera_info']
        model_3d_info_input = kwargs.get("model_3d_info", None)
        model_3d_info = model_3d_info_input if model_3d_info_input is not None else viewport_state.get('model_3d_info', [])
        return IO.NodeOutput(
            model_3d,
            model_3d_info,
            camera_info,
            width,
            height,
            ui=UI.PreviewUI3DAdvanced(filename, camera_info, model_3d_info),
        )


MESH_EXTENSIONS = {'.gltf', '.glb', '.obj', '.fbx', '.stl'}


class Load3DAdvanced(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")
        os.makedirs(input_dir, exist_ok=True)

        input_path = Path(input_dir)
        base_path = Path(folder_paths.get_input_directory())

        files = [
            normalize_path(str(file_path.relative_to(base_path)))
            for file_path in input_path.rglob("*")
            if file_path.suffix.lower() in MESH_EXTENSIONS
        ]
        return IO.Schema(
            node_id="Load3DAdvanced",
            display_name="Load 3D (Advanced)",
            category="3d",
            search_aliases=[
                "load mesh",
                "load gltf",
                "load glb",
                "load obj",
                "load fbx",
                "load stl",
            ],
            is_experimental=True,
            inputs=[
                IO.Combo.Input("model_file", options=["none"] + sorted(files), upload=IO.UploadType.model),
                IO.Load3D.Input("viewport_state"),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                IO.File3DAny.Output(display_name="model_3d"),
                IO.Load3DModelInfo.Output(display_name="model_3d_info"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def validate_inputs(cls, model_file, **kwargs) -> bool | str:
        if not model_file or model_file == "none":
            return True
        if not folder_paths.exists_annotated_filepath(model_file):
            return f"Invalid 3D model file: {model_file}"
        return True

    @classmethod
    def execute(cls, model_file, viewport_state, width: int, height: int, **kwargs) -> IO.NodeOutput:
        file_3d = None
        if model_file and model_file != "none":
            file_3d = Types.File3D(folder_paths.get_annotated_filepath(model_file))
        model_3d_info = viewport_state.get('model_3d_info', [])
        return IO.NodeOutput(file_3d, model_3d_info, viewport_state['camera_info'], width, height)


class Load3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Load3D,
            Load3DAdvanced,
            Preview3D,
            Preview3DAdvanced,
            PreviewGaussianSplat,
            PreviewPointCloud,
        ]


async def comfy_entrypoint() -> Load3DExtension:
    return Load3DExtension()
