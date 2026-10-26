import os

from typing_extensions import override

import comfy.model_state
import folder_paths
from comfy_api.latest import ComfyExtension, io


def state_path(filename):
    output_dir = folder_paths.get_output_directory()
    path = os.path.abspath(os.path.join(output_dir, filename))
    if not folder_paths.is_within_directory(output_dir, path):
        raise ValueError(f"Model state files must be inside the output folder: {filename}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


class OpenModelState(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OpenModelState",
            display_name="Open Model State",
            category="model_state",
            search_aliases=["KV cache", "conversation", "resume"],
            description="Opens a model state file in the output folder, such as a text generation conversation, so a sampler can continue from it. A state created here is written by the first sampler that uses it.",
            inputs=[
                io.String.Input("filename", default="model_states/ComfyUI.safetensors"),
                io.Combo.Input("create", options=["if_missing", "never", "always"], default="if_missing", tooltip="always recreates the file on every run; never requires an existing file."),
            ],
            outputs=[io.ModelState.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, filename, create):
        if create == "always":
            return float("NaN")
        path = state_path(filename)
        return os.stat(path).st_mtime_ns if os.path.exists(path) else None

    @classmethod
    def execute(cls, filename, create) -> io.NodeOutput:
        return io.NodeOutput(comfy.model_state.ModelState(state_path(filename), create))


class SaveModelState(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveModelState",
            display_name="Save Model State",
            category="model_state",
            description="Writes a model state's current values back to its file.",
            is_output_node=True,
            inputs=[io.ModelState.Input("state")],
            outputs=[io.ModelState.Output(tooltip="The saved state, for chaining or ending a loop.")],
        )

    @classmethod
    def execute(cls, state) -> io.NodeOutput:
        state.save()
        return io.NodeOutput(state)


class CopyModelState(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CopyModelState",
            display_name="Copy Model State",
            category="model_state",
            description="Saves a model state and copies its file to a new file in the output folder.",
            is_output_node=True,
            inputs=[
                io.ModelState.Input("state"),
                io.String.Input("filename", default="model_states/ComfyUI_copy.safetensors"),
            ],
            outputs=[io.ModelState.Output()],
        )

    @classmethod
    def execute(cls, state, filename) -> io.NodeOutput:
        return io.NodeOutput(state.copy(state_path(filename)))


class ModelStateExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            OpenModelState,
            SaveModelState,
            CopyModelState,
        ]


async def comfy_entrypoint() -> ModelStateExtension:
    return ModelStateExtension()
