import os

import folder_paths
from comfy.text_encoders.rocmfpx import ROCmFPXCLIP
from comfy_api.latest import ComfyExtension, io


folder_paths.folder_names_and_paths["rocmfpx_text_encoders"] = (
    folder_paths.get_folder_paths("text_encoders"), {".gguf"})


class ROCmFPXCLIPLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ROCmFPXCLIPLoader",
            display_name="Load Text Encoder (ROCmFPX)",
            category="model/loaders",
            description="ROCmFPX Qwen Image text-to-image or MiniMax H3 conditioning, managed by ComfyUI model unloading. H3 uses native vision and token embeddings from the companion checkpoint.",
            is_experimental=True,
            inputs=[
                io.Combo.Input("gguf_name", options=folder_paths.get_filename_list("rocmfpx_text_encoders")),
                io.Combo.Input("type", options=["qwen_image_21", "minimax_h3"]),
                io.Combo.Input("companion_name", options=["none"] + folder_paths.get_filename_list("text_encoders")),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, gguf_name, type, companion_name) -> io.NodeOutput:
        dll_directory = os.environ.get("ROCMFPX_LIBRARY_PATH")
        if not dll_directory:
            raise RuntimeError("Set ROCMFPX_LIBRARY_PATH to the ROCmFPX library directory before starting ComfyUI")
        gguf_path = folder_paths.get_full_path_or_raise("rocmfpx_text_encoders", gguf_name)
        companion_path = None
        if type == "minimax_h3":
            if companion_name == "none":
                raise ValueError("Select the MiniMax H3 safetensors companion for native vision and embeddings")
            companion_path = folder_paths.get_full_path_or_raise("text_encoders", companion_name)
        clip = ROCmFPXCLIP(gguf_path, dll_directory, type, companion_path,
                          embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return io.NodeOutput(clip)


class ROCmFPXExtension(ComfyExtension):
    async def get_node_list(self):
        return [ROCmFPXCLIPLoader]


async def comfy_entrypoint():
    return ROCmFPXExtension()
