import os
import json
from typing_extensions import override
from comfy_api.latest import io, ComfyExtension, ui
import folder_paths


class SaveTextNode(io.ComfyNode):
    """Save text content to .txt, .md, or .json."""

    FORMAT_EXTENSIONS = {
        "txt": "txt",
        "csv": "csv",
        "md": "md",
        "json": "json",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveText",
            search_aliases=["save text", "write text", "export text"],
            display_name="Save Text",
            category="text",
            description="Save text content to a file in the output directory.",
            inputs=[
                io.String.Input("text", force_input=True),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.Combo.Input("format", options=list(cls.FORMAT_EXTENSIONS), default="txt"),
            ],
            outputs=[io.String.Output(display_name="text")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, text, filename_prefix, format):
        extension = cls.FORMAT_EXTENSIONS.get(format)
        if extension is None:
            raise ValueError(f"Unsupported text format: {format!r}")

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            1,
            1,
        )

        file = f"{filename}_{counter:05}.{extension}"
        filepath = os.path.join(full_output_folder, file)

        if extension == "json":
            # tries to pretty print otherwise saves normally
            try:
                data = json.loads(text)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

        return io.NodeOutput(
            text,
            ui={
                "text": (text,),
                "files": [
                    ui.SavedResult(file, subfolder, io.FolderType.output)
                ]
            }
        )

class TextExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SaveTextNode
        ]

async def comfy_entrypoint() -> TextExtension:
    return TextExtension()
