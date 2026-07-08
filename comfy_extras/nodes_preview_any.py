import json
from comfy.comfy_types.node_typing import IO
import torch

# Preview Any - original implement from
# https://github.com/rgthree/rgthree-comfy/blob/main/py/display_any.py
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes
class PreviewAny():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"source": (IO.ANY, {})},
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "main"
    OUTPUT_NODE = True

    CATEGORY = "utilities"
    SEARCH_ALIASES = ["show output", "inspect", "debug", "print value", "show text"]

    def main(self, source=None):
        torch.set_printoptions(edgeitems=6)
        value = 'None'
        if isinstance(source, str):
            value = source
        elif isinstance(source, (int, float, bool)):
            value = str(source)
        elif source is not None:
            try:
                value = json.dumps(source, indent=4)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = 'source exists, but could not be serialized.'

        torch.set_printoptions()
        return {"ui": {"text": (value,)}, "result": (value,)}

NODE_CLASS_MAPPINGS = {
    "PreviewAny": PreviewAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewAny": "Preview as Text",
}
