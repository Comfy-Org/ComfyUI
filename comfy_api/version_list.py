from comfy_api.v3 import ComfyAPI_v3
from comfy_api.latest import ComfyAPI_latest
from comfy_api.v0_0_2 import ComfyAPIAdapter_v0_0_2
from comfy_api.v0_0_1 import ComfyAPIAdapter_v0_0_1
from comfy_api.internal import ComfyAPIBase

supported_versions: list[type[ComfyAPIBase]] = [
    # v3_0_0 - the V3 API under its numbered name
    ComfyAPI_v3,
    # Deprecated: use comfy_api.v3 for the same V3 API.
    ComfyAPI_latest,
    ComfyAPIAdapter_v0_0_2,
    ComfyAPIAdapter_v0_0_1,
]
