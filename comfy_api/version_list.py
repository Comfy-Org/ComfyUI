from comfy_api.latest import ComfyAPI_latest
from comfy_api.v0_0_3 import ComfyAPIAdapter_v0_0_3
from comfy_api.v0_0_2 import ComfyAPIAdapter_v0_0_2
from comfy_api.v0_0_1 import ComfyAPIAdapter_v0_0_1
from comfy_api.internal import ComfyAPIBase

supported_versions: list[type[ComfyAPIBase]] = [
    ComfyAPI_latest,
    ComfyAPIAdapter_v0_0_3,
    ComfyAPIAdapter_v0_0_2,
    ComfyAPIAdapter_v0_0_1,
]

