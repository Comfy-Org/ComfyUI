import asyncio
import base64
from io import BytesIO

import torch
from PIL import Image

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes.apis.gemini import (  # noqa: E402
    GeminiCandidate,
    GeminiContent,
    GeminiGenerateContentResponse,
    GeminiInlineData,
    GeminiPart,
)
from comfy_api_nodes.nodes_gemini import get_image_from_response  # noqa: E402


def image_part(mode, color):
    buffer = BytesIO()
    Image.new(mode, (4, 4), color).save(buffer, format="PNG")
    return GeminiPart(
        inlineData=GeminiInlineData(
            data=base64.b64encode(buffer.getvalue()).decode(),
            mimeType="image/png",
        )
    )


def response(*parts):
    return GeminiGenerateContentResponse(
        candidates=[GeminiCandidate(content=GeminiContent(parts=list(parts), role="model"))]
    )


def test_rgb_only_response_stays_three_channels():
    out = asyncio.run(get_image_from_response(response(image_part("RGB", (10, 20, 30)))))
    assert out.shape == (1, 4, 4, 3)


def test_mixed_rgb_and_rgba_parts_are_padded_to_the_same_width():
    out = asyncio.run(
        get_image_from_response(
            response(
                image_part("RGB", (10, 20, 30)),
                image_part("RGBA", (10, 20, 30, 0)),
            )
        )
    )
    assert out.shape == (2, 4, 4, 4)
    # the part that had no alpha is padded opaque, the transparent one is preserved
    assert out[0, ..., 3].min() == 1.0
    assert out[1, ..., 3].max() == 0.0
