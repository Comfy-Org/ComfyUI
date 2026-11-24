"""Tests for the mask upload path of ``server.py`` (issue #1488).

``/upload/mask`` copies the metadata of the image the mask belongs to. When
that image lives on another drive, the containment check rejects it with a
403 — and that rejection has to reach the client. It used to be swallowed:
``store_image_upload`` discarded the callback's response and registered the
upload anyway, so the request could report success for a mask that was never
written.
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
import torch
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from comfy.cli_args import args

if not torch.cuda.is_available():
    # The comfy modules reach for the CUDA device at import time.
    args.cpu = True

import folder_paths  # noqa: E402
import server  # noqa: E402


@pytest.fixture
def prompt_server():
    """A PromptServer with the asset manager stubbed out.

    The asset manager only matters for the asset registration the upload
    performs; the test asserts on whether it happened at all.
    """
    asset_manager = MagicMock()
    asset_manager.register_upload.return_value = None
    loop = asyncio.new_event_loop()
    instance = server.PromptServer(loop, asset_manager)
    instance.add_routes()
    return instance


@pytest.mark.asyncio
async def test_mask_upload_on_another_drive_is_rejected(prompt_server, tmp_path, monkeypatch):
    """A mask whose original image is outside the output dir gets a 403.

    The 403 comes from the containment check on the original image's
    subfolder. It must be the response the client sees, and nothing may be
    registered as an uploaded asset.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr(
        folder_paths,
        "annotated_filepath",
        lambda filename: ("original.png", str(output_dir)),
    )

    form = FormData()
    form.add_field(
        "image",
        b"mask-bytes",
        filename="mask.png",
        content_type="image/png",
    )
    # A subfolder that escapes the output directory stands in for the
    # cross-drive case: the containment check rejects it either way.
    form.add_field(
        "original_ref",
        json.dumps(
            {
                "filename": "original.png",
                "type": "output",
                "subfolder": "..",
            },
        ),
    )

    async with TestClient(TestServer(prompt_server.app)) as client:
        response = await client.post("/upload/mask", data=form)

    assert response.status == 403, (
        f"expected the cross-drive mask to be rejected with 403, got {response.status}"
    )
    prompt_server.asset_manager.register_upload.assert_not_called()
