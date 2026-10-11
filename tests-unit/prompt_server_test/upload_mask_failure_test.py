import ast
import asyncio
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import tempfile
import unittest
from aiohttp import web
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _handlers(output_dir, assets):
    source = Path(__file__).resolve().parents[2] / "server.py"
    tree = ast.parse(source.read_text())
    functions = [
        next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
        for name in ("image_upload", "upload_mask")
    ]
    for function in functions:
        function.decorator_list = []
    factory = ast.FunctionDef(
        name="bind",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=functions + [ast.Return(ast.Tuple([ast.Name("image_upload", ast.Load()), ast.Name("upload_mask", ast.Load())], ast.Load()))],
        decorator_list=[],
    )
    namespace = {
        "os": os, "json": json, "web": web, "Image": Image, "PngInfo": PngInfo,
        "get_dir_by_type": lambda kind: (str(output_dir), kind or "input"),
        "compare_image_hash": lambda *_: False,
        "folder_paths": SimpleNamespace(
            annotated_filepath=lambda filename: (filename, None),
            get_directory_by_type=lambda _: str(output_dir),
        ),
    }
    exec(compile(ast.fix_missing_locations(ast.Module([factory], [])), str(source), "exec"), namespace)
    return namespace["bind"](SimpleNamespace(asset_manager=assets))


def _post(original_ref=None):
    stream = io.BytesIO()
    Image.new("RGBA", (2, 2), (0, 0, 0, 64)).save(stream, format="PNG")
    stream.seek(0)
    post = {"image": SimpleNamespace(filename="mask.png", file=stream)}
    if original_ref is not None:
        post["original_ref"] = json.dumps(original_ref)
    return post


def _mask_request(post):
    async def read_post():
        return post
    return SimpleNamespace(post=read_post)


class UploadMaskFailureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.assets = SimpleNamespace(register_upload=Mock(return_value=None))
        self.image_upload, self.upload_mask = _handlers(self.output, self.assets)

    def test_rejected_mask_does_not_report_success_or_register_an_asset(self):
        for reference, status in [
            ({"filename": "/invalid.png"}, 400),
            ({"filename": "original.png", "subfolder": ".."}, 403),
            ({"filename": "missing.png"}, 404),
        ]:
            with self.subTest(reference=reference):
                response = asyncio.run(self.upload_mask(_mask_request(_post(reference))))
                self.assertEqual(response.status, status)
                self.assertFalse((self.output / "mask.png").exists())
                self.assets.register_upload.assert_not_called()

    def test_successful_mask_saves_alpha_and_registers_once(self):
        Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(self.output / "original.png")
        response = asyncio.run(self.upload_mask(_mask_request(_post({"filename": "original.png"}))))
        self.assertEqual(response.status, 200)
        with Image.open(self.output / "mask.png") as saved:
            self.assertEqual(saved.getpixel((0, 0)), (255, 0, 0, 64))
        self.assets.register_upload.assert_called_once()

    def test_plain_upload_still_saves_and_registers_once(self):
        response = self.image_upload(_post())
        self.assertEqual(response.status, 200)
        self.assertTrue((self.output / "mask.png").is_file())
        self.assets.register_upload.assert_called_once()
