import asyncio
import json

import pytest
import torch
from PIL import Image

from comfy.cli_args import args
from comfy_api.latest import _sdk


WORKFLOW = json.dumps({"authored": "workflow caf\u00e9 \U0001f408", "nodes": [], "__ref__": "literal"})
PARAMETERS = "native metadata caf\u00e9 \U0001f408\nSteps: 20, Seed: 42"


def save(pixels, filenames, **options):
    async def run():
        refs = _sdk.InProcessRefResolver()
        plan = _sdk.ExecutionPlan("webp-workflow", "7", "Save")
        ctx = _sdk.InProcessCtxProvider().build(plan)
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            image = _sdk.ImageRef._wrap(await refs.create("IMAGE", pixels))
            return await ctx.output.save_images(image, filenames=filenames, **options)
    return asyncio.run(run())


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("save_metadata", [False, True])
@pytest.mark.parametrize("parameters", [None, PARAMETERS])
def test_explicit_workflow_is_independent_and_keeps_parameters(tmp_path, monkeypatch, disabled, save_metadata, parameters):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(args, "disable_metadata", disabled)
    pixels = torch.linspace(0, 1, 2 * 4 * 5 * 3).reshape(2, 4, 5, 3)
    result = save(pixels, ["first.webp", "second.webp"], image_format="webp", quality=80,
                  lossless=False, save_metadata=save_metadata, webp_workflow_json=WORKFLOW,
                  a1111_parameters=parameters)
    assert result == {"images": [{"filename": name, "subfolder": "", "type": "output"}
                                 for name in ["first.webp", "second.webp"]]}
    for name in ["first.webp", "second.webp"]:
        with Image.open(tmp_path / name) as saved:
            assert saved.format == "WEBP" and saved.size == (5, 4)
            assert saved.getexif()[0x010E] == "Workflow:" + WORKFLOW
            comment = saved.getexif().get_ifd(0x8769).get(0x9286)
            assert comment == (None if parameters is None else b"UNICODE\0" + parameters.encode("utf-16-be"))


def test_none_preserves_existing_parameter_only_bytes(tmp_path, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    pixels = torch.zeros((1, 4, 5, 3))
    options = dict(image_format="webp", save_metadata=False, a1111_parameters=PARAMETERS)
    save(pixels, ["omitted.webp"], **options)
    save(pixels, ["none.webp"], webp_workflow_json=None, **options)
    assert (tmp_path / "omitted.webp").read_bytes() == (tmp_path / "none.webp").read_bytes()
    with Image.open(tmp_path / "none.webp") as saved:
        assert 0x010E not in saved.getexif()


@pytest.mark.parametrize("text", ['null', '[]', 'false', '42', '{ "b": 2, "a": 1, "a": 3 }',
                                  json.dumps({"nonfinite": [float("nan"), float("inf")], "nul": "\0"})])
def test_workflow_json_is_not_reserialized_or_restricted_to_an_object(tmp_path, monkeypatch, text):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    save(torch.zeros((1, 1, 1, 3)), ["actual.webp"], image_format="webp",
         save_metadata=False, webp_workflow_json=text)
    with Image.open(tmp_path / "actual.webp") as saved:
        assert saved.getexif()[0x010E] == "Workflow:" + text


def test_description_ceiling_counts_prefix_and_nul(tmp_path, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    text = json.dumps("x" * (1024 * 1024 - len("Workflow:") - 3))
    assert len(text) + len("Workflow:") + 1 == 1024 * 1024
    save(torch.zeros((1, 1, 1, 3)), ["limit.webp"], image_format="webp",
         save_metadata=False, webp_workflow_json=text)
    with Image.open(tmp_path / "limit.webp") as saved:
        assert saved.getexif()[0x010E] == "Workflow:" + text
    with pytest.raises(ValueError, match="1 MiB EXIF description limit"):
        save(torch.zeros((1, 1, 1, 3)), ["over.webp"], image_format="webp",
             webp_workflow_json=text + " ")
    assert not (tmp_path / "over.webp").exists()


@pytest.mark.parametrize("text,format,error,message", [
    ({}, "webp", TypeError, "must be a string or None"),
    (b"{}", "webp", TypeError, "must be a string or None"),
    ("{}", "png", ValueError, "supported only for WebP"),
    ("{}", "jpeg", ValueError, "supported only for WebP"),
    ("{}", "avif", ValueError, "supported only for WebP"),
    ("", "webp", ValueError, "must contain one JSON value"),
    ("{} []", "webp", ValueError, "must contain one JSON value"),
    ('{"x":"\u00e9"}', "webp", ValueError, "must use ASCII JSON text"),
    ('"\ud800"', "webp", ValueError, "must use ASCII JSON text"),
    (" " * (1024 * 1024), "webp", ValueError, "1 MiB EXIF description limit"),
])
def test_invalid_workflow_refuses_before_output_paths(tmp_path, monkeypatch, text, format, error, message):
    import folder_paths

    output = tmp_path / "not-created"
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    with pytest.raises(error, match=message):
        save(torch.zeros((1, 1, 1, 3)), ["nested/fail." + format],
             image_format=format, webp_workflow_json=text)
    assert not output.exists()


def test_parameter_validation_precedes_workflow_validation(tmp_path, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "not-created"))
    with pytest.raises(ValueError, match="a1111_parameters must contain valid Unicode"):
        save(torch.zeros((1, 1, 1, 3)), ["fail.webp"], image_format="webp",
             a1111_parameters="\ud800", webp_workflow_json="bad")
    assert not (tmp_path / "not-created").exists()
