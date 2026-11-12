import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from comfy_api import v4
from comfy_api.internal.singleton import SingletonMetaclass


def test_version_check_compares_major_then_minor_then_patch(monkeypatch):
    monkeypatch.setattr(
        SingletonMetaclass, "_instances", dict(SingletonMetaclass._instances)
    )

    class OtherAPI(v4.ComfyAPI):
        VERSION = "1.2.3"

    other = OtherAPI()

    assert other.is_api_version_at_least("v1.2.3")
    assert other.is_api_version_at_least("v1.1.9")
    assert not other.is_api_version_at_least("v1.2.4")
    assert not other.is_api_version_at_least("v1.10.0")
    assert other.is_api_version_at_least("1.2.3")


@pytest.mark.parametrize(
    "version",
    [
        "v4",
        "4.0",
        "v4.0.0.1",
        "V4.0.0",
        " v4.0.0",
        "v4.0.0rc1",
        "v4.0.x",
        "v\u0664.0.0",
        "",
    ],
)
def test_version_check_refuses_a_version_not_written_as_three_numbers(version):
    with pytest.raises(ValueError):
        v4.eng.is_api_version_at_least(version)


def test_eng_uses_an_injected_api_instance(monkeypatch):
    instances = dict(SingletonMetaclass._instances)
    instances.pop(v4.ComfyAPI, None)
    monkeypatch.setattr(SingletonMetaclass, "_instances", instances)

    class InjectedAPI(v4.ComfyAPI):
        VERSION = "1.2.3"

    injected = InjectedAPI()
    v4.ComfyAPI.inject_instance(injected)

    assert v4.eng is injected
    assert v4.eng.get_api_version() == v4.ComfyAPISync().get_api_version() == (1, 2, 3)
    assert v4.ComfyAPISync().is_api_version_at_least("v1.2.3")
    assert not v4.ComfyAPISync().is_api_version_at_least("v1.2.4")


@pytest.mark.parametrize("name", list(v4.ComfyAPI.__annotations__))
def test_each_group_is_the_injected_provider(monkeypatch, name):
    group = v4.ComfyAPI.__annotations__[name]
    instances = dict(SingletonMetaclass._instances)
    instances.pop(v4.ComfyAPI, None)
    instances.pop(group, None)
    monkeypatch.setattr(SingletonMetaclass, "_instances", instances)
    provider = type("Provider", (group,), {})()

    group.inject_instance(provider)

    assert getattr(v4.eng, name) is provider


@pytest.mark.asyncio
async def test_latest_and_v4_packs_share_the_native_loader_and_executor(
    load_pack, run_prompt
):
    for package, node_id, expression in (
        ("latest", "LatestSquare", "value * value"),
        ("v4", "V4AddMajor", "value + eng.get_api_version()[0]"),
    ):
        await load_pack(
            dedent(f"""\
            from comfy_api.{package} import ComfyExtension, io, ui
            {"from comfy_api.v4 import eng" if package == "v4" else ""}

            class Scalar(io.ComfyNode):
                @classmethod
                def define_schema(cls):
                    return io.Schema(
                        node_id={node_id!r},
                        inputs=[io.Int.Input("value")],
                        outputs=[io.Int.Output()],
                        is_output_node=True,
                    )

                @classmethod
                def execute(cls, value):
                    result = {expression}
                    return io.NodeOutput(result, ui=ui.PreviewText(str(result)))

            class Extension(ComfyExtension):
                async def get_node_list(self):
                    return [Scalar]

            async def comfy_entrypoint():
                return Extension()
            """),
            f"pack_{package}",
        )

    executor, _ = await run_prompt(
        {
            "source": {"class_type": "LatestSquare", "inputs": {"value": 2}},
            "v4": {"class_type": "V4AddMajor", "inputs": {"value": ["source", 0]}},
            "sink": {"class_type": "LatestSquare", "inputs": {"value": ["v4", 0]}},
        },
        ["sink"],
    )

    assert executor.success, executor.status_messages
    assert executor.history_result["outputs"] == {
        "source": {"text": ["4"]},
        "v4": {"text": ["8"]},
        "sink": {"text": ["64"]},
    }


def test_package_import_does_not_load_model_or_device_management():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""\
                import sys
                import torch
                from comfy_api import v4

                v4.ComfyAPISync()
                v4.eng.get_api_version()
                assert "comfy.sd" not in sys.modules
                assert "comfy.model_management" not in sys.modules
                assert not torch.cuda.is_initialized()
            """),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
