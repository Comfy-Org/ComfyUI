import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


class WorkerConfigured(Exception):
    pass


@pytest.mark.parametrize(
    "total_ram_gb, cache_ram, expected_active, expected_inactive",
    [
        (128, [], 10, 32),
        (16, [8], 8, 8),
        (16, [8, 3], 8, 3),
    ],
)
def test_ram_cache_thresholds(total_ram_gb, cache_ram, expected_active, expected_inactive):
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(main_path.read_text(), filename=str(main_path))
    worker = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "prompt_worker")

    def capture_cache_args(*args, **kwargs):
        raise WorkerConfigured(kwargs["cache_args"])

    namespace = {
        "args": SimpleNamespace(cache_classic=False, cache_none=False, cache_lru=0, cache_ram=cache_ram),
        "comfy": SimpleNamespace(model_management=SimpleNamespace(total_ram=total_ram_gb * 1024)),
        "execution": SimpleNamespace(
            CacheType=SimpleNamespace(RAM_PRESSURE=0, CLASSIC=1, LRU=2, NONE=3),
            PromptExecutor=capture_cache_args,
        ),
    }
    exec(compile(ast.Module(body=[worker], type_ignores=[]), str(main_path), "exec"), namespace)  # noqa: S102 - trusted source from main.py

    with pytest.raises(WorkerConfigured) as configured:
        namespace["prompt_worker"](None, None, None)

    thresholds = configured.value.args[0]
    assert thresholds["ram"] == expected_active
    assert thresholds["ram_inactive"] == expected_inactive
