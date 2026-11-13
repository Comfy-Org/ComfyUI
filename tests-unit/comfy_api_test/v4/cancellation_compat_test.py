import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from comfy import model_management
from comfy_api import v4
from conftest import CANCEL


def test_exception_leaf_imports_without_engine_or_site_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; "
            "from comfy_api._exceptions import InterruptProcessingException; "
            "assert 'torch' not in sys.modules; "
            "assert 'comfy.model_management' not in sys.modules; "
            "assert 'comfy_api.v4' not in sys.modules; "
            "assert not issubclass(InterruptProcessingException, Exception)",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr


def test_existing_exception_pickles_and_current_round_trip_keep_their_args():
    old = pickle.loads(
        b"ccomfy.model_management\nInterruptProcessingException\n(Vcancelled\ntR."
    )
    current = pickle.loads(pickle.dumps(v4.InterruptProcessingException("cancelled")))

    assert type(old) is type(current) is v4.InterruptProcessingException
    assert (
        model_management.InterruptProcessingException is v4.InterruptProcessingException
    )
    assert old.args == current.args == ("cancelled",)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_pack_catches_native_cancel_before_executor_handles_it(
    run_node, synchronous
):
    executor, server = await run_node(
        CANCEL + "try:\n"
        "    {call}.execution.check_interrupted()\n"
        "except InterruptProcessingException as error:\n"
        "    assert type(error) is InterruptProcessingException\n"
        "    raise\n",
        synchronous,
    )

    assert not executor.success
    assert executor.status_messages[-1][0] == "execution_interrupted"
