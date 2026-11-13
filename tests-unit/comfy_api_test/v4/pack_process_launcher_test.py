import asyncio
import importlib.util
import os
import py_compile
import re
import signal
import socket
import subprocess
import sys
import threading

import psutil
import pytest

import folder_paths
from comfy.cli_args import args
from comfy_api.v4 import _channel, _pack_host, _pack_launch, _pack_processes
from conftest import (
    MODULE,
    PROBE,
    evaluate,
    needs_pack_process,
    start_probe,
    wait_until,
)

pytestmark = needs_pack_process

# What the launchers below were given and returned, by the test that set them.
launches: list[dict] = []
processes: list["Recorded"] = []
job_folders: dict[str, str] = {}


class Recorded:
    """A launcher's process object that records what ComfyUI does with it."""

    def __init__(self, process):
        self.process, self.pid, self.calls = process, process.pid, []

    def kill(self):
        self.calls.append("kill")
        self.process.kill()

    def poll(self):
        return self.process.poll()

    def wait(self, timeout=None):
        code = self.process.wait(timeout)
        self.calls.append("wait")
        return code


def per_job(start: dict):
    """A host's launcher: the pack process gets the job's folders instead of ComfyUI's."""
    launches.append(start)
    process, sock = _pack_launch.launch({**start, "folders": dict(job_folders)})
    processes.append(Recorded(process))
    return processes[-1], sock


class WaitFails(Recorded):
    def wait(self, timeout=None):
        raise ChildProcessError("this launcher cannot wait")


def wait_fails(start: dict):
    """A launcher whose process object cannot wait for the process."""
    process, sock = _pack_launch.launch(start)
    processes.append(WaitFails(process))
    return processes[-1], sock


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    """Make `per_job` the `--pack-launcher`, with folders of the test's own."""
    monkeypatch.setattr(args, "pack_launcher", f"{__name__}:per_job")
    for name in ("output", "input", "temp", "user"):
        job_folders[name] = str(tmp_path / f"job_{name}")
        os.mkdir(job_folders[name])
    yield
    launches.clear()
    processes.clear()
    job_folders.clear()


def load(folder):
    path = folder / f"{MODULE}.py"
    path.write_text(PROBE, encoding="utf-8")
    asyncio.run(_pack_host.load(str(path), MODULE))
    return _pack_processes.processes[str(path)]


SAVE = (
    "__import__('comfy_api.v4', fromlist=['ui']).ui.ImageSaveHelper.save_images("
    "torch.rand(1, 8, 8, 3), 'v4_job', io.FolderType.output, None)[0]['filename']"
)


def test_the_launcher_the_option_names_starts_the_pack_process_with_the_folders_it_gives(
    launcher, tmp_path
):
    (tmp_path / "job_input" / "given.txt").write_text("job input", encoding="utf-8")
    process = load(tmp_path)
    try:
        answers = asyncio.run(
            evaluate(
                process,
                "(folders.get_output_directory(), "
                "folders.get_input_directory(), folders.get_temp_directory(), "
                "folders.get_user_directory(), "
                "OUTPUT_AT_IMPORT, open(folders.get_annotated_filepath('given.txt')).read())",
            )
        )
        saved = asyncio.run(evaluate(process, SAVE))
    finally:
        _pack_processes.unload(process.module_path)

    jobs = job_folders
    assert answers == (
        jobs["output"],
        jobs["input"],
        jobs["temp"],
        jobs["user"],
        jobs["output"],
        "job input",
    )
    assert os.listdir(jobs["output"]) == [saved]
    assert not os.path.exists(os.path.join(folder_paths.get_output_directory(), saved))
    [start] = launches
    assert (start["pack"], start["name"]) == (process.module_path, MODULE)
    assert start["folders"] == {
        name: getattr(folder_paths, f"get_{name}_directory")()
        for name in ("output", "input", "temp", "user")
    }


def test_comfyui_ends_and_restarts_a_pack_process_only_through_what_its_launcher_returned(
    launcher, tmp_path
):
    process = load(tmp_path)
    first = processes[0]
    program = asyncio.run(evaluate(process, "start_program(own_session=True)"))

    process.kill()  # as after a cancel the pack ignores
    wait_until(lambda: not process.ready, "the process to be seen stopped")
    again = asyncio.run(evaluate(process, "os.getpid()"))
    _pack_processes.unload(process.module_path)
    wait_until(lambda: "wait" in processes[1].calls, "the unloaded process reaped")

    assert first.calls[0] == "kill" and first.calls[-1] == "wait"
    assert first.poll() is not None
    assert not psutil.pid_exists(program) or psutil.Process(program).status() == (
        psutil.STATUS_ZOMBIE
    )
    assert len(launches) == 2 and again == processes[1].pid != first.pid
    assert processes[1].calls[0] == "kill" and processes[1].poll() is not None


@pytest.mark.parametrize("name", ["v4_no_such_launcher:launch", "os.path", "os:sep"])
def test_a_launcher_that_cannot_be_used_fails_the_load_saying_why(
    monkeypatch, tmp_path, name
):
    monkeypatch.setattr(args, "pack_launcher", name)
    path = tmp_path / f"{MODULE}.py"
    path.write_text(PROBE, encoding="utf-8")

    with pytest.raises(
        RuntimeError, match=re.escape(f"--pack-launcher {name}: cannot")
    ):
        asyncio.run(_pack_host.load(str(path), MODULE))
    assert str(path) not in _pack_processes.processes


def test_comfyui_answers_a_pack_process_only_about_its_model_folders(tmp_path):
    (tmp_path / "secret.txt").write_text("not the pack's", encoding="utf-8")

    pack = _pack_host.PackProcess(str(tmp_path / "pack.py"), "pack")

    with pytest.raises(LookupError, match="not answered by ComfyUI"):
        _pack_host._folders(pack, "recursive_search", (str(tmp_path),), {})
    assert _pack_host._folders(pack, "get_folder_paths", ("checkpoints",), {}) == (
        folder_paths.get_folder_paths("checkpoints")
    )


def test_a_file_one_node_wrote_in_the_temp_folder_is_read_by_another_node_of_the_pack(
    launcher, tmp_path
):
    process = load(tmp_path)
    try:
        written = asyncio.run(
            evaluate(
                process,
                "(p := os.path.join(folders.get_temp_directory(), 'frames.txt'))"
                " and open(p, 'w').write('from the first node') and p",
            )
        )
        read = asyncio.run(
            evaluate(
                process,
                "open(os.path.join(folders.get_temp_directory(), 'frames.txt')).read()",
            )
        )
    finally:
        _pack_processes.unload(process.module_path)

    assert read == "from the first node"
    assert os.path.dirname(written) == job_folders["temp"]


UNDER_A_POLICY = """\
import runpy, sys
sys.path.insert(0, {root!r})
from app import governance
governance.initialize = lambda: governance.set_custom_node_policy("blocklist", frozenset(), {{}})
runpy.run_path({main!r}, run_name="__main__")
"""

POLICY_PACK = """\
from comfy_api.v4 import ComfyExtension, io


class Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="{node_id}")

    @classmethod
    def execute(cls):
        return io.NodeOutput()


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Node]


async def comfy_entrypoint():
    return Extension()
"""


def under_a_policy(start: dict):
    """A launcher whose pack process finds a custom-node policy when it applies ComfyUI's."""
    script = UNDER_A_POLICY.format(root=os.getcwd(), main=_pack_launch.pack_main())
    return _pack_launch.launch(start, [sys.executable, "-c", script])


@pytest.mark.parametrize(
    ("launcher_name", "run"),
    [("under_a_policy", "FromSource"), (None, "OfBytecode")],
    ids=["under a policy", "positive control: without one"],
)
def test_under_a_custom_node_policy_a_pack_process_runs_the_packs_source_not_cached_bytecode(
    tmp_path, monkeypatch, launcher_name, run
):
    monkeypatch.setattr(
        args, "pack_launcher", launcher_name and f"{__name__}:{launcher_name}"
    )
    path = tmp_path / "v4_policy_pack.py"
    path.write_text(POLICY_PACK.format(node_id="OfBytecode"), encoding="utf-8")
    py_compile.compile(str(path), cfile=importlib.util.cache_from_source(str(path)))
    written = os.stat(path)
    # The cached bytecode still passes for this source: same size, same time.
    path.write_text(POLICY_PACK.format(node_id="FromSource"), encoding="utf-8")
    os.utime(path, ns=(written.st_atime_ns, written.st_mtime_ns))

    try:
        module = asyncio.run(_pack_host.load(str(path), "v4_policy_pack"))
        node_ids = [
            node.GET_SCHEMA().node_id for node in module.comfy_entrypoint().nodes
        ]
    finally:
        _pack_processes.unload(str(path))

    assert node_ids == [run]


def test_restarting_a_pack_process_leaves_comfyui_no_more_open_descriptors(tmp_path):
    process = load(tmp_path)

    def restart():
        stopping = process.process
        process.kill()
        wait_until(lambda: stopping.poll() is not None, "the process reaped")
        wait_until(lambda: not process.ready, "the process to be seen stopped")
        asyncio.run(evaluate(process, "1"))

    try:
        restart()
        before = psutil.Process().num_fds()
        for _ in range(5):
            restart()
        after = psutil.Process().num_fds()
    finally:
        _pack_processes.unload(process.module_path)

    assert after == before


ANSWERS_ITS_LOAD_WITH = """
import pickle, socket, sys, time

start = pickle.load(sys.stdin.buffer)
sys.argv = start["argv"]
from comfy_api.v4 import _channel, _codec


class Side:
    peer, io_types = "ComfyUI", {{}}


sock = socket.socket(fileno=start["socket"])
head, _, _ = _channel.read_frame(sock, bool)
call = pickle.loads(bytes(head))
body, _ = _codec.dumps({answer!r}, Side(), call["chain"])
reply = {{"to": call["id"], "chain": call["chain"]}}
_channel.write_frame(sock, pickle.dumps(reply, protocol=5), body)
time.sleep(60)
"""
answers: list[dict] = []


def answers_its_load(start: dict):
    """A launcher whose process is not a pack process: it answers the load as the test says."""
    script = ANSWERS_ITS_LOAD_WITH.format(answer=answers[-1])
    repository = os.path.dirname(os.path.dirname(os.path.dirname(_pack_launch.__file__)))
    process, sock = _pack_launch.launch(
        start,
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": repository},
    )
    processes.append(Recorded(process))
    return processes[-1], sock


@pytest.mark.parametrize(
    "answer",
    [
        {"nodes": 5, "routes": [], "web_directory": None},
        {"nodes": [], "routes": []},
    ],
    ids=["nodes that are not a list", "no web folder entry"],
)
def test_a_process_that_answers_its_load_with_something_else_is_ended(
    tmp_path, monkeypatch, answer
):
    monkeypatch.setattr(args, "pack_launcher", f"{__name__}:answers_its_load")
    answers.append(answer)
    path = str(tmp_path / f"{MODULE}.py")
    try:
        with pytest.raises((TypeError, KeyError)):
            asyncio.run(_pack_host.load(path, MODULE))
        wait_until(lambda: processes[-1].poll() is not None, "the process to end")
        assert path not in _pack_processes.processes
    finally:
        processes[-1].kill()
        answers.clear()
        processes.clear()


def test_a_pack_process_comfyui_ends_leaves_no_watchdog_to_act_later(tmp_path):
    process = start_probe(tmp_path)
    watchdog = process.process.watchdog

    _pack_processes.unload(process.module_path)

    assert watchdog.poll() is not None
    assert process.process.process.returncode is not None


def test_the_default_launchers_process_can_be_waited_for_with_a_timeout_and_twice():
    process, sock = _pack_launch.launch(
        {}, [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    sock.close()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.1)
    finally:
        process.kill()

    assert process.wait() == process.wait() == -signal.SIGKILL


def test_a_pack_process_reads_comfyuis_model_folders_as_they_are_now(
    tmp_path, monkeypatch
):
    process = load(tmp_path)
    folder = tmp_path / "added_after_start"
    folder.mkdir()
    (folder / "weights.bin").write_bytes(b"")
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models_now"))
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths, "v4_added", ([str(folder)], {".bin"})
    )
    try:
        seen = asyncio.run(
            evaluate(
                process,
                "folders.models_dir, folders.get_full_path('v4_added', 'weights.bin')",
            )
        )
    finally:
        _pack_processes.unload(process.module_path)

    assert seen == (str(tmp_path / "models_now"), str(folder / "weights.bin"))


def test_a_folder_comfyui_lists_is_a_model_folder_however_it_is_spelled(
    tmp_path, monkeypatch
):
    listed = tmp_path / "listed"
    listed.mkdir()
    (tmp_path / "other_spelling").symlink_to(listed)
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths,
        "v4_listed",
        ([str(tmp_path / "other_spelling")], set()),
    )
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "v4_again", ([], set()))

    _pack_host._folders(
        _pack_host.PackProcess(str(tmp_path / "pack.py"), "pack"),
        "add_model_folder_path",
        ("v4_again", str(listed)),
        {},
    )

    assert folder_paths.get_folder_paths("v4_again") == [str(listed)]


def test_a_launcher_whose_wait_raises_fails_the_node_and_the_pack_starts_again(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(args, "pack_launcher", f"{__name__}:wait_fails")
    process = load(tmp_path)
    try:
        with pytest.raises(
            _channel.PackStopped, match="exit code unknown: .*cannot wait"
        ):
            asyncio.run(asyncio.wait_for(evaluate(process, "os._exit(3)"), 30))
        wait_until(lambda: not process.ready, "the process to be seen stopped")
        again = asyncio.run(asyncio.wait_for(evaluate(process, "os.getpid()"), 30))
        restarted = processes[-1].pid
    finally:
        _pack_processes.unload(process.module_path)
        for recorded in processes:
            recorded.process.wait()
        processes.clear()

    assert again == restarted


class Pretends:
    """A process object for a process that does not exist."""

    killed = False

    def kill(self):
        self.killed = True

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


pretends = Pretends()


def returns_none(start):
    return None


def returns_three(start):
    return pretends, socket.socketpair()[0], "more"


def returns_swapped(start):
    return socket.socketpair()[0], pretends


def returns_no_socket(start):
    return pretends, "not a socket"


@pytest.mark.parametrize(
    "launcher_name",
    ["returns_none", "returns_three", "returns_swapped", "returns_no_socket"],
)
def test_a_launcher_that_returns_something_else_fails_the_load_naming_it(
    monkeypatch, tmp_path, launcher_name
):
    name = f"{__name__}:{launcher_name}"
    monkeypatch.setattr(args, "pack_launcher", name)
    monkeypatch.setattr(pretends, "killed", False)
    path = tmp_path / f"{MODULE}.py"
    path.write_text(PROBE, encoding="utf-8")

    with pytest.raises(TypeError, match=re.escape(f"--pack-launcher {name} returned")):
        asyncio.run(_pack_host.load(str(path), MODULE))

    assert str(path) not in _pack_processes.processes
    assert pretends.killed == (launcher_name == "returns_no_socket")


def silent(start: dict):
    """A launcher whose process lives and never answers."""
    return _pack_launch.launch(
        start, [sys.executable, "-c", "import time; time.sleep(60)"]
    )


def test_a_pack_process_that_does_not_answer_its_load_is_named_in_the_log(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(args, "pack_launcher", f"{__name__}:silent")
    monkeypatch.setattr(_pack_host, "SLOW_LOAD", 0.2)
    path = tmp_path / f"{MODULE}.py"
    path.write_text(PROBE, encoding="utf-8")
    failures = []

    def load_it():
        try:
            asyncio.run(_pack_host.load(str(path), MODULE))
        except _channel.PackStopped as error:
            failures.append(error)

    loading = threading.Thread(target=load_it, daemon=True)
    loading.start()
    try:
        wait_until(lambda: "has not loaded" in caplog.text, "the pack named in the log")
    finally:
        wait_until(lambda: str(path) in _pack_processes.processes, "the process listed")
        _pack_processes.unload(str(path))
        loading.join(10)

    assert f"pack {MODULE}.py has not loaded 0.2 s after its process started" in (
        caplog.text
    )
    assert len(failures) == 1


POLICY_MISSING = """\
import runpy, sys
sys.path.insert(0, {root!r})
from app import governance
governance.GOVERNANCE_REQUIRED = True
governance._POLICY_PATH = governance.Path({missing!r})
runpy.run_path({main!r}, run_name="__main__")
"""


def policy_missing(start: dict):
    """A launcher whose pack process cannot read the custom-node policy ComfyUI requires."""
    script = POLICY_MISSING.format(
        root=os.getcwd(),
        missing=start["folders"]["temp"] + "/none",
        main=_pack_launch.pack_main(),
    )
    return _pack_launch.launch(start, [sys.executable, "-c", script])


def test_a_pack_process_that_cannot_apply_the_policy_says_why_in_comfyuis_log(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(args, "pack_launcher", f"{__name__}:policy_missing")
    path = tmp_path / f"{MODULE}.py"
    path.write_text(PROBE, encoding="utf-8")

    with pytest.raises(_channel.PackStopped, match="exit code 1"):
        asyncio.run(_pack_host.load(str(path), MODULE))

    assert "could not apply your organization's policy" in caplog.text
    assert str(path) not in _pack_processes.processes
