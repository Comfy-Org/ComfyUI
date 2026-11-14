import asyncio
import importlib.util
import os
import py_compile
import re
import signal
import subprocess
import sys

import psutil
import pytest

import folder_paths
from comfy.cli_args import args
from comfy_api.v4 import _pack_host, _pack_launch, _pack_processes
from conftest import MODULE, PROBE, evaluate, needs_pack_process, wait_until

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

    def wait(self):
        self.calls.append("wait")
        return self.process.wait()


def per_job(start: dict):
    """A host's launcher: the pack process gets the job's folders instead of ComfyUI's."""
    launches.append(start)
    process, sock = _pack_launch.launch({**start, "folders": dict(job_folders)})
    processes.append(Recorded(process))
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
                "(folders.get_output_directory(), folders.output_directory, "
                "folders.get_input_directory(), folders.get_temp_directory(), "
                "folders.get_directory_by_type('temp'), folders.get_user_directory(), "
                "OUTPUT_AT_IMPORT, open(folders.get_annotated_filepath('given.txt')).read())",
            )
        )
        saved = asyncio.run(evaluate(process, SAVE))
    finally:
        _pack_processes.unload(process.module_path)

    jobs = job_folders
    assert answers == (
        jobs["output"],
        jobs["output"],
        jobs["input"],
        jobs["temp"],
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

    pack = str(tmp_path / "pack.py")

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
        str(tmp_path / "pack.py"),
        "add_model_folder_path",
        ("v4_again", str(listed)),
        {},
    )

    assert folder_paths.get_folder_paths("v4_again") == [str(listed)]
