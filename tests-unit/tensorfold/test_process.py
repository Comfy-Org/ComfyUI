import os
import sys
import time

import pytest

from comfy.tensorfold.process import LOG_LIMIT, run_native


fcntl = pytest.importorskip("fcntl", reason="Native process-group tests require a POSIX host")


class Interrupted(Exception):
    pass


def test_real_child_streams_progress_without_shell(tmp_path):
    progress = []
    command = [sys.executable, "-c", "import sys; print('step 1/2: running', flush=True); print('step 2/2: done', file=sys.stderr)"]
    tail = run_native(command, cwd=tmp_path, interrupt=lambda: None, progress=lambda n, total: progress.append((n, total)))
    assert progress == [(1, 2), (2, 2)]
    assert "done" in tail


def test_paths_and_arguments_are_literal(tmp_path):
    text = "$(touch SHOULD_NOT_EXIST); spaces and 'quotes'"
    command = [sys.executable, "-c", "import sys; print(sys.argv[1])", text]
    assert text in run_native(command, cwd=tmp_path, interrupt=lambda: None, progress=lambda *args: None)
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_failed_child_never_looks_successful(tmp_path):
    with pytest.raises(RuntimeError, match="status 7.*allocation failed"):
        run_native([sys.executable, "-c", "print('allocation failed'); raise SystemExit(7)"], cwd=tmp_path,
                   interrupt=lambda: None, progress=lambda *args: None)


def test_output_tail_is_bounded(tmp_path):
    tail = run_native([sys.executable, "-c", "print('a' * 2000000); print('complete')"], cwd=tmp_path,
                      interrupt=lambda: None, progress=lambda *args: None)
    assert len(tail.encode()) <= LOG_LIMIT
    assert tail.endswith("complete\n")


def test_cancel_stops_child_and_reaps_it(tmp_path):
    pidfile = tmp_path / "pid"
    code = "import os,time,pathlib; pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(60)"

    def interrupt():
        if pidfile.exists():
            raise Interrupted()

    started = time.monotonic()
    with pytest.raises(Interrupted):
        run_native([sys.executable, "-c", code], cwd=tmp_path, interrupt=interrupt,
                   progress=lambda *args: None)
    assert time.monotonic() - started < 3
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_hung_child_has_deadline(tmp_path):
    with pytest.raises(TimeoutError, match="deadline"):
        run_native([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path,
                   interrupt=lambda: None, progress=lambda *args: None, timeout=0.15)


def test_environment_override_is_scoped_to_child(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    tail = run_native([sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"], cwd=tmp_path,
                      interrupt=lambda: None, progress=lambda *args: None, environment={"CUDA_VISIBLE_DEVICES": "3"})
    assert tail.strip() == "3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_cancel_before_start_has_no_side_effect(tmp_path):
    def interrupt():
        raise Interrupted()

    with pytest.raises(Interrupted):
        run_native([sys.executable, "-c", "open('started', 'w').close()"], cwd=tmp_path,
                   interrupt=interrupt, progress=lambda *args: None)
    assert not (tmp_path / "started").exists()


@pytest.mark.parametrize("ignore_term", [False, True])
def test_successful_parent_does_not_leave_descendants(tmp_path, ignore_term):
    worker = tmp_path / "worker.py"
    worker.write_text("import fcntl,pathlib,signal,time\n"
                      + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
                      + "lock = open('lock', 'w')\n"
                      "fcntl.flock(lock, fcntl.LOCK_EX)\n"
                      "pathlib.Path('ready').touch()\n"
                      "time.sleep(60)\n")
    parent = ("import pathlib,subprocess,sys,time\n"
              "subprocess.Popen([sys.executable, 'worker.py'])\n"
              "while not pathlib.Path('ready').exists(): time.sleep(0.01)\n")
    started = time.monotonic()
    run_native([sys.executable, "-c", parent], cwd=tmp_path,
               interrupt=lambda: None, progress=lambda *args: None)
    with (tmp_path / "lock").open("w") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                assert time.monotonic() - started < 3
                time.sleep(0.01)


def test_cleanup_reports_real_permission_errors(monkeypatch):
    from comfy.tensorfold.process import _stop
    from unittest.mock import Mock

    def denied(*args):
        raise PermissionError("signal denied")

    monkeypatch.setattr(os, "killpg", denied)
    with pytest.raises(PermissionError, match="signal denied"):
        _stop(Mock(pid=12345))


def test_cleanup_reap_has_a_deadline(monkeypatch):
    import subprocess
    from comfy.tensorfold.process import _stop
    from unittest.mock import Mock

    monkeypatch.setattr(os, "killpg", lambda *args: None)
    process = Mock(pid=12345)
    process.wait.side_effect = subprocess.TimeoutExpired("worker", 5)
    with pytest.raises(RuntimeError, match="Restart ComfyUI"):
        _stop(process)
    process.wait.assert_called_once_with(timeout=5)
