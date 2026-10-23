"""Contract test for app/agent_host.py against a fake v1 host."""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import os
import threading
import time
from types import SimpleNamespace

import pytest

import folder_paths
from app import agent_host
from comfy_api import feature_flags


class FakeHost:
    def __init__(self):
        self.state = {"state": "starting", "url": None, "reason": None, "restarts": 0, "max_restarts": 3}
        self.started = None
        self.on_change = None
        self.stopped = 0
        self.registered = []

    def start(self, *, comfy_url, comfy_path, logger, on_change=None):
        self.started = {"comfy_url": comfy_url, "comfy_path": comfy_path, "logger": logger}
        self.on_change = on_change

    def stop(self, timeout=10.0):
        self.stopped += 1

    def status(self):
        return dict(self.state)

    def register(self, app):
        # The real host mounts its /api/agent + /api/workflows subapps here; the
        # proxy + trust boundary live in the wheel, so core only hands over app.
        self.registered.append(app)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(agent_host, "_host", None)
    monkeypatch.setattr(agent_host, "_failure", None)
    monkeypatch.setattr(agent_host, "_status_logged", False)
    # The loader publishes the agent flag by mutating this shared dict; keep that
    # leakage out of other tests.
    feature_flags.SERVER_FEATURE_FLAGS.pop("agent", None)
    yield
    feature_flags.SERVER_FEATURE_FLAGS.pop("agent", None)


def _entry_points(monkeypatch, eps: dict):
    points = [SimpleNamespace(name=name, load=lambda f=factory: f) for name, factory in eps.items()]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: points if group == "comfyui.agent" else [])


def _raise(*args, **kwargs):
    raise RuntimeError("broken install")


def _exit(*args, **kwargs):
    raise SystemExit("wrong-architecture wheel")


async def _until(predicate):
    while not predicate():
        await asyncio.sleep(0.02)


def _agent_records(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[agent-event]")]


def _comfy_path():
    return os.path.dirname(os.path.abspath(folder_paths.__file__))


@pytest.mark.parametrize(
    "eps, loaded, record",
    [
        ({"v1": FakeHost}, True, None),
        ({}, False, "[agent-event] package_missing"),
        ({"v9": FakeHost}, False, "[agent-event] agent_error reason=outdated"),
        ({"v1": _raise}, False, "[agent-event] agent_error reason=unknown"),
        ({"v1": _exit}, False, "[agent-event] agent_error reason=unknown"),
    ],
)
def test_load_finds_v1_or_reports_why_not(monkeypatch, caplog, eps, loaded, record):
    _entry_points(monkeypatch, eps)
    with caplog.at_level(logging.INFO):
        assert agent_host.load() is loaded
    assert _agent_records(caplog) == ([record] if record else [])


def test_broken_entry_point_metadata_does_not_stop_comfyui(monkeypatch, caplog):
    monkeypatch.setattr(importlib.metadata, "entry_points", _raise)
    with caplog.at_level(logging.INFO):
        assert agent_host.load() is False
    assert _agent_records(caplog) == ["[agent-event] agent_error reason=unknown"]


def test_status_reaches_the_flag_and_the_websocket(monkeypatch):
    _entry_points(monkeypatch, {"v1": FakeHost})
    assert agent_host.feature_flag() is None
    agent_host.load()
    sent = []
    agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: sent.append((event, data))), "http", "0.0.0.0", 8188)
    # Records must leave through ComfyUI's own logger; the host is told where
    # ComfyUI lives and the loopback URL to reach it on.
    assert agent_host._host.started == {"comfy_url": "http://127.0.0.1:8188", "comfy_path": _comfy_path(), "logger": logging.getLogger()}
    assert agent_host.feature_flag() == {"state": "starting"}
    agent_host._host.state = {"state": "ready", "url": None, "reason": None}
    agent_host._host.on_change()
    assert sent == [("agent_status", {"state": "ready"})]
    assert feature_flags.get_server_features()["agent"] == sent[0][1]


@pytest.mark.parametrize("url", ["http://127.0.0.1:5555", "/agent", None])
@pytest.mark.parametrize("state", ["starting", "ready"])
def test_the_flag_never_publishes_a_url(monkeypatch, url, state):
    # The proxy is mandatory: the frontend always reaches the agent at /api/agent.
    monkeypatch.setattr(agent_host, "_host", FakeHost())
    agent_host._host.state = {"state": state, "url": url, "reason": None}
    assert agent_host.feature_flag() == {"state": state}


def test_the_flag_carries_a_reason_only_when_set(monkeypatch):
    monkeypatch.setattr(agent_host, "_host", FakeHost())
    agent_host._host.state = {"state": "starting", "url": None, "reason": "crashed"}
    assert agent_host.feature_flag() == {"state": "starting", "reason": "crashed"}
    agent_host._host.state = {"state": "ready", "url": None, "reason": ""}
    assert agent_host.feature_flag() == {"state": "ready"}


@pytest.mark.parametrize(("eps", "reason"), [({}, "package_missing"), ({"v9": FakeHost}, "outdated"), ({"v1": _raise}, "unknown")])
def test_a_package_that_does_not_load_publishes_failed(monkeypatch, eps, reason):
    _entry_points(monkeypatch, eps)
    agent_host.load()
    assert feature_flags.get_server_features()["agent"] == {"state": "failed", "reason": reason}


def test_status_changes_the_flag_does_not_carry_send_nothing(monkeypatch):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    sent = []
    agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: sent.append(data)), "http", "127.0.0.1", 8188)
    agent_host._host.on_change()
    agent_host._host.state = {**agent_host._host.state, "restarts": 1}
    agent_host._host.on_change()  # only the restart count moved
    assert sent == [{"state": "starting"}]


def test_a_raising_status_is_logged_once(monkeypatch, caplog):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    monkeypatch.setattr(agent_host._host, "status", _raise)
    for _ in range(3):
        assert agent_host.feature_flag()["state"] == "failed"
    assert sum("status could not be read" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize("error", [SystemExit])
def test_any_exception_from_the_package_fails_the_agent_not_comfyui(monkeypatch, error):
    def create():
        raise error()

    _entry_points(monkeypatch, {"v1": create})
    assert agent_host.load() is False


def test_a_status_that_exits_during_load_fails_the_agent_not_comfyui(monkeypatch):
    class ExitingHost(FakeHost):
        status = _exit

    _entry_points(monkeypatch, {"v1": ExitingHost})
    assert agent_host.load() is False
    assert feature_flags.get_server_features()["agent"] == {"state": "failed", "reason": "unknown"}
    agent_host.register(object())
    agent_host.start(SimpleNamespace(loop=None, send_sync=lambda *a: None), "http", "127.0.0.1", 8188)
    assert agent_host._host.registered == [] and agent_host._host.started is None, "a host that exits is disabled"


def test_a_factory_that_returns_no_host_fails_the_agent(monkeypatch):
    _entry_points(monkeypatch, {"v1": object})
    assert agent_host.load() is False
    assert agent_host.feature_flag() == {"state": "failed", "reason": "unknown"}


def test_a_start_that_exits_fails_the_agent_not_comfyui(monkeypatch):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()

    def start(**kwargs):
        raise SystemExit(3)  # what a corrupt or wrong-architecture wheel does

    monkeypatch.setattr(agent_host._host, "start", start)
    sent = []
    agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: sent.append(data)), "http", "127.0.0.1", 8188)
    assert sent == [{"state": "failed", "reason": "spawn_failed"}]


@pytest.mark.parametrize(("address", "url"), [("0.0.0.0", "http://127.0.0.1:8188"), ("", "http://127.0.0.1:8188"), ("::", "http://[::1]:8188"), ("::1", "http://[::1]:8188"), ("192.168.1.5", "http://192.168.1.5:8188")])
def test_comfyui_tells_the_agent_a_url_it_can_reach(monkeypatch, address, url):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: None), "http", address, 8188)
    assert agent_host._host.started["comfy_url"] == url


def test_a_host_that_fails_to_start_reports_failed(monkeypatch, caplog):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    monkeypatch.setattr(agent_host._host, "start", _raise)
    sent = []
    with caplog.at_level(logging.INFO):
        agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: sent.append((event, data))), "http", "127.0.0.1", 8188)
    failed = {"state": "failed", "reason": "spawn_failed"}
    assert agent_host.feature_flag() == failed
    assert sent == [("agent_status", failed)]
    assert _agent_records(caplog) == ["[agent-event] agent_error reason=spawn_failed"]


def test_register_hands_the_host_comfyuis_app(monkeypatch):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    app = object()
    agent_host.register(app)
    assert agent_host._host.registered == [app], "register() hands the host ComfyUI's app to mount its own routes on"


def test_register_without_a_loaded_package_is_a_noop():
    # package_missing / outdated / broken: load() left _host None, so there is
    # nothing to mount and add_routes must carry on.
    agent_host.register(object())


def test_a_register_failure_fails_the_agent_not_comfyui(monkeypatch, caplog):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    monkeypatch.setattr(agent_host._host, "register", _raise)
    with caplog.at_level(logging.INFO):
        agent_host.register(object())  # a broken wheel must not end add_routes
    assert agent_host.feature_flag() == {"state": "failed", "reason": "unknown"}
    assert feature_flags.get_server_features()["agent"] == {"state": "failed", "reason": "unknown"}
    assert "[agent-event] agent_error reason=unknown" in _agent_records(caplog)


def test_start_does_not_run_the_host_after_a_register_failure(monkeypatch):
    # register() raising sets _failure while _host stays set. start() must bail
    # out instead of spawning an agent whose routes were never mounted.
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    monkeypatch.setattr(agent_host._host, "register", _raise)
    agent_host.register(object())
    sent = []
    agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: sent.append(data)), "http", "127.0.0.1", 8188)
    assert agent_host._host.started is None, "start() must not call the host after register() failed"
    assert sent == []
    assert agent_host.feature_flag() == {"state": "failed", "reason": "unknown"}


@pytest.mark.parametrize("raises", [None, RuntimeError, SystemExit])
def test_stop_stops_the_host(monkeypatch, raises):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    if raises:
        def stop(timeout=10.0):
            agent_host._host.stopped += 1
            raise raises()

        monkeypatch.setattr(agent_host._host, "stop", stop)
    agent_host.stop()
    assert agent_host._host.stopped == 1


def test_start_runs_the_host_off_the_event_loop(monkeypatch):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    real_start = agent_host._host.start
    released = threading.Event()

    def slow_start(**kwargs):
        released.wait(5)
        real_start(**kwargs)

    monkeypatch.setattr(agent_host._host, "start", slow_start)
    loop = asyncio.new_event_loop()
    try:
        began = time.monotonic()
        agent_host.start(SimpleNamespace(loop=loop, send_sync=lambda event, data: None), "http", "127.0.0.1", 8188)
        assert time.monotonic() - began < 0.5, "start() doesn't run the host on the event loop"
        released.set()
        loop.run_until_complete(asyncio.wait_for(_until(lambda: agent_host._host.started), 5))
    finally:
        loop.close()
