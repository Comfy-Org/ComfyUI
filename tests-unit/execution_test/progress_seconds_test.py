import types
from unittest.mock import Mock

import comfy_execution.progress as progress
from comfy_execution.progress import ProgressRegistry, WebUIProgressHandler


class Clock:
    def __init__(self):
        self.now = 100.0

    def advance(self, seconds):
        self.now += seconds

    def perf_counter(self):
        return self.now


def make_registry(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(progress, "time", types.SimpleNamespace(perf_counter=clock.perf_counter))
    dynprompt = Mock()
    dynprompt.get_display_node_id.side_effect = lambda n: n
    dynprompt.get_parent_node_id.return_value = None
    dynprompt.get_real_node_id.side_effect = lambda n: n
    dynprompt.get_node.return_value = {"class_type": "KSampler"}
    return ProgressRegistry(prompt_id="p", dynprompt=dynprompt), clock


def test_seconds_sum_every_stretch_of_an_activity(monkeypatch):
    registry, clock = make_registry(monkeypatch)
    registry.start_progress("1")
    clock.advance(1.0)
    registry.set_activity("1", "loading")
    clock.advance(2.0)
    registry.set_activity("1", None)
    clock.advance(0.5)
    registry.set_activity("1", "loading")
    clock.advance(3.0)
    registry.set_activity("1", None)
    clock.advance(0.25)
    registry.finish_progress("1")

    assert registry.nodes["1"]["seconds"] == {"loading": 5.0, "total": 6.75}
    assert "activity" not in registry.nodes["1"]


def test_switching_activities_closes_the_previous_stretch(monkeypatch):
    registry, clock = make_registry(monkeypatch)
    registry.start_progress("1")
    registry.set_activity("1", "loading")
    clock.advance(1.0)
    registry.set_activity("1", "downloading")
    clock.advance(4.0)
    registry.finish_progress("1")

    assert registry.nodes["1"]["seconds"] == {"loading": 1.0, "downloading": 4.0, "total": 5.0}


def test_cached_node_reports_no_seconds(monkeypatch):
    registry, _ = make_registry(monkeypatch)
    registry.finish_progress("1")

    assert "seconds" not in registry.nodes["1"]


def test_two_nodes_are_timed_separately(monkeypatch):
    registry, clock = make_registry(monkeypatch)
    registry.start_progress("1")
    registry.set_activity("1", "downloading")
    clock.advance(1.0)
    registry.start_progress("2")
    registry.set_activity("2", "loading")
    clock.advance(2.0)
    registry.set_activity("1", None)
    registry.finish_progress("1")
    clock.advance(1.0)
    registry.finish_progress("2")

    assert registry.nodes["1"]["seconds"] == {"downloading": 3.0, "total": 3.0}
    assert registry.nodes["2"]["seconds"] == {"loading": 3.0, "total": 3.0}


def test_seconds_are_sent_only_once_the_node_finishes(monkeypatch):
    registry, clock = make_registry(monkeypatch)
    server = Mock(client_id="c")
    handler = WebUIProgressHandler(server)
    handler.set_registry(registry)
    registry.register_handler(handler)

    registry.start_progress("1")
    assert "seconds" not in server.send_sync.call_args[0][1]["nodes"]["1"]

    clock.advance(2.0)
    registry.finish_progress("1")
    assert server.send_sync.call_args[0][1]["nodes"]["1"]["seconds"] == {"total": 2.0}
