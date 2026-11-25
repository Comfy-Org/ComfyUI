from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.text_encoders.token_stream as token_stream  # noqa: E402
from comfy_api.latest import io  # noqa: E402
from comfy_extras import nodes_textgen  # noqa: E402
from comfy_extras.nodes_textgen import ProgressText, TextGenerate, TextGenerateV2  # noqa: E402


class FakeServer:
    def __init__(self, client_id="tab"):
        self.client_id = client_id
        self.sent = []

    def send_progress_text(self, text, node_id, sid=None):
        self.sent.append((text, node_id, sid))


class FakeClip:
    """Generates the characters of `reply`, reporting the ids after each step as the generate loops do."""

    def __init__(self, reply):
        self.reply = reply

    def tokenize(self, prompt, **kwargs):
        return [[(1, 1.0)]]

    def generate(self, tokens, **kwargs):
        ids = []
        for ch in self.reply:
            ids.append(ord(ch))
            token_stream.report(ids)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i != 1)


@pytest.fixture
def server(monkeypatch):
    s = FakeServer()
    monkeypatch.setattr(nodes_textgen, "PromptServer", SimpleNamespace(instance=s))
    monkeypatch.setattr(ProgressText, "INTERVAL", 0.0)
    return s


def test_report_without_a_listener_does_nothing():
    token_stream.report([1, 2, 3])


def test_listen_is_scoped_and_nests():
    outer, inner = [], []
    with token_stream.listen(lambda ids: outer.append(list(ids))):
        token_stream.report([1])
        with token_stream.listen(lambda ids: inner.append(list(ids))):
            token_stream.report([1, 2])
        token_stream.report([1, 2, 3])
    token_stream.report([9])
    assert outer == [[1], [1, 2, 3]]
    assert inner == [[1, 2]]


def test_progress_text_goes_to_the_queuing_client_only(server):
    ProgressText(FakeClip(""), "7")([ord("h"), ord("i")])
    assert server.sent == [("hi", "7", "tab")]
    server.client_id = None
    ProgressText(FakeClip(""), "7")([ord("h")])
    assert len(server.sent) == 1  # no client: nothing is broadcast


def test_progress_text_is_throttled(server, monkeypatch):
    clock = [0.5]  # monotonic time can be below INTERVAL, e.g. on a freshly booted machine
    monkeypatch.setattr(nodes_textgen, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(ProgressText, "INTERVAL", 1.0)
    p = ProgressText(FakeClip(""), "7")
    p([ord("a")])
    clock[0] += 0.5
    p([ord("a"), ord("b")])
    assert [t for t, _, _ in server.sent] == ["a"]
    p([ord("a"), ord("b"), ord("c")], final=True)
    assert [t for t, _, _ in server.sent] == ["a", "abc"]
    clock[0] += 1.0
    p([ord("a"), ord("b"), ord("c"), ord("d")])
    assert [t for t, _, _ in server.sent] == ["a", "abc", "abcd"]


@pytest.mark.parametrize("node", [TextGenerateV2, TextGenerate])
def test_generate_text_shows_its_text_while_it_is_written(server, node):
    clone = node.PREPARE_CLASS_CLONE({"hidden_inputs": {io.Hidden.unique_id: "2"}})
    out = clone.execute(FakeClip("Hello"), "hi", 16, {"sampling_mode": "off"})
    assert out.args[0] == "Hello"
    assert [t for t, _, _ in server.sent] == ["H", "He", "Hel", "Hell", "Hello", "Hello"]  # the last, unthrottled
    assert {(n, sid) for _, n, sid in server.sent} == {("2", "tab")}


def test_generate_text_without_a_node_id_sends_nothing(server):
    clone = TextGenerateV2.PREPARE_CLASS_CLONE({"hidden_inputs": {}})
    assert clone.execute(FakeClip("Hi"), "hi", 16, {"sampling_mode": "off"}).args[0] == "Hi"
    assert server.sent == []
