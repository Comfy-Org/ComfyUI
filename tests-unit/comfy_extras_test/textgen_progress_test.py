from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.text_encoders.token_stream as token_stream  # noqa: E402
from comfy_api.latest import io  # noqa: E402
from comfy_extras import nodes_textgen  # noqa: E402
from comfy_extras.nodes_textgen import ProgressText, TextGenerate, TextGenerateLTX2Prompt, TextGenerateV2  # noqa: E402


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
        self.decodes = 0

    def tokenize(self, prompt, **kwargs):
        return [[(1, 1.0)]]

    def generate(self, tokens, **kwargs):
        ids = []
        for ch in self.reply:
            ids.append(ord(ch))
            token_stream.report(ids)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        self.decodes += 1
        return "".join(chr(i) for i in ids if i != 1)


@pytest.fixture
def server(monkeypatch):
    s = FakeServer()
    monkeypatch.setattr(nodes_textgen, "PromptServer", SimpleNamespace(instance=s))
    monkeypatch.setattr(ProgressText, "INTERVAL", 0.0)
    return s


@pytest.fixture
def clock(monkeypatch):
    now = [0.5]  # monotonic time can be below INTERVAL, e.g. on a freshly booted machine
    monkeypatch.setattr(nodes_textgen, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(ProgressText, "INTERVAL", 1.0)
    return now


def texts(server):
    return [t for t, _, _ in server.sent]


def test_listen_is_scoped():
    seen = []
    with token_stream.listen(lambda ids: seen.append(list(ids))):
        token_stream.report([1])
    token_stream.report([2])
    assert seen == [[1]]


def test_listen_ends_when_generation_fails():
    seen = []
    with pytest.raises(RuntimeError):
        with token_stream.listen(lambda ids: seen.append(list(ids))):
            raise RuntimeError("interrupted")
    token_stream.report([1])
    assert seen == []


def test_progress_text_goes_to_the_queuing_client(server):
    ProgressText(FakeClip(""), "7", "tab")([ord("h"), ord("i")])
    assert server.sent == [("hi", "7", "tab")]


def test_no_progress_text_without_a_client_or_a_node(server):
    clip, node = FakeClip(""), SimpleNamespace(unique_id="7", dynprompt=None)
    assert ProgressText.for_node(clip, node).client_id == "tab"
    assert ProgressText.for_node(clip, SimpleNamespace(unique_id=None, dynprompt=None)) is None
    assert ProgressText.for_node(clip, None) is None  # execute called on the bare class
    server.client_id = None
    assert ProgressText.for_node(clip, node) is None  # never broadcast another client's text


def test_progress_text_is_throttled_before_decoding(server, clock):
    clip = FakeClip("")
    p = ProgressText(clip, "7", "tab")
    p([ord("a")])
    clock[0] += 0.5
    p([ord("a"), ord("b")])
    assert texts(server) == ["a"] and clip.decodes == 1
    p.send("abc")
    clock[0] += 0.5
    p([ord("a"), ord("b"), ord("c"), ord("d")])  # a second after "a", but only half a second after "abc"
    assert texts(server) == ["a", "abc"]
    clock[0] += 0.5
    p([ord("a"), ord("b"), ord("c"), ord("d"), ord("e")])
    assert texts(server) == ["a", "abc", "abcde"] and clip.decodes == 2


def test_a_slow_decode_does_not_run_back_to_back(server, clock):
    class SlowClip(FakeClip):
        def decode(self, ids, skip_special_tokens=True):
            clock[0] += 2.0  # slower than INTERVAL
            return super().decode(ids, skip_special_tokens)

    p = ProgressText(SlowClip(""), "7", "tab")
    p([ord("a")])
    p([ord("a"), ord("b")])
    assert texts(server) == ["a"]


def test_progress_text_in_a_loop_body_shows_on_the_node_it_copies(server):
    copies = {"4.0.0.0_2": "2"}
    hidden = SimpleNamespace(unique_id="4.0.0.0_2", dynprompt=SimpleNamespace(get_display_node_id=lambda i: copies.get(i, i)))
    assert ProgressText.for_node(FakeClip(""), hidden).node_id == "2"


@pytest.mark.parametrize("node", [TextGenerateV2, TextGenerate, TextGenerateLTX2Prompt])
def test_generate_text_nodes_ask_the_executor_for_their_node_id(node):
    assert {io.Hidden.unique_id, io.Hidden.dynprompt} <= set(node.GET_SCHEMA().hidden)


def test_generate_text_shows_its_text_while_it_is_written(server):
    clone = TextGenerateV2.PREPARE_CLASS_CLONE({"hidden_inputs": {io.Hidden.unique_id: "2"}})
    out = clone.execute(FakeClip("Hello"), "hi", 16, {"sampling_mode": "off"})
    assert out.args[0] == "Hello"
    assert texts(server) == ["H", "He", "Hel", "Hell", "Hello", "Hello"]  # the last, unthrottled
    assert {(n, sid) for _, n, sid in server.sent} == {("2", "tab")}


def test_generate_text_without_a_client_sends_nothing(server):
    server.client_id = None
    clone = TextGenerateV2.PREPARE_CLASS_CLONE({"hidden_inputs": {io.Hidden.unique_id: "2"}})
    assert clone.execute(FakeClip("Hi"), "hi", 16, {"sampling_mode": "off"}).args[0] == "Hi"
    assert server.sent == []
