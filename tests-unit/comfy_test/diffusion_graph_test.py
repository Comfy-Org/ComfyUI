from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import PerformanceFeature, parser
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
import comfy.ldm.qwen_image21.model as qwen


def test_diffusion_graphs_cli():
    assert parser.parse_args(["--fast", "diffusion_graphs"]).fast == [PerformanceFeature.DiffusionGraphs]


@pytest.mark.parametrize("kind", ["qwen21", "krea2"])
def test_graph_path_preserves_tiny_cpu_forward(monkeypatch, kind):
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    torch.manual_seed(694)
    common = {"dtype": torch.float32, "operations": comfy.ops.disable_weight_init}
    if kind == "qwen21":
        cls = QwenImage21Transformer2DModel
        kwargs = dict(in_channels=4, out_channels=4, num_layers=2, attention_head_dim=16,
                      num_attention_heads=2, context_in_dim=8, mlp_ratio=2, axes_dims_rope=(4, 6, 6))
        inputs = (torch.randn(1, 4, 2, 2), torch.tensor([0.7]), torch.randn(1, 3, 8))
    else:
        cls = SingleStreamDiT
        kwargs = dict(features=32, tdim=8, txtdim=16, heads=2, kvheads=1, multiplier=2,
                      layers=2, patch=2, channels=4, txtlayers=2, txtheads=2, txtkvheads=2)
        inputs = (torch.randn(1, 4, 4, 4), torch.tensor([0.7]), torch.randn(1, 3, 32))
    reference = cls(**kwargs, **common)
    assert sum(p.numel() for p in reference.parameters()) < 1_000_000
    state = {name: torch.randn_like(value) * 0.02 for name, value in reference.state_dict().items()}
    reference.load_state_dict(state)
    reference.diffusion_graphs = False
    candidate = cls(**kwargs, **common)
    candidate.load_state_dict(state)
    candidate.diffusion_graphs = True
    torch.testing.assert_close(candidate(*inputs), reference(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize("ready", [None, False, True])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("graphs", [False, True])
def test_qwen_records_only_ready_positive_cache(monkeypatch, ready, negative, graphs):
    events = []

    @contextmanager
    def scope(device):
        events.append("begin")
        yield
        events.append("end")

    monkeypatch.setattr(comfy.model_prefetch, "malloc_graph_scope", scope)
    model = object.__new__(QwenImage21Transformer2DModel)
    torch.nn.Module.__init__(model)
    model.diffusion_graphs = graphs
    model._forward = lambda x, *args, **kwargs: x
    x = torch.empty(1, 4, 2, 2)
    slots = {} if ready is None else {x.device: [SimpleNamespace(key=object() if ready else None)]}
    result = model(x, torch.ones(1), torch.empty(1, 3, 8), prefix_cache=slots,
                   transformer_options={"cond_or_uncond": [int(negative)]})
    assert result is x
    assert events == (["begin", "end"] if not negative and ready is not False else [])


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_qwen_cache_pins_surround_prefetch_and_replay(monkeypatch, graphs, replay, cached):
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    model = QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=2, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=8, mlp_ratio=2, axes_dims_rope=(4, 6, 6),
        dtype=torch.float32, operations=comfy.ops.disable_weight_init,
    )
    model.load_state_dict({name: torch.randn_like(value) * 0.02 for name, value in model.state_dict().items()})
    model.diffusion_graphs = graphs
    model.prefix_cache_enabled = True
    events = []
    active = []

    class Cache:
        def __init__(self, slots):
            self.slots = slots

        def filled(self, key):
            return cached

        @contextmanager
        def layer(self, i, device, write):
            assert write == (not cached)
            assert not active
            active.append(i)
            events.append(("pin", i))
            yield
            events.append(("unpin", active.pop()))

        def take(self, i, device, dtype, batch_size):
            assert active == [i]
            events.append(("take", i))
            return torch.zeros(batch_size, 2, 3, 2, 16, device=device, dtype=dtype)

        def put(self, i, tensor):
            assert active == [i]
            events.append(("put", i))

        def finish(self, key):
            events.append("finish")

    def queue(units, *args):
        assert units == list(model.transformer_blocks)
        return None

    def pop(queue, device, module, dtype=None, core=None, enable_graph=False, malloc_scope=None):
        assert malloc_scope == "block"
        if module is None:
            assert not active
            events.append("drain")
        else:
            i = list(model.transformer_blocks).index(module)
            assert active == [i]
            assert enable_graph == (graphs and cached)
            events.append(("pop", i))
            if not replay or not cached:
                core()

    monkeypatch.setattr(qwen, "PrefixCache", Cache)
    monkeypatch.setattr(comfy.model_prefetch, "make_prefetch_queue", queue)
    monkeypatch.setattr(comfy.model_prefetch, "prefetch_queue_pop", pop)
    x = torch.randn(1, 4, 2, 2)
    model._forward(x, torch.ones(1), torch.randn(1, 3, 8),
                   prefix_cache={x.device: [[SimpleNamespace(), SimpleNamespace()]]})
    expected = []
    for i in range(2):
        expected.extend([("pin", i), ("pop", i)])
        if not cached:
            expected.append(("put", i))
        if not replay or not cached:
            expected.append(("take", i))
        expected.append(("unpin", i))
    expected.append("drain")
    if not cached:
        expected.append("finish")
    assert events == expected


@pytest.mark.parametrize("graphs", [False, True])
def test_krea_records_preparation_in_both_modes(monkeypatch, graphs):
    events = []

    @contextmanager
    def scope(device):
        events.append("begin")
        yield
        events.append("end")

    monkeypatch.setattr(comfy.model_prefetch, "malloc_graph_scope", scope)
    model = object.__new__(SingleStreamDiT)
    torch.nn.Module.__init__(model)
    model.diffusion_graphs = graphs
    x = torch.empty(1, 4, 2, 2)
    model._forward = lambda *args, **kwargs: events.append("forward") or x
    assert model(x, torch.ones(1), torch.empty(1, 3, 8)) is x
    assert events == ["begin", "forward", "end"]


def test_blas_warmup_uses_requested_device(monkeypatch):
    events = []
    device = torch.device("cuda:1")
    one = torch.empty(2, 2)

    @contextmanager
    def use_device(actual):
        assert actual == device
        events.append("enter")
        yield
        events.append("exit")

    def empty(shape, device):
        assert shape == (2, 2) and device == torch.device("cuda:1")
        events.append("allocate")
        return one

    def addmm(bias, a, b):
        assert a is one and b is one
        assert bias.shape == (2,)
        assert bias.untyped_storage().data_ptr() == one.untyped_storage().data_ptr()
        events.append("gemm")

    monkeypatch.setattr(torch.cuda, "device", use_device)
    monkeypatch.setattr(torch.cuda, "current_blas_handle", lambda: events.append("handle"))
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch, "addmm", addmm)
    comfy.model_prefetch._warmup_blas(device)
    assert events == ["enter", "handle", "allocate", "gemm", "exit"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("retained_capture_stream", [False, True])
def test_blas_warmup_precedes_new_allocation_recordings(monkeypatch, enabled, retained_capture_stream):
    prefetch = comfy.model_prefetch
    events = []
    device = torch.device("cuda:1")
    stream = object()
    capture_stream = object()
    graphs = {}

    @contextmanager
    def use_capture_stream(actual_stream):
        assert actual_stream is capture_stream
        events.append("capture_stream")
        yield
        events.append("default_stream")

    def record(actual_stream, assert_breaks):
        assert actual_stream is stream
        assert assert_breaks == prefetch.args.assert_graph_breaks
        events.append("record")
        return SimpleNamespace(push=lambda: events.append("push"), pop=lambda: events.append("pop") or False)

    def warmup(actual_device):
        assert actual_device == device
        assert not graphs
        events.append("warmup")

    monkeypatch.setattr(prefetch, "MALLOC_GRAPHS", graphs)
    monkeypatch.setattr(prefetch, "MALLOC_GRAPH_USED", False)
    monkeypatch.setattr(prefetch, "GRAPH_CAPTURE_STREAMS", {device: capture_stream} if retained_capture_stream else {torch.device("cuda:0"): object()})
    monkeypatch.setattr(prefetch, "malloc_graph_enabled", lambda device: enabled)
    monkeypatch.setattr(prefetch, "_warmup_blas", warmup)
    monkeypatch.setattr(prefetch.comfy_aimdo.malloc_graph, "record", record)
    monkeypatch.setattr(comfy.model_management, "current_stream", lambda device: stream)
    monkeypatch.setattr(prefetch.ck, "set_allocation_context", lambda context: None, raising=False)
    monkeypatch.setattr(torch.cuda, "stream", use_capture_stream)
    for _ in range(2):
        prefetch.malloc_graph_begin(device)
        prefetch.malloc_graph_end()
    graphs.clear()
    prefetch.malloc_graph_begin(device)
    prefetch.malloc_graph_end()
    warmup_events = ["warmup"] + (["capture_stream", "warmup", "default_stream"] if retained_capture_stream else [])
    recording_events = warmup_events + ["record", "pop"]
    assert events == (recording_events + ["push", "pop"] + recording_events if enabled else [])
    assert prefetch.MALLOC_GRAPH_USED == enabled


@pytest.mark.parametrize("recapture", [False, True])
@pytest.mark.parametrize("core_error", [False, True])
def test_blas_workspace_is_restored_after_graph_reset(monkeypatch, recapture, core_error):
    prefetch = comfy.model_prefetch
    device = torch.device("cuda:1")
    events = []
    workspace = {"ready": True, "paused": False}
    module = torch.nn.Module()
    module._v_block = object()
    signature = object()
    capture_stream = SimpleNamespace(wait_stream=lambda stream: None)
    current_stream = SimpleNamespace(wait_stream=lambda stream: None)

    class Graph:
        def reset(self):
            events.append("reset")
            workspace["ready"] = False

        def replay(self):
            assert workspace["ready"]
            events.append("replay")

    if recapture:
        module._comfy_graph = {"graph": Graph(), "signature": object()}

    def pause(sync=False):
        workspace["paused"] = True
        events.append("pause")

    def resume(sync=False):
        workspace["paused"] = False

    def synchronize():
        assert not workspace["paused"]
        events.append("synchronize")

    @contextmanager
    def capture(graph, stream, capture_error_mode):
        assert workspace["ready"] and workspace["paused"]
        assert stream is capture_stream and capture_error_mode == "thread_local"
        assert events[-2:] == ["synchronize", "pause"]
        events.append("capture")
        try:
            yield
        finally:
            events.append("capture_end")

    @contextmanager
    def stream(actual):
        assert actual is capture_stream
        yield

    def warmup(actual):
        assert actual == device
        assert workspace["paused"] and not workspace["ready"]
        events.append("warmup")
        workspace["ready"] = True

    def core():
        assert workspace["ready"]
        events.append("core")
        if core_error:
            raise ValueError("core failed")

    malloc = SimpleNamespace(
        _comfy_active=True, _comfy_cuda_graph_modules=set(),
        iterate=lambda name: False, pause=pause, resume=resume, use_stream=stream,
    )
    monkeypatch.setattr(prefetch, "MALLOC_GRAPHS", {prefetch.threading.get_ident(): malloc})
    monkeypatch.setattr(prefetch, "GRAPH_CAPTURE_STREAMS", {device: capture_stream})
    monkeypatch.setattr(prefetch, "GRAPH_WARMED_MODULES", {module})
    monkeypatch.setattr(prefetch.args, "disable_cuda_graphs", False)
    monkeypatch.setattr(prefetch, "pin_modules", lambda *args: (None, True))
    monkeypatch.setattr(prefetch, "_warmup_blas", warmup)
    monkeypatch.setattr(prefetch.comfy_aimdo.model_vbar, "vbar_fault", lambda block: signature)
    monkeypatch.setattr(prefetch.comfy_aimdo.model_vbar, "vbar_unpin", lambda block: None)
    monkeypatch.setattr(prefetch.comfy_aimdo.model_vbar, "vbar_signature_compare", lambda a, b: False)
    monkeypatch.setattr(comfy.model_management, "synchronize", synchronize)
    monkeypatch.setattr(comfy.model_management, "current_stream", lambda device: current_stream)
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", capture)

    if core_error:
        with pytest.raises(ValueError, match="core failed"):
            prefetch.prefetch_queue_pop([None, module, None], device, module, core=core, enable_graph=True, malloc_scope="block")
        assert events[-2:] == ["core", "capture_end"]
        assert events.count("synchronize") == 1
        assert not hasattr(module, "_comfy_graph")
        return
    prefetch.prefetch_queue_pop([None, module, None], device, module, core=core, enable_graph=True, malloc_scope="block")
    assert events.count("warmup") == int(recapture)
    if recapture:
        assert events.index("reset") < events.index("pause") < events.index("warmup") < events.index("capture")
    assert events[-4:] == ["core", "pause", "capture_end", "replay"]
    assert module._comfy_graph["signature"] is signature
    monkeypatch.setattr(prefetch.comfy_aimdo.model_vbar, "vbar_signature_compare", lambda a, b: True)
    prefetch.prefetch_queue_pop([None, module, None], device, module, core=core, enable_graph=True, malloc_scope="block")
    assert events.count("warmup") == int(recapture)
    assert events.count("capture") == events.count("core") == 1
    assert events[-1] == "replay"
    assert events.count("synchronize") == 1

    other = torch.nn.Module()
    other._v_block = object()
    prefetch.GRAPH_WARMED_MODULES.add(other)
    prefetch.prefetch_queue_pop([None, other, None], device, other, core=core, enable_graph=True, malloc_scope="block")
    assert events.count("capture") == 2
    assert events.count("synchronize") == 2

    malloc._comfy_cuda_graph_modules = set()
    prefetch._drop_graph(module)
    workspace["ready"] = True
    prefetch.prefetch_queue_pop([None, module, None], device, module, core=core, enable_graph=True, malloc_scope="block")
    assert events.count("synchronize") == 3
