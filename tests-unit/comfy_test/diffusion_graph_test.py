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
def test_blas_warmup_precedes_new_allocation_recordings(monkeypatch, enabled):
    prefetch = comfy.model_prefetch
    events = []
    device = torch.device("cuda:1")
    stream = object()
    graphs = {}

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
    monkeypatch.setattr(prefetch, "malloc_graph_enabled", lambda device: enabled)
    monkeypatch.setattr(prefetch, "_warmup_blas", warmup)
    monkeypatch.setattr(prefetch.comfy_aimdo.malloc_graph, "record", record)
    monkeypatch.setattr(comfy.model_management, "current_stream", lambda device: stream)
    monkeypatch.setattr(prefetch.ck, "set_allocation_context", lambda context: None, raising=False)
    for _ in range(2):
        prefetch.malloc_graph_begin(device)
        prefetch.malloc_graph_end()
    graphs.clear()
    prefetch.malloc_graph_begin(device)
    prefetch.malloc_graph_end()
    assert events == (["warmup", "record", "pop", "push", "pop", "warmup", "record", "pop"] if enabled else [])
    assert prefetch.MALLOC_GRAPH_USED == enabled
