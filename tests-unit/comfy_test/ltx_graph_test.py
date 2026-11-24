from contextlib import contextmanager
import weakref

import pytest
import torch

import comfy.model_prefetch as prefetch
from comfy.ldm.lightricks.av_model import LTXAVModel


@pytest.mark.parametrize("stg", [False, True])
@pytest.mark.parametrize("graphs", [False, True])
def test_ltxav_forward_scopes_preparation_and_not_outputs(monkeypatch, stg, graphs):
    events = []
    prepared = []
    freed = []

    @contextmanager
    def scope(device):
        events.append("begin")
        yield
        assert all(reference() is None for reference in prepared)
        assert freed and all(event == "resume" for event in freed)
        events.append("end")

    @contextmanager
    def pause():
        events.append("pause")
        yield
        events.append("resume")

    monkeypatch.setattr(prefetch, "malloc_graph_scope", scope)
    monkeypatch.setattr(prefetch, "pause_malloc_graph", pause)
    model = object.__new__(LTXAVModel)
    torch.nn.Module.__init__(model)
    model.diffusion_graphs = graphs
    scoped = not stg

    def prepare(x, *args, **kwargs):
        assert events == (["begin"] if scoped else [])
        values = [value.clone() for value in x]
        prepared.extend(weakref.ref(value, lambda reference: freed.append(events[-1])) for value in values)
        return values, None, {}

    model._process_input = prepare
    model._prepare_timestep = lambda *args, **kwargs: (None, None, None)
    model._prepare_context = lambda context, batch, x, mask: (context, mask)
    model._prepare_attention_mask = lambda mask, dtype: mask
    model._prepare_positional_embeddings = lambda *args: None
    model._build_guide_self_attention_mask = lambda *args: None
    model._process_transformer_blocks = lambda x, *args, **kwargs: x

    def output(x, *args, **kwargs):
        assert events == (["begin", "pause"] if scoped else ["pause"])
        return [value.clone() for value in x]

    model._process_output = output
    x = torch.empty(1, 2, 3)
    result = model([x, x], torch.ones(1), x, attention_mask=None,
                   transformer_options={"stg_self_attn_blocks": (0,)} if stg else {})
    assert torch.equal(result[0], x)
    assert events == (["begin", "pause", "resume", "end"] if scoped else ["pause", "resume"])
    assert all(reference() is None for reference in prepared)


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_ltxav_prepared_context_uses_one_buffer(graphs, batch_size):
    model = object.__new__(LTXAVModel)
    torch.nn.Module.__init__(model)
    model.diffusion_graphs = graphs
    model.caption_proj_before_connector = True
    context = torch.arange(batch_size * 3 * 12, dtype=torch.float32).reshape(batch_size, 3, 12)
    mask = torch.ones(batch_size, 3)
    x = [torch.empty(batch_size, 2, 8), torch.empty(batch_size, 4, 4)]

    (video, audio), output_mask = model._prepare_context(context, batch_size, x, mask)

    assert output_mask is mask
    torch.testing.assert_close(video, context[..., :8], rtol=0, atol=0)
    torch.testing.assert_close(audio, context[..., 8:], rtol=0, atol=0)
    assert video.untyped_storage().data_ptr() == audio.untyped_storage().data_ptr()
    assert (video.untyped_storage().data_ptr() != context.untyped_storage().data_ptr()) == graphs
    assert video.stride(-2) == audio.stride(-2) == 12


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("stg", [False, True])
def test_ltxav_prefetch_owns_both_block_paths(monkeypatch, graphs, replacement, stg):
    events = []
    queue = object()

    class Block(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, x, transformer_options, **kwargs):
            assert transformer_options.get("stg_skip_self_attn", False) == (stg and self.index == 1)
            events.append(("block", self.index))
            return x[0] + self.index + 1, x[1] + 2 * (self.index + 1)

    def replace(args, extra):
        events.append(("replace", 1))
        out = extra["original_block"](args)
        out["img"] = (out["img"][0] + 10, out["img"][1])
        return out

    def pop(actual_queue, device, module, dtype=None, core=None, enable_graph=False, malloc_scope=None):
        assert actual_queue is queue
        assert malloc_scope == "block"
        if module is None:
            assert core is None
            events.append("drain")
        else:
            assert dtype == torch.float32
            assert enable_graph == (graphs and not replacement)
            events.append(("prefetch", module.index))
            core()

    monkeypatch.setattr(prefetch, "make_prefetch_queue", lambda *args: queue)
    monkeypatch.setattr(prefetch, "prefetch_queue_pop", pop)
    monkeypatch.setattr(prefetch, "malloc_graph_begin", lambda device: events.append("begin"))
    monkeypatch.setattr(prefetch, "malloc_graph_end", lambda: events.append("end"))
    model = object.__new__(LTXAVModel)
    torch.nn.Module.__init__(model)
    model.diffusion_graphs = graphs
    model.transformer_blocks = torch.nn.ModuleList([Block(i) for i in range(3)])
    options = {"stg_self_attn_blocks": (1,)} if stg else {}
    if replacement:
        options["patches_replace"] = {"dit": {("double_block", 1): replace}}
    result = model._process_transformer_blocks(
        [torch.ones(2), torch.full((2,), 2.0)], (None, None), None,
        (None, None, (None, None, None, None), None, None),
        ((None, None), (None, None)), transformer_options=options,
    )
    torch.testing.assert_close(result[0], torch.full((2,), 17.0 if replacement else 7.0))
    torch.testing.assert_close(result[1], torch.full((2,), 14.0))
    expected = []
    for i in range(3):
        expected.append(("prefetch", i))
        if replacement and i == 1:
            expected.append(("replace", i))
        expected.append(("block", i))
    expected.append("drain")
    assert events == expected
