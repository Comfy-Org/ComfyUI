import json
import types

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ops  # noqa: E402
from comfy.text_encoders.llama import MLP, RMSNorm  # noqa: E402


def _mlp(merged):
    config = types.SimpleNamespace(hidden_size=16, intermediate_size=24, mlp_activation="silu", merged_mlp=merged)
    mlp = MLP(config, dtype=torch.float32, ops=comfy.ops.manual_cast)
    for p in mlp.parameters():
        p.data = torch.randn_like(p)
    return mlp


def _norm(add):
    norm = RMSNorm(16, eps=1e-6, add=add, dtype=torch.float32)
    norm.weight.data = torch.randn(16) * 0.1
    return norm


def test_mlp_folded_norm_matches_eager_norm():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 16)
    for merged in (False, True):
        mlp = _mlp(merged)
        for add in (False, True):
            norm = _norm(add)
            assert torch.equal(mlp(x, norm=norm), mlp(norm(x)))


def test_rms_norm_scale_adds_one_without_caching():
    norm = _norm(True)
    assert torch.equal(norm.scale(), norm.weight + 1.0)
    norm.weight.data.add_(1.0)
    assert torch.equal(norm.scale(), norm.weight + 1.0)
    plain = _norm(False)
    assert plain.scale() is plain.weight


@pytest.mark.parametrize("bits", [4, 6, 8, "fp8", "transposed"])
@pytest.mark.parametrize("activation", ["rms_norm", "swiglu"])
def test_input_activation_preserves_linear_and_patched_forward(bits, activation):
    from comfy.quant_ops import QuantizedTensor

    torch.manual_seed(7)
    w = torch.randn(32, 256, dtype=torch.bfloat16)
    if bits in (4, 6):
        weight = QuantizedTensor.from_float(w, "AsymW4A8Int8Layout", bits=bits)
    elif bits == "fp8":
        weight = QuantizedTensor.from_float(w, "TensorCoreFP8Layout")
    elif bits == "transposed":
        weight = QuantizedTensor.from_float(w.t().contiguous(), "TensorWiseINT8Layout").t()
    else:
        weight = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
    x = torch.randn(3, 512 if activation == "swiglu" else 256, dtype=torch.bfloat16)
    norm_weight = torch.randn(256, dtype=torch.bfloat16)
    bias = torch.randn(32, dtype=torch.bfloat16)
    residual = torch.randn(3, 32, dtype=torch.bfloat16)
    scale = torch.linspace(-1, 2, 32, dtype=torch.bfloat16)
    if activation == "rms_norm":
        activated = torch.nn.functional.rms_norm(x, (256,), norm_weight, 1e-6)
    else:
        gate, up = x.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
    linear = torch.nn.functional.linear(activated, weight, bias)
    expected = torch.addcmul(residual, linear, scale)
    got = comfy.ops.linear_input_act_(x, weight, bias, activation, norm_weight, 1e-6, residual, scale)
    assert torch.equal(got, expected)

    layer = comfy.ops.manual_cast.Linear(256, 32, dtype=torch.bfloat16)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    got = comfy.ops.linear_input_act(layer, x, activation, norm_weight, 1e-6, residual, scale)
    assert torch.equal(got, expected)

    if bits in ("fp8", "transposed"):
        layer = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(256, 32)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        layer.layout_type = weight._layout_cls
        layer.quant_format = "float8_e4m3fn" if bits == "fp8" else "int8_tensorwise"
        layer.pre_quant_scale = torch.linspace(0.25, 1.75, 256, dtype=torch.bfloat16)
        layer.input_scale = torch.tensor(0.01)
        expected = torch.addcmul(residual, layer(activated), scale)
        got = comfy.ops.linear_input_act(layer, x, activation, norm_weight, 1e-6, residual, scale)
        assert torch.equal(got, expected)

    class Patched(torch.nn.Module):
        comfy_force_forward = True

        def __init__(self):
            super().__init__()
            self.weight = weight

        def forward(self, value):
            return torch.nn.functional.linear(value, self.weight, bias) + 7

    got = comfy.ops.linear_input_act(Patched(), x, activation, norm_weight, 1e-6, residual, scale)
    assert torch.equal(got, torch.addcmul(residual, linear + 7, scale))


def _fake_kitchen(monkeypatch):
    import comfy.text_encoders.llama as llama
    calls = []

    def flash_gqa(q, k, v, kv_lengths, **kwargs):
        calls.append((q.shape, kv_lengths.clone(), kwargs))
        return torch.zeros(q.shape[0], q.shape[2], q.shape[1] * q.shape[3], dtype=q.dtype)
    monkeypatch.setattr(llama.comfy_kitchen, "flash_attention_decode_gqa", flash_gqa)
    return calls


def _gated_attention():
    from comfy.text_encoders.qwen35 import GatedAttention, Qwen35Config
    config = Qwen35Config(hidden_size=32, num_attention_heads=2, num_key_value_heads=1, head_dim=256)
    attn = GatedAttention(config, dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
    for p in attn.parameters():
        p.data = torch.randn_like(p) * 0.1
    return attn


def test_int8_cache_requires_cli_flag_and_native_support(monkeypatch):
    import comfy.text_encoders.qwen35 as qwen35
    from comfy.text_encoders.llama import FixedKVCache
    attn = _gated_attention()
    device = torch.device("cpu")
    monkeypatch.setattr(qwen35, "int8_decode_is_available", lambda device: True)
    monkeypatch.setattr(cli_args, "use_ck_attention", True)
    monkeypatch.setenv("COMFY_INT8_KV", "0")
    assert attn.kv_cache_type(device, torch.bfloat16) is qwen35.Int8FixedKVCache
    for dtype, head_dim in ((torch.float16, 256), (torch.float32, 256), (torch.bfloat16, 128)):
        attn.head_dim = head_dim
        assert attn.kv_cache_type(device, dtype) is FixedKVCache
    attn.head_dim = 256
    monkeypatch.setattr(qwen35, "int8_decode_is_available", lambda device: False)
    assert attn.kv_cache_type(device, torch.bfloat16) is FixedKVCache
    monkeypatch.setattr(qwen35, "int8_decode_is_available", lambda device: True)
    monkeypatch.setattr(cli_args, "use_ck_attention", False)
    monkeypatch.setenv("COMFY_INT8_KV", "1")
    assert attn.kv_cache_type(device, torch.bfloat16) is FixedKVCache
    for dtype, head_dim in ((torch.float16, 256), (torch.float32, 256), (torch.bfloat16, 128)):
        attn.head_dim = head_dim
        assert attn.kv_cache_type(device, dtype) is FixedKVCache


def test_cache_masks_with_device_length(monkeypatch):
    from comfy.text_encoders.llama import FixedKVCache
    calls = _fake_kitchen(monkeypatch)
    shared = FixedKVCache.shared(1, "cpu")
    kv = FixedKVCache.zeros(1, 2, 32, 256, "cpu", torch.bfloat16, shared)
    kv.index = 10
    kv.prepare(3)
    assert kv.position.tolist()[:3] == [10, 11, 12] and kv.seqlen.tolist() == [13]
    xq = torch.randn(1, 4, 3, 256, dtype=torch.bfloat16)
    xk = torch.randn(1, 2, 3, 256, dtype=torch.bfloat16)
    kv.write(xk, xk)
    out = kv.attend(xq, xk, xk)
    assert out.shape == (1, 3, 1024) and calls[-1][1].tolist() == [13]
    assert torch.equal(kv.key[:, :, 10:13], xk)


def _masked_attention(q, k, v, mask):
    groups = q.shape[1] // k.shape[1]
    scores = q.float() @ k.float().repeat_interleave(groups, 1).transpose(-1, -2) * q.shape[-1] ** -0.5 + mask.float()
    out = scores.softmax(-1) @ v.float().repeat_interleave(groups, 1)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def test_cache_mask_matches_attend(monkeypatch):
    # the additive mask reproduces attend()'s chain staircase and tree ancestry over the whole cache
    from comfy.text_encoders.llama import FixedKVCache
    tree = _tree2s2()
    torch.manual_seed(0)
    shared = FixedKVCache.shared(2, "cpu")
    kv = FixedKVCache.zeros(2, 1, 24, 64, "cpu", torch.float32, shared)
    kv.key.normal_()
    kv.value.normal_()
    kv.index = 9
    for seq, shaped in ((1, None), (3, None), (8, tree)):
        kv.tracker["tree"] = shaped
        kv.prepare(seq)
        xq = torch.randn(2, 2, seq, 64)
        xk = torch.randn(2, 1, seq, 64)
        xv = torch.randn(2, 1, seq, 64)
        kv.write(xk, xv)
        expected = kv.attend(xq, xk, xv)
        mask = kv.mask(seq)
        assert mask.shape == (2, 1, seq, 24) and mask.dtype == torch.float32
        visible = (mask == 0)[0, 0]
        if shaped is None:
            assert visible.sum(-1).tolist() == [10 + j for j in range(seq)]
        else:
            assert visible[:, 9:17].int().tolist() == [[(m >> t) & 1 for t in range(8)] for m in tree.mask.tolist()]
            assert visible[:, :9].all() and not visible[:, 17:].any()
        torch.testing.assert_close(_masked_attention(xq, kv.key, kv.value, mask), expected, rtol=1e-4, atol=1e-5)


def test_deferred_ctl_rows_wait_for_their_previous_copy(monkeypatch):
    from comfy.text_encoders.qwen35 import LinearKV
    log = []

    class Event:
        def __init__(self, i):
            self.i = i

        def synchronize(self):
            log.append(("sync", self.i))

        def record(self, stream=None):
            log.append(("record", self.i))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    tracker = LinearKV.deferred_tracker("cpu")
    tracker["copied"] = [Event(0), Event(1)]
    ctl = tracker["ctl"]
    assert ctl.shape == (26,)  # {pending, parity, slots[8], parents[8], program[8]}
    layers = [LinearKV(torch.zeros(1), torch.zeros(1), 0, None, None, ctl=ctl, ctl_tracker=tracker) for _ in range(2)]

    def step(n, discard=0):
        for kv in layers:
            kv.prepare(n)
        for kv in layers:
            kv.advance(n)
            if discard:
                kv.rollback(discard)
        return ctl[:2].tolist()

    assert step(10) == [0, 1]  # prefill commits directly
    assert step(3, discard=2) == [0, 0]
    assert [kv.uncommitted for kv in layers] == [1, 1] and layers[0].index == 11
    log.clear()
    assert step(3) == [1, 1]
    assert log == [("sync", 1), ("record", 1)]  # one copy per step, after its row's previous copy
    assert step(1) == [3, 0]


def _tree2s2():
    from comfy.text_encoders.qwen35 import MTP_MODES, VerifyTree
    capacity, spec = MTP_MODES["tree2s2"]
    return VerifyTree(spec, capacity, "cpu")


def test_verify_tree_shapes():
    from comfy.text_encoders.qwen35 import MTP_MODES, VerifyTree
    tree = _tree2s2()
    assert (tree.rows, tree.chain, tree.fanout) == (8, 2, 3)
    assert tree.parent == [-1, 0, 1, 1, 1, 0, 0, -1] and tree.depth == [0, 1, 2, 2, 2, 1, 1, 0]
    assert tree.mask.tolist() == [1, 3, 7, 11, 19, 33, 65, 128]
    assert tree.ctl.tolist() == [-1, 0, 1, 1, 1, 0, 0, -1, 32, 5, 6, 33, 3, 4, 2, 7]
    tree = VerifyTree(MTP_MODES["tree4s1m8"][1], 8, "cpu")
    assert (tree.rows, tree.chain, tree.fanout) == (8, 4, 2)
    assert tree.parent == [-1, 0, 1, 2, 3, 2, 1, 0] and tree.depth == [0, 1, 2, 3, 4, 3, 2, 1]
    assert tree.mask.tolist() == [1, 3, 7, 15, 31, 39, 67, 129]
    assert tree.ctl.tolist() == [-1, 0, 1, 2, 3, 2, 1, 0, 32, 7, 33, 6, 34, 5, 35, 4]
    mask_ptr, pos_ptr = tree.mask.data_ptr(), tree.positions.data_ptr()
    tree.update([[[[[[[[]]]]]]]])
    assert tree.mask.tolist() == [1, 3, 7, 15, 31, 63, 127, 255]
    assert tree.positions.tolist() == [list(range(8))]
    tree.update(MTP_MODES["tree4s1m8"][1])
    assert tree.mask.data_ptr() == mask_ptr and tree.positions.data_ptr() == pos_ptr
    assert tree.mask.tolist() == [1, 3, 7, 15, 31, 39, 67, 129]
    assert tree.positions.tolist() == [[0, 1, 2, 3, 4, 3, 2, 1]]
    for name, depth in (("tree2s1", 2), ("tree3s1", 3)):
        tree = VerifyTree(MTP_MODES[name][1], MTP_MODES[name][0], "cpu")
        assert (tree.rows, tree.chain, tree.fanout) == (8, depth, 2)
    chain = VerifyTree(MTP_MODES["3"][1], 4, "cpu")
    assert (chain.rows, chain.chain, chain.fanout, chain.children[:3]) == (4, 3, 1, [[1], [2], [3]])
    assert chain.mask.tolist() == [1, 3, 7, 15]
    for depth, capacity in ((2, 4), (4, 8), (5, 8)):
        tree = VerifyTree(MTP_MODES[str(depth)][1], capacity, "cpu")
        assert tree.chain == depth and tree.rows == capacity
        assert tree.parent[depth + 1:] == [-1] * (capacity - depth - 1)
        assert tree.ctl[8 + depth + 1:].tolist() == list(range(depth + 1, 8))
        assert all(not children for children in tree.children[depth:])


def test_tree_verify_cache_prefix_and_commit_path():
    from comfy.text_encoders.llama import FixedKVCache
    tree = _tree2s2()
    shared = FixedKVCache.shared(1, "cpu")
    kv = FixedKVCache.zeros(1, 1, 32, 256, "cpu", torch.bfloat16, shared)
    kv.index = 10
    kv.tracker["tree"] = tree
    kv.prepare(8)
    assert kv.seqlen.tolist() == [10]
    kv.tracker["tree"] = None
    kv.prepare(8)
    assert kv.seqlen.tolist() == [18]
    for r in range(7):
        kv.key[:, :, 10 + r] = r
        kv.value[:, :, 10 + r] = -r
    kv.commit_path(10, [0, 1, 4])
    assert kv.key[0, 0, 10:13, 0].tolist() == [0, 1, 4] and kv.value[0, 0, 12, 0].item() == -4


def test_deferred_ctl_carries_accepted_path_and_tree_program():
    from comfy.text_encoders.qwen35 import LinearKV
    tree = _tree2s2()
    tracker = LinearKV.deferred_tracker("cpu")
    tracker["tree"] = tree
    ctl = tracker["ctl"]
    kv = LinearKV(torch.zeros(1), torch.zeros(1), 5, None, None, ctl=ctl, ctl_tracker=tracker)
    kv.prepare(8)
    assert ctl[2:10].tolist() == list(range(8)) and ctl[10:].tolist() == tree.ctl.tolist()
    kv.advance(8)
    kv.rollback(5)
    kv.commit_path(5, [0, 1, 4])
    kv.prepare(8)
    assert ctl[0].item() == 3 and ctl[2:10].tolist() == [0, 1, 4, 3, 4, 5, 6, 7]
    tracker["tree"] = None
    kv.advance(8)
    kv.prepare(1)
    assert ctl[2:].tolist() == tracker["single"].tolist()


def _tiny_model(monkeypatch, layers=4, head_dim=64):
    import comfy.text_encoders.qwen35 as qwen35
    monkeypatch.setitem(qwen35.QWEN35_MODELS, "tiny", dict(vision=dict(hidden_size=32, num_heads=2, intermediate_size=64, depth=1)))

    class Tiny(qwen35.Qwen35):
        model_type = "tiny"
    config = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=layers, layer_types=qwen35._qwen35_layer_types(layers),
                  num_attention_heads=2, num_key_value_heads=1, head_dim=head_dim, linear_num_key_heads=2, linear_num_value_heads=2,
                  linear_key_head_dim=16, linear_value_head_dim=16, mtp=True, lm_head=True, stop_tokens=[])
    torch.manual_seed(0)
    model = Tiny(config, torch.float32, "cpu", comfy.ops.manual_cast)
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        p.data = torch.randn_like(p) * (0.05 if "norm" in name else 0.3)
    model.model.lm_head.weight.data[:8] *= 4  # a few likely tokens, so some drafts survive
    return model


@pytest.mark.parametrize("image_prompt", [False, True])
def test_greedy_mtp_matches_plain_greedy(monkeypatch, image_prompt):
    # chain and tree verify through the kitchen torch paths: GQA prefix + tree merge, deferred DeltaNet
    from comfy.text_encoders.llama import FixedKVCache
    from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids
    model = _tiny_model(monkeypatch)
    discards = []
    rollback = FixedKVCache.rollback
    monkeypatch.setattr(FixedKVCache, "rollback", lambda self, discard=1: discards.append(discard) or rollback(self, discard))
    embeds = torch.randn(1, 24 if image_prompt else 5, 64)
    image_info = [{"type": "image", "index": 3, "size": 16, "extra": torch.tensor([[1, 8, 8]])}] if image_prompt else []
    positions = qwen2vl_mrope_position_ids(image_info, embeds.shape[1], "cpu")
    if image_prompt:
        assert positions.shape == (3, 24) and positions[:, -1].tolist() == [11, 11, 11]
    plain = model.generate(embeds.clone(), do_sample=False, max_length=40, mtp=False, position_ids=positions)
    for mtp in (True, "2", "3", "4", "5", "tree2s1", "tree2s2", "tree3s1", "tree4s1m8"):
        discards.clear()
        assert model.generate(embeds.clone(), do_sample=False, max_length=40, mtp=mtp, position_ids=positions) == plain
        assert discards and min(discards) < max(discards)  # partial acceptance happened


@pytest.mark.parametrize("quantile", [0.42, 0.8, 0.92])
@pytest.mark.parametrize("top_k", [0, 7])
def test_sampled_tree_matches_sequential_with_same_draws(monkeypatch, quantile, top_k):
    # Couple draws by quantile rather than RNG consumption: unused tree rows also sample.
    def draw(weights, num_samples, generator=None):
        assert num_samples == 1
        cdf = weights.cumsum(-1) / weights.sum(-1, keepdim=True)
        return (cdf < quantile).sum(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", draw)
    model = _tiny_model(monkeypatch)
    embeds = torch.randn(1, 5, 64)
    settings = dict(do_sample=True, max_length=65, temperature=0.7, top_k=top_k, top_p=0.95,
                    min_p=0.05, repetition_penalty=1.2, presence_penalty=0.8)
    expected = model.generate(embeds.clone(), mtp=False, **settings)
    for preset in (True, "2", "3", "4", "5", "tree2s1", "tree2s2", "tree3s1", "tree4s1m8"):
        assert model.generate(embeds.clone(), mtp=preset, **settings) == expected, preset
        stop = expected[4]
        stopped = expected[:expected.index(stop) + 1]
        assert model.generate(embeds.clone(), mtp=preset, stop_tokens=[stop], **settings) == stopped


@pytest.mark.parametrize("sampled", [False, True])
def test_adaptive_switch_preserves_sequential_output(monkeypatch, sampled):
    import comfy.text_encoders.qwen35 as qwen35

    def draw(weights, num_samples, generator=None):
        cdf = weights.cumsum(-1) / weights.sum(-1, keepdim=True)
        return (cdf < 0.8).sum(-1, keepdim=True)

    switches = []

    def best_widths(widths, hits, draft_ms, verify_ms):
        # every 32nd step: cycle through shapes that differ in depth and fanout
        assert len(hits) == 7 and all(h >= 0 for row in hits for h in row)
        switches.append(((1,) * 7, (2, 1, 4), (7,), (4, 3))[len(switches) % 4])
        return switches[-1]

    monkeypatch.setattr(torch, "multinomial", draw)
    monkeypatch.setattr(qwen35, "best_widths", best_widths)
    model = _tiny_model(monkeypatch)
    embeds = torch.randn(1, 5, 64)
    settings = dict(do_sample=sampled, max_length=400, temperature=0.7, top_k=7,
                    top_p=0.95, min_p=0.05, repetition_penalty=1.2, presence_penalty=0.8)
    expected = model.generate(embeds.clone(), mtp=False, **settings)
    assert model.generate(embeds.clone(), mtp=True, **settings) == expected
    assert len(set(switches)) == 4, switches


class _FakePacked:
    # Int8DecodeCache boundary with 4-row pages: records the BF16 refreshes it is asked for
    def __init__(self, batch, heads, capacity, device):
        self.key = torch.zeros(batch, (capacity + 3) // 4, heads, 4, 256, dtype=torch.int8)
        self.updates = []

    def update(self, key, value, length, initialize=False):
        self.updates.append((length.tolist(), initialize))

    def attend(self, query, length):
        batch, heads, seq, dim = query.shape
        return torch.zeros(batch, seq, heads * dim, dtype=query.dtype), torch.zeros(batch, heads, seq)


def _int8_cache(monkeypatch, capacity=32):
    import comfy.text_encoders.llama as llama
    from comfy.text_encoders.llama import FixedKVCache
    _fake_kitchen(monkeypatch)
    merges = []

    def tree_merge(out, lse, q, k, v, mask, result):
        merges.append((k, v, mask.clone()))
        return out
    monkeypatch.setattr(llama, "Int8DecodeCache", _FakePacked, raising=False)
    monkeypatch.setattr(llama.comfy_kitchen, "flash_attention_decode_tree_merge", tree_merge)
    shared = FixedKVCache.shared(1, "cpu")
    return llama.Int8FixedKVCache.zeros(1, 1, capacity, 256, "cpu", torch.bfloat16, shared), merges


def test_int8_cache_merges_current_chain_rows(monkeypatch):
    kv, merges = _int8_cache(monkeypatch)
    prefill = torch.randn(1, 1, 10, 256, dtype=torch.bfloat16)
    kv.prepare(10)
    kv.append(prefill, prefill)
    kv.advance(10)
    kv.prepare(3)
    xq = torch.randn(1, 2, 3, 256, dtype=torch.bfloat16)
    xk = torch.randn(1, 1, 3, 256, dtype=torch.bfloat16)
    xv = torch.randn(1, 1, 3, 256, dtype=torch.bfloat16)
    kv.write(xk, xv)
    assert kv.attend(xq, xk, xv).shape == (1, 3, 512)
    # the committed prefix is quantized once, then refreshed to the same length before attending
    assert kv.packed.updates == [([10], True), ([10], False)]
    k, v, mask = merges[-1]
    assert k is xk and v is xv and mask.tolist() == [1, 3, 7]


def test_int8_cache_keeps_bounded_circular_window(monkeypatch):
    for capacity in (3, 5, 11, 12, 13):
        kv, _ = _int8_cache(monkeypatch, capacity)
        assert kv.key.shape == (1, 1, min(capacity, 12), 256)
    kv, _ = _int8_cache(monkeypatch)
    # three 4-row pages of BF16 window for a 32-row capacity
    assert kv.key.shape == (1, 1, 12, 256)
    prefill = torch.randn(1, 1, 14, 256, dtype=torch.bfloat16)
    kv.prepare(14)
    out_k, _ = kv.append(prefill, prefill)
    assert out_k is prefill and kv.packed.updates == [([14], True)]
    # only the newest rows survive, absolute row i at slot i % 12
    assert torch.equal(kv.key[:, :, 2:12], prefill[:, :, 2:12])
    assert torch.equal(kv.key[:, :, :2], prefill[:, :, 12:14])
    kv.advance(14)
    kv.prepare(3)
    assert kv.position[:3].tolist() == [14, 15, 16]
    xk = torch.randn(1, 1, 3, 256, dtype=torch.bfloat16)
    kv.write(xk, xk)
    kv.attend(torch.randn(1, 2, 3, 256, dtype=torch.bfloat16), xk, xk)
    assert torch.equal(kv.key[:, :, 2:5], xk) and kv.packed.updates[-1] == ([14], False)


@pytest.mark.parametrize("path", [[0, 1, 2, 3, 4], [0, 1, 2, 5], [0, 7]])
def test_tree4s1m8_cache_commits_deep_and_sibling_paths(monkeypatch, path):
    from comfy.text_encoders.qwen35 import MTP_MODES, VerifyTree
    tree = VerifyTree(MTP_MODES["tree4s1m8"][1], 8, "cpu")
    kv, merges = _int8_cache(monkeypatch)
    kv.index = 10
    kv.tracker["tree"] = tree
    kv.prepare(8)
    xk = torch.randn(1, 1, 8, 256, dtype=torch.bfloat16)
    kv.write(xk, -xk)
    kv.attend(torch.randn(1, 2, 8, 256, dtype=torch.bfloat16), xk, -xk)
    assert merges[-1][2].tolist() == [1, 3, 7, 15, 31, 39, 67, 129]
    kv.advance(8)
    kv.rollback(8 - len(path))
    kv.commit_path(10, path)
    assert kv.index == 10 + len(path)
    for token, row in enumerate(path, 10):
        assert torch.equal(kv.key[:, :, token % 12], xk[:, :, row])
        assert torch.equal(kv.value[:, :, token % 12], -xk[:, :, row])


def _attention_config(config):
    return torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)


def test_checkpoint_and_lora_attention_select_each_kv_layer(monkeypatch):
    import comfy.ldm.modules.attention as attention
    import comfy.lora
    from comfy.model_patcher import ModelPatcher
    import comfy.text_encoders.llama as llama
    import comfy.text_encoders.qwen35 as qwen35
    from comfy.text_encoders.llama import FixedKVCache

    monkeypatch.setattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(qwen35.comfy_kitchen, "int8_attention_is_available", lambda device: True)
    monkeypatch.setattr(qwen35.comfy_kitchen, "sol_attn_is_available", lambda device: False)
    monkeypatch.setattr(qwen35, "int8_decode_is_available", lambda device: True)
    monkeypatch.setattr(llama, "Int8DecodeCache", _FakePacked)
    monkeypatch.setattr(cli_args, "use_ck_attention", False)
    model = _tiny_model(monkeypatch, layers=8, head_dim=256)
    key = "model.layers.3.self_attn.comfy_attention.config"
    mtp_key = "mtp.layers.0.self_attn.comfy_attention.config"
    encoded = _attention_config([{"attention": "comfy_kitchen_sol"}, {"attention": "comfy_kitchen_int8"}])
    sd = model.state_dict()
    sd[key] = sd[mtp_key] = encoded
    model.load_state_dict(sd, strict=True)
    assert torch.equal(model.state_dict()[key], encoded)
    device = torch.device("cpu")

    def cache_types():
        kv = model.init_kv_cache(1, 32, device, torch.bfloat16)
        draft = model.mtp.layers[0].self_attn.init_kv_cache(1, 32, device, torch.bfloat16, {})
        return type(kv[3]), type(kv[7]), type(draft)

    assert cache_types() == (qwen35.Int8FixedKVCache, FixedKVCache, qwen35.Int8FixedKVCache)
    monkeypatch.setattr(qwen35.comfy_kitchen, "sol_attn_is_available", lambda device: True)
    sol = _attention_config({"attention": "comfy_kitchen_sol", "tau": 0.75})
    patcher = ModelPatcher(model, device, device)
    patches = comfy.lora.load_lora({key: sol, mtp_key: sol}, {})
    assert set(patcher.add_patches(patches, strength_patch=0)) == {key, mtp_key}
    patcher.patch_model(load_weights=False)
    assert cache_types() == (qwen35.Int8FixedKVCache, FixedKVCache, qwen35.Int8FixedKVCache)
    patcher.unpatch_model(unpatch_weights=False)
    patcher.add_patches(patches)
    patcher.patch_model(load_weights=False)
    try:
        # A LoRA's supported setting beats both the checkpoint and the CLI default.
        monkeypatch.setattr(cli_args, "use_ck_attention", True)
        assert cache_types() == (FixedKVCache, qwen35.Int8FixedKVCache, FixedKVCache)
    finally:
        patcher.unpatch_model(unpatch_weights=False)
    monkeypatch.setattr(cli_args, "use_ck_attention", False)
    assert cache_types() == (qwen35.Int8FixedKVCache, FixedKVCache, qwen35.Int8FixedKVCache)
    # Reloading an unavailable setting clears the previously resolved INT8 selection.
    monkeypatch.setattr(qwen35.comfy_kitchen, "int8_attention_is_available", lambda device: False)
    sd[key] = sd[mtp_key] = _attention_config({"attention": "comfy_kitchen_int8"})
    model.load_state_dict(sd, strict=True)
    assert cache_types() == (FixedKVCache, FixedKVCache, FixedKVCache)


@pytest.mark.parametrize("seq,tree_verify", [(1, False), (3, False), (8, True)])
def test_configured_decode_preserves_causal_and_tree_visibility(monkeypatch, seq, tree_verify):
    import comfy.ldm.modules.attention as attention
    # Make the configured backend observably different from the default backend.
    @attention.wrap_attn
    def scaled_attention(q, k, v, heads, *, tau, **kwargs):
        return tau * attention.attention_pytorch(q, k, v, heads, **kwargs)

    monkeypatch.setattr(attention.comfy_kitchen, "sol_attn_is_available", lambda device: True)
    monkeypatch.setattr(attention, "attention_comfy_kitchen_sol", scaled_attention)
    attn = _gated_attention().float()
    attn.load_state_dict({"comfy_attention.config": _attention_config({"attention": "comfy_kitchen_sol", "tau": 0.25})}, strict=False)
    kv = attn.init_kv_cache(2, 24, torch.device("cpu"), torch.float32, {})
    torch.manual_seed(35)
    prefix, current = torch.randn(2, 9, 32), torch.randn(2, seq, 32)
    kv.prepare(9)
    attn(prefix, attention_mask=torch.full((9, 9), -torch.inf).triu(1),
         freqs_cis=torch.zeros(1, 1, 9, 32, 2, 2), optimized_attention=attention.attention_pytorch, past_key_value=kv)
    tree = _tree2s2() if tree_verify else None
    kv.tracker["tree"] = tree
    # Repeat after rollback; stale tree rows must not become visible.
    for start in (9, 4):
        kv.index = start
        kv.prepare(seq)
        out, _ = attn(current, freqs_cis=torch.zeros(1, 1, seq, 32, 2, 2), past_key_value=kv)
        full = torch.cat((prefix[:, :start], current), dim=1)
        mask = torch.full((start + seq, start + seq), -torch.inf).triu(1)
        parents = [-1, 0, 1, 1, 1, 0, 0, -1]
        for row in range(seq):
            ancestors = []
            r = row
            while r >= 0:
                ancestors.append(start + r)
                r = parents[r] if tree_verify else r - 1
            visible = list(range(start)) + sorted(ancestors)
            mask[start + row] = -torch.inf
            mask[start + row, visible] = 0
        expected, _ = attn(full, attention_mask=mask, freqs_cis=torch.zeros(1, 1, start + seq, 32, 2, 2),
                           optimized_attention=attention.attention_pytorch)
        torch.testing.assert_close(out, expected[:, start:])

    # The same checkpoint setting must also affect ordinary prefill/encode.
    x = torch.randn(1, 3, 32)
    freqs = torch.zeros(1, 1, 3, 32, 2, 2)
    mask = torch.full((3, 3), -torch.inf).triu(1)
    configured, _ = attn(x, attention_mask=mask, freqs_cis=freqs, optimized_attention=attention.attention_pytorch)
    attn.comfy_attention.load_state_dict({})
    default, _ = attn(x, attention_mask=mask, freqs_cis=freqs, optimized_attention=attention.attention_pytorch)
    assert default.abs().max() > 0
    torch.testing.assert_close(configured, default * 0.25)
