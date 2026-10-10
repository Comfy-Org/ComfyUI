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
    # the additive mask reproduces attend()'s chain staircase over the whole cache
    from comfy.text_encoders.llama import FixedKVCache
    torch.manual_seed(0)
    shared = FixedKVCache.shared(2, "cpu")
    kv = FixedKVCache.zeros(2, 1, 24, 64, "cpu", torch.float32, shared)
    kv.key.normal_()
    kv.value.normal_()
    kv.index = 9
    for seq in (1, 3):
        kv.prepare(seq)
        xq = torch.randn(2, 2, seq, 64)
        xk = torch.randn(2, 1, seq, 64)
        xv = torch.randn(2, 1, seq, 64)
        kv.write(xk, xv)
        expected = kv.attend(xq, xk, xv)
        mask = kv.mask(seq)
        assert mask.shape == (2, 1, seq, 24) and mask.dtype == torch.float32
        visible = (mask == 0)[0, 0]
        assert visible.sum(-1).tolist() == [10 + j for j in range(seq)]
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
    assert ctl.shape == (2,)  # {pending, parity}
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
    # auto and fixed-depth chain verify through the kitchen torch paths: causal GQA, deferred DeltaNet
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
    for mtp in (True, 3, 5):
        discards.clear()
        assert model.generate(embeds.clone(), do_sample=False, max_length=40, mtp=mtp, position_ids=positions) == plain
        assert discards and min(discards) < max(discards)  # partial acceptance happened


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

    def step_merge(out, lse, q, k, v, result):
        merges.append((k, v))
        return out
    monkeypatch.setattr(llama, "Int8DecodeCache", _FakePacked, raising=False)
    monkeypatch.setattr(llama.comfy_kitchen, "flash_attention_decode_step_merge", step_merge)
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
    k, v = merges[-1]
    assert k is xk and v is xv


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


@pytest.mark.parametrize("seq", [1, 3])
def test_configured_decode_preserves_causal_visibility(monkeypatch, seq):
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
    # Repeat after rollback; stale rows past the current chain must not become visible.
    for start in (9, 4):
        kv.index = start
        kv.prepare(seq)
        out, _ = attn(current, freqs_cis=torch.zeros(1, 1, seq, 32, 2, 2), past_key_value=kv)
        full = torch.cat((prefix[:, :start], current), dim=1)
        mask = torch.full((start + seq, start + seq), -torch.inf).triu(1)
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
