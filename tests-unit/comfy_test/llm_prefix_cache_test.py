"""Prefix cache: cached generation must match an uncached run (small random Qwen3.5-shaped model, fp32)."""

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops  # noqa: E402
from comfy.text_encoders import llm_prefix_cache as pc  # noqa: E402
from comfy.text_encoders.qwen35 import Qwen35  # noqa: E402

MSG_END = (510, 509)  # stands in for "<|im_end|>\n" in the small vocab


class Tiny(Qwen35):
    model_type = "tiny"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = dict(hidden_size=128, intermediate_size=256, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
               head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
               vocab_size=512, layer_types=[("full_attention" if (i + 1) % 4 == 0 else "linear_attention") for i in range(8)],
               stop_tokens=[511], mtp=True, lm_head=True)
    m = Tiny(cfg, torch.float32, "cpu", comfy.ops.disable_weight_init)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("A_log"):
                p.uniform_(-2, 0.5)
            elif p.ndim == 1:
                p.normal_(0, 0.1)
            else:
                p.normal_(0, 0.08)
        m.model.lm_head.weight.mul_(4)  # clear greedy winners
    return m.eval()


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(pc, "MSG_END", MSG_END)
    monkeypatch.setattr(pc, "HEADROOM", 64)
    pc.drop()
    yield
    pc.drop()


def generate(model, ids, mtp, cache, max_length=24):
    embeds = model.model.embed_tokens(torch.tensor([ids]))
    kw = {"cache_ids": list(ids)} if cache else {}
    with torch.no_grad():
        return model.generate(embeds, do_sample=False, max_length=max_length, temperature=1.0, stop_tokens=[511], mtp=mtp, **kw)


def tokens(seed, n):
    return torch.randint(0, 500, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def messages(seed, count, size):
    out = []
    for i in range(count):
        out += tokens(seed * 100 + i, size) + list(MSG_END)
    return out


@pytest.mark.parametrize("mtp", [False, True])
def test_reuse_matches_uncached(model, mtp):
    prompt = tokens(1, 300)
    first = generate(model, prompt, mtp, cache=True)
    assert first == generate(model, prompt, mtp, cache=False)
    # the next prompt carries the reply: resumes from the end-of-generation checkpoint
    second = prompt + first + tokens(2, 40)
    assert generate(model, second, mtp, cache=True) == generate(model, second, mtp, cache=False)
    assert pc.stats["reused"] > len(prompt)
    # extends the previous prompt but not its reply: resumes from the prompt-end checkpoint
    third = second + tokens(3, 60)
    assert generate(model, third, mtp, cache=True) == generate(model, third, mtp, cache=False)
    assert pc.stats["reused"] == len(second)


@pytest.mark.parametrize("mtp", [False, True])
def test_rewritten_message_resumes_at_boundary(model, mtp):
    system = messages(5, 1, 200)
    base = system + messages(6, 8, 30)
    generate(model, base, mtp, cache=True)
    # an earlier message shrinks, as when an agent elides an old tool result
    cut = len(system) + 4 * 32
    changed = base[:cut] + [7, 8, 9] + list(MSG_END) + base[cut + 32:] + messages(7, 1, 20)
    assert generate(model, changed, mtp, cache=True) == generate(model, changed, mtp, cache=False)
    assert pc.stats["reused"] == cut


def test_diverging_prompt_misses(model):
    prompt = tokens(1, 300)
    generate(model, prompt, False, cache=True)
    other = prompt[:100] + tokens(4, 60) + prompt[100:]
    assert generate(model, other, False, cache=True) == generate(model, other, False, cache=False)
    assert pc.stats["reused"] == 0


def test_longer_prompt_grows_the_cache(model, monkeypatch):
    monkeypatch.setattr(pc, "HEADROOM", 8)
    prompt = tokens(1, 300)
    generate(model, prompt, True, cache=True)
    longer = prompt + tokens(2, 40)
    assert generate(model, longer, True, cache=True) == generate(model, longer, True, cache=False)
    assert pc.stats["restore"] == "grow"


def test_other_weights_miss(model):
    prompt = tokens(1, 300)
    generate(model, prompt, False, cache=True)
    with torch.no_grad():
        model.model.norm.weight.add_(0.01)
    try:
        longer = prompt + tokens(2, 40)
        assert generate(model, longer, False, cache=True) == generate(model, longer, False, cache=False)
        assert pc.stats["reused"] == 0
    finally:
        with torch.no_grad():
            model.model.norm.weight.sub_(0.01)


def test_suffix_prefill_matches_full_prefill(model):
    prompt, suffix = tokens(1, 300), tokens(2, 40)

    def forward(e, kv):
        return model.model.forward(None, embeds=e, attention_mask=None, past_key_values=kv)[0]

    with torch.no_grad():
        embeds = model.model.embed_tokens(torch.tensor([prompt + suffix]))  # forward writes into its input: clone per use
        pc.prefill(model, embeds[:, :len(prompt)].clone(), prompt, len(prompt) + 100, forward)
        _, cached = pc.prefill(model, embeds.clone(), prompt + suffix, len(prompt) + 100, forward)
        assert pc.stats["reused"] == len(prompt)
        pc.drop()
        _, full = pc.prefill(model, embeds.clone(), prompt + suffix, len(prompt) + 100, forward)
    assert torch.allclose(cached[:, -1], full[:, -1], atol=1e-4)


@pytest.mark.parametrize("mtp", [False, True])
def test_ram_mode_parks_and_restores(model, mtp, monkeypatch):
    monkeypatch.setattr(pc, "MODE", "ram")
    prompt = messages(5, 1, 200) + tokens(1, 100)
    generate(model, prompt, mtp, cache=True)
    assert pc._slot.parked is not None
    longer = prompt + tokens(2, 40)
    assert generate(model, longer, mtp, cache=True) == generate(model, longer, mtp, cache=False)
    assert pc.stats["restore"] == "from-ram"


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("mtp", [False, True])
def test_disk_restore_in_a_new_session(model, mtp, direct, monkeypatch, tmp_path):
    from comfy.text_encoders import llm_prefix_cache_disk as disk
    monkeypatch.setattr(pc, "DISK", str(tmp_path))
    monkeypatch.setattr(disk, "DIRECT", direct)
    system = messages(5, 1, 200)
    generate(model, system + messages(6, 3, 30), mtp, cache=True)
    assert pc.stats["disk_saved"] == len(system)
    pc.drop()  # a fresh process: nothing in memory, the system prompt on disk
    other = system + messages(8, 2, 25)
    assert generate(model, other, mtp, cache=True) == generate(model, other, mtp, cache=False)
    assert pc.stats["restore"] == "from-disk" and pc.stats["reused"] == len(system)
    # a different system prompt must not hit
    pc.drop()
    changed = messages(9, 1, 200) + messages(8, 2, 25)
    generate(model, changed, mtp, cache=True)
    assert pc.stats["reused"] == 0
