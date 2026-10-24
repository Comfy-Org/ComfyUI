import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy import conds, memory_management, model_base, model_management, model_patcher, ops, supported_models, system_memory


@pytest.mark.parametrize("vram_margin,ram_margin,reclaimable,transformer_options,admitted", [
    (-1, -1, 0, {}, False),
    (0, -1, 0, {}, False),
    (1, -1, 0, {}, True),
    (-1, 0, 0, {}, True),
    (-1, -4096, 4095, {}, False),
    (-1, -4096, 4096, {}, True),
    (1, 0, 0, {"patches_replace": {"dit": {("single_block", 0): object()}}}, False),
    (1, 0, 0, {"patches": {"post_input": [object()]}}, False),
    (1, 0, 0, {"patches": {"single_block": [object()]}}, False),
    (1, 0, 0, {"patches": {"attn1_patch": [object()]}}, False),
    (1, 0, 0, {"patches_replace": {"dit": {}}, "patches": {"post_input": [], "single_block": [], "attn1_patch": []}}, True),
    (1, 0, 0, {"patches": {"unrelated": [object()]}}, True),
])
def test_cache_budget_admits_all_conditions_together(monkeypatch, vram_margin, ram_margin, reclaimable, transformer_options, admitted):
    config = supported_models.QwenImage21(dict(num_layers=2, attention_head_dim=8, num_attention_heads=2,
        context_in_dim=16, in_channels=4, out_channels=4, axes_dims_rope=(2, 2, 4), dtype=torch.float32))
    config.custom_operations = ops.manual_cast
    model = model_base.QwenImage21(config, device="meta")
    patcher = model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    patcher.fast_disk = True
    model.current_patcher = patcher
    conditions = {name: [{"model_conds": {
        "c_crossattn": conds.CONDRegular(torch.empty(1, length, 16, device="meta")),
        "ref_latents": conds.CONDList([torch.empty(1, 4, 2, 3, device="meta")]),
    }}] for name, length in (("positive", 3), ("negative", 17))}

    # Two blocks: positive (9 tokens) rounds to 2 KiB; negative (23) to 3 KiB.
    cache_bytes = 2 * (2048 + 3072)
    monkeypatch.setattr(args, "high_ram", False)
    monkeypatch.setattr(memory_management, "RAM_CACHE_HEADROOM", 6 * 1024 ** 3)
    monkeypatch.setattr(system_memory, "virtual_memory_available", lambda: 3 * 1024 ** 3 + cache_bytes + ram_margin)
    monkeypatch.setattr(model_management, "free_pins", lambda size, **kwargs: min(size, reclaimable))
    monkeypatch.setattr(patcher, "get_free_memory", lambda device: 4 * cache_bytes + vram_margin)
    monkeypatch.setattr(patcher, "is_dynamic", lambda: True)

    cache = model.get_writable_model(conditions, (1, 4, 3, 5), {"transformer_options": transformer_options}, torch.device("cpu"))
    assert (cache is not None) == admitted
    assert all(("prefix_cache" in c[0]["model_conds"]) == admitted for c in conditions.values())
    if cache is not None:
        assert sum(p.nbytes for p in cache.parameters()) == cache_bytes
