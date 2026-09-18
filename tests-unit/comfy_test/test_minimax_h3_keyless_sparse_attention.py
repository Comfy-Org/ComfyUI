from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy_extras.nodes_sparse_attention as sparse  # noqa: E402


class _Contract:
    api = 1
    architecture = "h3_keyless_core50_v1"
    core_blocks = 50
    token_refiner = "native_qkv"
    token_refiner_blocks = 2
    routing_source = "value"
    retrieval_source = "raw_projected_value"
    projection_attr = "qv_proj"
    qv_order = "q_effective;v"


class _KeylessAttention:
    def __init__(self):
        self.qv_proj = object()
        self.route_norm = object()


class _KeylessDiffusion:
    def __init__(self, *, contract=None):
        self.blocks = [SimpleNamespace(attn=_KeylessAttention()) for _ in range(50)]
        setattr(
            self,
            sparse.KEYLESS_H3_CONTRACT_KEY,
            _Contract() if contract is None else contract,
        )


class _Sampling:
    def percent_to_sigma(self, value):
        return 1.0 - float(value)


class _Patcher:
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self.sampling = _Sampling()
        self.model_options = {"transformer_options": {}}
        self.callbacks = []
        self.replacements = []
        self.clone_calls = 0

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self.diffusion
        if name == "model_sampling":
            return self.sampling
        raise KeyError(name)

    def clone(self):
        self.clone_calls += 1
        clone = _Patcher(self.diffusion)
        clone.model_options = {
            "transformer_options": dict(self.model_options["transformer_options"])
        }
        return clone

    def add_callback_with_key(self, *args):
        self.callbacks.append(args)

    def set_model_patch_replace(self, patch, *keys):
        self.replacements.append((patch, keys))


def _apply(model, *, vsa=False):
    return sparse.apply_block_sparse_attention(
        model,
        tau=1.3,
        topk_ratio=0.0,
        vsa=vsa,
        start_percent=0.2,
        end_percent=1.0,
        min_tokens=0,
        dense_blocks=set(),
        sink_conditioning="exact_kv_and_rows",
        extra_tokens=0,
        verbose=False,
    )


def test_keyless_contract_requires_qv_route_topology_without_fake_k():
    diffusion = _KeylessDiffusion()
    contract = sparse._keyless_h3_contract(diffusion)

    assert contract is getattr(diffusion, sparse.KEYLESS_H3_CONTRACT_KEY)
    assert all(hasattr(block.attn, "qv_proj") for block in diffusion.blocks)
    assert all(not hasattr(block.attn, "qkv_proj") for block in diffusion.blocks)
    assert all(not hasattr(block.attn, "k_norm") for block in diffusion.blocks)


def test_malformed_explicit_keyless_contract_fails_closed():
    diffusion = _KeylessDiffusion(
        contract=SimpleNamespace(
            api=1,
            architecture="h3_keyless_core50_v1",
            core_blocks=50,
            token_refiner="native_qkv",
            token_refiner_blocks=2,
            routing_source="key",
            retrieval_source="raw_projected_value",
            projection_attr="qv_proj",
            qv_order="q_effective;v",
        )
    )

    with pytest.raises(ValueError, match="routing_source"):
        sparse._keyless_h3_contract(diffusion)


def test_keyless_sol_sparse_uses_generic_override_without_qkv_block_replacements(monkeypatch):
    class NativeH3:
        pass

    class KeylessNativeSubclass(_KeylessDiffusion, NativeH3):
        pass

    monkeypatch.setattr(sparse, "MiniMaxH3Model", NativeH3)
    source = _Patcher(KeylessNativeSubclass())

    patched = _apply(source, vsa=False)

    assert source.clone_calls == 1
    assert source.model_options["transformer_options"] == {}
    assert "optimized_attention_override" in patched.model_options["transformer_options"]
    assert patched.replacements == []
    assert len(patched.callbacks) == 2


def test_keyless_vsa_fails_before_clone_or_callback_installation(monkeypatch):
    class NativeH3:
        pass

    class KeylessNativeSubclass(_KeylessDiffusion, NativeH3):
        pass

    monkeypatch.setattr(sparse, "MiniMaxH3Model", NativeH3)
    source = _Patcher(KeylessNativeSubclass())

    with pytest.raises(ValueError, match="VSA selection is not compatible"):
        _apply(source, vsa=True)

    assert source.clone_calls == 0
    assert source.model_options["transformer_options"] == {}
    assert source.callbacks == []


def test_native_h3_still_installs_chunked_block_producer(monkeypatch):
    class NativeH3:
        def __init__(self):
            self.blocks = [
                SimpleNamespace(attn=SimpleNamespace(to_gate_compress=None))
                for _ in range(3)
            ]

    monkeypatch.setattr(sparse, "MiniMaxH3Model", NativeH3)
    source = _Patcher(NativeH3())

    patched = _apply(source, vsa=False)

    assert len(patched.replacements) == 3
    assert all(keys == ("dit", "double_block", i) for i, (_, keys) in enumerate(patched.replacements))
