from types import SimpleNamespace

from comfy_extras.sparse_attention_history import (
    H3_BLOCK_SCOPE,
    PROVIDER_KEY,
    SparseAttentionHistoryPolicy,
    append_history_receipt,
    install_history_policy,
    mark_attention_override,
)


def _forward():
    return None


def _model(blocks=3):
    return SimpleNamespace(
        blocks=[SimpleNamespace(attn=SimpleNamespace(forward=_forward)) for _ in range(blocks)],
        dtype="bf16",
    )


def _layout(rows=128):
    return SimpleNamespace(
        seq_len=rows,
        signature=(16, 2, 8, 8, 4),
        segments=((0, 16, "text"), (16, 24, "audio"), (24, rows, "video")),
    )


def _patch(blocks=3):
    return SimpleNamespace(
        vsa=False,
        h3_block_count=blocks,
        tau=1.0,
        topk_ratio=0.0,
        extra_tokens=256,
        sigma_start=0.8,
        sigma_end=0.0,
        min_tokens=64,
        dense_blocks={0},
        sink_conditioning="exact_kv",
        pooled={},
        installed=set(),
        history_policy=None,
    )


def _options(patch, sigma=0.5, uuids=("positive",)):
    previous = lambda *args, **kwargs: None
    override = mark_attention_override(lambda *args, **kwargs: None, patch, previous)
    patch.installed.add(override)
    return {
        "optimized_attention_override": override,
        "sigmas": [sigma],
        "uuids": uuids,
        "patches_replace": {
            "dit": {
                ("double_block", index): (lambda index=index: index)
                for index in range(patch.h3_block_count)
            }
        },
    }


def test_history_policy_registration_preserves_other_providers():
    patch = _patch()
    options = {"attention_backend_history_v1": {"other": object()}}
    assert install_history_policy(patch, options)
    providers = options["attention_backend_history_v1"]
    assert "other" in providers
    assert providers[PROVIDER_KEY] is patch.history_policy


def test_history_policy_does_not_overwrite_unknown_contract_shape():
    patch = _patch()
    options = {"attention_backend_history_v1": object()}
    assert not install_history_policy(patch, options)
    assert PROVIDER_KEY not in options


def test_history_identity_is_hashable_and_tracks_sparse_state():
    patch = _patch()
    options = _options(patch)
    policy = SparseAttentionHistoryPolicy(patch)
    identity_cold = policy(layout=_layout(), options=options, model=_model())
    assert identity_cold is not None
    hash(identity_cold)

    patch.pooled[(1, 128, ("positive",))] = (object(), object())
    identity_primed = policy(layout=_layout(), options=options, model=_model())
    assert identity_primed is not None
    assert identity_primed != identity_cold


def test_history_identity_tracks_dense_sparse_window_transition():
    patch = _patch()
    policy = SparseAttentionHistoryPolicy(patch)
    dense_options = _options(patch, sigma=0.9)
    sparse_options = dict(dense_options)
    sparse_options["sigmas"] = [0.5]
    dense_identity = policy(layout=_layout(), options=dense_options, model=_model())
    sparse_identity = policy(layout=_layout(), options=sparse_options, model=_model())
    assert dense_identity is not None
    assert sparse_identity is not None
    assert dense_identity != sparse_identity


def test_history_policy_refuses_vsa_stacked_bsa_and_lost_override():
    patch = _patch()
    options = _options(patch)
    policy = SparseAttentionHistoryPolicy(patch)

    patch.vsa = True
    assert policy(layout=_layout(), options=options, model=_model()) is None
    patch.vsa = False

    previous = lambda *args, **kwargs: None
    previous._block_sparse_attention_patch = object()
    options["optimized_attention_override"] = mark_attention_override(
        lambda *args, **kwargs: None, patch, previous
    )
    assert policy(layout=_layout(), options=options, model=_model()) is None

    options["optimized_attention_override"] = lambda *args, **kwargs: None
    assert policy(layout=_layout(), options=options, model=_model()) is None


def test_history_receipts_require_exact_h3_block_coverage():
    patch = _patch(blocks=3)
    policy = SparseAttentionHistoryPolicy(patch)
    options = {"attention_backend_receipts_v1": []}
    routes = ("h3_dense", "h3_chunked_sparse_cold", "h3_chunked_sparse_primed")
    for block, route in enumerate(routes):
        append_history_receipt(options, patch, block, route, 128, (0, 1), (0, 0))
    receipts = tuple(options["attention_backend_receipts_v1"])
    assert policy.accept_receipts(receipts)

    assert not policy.accept_receipts(receipts[:-1])
    duplicate = list(receipts)
    duplicate[-1] = duplicate[0]
    assert not policy.accept_receipts(tuple(duplicate))

    foreign = list(receipts)
    foreign[0] = ("other",) + foreign[0][1:]
    assert not policy.accept_receipts(tuple(foreign))

    wrong_tokens = list(receipts)
    wrong_tokens[-1] = wrong_tokens[-1][:5] + (64,) + wrong_tokens[-1][6:]
    assert not policy.accept_receipts(tuple(wrong_tokens))


def test_h3_block_patch_preserves_previous_replacement_and_scope(monkeypatch):
    import comfy_extras.nodes_sparse_attention as sparse

    patch = _patch(blocks=1)
    block = SimpleNamespace(attn=SimpleNamespace())
    calls = []

    monkeypatch.setattr(sparse, "h3_eligible", lambda *args, **kwargs: False)

    def previous(args, extra):
        calls.append(("previous", args["transformer_options"].get(H3_BLOCK_SCOPE)))
        return extra["original_block"](args)

    def original(args):
        calls.append(("original", args["transformer_options"].get(H3_BLOCK_SCOPE)))
        return {"img": "ok"}

    options = {}
    block_patch = sparse.make_h3_block_patch(block, 0, patch, previous)
    result = block_patch(
        {"img": object(), "rope_freqs": None, "transformer_options": options},
        {"original_block": original},
    )

    assert result == {"img": "ok"}
    assert calls == [("previous", (patch, 0)), ("original", (patch, 0))]
    assert H3_BLOCK_SCOPE not in options
    assert block_patch._block_sparse_attention_previous is previous


def test_h3_scope_key_is_private_and_stable():
    assert H3_BLOCK_SCOPE.startswith("_")
    assert "block_sparse_attention" in H3_BLOCK_SCOPE
