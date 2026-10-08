import ast
import builtins
import functools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from comfy.cli_args import parser


ATTENTION_SOURCE = Path(__file__).parents[2] / "comfy" / "ldm" / "modules" / "attention.py"


def _run_aiter_import_probe(
    hip, explicitly_requested, other_backend_requested, import_error=None, cpu=False, arch="gfx1151", arch_error=None,
):
    source = ast.parse(ATTENTION_SOURCE.read_text(encoding="utf-8"))
    supported_arches = next(
        node for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_AITER_SUPPORTED_ARCHES" for target in node.targets)
    )
    device_arch_cache = next(
        node for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_AITER_DEVICE_ARCHES" for target in node.targets)
    )
    device_arch = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_aiter_device_arch"
    )
    probe = next(
        node for node in source.body
        if isinstance(node, ast.If) and "AITER_EXPLICITLY_REQUESTED" in ast.unparse(node.test)
    )

    imported = []

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "aiter.ops.mha":
            imported.append(name)
            if import_error is not None:
                raise import_error
            return SimpleNamespace(mha_fwd=object())
        return builtins.__import__(name, globals, locals, fromlist, level)

    class ProbeExit(Exception):
        pass

    def stop(code):
        raise ProbeExit(code)

    def get_device_properties(device):
        if arch_error is not None:
            raise arch_error
        return SimpleNamespace(gcnArchName=arch) if arch is not None else SimpleNamespace()

    properties = Mock(side_effect=get_device_properties)

    namespace = {
        "AITER_ATTENTION_IS_AVAILABLE": False,
        "AITER_EXPLICITLY_REQUESTED": explicitly_requested,
        "AITER_OTHER_BACKEND_REQUESTED": other_backend_requested,
        "args": SimpleNamespace(cpu=cpu),
        "torch": SimpleNamespace(
            version=SimpleNamespace(hip=hip),
            cuda=SimpleNamespace(get_device_properties=properties),
            device=torch.device,
        ),
        "model_management": SimpleNamespace(get_torch_device=lambda: torch.device("cuda:0")),
        "functools": functools,
        "logging": Mock(),
        "exit": stop,
        "__builtins__": {**vars(builtins), "__import__": fake_import},
    }
    try:
        exec(compile(ast.Module(body=[supported_arches, device_arch_cache, device_arch, probe], type_ignores=[]), str(ATTENTION_SOURCE), "exec"), namespace)
    except ProbeExit as e:
        return namespace, imported, e.args[0], properties
    return namespace, imported, None, properties


def _other_backend_requested(flags):
    source = ast.parse(ATTENTION_SOURCE.read_text(encoding="utf-8"))
    assignment = next(
        node for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "AITER_OTHER_BACKEND_REQUESTED" for target in node.targets)
    )
    args = SimpleNamespace(**{
        name: name in flags for name in (
            "use_split_cross_attention",
            "use_quad_cross_attention",
            "use_pytorch_cross_attention",
            "use_sage_attention",
            "use_flash_attention",
            "use_ck_attention",
        )
    })
    namespace = {"args": args}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(ATTENTION_SOURCE), "exec"), namespace)
    return namespace["AITER_OTHER_BACKEND_REQUESTED"]


@pytest.mark.parametrize(
    ("hip", "explicit", "other", "expected_import"),
    [
        (None, False, False, False),
        ("test-hip", False, False, True),
        ("test-hip", False, True, False),
        ("test-hip", True, False, True),
    ],
)
def test_aiter_import_probe_selection(hip, explicit, other, expected_import):
    namespace, imported, exit_code, properties = _run_aiter_import_probe(hip, explicit, other)

    assert bool(imported) is expected_import
    assert namespace["AITER_ATTENTION_IS_AVAILABLE"] is expected_import
    assert exit_code is None
    assert properties.call_count == expected_import


@pytest.mark.parametrize("arch", ["gfx1030", None])
def test_aiter_import_probe_skips_unsupported_or_unknown_auto_target(arch):
    namespace, imported, exit_code, properties = _run_aiter_import_probe("test-hip", False, False, arch=arch)

    assert not namespace["AITER_ATTENTION_IS_AVAILABLE"]
    assert imported == []
    assert exit_code is None
    assert properties.call_count == 1


@pytest.mark.parametrize("arch,arch_error", [("gfx1030", None), (None, None), (None, RuntimeError("unknown device"))])
def test_explicit_aiter_rejects_unsupported_or_unknown_target_before_import(arch, arch_error):
    namespace, imported, exit_code, properties = _run_aiter_import_probe(
        "test-hip", True, False, arch=arch, arch_error=arch_error,
    )

    assert exit_code == -1
    assert imported == []
    assert properties.call_count == 1
    namespace["logging"].error.assert_called_once()


def test_cpu_mode_does_not_auto_import_aiter():
    namespace, imported, exit_code, properties = _run_aiter_import_probe("test-hip", False, False, cpu=True)

    assert imported == []
    assert namespace["AITER_ATTENTION_IS_AVAILABLE"] is False
    assert exit_code is None
    properties.assert_not_called()


def test_other_explicit_backend_does_not_query_or_import_aiter():
    namespace, imported, exit_code, properties = _run_aiter_import_probe("test-hip", False, True)

    assert imported == []
    assert namespace["AITER_ATTENTION_IS_AVAILABLE"] is False
    assert exit_code is None
    properties.assert_not_called()


def test_aiter_import_probe_errors_only_for_explicit_request():
    namespace, imported, exit_code, _ = _run_aiter_import_probe(None, True, False)
    assert imported == []
    assert exit_code == -1
    assert namespace["logging"].error.called
    _, imported, exit_code, properties = _run_aiter_import_probe("test-hip", True, False, cpu=True)
    assert imported == []
    assert exit_code == -1
    properties.assert_not_called()

    namespace, imported, exit_code, _ = _run_aiter_import_probe("test-hip", False, False, ImportError("missing"))
    assert imported == ["aiter.ops.mha"]
    assert namespace["AITER_ATTENTION_IS_AVAILABLE"] is False
    assert exit_code is None

    namespace, imported, exit_code, _ = _run_aiter_import_probe("test-hip", True, False, OSError("native module unavailable"))
    assert imported == ["aiter.ops.mha"]
    assert exit_code == -1
    assert namespace["logging"].error.called


def test_aiter_flag_is_exclusive_with_existing_attention_backends(capsys):
    existing_flags = (
        "--use-split-cross-attention",
        "--use-quad-cross-attention",
        "--use-pytorch-cross-attention",
        "--use-sage-attention",
        "--use-flash-attention",
        "--use-ck-attention",
    )
    assert parser.parse_args(["--use-aiter-attention"]).use_aiter_attention
    for flag in existing_flags:
        with pytest.raises(SystemExit):
            parser.parse_args(["--use-aiter-attention", flag])
        capsys.readouterr()

    assert not _other_backend_requested(set())
    for name in (
        "use_split_cross_attention",
        "use_quad_cross_attention",
        "use_pytorch_cross_attention",
        "use_sage_attention",
        "use_flash_attention",
        "use_ck_attention",
    ):
        assert _other_backend_requested({name})


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1151:sramecc+:xnack-"])
def test_aiter_device_arch_reads_gcn_target_and_strips_suffix(arch):
    source = ast.parse(ATTENTION_SOURCE.read_text(encoding="utf-8"))
    supported_arches = next(
        node for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_AITER_SUPPORTED_ARCHES" for target in node.targets)
    )
    device_arch_cache = next(
        node for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_AITER_DEVICE_ARCHES" for target in node.targets)
    )
    device_arch = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_aiter_device_arch"
    )
    properties = Mock(return_value=SimpleNamespace(gcnArchName=arch))
    namespace = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(get_device_properties=properties)),
        "functools": functools,
    }
    exec(compile(ast.Module(body=[supported_arches, device_arch_cache, device_arch], type_ignores=[]), str(ATTENTION_SOURCE), "exec"), namespace)

    assert namespace["_aiter_device_arch"](torch.device("cuda:0")) == "gfx1151"
    properties.assert_called_once_with(torch.device("cuda:0"))


@pytest.mark.parametrize("properties_result", [SimpleNamespace(), RuntimeError("unknown device")])
def test_aiter_device_arch_returns_unknown_when_runtime_property_is_unavailable(properties_result):
    source = ast.parse(ATTENTION_SOURCE.read_text(encoding="utf-8"))
    device_arch = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_aiter_device_arch"
    )
    device_arch_cache = next(
        node for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_AITER_DEVICE_ARCHES" for target in node.targets)
    )
    properties = Mock()
    if isinstance(properties_result, Exception):
        properties.side_effect = properties_result
    else:
        properties.return_value = properties_result
    namespace = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(get_device_properties=properties)),
        "functools": functools,
    }
    exec(compile(ast.Module(body=[device_arch_cache, device_arch], type_ignores=[]), str(ATTENTION_SOURCE), "exec"), namespace)

    assert namespace["_aiter_device_arch"](torch.device("cuda:0")) is None


def test_aiter_device_arch_skips_device_query_for_cpu():
    source = ast.parse(ATTENTION_SOURCE.read_text(encoding="utf-8"))
    device_arch = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_aiter_device_arch"
    )
    device_arch_cache = next(
        node for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_AITER_DEVICE_ARCHES" for target in node.targets)
    )
    properties = Mock()
    namespace = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(get_device_properties=properties)),
        "functools": functools,
    }
    exec(compile(ast.Module(body=[device_arch_cache, device_arch], type_ignores=[]), str(ATTENTION_SOURCE), "exec"), namespace)

    assert namespace["_aiter_device_arch"](torch.device("cpu")) is None
    properties.assert_not_called()


def test_aiter_selector_captures_and_replaces_only_unmasked_backend():
    source = ast.parse(ATTENTION_SOURCE.read_text(encoding="utf-8"))
    masked_alias = next(
        node for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "optimized_attention_masked" for target in node.targets)
    )
    selector = next(
        node for node in source.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "AITER_ATTENTION_IS_AVAILABLE"
    )

    baseline = object()
    aiter = object()
    namespace = {
        "optimized_attention": baseline,
        "AITER_ATTENTION_IS_AVAILABLE": True,
        "attention_aiter": aiter,
        "logging": Mock(),
    }
    exec(compile(ast.Module(body=[masked_alias, selector], type_ignores=[]), str(ATTENTION_SOURCE), "exec"), namespace)

    assert namespace["optimized_attention"] is aiter
    assert namespace["optimized_attention_masked"] is baseline
    assert namespace["_AITER_FALLBACK"] is baseline


@pytest.fixture
def attention_namespace():
    source = ast.parse(ATTENTION_SOURCE.read_text(encoding="utf-8"))
    wanted = {"AttentionTensorContainer", "wrap_attn", "get_attn_precision", "attention_aiter"}
    nodes = [
        node for node in source.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    ]
    namespace = {
        "torch": torch,
        "functools": functools,
        "args": SimpleNamespace(dont_upcast_attention=False),
        "FORCE_UPCAST_ATTENTION_DTYPE": {},
        "_AITER_SUPPORTED_ARCHES": {"gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201"},
        "_aiter_device_arch": lambda device: "gfx1151",
        "attention_sub_quad": lambda q, k, v, *args, **kwargs: q + k + v,
        "_AITER_FALLBACK": None,
        "mha_fwd": Mock(),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ATTENTION_SOURCE), "exec"), namespace)
    return namespace


def test_aiter_cpu_fallback_forwards_gradients(attention_namespace):
    q = torch.randn(1, 4, 8, requires_grad=True)
    k = torch.randn(1, 4, 8, requires_grad=True)
    v = torch.randn(1, 4, 8, requires_grad=True)

    output = attention_namespace["attention_aiter"](q, k, v, 2)

    assert output.shape == q.shape
    assert output.dtype == q.dtype
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_preferred_attention_keeps_precedence_over_aiter(attention_namespace):
    q, k, v = (torch.randn(1, 2, 4) for _ in range(3))
    preferred = SimpleNamespace(function=Mock(return_value="preferred-result"))

    result = attention_namespace["attention_aiter"](q, k, v, 2, preferred_attention=preferred)

    assert result == "preferred-result"
    preferred.function.assert_called_once()


def test_aiter_fallback_uses_pre_aiter_backend_once_and_forwards_kwargs(attention_namespace, monkeypatch):
    q = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    k = object()
    v = object()
    mask = object()
    options = {"test_option": object()}
    fallback = Mock(return_value="fallback-result")
    attention_namespace["_AITER_FALLBACK"] = fallback

    result = attention_namespace["attention_aiter"](
        q, k, v, 3, mask=mask, attn_precision=torch.float32,
        skip_reshape=True, skip_output_reshape=True,
        transformer_options=options,
    )

    assert result == "fallback-result"
    fallback.assert_called_once_with(
        q, k, v, 3, mask=mask, attn_precision=torch.float32,
        skip_reshape=True, skip_output_reshape=True,
        transformer_options=options, _inside_attn_wrapper=True,
    )


@pytest.mark.parametrize("fallback_reason", ["upcast", "autograd"])
def test_aiter_ineligible_cuda_inputs_fall_back(attention_namespace, monkeypatch, fallback_reason):
    q = SimpleNamespace(
        device=SimpleNamespace(type="cuda"), dtype=torch.float16,
        requires_grad=(fallback_reason == "autograd"),
    )
    k = SimpleNamespace(device=q.device, dtype=q.dtype, requires_grad=False)
    v = SimpleNamespace(device=q.device, dtype=q.dtype, requires_grad=False)
    fallback = Mock(return_value="fallback-result")
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    attention_namespace["_AITER_FALLBACK"] = fallback
    attention_namespace["FORCE_UPCAST_ATTENTION_DTYPE"] = {torch.float16: torch.float32}
    if fallback_reason == "upcast":
        monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)
    else:
        attention_namespace["FORCE_UPCAST_ATTENTION_DTYPE"] = {}

    result = attention_namespace["attention_aiter"](q, k, v, 1)

    assert result == "fallback-result"
    assert fallback.call_count == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["flat", "flat_head_output", "headed"])
@pytest.mark.parametrize("scale", [None, 0.125])
def test_aiter_eligible_fake_cuda_shapes_and_output_layout(attention_namespace, monkeypatch, dtype, layout, scale):
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mha_calls = []

    def fake_mha_fwd(q, k, v, dropout, scale, causal, window_left, window_right, sink, return_lse, return_dropout):
        mha_calls.append((q.shape, k.shape, v.shape, scale))
        return (torch.empty_like(q),)

    attention_namespace["mha_fwd"] = fake_mha_fwd
    with FakeTensorMode():
        if layout == "headed":
            q = torch.randn((1, 2, 3, 8), device="cuda", dtype=dtype)
            k = torch.randn((1, 2, 5, 8), device="cuda", dtype=dtype)
            v = torch.randn((1, 2, 5, 8), device="cuda", dtype=dtype)
            output = attention_namespace["attention_aiter"](
                q, k, v, 2, skip_reshape=True, skip_output_reshape=True, scale=scale,
            )
            expected_native_shape = (1, 3, 2, 8)
            expected_output_shape = q.shape
        else:
            q = torch.randn((1, 3, 16), device="cuda", dtype=dtype)
            k = torch.randn((1, 5, 16), device="cuda", dtype=dtype)
            v = torch.randn((1, 5, 16), device="cuda", dtype=dtype)
            output = attention_namespace["attention_aiter"](
                q, k, v, 2, skip_output_reshape=(layout == "flat_head_output"), scale=scale,
            )
            expected_native_shape = (1, 3, 2, 8)
            expected_output_shape = (1, 2, 3, 8) if layout == "flat_head_output" else q.shape

    assert output.shape == expected_output_shape
    assert output.dtype == dtype
    assert len(mha_calls) == 1
    assert mha_calls[0][:3] == (expected_native_shape, (1, 5, 2, 8), (1, 5, 2, 8))
    assert mha_calls[0][3] == pytest.approx(8 ** -0.5 if scale is None else scale)


@pytest.mark.parametrize(
    ("case", "mask_shape"),
    [
        ("fp32", None),
        ("mixed_dtype", None),
        ("gqa", None),
        ("causal", None),
        ("mask", (3, 5)),
        ("mask", (1, 3, 5)),
        ("mask", (1, 1, 3, 5)),
        ("head_dim_mismatch", None),
        ("headed_layout_mismatch", None),
        ("empty_batch", None),
        ("empty_q", None),
        ("head_dim_zero", None),
        ("head_dim4", None),
        ("head_dim10", None),
        ("head_dim264", None),
    ],
)
def test_aiter_unsupported_cuda_inputs_use_saved_backend(attention_namespace, monkeypatch, case, mask_shape):
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fallback = Mock(return_value="fallback-result")
    native = Mock()
    attention_namespace["_AITER_FALLBACK"] = fallback
    attention_namespace["mha_fwd"] = native

    with FakeTensorMode():
        if case in {"empty_batch", "empty_q", "head_dim_zero", "head_dim4", "head_dim10", "head_dim264"}:
            batch = 0 if case == "empty_batch" else 1
            q_length = 0 if case == "empty_q" else 3
            head_dim = {
                "head_dim_zero": 0,
                "head_dim4": 4,
                "head_dim10": 10,
                "head_dim264": 264,
            }.get(case, 8)
            q = torch.randn((batch, q_length, 2 * head_dim), device="cuda", dtype=torch.float16)
            k = torch.randn((batch, 5, 2 * head_dim), device="cuda", dtype=torch.float16)
            v = torch.randn((batch, 5, 2 * head_dim), device="cuda", dtype=torch.float16)
            result = attention_namespace["attention_aiter"](q, k, v, 2, scale=0.125)
            kwargs = {"scale": 0.125}
            mask = None
        else:
            q_dtype = torch.float32 if case == "fp32" else torch.float16
            k_dtype = torch.bfloat16 if case == "mixed_dtype" else q_dtype
            q_shape = (1, 2, 3, 4) if case == "headed_layout_mismatch" else (1, 3, 16)
            k_shape = (1, 1, 5, 4) if case == "headed_layout_mismatch" else (1, 5, 24 if case == "head_dim_mismatch" else 16)
            v_shape = (1, 2, 5, 4) if case == "headed_layout_mismatch" else (1, 5, 24 if case == "head_dim_mismatch" else 16)
            q = torch.randn(q_shape, device="cuda", dtype=q_dtype)
            k = torch.randn(k_shape, device="cuda", dtype=k_dtype)
            v = torch.randn(v_shape, device="cuda", dtype=q_dtype)
            mask = torch.zeros(mask_shape) if mask_shape is not None else None
            kwargs = {"scale": 0.125}
            if case == "gqa":
                kwargs["enable_gqa"] = True
            if case == "causal":
                kwargs["is_causal"] = True
            if case == "headed_layout_mismatch":
                kwargs["skip_reshape"] = True

            result = attention_namespace["attention_aiter"](q, k, v, 2, mask=mask, **kwargs)

    assert result == "fallback-result"
    fallback.assert_called_once()
    assert all(actual is expected for actual, expected in zip(fallback.call_args.args[:3], (q, k, v)))
    assert fallback.call_args.kwargs["scale"] == 0.125
    if mask is not None:
        assert fallback.call_args.kwargs["mask"] is mask
    if case == "gqa":
        assert fallback.call_args.kwargs["enable_gqa"] is True
    if case == "causal":
        assert fallback.call_args.kwargs["is_causal"] is True
    if case == "headed_layout_mismatch":
        assert fallback.call_args.kwargs["skip_reshape"] is True
    native.assert_not_called()


@pytest.mark.parametrize("explicit", [False, True])
def test_aiter_unsupported_secondary_device_falls_back_or_errors(attention_namespace, monkeypatch, explicit):
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    attention_namespace["_aiter_device_arch"] = lambda device: "gfx1030"
    attention_namespace["AITER_EXPLICITLY_REQUESTED"] = explicit
    attention_namespace["_AITER_FALLBACK"] = Mock(return_value="fallback-result")
    attention_namespace["mha_fwd"] = Mock()
    q = SimpleNamespace(device=torch.device("cuda:1"))
    k, v = object(), object()

    if explicit:
        with pytest.raises(RuntimeError, match="gfx1030.*not supported"):
            attention_namespace["attention_aiter"](q, k, v, 1)
        attention_namespace["_AITER_FALLBACK"].assert_not_called()
    else:
        result = attention_namespace["attention_aiter"](q, k, v, 1)
        assert result == "fallback-result"
        attention_namespace["_AITER_FALLBACK"].assert_called_once()
    attention_namespace["mha_fwd"].assert_not_called()


def test_attention_container_and_model_override_contract(attention_namespace):
    source = ast.parse(Path(__file__).parents[2].joinpath("comfy", "model_patcher.py").read_text(encoding="utf-8"))
    patcher = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ModelPatcher")
    method = next(node for node in patcher.body if isinstance(node, ast.FunctionDef) and node.name == "set_model_optimized_attention")
    exec(compile(ast.Module(body=[method], type_ignores=[]), "comfy/model_patcher.py", "exec"), attention_namespace)
    patcher = SimpleNamespace(model_options={"transformer_options": {}})
    attention_namespace["set_model_optimized_attention"](patcher, attention_namespace["attention_aiter"])
    transformer_options = patcher.model_options["transformer_options"]
    q = attention_namespace["AttentionTensorContainer"](torch.randn(1, 4, 8))
    k = attention_namespace["AttentionTensorContainer"](torch.randn(1, 4, 8))
    v = attention_namespace["AttentionTensorContainer"](torch.randn(1, 4, 8))

    output = attention_namespace["attention_aiter"](
        q, k, v, 2, transformer_options=transformer_options,
    )

    assert output.shape == (1, 4, 8)
    for container in (q, k, v):
        with pytest.raises(RuntimeError, match="already been consumed"):
            container.peek()
