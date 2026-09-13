"""An incompatible comfy_kitchen must turn its features off, not kill startup.

comfy_kitchen registers custom ops in its module body. Before torch 2.7,
`torch.library.custom_op` rejects a PEP-585 annotation and raises

    ValueError: infer_schema(func): Parameter kernel_size has unsupported
    type list[int]

which is not an ImportError. The guards in `comfy.quant_ops` and
`comfy.model_prefetch` have to catch it, or ComfyUI cannot start at all on the
older torch builds legacy GPUs are pinned to (#16300).

Each case runs in its own subprocess: the failure is an import-time one, and
reloading these modules in-process would hand the rest of the session a second
copy of `QuantizedTensor`.
"""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[2]

# The reported message, verbatim, so the test fails for the reported reason.
CK_FAILURE = (
    "infer_schema(func): Parameter kernel_size has unsupported type list[int]"
)

_PRELUDE = dedent(
    f"""
    import sys

    class RaisingFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "comfy_kitchen" or fullname.startswith("comfy_kitchen."):
                raise ValueError({CK_FAILURE!r})
            return None

    for name in [m for m in sys.modules if m == "comfy_kitchen" or m.startswith("comfy_kitchen.")]:
        del sys.modules[name]
    sys.meta_path.insert(0, RaisingFinder())
    """
)


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + dedent(body)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_quant_ops_degrades_instead_of_raising():
    result = _run(
        """
        import comfy.quant_ops as quant_ops
        print("CK_AVAILABLE:", quant_ops._CK_AVAILABLE)
        print("QUANTIZED_TENSOR:", quant_ops.QuantizedTensor.__name__)
        """
    )

    assert result.returncode == 0, result.stderr
    assert "CK_AVAILABLE: False" in result.stdout, result.stdout
    # The fallback stubs are in place, so importers downstream still resolve.
    assert "QUANTIZED_TENSOR: QuantizedTensor" in result.stdout, result.stdout


def test_model_prefetch_imports_without_comfy_kitchen():
    """execution.py, latent_preview.py and comfy.sd all import this at startup,
    so quant_ops degrading on its own would not be enough."""
    result = _run(
        """
        import comfy.model_prefetch as model_prefetch
        print("CK_IS_NONE:", model_prefetch.ck is None)
        """
    )

    assert result.returncode == 0, result.stderr
    assert "CK_IS_NONE: True" in result.stdout, result.stdout


def test_float_falls_back_to_the_unavailable_stub():
    """The third import site, and the one quant_ops pulls in transitively: its
    guard listed AttributeError and ImportError, neither of which is raised."""
    result = _run(
        """
        import comfy.float as float_mod
        print("STOCHASTIC_AVAILABLE:", float_mod._CK_STOCHASTIC_ROUNDING_AVAILABLE)
        """
    )

    assert result.returncode == 0, result.stderr
    assert "STOCHASTIC_AVAILABLE: False" in result.stdout, result.stdout


def test_the_finder_really_breaks_comfy_kitchen():
    """Positive control: without it the two tests above would pass on any tree,
    including one where the import never failed in the first place."""
    result = _run(
        """
        try:
            import comfy_kitchen
        except ValueError as e:
            print("RAISED:", e)
        else:
            print("NOT_RAISED")
        """
    )

    assert result.returncode == 0, result.stderr
    assert f"RAISED: {CK_FAILURE}" in result.stdout, result.stdout
