import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def run_qwen35_script(body):
    script = textwrap.dedent(
        """
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.cli_args
        """
    ) + textwrap.dedent(body)
    result = subprocess.run(
        [sys.executable, "-c", script, "--cpu"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gated_delta_decode_skips_disabled_backends():
    run_qwen35_script(
        """
        from unittest.mock import patch
        from comfy.text_encoders import qwen35

        checked_backends = []
        capability_checks = []

        def backend_disabled(backend):
            checked_backends.append(backend)
            return False

        with patch.object(qwen35.comfy_kitchen.registry, "is_available", side_effect=backend_disabled), \\
             patch.object(qwen35.comfy_kitchen, "gated_delta_decode_is_available", side_effect=lambda *args: capability_checks.append(args) or True):
            assert not qwen35._can_use_gated_delta_decode("cuda:0", 128, 128)

        assert checked_backends == ["cuda", "hip"]
        assert capability_checks == []
        """
    )


@pytest.mark.parametrize("enabled_backend", ["cuda", "hip"])
def test_gated_delta_decode_keeps_enabled_backend(enabled_backend):
    run_qwen35_script(
        f"""
        from unittest.mock import patch
        from comfy.text_encoders import qwen35

        with patch.object(qwen35.comfy_kitchen.registry, "is_available", side_effect=lambda backend: backend == {enabled_backend!r}), \\
             patch.object(qwen35.comfy_kitchen, "gated_delta_decode_is_available", return_value=True):
            assert qwen35._can_use_gated_delta_decode("cuda:0", 128, 128)
        """
    )


def test_gated_delta_forward_uses_eager_path_when_backends_are_disabled():
    run_qwen35_script(
        """
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        import torch
        from comfy.text_encoders import qwen35

        class Config:
            hidden_size = 4
            linear_num_key_heads = 1
            linear_num_value_heads = 1
            linear_key_head_dim = 32
            linear_value_head_dim = 32
            conv_kernel_size = 2
            rms_norm_eps = 1e-6

        layer = qwen35.GatedDeltaNet(
            Config(), device="cpu", dtype=torch.float32, ops=qwen35.comfy.ops.manual_cast
        )
        for parameter in layer.parameters():
            parameter.data.zero_()
        state = SimpleNamespace(
            index=1,
            conv_state=torch.zeros(1, 96, 1),
            recurrent_state=torch.zeros(1, 1, 32, 32),
            g_decay=torch.tensor([-1.0]),
            dt_bias=torch.zeros(1),
            snap_backing=None,
            conv_snap_backing=None,
        )
        conv_step = Mock(side_effect=AssertionError("disabled backend kernel was called"))
        fused_decode = Mock(side_effect=AssertionError("disabled backend kernel was called"))

        with patch.object(qwen35.comfy_kitchen.registry, "is_available", return_value=False), \\
             patch.object(qwen35.comfy_kitchen, "gated_delta_decode_is_available", return_value=True), \\
             patch.object(qwen35.comfy_kitchen, "deltanet_conv_step", conv_step), \\
             patch.object(qwen35.comfy_kitchen, "gated_delta_decode_fused", fused_decode):
            output, returned_state = layer(torch.randn(1, 1, 4), state)

        assert output.shape == (1, 1, 4)
        assert returned_state is state
        conv_step.assert_not_called()
        fused_decode.assert_not_called()
        """
    )
