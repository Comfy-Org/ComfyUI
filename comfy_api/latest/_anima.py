from __future__ import annotations

import math
import os
from typing import Any


class InProcessAnima:
    async def apply_lllite(
        self,
        model: Any,
        weights: Any,
        image: Any,
        *,
        strength: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        preserve_wrapper: bool = True,
    ) -> Any:
        import torch

        import comfy.ldm.anima.lllite
        import comfy.model_base
        import comfy.model_management
        import comfy.model_patcher
        import comfy.ops
        import comfy.utils
        import folder_paths

        from . import _sdk

        if not isinstance(model, _sdk.ModelRef) or model.kind != "MODEL":
            raise TypeError("Anima LLLite needs a MODEL ref")
        if not isinstance(weights, _sdk.AssetRef) or weights.kind != "ASSET":
            raise TypeError("Anima LLLite weights must be an ASSET ref")
        if not isinstance(image, _sdk.ImageRef) or image.kind != "IMAGE":
            raise TypeError("Anima LLLite image must be an IMAGE ref")
        checked = {}
        for name, value, minimum, maximum in (
            ("strength", strength, -10.0, 10.0),
            ("start_percent", start_percent, 0.0, 1.0),
            ("end_percent", end_percent, 0.0, 1.0),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            number = float(value)
            if not math.isfinite(number) or not minimum <= number <= maximum:
                raise ValueError(
                    f"{name} must be finite and in [{minimum}, {maximum}]")
            checked[name] = number
        if not isinstance(preserve_wrapper, bool):
            raise TypeError("preserve_wrapper must be a boolean")

        runtime = _sdk.current_runtime()
        source_model = await runtime.refs.resolve(model)
        source_image = await runtime.refs.resolve(image)
        path = await runtime.refs.resolve(weights)
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("Anima LLLite ASSET ref does not contain a path")
        path = _sdk._InProcessAssets._confined_resolved_path(
            path, folder_paths.get_folder_paths("controlnet"), "controlnet")
        if os.path.splitext(path)[1].lower() not in {".safetensors", ".sft"}:
            raise ValueError("Anima LLLite weights must use SafeTensors")
        size = os.path.getsize(path)
        if not 0 < size <= 8 * 1024**3:
            raise ValueError("Anima LLLite weights exceed the 8 GiB limit")
        if not isinstance(source_image, torch.Tensor) or (
            source_image.ndim != 4
            or not 1 <= source_image.shape[0] <= 64
            or source_image.shape[-1] < 3
            or source_image.shape[1] < 1
            or source_image.shape[2] < 1
            or source_image.numel() > 268_435_456
        ):
            raise ValueError("Anima LLLite needs a bounded BHWC image batch")
        if not isinstance(
            getattr(source_model, "model", None), comfy.model_base.Anima,
        ):
            raise ValueError("Anima LLLite requires an Anima model")

        state, metadata = comfy.utils.load_torch_file(
            path, safe_load=True, return_metadata=True)
        if (
            not isinstance(state, dict)
            or not state
            or len(state) > 100_000
            or any(
                not isinstance(key, str) or not isinstance(value, torch.Tensor)
                for key, value in state.items()
            )
        ):
            raise ValueError(
                "Anima LLLite weights must be a bounded tensor-only state dict")
        dtype = comfy.utils.weight_dtype(state)
        lllite = comfy.ldm.anima.lllite.AnimaLLLite(
            state,
            metadata,
            device=comfy.model_management.unet_offload_device(),
            dtype=dtype,
            operations=comfy.ops.manual_cast,
        )
        if lllite.cond_in_channels != 3:
            raise ValueError(
                "this Anima LLLite integration supports RGB control weights only")
        model_patch = comfy.model_patcher.CoreModelPatcher(
            lllite,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )
        lllite.load_state_dict(state, assign=model_patch.is_dynamic())

        sampling = source_model.get_model_object("model_sampling")
        sigma_start = float(sampling.percent_to_sigma(checked["start_percent"]))
        sigma_end = float(sampling.percent_to_sigma(checked["end_percent"]))
        patch = comfy.ldm.anima.lllite.AnimaLLLitePatch(
            model_patch,
            source_image[..., :3],
            None,
            checked["strength"],
            sigma_start,
            sigma_end,
        )
        result = source_model.clone()
        if not preserve_wrapper:
            result.model_options.pop("model_function_wrapper", None)
        result.set_model_post_input_patch(patch)
        result.set_model_attn1_patch(
            comfy.ldm.anima.lllite.AnimaLLLiteAttentionPatch(
                patch,
                {
                    "q": "self_attn_q_proj",
                    "k": "self_attn_k_proj",
                    "v": "self_attn_v_proj",
                },
            ))
        result.set_model_attn2_patch(
            comfy.ldm.anima.lllite.AnimaLLLiteAttentionPatch(
                patch, {"q": "cross_attn_q_proj"}))
        result.set_model_patch(
            comfy.ldm.anima.lllite.AnimaLLLiteMLPPatch(patch), "mlp_patch")
        return _sdk.ModelRef._wrap(
            await runtime.refs.create("MODEL", result))
