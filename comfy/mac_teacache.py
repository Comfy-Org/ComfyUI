"""TeaCache-style timestep residual reuse for Flux — automatic on MPS.

Existing Flux workflows get fewer DiT block evaluations when consecutive steps
are similar. No new nodes required. Opt out with --disable-mac-teacache.
"""

from __future__ import annotations

import logging
import os
from types import MethodType

import torch
from torch import Tensor

_logger = logging.getLogger("comfy.mac_teacache")

FLUX_COEFFICIENTS = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]

_INSTALLED = False
_ORIGINAL_FORWARD_ORIG = None


def poly1d(coefficients, x: Tensor) -> Tensor:
    result = torch.zeros_like(x)
    n = len(coefficients)
    for i, coeff in enumerate(coefficients):
        result = result + coeff * (x ** (n - 1 - i))
    return result


def _rel_l1(a: Tensor, b: Tensor) -> Tensor:
    denom = b.abs().mean().clamp(min=1e-6)
    return (a - b).abs().mean() / denom


def _default_state(rel_l1_thresh: float, start_percent: float, end_percent: float, coefficients):
    return {
        "accumulated_rel_l1_distance": 0.0,
        "previous_modulated_input": None,
        "previous_residual": None,
        "cnt": 0,
        "skipped": 0,
        "computed": 0,
        "rel_l1_thresh": float(rel_l1_thresh),
        "start_percent": float(start_percent),
        "end_percent": float(end_percent),
        "coefficients": list(coefficients),
    }


def _make_teacache_forward(original_forward, *, rel_l1_thresh, start_percent, end_percent, coefficients):
    from comfy.ldm.flux.layers import apply_mod, timestep_embedding

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control=None,
        timestep_zero_index=None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        state = self.__dict__.get("_comfy_mac_teacache_state")
        if state is None:
            state = _default_state(rel_l1_thresh, start_percent, end_percent, coefficients)
            self.__dict__["_comfy_mac_teacache_state"] = state

        enable = True
        sigmas = transformer_options.get("sample_sigmas", None)
        step_index = transformer_options.get("current_step", None)
        if sigmas is not None and step_index is not None and len(sigmas) > 1:
            percent = float(step_index) / max(len(sigmas) - 1, 1)
            if percent < state["start_percent"] or percent > state["end_percent"]:
                enable = False

        if not enable or timestep_zero_index is not None:
            return original_forward(
                self, img, img_ids, txt, txt_ids, timesteps, y,
                guidance=guidance, control=control,
                timestep_zero_index=timestep_zero_index,
                transformer_options=transformer_options, attn_mask=attn_mask,
            )

        transformer_options = transformer_options.copy()
        patches = transformer_options.get("patches", {})
        patches_replace = transformer_options.get("patches_replace", {})

        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if self.vector_in is not None:
            if y is None:
                y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
            vec = vec + self.vector_in(y[:, : self.params.vec_in_dim])

        if self.txt_norm is not None:
            txt = self.txt_norm(txt)
        txt = self.txt_in(txt)

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
                img = out["img"]
                txt = out["txt"]
                img_ids = out["img_ids"]
                txt_ids = out["txt_ids"]

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        vec_orig = vec
        txt_vec = vec
        extra_kwargs = {}
        if self.params.global_modulation:
            vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(txt_vec))

        block0 = self.double_blocks[0]
        if self.params.global_modulation:
            (img_mod1, _), _ = vec
        else:
            img_mod1, _ = block0.img_mod(vec)
        modulated_inp = apply_mod(block0.img_norm1(img), (1 + img_mod1.scale), img_mod1.shift)

        should_calc = True
        if state["previous_modulated_input"] is None or state["previous_residual"] is None:
            should_calc = True
            state["accumulated_rel_l1_distance"] = 0.0
        else:
            try:
                prev = state["previous_modulated_input"]
                if prev.shape != modulated_inp.shape:
                    should_calc = True
                    state["accumulated_rel_l1_distance"] = 0.0
                else:
                    delta = poly1d(state["coefficients"], _rel_l1(modulated_inp, prev)).abs().item()
                    state["accumulated_rel_l1_distance"] += delta
                    if state["accumulated_rel_l1_distance"] < state["rel_l1_thresh"]:
                        should_calc = False
                    else:
                        should_calc = True
                        state["accumulated_rel_l1_distance"] = 0.0
            except Exception:
                should_calc = True
                state["accumulated_rel_l1_distance"] = 0.0

        state["previous_modulated_input"] = modulated_inp.detach()
        state["cnt"] += 1

        if not should_calc:
            state["skipped"] += 1
            img = img + state["previous_residual"].to(device=img.device, dtype=img.dtype)
            return self.final_layer(img, vec_orig)

        state["computed"] += 1
        ori_img = img

        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.double_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args, _block=block):
                    out = {}
                    out["img"], out["txt"] = _block(
                        img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"],
                        attn_mask=args.get("attn_mask"),
                        transformer_options=args.get("transformer_options"),
                        **extra_kwargs,
                    )
                    return out

                out = blocks_replace[("double_block", i)](
                    {"img": img, "txt": txt, "vec": vec, "pe": pe, "attn_mask": attn_mask, "transformer_options": transformer_options},
                    {"original_block": block_wrap},
                )
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(
                    img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask,
                    transformer_options=transformer_options, **extra_kwargs,
                )

            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:, : add.shape[1]] += add

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        img = torch.cat((txt, img), 1)

        if self.params.global_modulation:
            vec, _ = self.single_stream_modulation(vec_orig)

        transformer_options["total_blocks"] = len(self.single_blocks)
        transformer_options["block_type"] = "single"
        transformer_options["img_slice"] = [txt.shape[1], img.shape[1]]
        for i, block in enumerate(self.single_blocks):
            transformer_options["block_index"] = i
            if ("single_block", i) in blocks_replace:
                def block_wrap(args, _block=block):
                    out = {}
                    out["img"] = _block(
                        args["img"], vec=args["vec"], pe=args["pe"],
                        attn_mask=args.get("attn_mask"),
                        transformer_options=args.get("transformer_options"),
                        **extra_kwargs,
                    )
                    return out

                out = blocks_replace[("single_block", i)](
                    {"img": img, "vec": vec, "pe": pe, "attn_mask": attn_mask, "transformer_options": transformer_options},
                    {"original_block": block_wrap},
                )
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options, **extra_kwargs)

            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] : txt.shape[1] + add.shape[1], ...] += add

        img = img[:, txt.shape[1] :, ...]
        state["previous_residual"] = (img - ori_img).detach()
        return self.final_layer(img, vec_orig)

    return forward_orig


def install_auto_teacache(
    *,
    rel_l1_thresh: float = 0.4,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    coefficients=None,
) -> bool:
    """Patch Flux.forward_orig once so every Flux checkpoint on this process gets TeaCache."""
    global _INSTALLED, _ORIGINAL_FORWARD_ORIG
    if _INSTALLED:
        return True
    if coefficients is None:
        coefficients = FLUX_COEFFICIENTS
    try:
        from comfy.ldm.flux.model import Flux
    except Exception as e:
        _logger.info("TeaCache: Flux import failed (%s)", e)
        return False

    _ORIGINAL_FORWARD_ORIG = Flux.forward_orig
    Flux.forward_orig = _make_teacache_forward(
        _ORIGINAL_FORWARD_ORIG,
        rel_l1_thresh=rel_l1_thresh,
        start_percent=start_percent,
        end_percent=end_percent,
        coefficients=coefficients,
    )
    _INSTALLED = True
    _logger.info("Flux TeaCache auto-enabled (rel_l1_thresh=%.3f)", rel_l1_thresh)
    return True


def _mac_teacache_disabled() -> bool:
    if os.environ.get("COMFY_DISABLE_MAC_TEACACHE", "").strip() in ("1", "true", "yes"):
        return True
    try:
        from comfy.cli_args import args
        return bool(getattr(args, "disable_mac_teacache", False))
    except Exception:
        return False


def maybe_enable_mac_teacache() -> bool:
    """Enable Flux TeaCache automatically on Apple Silicon unless opted out."""
    if _mac_teacache_disabled():
        _logger.info("Flux TeaCache disabled by flag/env")
        return False
    try:
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return False
    except Exception:
        return False
    return install_auto_teacache()


def patch_flux_teacache(
    diffusion_model,
    *,
    rel_l1_thresh: float = 0.4,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    coefficients=None,
):
    """Instance-level patch (tests / rare overrides). Prefer install_auto_teacache()."""
    if coefficients is None:
        coefficients = FLUX_COEFFICIENTS
    if not hasattr(diffusion_model, "forward_orig"):
        raise TypeError("TeaCache Flux patch requires diffusion_model.forward_orig")
    original = diffusion_model.forward_orig
    # Bound method → underlying function for the wrapper's original_forward(self, ...)
    original_fn = original.__func__ if hasattr(original, "__func__") else original
    wrapped = _make_teacache_forward(
        original_fn,
        rel_l1_thresh=rel_l1_thresh,
        start_percent=start_percent,
        end_percent=end_percent,
        coefficients=coefficients,
    )
    diffusion_model.forward_orig = MethodType(wrapped, diffusion_model)
    diffusion_model._comfy_mac_teacache_state = _default_state(
        rel_l1_thresh, start_percent, end_percent, coefficients
    )
    return diffusion_model._comfy_mac_teacache_state


def teacache_stats(diffusion_model) -> dict:
    state = getattr(diffusion_model, "_comfy_mac_teacache_state", None)
    if state is None:
        return {}
    total = state["skipped"] + state["computed"]
    return {
        "skipped": state["skipped"],
        "computed": state["computed"],
        "skip_ratio": (state["skipped"] / total) if total else 0.0,
    }
