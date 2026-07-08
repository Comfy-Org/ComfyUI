import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import Upsample


class SUPIRPatch:
    """
    Holds GLVControl (control encoder) + project_modules (ZeroSFT/ZeroCrossAttn adapters).
    Runs GLVControl lazily on first patch invocation per step, applies adapters through
    middle_block_after_patch, output_block_merge_patch, and forward_timestep_embed_patch.
    """
    SIGMA_MAX = 14.6146

    def __init__(self, model_patch, project_modules, hint_latent, strength_start, strength_end):
        self.model_patch = model_patch           # CoreModelPatcher wrapping GLVControl
        self.project_modules = project_modules   # nn.ModuleList of ZeroSFT/ZeroCrossAttn
        self.hint_latent = hint_latent           # encoded LQ image latent
        self.strength_start = strength_start
        self.strength_end = strength_end
        self.cached_features = None
        self.adapter_idx = 0
        self.control_idx = 0
        self.current_control_idx = 0
        self.active = True

    def _ensure_features(self, kwargs):
        """Run GLVControl on first call per step, cache results."""
        if self.cached_features is not None:
            return
        x = kwargs["x"]
        b = x.shape[0]
        hint = self.hint_latent.to(device=x.device, dtype=x.dtype)
        if hint.shape[0] != b:
            hint = hint.expand(b, -1, -1, -1) if hint.shape[0] == 1 else hint.repeat((b + hint.shape[0] - 1) // hint.shape[0], 1, 1, 1)[:b]
        self.cached_features = self.model_patch.model.control_model(
            hint, kwargs["timesteps"], x,
            kwargs["context"], kwargs["y"]
        )
        self.adapter_idx = len(self.project_modules) - 1
        self.control_idx = len(self.cached_features) - 1

    def _get_control_scale(self, kwargs):
        if self.strength_start == self.strength_end:
            return self.strength_end
        sigma = kwargs["transformer_options"].get("sigmas")
        if sigma is None:
            return self.strength_end
        s = sigma[0].item() if sigma.dim() > 0 else sigma.item()
        t = min(s / self.SIGMA_MAX, 1.0)
        return t * (self.strength_start - self.strength_end) + self.strength_end

    def middle_after(self, kwargs):
        """middle_block_after_patch: run GLVControl lazily, apply last adapter after middle block."""
        self.cached_features = None  # reset from previous step
        self.current_scale = self._get_control_scale(kwargs)
        self.active = self.current_scale > 0
        if not self.active:
            return {"h": kwargs["h"]}
        self._ensure_features(kwargs)
        h = kwargs["h"]
        h = self.project_modules[self.adapter_idx](
            self.cached_features[self.control_idx], h, control_scale=self.current_scale
        )
        self.adapter_idx -= 1
        self.control_idx -= 1
        return {"h": h}

    def output_block(self, h, hsp, transformer_options):
        """output_block_patch: ZeroSFT adapter fusion replaces cat([h, hsp]). Returns (h, None) to skip cat."""
        if not self.active:
            return h, hsp
        self.current_control_idx = self.control_idx
        h = self.project_modules[self.adapter_idx](
            self.cached_features[self.control_idx], hsp, h, control_scale=self.current_scale
        )
        self.adapter_idx -= 1
        self.control_idx -= 1
        return h, None

    def pre_upsample(self, layer, x, emb, context, transformer_options, output_shape, *args, **kw):
        """forward_timestep_embed_patch for Upsample: extra cross-attn adapter before upsample."""
        block_type, _ = transformer_options["block"]
        if block_type == "output" and self.active and self.cached_features is not None:
            x = self.project_modules[self.adapter_idx](
                self.cached_features[self.current_control_idx], x, control_scale=self.current_scale
            )
            self.adapter_idx -= 1
        return layer(x, output_shape=output_shape)

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.cached_features = None
            if self.hint_latent is not None:
                self.hint_latent = self.hint_latent.to(device_or_dtype)
        return self

    def models(self):
        return [self.model_patch]

    def register(self, model_patcher):
        """Register all patches on a cloned model patcher."""
        model_patcher.set_model_patch(self.middle_after, "middle_block_after_patch")
        model_patcher.set_model_output_block_patch(self.output_block)
        model_patcher.set_model_patch((Upsample, self.pre_upsample), "forward_timestep_embed_patch")
