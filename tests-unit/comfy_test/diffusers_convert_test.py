import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.diffusers_convert import convert_text_enc_state_dict_v20  # noqa: E402
from comfy.model_patcher import LazyCastingParam  # noqa: E402


class PatchedModel:
    load_device = torch.device("cpu")

    def __init__(self, weights):
        self.weights = weights

    def patch_weight_to_device(self, key, device_to=None, return_weight=False):
        return self.weights[key] + 1


def test_text_encoder_conversion_uses_patched_lazy_weights():
    prefix = "cond_stage_model.model.transformer.text_model.encoder.layers.0.self_attn."
    keys = [prefix + f"{p}_proj.{t}" for p in "qkv" for t in ("weight", "bias")]
    keys.append("cond_stage_model.model.transformer.text_projection.weight")
    weights = {k: torch.randn(4, 4) if k.endswith("weight") else torch.randn(4) for k in keys}
    model = PatchedModel(weights)
    sd = {k: LazyCastingParam(model, k, v) for k, v in weights.items()}

    out = convert_text_enc_state_dict_v20(sd)

    attn = "cond_stage_model.model.transformer.resblocks.0.attn."
    expected_weight = torch.cat([weights[prefix + f"{p}_proj.weight"] for p in "qkv"]) + 1
    expected_bias = torch.cat([weights[prefix + f"{p}_proj.bias"] for p in "qkv"]) + 1
    assert torch.equal(out[attn + "in_proj_weight"], expected_weight)
    assert torch.equal(out[attn + "in_proj_bias"], expected_bias)
    expected_proj = (weights["cond_stage_model.model.transformer.text_projection.weight"] + 1).transpose(0, 1)
    assert torch.equal(out["cond_stage_model.model.text_projection"], expected_proj)
