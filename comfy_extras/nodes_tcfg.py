# TCFG: Tangential Damping Classifier-free Guidance - (arXiv: https://arxiv.org/abs/2503.18137)

from typing_extensions import override
import torch

from comfy_api.v3 import ComfyExtension, io


def score_tangential_damping(cond_score: torch.Tensor, uncond_score: torch.Tensor) -> torch.Tensor:
    """Drop tangential components from uncond score to align with cond score."""
    # (B, 1, ...)
    batch_num = cond_score.shape[0]
    cond_score_flat = cond_score.reshape(batch_num, 1, -1).float()
    uncond_score_flat = uncond_score.reshape(batch_num, 1, -1).float()

    # Score matrix A (B, 2, ...)
    score_matrix = torch.cat((uncond_score_flat, cond_score_flat), dim=1)
    try:
        _, _, Vh = torch.linalg.svd(score_matrix, full_matrices=False)
    except RuntimeError:
        # Fallback to CPU
        _, _, Vh = torch.linalg.svd(score_matrix.cpu(), full_matrices=False)

    # Drop the tangential components
    v1 = Vh[:, 0:1, :].to(uncond_score_flat.device)  # (B, 1, ...)
    uncond_score_td = (uncond_score_flat @ v1.transpose(-2, -1)) * v1
    return uncond_score_td.reshape_as(uncond_score).to(uncond_score.dtype)


class TCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TCFG",
            display_name="Tangential Damping CFG",
            category="advanced/guidance",
            description="TCFG – Tangential Damping CFG (2503.18137)\n\nRefine the uncond (negative) to align with the cond (positive) for improving quality.",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[
                io.Model.Output(display_name="patched_model"),
            ],
        )

    @classmethod
    def execute(cls, model):
        m = model.clone()

        def tangential_damping_cfg(args):
            #  Assume [cond, uncond, ...]
            x = args["input"]
            conds_out = args["conds_out"]
            if len(conds_out) <= 1 or None in args["conds"][:2]:
                # Skip when either cond or uncond is None
                return conds_out
            cond_pred = conds_out[0]
            uncond_pred = conds_out[1]
            uncond_td = score_tangential_damping(x - cond_pred, x - uncond_pred)
            uncond_pred_td = x - uncond_td
            return [cond_pred, uncond_pred_td] + conds_out[2:]

        m.set_model_sampler_pre_cfg_function(tangential_damping_cfg)
        return io.NodeOutput(m)


class TcfgExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TCFG,
        ]


async def comfy_entrypoint() -> TcfgExtension:
    return TcfgExtension()
