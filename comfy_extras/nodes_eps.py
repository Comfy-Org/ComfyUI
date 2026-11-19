import torch
from typing_extensions import override

from comfy.k_diffusion.sampling import sigma_to_half_log_snr
from comfy_api.latest import ComfyExtension, io


class EpsilonScaling(io.ComfyNode):
    """
    Implements the Epsilon Scaling method from 'Elucidating the Exposure Bias in Diffusion Models'
    (https://arxiv.org/abs/2308.15321v6).

    This method mitigates exposure bias by scaling the predicted noise during sampling,
    which can significantly improve sample quality. This implementation uses the "uniform schedule"
    recommended by the paper for its practicality and effectiveness.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Epsilon Scaling",
            category="model/patch/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "scaling_factor",
                    default=1.005,
                    min=0.5,
                    max=1.5,
                    step=0.001,
                    display_mode=io.NumberDisplay.number,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, scaling_factor) -> io.NodeOutput:
        # Prevent division by zero, though the UI's min value should prevent this.
        if scaling_factor == 0:
            scaling_factor = 1e-9

        def epsilon_scaling_function(args):
            """
            This function is applied after the CFG guidance has been calculated.
            It recalculates the denoised latent by scaling the predicted noise.
            """
            denoised = args["denoised"]
            x = args["input"]

            noise_pred = x - denoised

            scaled_noise_pred = noise_pred / scaling_factor

            new_denoised = x - scaled_noise_pred

            return new_denoised

        # Clone the model patcher to avoid modifying the original model in place
        model_clone = model.clone()

        model_clone.set_model_sampler_post_cfg_function(epsilon_scaling_function)

        return io.NodeOutput(model_clone)


def compute_tsr_rescaling_factor(
    snr: torch.Tensor, tsr_k: float, tsr_variance: float
) -> torch.Tensor:
    """Compute the rescaling score ratio in Temporal Score Rescaling.

    See equation (6) in https://arxiv.org/pdf/2510.01184v1.
    """
    posinf_mask = torch.isposinf(snr)
    rescaling_factor = (snr * tsr_variance + 1) / (snr * tsr_variance / tsr_k + 1)
    return torch.where(posinf_mask, tsr_k, rescaling_factor) # when snr → inf, r = tsr_k


class TemporalScoreRescaling(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TemporalScoreRescaling",
            display_name="TSR - Temporal Score Rescaling",
            category="model/patch/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "tsr_k",
                    tooltip=(
                        "Controls the rescaling strength.\n"
                        "Lower k produces more detailed results; higher k produces smoother results in image generation. Setting k = 1 disables rescaling."
                    ),
                    default=0.95,
                    min=0.01,
                    max=100.0,
                    step=0.001,
                    display_mode=io.NumberDisplay.number,
                    advanced=True,
                ),
                io.Float.Input(
                    "tsr_sigma",
                    tooltip=(
                        "Controls how early rescaling takes effect.\n"
                        "Larger values take effect earlier."
                    ),
                    default=1.0,
                    min=0.01,
                    max=100.0,
                    step=0.001,
                    display_mode=io.NumberDisplay.number,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="patched_model",
                ),
            ],
            description=(
                "[Post-CFG Function]\n"
                "TSR - Temporal Score Rescaling (2510.01184)\n\n"
                "Rescaling the model's score or noise to steer the sampling diversity.\n"
            ),
        )

    @classmethod
    def execute(cls, model, tsr_k, tsr_sigma) -> io.NodeOutput:
        tsr_variance = tsr_sigma**2

        def temporal_score_rescaling(args):
            denoised = args["denoised"]
            x = args["input"]
            sigma = args["sigma"]
            curr_model = args["model"]

            # No rescaling (r = 1) or no noise
            if tsr_k == 1 or sigma == 0:
                return denoised

            model_sampling = curr_model.current_patcher.get_model_object("model_sampling")
            half_log_snr = sigma_to_half_log_snr(sigma, model_sampling)
            snr = (2 * half_log_snr).exp()

            # No rescaling needed (r = 1)
            if snr == 0:
                return denoised

            rescaling_r = compute_tsr_rescaling_factor(snr, tsr_k, tsr_variance)

            # Derived from scaled_denoised = (x - r * sigma * noise) / alpha
            alpha = sigma * half_log_snr.exp()
            return torch.lerp(x / alpha, denoised, rescaling_r)

        m = model.clone()
        m.set_model_sampler_post_cfg_function(temporal_score_rescaling)
        return io.NodeOutput(m)


class EpsilonScalingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EpsilonScaling,
            TemporalScoreRescaling,
        ]


async def comfy_entrypoint() -> EpsilonScalingExtension:
    return EpsilonScalingExtension()
