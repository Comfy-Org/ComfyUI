# Code based on https://github.com/WikiChao/FreSca (MIT License)
import torch
import torch.fft as fft
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


def Fourier_filter(x, scale_low=1.0, scale_high=1.5, freq_cutoff=20):
    """
    Apply frequency-dependent scaling to an image tensor using Fourier transforms.

    Parameters:
        x:           Input tensor of shape (B, C, H, W)
        scale_low:   Scaling factor for low-frequency components (default: 1.0)
        scale_high:  Scaling factor for high-frequency components (default: 1.5)
        freq_cutoff: Number of frequency indices around center to consider as low-frequency (default: 20)

    Returns:
        x_filtered: Filtered version of x in spatial domain with frequency-specific scaling applied.
    """
    # Preserve input dtype and device
    dtype, device = x.dtype, x.device

    # Convert to float32 for FFT computations
    x = x.to(torch.float32)

    # 1) Apply FFT and shift low frequencies to center
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    # Initialize mask with high-frequency scaling factor
    mask = torch.ones(x_freq.shape, device=device) * scale_high
    m = mask
    for d in range(len(x_freq.shape) - 2):
        dim = d + 2
        cc = x_freq.shape[dim] // 2
        f_c = min(freq_cutoff, cc)
        m = m.narrow(dim, cc - f_c, f_c * 2)

    # Apply low-frequency scaling factor to center region
    m[:] = scale_low

    # 3) Apply frequency-specific scaling
    x_freq = x_freq * mask

    # 4) Convert back to spatial domain
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    # 5) Restore original dtype
    x_filtered = x_filtered.to(dtype)

    return x_filtered


class FreSca(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FreSca",
            search_aliases=["frequency guidance"],
            display_name="FreSca",
            category="experimental",
            description="Applies frequency-dependent scaling to the guidance",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("scale_low", default=1.0, min=0, max=10, step=0.01,
                               tooltip="Scaling factor for low-frequency components", advanced=True),
                io.Float.Input("scale_high", default=1.25, min=0, max=10, step=0.01,
                               tooltip="Scaling factor for high-frequency components", advanced=True),
                io.Int.Input("freq_cutoff", default=20, min=1, max=10000, step=1,
                             tooltip="Number of frequency indices around center to consider as low-frequency", advanced=True),
            ],
            outputs=[
                io.Model.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, scale_low, scale_high, freq_cutoff):
        def custom_cfg_function(args):
            conds_out = args["conds_out"]
            if len(conds_out) <= 1 or None in args["conds"][:2]:
                return conds_out
            cond = conds_out[0]
            uncond = conds_out[1]

            guidance = cond - uncond
            filtered_guidance = Fourier_filter(
                guidance,
                scale_low=scale_low,
                scale_high=scale_high,
                freq_cutoff=freq_cutoff,
            )
            filtered_cond = filtered_guidance + uncond

            return [filtered_cond, uncond] + conds_out[2:]

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(custom_cfg_function)

        return io.NodeOutput(m)


class FreScaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FreSca,
        ]


async def comfy_entrypoint() -> FreScaExtension:
    return FreScaExtension()
