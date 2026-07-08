#code originally taken from: https://github.com/ChenyangSi/FreeU (under MIT License)

import torch
import logging
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO

def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


class FreeU(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FreeU",
            category="model/patch/unet",
            inputs=[
                IO.Model.Input("model"),
                IO.Float.Input("b1", default=1.1, min=0.0, max=10.0, step=0.01, advanced=True),
                IO.Float.Input("b2", default=1.2, min=0.0, max=10.0, step=0.01, advanced=True),
                IO.Float.Input("s1", default=0.9, min=0.0, max=10.0, step=0.01, advanced=True),
                IO.Float.Input("s2", default=0.2, min=0.0, max=10.0, step=0.01, advanced=True),
            ],
            outputs=[
                IO.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, b1, b2, s1, s2) -> IO.NodeOutput:
        model_channels = model.model.model_config.unet_config["model_channels"]
        scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
        on_cpu_devices = {}

        def output_block_patch(h, hsp, transformer_options):
            scale = scale_dict.get(int(h.shape[1]), None)
            if scale is not None:
                h[:,:h.shape[1] // 2] = h[:,:h.shape[1] // 2] * scale[0]
                if hsp.device not in on_cpu_devices:
                    try:
                        hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                    except:
                        logging.warning("Device {} does not support the torch.fft functions used in the FreeU node, switching to CPU.".format(hsp.device))
                        on_cpu_devices[hsp.device] = True
                        hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
                else:
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

            return h, hsp

        m = model.clone()
        m.set_model_output_block_patch(output_block_patch)
        return IO.NodeOutput(m)

    patch = execute  # TODO: remove


class FreeU_V2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FreeU_V2",
            category="model/patch/unet",
            inputs=[
                IO.Model.Input("model"),
                IO.Float.Input("b1", default=1.3, min=0.0, max=10.0, step=0.01, advanced=True),
                IO.Float.Input("b2", default=1.4, min=0.0, max=10.0, step=0.01, advanced=True),
                IO.Float.Input("s1", default=0.9, min=0.0, max=10.0, step=0.01, advanced=True),
                IO.Float.Input("s2", default=0.2, min=0.0, max=10.0, step=0.01, advanced=True),
            ],
            outputs=[
                IO.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, b1, b2, s1, s2) -> IO.NodeOutput:
        model_channels = model.model.model_config.unet_config["model_channels"]
        scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
        on_cpu_devices = {}

        def output_block_patch(h, hsp, transformer_options):
            scale = scale_dict.get(int(h.shape[1]), None)
            if scale is not None:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:,:h.shape[1] // 2] = h[:,:h.shape[1] // 2] * ((scale[0] - 1 ) * hidden_mean + 1)

                if hsp.device not in on_cpu_devices:
                    try:
                        hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                    except:
                        logging.warning("Device {} does not support the torch.fft functions used in the FreeU node, switching to CPU.".format(hsp.device))
                        on_cpu_devices[hsp.device] = True
                        hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
                else:
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

            return h, hsp

        m = model.clone()
        m.set_model_output_block_patch(output_block_patch)
        return IO.NodeOutput(m)

    patch = execute  # TODO: remove


class FreelunchExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            FreeU,
            FreeU_V2,
        ]


async def comfy_entrypoint() -> FreelunchExtension:
    return FreelunchExtension()
