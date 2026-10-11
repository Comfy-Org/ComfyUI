import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.modules.diffusionmodules.model import ResnetBlock, VideoConv3d
from comfy.ldm.hunyuan_video.vae_refiner import RMS_norm
import comfy.model_management
import comfy.model_patcher

class SRResidualCausalBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            VideoConv3d(channels, channels, kernel_size=3),
            nn.SiLU(inplace=True),
            VideoConv3d(channels, channels, kernel_size=3),
            nn.SiLU(inplace=True),
            VideoConv3d(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class SRModel3DV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        global_residual: bool = False,
    ):
        super().__init__()
        self.in_conv = VideoConv3d(in_channels, hidden_channels, kernel_size=3)
        self.blocks = nn.ModuleList([SRResidualCausalBlock3D(hidden_channels) for _ in range(num_blocks)])
        self.out_conv = VideoConv3d(hidden_channels, out_channels, kernel_size=3)
        self.global_residual = bool(global_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.in_conv(x)
        for blk in self.blocks:
            y = blk(y)
        y = self.out_conv(y)
        if self.global_residual and (y.shape == residual.shape):
            y = y + residual
        return y


class Upsampler(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.block_out_channels = block_out_channels
        self.z_channels = z_channels

        ch = block_out_channels[0]
        self.conv_in = VideoConv3d(z_channels, ch, kernel_size=3)

        self.up = nn.ModuleList()

        for i, tgt in enumerate(block_out_channels):
            stage = nn.Module()
            stage.block = nn.ModuleList([ResnetBlock(in_channels=ch if j == 0 else tgt,
                                                    out_channels=tgt,
                                                    temb_channels=0,
                                                    conv_shortcut=False,
                                                    conv_op=VideoConv3d, norm_op=RMS_norm)
                                        for j in range(num_res_blocks + 1)])
            ch = tgt
            self.up.append(stage)

        self.norm_out = RMS_norm(ch)
        self.conv_out = VideoConv3d(ch, out_channels, kernel_size=3)

    def forward(self, z):
        """
        Args:
            z: (B, C, T, H, W)
            target_shape: (H, W)
        """
        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        x = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # upsampling
        for stage in self.up:
            for blk in stage.block:
                x = blk(x)

        out = self.conv_out(F.silu(self.norm_out(x)))
        return out

UPSAMPLERS = {
    "720p": SRModel3DV2,
    "1080p": Upsampler,
}

class HunyuanVideo15SRModel():
    def __init__(self, model_type, config, fast_disk=False):
        self.load_device = comfy.model_management.vae_device()
        offload_device = comfy.model_management.vae_offload_device()
        self.dtype = comfy.model_management.vae_dtype(self.load_device)
        self.model_class = UPSAMPLERS.get(model_type)
        self.model = self.model_class(**config).eval()

        self.patcher = comfy.model_patcher.CoreModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device, fast_disk=fast_disk)

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=True, assign=self.patcher.is_dynamic())

    def get_sd(self):
        return self.model.state_dict()

    def resample_latent(self, latent):
        comfy.model_management.load_model_gpu(self.patcher)
        return self.model(latent.to(self.load_device))
