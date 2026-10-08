# Mage-VAE (https://github.com/microsoft/Mage) (MIT)
# Symmetric one-step diffusion codec: DConvEncoder (image -> 128ch latent) and
# DConvDenoiser + CoD Decoder (latent -> image). 16x downsample, latents in the
# Flux.2-VAE-anchored space (no patch packing, no BN normalization).
# Both encode and decode are single forward passes at t=0.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.modules.diffusionmodules.model import vae_attention

ops = comfy.ops.disable_weight_init


def nonlinearity(x):
    return torch.nn.functional.silu(x)


def Normalize(in_channels):
    return ops.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def modulate(x, shift, scale):
    if x.dim() == 4:
        b, c = x.shape[:2]
        return x * (1 + scale.view(b, c, 1, 1)) + shift.view(b, c, 1, 1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LayerNorm2d(ops.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class TimestepEmbedder(nn.Module):
    """DConv-style timestep MLP (max_period=10000, freq_size=256)."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            ops.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            ops.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t, dtype):
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb.to(dtype))


class BottleneckPatchEmbed(nn.Module):
    """Image patch embed concatenated with a per-patch conditioning vector."""

    def __init__(self, patch_size=16, in_chans=3, pca_dim=128, embed_dim=384, bias=True):
        super().__init__()
        self.proj1 = ops.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = ops.Conv2d(pca_dim + embed_dim, embed_dim, kernel_size=1, bias=bias)

    def forward(self, x, cond):
        return self.proj2(torch.cat([self.proj1(x), cond], dim=1))


class DiCoBlock(nn.Module):
    """DConv block with adaLN modulation."""

    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.conv1 = ops.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = ops.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size, bias=True)
        self.conv3 = ops.Conv2d(hidden_size, hidden_size, 1, bias=True)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ops.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )

        ffn = int(mlp_ratio * hidden_size)
        self.conv4 = ops.Conv2d(hidden_size, ffn, 1, bias=True)
        self.conv5 = ops.Conv2d(ffn, hidden_size, 1, bias=True)

        self.norm1 = LayerNorm2d(hidden_size, affine=False)
        self.norm2 = LayerNorm2d(hidden_size, affine=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ops.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, inp, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(inp), shift_msa, scale_msa)
        x = F.gelu(self.conv2(self.conv1(x)))
        x = x * self.ca(x)
        x = self.conv3(x)
        x = inp + gate_msa[..., None, None] * x
        x = x + gate_mlp[..., None, None] * self.conv5(
            F.gelu(self.conv4(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        )
        return x


class EncoderDiCoBlock(nn.Module):
    """DiCoBlock without adaLN, for the encoder head pathway."""

    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.conv1 = ops.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = ops.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size, bias=True)
        self.conv3 = ops.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ops.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )
        ffn = int(mlp_ratio * hidden_size)
        self.conv4 = ops.Conv2d(hidden_size, ffn, 1, bias=True)
        self.conv5 = ops.Conv2d(ffn, hidden_size, 1, bias=True)
        self.norm1 = LayerNorm2d(hidden_size)
        self.norm2 = LayerNorm2d(hidden_size)

    def forward(self, inp):
        x = self.norm1(inp)
        x = F.gelu(self.conv2(self.conv1(x)))
        x = x * self.ca(x)
        x = self.conv3(x)
        x = inp + x
        return x + self.conv5(F.gelu(self.conv4(self.norm2(x))))


class NerfEmbedder(nn.Module):
    """Patch-position embedder used by the DConv decoder x-pathway."""

    def __init__(self, in_channels, hidden_size_input, max_freqs=8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(
            ops.Linear(in_channels + max_freqs ** 2, hidden_size_input, bias=True),
        )

    def fetch_pos(self, patch_size, device, dtype):
        pos = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos, pos, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        freqs = torch.linspace(0, self.max_freqs, self.max_freqs, dtype=dtype, device=device)
        fx = freqs[None, :, None]
        fy = freqs[None, None, :]
        coeffs = (1 + fx * fy) ** -1
        dct_x = torch.cos(pos_x * fx * torch.pi)
        dct_y = torch.cos(pos_y * fy * torch.pi)
        return (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)

    def forward(self, x):
        B, P2, _ = x.shape
        ps = int(P2 ** 0.5)
        dct = self.fetch_pos(ps, x.device, x.dtype).expand(B, -1, -1)
        return self.embedder(torch.cat([x, dct], dim=-1))


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = ops.RMSNorm(hidden_size, eps=1e-6)
        self.linear = ops.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        return self.linear(self.norm(x))


class MLPResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = ops.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            ops.Linear(channels, channels, bias=True),
            nn.SiLU(),
            ops.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ops.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x, y):
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = self.in_ln(x) * (1 + scale) + shift
        return x + gate * self.mlp(h)


class SimpleMLPAdaLN(nn.Module):
    """Final small MLP that maps NerfEmbedder features to per-patch RGB."""

    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.patch_size = patch_size

        self.cond_embed = ops.Linear(z_channels, patch_size ** 2 * model_channels)
        self.input_proj = ops.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList(MLPResBlock(model_channels) for _ in range(num_res_blocks))

    def forward(self, x, c):
        x = self.input_proj(x)
        c = self.cond_embed(c).reshape(c.shape[0], self.patch_size ** 2, -1)
        for block in self.res_blocks:
            x = block(x, c)
        return x


class ResnetBlock(nn.Module):
    """GroupNorm + Conv ResBlock used by the CoD Decoder."""

    def __init__(self, *, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = ops.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = ops.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = ops.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(nonlinearity(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Patched (windowed) self-attention used by the CoD Decoder."""

    def __init__(self, in_channels, patch_size=32):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.norm = Normalize(in_channels)
        self.q = ops.Conv2d(in_channels, in_channels, 1)
        self.k = ops.Conv2d(in_channels, in_channels, 1)
        self.v = ops.Conv2d(in_channels, in_channels, 1)
        self.proj_out = ops.Conv2d(in_channels, in_channels, 1)
        # VAE attention selection: full-precision backends only (no sage/quantized attention)
        self.optimized_attention = vae_attention()

    def forward(self, x):
        h_ = self.norm(x)
        Q = self.q(h_)
        K = self.k(h_)
        V = self.v(h_)

        d = self.patch_size
        b, c, H, W = Q.shape
        pad_h = (d - H % d) % d
        pad_w = (d - W % d) % d
        if pad_h or pad_w:
            Q = F.pad(Q, (0, pad_w, 0, pad_h), mode="replicate")
            K = F.pad(K, (0, pad_w, 0, pad_h), mode="replicate")
            V = F.pad(V, (0, pad_w, 0, pad_h), mode="replicate")
        _, _, H_pad, W_pad = Q.shape
        nph, npw = H_pad // d, W_pad // d
        np_ = nph * npw

        def to_patches(t):
            return (t.reshape(b, c, nph, d, npw, d)
                    .permute(0, 2, 4, 1, 3, 5)
                    .reshape(b * np_, c, d * d))

        # [b*np, c, d*d]: attention over the d*d spatial positions of each window
        Q = to_patches(Q)
        K = to_patches(K)
        V = to_patches(V)

        h_ = self.optimized_attention(Q, K, V)
        h_ = h_.reshape(b, nph, npw, c, d, d).permute(0, 3, 1, 4, 2, 5).reshape(b, c, H_pad, W_pad)
        if pad_h or pad_w:
            h_ = h_[:, :, :H, :W]
        return x + self.proj_out(h_)


class CoDDecoder(nn.Module):
    """CoD Decoder: latent -> conditioning features for the denoiser (ds=16, light)."""

    def __init__(self, out_ch=384, z_ch=128):
        super().__init__()
        self.conv_in = ops.Conv2d(z_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.block = nn.Sequential(
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
        )
        self.norm_out = Normalize(out_ch)
        self.conv_out = ops.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.ada = nn.Identity()

    def forward(self, z):
        h = self.block(self.conv_in(z))
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return self.ada(h)


class DConvEncoder(nn.Module):
    """DConvEncoder: image -> packed (mean, logvar) latent."""

    def __init__(
        self,
        z_ch=128,
        hidden_size=384,
        num_blocks=21,
        patch_size=16,
        mlp_ratio=4.0,
        head_size=768,
        num_head_blocks=2,
        out_ch_mult=2,
    ):
        super().__init__()
        self.z_ch = z_ch
        self.patch_size = patch_size
        self.patch_cond_embed = ops.Conv2d(3, head_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.head_blocks = nn.ModuleList([
            EncoderDiCoBlock(head_size, mlp_ratio=mlp_ratio) for _ in range(num_head_blocks)
        ])
        self.proj_down = ops.Conv2d(head_size, hidden_size, kernel_size=1, bias=True)
        self.z_proj = ops.Conv2d(z_ch, hidden_size, kernel_size=1, bias=True)
        self.fuse_proj = ops.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([
            DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_blocks)
        ])
        self.norm_out = LayerNorm2d(hidden_size)
        self.proj_out = ops.Conv2d(hidden_size, z_ch * out_ch_mult, kernel_size=1, bias=True)

    def forward_pred(self, z_t, t, y):
        cond = self.patch_cond_embed(y)
        for block in self.head_blocks:
            cond = block(cond)
        cond = self.proj_down(cond)

        s = self.fuse_proj(torch.cat([cond, self.z_proj(z_t)], dim=1))
        c = self.t_embedder(t.view(-1), y.dtype)
        for block in self.blocks:
            s = block(s, c)
        return self.proj_out(self.norm_out(s))


class YEmbedder(nn.Module):
    """Holds only the CoD decoder (the original Flux2-VAE encoder side is dropped at load)."""

    def __init__(self, ch=384, z_ch=128):
        super().__init__()
        self.decoder = CoDDecoder(out_ch=ch, z_ch=z_ch)


class DConvDenoiser(nn.Module):
    """One-step DConv denoiser: latent (via cond) + zero noise -> reconstructed image."""

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        hidden_size=384,
        hidden_size_x=32,
        mlp_ratio=4.0,
        num_blocks=24,
        num_cond_blocks=21,
        bottleneck_dim=128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_cond_blocks = num_cond_blocks

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder_x = ops.Conv2d(hidden_size, hidden_size_x * patch_size ** 2, 1, 1, 0)
        self.x_embedder = NerfEmbedder(in_channels + hidden_size_x, hidden_size_x, max_freqs=8)
        self.s_embedder = BottleneckPatchEmbed(patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)
        self.blocks = nn.ModuleList([
            DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_cond_blocks)
        ])
        self.dec_net = SimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_blocks - num_cond_blocks,
            patch_size=patch_size,
        )
        self.final_layer = NerfFinalLayer(hidden_size_x, in_channels)
        self.y_embedder = YEmbedder(ch=hidden_size, z_ch=bottleneck_dim)

    def forward(self, x, t, cond):
        b, _, h, w = x.shape
        c = self.t_embedder(t.view(-1), x.dtype)

        s = self.s_embedder(x, cond)
        for block in self.blocks:
            s = block(s, c)

        length = s.shape[-2] * s.shape[-1]
        s = s.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)

        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = torch.cat([x, self.y_embedder_x(cond).flatten(2)], dim=1)
        x = x.reshape(b, -1, self.patch_size ** 2, length).permute(0, 3, 2, 1).flatten(0, 1)
        x = self.x_embedder(x)

        x = self.dec_net(x, s)
        x = self.final_layer(x)
        x = x.transpose(1, 2).reshape(b, length, -1)
        return torch.nn.functional.fold(
            x.transpose(1, 2).contiguous(), (h, w),
            kernel_size=self.patch_size, stride=self.patch_size,
        )


class MageVAE(nn.Module):
    """
    Encode: DConvEncoder (one-step at t=0) -> posterior mean [B, 128, H/16, W/16]
    Decode: DConvDenoiser + CoD Decoder    -> image [B, 3, H, W] in [-1, 1]
    """

    latent_channels = 128
    downsample_factor = 16

    def __init__(self):
        super().__init__()
        self.dconv_encoder = DConvEncoder()
        self.decoder_model = DConvDenoiser()

    def encode(self, x):
        B, _, H, W = x.shape
        ps = self.dconv_encoder.patch_size
        z_t = torch.zeros(B, self.dconv_encoder.z_ch, H // ps, W // ps, device=x.device, dtype=x.dtype)
        t = torch.zeros(B, device=x.device, dtype=x.dtype)
        out = self.dconv_encoder.forward_pred(z_t, t, x)
        return out[:, : self.latent_channels]  # posterior mean (sample_posterior=False)

    def decode(self, z):
        cond = self.decoder_model.y_embedder.decoder(z)
        B = z.shape[0]
        H = z.shape[2] * self.downsample_factor
        W = z.shape[3] * self.downsample_factor
        noise = torch.zeros(B, 3, H, W, device=z.device, dtype=z.dtype)
        t = torch.zeros(B, device=z.device, dtype=z.dtype)
        return self.decoder_model.forward(noise, t, cond)
