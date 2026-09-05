import torch
import torch.nn.functional as F
import torch.nn as nn
import comfy.ops
import comfy.model_management
import numpy as np
import math

ops = comfy.ops.disable_weight_init

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# ---------------------------------------------------------------------------
# Anti-aliased resampling helpers (kaiser-sinc filters) for BigVGAN v2
# Adopted from https://github.com/NVIDIA/BigVGAN
# ---------------------------------------------------------------------------


def _sinc(x: torch.Tensor):
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / math.pi / x,
    )


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)
    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride=1,
        padding=True,
        padding_mode="replicate",
        kernel_size=12,
    ):
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, comfy.model_management.cast_to(self.filter.expand(C, -1, -1), dtype=x.dtype, device=x.device), stride=self.stride, groups=C)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, persistent=True, window_type="kaiser"):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        if window_type == "hann":
            # Hann-windowed sinc filter — identical to torchaudio.functional.resample
            # with its default parameters (rolloff=0.99, lowpass_filter_width=6).
            # Uses replicate boundary padding, matching the reference resampler exactly.
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            t = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            t_clamped = t.clamp(-lowpass_filter_width, lowpass_filter_width)
            window = torch.cos(t_clamped * math.pi / lowpass_filter_width / 2) ** 2
            filter = (torch.sinc(t) * window * rolloff / ratio).view(1, 1, -1)
        else:
            # Kaiser-windowed sinc filter (BigVGAN default).
            self.kernel_size = (
                int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            )
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = (
                self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            )
            filter = kaiser_sinc_filter1d(
                cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
            )

        self.register_buffer("filter", filter, persistent=persistent)

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, comfy.model_management.cast_to(self.filter.expand(C, -1, -1), dtype=x.dtype, device=x.device), stride=self.stride, groups=C
        )
        x = x[..., self.pad_left : -self.pad_right]
        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio=2,
        down_ratio=2,
        up_kernel_size=12,
        down_kernel_size=12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# BigVGAN v2 activations (Snake / SnakeBeta)
# ---------------------------------------------------------------------------


class Snake(nn.Module):
    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True
    ):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(
            torch.zeros(in_features)
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.alpha.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x):
        a = comfy.model_management.cast_to(self.alpha.unsqueeze(0).unsqueeze(-1), dtype=x.dtype, device=x.device)
        if self.alpha_logscale:
            a = torch.exp(a)
        return x + (1.0 / (a + self.eps)) * torch.sin(x * a).pow(2)


class SnakeBeta(nn.Module):
    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True
    ):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(
            torch.zeros(in_features)
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.alpha.requires_grad = alpha_trainable
        self.beta = nn.Parameter(
            torch.zeros(in_features)
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.beta.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x):
        a = comfy.model_management.cast_to(self.alpha.unsqueeze(0).unsqueeze(-1), dtype=x.dtype, device=x.device)
        b = comfy.model_management.cast_to(self.beta.unsqueeze(0).unsqueeze(-1), dtype=x.dtype, device=x.device)
        if self.alpha_logscale:
            a = torch.exp(a)
            b = torch.exp(b)
        return x + (1.0 / (b + self.eps)) * torch.sin(x * a).pow(2)


# ---------------------------------------------------------------------------
# BigVGAN v2 AMPBlock (Anti-aliased Multi-Periodicity)
# ---------------------------------------------------------------------------


class AMPBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation="snake"):
        super().__init__()
        act_cls = SnakeBeta if activation == "snakebeta" else Snake
        self.convs1 = nn.ModuleList(
            [
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
            ]
        )

        self.acts1 = nn.ModuleList(
            [Activation1d(act_cls(channels)) for _ in range(len(self.convs1))]
        )
        self.acts2 = nn.ModuleList(
            [Activation1d(act_cls(channels)) for _ in range(len(self.convs2))]
        )

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = x + xt
        return x


# ---------------------------------------------------------------------------
# HiFi-GAN residual blocks
# ---------------------------------------------------------------------------


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                ),
            ]
        )

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class Vocoder(torch.nn.Module):
    """
    Vocoder model for synthesizing audio from spectrograms, based on: https://github.com/jik876/hifi-gan.

    Supports both HiFi-GAN (resblock "1"/"2") and BigVGAN v2 (resblock "AMP1").
    """

    def __init__(self, config=None):
        super(Vocoder, self).__init__()

        if config is None:
            config = self.get_default_config()

        resblock_kernel_sizes = config.get("resblock_kernel_sizes", [3, 7, 11])
        upsample_rates = config.get("upsample_rates", [5, 4, 2, 2, 2])
        upsample_kernel_sizes = config.get("upsample_kernel_sizes", [16, 16, 8, 4, 4])
        resblock_dilation_sizes = config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        upsample_initial_channel = config.get("upsample_initial_channel", 1024)
        stereo = config.get("stereo", True)
        activation = config.get("activation", "snake")
        use_bias_at_final = config.get("use_bias_at_final", True)


        # "output_sample_rate" is not present in recent checkpoint configs.
        # When absent (None), AudioVAE.output_sample_rate computes it as:
        #   sample_rate * vocoder.upsample_factor / mel_hop_length
        # where upsample_factor = product of all upsample stride lengths,
        # and mel_hop_length is loaded from the autoencoder config at
        # preprocessing.stft.hop_length (see CausalAudioAutoencoder).
        self.output_sample_rate = config.get("output_sample_rate")
        self.resblock = config.get("resblock", "1")
        self.use_tanh_at_final = config.get("use_tanh_at_final", True)
        self.apply_final_activation = config.get("apply_final_activation", True)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        in_channels = 128 if stereo else 64
        self.conv_pre = ops.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

        if self.resblock == "1":
            resblock_cls = ResBlock1
        elif self.resblock == "2":
            resblock_cls = ResBlock2
        elif self.resblock == "AMP1":
            resblock_cls = AMPBlock1
        else:
            raise ValueError(f"Unknown resblock type: {self.resblock}")

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ops.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                if self.resblock == "AMP1":
                    self.resblocks.append(resblock_cls(ch, k, d, activation=activation))
                else:
                    self.resblocks.append(resblock_cls(ch, k, d))

        out_channels = 2 if stereo else 1
        if self.resblock == "AMP1":
            act_cls = SnakeBeta if activation == "snakebeta" else Snake
            self.act_post = Activation1d(act_cls(ch))
        else:
            self.act_post = nn.LeakyReLU()

        self.conv_post = ops.Conv1d(
            ch, out_channels, 7, 1, padding=3, bias=use_bias_at_final
        )

        self.upsample_factor = np.prod([self.ups[i].stride[0] for i in range(len(self.ups))])


    def get_default_config(self):
        """Generate default configuration for the vocoder."""

        config = {
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_rates": [5, 4, 2, 2, 2],
            "upsample_kernel_sizes": [16, 16, 8, 4, 4],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_initial_channel": 1024,
            "stereo": True,
            "resblock": "1",
            "activation": "snake",
            "use_bias_at_final": True,
            "use_tanh_at_final": True,
        }

        return config

    def forward(self, x):
        """
        Forward pass of the vocoder.

        Args:
            x: Input spectrogram tensor. Can be:
               - 3D: (batch_size, channels, time_steps) for mono
               - 4D: (batch_size, 2, channels, time_steps) for stereo

        Returns:
            Audio tensor of shape (batch_size, out_channels, audio_length)
        """
        if x.dim() == 4:  # stereo
            assert x.shape[1] == 2, "Input must have 2 channels for stereo"
            x = torch.cat((x[:, 0, :, :], x[:, 1, :, :]), dim=1)
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            if self.resblock != "AMP1":
                x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.act_post(x)
        x = self.conv_post(x)

        if self.apply_final_activation:
            if self.use_tanh_at_final:
                x = torch.tanh(x)
            else:
                x = torch.clamp(x, -1, 1)

        return x


class _STFTFn(nn.Module):
    """Implements STFT as a convolution with precomputed DFT × Hann-window bases.

    The DFT basis rows (real and imaginary parts interleaved) multiplied by the causal
    Hann window are stored as buffers and loaded from the checkpoint. Using the exact
    bfloat16 bases from training ensures the mel values fed to the BWE generator are
    bit-identical to what it was trained on.
    """

    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, filter_length))
        self.register_buffer("inverse_basis", torch.zeros(n_freqs * 2, 1, filter_length))

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute magnitude and phase spectrogram from a batch of waveforms.

        Applies causal (left-only) padding of win_length - hop_length samples so that
        each output frame depends only on past and present input — no lookahead.
        The STFT is computed by convolving the padded signal with forward_basis.

        Args:
            y: Waveform tensor of shape (B, T).

        Returns:
            magnitude: Linear amplitude spectrogram, shape (B, n_freqs, T_frames).
            phase:     Phase spectrogram in radians, shape (B, n_freqs, T_frames).
                       Computed in float32 for numerical stability, then cast back to
                       the input dtype.
        """
        if y.dim() == 2:
            y = y.unsqueeze(1)                                # (B, 1, T)
        left_pad = max(0, self.win_length - self.hop_length)  # causal: left-only
        y = F.pad(y, (left_pad, 0))
        spec = F.conv1d(y, comfy.model_management.cast_to(self.forward_basis, dtype=y.dtype, device=y.device), stride=self.hop_length, padding=0)
        n_freqs = spec.shape[1] // 2
        real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
        magnitude = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.atan2(imag.float(), real.float()).to(real.dtype)
        return magnitude, phase


class MelSTFT(nn.Module):
    """Causal log-mel spectrogram module whose buffers are loaded from the checkpoint.

    Computes a log-mel spectrogram by running the causal STFT (_STFTFn) on the input
    waveform and projecting the linear magnitude spectrum onto the mel filterbank.

    The module's state dict layout matches the 'mel_stft.*' keys stored in the checkpoint
    (mel_basis, stft_fn.forward_basis, stft_fn.inverse_basis).
    """

    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
        sampling_rate: int,
        mel_fmin: float,
        mel_fmax: float,
    ):
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)

        n_freqs = filter_length // 2 + 1
        self.register_buffer("mel_basis", torch.zeros(n_mel_channels, n_freqs))

    def mel_spectrogram(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-mel spectrogram and auxiliary spectral quantities.

        Args:
            y: Waveform tensor of shape (B, T).

        Returns:
            log_mel:   Log-compressed mel spectrogram, shape (B, n_mel_channels, T_frames).
                       Computed as log(clamp(mel_basis @ magnitude, min=1e-5)).
            magnitude: Linear amplitude spectrogram, shape (B, n_freqs, T_frames).
            phase:     Phase spectrogram in radians, shape (B, n_freqs, T_frames).
            energy:    Per-frame energy (L2 norm over frequency), shape (B, T_frames).
        """
        magnitude, phase = self.stft_fn(y)
        energy = torch.norm(magnitude, dim=1)
        mel = torch.matmul(comfy.model_management.cast_to(self.mel_basis, dtype=magnitude.dtype, device=y.device), magnitude)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, magnitude, phase, energy


class VocoderWithBWE(torch.nn.Module):
    """Vocoder with bandwidth extension (BWE) for higher sample rate output.

    Chains a base vocoder (mel -> low-rate waveform) with a BWE stage that upsamples
    to a higher rate. The BWE computes a mel spectrogram from the low-rate waveform.

    The forward pass is wrapped in ``torch.autocast(dtype=float32)`` so that
    all operations run in fp32 regardless of weight dtype or caller context.
    The BigVGAN v2 architecture passes signals through 108 sequential
    convolutions (18 AMPBlocks x 6 convs) plus 36 anti-aliased activations;
    bfloat16 accumulation errors compound through this chain and degrade
    spectral metrics by 40-90% while adding only ~70 MB peak VRAM and ~20 ms
    latency compared to native bf16.  Weights may remain in bf16 for storage
    savings -- autocast upcasts them per-op at kernel level.
    """

    def __init__(self, config):
        super().__init__()
        vocoder_config = config["vocoder"]
        bwe_config = config["bwe"]

        self.vocoder = Vocoder(config=vocoder_config)
        self.bwe_generator = Vocoder(
            config={**bwe_config, "apply_final_activation": False}
        )

        self.input_sample_rate = bwe_config["input_sampling_rate"]
        self.output_sample_rate = bwe_config["output_sampling_rate"]
        self.hop_length = bwe_config["hop_length"]

        self.mel_stft = MelSTFT(
            filter_length=bwe_config["n_fft"],
            hop_length=bwe_config["hop_length"],
            win_length=bwe_config["n_fft"],
            n_mel_channels=bwe_config["num_mels"],
            sampling_rate=bwe_config["input_sampling_rate"],
            mel_fmin=0.0,
            mel_fmax=bwe_config["input_sampling_rate"] / 2.0,
        )
        self.resampler = UpSample1d(
            ratio=bwe_config["output_sampling_rate"] // bwe_config["input_sampling_rate"],
            persistent=False,
            window_type="hann",
        )

    def _compute_mel(self, audio):
        """Compute log-mel spectrogram from waveform using causal STFT bases."""
        B, C, T = audio.shape
        flat = audio.reshape(B * C, -1)                         # (B*C, T)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)      # (B*C, n_mels, T_frames)
        return mel.reshape(B, C, mel.shape[1], mel.shape[2])    # (B, C, n_mels, T_frames)

    def forward(self, mel_spec):
        input_dtype = mel_spec.dtype
        with torch.autocast(device_type=mel_spec.device.type, dtype=torch.float32):
            x = self.vocoder(mel_spec.float())
            _, _, T_low = x.shape
            T_out = T_low * self.output_sample_rate // self.input_sample_rate

            remainder = T_low % self.hop_length
            if remainder != 0:
                x = F.pad(x, (0, self.hop_length - remainder))

            mel = self._compute_mel(x)
            residual = self.bwe_generator(mel)
            skip = self.resampler(x)
            assert residual.shape == skip.shape, f"residual {residual.shape} != skip {skip.shape}"

            return torch.clamp(residual + skip, -1, 1)[..., :T_out].to(input_dtype)
