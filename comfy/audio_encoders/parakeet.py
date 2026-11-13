import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention_masked
import comfy.audio
import comfy.model_management
import comfy.ops


def mel_features(audio, n_mels):
    # audio: [1, samples] float32 at 16 kHz
    audio = torch.cat([audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1]], dim=1)
    window = torch.hann_window(400, periodic=False, device=audio.device)
    spec = torch.stft(audio, 512, hop_length=160, win_length=400, window=window, return_complex=True, pad_mode="constant").abs().pow(2)
    mel_filters = comfy.audio.MelScale(n_mels, 16000, 0.0, 8000.0, 257).fb.to(audio.device)
    mel = torch.log(mel_filters.T @ spec + 2 ** -24)[:, :, :-1].transpose(1, 2)
    return (mel - mel.mean(dim=1, keepdim=True)) / (mel.std(dim=1, keepdim=True) + 1e-5)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_ff, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear1 = operations.Linear(dim, dim_ff, bias=False, dtype=dtype, device=device)
        self.linear2 = operations.Linear(dim_ff, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))


class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.pointwise_conv1 = operations.Conv1d(dim, 2 * dim, 1, bias=False, dtype=dtype, device=device)
        self.depthwise_conv = operations.Conv1d(dim, dim, kernel_size, padding=(kernel_size - 1) // 2, groups=dim, bias=False, dtype=dtype, device=device)
        self.norm = operations.BatchNorm2d(dim, dtype=dtype, device=device)
        self.pointwise_conv2 = operations.Conv1d(dim, dim, 1, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        x = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        x = self.norm(self.depthwise_conv(x).unsqueeze(-1)).squeeze(-1)
        return self.pointwise_conv2(F.silu(x)).transpose(1, 2)


class RelPosAttention(nn.Module):
    def __init__(self, dim, heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.k_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.v_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.o_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.relative_k_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.bias_u = nn.Parameter(torch.empty(heads, self.head_dim, dtype=dtype, device=device))
        self.bias_v = nn.Parameter(torch.empty(heads, self.head_dim, dtype=dtype, device=device))

    def forward(self, x, pos_emb):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q_u = (q + comfy.ops.cast_to_input(self.bias_u, q)).reshape(b, t, -1)
        q_v = (q + comfy.ops.cast_to_input(self.bias_v, q)).transpose(1, 2)

        rel_k = self.relative_k_proj(pos_emb).view(1, -1, self.heads, self.head_dim).permute(0, 2, 3, 1)
        bd = (q_v * self.head_dim ** -0.5) @ rel_k
        # row i of the relative scores starts at offset t - 1 - i
        bd = bd.as_strided((b, self.heads, t, t), (self.heads * t * (2 * t - 1), t * (2 * t - 1), 2 * t - 2, 1), t - 1)
        return self.o_proj(optimized_attention_masked(q_u, k, v, self.heads, bd))


class ConformerBlock(nn.Module):
    def __init__(self, dim, dim_ff, heads, kernel_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.feed_forward1 = FeedForward(dim, dim_ff, dtype=dtype, device=device, operations=operations)
        self.self_attn = RelPosAttention(dim, heads, dtype=dtype, device=device, operations=operations)
        self.conv = ConvolutionModule(dim, kernel_size, dtype=dtype, device=device, operations=operations)
        self.feed_forward2 = FeedForward(dim, dim_ff, dtype=dtype, device=device, operations=operations)
        self.norm_feed_forward1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.norm_self_att = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.norm_conv = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.norm_feed_forward2 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.norm_out = operations.LayerNorm(dim, dtype=dtype, device=device)

    def forward(self, x, pos_emb):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Subsampling(nn.Module):
    def __init__(self, n_mels, channels, dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            operations.Conv2d(1, channels, 3, stride=2, padding=1, dtype=dtype, device=device), nn.ReLU(),
            operations.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, dtype=dtype, device=device),
            operations.Conv2d(channels, channels, 1, dtype=dtype, device=device), nn.ReLU(),
            operations.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, dtype=dtype, device=device),
            operations.Conv2d(channels, channels, 1, dtype=dtype, device=device), nn.ReLU(),
        ])
        self.linear = operations.Linear(channels * (n_mels // 8), dim, dtype=dtype, device=device)

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return self.linear(x.transpose(1, 2).flatten(2))


class Encoder(nn.Module):
    def __init__(self, n_mels, dim, dim_ff, heads, layers, kernel_size, channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.dim = dim
        self.subsampling = Subsampling(n_mels, channels, dim, dtype=dtype, device=device, operations=operations)
        self.layers = nn.ModuleList([
            ConformerBlock(dim, dim_ff, heads, kernel_size, dtype=dtype, device=device, operations=operations) for _ in range(layers)
        ])

    def forward(self, x):
        x = self.subsampling(x)
        t = x.shape[1]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) / self.dim))
        freqs = torch.arange(t - 1, -t, -1, device=x.device, dtype=torch.float32)[:, None] * inv_freq
        pos_emb = torch.stack([freqs.sin(), freqs.cos()], dim=-1).flatten(1).to(x.dtype)[None]
        for layer in self.layers:
            x = layer(x, pos_emb)
        return x


class LSTM(nn.Module):
    def __init__(self, dim, layers, dtype=None, device=None):
        super().__init__()
        self.layers = layers
        for i in range(layers):
            for name in ("weight_ih", "weight_hh"):
                setattr(self, f"{name}_l{i}", nn.Parameter(torch.empty(4 * dim, dim, dtype=dtype, device=device)))
            for name in ("bias_ih", "bias_hh"):
                setattr(self, f"{name}_l{i}", nn.Parameter(torch.empty(4 * dim, dtype=dtype, device=device)))

    def forward(self, x, state):
        new_state = []
        for i, (h, c) in enumerate(state):
            w = [comfy.ops.cast_to_input(getattr(self, f"{name}_l{i}"), x) for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
            gate_i, gate_f, gate_g, gate_o = (F.linear(x, w[0], w[2]) + F.linear(h, w[1], w[3])).chunk(4, dim=-1)
            c = gate_f.sigmoid() * c + gate_i.sigmoid() * gate_g.tanh()
            x = gate_o.sigmoid() * c.tanh()
            new_state.append((x, c))
        return x, new_state


class Decoder(nn.Module):
    def __init__(self, vocab_size, dim, layers, dtype=None, device=None, operations=None):
        super().__init__()
        self.embedding = operations.Embedding(vocab_size, dim, dtype=dtype, device=device)
        self.lstm = LSTM(dim, layers, dtype=dtype, device=device)
        self.decoder_projector = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, token, state):
        out, state = self.lstm(self.embedding(token).to(state[0][0].dtype), state)
        return self.decoder_projector(out), state


class Joint(nn.Module):
    def __init__(self, dim, num_outputs, dtype=None, device=None, operations=None):
        super().__init__()
        self.head = operations.Linear(dim, num_outputs, dtype=dtype, device=device)

    def forward(self, enc, dec):
        return self.head(F.relu(enc + dec))


class ParakeetTDT(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        dim: int = 1024,
        dim_ff: int = 4096,
        heads: int = 8,
        layers: int = 24,
        kernel_size: int = 9,
        subsampling_channels: int = 256,
        decoder_dim: int = 640,
        decoder_layers: int = 2,
        vocab_size: int = 8192,
        durations=(0, 1, 2, 3, 4),
        max_symbols_per_step: int = 10,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.n_mels = n_mels
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.durations = durations
        self.max_symbols_per_step = max_symbols_per_step
        self.encoder = Encoder(n_mels, dim, dim_ff, heads, layers, kernel_size, subsampling_channels, dtype=dtype, device=device, operations=operations)
        self.encoder_projector = operations.Linear(dim, decoder_dim, dtype=dtype, device=device)
        self.decoder = Decoder(vocab_size + 1, decoder_dim, decoder_layers, dtype=dtype, device=device, operations=operations)
        self.joint = Joint(decoder_dim, vocab_size + 1 + len(durations), dtype=dtype, device=device, operations=operations)

    def transcribe(self, waveform):
        # greedy token-duration transducer decoding, returns (token, first frame, frame count) per emitted token
        enc = self.encoder_projector(self.encoder(mel_features(waveform, self.n_mels)))[0]
        blank = self.vocab_size
        zeros = torch.zeros(1, self.decoder_dim, device=enc.device, dtype=enc.dtype)
        state = [(zeros, zeros)] * self.decoder.lstm.layers
        dec, state = self.decoder(torch.full((1,), blank, device=enc.device), state)
        tokens = []
        frame = symbols = 0
        while frame < enc.shape[0]:
            comfy.model_management.throw_exception_if_processing_interrupted()
            logits = self.joint(enc[frame:frame + 1], dec)[0]
            token, duration = torch.stack([logits[:blank + 1].argmax(), logits[blank + 1:].argmax()]).tolist()
            duration = self.durations[duration]
            if token != blank:
                tokens.append((token, frame, duration))
                dec, state = self.decoder(torch.full((1,), token, device=enc.device), state)
                symbols += 1
            if duration == 0 and (token == blank or symbols >= self.max_symbols_per_step):
                duration = 1
            if duration > 0:
                frame += duration
                symbols = 0
        return tokens
