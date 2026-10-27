import torch
from einops import rearrange, repeat
import comfy
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


def calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, split_num=8):
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q.transpose(1, 2) * scale

    B, H, x_seqlens, K = visual_q.shape

    x_ref_attn_maps = []
    for class_idx, ref_target_mask in enumerate(ref_target_masks):
        ref_target_mask = ref_target_mask.view(1, 1, 1, -1)

        x_ref_attnmap = torch.zeros(B, H, x_seqlens, device=visual_q.device, dtype=visual_q.dtype)
        chunk_size = min(max(x_seqlens // split_num, 1), x_seqlens)

        for i in range(0, x_seqlens, chunk_size):
            end_i = min(i + chunk_size, x_seqlens)

            attn_chunk = visual_q[:, :, i:end_i] @ ref_k.permute(0, 2, 3, 1)  # B, H, chunk, ref_seqlens

            # Apply softmax
            attn_max = attn_chunk.max(dim=-1, keepdim=True).values
            attn_chunk = (attn_chunk - attn_max).exp()
            attn_sum = attn_chunk.sum(dim=-1, keepdim=True)
            attn_chunk = attn_chunk / (attn_sum + 1e-8)

            # Apply mask and sum
            masked_attn = attn_chunk * ref_target_mask
            x_ref_attnmap[:, :, i:end_i] = masked_attn.sum(-1) / (ref_target_mask.sum() + 1e-8)

            del attn_chunk, masked_attn

        # Average across heads
        x_ref_attnmap = x_ref_attnmap.mean(dim=1)  # B, x_seqlens
        x_ref_attn_maps.append(x_ref_attnmap)

    del visual_q, ref_k

    return torch.cat(x_ref_attn_maps, dim=0)

def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=2):
    """Args:
        query (torch.tensor): B M H K
        key (torch.tensor): B M H K
        shape (tuple): (N_t, N_h, N_w)
        ref_target_masks: [B, N_h * N_w]
    """

    N_t, N_h, N_w = shape

    x_seqlens = N_h * N_w
    ref_k     = ref_k[:, :x_seqlens]
    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(class_num, seq_lens).to(visual_q)

    split_chunk = heads // split_num

    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(
            visual_q[:, :, i*split_chunk:(i+1)*split_chunk, :],
            ref_k[:, :, i*split_chunk:(i+1)*split_chunk, :],
            ref_target_masks
            )
        x_ref_attn_maps += x_ref_attn_maps_perhead

    return x_ref_attn_maps / split_num


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):
    source_min, source_max = source_range
    new_min, new_max = target_range
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def get_audio_embeds(encoded_audio, audio_start, audio_end):
    audio_embs = []
    human_num = len(encoded_audio)
    audio_frames = encoded_audio[0].shape[0]

    indices = (torch.arange(4 + 1) - 2) * 1

    for human_idx in range(human_num):
        if audio_end > audio_frames: # in case of not enough audio for current window, pad with first audio frame as that's most likely silence
            pad_len = audio_end - audio_frames
            pad_shape = list(encoded_audio[human_idx].shape)
            pad_shape[0] = pad_len
            pad_tensor = encoded_audio[human_idx][:1].repeat(pad_len, *([1] * (encoded_audio[human_idx].dim() - 1)))
            encoded_audio_in = torch.cat([encoded_audio[human_idx], pad_tensor], dim=0)
        else:
            encoded_audio_in = encoded_audio[human_idx]
        center_indices = torch.arange(audio_start, audio_end, 1).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=encoded_audio_in.shape[0] - 1)
        audio_emb = encoded_audio_in[center_indices].unsqueeze(0)
        audio_embs.append(audio_emb)

    return torch.cat(audio_embs, dim=0)


def project_audio_features(audio_proj, encoded_audio, audio_start, audio_end):
    audio_embs = get_audio_embeds(encoded_audio, audio_start, audio_end)

    first_frame_audio_emb_s = audio_embs[:, :1, ...]
    latter_frame_audio_emb = audio_embs[:, 1:, ...]
    latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=4)

    middle_index = audio_proj.seq_len // 2

    latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...]
    latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
    latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...]
    latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_frame_audio_emb_s = torch.cat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2)

    audio_emb = audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
    audio_emb = torch.cat(audio_emb.split(1), dim=2)

    return audio_emb


class RotaryPositionalEmbedding1D(torch.nn.Module):
    def __init__(self,
                 head_dim,
                 ):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000

    def precompute_freqs_cis_1d(self, pos_indices):
        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x, pos_indices):
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)

        x_ = x.float()

        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        x_ = (x_ * cos) + (rotate_half(x_) * sin)

        return x_.type_as(x)

class SingleStreamAttention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        device=None, dtype=None, operations=None
    ) -> None:
        super().__init__()
        self.dim = dim
        self.comfy_attention = ComfyAttention()
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = operations.Linear(dim, dim, bias=qkv_bias, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.kv_linear = operations.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, shape=None) -> torch.Tensor:
        N_t, N_h, N_w = shape

        expected_tokens = N_t * N_h * N_w
        actual_tokens = x.shape[1]
        x_extra = None

        if actual_tokens != expected_tokens:
            x_extra = x[:, -N_h * N_w:, :]
            x = x[:, :-N_h * N_w, :]
            N_t = N_t - 1

        B = x.shape[0]
        S = N_h * N_w
        x = x.view(B * N_t, S, self.dim)

        # get q for hidden_state
        q = self.q_linear(x).view(B * N_t, S, self.num_heads, self.head_dim)

        # get kv from encoder_hidden_states # shape: (B, N, num_heads, head_dim)
        kv = self.kv_linear(encoder_hidden_states)
        encoder_k, encoder_v = kv.view(B * N_t, encoder_hidden_states.shape[1], 2, self.num_heads, self.head_dim).unbind(2)

        #print("q.shape", q.shape) #torch.Size([21, 1024, 40, 128])
        q = AttentionTensorContainer(q.transpose(1, 2))
        encoder_k = AttentionTensorContainer(encoder_k.transpose(1, 2))
        encoder_v = AttentionTensorContainer(encoder_v.transpose(1, 2))
        del kv
        x = optimized_attention(q, encoder_k, encoder_v, heads=self.num_heads, preferred_attention=self.comfy_attention, skip_reshape=True, skip_output_reshape=True).transpose(1, 2)

        # linear transform
        x = self.proj(x.reshape(B * N_t, S, self.dim))
        x = x.view(B, N_t * S, self.dim)

        if x_extra is not None:
            x = torch.cat([x, torch.zeros_like(x_extra)], dim=1)

        return x

class SingleStreamMultiAttention(SingleStreamAttention):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        class_range: int = 24,
        class_interval: int = 4,
        device=None, dtype=None, operations=None
    ) -> None:
        super().__init__(
            dim=dim,
            encoder_hidden_states_dim=encoder_hidden_states_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            device=device,
            dtype=dtype,
            operations=operations
        )

        # Rotary-embedding layout parameters
        self.class_interval = class_interval
        self.class_range = class_range
        self.max_humans = self.class_range // self.class_interval

        # Constant bucket used for background tokens
        self.rope_bak = int(self.class_range // 2)

        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        shape=None,
        x_ref_attn_map=None
    ) -> torch.Tensor:
        encoder_hidden_states = encoder_hidden_states.squeeze(0).to(x.device)
        human_num = x_ref_attn_map.shape[0] if x_ref_attn_map is not None else 1
        # Single-speaker fall-through
        if human_num <= 1:
            return super().forward(x, encoder_hidden_states, shape)

        N_t, N_h, N_w = shape

        x_extra = None
        if x.shape[0] * N_t != encoder_hidden_states.shape[0]:
            x_extra = x[:, -N_h * N_w:, :]
            x = x[:, :-N_h * N_w, :]
            N_t = N_t - 1
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        # Query projection
        B, N, C = x.shape
        q = self.q_linear(x)
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Use `class_range` logic for 2 speakers
        rope_h1 = (0, self.class_interval)
        rope_h2 = (self.class_range - self.class_interval, self.class_range)
        rope_bak = int(self.class_range // 2)

        # Normalize and scale attention maps for each speaker
        max_values = x_ref_attn_map.max(1).values[:, None, None]
        min_values = x_ref_attn_map.min(1).values[:, None, None]
        max_min_values = torch.cat([max_values, min_values], dim=2)

        human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
        human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

        human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), rope_h1)
        human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), rope_h2)
        back = torch.full((x_ref_attn_map.size(1),), rope_bak, dtype=human1.dtype, device=human1.device)

        # Token-wise speaker dominance
        max_indices = x_ref_attn_map.argmax(dim=0)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        normalized_pos = normalized_map[torch.arange(x_ref_attn_map.size(1)), max_indices]

        # Apply rotary to Q
        q = rearrange(q, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        q = self.rope_1d(q, normalized_pos)
        q = rearrange(q, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        # Keys / Values
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv = encoder_kv.view(B, N_a, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        encoder_k, encoder_v = encoder_kv.unbind(0)

        # Rotary for keys – assign centre of each speaker bucket to its context tokens
        per_frame = torch.zeros(N_a, dtype=encoder_k.dtype, device=encoder_k.device)
        per_frame[: per_frame.size(0) // 2] = (rope_h1[0] + rope_h1[1]) / 2
        per_frame[per_frame.size(0) // 2 :] = (rope_h2[0] + rope_h2[1]) / 2
        encoder_pos = torch.cat([per_frame] * N_t, dim=0)

        encoder_k = rearrange(encoder_k, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        encoder_k = self.rope_1d(encoder_k, encoder_pos)
        encoder_k = rearrange(encoder_k, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        # Final attention
        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")

        q = AttentionTensorContainer(q.transpose(1, 2))
        encoder_k = AttentionTensorContainer(encoder_k.transpose(1, 2))
        encoder_v = AttentionTensorContainer(encoder_v.transpose(1, 2))
        del encoder_kv
        x = optimized_attention(q, encoder_k, encoder_v, heads=self.num_heads, preferred_attention=self.comfy_attention, skip_reshape=True, skip_output_reshape=True).transpose(1, 2)

        # Linear projection
        x = x.reshape(B, N, C)
        x = self.proj(x)

        # Restore original layout
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)
        if x_extra is not None:
            x = torch.cat([x, torch.zeros_like(x_extra)], dim=1)

        return x


class MultiTalkAudioProjModel(torch.nn.Module):
    def __init__(
        self,
        seq_len: int = 5,
        seq_len_vf: int = 12,
        blocks: int = 12,
        channels: int = 768,
        intermediate_dim: int = 512,
        out_dim: int = 768,
        context_tokens: int = 32,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.out_dim = out_dim

        # define multiple linear layers
        self.proj1 = operations.Linear(self.input_dim, intermediate_dim, device=device, dtype=dtype)
        self.proj1_vf = operations.Linear(self.input_dim_vf, intermediate_dim, device=device, dtype=dtype)
        self.proj2 = operations.Linear(intermediate_dim, intermediate_dim, device=device, dtype=dtype)
        self.proj3 = operations.Linear(intermediate_dim, context_tokens * out_dim, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(out_dim, device=device, dtype=dtype)

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c*N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c*N_t, self.context_tokens, self.out_dim)

        # normalization and reshape
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens


class WanMultiTalkAttentionBlock(torch.nn.Module):
    def __init__(self, in_dim=5120, out_dim=768, device=None, dtype=None, operations=None):
        super().__init__()
        self.audio_cross_attn = SingleStreamMultiAttention(in_dim, out_dim, num_heads=40, qkv_bias=True, device=device, dtype=dtype, operations=operations)
        self.norm_x = operations.LayerNorm(in_dim, device=device, dtype=dtype, elementwise_affine=True)


class MultiTalkGetAttnMapPatch:
    def __init__(self, ref_target_masks=None):
        self.ref_target_masks = ref_target_masks

    def __call__(self, kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        x = kwargs["x"]

        if self.ref_target_masks is not None:
            x_ref_attn_map = get_attn_map_with_target(kwargs["q"], kwargs["k"], transformer_options["grid_sizes"], ref_target_masks=self.ref_target_masks.to(x.device))
            transformer_options["x_ref_attn_map"] = x_ref_attn_map
        return x


class MultiTalkCrossAttnPatch:
    def __init__(self, model_patch, audio_scale=1.0, ref_target_masks=None):
        self.model_patch = model_patch
        self.audio_scale = audio_scale
        self.ref_target_masks = ref_target_masks

    def __call__(self, kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        block_idx = transformer_options.get("block_index", None)
        x = kwargs["x"]
        if block_idx is None:
            return torch.zeros_like(x)

        audio_embeds = transformer_options.get("audio_embeds")
        x_ref_attn_map = transformer_options.pop("x_ref_attn_map", None)

        norm_x = self.model_patch.model.blocks[block_idx].norm_x(x)
        x_audio = self.model_patch.model.blocks[block_idx].audio_cross_attn(
            norm_x, audio_embeds.to(x.dtype),
            shape=transformer_options["grid_sizes"],
            x_ref_attn_map=x_ref_attn_map
        )
        x = x + x_audio * self.audio_scale
        return x

    def models(self):
        return [self.model_patch]

class MultiTalkApplyModelWrapper:
    def __init__(self, init_latents):
        self.init_latents = init_latents

    def __call__(self, executor, x, *args, **kwargs):
        x[:, :, :self.init_latents.shape[2]] = self.init_latents.to(x)
        samples = executor(x, *args, **kwargs)
        return samples


class InfiniteTalkOuterSampleWrapper:
    def __init__(self, motion_frames_latent, model_patch, is_extend=False):
        self.motion_frames_latent = motion_frames_latent
        self.model_patch = model_patch
        self.is_extend = is_extend

    def __call__(self, executor, *args, **kwargs):
        model_patcher = executor.class_obj.model_patcher
        model_options = executor.class_obj.model_options
        process_latent_in = model_patcher.model.process_latent_in

        # for InfiniteTalk, model input first latent(s) need to always be replaced on every step
        if self.motion_frames_latent is not None:
            wrappers = model_options["transformer_options"]["wrappers"]
            w = wrappers.setdefault(comfy.patcher_extension.WrappersMP.APPLY_MODEL, {})
            w["MultiTalk_apply_model"] = [MultiTalkApplyModelWrapper(process_latent_in(self.motion_frames_latent))]

        # run the sampling process
        result = executor(*args, **kwargs)

        # insert motion frames before decoding
        if self.is_extend:
            overlap = self.motion_frames_latent.shape[2]
            result = torch.cat([self.motion_frames_latent.to(result), result[:, :, overlap:]], dim=2)

        return result

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            if self.motion_frames_latent is not None:
                self.motion_frames_latent = self.motion_frames_latent.to(device_or_dtype)
        return self
