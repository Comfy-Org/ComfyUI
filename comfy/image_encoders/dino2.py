import torch
import torch.nn.functional as F

from comfy.text_encoders.bert import BertAttention
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.ldm.depth_anything_3.reference_view_selector import (
    select_reference_view, reorder_by_reference, restore_original_order,
    THRESH_FOR_REF_SELECTION,
)


class Dino2AttentionOutput(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.dense = operations.Linear(input_dim, output_dim, dtype=dtype, device=device)

    def forward(self, x):
        return self.dense(x)


class Dino2AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, heads, layer_norm_eps, dtype, device, operations,
                 qk_norm=False):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.attention = BertAttention(embed_dim, heads, dtype, device, operations)
        self.output = Dino2AttentionOutput(embed_dim, embed_dim, layer_norm_eps, dtype, device, operations)
        if qk_norm:
            self.q_norm = operations.LayerNorm(self.head_dim, dtype=dtype, device=device)
            self.k_norm = operations.LayerNorm(self.head_dim, dtype=dtype, device=device)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x, mask, optimized_attention, pos=None, rope=None):
        # Fast path used by the existing CLIP-vision DINOv2 (no DA3 extensions).
        if self.q_norm is None and rope is None:
            return self.output(self.attention(x, mask, optimized_attention))

        # DA3 path: do QKV manually so we can apply per-head QK-norm and 2D RoPE.
        attn = self.attention
        B, N, C = x.shape
        h = self.heads
        d = self.head_dim
        q = attn.query(x).view(B, N, h, d).transpose(1, 2)
        k = attn.key(x).view(B, N, h, d).transpose(1, 2)
        v = attn.value(x).view(B, N, h, d).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None and pos is not None:
            q = rope(q, pos)
            k = rope(k, pos)
        out = optimized_attention(q, k, v, h, mask=mask, skip_reshape=True)
        return self.output(out)


class LayerScale(torch.nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.lambda1 = torch.nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x):
        return x * comfy.model_management.cast_to_device(self.lambda1, x.device, x.dtype)

class Dinov2MLP(torch.nn.Module):
    def __init__(self, hidden_size: int, dtype, device, operations):
        super().__init__()

        mlp_ratio = 4
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = operations.Linear(hidden_size, hidden_features, bias = True, device=device, dtype=dtype)
        self.fc2 = operations.Linear(hidden_features, hidden_size, bias = True, device=device, dtype=dtype)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = torch.nn.functional.gelu(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state

class SwiGLUFFN(torch.nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        in_features = out_features = dim
        hidden_features = int(dim * 4)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = operations.Linear(in_features, 2 * hidden_features, bias=True, device=device, dtype=dtype)
        self.weights_out = operations.Linear(hidden_features, out_features, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        x = self.weights_in(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = torch.nn.functional.silu(x1) * x2
        return self.weights_out(x)


class Dino2Block(torch.nn.Module):
    def __init__(self, dim, num_heads, layer_norm_eps, dtype, device, operations, use_swiglu_ffn,
                 qk_norm=False):
        super().__init__()
        self.attention = Dino2AttentionBlock(dim, num_heads, layer_norm_eps, dtype, device, operations,
                                             qk_norm=qk_norm)
        self.layer_scale1 = LayerScale(dim, dtype, device, operations)
        self.layer_scale2 = LayerScale(dim, dtype, device, operations)
        if use_swiglu_ffn:
            self.mlp = SwiGLUFFN(dim, dtype, device, operations)
        else:
            self.mlp = Dinov2MLP(dim, dtype, device, operations)
        self.norm1 = operations.LayerNorm(dim, eps=layer_norm_eps, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(dim, eps=layer_norm_eps, dtype=dtype, device=device)

    def forward(self, x, optimized_attention, pos=None, rope=None, attn_mask=None):
        x = x + self.layer_scale1(self.attention(self.norm1(x), attn_mask, optimized_attention,
                                                 pos=pos, rope=rope))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


# -----------------------------------------------------------------------------
# 2D Rotary position embedding (DA3 extension)
# -----------------------------------------------------------------------------


class _PositionGetter:
    """Cache (h, w) -> flat (y, x) position grid used to feed ``rope``."""

    def __init__(self):
        self._cache: dict = {}

    def __call__(self, batch_size: int, height: int, width: int, device) -> torch.Tensor:
        key = (height, width, device)
        if key not in self._cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            self._cache[key] = torch.cartesian_prod(y, x)
        cached = self._cache[key]
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(torch.nn.Module):
    """2D RoPE used by DA3-Small/Base. No learnable parameters."""

    def __init__(self, frequency: float = 100.0):
        super().__init__()
        self.base_frequency = frequency
        self._freq_cache: dict = {}

    def _components(self, dim: int, seq_len: int, device, dtype):
        key = (dim, seq_len, device, dtype)
        if key not in self._freq_cache:
            exp = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exp)
            pos = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            ang = torch.einsum("i,j->ij", pos, inv_freq)
            ang = ang.to(dtype)
            ang = torch.cat((ang, ang), dim=-1)
            self._freq_cache[key] = (ang.cos().to(dtype), ang.sin().to(dtype))
        return self._freq_cache[key]

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d(self, tokens, positions, cos_c, sin_c):
        cos = F.embedding(positions, cos_c)[:, None, :, :]
        sin = F.embedding(positions, sin_c)[:, None, :, :]
        return (tokens * cos) + (self._rotate(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        feature_dim = tokens.size(-1) // 2
        max_pos = int(positions.max()) + 1
        cos_c, sin_c = self._components(feature_dim, max_pos, tokens.device, tokens.dtype)
        v, h = tokens.chunk(2, dim=-1)
        v = self._apply_1d(v, positions[..., 0], cos_c, sin_c)
        h = self._apply_1d(h, positions[..., 1], cos_c, sin_c)
        return torch.cat((v, h), dim=-1)


class Dino2Encoder(torch.nn.Module):
    def __init__(self, dim, num_heads, layer_norm_eps, num_layers, dtype, device, operations, use_swiglu_ffn,
                 qknorm_start: int = -1):
        super().__init__()
        self.layer = torch.nn.ModuleList([
            Dino2Block(
                dim, num_heads, layer_norm_eps, dtype, device, operations,
                use_swiglu_ffn=use_swiglu_ffn,
                qk_norm=(qknorm_start != -1 and i >= qknorm_start),
            )
            for i in range(num_layers)
        ])

    def forward(self, x, intermediate_output=None):
        # Backward-compat path used by ``ClipVisionModel`` (no DA3 extensions).
        optimized_attention = optimized_attention_for_device(x.device, False, small_input=True)

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layer) + intermediate_output

        intermediate = None
        for i, layer in enumerate(self.layer):
            x = layer(x, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class Dino2PatchEmbeddings(torch.nn.Module):
    def __init__(self, dim, num_channels=3, patch_size=14, image_size=518, dtype=None, device=None, operations=None):
        super().__init__()
        self.patch_size = patch_size
        self.projection = operations.Conv2d(
            in_channels=num_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
            dtype=dtype,
            device=device
        )

    def forward(self, pixel_values):
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class Dino2Embeddings(torch.nn.Module):
    def __init__(self, dim, dtype, device, operations,
                 patch_size: int = 14, image_size: int = 518,
                 use_mask_token: bool = True,
                 num_camera_tokens: int = 0):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size

        self.patch_embeddings = Dino2PatchEmbeddings(dim, patch_size=patch_size, image_size=image_size, dtype=dtype, device=device, operations=operations)
        self.position_embeddings = torch.nn.Parameter(torch.empty(1, (image_size // patch_size) ** 2 + 1, dim, dtype=dtype, device=device))
        self.cls_token = torch.nn.Parameter(torch.empty(1, 1, dim, dtype=dtype, device=device)) # mask_token is a pre-training param, kept only so strict loading accepts the key.
        if use_mask_token:
            self.mask_token = torch.nn.Parameter(torch.empty(1, dim, dtype=dtype, device=device))
        else:
            self.mask_token = None
        if num_camera_tokens > 0:
            # DA3 stores (ref_token, src_token) pairs that get injected at the
            # alt-attn boundary; see ``Dinov2Model._inject_camera_token``.
            self.camera_token = torch.nn.Parameter(torch.empty(1, num_camera_tokens, dim, dtype=dtype, device=device))
        else:
            self.camera_token = None

    def interpolate_pos_encoding(self, x, h_pixels, w_pixels):
        pos_embed = comfy.model_management.cast_to_device(self.position_embeddings, x.device, torch.float32)

        class_pos = pos_embed[:, 0:1]
        patch_pos = pos_embed[:, 1:]
        N = patch_pos.shape[1]
        M = int(N ** 0.5)
        assert N == M * M, f"DINOv2 position grid must be square, got N={N} patches (sqrt={M})"
        h0 = h_pixels // self.patch_size
        w0 = w_pixels // self.patch_size
        # +0.1 matches upstream DINOv2's FP-rounding workaround so the interpolate output size lands on (h0, w0).
        # scale_factor is (height_scale, width_scale) -- height MUST come first;
        # swapping these only happens to work for square inputs and breaks
        # non-square paths like DA3-Small / DA3-Base multi-view.
        scale_factor = ((h0 + 0.1) / M, (w0 + 0.1) / M)

        patch_pos = patch_pos.reshape(1, M, M, -1).permute(0, 3, 1, 2)
        patch_pos = torch.nn.functional.interpolate(patch_pos, scale_factor=scale_factor, mode="bicubic", antialias=False)
        assert (h0, w0) == patch_pos.shape[-2:], (
            f"Interpolated pos-embed grid {tuple(patch_pos.shape[-2:])} does not match "
            f"target patch grid ({h0}, {w0}) for input {h_pixels}x{w_pixels} (patch_size={self.patch_size}); "
            f"check scale_factor axis order and +0.1 rounding workaround"
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat((class_pos, patch_pos), dim=1).to(x.dtype)

    def forward(self, pixel_values):
        x = self.patch_embeddings(pixel_values)
        x = torch.cat((self.cls_token.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1), x), dim=1)
        if x.shape[1] - 1 == self.position_embeddings.shape[1] - 1:
            x = x + comfy.model_management.cast_to_device(self.position_embeddings, x.device, x.dtype)
        else:
            h, w = pixel_values.shape[-2:]
            x = x + self.interpolate_pos_encoding(x, h, w)
        return x


class Dinov2Model(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        num_layers = config_dict["num_hidden_layers"]
        dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        layer_norm_eps = config_dict["layer_norm_eps"]
        use_swiglu_ffn = config_dict["use_swiglu_ffn"]
        patch_size = config_dict.get("patch_size", 14)
        image_size = config_dict.get("image_size", 518)
        use_mask_token = config_dict.get("use_mask_token", True)

        # DA3 extensions (all default to disabled).
        self.alt_start = config_dict.get("alt_start", -1)
        self.qknorm_start = config_dict.get("qknorm_start", -1)
        self.rope_start = config_dict.get("rope_start", -1)
        self.cat_token = config_dict.get("cat_token", False)
        rope_freq = config_dict.get("rope_freq", 100.0)

        self.embed_dim = dim
        self.patch_size = patch_size
        self.num_register_tokens = 0
        self.patch_start_idx = 1

        if self.rope_start != -1 and rope_freq > 0:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
            self._position_getter = _PositionGetter()
        else:
            self.rope = None
            self._position_getter = None

        # camera_token shape: (1, 2, dim) -> (ref_token, src_token).
        num_cam_tokens = 2 if self.alt_start != -1 else 0

        self.embeddings = Dino2Embeddings(
            dim, dtype, device, operations,
            patch_size=patch_size, image_size=image_size,
            use_mask_token=use_mask_token, num_camera_tokens=num_cam_tokens,
        )
        self.encoder = Dino2Encoder(
            dim, heads, layer_norm_eps, num_layers, dtype, device, operations,
            use_swiglu_ffn=use_swiglu_ffn,
            qknorm_start=self.qknorm_start,
        )
        self.layernorm = operations.LayerNorm(dim, eps=layer_norm_eps, dtype=dtype, device=device)

    def forward(self, pixel_values, attention_mask=None, intermediate_output=None):
        if self.alt_start != -1:
            raise RuntimeError(
                "Dinov2Model.forward() is the backward-compatible CLIP-vision path and does not "
                "apply DA3 extensions (RoPE, alternating attention, camera-token injection). "
                "Use get_intermediate_layers_da3() for Depth Anything 3 models."
            )
        x = self.embeddings(pixel_values)
        x, i = self.encoder(x, intermediate_output=intermediate_output)
        x = self.layernorm(x)
        pooled_output = x[:, 0, :]
        return x, i, pooled_output, None

    def get_intermediate_layers(self, pixel_values, indices, apply_norm=True):
        """Single-view multi-layer feature extraction."""
        x = self.embeddings(pixel_values)
        optimized_attention = optimized_attention_for_device(x.device, False, small_input=True)
        n_layers = len(self.encoder.layer)
        resolved = [(i if i >= 0 else n_layers + i) for i in indices]
        target = set(resolved)
        max_idx = max(resolved)
        n_skip = 1  # skip cls token
        cache = {}
        for i, layer in enumerate(self.encoder.layer):
            x = layer(x, optimized_attention)
            if i in target:
                normed = self.layernorm(x) if apply_norm else x
                cache[i] = (normed[:, n_skip:], normed[:, 0])
            if i >= max_idx:
                break
        return [cache[i] for i in resolved]

    # ------------------------------------------------------------------
    # Depth Anything 3 forward
    # ------------------------------------------------------------------
    def _prepare_rope_positions(self, B, S, H, W, device):
        if self.rope is None:
            return None, None
        ph, pw = H // self.patch_size, W // self.patch_size
        pos = self._position_getter(B * S, ph, pw, device=device)
        # Shift so the cls/cam token at position 0 is reserved for "no diff".
        pos = pos + 1
        cls_pos = torch.zeros(B * S, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
        # Per-view local: real grid positions for patches, 0 for cls token.
        pos_local = torch.cat([cls_pos, pos], dim=1)
        # Global (across views): same grid positions; cls token still at 0,
        # but patches share the same positions in every view.
        pos_global = torch.cat([cls_pos, torch.zeros_like(pos) + 1], dim=1)
        return pos_local, pos_global

    def _inject_camera_token(self, x: torch.Tensor, B: int, S: int, cam_token: "torch.Tensor | None") -> torch.Tensor:
        # x: (B, S, N, C). Replace token at index 0 with the camera token.
        if cam_token is not None:
            inj = cam_token
        else:
            ct = comfy.model_management.cast_to_device(self.embeddings.camera_token, x.device, x.dtype)
            ref_token = ct[:, :1].expand(B, -1, -1)
            src_token = ct[:, 1:].expand(B, max(S - 1, 0), -1)
            inj = torch.cat([ref_token, src_token], dim=1)
        x = x.clone()
        x[:, :, 0] = inj
        return x

    def get_intermediate_layers_da3(self, pixel_values, out_layers, cam_token=None, ref_view_strategy="saddle_balanced", export_feat_layers=None):
        """Multi-view multi-layer feature extraction used by Depth Anything 3."""
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        assert pixel_values.ndim == 5 and pixel_values.shape[2] == 3, \
            f"expected (B,3,H,W) or (B,S,3,H,W); got {tuple(pixel_values.shape)}"
        B, S, _, H, W = pixel_values.shape

        # Patch + cls + (interpolated) pos embed for each view.
        x = pixel_values.reshape(B * S, 3, H, W)
        x = self.embeddings(x)                          # (B*S, 1+N, C)
        x = x.reshape(B, S, x.shape[-2], x.shape[-1])    # (B, S, 1+N, C)

        pos_local, pos_global = self._prepare_rope_positions(B, S, H, W, x.device)
        # optimized_attention is only used by blocks without QK-norm/RoPE
        # (vanilla DINOv2 path); enabling-aware blocks fall through to SDPA.
        optimized_attention = optimized_attention_for_device(x.device, False, small_input=True)

        out_set = set(out_layers)
        export_set = set(export_feat_layers) if export_feat_layers else set()
        outputs: list[torch.Tensor] = []
        aux_outputs: list[torch.Tensor] = []
        local_x = x
        b_idx = None


        for i, blk in enumerate(self.encoder.layer):
            apply_rope = self.rope is not None and i >= self.rope_start
            block_rope = self.rope if apply_rope else None
            l_pos = pos_local if apply_rope else None
            g_pos = pos_global if apply_rope else None

            # Reference-view selection threshold: matches the upstream constant
            # THRESH_FOR_REF_SELECTION = 3. Skipped when a user-supplied
            # cam_token is provided (camera info already pins the geometry).
            if (self.alt_start != -1 and i == self.alt_start - 1 and S >= THRESH_FOR_REF_SELECTION and cam_token is None):
                b_idx = select_reference_view(x, strategy=ref_view_strategy)
                x = reorder_by_reference(x, b_idx)
                local_x = reorder_by_reference(local_x, b_idx)

            if self.alt_start != -1 and i == self.alt_start:
                x = self._inject_camera_token(x, B, S, cam_token)

            if self.alt_start != -1 and i >= self.alt_start and (i % 2 == 1):
                # Global attention across views: flatten S into the seq dim.
                t = x.reshape(B, S * x.shape[-2], x.shape[-1])
                p = g_pos.reshape(B, S * g_pos.shape[-2], g_pos.shape[-1]) if g_pos is not None else None
                t = blk(t, optimized_attention=optimized_attention, pos=p, rope=block_rope)
                x = t.reshape(B, S, x.shape[-2], x.shape[-1])
            else:
                # Per-view local attention.
                t = x.reshape(B * S, x.shape[-2], x.shape[-1])
                p = l_pos.reshape(B * S, l_pos.shape[-2], l_pos.shape[-1]) if l_pos is not None else None
                t = blk(t, optimized_attention=optimized_attention, pos=p, rope=block_rope)
                x = t.reshape(B, S, x.shape[-2], x.shape[-1])
                local_x = x

            if i in out_set:
                if self.cat_token:
                    out_x = torch.cat([local_x, x], dim=-1)
                else:
                    out_x = x
                # Restore original view order on the way out so heads see views
                # in the user's expected order.
                if b_idx is not None and self.alt_start != -1:
                    out_x = restore_original_order(out_x, b_idx)
                outputs.append(out_x)

            if i in export_set:
                aux = x
                if b_idx is not None and self.alt_start != -1:
                    aux = restore_original_order(aux, b_idx)
                aux_outputs.append(aux)

        # Apply final norm. When cat_token is set, only the right half
        # ("global" features) is normalised; the left half is left as-is to
        # match the upstream DA3 head signature.
        normed: list[torch.Tensor] = []
        cls_tokens: list[torch.Tensor] = []
        for out_x in outputs:
            cls_tokens.append(out_x[:, :, 0])
            if out_x.shape[-1] == self.embed_dim:
                normed.append(self.layernorm(out_x))
            elif out_x.shape[-1] == self.embed_dim * 2:
                left = out_x[..., :self.embed_dim]
                right = self.layernorm(out_x[..., self.embed_dim:])
                normed.append(torch.cat([left, right], dim=-1))
            else:
                raise ValueError(f"Unexpected token width: {out_x.shape[-1]}")

        # Drop cls/cam token from the patch sequence.
        normed = [o[..., 1 + self.num_register_tokens:, :] for o in normed]

        # Final layernorm + drop cls token from auxiliary features too.
        aux_normed = [self.layernorm(o)[..., 1 + self.num_register_tokens:, :]
                      for o in aux_outputs]
        return list(zip(normed, cls_tokens)), aux_normed
