# Wan-Animate-2: https://github.com/Wan-Video/Wan-Animate-2
"""Wan2.1-I2V-14B weights, driven by a video instead of a motion extractor.

A pose branch over the pose video's latents runs in lockstep with the generation
branch, feeding it K/V per block. The reference image is one extra latent frame at the
front of the generation branch, trimmed off by the caller. Upstream calls the pose video
the driving video and its branch forward_ref, not to be confused with the reference image.
"""

import torch

import comfy.ldm.common_dit
import comfy.model_management
import comfy.quant_ops
import comfy.utils
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import optimized_attention

from .model import WanAttentionBlock, WanModel, WanSelfAttention, repeat_e, sinusoidal_embedding_1d


class WanAnimate2SelfAttention(WanSelfAttention):

    def qkv(self, x, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = apply_rope1(self.norm_q(self.q(x)).view(b, s, n, d), freqs)
        k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs)
        return q, k, self.v(x).view(b, s, n, d)

    def _attn1_patch(self, x, q, k, transformer_options):
        for p in transformer_options.get("patches", {}).get("attn1_patch", []):
            x = p({"x": x, "q": q, "k": k, "transformer_options": transformer_options})
        return x

    def kv(self, x, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        return apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs), self.v(x).view(b, s, n, d)

    def forward_pose(self, x, freqs, transformer_options={}):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q, k, v = self.qkv(x, freqs)
        out = optimized_attention(q.reshape(b, s, n * d), k.reshape(b, s, n * d), v.reshape(b, s, n * d), heads=self.num_heads, transformer_options=transformer_options)
        return self.o(self._attn1_patch(out, q, k, transformer_options)), k, v

    def forward_gen(self, x, freqs, k_pose, v_pose, f_gen, hw, buffers, ref_strength=1.0, transformer_options={}):
        # frame j attends every gen token plus pose frame j-1 (frame 0 is the reference slot and has none)
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q, k, v = self.qkv(x, freqs)
        if ref_strength != 1.0:
            v[:, :hw] *= ref_strength  # frame 0 is the reference image's slot

        if k_pose is None:  # pose influence windowed out: plain self-attention, no per-frame loop
            out = optimized_attention(q.reshape(b, s, n * d), k.reshape(b, s, n * d), v.reshape(b, s, n * d), heads=self.num_heads, transformer_options=transformer_options)
            return self.o(self._attn1_patch(out, q, k, transformer_options))

        # gen half is the same every frame; only the hw-token pose tail is rewritten
        kbuf, vbuf, out = buffers
        kbuf[:, :s] = k
        vbuf[:, :s] = v

        for j in range(f_gen):
            q_j = q[:, j * hw:(j + 1) * hw].reshape(b, hw, n * d)
            if j == 0:
                kk, vv = k, v
            else:
                kbuf[:, s:] = k_pose[:, (j - 1) * hw:j * hw]
                vbuf[:, s:] = v_pose[:, (j - 1) * hw:j * hw]
                kk, vv = kbuf, vbuf
            out[:, j * hw:(j + 1) * hw] = optimized_attention(q_j, kk.reshape(b, kk.shape[1], n * d), vv.reshape(b, kk.shape[1], n * d), heads=self.num_heads, transformer_options=transformer_options)
        return self.o(self._attn1_patch(out, q, k, transformer_options))


class WanAnimate2Block(WanAttentionBlock):

    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6, operation_settings={}):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, operation_settings=operation_settings)
        self.self_attn = WanAnimate2SelfAttention(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)

    def _modulation(self, e, x):
        if e.ndim < 4:
            return (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        return (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)

    def _cross_attn_ffn(self, x, e, context, context_img_len, transformer_options):
        x = x + self.cross_attn(self.norm3(x), context, context_img_len=context_img_len, transformer_options=transformer_options)
        for p in transformer_options.get("patches", {}).get("attn2_patch", []):
            x = p({"x": x, "transformer_options": transformer_options})
        y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
        return torch.addcmul(x, y, repeat_e(e[5], x))

    def forward_pose(self, x, e, freqs, context, context_img_len=257, transformer_options={}):
        e = self._modulation(e, x)
        x = x.contiguous()
        y, k, v = self.self_attn.forward_pose(torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)), freqs, transformer_options=transformer_options)
        x = torch.addcmul(x, y, repeat_e(e[2], x))
        del y
        return self._cross_attn_ffn(x, e, context, context_img_len, transformer_options), k, v

    def kv_from_input(self, x_pose, e, freqs, transformer_options={}):
        e = self._modulation(e, x_pose)
        x_pose = x_pose.contiguous()
        return self.self_attn.kv(torch.addcmul(repeat_e(e[0], x_pose), self.norm1(x_pose), 1 + repeat_e(e[1], x_pose)), freqs)

    def forward_gen(self, x, e, freqs, context, k_pose, v_pose, f_gen, hw, buffers, ref_strength=1.0, context_img_len=257, transformer_options={}):
        e = self._modulation(e, x)
        x = x.contiguous()
        y = self.self_attn.forward_gen(torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)), freqs, k_pose, v_pose, f_gen, hw, buffers, ref_strength=ref_strength, transformer_options=transformer_options)
        x = torch.addcmul(x, y, repeat_e(e[2], x))
        del y
        return self._cross_attn_ffn(x, e, context, context_img_len, transformer_options)


class PoseBranchCache:
    """Pose-branch block inputs, reused across the sampling steps of one execution.

    Caching the block input rather than its K/V halves the memory; reprojecting K/V on read
    costs ~4% of re-running the block. One slot per distinct pose sequence, so under
    context windows each window keeps its own; least recently used slots are evicted when
    the store device runs low on memory. Created and freed by WanAnimate2Cache; Qwen-Image 2.1
    also stores its prefix K/V in it.
    """

    CONVROT_GROUPSIZE = 256

    def __init__(self, store_device=None, dtype="default"):
        self.store_device = torch.device(store_device) if store_device is not None else torch.device("cpu")
        self.dtype = dtype
        self.slots = []  # most recently used last
        self.slot = None
        self._pending = {}
        self._staging = {}

    def select(self, k, create=True):
        # select runs at a forward boundary: an interrupted forward can leave copies in flight that a different slot's forward would then mistake for its own
        if self._pending:
            for t, stream in self._pending.values():
                if stream is not None:
                    stream.synchronize()
            self._pending = {}
        for s in self.slots:
            if s["key"].shape == k.shape and torch.equal(s["key"], k.to(s["key"].device)):
                self.slots.remove(s)
                self.slots.append(s)
                self.slot = s
                return True
        self.slot = None
        if not create:
            return False
        # cache what fits: a filled slot is the size estimate for the next one, and least recently used slots make room when the store device runs low
        est = max((self._slot_bytes(s) for s in self.slots), default=0) * 1.5
        while self.slots and comfy.model_management.get_free_memory(self.store_device) < est:
            self._free_slot(self.slots.pop(0))
        self.slot = {"key": k.clone().to(self.store_device), "blocks": {}, "params": {}, "shape": None, "pinned": []}
        self.slots.append(self.slot)

    def _free_slot(self, s):
        for t, stream in self._pending.values():
            if stream is not None:
                stream.synchronize()  # an aborted forward can leave a copy in flight, still reading memory we are about to unpin
        self._pending = {}
        for t in s["pinned"]:
            comfy.model_management.unpin_memory(t)

    def free(self):
        for s in self.slots:
            self._free_slot(s)
        self.slots = []
        self.slot = None
        self._staging = {}

    def filled(self, num_blocks):
        return self.slot is not None and len(self.slot["blocks"]) == num_blocks

    def put(self, i, t):
        params = None
        if self.dtype in ("int8", "int4"):
            # convrot is what lets low-bit survive the ~125x per-channel outliers here, and over a [tokens, dim] view per-row scale means per-token. The kernels want 2D and a power-of-4 group that divides dim.
            self.slot["shape"] = t.shape
            g = self.CONVROT_GROUPSIZE
            while g > 4 and t.shape[-1] % g:
                g //= 4
            if self.dtype == "int4":
                t, params = comfy.quant_ops.TensorCoreConvRotW4A4Layout.quantize(t.reshape(-1, t.shape[-1]), convrot_groupsize=g)
            else:
                t, params = comfy.quant_ops.TensorWiseINT8Layout.quantize(t.reshape(-1, t.shape[-1]), is_weight=True, per_channel=True, convrot=True, convrot_groupsize=g)

        t = t.to(self.store_device, copy=True)
        if comfy.model_management.pin_memory(t):
            self.slot["pinned"].append(t)
        self.slot["blocks"][i] = t
        # the scales follow the blocks off the GPU: per-window slots would otherwise pile them up in VRAM (~200 MB per window at 480p int4)
        self.slot["params"][i] = params if params is None else params.to_device(self.store_device)

    def prefetch(self, i, device, dtype):
        # call before the compute this should overlap, so the stream waits only on work already enqueued
        if i not in self.slot["blocks"] or i in self._pending:
            return
        t = self.slot["blocks"][i]
        cast_dtype = None if self.slot["params"][i] is not None else dtype  # int8 entries move in their stored dtype and widen in take()
        stream = None
        r = None
        if t.device != device:
            stream = comfy.model_management.get_offload_stream(device)
            cs = comfy.model_management.current_stream(device)
            if stream is not None and cs is not None:
                # the handed-out stream last waited on the main stream a full rotation ago, which does not cover the previous consumer's reads of this slot; wait now so the copy cannot overwrite a slot still being read
                stream.wait_stream(cs)
            # two persistent staging buffers per tensor shape instead of a fresh allocation per block (~29 GB of churn per pass at 720p); windows of different lengths get their own pair
            buf_key = (tuple(t.shape), cast_dtype if cast_dtype is not None else t.dtype)
            if buf_key not in self._staging:
                self._staging[buf_key] = [torch.empty(t.shape, dtype=buf_key[1], device=device) for _ in range(2)]
            r = self._staging[buf_key][i % 2]
        self._pending[i] = (comfy.model_management.cast_to(t, cast_dtype, device, non_blocking=True, stream=stream, r=r), stream)

    def take(self, i, device, dtype, batch_size):
        if i not in self._pending:
            self.prefetch(i, device, dtype)
        t, stream = self._pending.pop(i)
        comfy.model_management.sync_stream(device, stream)
        params = self.slot["params"][i]
        if params is not None:
            layout = comfy.quant_ops.TensorCoreConvRotW4A4Layout if self.dtype == "int4" else comfy.quant_ops.TensorWiseINT8Layout
            t = layout.dequantize(t, params.to_device(t.device)).reshape(self.slot["shape"]).to(dtype)
        return comfy.utils.repeat_to_batch_size(t, batch_size)

    def _slot_bytes(self, s):
        return sum(t.numel() * t.element_size() for t in s["blocks"].values())

    def memory_bytes(self):
        return sum(self._slot_bytes(s) for s in self.slots)


class WanAnimate2Model(WanModel):

    def __init__(self,
                 model_type='animate2',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=36,
                 dim=5120,
                 ffn_dim=13824,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=40,
                 num_layers=40,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 in_dim_ref_conv=None,
                 image_model=None,
                 device=None, dtype=None, operations=None,
                 ):
        # model_type is 'animate2' in unet_config, but the checkpoint is i2v-shaped
        super().__init__(model_type='i2v', patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim,
                         text_dim=text_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm,
                         cross_attn_norm=cross_attn_norm, eps=eps, flf_pos_embed_token_number=flf_pos_embed_token_number, in_dim_ref_conv=in_dim_ref_conv,
                         wan_attn_block_class=WanAnimate2Block, image_model=image_model, device=device, dtype=dtype, operations=operations)

    def rope_encode_pose(self, t, h, w, w_patches, device=None, dtype=None):
        # t_start=1 lines pose frame j up with gen frame j+1, past the reference slot; shift_x parks it in its own strip of rope space.
        # The caller's rope_options are a user scaling knob and deliberately not forwarded.
        return super().rope_encode(t, h, w, t_start=1, device=device, dtype=dtype, transformer_options={"rope_options": {"shift_x": float(w_patches)}})

    def _forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, pose_latents=None, clip_fea_pose=None, context_pose=None, **kwargs):
        bs, c, t, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

        # h/w pre-pad: rope_encode's rounding reproduces the post-pad grid
        freqs = self.rope_encode(t, h, w, device=x.device, dtype=x.dtype, transformer_options=transformer_options)
        freqs_pose = None
        if pose_latents is not None:  # absent when the node's timestep window excludes this step
            pose_latents = comfy.ldm.common_dit.pad_to_patch_size(pose_latents.to(x.dtype), self.patch_size)
            w_patches = (w + (self.patch_size[2] // 2)) // self.patch_size[2]
            freqs_pose = self.rope_encode_pose(pose_latents.shape[2], h, w, w_patches, device=x.device, dtype=x.dtype)

        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs, freqs_pose=freqs_pose, pose_latents=pose_latents,
                                 clip_fea_pose=clip_fea_pose, context_pose=context_pose, transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]

    def forward_orig(self, x, t, context, clip_fea=None, freqs=None, freqs_pose=None, pose_latents=None, clip_fea_pose=None, context_pose=None, pose_strength=1.0, reference_strength=1.0, transformer_options={}, **kwargs):
        x_input = x[:, :, 1:]  # video-only: frame 0 is the reference slot, offset past it below
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        transformer_options["grid_sizes"] = grid_sizes
        f_gen, gh, gw = grid_sizes
        hw = gh * gw
        x = x.flatten(2).transpose(1, 2)

        # the node windows the pose influence via cond timestep ranges: outside the window the cond carries no pose latents, and the branch, its cache traffic and the per-frame attention loop are all skipped
        apply_pose = pose_latents is not None
        if apply_pose and pose_latents.shape[2] != f_gen - 1:  # before cache.select, which would otherwise keep an empty slot keyed to the rejected latents
            raise ValueError("pose branch has {} latent frames, expected {} (generation frames minus the reference-image slot)".format(pose_latents.shape[2], f_gen - 1))

        cache = transformer_options.get("animate2_cache", None) if apply_pose else None
        if cache is not None:
            cache.select(pose_latents[:1])  # keyed on batch element 0, so a cond batch size change mid-run stays valid
        cached = cache is not None and cache.filled(len(self.blocks))

        x_pose = None
        if not cached and apply_pose:
            # 36ch = [latents(16) | mask(4) | latents(16)]; latents twice, and the mask is all ones since every pose frame is known
            x_pose = self.patch_embedding(torch.cat([pose_latents, torch.ones_like(pose_latents[:, :4]), pose_latents], dim=1).float()).to(x.dtype)
            x_pose = x_pose.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x.dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        e0_pose = None
        if apply_pose:
            t_pose = torch.ones_like(t.flatten())
            e_pose = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_pose).to(dtype=x.dtype))
            e_pose = e_pose.reshape(t.shape[0], -1, e_pose.shape[-1])
            e0_pose = self.time_projection(e_pose).unflatten(2, (6, self.dim))

        context_gen = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_gen = torch.cat([self.img_emb(clip_fea), context_gen], dim=1)
            context_img_len = clip_fea.shape[-2]

        context_img_len_pose = None
        if not cached and apply_pose:
            context_pose = self.text_embedding(context if context_pose is None else context_pose)
            clip_fea_pose = clip_fea if clip_fea_pose is None else clip_fea_pose
            if clip_fea_pose is not None:
                if self.img_emb is not None:
                    context_pose = torch.cat([self.img_emb(clip_fea_pose), context_pose], dim=1)
                context_img_len_pose = clip_fea_pose.shape[-2]

        patches_replace = transformer_options.get("patches_replace", {})
        patches = transformer_options.get("patches", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"

        if cache is not None and not cached and apply_pose and "context_window" in transformer_options:
            # pose-only prepass, to avoid inflating dynamic VRAM calibration when using multiple context windows
            for i, block in enumerate(self.blocks):
                transformer_options["block_index"] = i
                cache.put(i, x_pose[:1])
                x_pose = block.forward_pose(x_pose, e0_pose, freqs_pose, context_pose, context_img_len=context_img_len_pose, transformer_options=transformer_options)[0]
            x_pose = None
            cached = True

        buffers = None
        if apply_pose:
            # allocated once and reused by every block
            n, d = self.num_heads, self.dim // self.num_heads
            buffers = (x.new_empty(x.shape[0], x.shape[1] + hw, n, d), x.new_empty(x.shape[0], x.shape[1] + hw, n, d), x.new_empty(x.shape[0], x.shape[1], self.dim))

        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i

            if not apply_pose:
                k_pose = v_pose = None
            elif cached:
                x_pose_in = cache.take(i, x.device, x.dtype, x.shape[0])
                cache.prefetch(i + 1, x.device, x.dtype)  # queue the next block before the gen compute it should overlap
                k_pose, v_pose = block.kv_from_input(x_pose_in, e0_pose, freqs_pose, transformer_options=transformer_options)
                del x_pose_in
            else:
                if cache is not None:
                    cache.put(i, x_pose[:1])
                # runs even under a block replace: its state has to reach block i+1
                x_pose, k_pose, v_pose = block.forward_pose(x_pose, e0_pose, freqs_pose, context_pose, context_img_len=context_img_len_pose, transformer_options=transformer_options)
            if v_pose is not None and pose_strength != 1.0:
                v_pose = v_pose * pose_strength

            if ("double_block", i) in blocks_replace:
                def block_wrap(args, block=block, k_pose=k_pose, v_pose=v_pose):
                    return {"img": block.forward_gen(args["img"], args["vec"], args["pe"], args["txt"], k_pose, v_pose, f_gen, hw, buffers, ref_strength=reference_strength, context_img_len=context_img_len, transformer_options=args["transformer_options"])}
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context_gen, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block.forward_gen(x, e0, freqs, context_gen, k_pose, v_pose, f_gen, hw, buffers, ref_strength=reference_strength, context_img_len=context_img_len, transformer_options=transformer_options)

            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": x, "x": x_input, "vec": e, "block_index": i, "img_offset": hw, "transformer_options": transformer_options})
                    x = out["img"]

        return self.unpatchify(self.head(x, e), grid_sizes)
