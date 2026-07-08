import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management

class GELU(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, operations, device, dtype):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out, device = device, dtype = dtype)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:

        if gate.device.type == "mps":
            return F.gelu(gate.to(dtype = torch.float32)).to(dtype = gate.dtype)

        return F.gelu(gate)

    def forward(self, hidden_states):

        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)

        return hidden_states

class FeedForward(nn.Module):

    def __init__(self, dim: int, dim_out = None, mult: int = 4,
                dropout: float = 0.0, inner_dim = None, operations = None, device = None, dtype = None):

        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)

        dim_out = dim_out if dim_out is not None else dim

        act_fn = GELU(dim, inner_dim, operations = operations, device = device, dtype = dtype)

        self.net = nn.ModuleList([])
        self.net.append(act_fn)

        self.net.append(nn.Dropout(dropout))
        self.net.append(operations.Linear(inner_dim, dim_out, device = device, dtype = dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class AddAuxLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, loss):
        # do nothing in forward (no computation)
        ctx.requires_aux_loss = loss.requires_grad
        ctx.dtype = loss.dtype

        return x

    @staticmethod
    def backward(ctx, grad_output):
        # add the aux loss gradients
        grad_loss = None
        # put the aux grad the same as the main grad loss
        # aux grad contributes equally
        if ctx.requires_aux_loss:
            grad_loss = torch.ones(1, dtype = ctx.dtype, device = grad_output.device)

        return grad_output, grad_loss

class MoEGate(nn.Module):

    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01, device = None, dtype = None):

        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.alpha = aux_loss_alpha

        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim), device = device, dtype = dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        # flatten hidden states
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        # get logits and pass it to softmax
        logits = F.linear(hidden_states, comfy.model_management.cast_to(self.weight, dtype=hidden_states.dtype, device=hidden_states.device), bias = None)
        scores = logits.softmax(dim = -1)

        topk_weight, topk_idx = torch.topk(scores, k = self.top_k, dim = -1, sorted = False)

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores

            # used bincount instead of one hot encoding
            counts = torch.bincount(topk_idx.view(-1), minlength = self.n_routed_experts).float()
            ce = counts / topk_idx.numel()  # normalized expert usage

            # mean expert score
            Pi = scores_for_aux.mean(0)

            # expert balance loss
            aux_loss = (Pi * ce * self.n_routed_experts).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss

class MoEBlock(nn.Module):
    def __init__(self, dim, num_experts: int = 6, moe_top_k: int = 2, dropout: float = 0.0,
                 ff_inner_dim: int = None, operations = None, device = None, dtype = None):
        super().__init__()

        self.moe_top_k = moe_top_k
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            FeedForward(dim, dropout = dropout, inner_dim = ff_inner_dim, operations = operations, device = device, dtype = dtype)
            for _ in range(num_experts)
        ])

        self.gate = MoEGate(dim, num_experts = num_experts, num_experts_per_tok = moe_top_k, device = device, dtype = dtype)
        self.shared_experts = FeedForward(dim, dropout = dropout, inner_dim = ff_inner_dim, operations = operations, device = device, dtype = dtype)

    def forward(self, hidden_states) -> torch.Tensor:

        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:

            hidden_states = hidden_states.repeat_interleave(self.moe_top_k, dim = 0)
            y = torch.empty_like(hidden_states, dtype = hidden_states.dtype)

            for i, expert in enumerate(self.experts):
                tmp = expert(hidden_states[flat_topk_idx == i])
                y[flat_topk_idx == i] = tmp.to(hidden_states.dtype)

            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim = 1)
            y =  y.view(*orig_shape)

            y = AddAuxLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_expert_indices = flat_topk_idx,flat_expert_weights = topk_weight.view(-1, 1)).view(*orig_shape)

        y = y + self.shared_experts(identity)

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):

        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()

        # no need for .numpy().cpu() here
        tokens_per_expert = flat_expert_indices.bincount().cumsum(0)
        token_idxs = idxs // self.moe_top_k

        for i, end_idx in enumerate(tokens_per_expert):

            start_idx = 0 if i == 0 else tokens_per_expert[i-1]

            if start_idx == end_idx:
                continue

            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]

            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)

            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # use index_add_ with a 1-D index tensor directly avoids building a large [N, D] index map and extra memcopy required by scatter_reduce_
            # + avoid dtype conversion
            expert_cache.index_add_(0, exp_token_idx, expert_out)

        return expert_cache

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, downscale_freq_shift: float = 0.0,
                 scale: float = 1.0, max_period: int = 10000):
        super().__init__()

        self.num_channels = num_channels
        half_dim = num_channels // 2

        # precompute the “inv_freq” vector once
        exponent = -math.log(max_period) * torch.arange(
            half_dim, dtype=torch.float32
        ) / (half_dim - downscale_freq_shift)

        inv_freq = torch.exp(exponent)

        # pad
        if num_channels % 2 == 1:
            # we’ll pad a zero at the end of the cos-half
            inv_freq = torch.cat([inv_freq, inv_freq.new_zeros(1)])

        # register to buffer so it moves with the device
        self.register_buffer("inv_freq", inv_freq, persistent = False)
        self.scale = scale

    def forward(self, timesteps: torch.Tensor):

        x = timesteps.float().unsqueeze(1) * self.inv_freq.to(timesteps.device).unsqueeze(0)


        # fused CUDA kernels for sin and cos
        sin_emb = x.sin()
        cos_emb = x.cos()

        emb = torch.cat([sin_emb, cos_emb], dim = 1)

        # scale factor
        if self.scale != 1.0:
            emb = emb * self.scale

        # If we padded inv_freq for odd, emb is already wide enough; otherwise:
        if emb.shape[1] > self.num_channels:
            emb = emb[:, :self.num_channels]

        return emb

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size = 256, cond_proj_dim = None, operations = None, device = None, dtype = None):
        super().__init__()

        self.mlp = nn.Sequential(
            operations.Linear(hidden_size, frequency_embedding_size, bias=True, device = device, dtype = dtype),
            nn.GELU(),
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, device = device, dtype = dtype),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = operations.Linear(cond_proj_dim, frequency_embedding_size, bias=False, device = device, dtype = dtype)

        self.time_embed = Timesteps(hidden_size)

    def forward(self, timesteps, condition):

        timestep_embed = self.time_embed(timesteps).type(self.mlp[0].weight.dtype)

        if condition is not None:
            cond_embed = self.cond_proj(condition)
            timestep_embed = timestep_embed + cond_embed

        time_conditioned = self.mlp(timestep_embed)

        # for broadcasting with image tokens
        return time_conditioned.unsqueeze(1)

class MLP(nn.Module):
    def __init__(self, *, width: int, operations = None, device = None, dtype = None):
        super().__init__()
        self.width = width
        self.fc1 = operations.Linear(width, width * 4, device = device, dtype = dtype)
        self.fc2 = operations.Linear(width * 4, width, device = device, dtype = dtype)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        use_fp16: bool = False,
        operations = None,
        dtype = None,
        device = None,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim

        self.num_heads = num_heads
        self.head_dim = self.qdim // num_heads

        self.scale = self.head_dim ** -0.5

        self.to_q = operations.Linear(qdim, qdim, bias=qkv_bias, device = device, dtype = dtype)
        self.to_k = operations.Linear(kdim, qdim, bias=qkv_bias, device = device, dtype = dtype)
        self.to_v = operations.Linear(kdim, qdim, bias=qkv_bias, device = device, dtype = dtype)

        if use_fp16:
            eps = 1.0 / 65504
        else:
            eps = 1e-6

        if norm_layer == nn.LayerNorm:
            norm_layer = operations.LayerNorm
        else:
            norm_layer = operations.RMSNorm

        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps = eps, device = device, dtype = dtype) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps = eps, device = device, dtype = dtype) if qk_norm else nn.Identity()
        self.out_proj = operations.Linear(qdim, qdim, bias=True, device = device, dtype = dtype)

    def forward(self, x, y):

        b, s1, _ = x.shape
        _, s2, _ = y.shape

        y = y.to(next(self.to_k.parameters()).dtype)

        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2

        kv = kv.view(b, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)
        k = k.view(b, s2, self.num_heads, self.head_dim)
        v = v.reshape(b, s2, self.num_heads * self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = optimized_attention(
            q.reshape(b, s1, self.num_heads * self.head_dim),
            k.reshape(b, s2, self.num_heads * self.head_dim),
            v,
            heads=self.num_heads,
            low_precision_attention=False,
        )

        out = self.out_proj(x)

        return out

class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias = True,
        qk_norm = False,
        norm_layer = nn.LayerNorm,
        use_fp16: bool = False,
        operations = None,
        device = None,
        dtype = None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = operations.Linear(dim, dim, bias = qkv_bias, device = device, dtype = dtype)
        self.to_k = operations.Linear(dim, dim, bias = qkv_bias, device = device, dtype = dtype)
        self.to_v = operations.Linear(dim, dim, bias = qkv_bias, device = device, dtype = dtype)

        if use_fp16:
            eps = 1.0 / 65504
        else:
            eps = 1e-6

        if norm_layer == nn.LayerNorm:
            norm_layer = operations.LayerNorm
        else:
            norm_layer = operations.RMSNorm

        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps = eps, device = device, dtype = dtype) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps = eps, device = device, dtype = dtype) if qk_norm else nn.Identity()
        self.out_proj = operations.Linear(dim, dim, device = device, dtype = dtype)

    def forward(self, x):
        B, N, _ = x.shape

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        qkv_combined = torch.cat((query, key, value), dim=-1)
        split_size = qkv_combined.shape[-1] // self.num_heads // 3

        qkv = qkv_combined.view(B, -1, self.num_heads, split_size * 3)
        query, key, value = torch.split(qkv, split_size, dim=-1)

        query = query.reshape(B, N, self.num_heads, self.head_dim)
        key = key.reshape(B, N, self.num_heads, self.head_dim)
        value = value.reshape(B, N, self.num_heads * self.head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)

        x = optimized_attention(
            query.reshape(B, N, self.num_heads * self.head_dim),
            key.reshape(B, N, self.num_heads * self.head_dim),
            value,
            heads=self.num_heads,
            low_precision_attention=False,
        )

        x = self.out_proj(x)
        return x

class HunYuanDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        text_states_dim=1024,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=True,
        qkv_bias=True,
        skip_connection=True,
        timested_modulate=False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        use_fp16: bool = False,
        operations = None,
        device = None, dtype = None
    ):
        super().__init__()

        # eps can't be 1e-6 in fp16 mode because of numerical stability issues
        if use_fp16:
            eps = 1.0 / 65504
        else:
            eps = 1e-6

        self.norm1 = norm_layer(hidden_size, elementwise_affine = True, eps = eps, device = device, dtype = dtype)

        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                               norm_layer=qk_norm_layer, use_fp16 = use_fp16, device = device, dtype = dtype, operations = operations)

        self.norm2 = norm_layer(hidden_size, elementwise_affine = True, eps = eps, device = device, dtype = dtype)

        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(c_emb_size, hidden_size, bias=True, device = device, dtype = dtype)
            )

        self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_norm=qk_norm, norm_layer=qk_norm_layer, use_fp16 = use_fp16,
                                    device = device, dtype = dtype, operations = operations)

        self.norm3 = norm_layer(hidden_size, elementwise_affine = True, eps = eps, device = device, dtype = dtype)

        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine = True, eps = eps, device = device, dtype = dtype)
            self.skip_linear = operations.Linear(2 * hidden_size, hidden_size, device = device, dtype = dtype)
        else:
            self.skip_linear = None

        self.use_moe = use_moe

        if self.use_moe:
            self.moe = MoEBlock(
                hidden_size,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                dropout = 0.0,
                ff_inner_dim = int(hidden_size * 4.0),
                device = device, dtype = dtype,
                operations = operations
            )
        else:
            self.mlp = MLP(width=hidden_size, operations=operations, device = device, dtype = dtype)

    def forward(self, hidden_states, conditioning=None, text_states=None, skip_tensor=None):

        if self.skip_linear is not None:
            combined = torch.cat([skip_tensor, hidden_states], dim=-1)
            hidden_states = self.skip_linear(combined)
            hidden_states = self.skip_norm(hidden_states)

        # self attention
        if self.timested_modulate:
            modulation_shift = self.default_modulation(conditioning).unsqueeze(dim=1)
            hidden_states = hidden_states + modulation_shift

        self_attn_out = self.attn1(self.norm1(hidden_states))
        hidden_states = hidden_states + self_attn_out

        # cross attention
        hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), text_states)

        # MLP Layer
        mlp_input = self.norm3(hidden_states)

        if self.use_moe:
            hidden_states = hidden_states + self.moe(mlp_input)
        else:
            hidden_states = hidden_states + self.mlp(mlp_input)

        return hidden_states

class FinalLayer(nn.Module):

    def __init__(self, final_hidden_size, out_channels, operations, use_fp16: bool = False, device = None, dtype = None):
        super().__init__()

        if use_fp16:
            eps = 1.0 / 65504
        else:
            eps = 1e-6

        self.norm_final = operations.LayerNorm(final_hidden_size, elementwise_affine = True, eps = eps, device = device, dtype = dtype)
        self.linear = operations.Linear(final_hidden_size, out_channels, bias = True, device = device, dtype = dtype)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x

class HunYuanDiTPlain(nn.Module):

    # init with the defaults values from https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/hunyuan3d-dit-v2-1/config.yaml
    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 2048,
        context_dim: int = 1024,
        depth: int = 21,
        num_heads: int = 16,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        num_moe_layers: int = 6,
        guidance_cond_proj_dim = 2048,
        norm_type = 'layer',
        num_experts: int = 8,
        moe_top_k: int = 2,
        use_fp16: bool = False,
        dtype = None,
        device = None,
        operations = None,
        **kwargs
        ):

        self.dtype = dtype

        super().__init__()

        self.depth = depth

        self.in_channels = in_channels
        self.out_channels = in_channels

        self.num_heads = num_heads
        self.hidden_size = hidden_size

        norm = operations.LayerNorm if norm_type == 'layer' else operations.RMSNorm
        qk_norm = operations.RMSNorm

        self.context_dim = context_dim
        self.guidance_cond_proj_dim = guidance_cond_proj_dim

        self.x_embedder = operations.Linear(in_channels, hidden_size, bias = True, device = device, dtype = dtype)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4, cond_proj_dim = guidance_cond_proj_dim, device = device, dtype = dtype, operations = operations)


        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            text_states_dim=context_dim,
                            qk_norm=qk_norm,
                            norm_layer = norm,
                            qk_norm_layer = qk_norm,
                            skip_connection=layer > depth // 2,
                            qkv_bias=qkv_bias,
                            use_moe=True if depth - layer <= num_moe_layers else False,
                            num_experts=num_experts,
                            moe_top_k=moe_top_k,
                            use_fp16 = use_fp16,
                            device = device, dtype = dtype, operations = operations)
            for layer in range(depth)
        ])

        self.depth = depth

        self.final_layer = FinalLayer(hidden_size, self.out_channels, use_fp16 = use_fp16, operations = operations, device = device, dtype = dtype)

    def forward(self, x, t, context, transformer_options = {}, **kwargs):

        x = x.movedim(-1, -2)

        swap_cfg_halves = context.shape[0] >= 2

        if swap_cfg_halves:
            first_half, second_half = context.chunk(2, dim = 0)
            context = torch.cat([second_half, first_half], dim = 0)

        main_condition = context

        t = 1.0 - t

        time_embedded = self.t_embedder(t, condition = kwargs.get('guidance_cond'))

        x = x.to(dtype = next(self.x_embedder.parameters()).dtype)
        x_embedded = self.x_embedder(x)

        combined = torch.cat([time_embedded, x_embedded], dim=1)

        def block_wrap(args):
            return block(
                args["x"],
                args["t"],
                args["cond"],
                skip_tensor=args.get("skip"),)

        skip_stack = []
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        for idx, block in enumerate(self.blocks):
            if idx <= self.depth // 2:
                skip_input = None
            else:
                skip_input = skip_stack.pop()

            if ("block", idx) in blocks_replace:

                combined = blocks_replace[("block", idx)](
                    {
                        "x": combined,
                        "t": time_embedded,
                        "cond": main_condition,
                        "skip": skip_input,
                    },
                    {"original_block": block_wrap},
                )
            else:
                combined = block(combined, time_embedded, main_condition, skip_tensor=skip_input)

            if idx < self.depth // 2:
                skip_stack.append(combined)

        output = self.final_layer(combined)
        output =  output.movedim(-2, -1) * (-1.0)

        if swap_cfg_halves:
            first_half, second_half = output.chunk(2, dim = 0)
            output = torch.cat([second_half, first_half], dim = 0)

        return output
