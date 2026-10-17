import math
import torch
from torch import nn
from .ldm.modules.attention import CrossAttention, FeedForward
import comfy.ops
ops = comfy.ops.manual_cast


class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        self.attn = CrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            operations=ops)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = ops.LayerNorm(query_dim)
        self.norm2 = ops.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as
        # original one
        self.scale = 1

    def forward(self, x, objs):

        x = x + self.scale * \
            torch.tanh(self.alpha_attn) * self.attn(self.norm1(x), objs, objs)
        x = x + self.scale * \
            torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x


class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj
        # feature
        self.linear = ops.Linear(context_dim, query_dim)

        self.attn = CrossAttention(
            query_dim=query_dim,
            context_dim=query_dim,
            heads=n_heads,
            dim_head=d_head,
            operations=ops)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = ops.LayerNorm(query_dim)
        self.norm2 = ops.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as
        # original one
        self.scale = 1

    def forward(self, x, objs):

        N_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(
            self.norm1(torch.cat([x, objs], dim=1)))[:, 0:N_visual, :]
        x = x + self.scale * \
            torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x


class GatedSelfAttentionDense2(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj
        # feature
        self.linear = ops.Linear(context_dim, query_dim)

        self.attn = CrossAttention(
            query_dim=query_dim, context_dim=query_dim, dim_head=d_head, operations=ops)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = ops.LayerNorm(query_dim)
        self.norm2 = ops.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as
        # original one
        self.scale = 1

    def forward(self, x, objs):

        B, N_visual, _ = x.shape
        B, N_ground, _ = objs.shape

        objs = self.linear(objs)

        # sanity check
        size_v = math.sqrt(N_visual)
        size_g = math.sqrt(N_ground)
        assert int(size_v) == size_v, "Visual tokens must be square rootable"
        assert int(size_g) == size_g, "Grounding tokens must be square rootable"
        size_v = int(size_v)
        size_g = int(size_g)

        # select grounding token and resize it to visual token size as residual
        out = self.attn(self.norm1(torch.cat([x, objs], dim=1)))[
            :, N_visual:, :]
        out = out.permute(0, 2, 1).reshape(B, -1, size_g, size_g)
        out = torch.nn.functional.interpolate(
            out, (size_v, size_v), mode='bicubic')
        residual = out.reshape(B, -1, N_visual).permute(0, 2, 1)

        # add residual to visual feature
        x = x + self.scale * torch.tanh(self.alpha_attn) * residual
        x = x + self.scale * \
            torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x


class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)


class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            ops.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            ops.Linear(512, 512),
            nn.SiLU(),
            ops.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(
            torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(
            torch.zeros([self.position_dim]))

    def forward(self, boxes, masks, positive_embeddings):
        B, N, _ = boxes.shape
        masks = masks.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 --> B*N*C

        # learnable null embedding
        positive_null = self.null_positive_feature.to(device=boxes.device, dtype=boxes.dtype).view(1, 1, -1)
        xyxy_null = self.null_position_feature.to(device=boxes.device, dtype=boxes.dtype).view(1, 1, -1)

        # replace padding with learnable null embedding
        positive_embeddings = positive_embeddings * \
            masks + (1 - masks) * positive_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        objs = self.linears(
            torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs


class Gligen(nn.Module):
    def __init__(self, modules, position_net, key_dim):
        super().__init__()
        self.module_list = nn.ModuleList(modules)
        self.position_net = position_net
        self.key_dim = key_dim
        self.max_objs = 30
        self.current_device = torch.device("cpu")

    def _set_position(self, boxes, masks, positive_embeddings):
        objs = self.position_net(boxes, masks, positive_embeddings)
        def func(x, extra_options):
            key = extra_options["transformer_index"]
            module = self.module_list[key]
            return module(x, objs.to(device=x.device, dtype=x.dtype))
        return func

    def set_position(self, latent_image_shape, position_params, device):
        batch, c, h, w = latent_image_shape
        masks = torch.zeros([self.max_objs], device="cpu")
        boxes = []
        positive_embeddings = []
        for p in position_params:
            x1 = (p[4]) / w
            y1 = (p[3]) / h
            x2 = (p[4] + p[2]) / w
            y2 = (p[3] + p[1]) / h
            masks[len(boxes)] = 1.0
            boxes += [torch.tensor((x1, y1, x2, y2)).unsqueeze(0)]
            positive_embeddings += [p[0]]
        append_boxes = []
        append_conds = []
        if len(boxes) < self.max_objs:
            append_boxes = [torch.zeros(
                [self.max_objs - len(boxes), 4], device="cpu")]
            append_conds = [torch.zeros(
                [self.max_objs - len(boxes), self.key_dim], device="cpu")]

        box_out = torch.cat(
            boxes + append_boxes).unsqueeze(0).repeat(batch, 1, 1)
        masks = masks.unsqueeze(0).repeat(batch, 1)
        conds = torch.cat(positive_embeddings +
                          append_conds).unsqueeze(0).repeat(batch, 1, 1)
        return self._set_position(
            box_out.to(device),
            masks.to(device),
            conds.to(device))

    def set_empty(self, latent_image_shape, device):
        batch, c, h, w = latent_image_shape
        masks = torch.zeros([self.max_objs], device="cpu").repeat(batch, 1)
        box_out = torch.zeros([self.max_objs, 4],
                              device="cpu").repeat(batch, 1, 1)
        conds = torch.zeros([self.max_objs, self.key_dim],
                            device="cpu").repeat(batch, 1, 1)
        return self._set_position(
            box_out.to(device),
            masks.to(device),
            conds.to(device))


def load_gligen(sd):
    sd_k = sd.keys()
    output_list = []
    key_dim = 768
    for a in ["input_blocks", "middle_block", "output_blocks"]:
        for b in range(20):
            k_temp = filter(lambda k: "{}.{}.".format(a, b)
                            in k and ".fuser." in k, sd_k)
            k_temp = map(lambda k: (k, k.split(".fuser.")[-1]), k_temp)

            n_sd = {}
            for k in k_temp:
                n_sd[k[1]] = sd[k[0]]
            if len(n_sd) > 0:
                query_dim = n_sd["linear.weight"].shape[0]
                key_dim = n_sd["linear.weight"].shape[1]

                if key_dim == 768:  # SD1.x
                    n_heads = 8
                    d_head = query_dim // n_heads
                else:
                    d_head = 64
                    n_heads = query_dim // d_head

                gated = GatedSelfAttentionDense(
                    query_dim, key_dim, n_heads, d_head)
                gated.load_state_dict(n_sd, strict=False)
                output_list.append(gated)

    if "position_net.null_positive_feature" in sd_k:
        in_dim = sd["position_net.null_positive_feature"].shape[0]
        out_dim = sd["position_net.linears.4.weight"].shape[0]

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.position_net = PositionNet(in_dim, out_dim)
        w.load_state_dict(sd, strict=False)

    gligen = Gligen(output_list, w.position_net, key_dim)
    return gligen
