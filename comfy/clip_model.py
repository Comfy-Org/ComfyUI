import torch
from comfy.ldm.modules.attention import optimized_attention_for_device
import comfy.ops
import math

def clip_preprocess(image, size=224, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], crop=True):
    image = image[:, :, :, :3] if image.shape[3] > 3 else image
    mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std = torch.tensor(std, device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        if crop:
            scale = (size / min(image.shape[2], image.shape[3]))
            scale_size = (round(scale * image.shape[2]), round(scale * image.shape[3]))
        else:
            scale_size = (size, size)

        image = torch.nn.functional.interpolate(image, size=scale_size, mode="bicubic", antialias=True)
        h = (image.shape[2] - size)//2
        w = (image.shape[3] - size)//2
        image = image[:,:,h:h+size,w:w+size]
    image = torch.clip((255. * image), 0, 255).round() / 255.0
    return (image - mean.view([3,1,1])) / std.view([3,1,1])

def siglip2_flex_calc_resolution(oh, ow, patch_size, max_num_patches, eps=1e-5):
    def scale_dim(size, scale):
        scaled = math.ceil(size * scale / patch_size) * patch_size
        return max(patch_size, int(scaled))

    # Binary search for optimal scale
    lo, hi = eps / 10, 100.0
    while hi - lo >= eps:
        mid = (lo + hi) / 2
        h, w = scale_dim(oh, mid), scale_dim(ow, mid)
        if (h // patch_size) * (w // patch_size) <= max_num_patches:
            lo = mid
        else:
            hi = mid

    return scale_dim(oh, lo), scale_dim(ow, lo)

def siglip2_preprocess(image, size, patch_size, num_patches, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], crop=True):
    if size > 0:
        return clip_preprocess(image, size=size, mean=mean, std=std, crop=crop)

    image = image[:, :, :, :3] if image.shape[3] > 3 else image
    mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std = torch.tensor(std, device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)

    b, c, h, w = image.shape
    h, w = siglip2_flex_calc_resolution(h, w, patch_size, num_patches)

    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear", antialias=True)
    image = torch.clip((255. * image), 0, 255).round() / 255.0
    return (image - mean.view([3, 1, 1])) / std.view([3, 1, 1])

class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device, operations):
        super().__init__()

        self.heads = heads
        self.q_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None, optimized_attention=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out = optimized_attention(q, k, v, self.heads, mask)
        return self.out_proj(out)

ACTIVATIONS = {"quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
               "gelu": torch.nn.functional.gelu,
               "gelu_pytorch_tanh": lambda a: torch.nn.functional.gelu(a, approximate="tanh"),
}

class CLIPMLP(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_size, activation, dtype, device, operations):
        super().__init__()
        self.fc1 = operations.Linear(embed_dim, intermediate_size, bias=True, dtype=dtype, device=device)
        self.activation = ACTIVATIONS[activation]
        self.fc2 = operations.Linear(intermediate_size, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class CLIPLayer(torch.nn.Module):
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations):
        super().__init__()
        self.layer_norm1 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device, operations)
        self.layer_norm2 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device, operations)

    def forward(self, x, mask=None, optimized_attention=None):
        x += self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations):
        super().__init__()
        self.layers = torch.nn.ModuleList([CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

        all_intermediate = None
        if intermediate_output is not None:
            if intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
            if all_intermediate is not None:
                all_intermediate.append(x.unsqueeze(1).clone())

        if all_intermediate is not None:
            intermediate = torch.cat(all_intermediate, dim=1)

        return x, intermediate

class CLIPEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None, operations=None):
        super().__init__()
        self.token_embedding = operations.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)
        self.position_embedding = operations.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, dtype=torch.float32):
        return self.token_embedding(input_tokens, out_dtype=dtype) + comfy.ops.cast_to(self.position_embedding.weight, dtype=dtype, device=input_tokens.device)


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        num_positions = config_dict["max_position_embeddings"]
        self.eos_token_id = config_dict["eos_token_id"]

        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, num_positions=num_positions, dtype=dtype, device=device, operations=operations)
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
        self.final_layer_norm = operations.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens=None, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=torch.float32, embeds_info=[]):
        if embeds is not None:
            x = embeds + comfy.ops.cast_to(self.embeddings.position_embedding.weight, dtype=dtype, device=embeds.device)
        else:
            x = self.embeddings(input_tokens, dtype=dtype)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        causal_mask = torch.full((x.shape[1], x.shape[1]), -torch.finfo(x.dtype).max, dtype=x.dtype, device=x.device).triu_(1)

        if mask is not None:
            mask += causal_mask
        else:
            mask = causal_mask

        x, i = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)

        if num_tokens is not None:
            pooled_output = x[list(range(x.shape[0])), list(map(lambda a: a - 1, num_tokens))]
        else:
            pooled_output = x[torch.arange(x.shape[0], device=x.device), (torch.round(input_tokens).to(dtype=torch.int, device=x.device) == self.eos_token_id).int().argmax(dim=-1),]
        return x, i, pooled_output

class CLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device, operations)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])

def siglip2_pos_embed(embed_weight, embeds, orig_shape):
    embed_weight_len = round(embed_weight.shape[0] ** 0.5)
    embed_weight = comfy.ops.cast_to_input(embed_weight, embeds).movedim(1, 0).reshape(1, -1, embed_weight_len, embed_weight_len)
    embed_weight = torch.nn.functional.interpolate(embed_weight, size=orig_shape, mode="bilinear", align_corners=False, antialias=True)
    embed_weight = embed_weight.reshape(-1, embed_weight.shape[-2] * embed_weight.shape[-1]).movedim(0, 1)
    return embeds + embed_weight

class Siglip2Embeddings(torch.nn.Module):
    def __init__(self, embed_dim, num_channels=3, patch_size=14, image_size=224, model_type="", num_patches=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.patch_embedding = operations.Linear(num_channels * patch_size * patch_size, embed_dim, dtype=dtype, device=device)
        self.position_embedding = operations.Embedding(num_patches, embed_dim, dtype=dtype, device=device)
        self.patch_size = patch_size

    def forward(self, pixel_values):
        b, c, h, w = pixel_values.shape
        img = pixel_values.movedim(1, -1).reshape(b, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size, c)
        img = img.permute(0, 1, 3, 2, 4, 5)
        img = img.reshape(b, img.shape[1] * img.shape[2], -1)
        img = self.patch_embedding(img)
        return siglip2_pos_embed(self.position_embedding.weight, img, (h // self.patch_size, w // self.patch_size))

class CLIPVisionEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, num_channels=3, patch_size=14, image_size=224, model_type="", dtype=None, device=None, operations=None):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2
        if model_type == "siglip_vision_model":
            self.class_embedding = None
            patch_bias = True
        else:
            num_patches = num_patches + 1
            self.class_embedding = torch.nn.Parameter(torch.empty(embed_dim, dtype=dtype, device=device))
            patch_bias = False

        self.patch_embedding = operations.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=patch_bias,
            dtype=dtype,
            device=device
        )

        self.position_embedding = operations.Embedding(num_patches, embed_dim, dtype=dtype, device=device)

    def forward(self, pixel_values):
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        if self.class_embedding is not None:
            embeds = torch.cat([comfy.ops.cast_to_input(self.class_embedding, embeds).expand(pixel_values.shape[0], 1, -1), embeds], dim=1)
        return embeds + comfy.ops.cast_to_input(self.position_embedding.weight, embeds)


class CLIPVision(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        model_type = config_dict["model_type"]

        if model_type in ["siglip2_vision_model"]:
            self.embeddings = Siglip2Embeddings(embed_dim, config_dict["num_channels"], config_dict["patch_size"], config_dict["image_size"], model_type=model_type, num_patches=config_dict.get("num_patches", None), dtype=dtype, device=device, operations=operations)
        else:
            self.embeddings = CLIPVisionEmbeddings(embed_dim, config_dict["num_channels"], config_dict["patch_size"], config_dict["image_size"], model_type=model_type, dtype=dtype, device=device, operations=operations)
        if model_type in ["siglip_vision_model", "siglip2_vision_model"]:
            self.pre_layrnorm = lambda a: a
            self.output_layernorm = True
        else:
            self.pre_layrnorm = operations.LayerNorm(embed_dim)
            self.output_layernorm = False
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
        self.post_layernorm = operations.LayerNorm(embed_dim)

    def forward(self, pixel_values, attention_mask=None, intermediate_output=None):
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)
        #TODO: attention_mask?
        x, i = self.encoder(x, mask=None, intermediate_output=intermediate_output)
        if self.output_layernorm:
            x = self.post_layernorm(x)
            pooled_output = x
        else:
            pooled_output = self.post_layernorm(x[:, 0, :])
        return x, i, pooled_output

class LlavaProjector(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dtype, device, operations):
        super().__init__()
        self.linear_1 = operations.Linear(in_dim, out_dim, bias=True, device=device, dtype=dtype)
        self.linear_2 = operations.Linear(out_dim, out_dim, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear_2(torch.nn.functional.gelu(self.linear_1(x[:, 1:])))

class CLIPVisionModelProjection(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.vision_model = CLIPVision(config_dict, dtype, device, operations)
        if "projection_dim" in config_dict:
            self.visual_projection = operations.Linear(config_dict["hidden_size"], config_dict["projection_dim"], bias=False)
        else:
            self.visual_projection = torch.nn.Identity()

        if "llava3" == config_dict.get("projector_type", None):
            self.multi_modal_projector = LlavaProjector(config_dict["hidden_size"], 4096, dtype, device, operations)
        else:
            self.multi_modal_projector = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if "{}visual_projection.weight".format(prefix) not in state_dict:
            self.visual_projection = torch.nn.Identity()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, *args, **kwargs):
        x = self.vision_model(*args, **kwargs)
        out = self.visual_projection(x[2])
        projected = None
        if self.multi_modal_projector is not None:
            projected = self.multi_modal_projector(x[1])

        return (x[0], x[1], out, projected)
