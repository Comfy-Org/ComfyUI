import torch
import comfy.ops
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from comfy.ldm.modules.attention import optimized_attention_for_device

CXT = [3072, 1536, 768, 384][1:][::-1][-3:]

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, device=None, dtype=None, operations=None):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = operations.Linear(dim, dim, bias=qkv_bias, device=device, dtype=dtype)
        self.kv = operations.Linear(dim, dim * 2, bias=qkv_bias, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x):
        B, N, C = x.shape
        optimized_attention = optimized_attention_for_device(x.device, mask=False, small_input=True)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        x = optimized_attention(
            q, k, v, heads=self.num_heads, skip_output_reshape=True, skip_reshape=True
        ).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, device=None, dtype=None, operations=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = operations.Linear(in_features, hidden_features, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.fc2 = operations.Linear(hidden_features, out_features, device=device, dtype=dtype)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, device=None, dtype=None, operations=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads, device=device, dtype=dtype))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.long().view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = comfy.ops.cast_to_input(relative_position_bias.permute(2, 0, 1).contiguous(), attn)  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim, device=device, dtype=dtype)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, device=device, dtype=dtype, operations=operations)

        self.norm2 = norm_layer(dim, device=device, dtype=dtype)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, device=device, dtype=dtype, operations=operations)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.reduction = operations.Linear(4 * dim, 2 * dim, bias=False, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(4 * dim, device=device, dtype=dtype)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                device=device, dtype=dtype, operations=operations)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, device=device, dtype=dtype, operations=operations)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None, device=None, dtype=None, operations=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = operations.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim, device=device, dtype=dtype)
        else:
            self.norm = None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 device=None, dtype=None, operations=None):
        super().__init__()

        norm_layer = partial(operations.LayerNorm, device=device, dtype=dtype)
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            device=device, dtype=dtype, operations=operations,
            norm_layer=norm_layer if self.patch_norm else None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                device=device, dtype=dtype, operations=operations)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


    def forward(self, x):
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)

        outs = []
        x = x.flatten(2).transpose(1, 2)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False, device=None, dtype=None, operations=None):

        super(DeformableConv2d, self).__init__()

        kernel_size = kernel_size if type(kernel_size) is tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) is tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = operations.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True, device=device, dtype=dtype)

        self.modulator_conv = operations.Conv2d(in_channels,
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True, device=device, dtype=dtype)

        self.regular_conv = operations.Conv2d(in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        weight, bias, offload_info = comfy.ops.cast_bias_weight(self.regular_conv, x, offloadable=True)

        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=weight,
            bias=None,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
        comfy.ops.uncast_bias_weight(self.regular_conv, weight, bias, offload_info)
        return x

class BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64, device=None, dtype=None, operations=None):
        super(BasicDecBlk, self).__init__()
        inter_channels = 64
        self.conv_in = operations.Conv2d(in_channels, inter_channels, 3, 1, padding=1, device=device, dtype=dtype)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(in_channels=inter_channels, device=device, dtype=dtype, operations=operations)
        self.conv_out = operations.Conv2d(inter_channels, out_channels, 3, 1, padding=1, device=device, dtype=dtype)
        self.bn_in = operations.BatchNorm2d(inter_channels, device=device, dtype=dtype)
        self.bn_out = operations.BatchNorm2d(out_channels, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, device=None, dtype=None, operations=None):
        super(BasicLatBlk, self).__init__()
        self.conv = operations.Conv2d(in_channels, out_channels, 1, 1, 0, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        return x


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding, device, dtype, operations):
        super(_ASPPModuleDeformable, self).__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, bias=False, device=device, dtype=dtype, operations=operations)
        self.bn = operations.BatchNorm2d(planes, device=device, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None, parallel_block_sizes=[1, 3, 7], device=None, dtype=None, operations=None):
        super(ASPPDeformable, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale

        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0, device=device, dtype=dtype, operations=operations)
        self.aspp_deforms = nn.ModuleList([
            _ASPPModuleDeformable(in_channels, self.in_channelster, conv_size, padding=int(conv_size//2), device=device, dtype=dtype, operations=operations)
              for conv_size in parallel_block_sizes
        ])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             operations.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False, device=device, dtype=dtype),
                                             operations.BatchNorm2d(self.in_channelster, device=device, dtype=dtype),
                                             nn.ReLU(inplace=True))
        self.conv1 = operations.Conv2d(self.in_channelster * (2 + len(self.aspp_deforms)), out_channels, 1, bias=False, device=device, dtype=dtype)
        self.bn1 = operations.BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

class BiRefNet(nn.Module):
    def __init__(self, config=None, dtype=None, device=None, operations=None):
        super(BiRefNet, self).__init__()
        self.bb = SwinTransformer(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12, device=device, dtype=dtype, operations=operations)

        channels = [1536, 768, 384, 192]
        channels = [c * 2 for c in channels]
        self.cxt = channels[1:][::-1][-3:]
        self.squeeze_module = nn.Sequential(*[
            BasicDecBlk(channels[0]+sum(self.cxt), channels[0], device=device, dtype=dtype, operations=operations)
            for _ in range(1)
        ])

        self.decoder = Decoder(channels, device=device, dtype=dtype, operations=operations)

    def forward_enc(self, x):
        x1, x2, x3, x4 = self.bb(x)
        B, C, H, W = x.shape
        x1_, x2_, x3_, x4_ = self.bb(F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=True))
        x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x4 = torch.cat(
            (
                *[
                    F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True),
                    F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                    F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                ][-len(CXT):],
                x4
            ),
            dim=1
        )
        return (x1, x2, x3, x4)

    def forward_ori(self, x):
        (x1, x2, x3, x4) = self.forward_enc(x)
        x4 = self.squeeze_module(x4)
        features = [x, x1, x2, x3, x4]
        scaled_preds = self.decoder(features)
        return scaled_preds

    def forward(self, pixel_values, intermediate_output=None):
        scaled_preds = self.forward_ori(pixel_values)
        return scaled_preds


class Decoder(nn.Module):
    def __init__(self, channels, device, dtype, operations):
        super(Decoder, self).__init__()
        # factory kwargs
        fk = {"device":device, "dtype":dtype, "operations":operations}
        DecoderBlock = partial(BasicDecBlk, **fk)
        LateralBlock = partial(BasicLatBlk, **fk)
        DBlock = partial(SimpleConvs, **fk)

        self.split = True
        N_dec_ipt = 64
        ic = 64
        ipt_cha_opt = 1
        self.ipt_blk5 = DBlock(2**10*3 if self.split else 3, [N_dec_ipt, channels[0]//8][ipt_cha_opt], inter_channels=ic)
        self.ipt_blk4 = DBlock(2**8*3 if self.split else 3, [N_dec_ipt, channels[0]//8][ipt_cha_opt], inter_channels=ic)
        self.ipt_blk3 = DBlock(2**6*3 if self.split else 3, [N_dec_ipt, channels[1]//8][ipt_cha_opt], inter_channels=ic)
        self.ipt_blk2 = DBlock(2**4*3 if self.split else 3, [N_dec_ipt, channels[2]//8][ipt_cha_opt], inter_channels=ic)
        self.ipt_blk1 = DBlock(2**0*3 if self.split else 3, [N_dec_ipt, channels[3]//8][ipt_cha_opt], inter_channels=ic)

        self.decoder_block4 = DecoderBlock(channels[0]+([N_dec_ipt, channels[0]//8][ipt_cha_opt]), channels[1])
        self.decoder_block3 = DecoderBlock(channels[1]+([N_dec_ipt, channels[0]//8][ipt_cha_opt]), channels[2])
        self.decoder_block2 = DecoderBlock(channels[2]+([N_dec_ipt, channels[1]//8][ipt_cha_opt]), channels[3])
        self.decoder_block1 = DecoderBlock(channels[3]+([N_dec_ipt, channels[2]//8][ipt_cha_opt]), channels[3]//2)

        fk = {"device":device, "dtype":dtype}

        self.conv_out1 = nn.Sequential(operations.Conv2d(channels[3]//2+([N_dec_ipt, channels[3]//8][ipt_cha_opt]), 1, 1, 1, 0, **fk))

        self.lateral_block4 = LateralBlock(channels[1], channels[1])
        self.lateral_block3 = LateralBlock(channels[2], channels[2])
        self.lateral_block2 = LateralBlock(channels[3], channels[3])

        self.conv_ms_spvn_4 = operations.Conv2d(channels[1], 1, 1, 1, 0, **fk)
        self.conv_ms_spvn_3 = operations.Conv2d(channels[2], 1, 1, 1, 0, **fk)
        self.conv_ms_spvn_2 = operations.Conv2d(channels[3], 1, 1, 1, 0, **fk)

        _N = 16

        self.gdt_convs_4 = nn.Sequential(operations.Conv2d(channels[0] // 2, _N, 3, 1, 1, **fk), operations.BatchNorm2d(_N, **fk), nn.ReLU(inplace=True))
        self.gdt_convs_3 = nn.Sequential(operations.Conv2d(channels[1] // 2, _N, 3, 1, 1, **fk), operations.BatchNorm2d(_N, **fk), nn.ReLU(inplace=True))
        self.gdt_convs_2 = nn.Sequential(operations.Conv2d(channels[2] // 2, _N, 3, 1, 1, **fk), operations.BatchNorm2d(_N, **fk), nn.ReLU(inplace=True))

        [setattr(self, f"gdt_convs_pred_{i}", nn.Sequential(operations.Conv2d(_N, 1, 1, 1, 0, **fk))) for i in range(2, 5)]
        [setattr(self, f"gdt_convs_attn_{i}", nn.Sequential(operations.Conv2d(_N, 1, 1, 1, 0, **fk))) for i in range(2, 5)]

    def get_patches_batch(self, x, p):
        _size_h, _size_w = p.shape[2:]
        patches_batch = []
        for idx in range(x.shape[0]):
            columns_x = torch.split(x[idx], split_size_or_sections=_size_w, dim=-1)
            patches_x = []
            for column_x in columns_x:
                patches_x += [p.unsqueeze(0) for p in torch.split(column_x, split_size_or_sections=_size_h, dim=-2)]
            patch_sample = torch.cat(patches_x, dim=1)
            patches_batch.append(patch_sample)
        return torch.cat(patches_batch, dim=0)

    def forward(self, features):
        x, x1, x2, x3, x4 = features

        patches_batch = self.get_patches_batch(x, x4) if self.split else x
        x4 = torch.cat((x4, self.ipt_blk5(F.interpolate(patches_batch, size=x4.shape[2:], mode='bilinear', align_corners=True))), 1)
        p4 = self.decoder_block4(x4)
        p4_gdt = self.gdt_convs_4(p4)
        gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
        p4 = p4 * gdt_attn_4
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)

        patches_batch = self.get_patches_batch(x, _p3) if self.split else x
        _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(patches_batch, size=x3.shape[2:], mode='bilinear', align_corners=True))), 1)
        p3 = self.decoder_block3(_p3)

        p3_gdt = self.gdt_convs_3(p3)
        gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
        p3 = p3 * gdt_attn_3
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)

        patches_batch = self.get_patches_batch(x, _p2) if self.split else x
        _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(patches_batch, size=x2.shape[2:], mode='bilinear', align_corners=True))), 1)
        p2 = self.decoder_block2(_p2)

        p2_gdt = self.gdt_convs_2(p2)
        gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
        p2 = p2 * gdt_attn_2

        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)

        patches_batch = self.get_patches_batch(x, _p1) if self.split else x
        _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(patches_batch, size=x1.shape[2:], mode='bilinear', align_corners=True))), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)

        patches_batch = self.get_patches_batch(x, _p1) if self.split else x
        _p1 = torch.cat((_p1, self.ipt_blk1(F.interpolate(patches_batch, size=x.shape[2:], mode='bilinear', align_corners=True))), 1)
        p1_out = self.conv_out1(_p1)
        return p1_out


class SimpleConvs(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, inter_channels=64, device=None, dtype=None, operations=None
    ) -> None:
        super().__init__()
        self.conv1 = operations.Conv2d(in_channels, inter_channels, 3, 1, 1, device=device, dtype=dtype)
        self.conv_out = operations.Conv2d(inter_channels, out_channels, 3, 1, 1, device=device, dtype=dtype)

    def forward(self, x):
        return self.conv_out(self.conv1(x))
