import logging
from typing import Optional

import torch
import torch.nn.functional as F
import comfy.model_management
from .base import (
    WeightAdapterBase,
    WeightAdapterTrainBase,
    weight_decompose,
    pad_tensor_to_shape,
    tucker_weight_from_conv,
)


class LoraDiff(WeightAdapterTrainBase):
    def __init__(self, weights):
        super().__init__()
        mat1, mat2, alpha, mid, dora_scale, reshape = weights
        out_dim, rank = mat1.shape[0], mat1.shape[1]
        rank, in_dim = mat2.shape[0], mat2.shape[1]
        if mid is not None:
            convdim = mid.ndim - 2
            layer = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)[convdim]
        else:
            layer = torch.nn.Linear
        self.lora_up = layer(rank, out_dim, bias=False)
        self.lora_down = layer(in_dim, rank, bias=False)
        self.lora_up.weight.data.copy_(mat1)
        self.lora_down.weight.data.copy_(mat2)
        if mid is not None:
            self.lora_mid = layer(mid, rank, bias=False)
            self.lora_mid.weight.data.copy_(mid)
        else:
            self.lora_mid = None
        self.rank = rank
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def __call__(self, w):
        org_dtype = w.dtype
        if self.lora_mid is None:
            diff = self.lora_up.weight @ self.lora_down.weight
        else:
            diff = tucker_weight_from_conv(
                self.lora_up.weight, self.lora_down.weight, self.lora_mid.weight
            )
        scale = self.alpha / self.rank
        weight = w + scale * diff.reshape(w.shape)
        return weight.to(org_dtype)

    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """
        Additive bypass component for LoRA training: h(x) = up(down(x)) * scale

        Simple implementation using the nn.Module weights directly.
        No mid/dora/reshape branches (create_train doesn't create them).

        Args:
            x: Input tensor
            base_out: Output from base forward (unused, for API consistency)
        """
        # Compute scale = alpha / rank * multiplier
        scale = (self.alpha / self.rank) * getattr(self, "multiplier", 1.0)

        # Get module info from bypass injection
        is_conv = getattr(self, "is_conv", False)
        conv_dim = getattr(self, "conv_dim", 0)
        kw_dict = getattr(self, "kw_dict", {})

        # Get weights (keep in original dtype for numerical stability)
        down_weight = self.lora_down.weight
        up_weight = self.lora_up.weight

        if is_conv:
            # Conv path: use functional conv
            # conv_dim: 1=conv1d, 2=conv2d, 3=conv3d
            conv_fn = (F.conv1d, F.conv2d, F.conv3d)[conv_dim - 1]

            # Reshape 2D weights to conv format if needed
            # down: [rank, in_features] -> [rank, in_channels, *kernel_size]
            # up: [out_features, rank] -> [out_features, rank, 1, 1, ...]
            if down_weight.dim() == 2:
                kernel_size = getattr(self, "kernel_size", (1,) * conv_dim)
                in_channels = getattr(self, "in_channels", None)
                if in_channels is not None:
                    down_weight = down_weight.view(
                        down_weight.shape[0], in_channels, *kernel_size
                    )
                else:
                    # Fallback: assume 1x1 kernel
                    down_weight = down_weight.view(
                        *down_weight.shape, *([1] * conv_dim)
                    )
            if up_weight.dim() == 2:
                # up always uses 1x1 kernel
                up_weight = up_weight.view(*up_weight.shape, *([1] * conv_dim))

            # down conv uses stride/padding from module, up is 1x1
            hidden = conv_fn(x, down_weight, **kw_dict)

            # mid layer if exists (tucker decomposition)
            if self.lora_mid is not None:
                mid_weight = self.lora_mid.weight
                if mid_weight.dim() == 2:
                    mid_weight = mid_weight.view(*mid_weight.shape, *([1] * conv_dim))
                hidden = conv_fn(hidden, mid_weight)

            # up conv is always 1x1 (no stride/padding)
            out = conv_fn(hidden, up_weight)
        else:
            # Linear path: simple matmul chain
            hidden = F.linear(x, down_weight)

            # mid layer if exists
            if self.lora_mid is not None:
                mid_weight = self.lora_mid.weight
                hidden = F.linear(hidden, mid_weight)

            out = F.linear(hidden, up_weight)

        return out * scale

    def passive_memory_usage(self):
        return sum(param.numel() * param.element_size() for param in self.parameters())


class LoRAAdapter(WeightAdapterBase):
    name = "lora"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def create_train(cls, weight, rank=1, alpha=1.0):
        out_dim = weight.shape[0]
        in_dim = weight.shape[1:].numel()
        mat1 = torch.empty(out_dim, rank, device=weight.device, dtype=torch.float32)
        mat2 = torch.empty(rank, in_dim, device=weight.device, dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(mat1, a=5**0.5)
        torch.nn.init.constant_(mat2, 0.0)
        return LoraDiff((mat1, mat2, alpha, None, None, None))

    def to_train(self):
        return LoraDiff(self.weights)

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor,
        loaded_keys: set[str] = None,
    ) -> Optional["LoRAAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()

        reshape_name = "{}.reshape_weight".format(x)
        regular_lora = "{}.lora_up.weight".format(x)
        diffusers_lora = "{}_lora.up.weight".format(x)
        diffusers2_lora = "{}.lora_B.weight".format(x)
        diffusers3_lora = "{}.lora.up.weight".format(x)
        mochi_lora = "{}.lora_B".format(x)
        transformers_lora = "{}.lora_linear_layer.up.weight".format(x)
        qwen_default_lora = "{}.lora_B.default.weight".format(x)
        A_name = None

        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            mid_name = "{}.lora_mid.weight".format(x)
        elif diffusers_lora in lora.keys():
            A_name = diffusers_lora
            B_name = "{}_lora.down.weight".format(x)
            mid_name = None
        elif diffusers2_lora in lora.keys():
            A_name = diffusers2_lora
            B_name = "{}.lora_A.weight".format(x)
            mid_name = None
        elif diffusers3_lora in lora.keys():
            A_name = diffusers3_lora
            B_name = "{}.lora.down.weight".format(x)
            mid_name = None
        elif mochi_lora in lora.keys():
            A_name = mochi_lora
            B_name = "{}.lora_A".format(x)
            mid_name = None
        elif transformers_lora in lora.keys():
            A_name = transformers_lora
            B_name = "{}.lora_linear_layer.down.weight".format(x)
            mid_name = None
        elif qwen_default_lora in lora.keys():
            A_name = qwen_default_lora
            B_name = "{}.lora_A.default.weight".format(x)
            mid_name = None

        if A_name is not None:
            mid = None
            if mid_name is not None and mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            reshape = None
            if reshape_name in lora.keys():
                try:
                    reshape = lora[reshape_name].tolist()
                    loaded_keys.add(reshape_name)
                except:
                    pass
            weights = (lora[A_name], lora[B_name], alpha, mid, dora_scale, reshape)
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)
            return cls(loaded_keys, weights)
        else:
            return None

    def calculate_shape(
        self,
        key
    ):
        reshape = self.weights[5]
        return tuple(reshape) if reshape is not None else None

    def calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=torch.float32,
        original_weight=None,
    ):
        v = self.weights
        mat1 = comfy.model_management.cast_to_device(
            v[0], weight.device, intermediate_dtype
        )
        mat2 = comfy.model_management.cast_to_device(
            v[1], weight.device, intermediate_dtype
        )
        dora_scale = v[4]
        reshape = v[5]

        if reshape is not None:
            weight = pad_tensor_to_shape(weight, reshape)

        if v[2] is not None:
            alpha = v[2] / mat2.shape[0]
        else:
            alpha = 1.0

        if v[3] is not None:
            # locon mid weights, hopefully the math is fine because I didn't properly test it
            mat3 = comfy.model_management.cast_to_device(
                v[3], weight.device, intermediate_dtype
            )
            final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
            mat2 = (
                torch.mm(
                    mat2.transpose(0, 1).flatten(start_dim=1),
                    mat3.transpose(0, 1).flatten(start_dim=1),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )
        try:
            lora_diff = torch.mm(
                mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)
            ).reshape(weight.shape)
            del mat1, mat2
            if dora_scale is not None:
                weight = weight_decompose(
                    dora_scale,
                    weight,
                    lora_diff,
                    alpha,
                    strength,
                    intermediate_dtype,
                    function,
                )
            else:
                weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight

    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """
        Additive bypass component for LoRA: h(x) = up(down(x)) * scale

        Note:
            Does not access original model weights - bypass mode is designed
            for quantized models where weights may not be accessible.

        Args:
            x: Input tensor
            base_out: Output from base forward (unused, for API consistency)

        Reference: LyCORIS functional/locon.py bypass_forward_diff
        """
        # FUNC_LIST: [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]
        FUNC_LIST = [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]

        v = self.weights
        # v[0]=up, v[1]=down, v[2]=alpha, v[3]=mid, v[4]=dora_scale, v[5]=reshape
        up = v[0]
        down = v[1]
        alpha = v[2]
        mid = v[3]

        # Compute scale = alpha / rank
        rank = down.shape[0]
        if alpha is not None:
            scale = alpha / rank
        else:
            scale = 1.0
        scale = scale * getattr(self, "multiplier", 1.0)

        # Cast dtype
        up = up.to(dtype=x.dtype)
        down = down.to(dtype=x.dtype)

        # Use module info from bypass injection, not weight dimension
        is_conv = getattr(self, "is_conv", False)
        conv_dim = getattr(self, "conv_dim", 0)
        kw_dict = getattr(self, "kw_dict", {})

        if is_conv:
            op = FUNC_LIST[
                conv_dim + 2
            ]  # conv_dim 1->conv1d(3), 2->conv2d(4), 3->conv3d(5)
            kernel_size = getattr(self, "kernel_size", (1,) * conv_dim)
            in_channels = getattr(self, "in_channels", None)

            # Reshape 2D weights to conv format using kernel_size
            # down: [rank, in_channels * prod(kernel_size)] -> [rank, in_channels, *kernel_size]
            # up: [out_channels, rank] -> [out_channels, rank, 1, 1, ...] (1x1 kernel)
            if down.dim() == 2:
                # down.shape[1] = in_channels * prod(kernel_size)
                if in_channels is not None:
                    down = down.view(down.shape[0], in_channels, *kernel_size)
                else:
                    # Fallback: assume 1x1 kernel if in_channels unknown
                    down = down.view(*down.shape, *([1] * conv_dim))
            if up.dim() == 2:
                # up always uses 1x1 kernel
                up = up.view(*up.shape, *([1] * conv_dim))
            if mid is not None:
                mid = mid.to(dtype=x.dtype)
                if mid.dim() == 2:
                    mid = mid.view(*mid.shape, *([1] * conv_dim))
        else:
            op = F.linear
            kw_dict = {}  # linear doesn't take stride/padding

        # Simple chain: down -> mid (if tucker) -> up
        if mid is not None:
            if not is_conv:
                mid = mid.to(dtype=x.dtype)
            hidden = op(x, down)
            hidden = op(hidden, mid, **kw_dict)
            out = op(hidden, up)
        else:
            hidden = op(x, down, **kw_dict)
            out = op(hidden, up)

        if (not torch.is_grad_enabled() and type(out) is torch.Tensor
                and (not torch.is_tensor(scale) or scale.ndim == 0)
                and torch.result_type(out, scale) == out.dtype):
            return out.mul_(scale)
        return out * scale

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        # The native h() owns its projection output; base output may alias caller state.
        if torch.is_grad_enabled() or getattr(self.h, "__func__", None) is not LoRAAdapter.h:
            return super().bypass_forward(org_forward, x, *args, **kwargs)

        base_out = org_forward(x, *args, **kwargs)
        h_out = self.h(x, base_out)
        if (type(base_out) is torch.Tensor and type(h_out) is torch.Tensor
                and h_out.shape == base_out.shape and h_out.dtype == base_out.dtype
                and h_out.stride() == base_out.stride()):
            torch.add(base_out, h_out, out=h_out)
            del base_out
            return self.g(h_out)
        return self.g(base_out + h_out)
