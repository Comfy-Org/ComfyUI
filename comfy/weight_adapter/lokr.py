import logging
from typing import Optional

import torch
import torch.nn.functional as F
import comfy.model_management
from .base import (
    WeightAdapterBase,
    WeightAdapterTrainBase,
    weight_decompose,
    factorization,
)


class LokrDiff(WeightAdapterTrainBase):
    def __init__(self, weights):
        super().__init__()
        (
            lokr_w1,
            lokr_w2,
            alpha,
            lokr_w1_a,
            lokr_w1_b,
            lokr_w2_a,
            lokr_w2_b,
            lokr_t2,
            dora_scale,
        ) = weights
        self.use_tucker = False
        if lokr_w1_a is not None:
            _, rank_a = lokr_w1_a.shape[0], lokr_w1_a.shape[1]
            rank_a, _ = lokr_w1_b.shape[0], lokr_w1_b.shape[1]
            self.lokr_w1_a = torch.nn.Parameter(lokr_w1_a)
            self.lokr_w1_b = torch.nn.Parameter(lokr_w1_b)
            self.w1_rebuild = True
            self.ranka = rank_a

        if lokr_w2_a is not None:
            _, rank_b = lokr_w2_a.shape[0], lokr_w2_a.shape[1]
            rank_b, _ = lokr_w2_b.shape[0], lokr_w2_b.shape[1]
            self.lokr_w2_a = torch.nn.Parameter(lokr_w2_a)
            self.lokr_w2_b = torch.nn.Parameter(lokr_w2_b)
            if lokr_t2 is not None:
                self.use_tucker = True
                self.lokr_t2 = torch.nn.Parameter(lokr_t2)
            self.w2_rebuild = True
            self.rankb = rank_b

        if lokr_w1 is not None:
            self.lokr_w1 = torch.nn.Parameter(lokr_w1)
            self.w1_rebuild = False

        if lokr_w2 is not None:
            self.lokr_w2 = torch.nn.Parameter(lokr_w2)
            self.w2_rebuild = False

        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)

    @property
    def w1(self):
        if self.w1_rebuild:
            return (self.lokr_w1_a @ self.lokr_w1_b) * (self.alpha / self.ranka)
        else:
            return self.lokr_w1

    @property
    def w2(self):
        if self.w2_rebuild:
            if self.use_tucker:
                w2 = torch.einsum(
                    "i j k l, j r, i p -> p r k l",
                    self.lokr_t2,
                    self.lokr_w2_b,
                    self.lokr_w2_a,
                )
            else:
                w2 = self.lokr_w2_a @ self.lokr_w2_b
            return w2 * (self.alpha / self.rankb)
        else:
            return self.lokr_w2

    def __call__(self, w):
        w1 = self.w1
        w2 = self.w2
        # Unsqueeze w1 to match w2 dims for proper kron product (like LyCORIS make_kron)
        for _ in range(w2.dim() - w1.dim()):
            w1 = w1.unsqueeze(-1)
        diff = torch.kron(w1, w2)
        return w + diff.reshape(w.shape).to(w)

    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """
        Additive bypass component for LoKr training: efficient Kronecker product.

        Uses w1/w2 properties which handle both direct and decomposed cases.
        For create_train (direct w1/w2), no alpha scaling in properties.
        For to_train (decomposed), alpha/rank scaling is in properties.

        Args:
            x: Input tensor
            base_out: Output from base forward (unused, for API consistency)
        """
        # Get w1, w2 from properties (handles rebuild vs direct)
        w1 = self.w1
        w2 = self.w2

        # Multiplier from bypass injection
        multiplier = getattr(self, "multiplier", 1.0)

        # Get module info from bypass injection
        is_conv = getattr(self, "is_conv", False)
        conv_dim = getattr(self, "conv_dim", 0)
        kw_dict = getattr(self, "kw_dict", {})

        # Efficient Kronecker application without materializing full weight
        # kron(w1, w2) @ x can be computed as nested operations
        # w1: [out_l, in_m], w2: [out_k, in_n, *k_size]
        # Full weight would be [out_l*out_k, in_m*in_n, *k_size]

        uq = w1.size(1)  # in_m - inner grouping dimension

        if is_conv:
            conv_fn = (F.conv1d, F.conv2d, F.conv3d)[conv_dim - 1]

            B, C_in, *spatial = x.shape
            # Reshape input for grouped application: [B * uq, C_in // uq, *spatial]
            h_in_group = x.reshape(B * uq, -1, *spatial)

            # Ensure w2 has conv dims
            if w2.dim() == 2:
                w2 = w2.view(*w2.shape, *([1] * conv_dim))

            # Apply w2 path with stride/padding
            hb = conv_fn(h_in_group, w2, **kw_dict)

            # Reshape for cross-group operation
            hb = hb.view(B, -1, *hb.shape[1:])
            h_cross = hb.transpose(1, -1)

            # Apply w1 (always 2D, applied as linear on channel dim)
            hc = F.linear(h_cross, w1)
            hc = hc.transpose(1, -1)

            # Reshape to output
            out = hc.reshape(B, -1, *hc.shape[3:])
        else:
            # Linear case
            # Reshape input: [..., in_m * in_n] -> [..., uq (in_m), in_n]
            h_in_group = x.reshape(*x.shape[:-1], uq, -1)

            # Apply w2: [..., uq, in_n] @ [out_k, in_n].T -> [..., uq, out_k]
            hb = F.linear(h_in_group, w2)

            # Transpose for w1: [..., uq, out_k] -> [..., out_k, uq]
            h_cross = hb.transpose(-1, -2)

            # Apply w1: [..., out_k, uq] @ [out_l, uq].T -> [..., out_k, out_l]
            hc = F.linear(h_cross, w1)

            # Transpose back and flatten: [..., out_k, out_l] -> [..., out_l * out_k]
            hc = hc.transpose(-1, -2)
            out = hc.reshape(*hc.shape[:-2], -1)

        return out * multiplier

    def passive_memory_usage(self):
        return sum(param.numel() * param.element_size() for param in self.parameters())


class LoKrAdapter(WeightAdapterBase):
    name = "lokr"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def create_train(cls, weight, rank=1, alpha=1.0):
        out_dim = weight.shape[0]
        in_dim = weight.shape[1]  # Just in_channels, not flattened with kernel
        k_size = weight.shape[2:] if weight.dim() > 2 else ()

        out_l, out_k = factorization(out_dim, rank)
        in_m, in_n = factorization(in_dim, rank)

        # w1: [out_l, in_m]
        mat1 = torch.empty(out_l, in_m, device=weight.device, dtype=torch.float32)
        # w2: [out_k, in_n, *k_size] for conv, [out_k, in_n] for linear
        mat2 = torch.empty(
            out_k, in_n, *k_size, device=weight.device, dtype=torch.float32
        )

        torch.nn.init.kaiming_uniform_(mat2, a=5**0.5)
        torch.nn.init.constant_(mat1, 0.0)
        return LokrDiff((mat1, mat2, alpha, None, None, None, None, None, None))

    def to_train(self):
        return LokrDiff(self.weights)

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor,
        loaded_keys: set[str] = None,
    ) -> Optional["LoKrAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()
        lokr_w1_name = "{}.lokr_w1".format(x)
        lokr_w2_name = "{}.lokr_w2".format(x)
        lokr_w1_a_name = "{}.lokr_w1_a".format(x)
        lokr_w1_b_name = "{}.lokr_w1_b".format(x)
        lokr_t2_name = "{}.lokr_t2".format(x)
        lokr_w2_a_name = "{}.lokr_w2_a".format(x)
        lokr_w2_b_name = "{}.lokr_w2_b".format(x)

        lokr_w1 = None
        if lokr_w1_name in lora.keys():
            lokr_w1 = lora[lokr_w1_name]
            loaded_keys.add(lokr_w1_name)

        lokr_w2 = None
        if lokr_w2_name in lora.keys():
            lokr_w2 = lora[lokr_w2_name]
            loaded_keys.add(lokr_w2_name)

        lokr_w1_a = None
        if lokr_w1_a_name in lora.keys():
            lokr_w1_a = lora[lokr_w1_a_name]
            loaded_keys.add(lokr_w1_a_name)

        lokr_w1_b = None
        if lokr_w1_b_name in lora.keys():
            lokr_w1_b = lora[lokr_w1_b_name]
            loaded_keys.add(lokr_w1_b_name)

        lokr_w2_a = None
        if lokr_w2_a_name in lora.keys():
            lokr_w2_a = lora[lokr_w2_a_name]
            loaded_keys.add(lokr_w2_a_name)

        lokr_w2_b = None
        if lokr_w2_b_name in lora.keys():
            lokr_w2_b = lora[lokr_w2_b_name]
            loaded_keys.add(lokr_w2_b_name)

        lokr_t2 = None
        if lokr_t2_name in lora.keys():
            lokr_t2 = lora[lokr_t2_name]
            loaded_keys.add(lokr_t2_name)

        if (
            (lokr_w1 is not None)
            or (lokr_w2 is not None)
            or (lokr_w1_a is not None)
            or (lokr_w2_a is not None)
        ):
            weights = (
                lokr_w1,
                lokr_w2,
                alpha,
                lokr_w1_a,
                lokr_w1_b,
                lokr_w2_a,
                lokr_w2_b,
                lokr_t2,
                dora_scale,
            )
            return cls(loaded_keys, weights)
        else:
            return None

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
        w1 = v[0]
        w2 = v[1]
        w1_a = v[3]
        w1_b = v[4]
        w2_a = v[5]
        w2_b = v[6]
        t2 = v[7]
        dora_scale = v[8]
        dim = None if w1 is None or w2 is None else w2.shape[0] // w1.shape[0]

        if w1 is None:
            dim = w1_b.shape[0]
            w1 = torch.mm(
                comfy.model_management.cast_to_device(
                    w1_a, weight.device, intermediate_dtype
                ),
                comfy.model_management.cast_to_device(
                    w1_b, weight.device, intermediate_dtype
                ),
            )
        else:
            w1 = comfy.model_management.cast_to_device(
                w1, weight.device, intermediate_dtype
            )

        if w2 is None:
            dim = w2_b.shape[0]
            if t2 is None:
                w2 = torch.mm(
                    comfy.model_management.cast_to_device(
                        w2_a, weight.device, intermediate_dtype
                    ),
                    comfy.model_management.cast_to_device(
                        w2_b, weight.device, intermediate_dtype
                    ),
                )
            else:
                w2 = torch.einsum(
                    "i j k l, j r, i p -> p r k l",
                    comfy.model_management.cast_to_device(
                        t2, weight.device, intermediate_dtype
                    ),
                    comfy.model_management.cast_to_device(
                        w2_b, weight.device, intermediate_dtype
                    ),
                    comfy.model_management.cast_to_device(
                        w2_a, weight.device, intermediate_dtype
                    ),
                )
        else:
            w2 = comfy.model_management.cast_to_device(
                w2, weight.device, intermediate_dtype
            )

        if len(w2.shape) == 4:
            w1 = w1.unsqueeze(2).unsqueeze(2)
        if v[2] is not None and dim is not None:
            alpha = v[2] / dim
        else:
            alpha = 1.0

        try:
            lora_diff = torch.kron(w1, w2).reshape(weight.shape)
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
        Additive bypass component for LoKr: efficient Kronecker product application.

        Note:
            Does not access original model weights - bypass mode is designed
            for quantized models where weights may not be accessible.

        Args:
            x: Input tensor
            base_out: Output from base forward (unused, for API consistency)

        Reference: LyCORIS functional/lokr.py bypass_forward_diff
        """
        # FUNC_LIST: [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]
        FUNC_LIST = [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]

        v = self.weights
        # v[0]=w1, v[1]=w2, v[2]=alpha, v[3]=w1_a, v[4]=w1_b, v[5]=w2_a, v[6]=w2_b, v[7]=t2, v[8]=dora
        w1 = v[0]
        w2 = v[1]
        alpha = v[2]
        w1_a = v[3]
        w1_b = v[4]
        w2_a = v[5]
        w2_b = v[6]
        t2 = v[7]

        use_w1 = w1 is not None
        use_w2 = w2 is not None
        tucker = t2 is not None

        # Use module info from bypass injection, not weight dimension
        is_conv = getattr(self, "is_conv", False)
        conv_dim = getattr(self, "conv_dim", 0)
        kw_dict = getattr(self, "kw_dict", {}) if is_conv else {}

        if is_conv:
            op = FUNC_LIST[conv_dim + 2]
        else:
            op = F.linear

        # Determine rank and scale
        if not use_w1:
            rank = w1_b.size(0)
        elif not use_w2:
            rank = w2_b.size(0)
        else:
            rank = w2.size(0) // w1.size(0)
        scale = (alpha / rank if alpha is not None else 1.0) * getattr(
            self, "multiplier", 1.0
        )

        # Build c (w1)
        if use_w1:
            c = w1.to(dtype=x.dtype)
        else:
            c = w1_a.to(dtype=x.dtype) @ w1_b.to(dtype=x.dtype)
        uq = c.size(1)

        # Build w2 components
        if use_w2:
            ba = w2.to(dtype=x.dtype)
        else:
            a = w2_b.to(dtype=x.dtype)
            b = w2_a.to(dtype=x.dtype)
            if is_conv:
                if tucker:
                    # Tucker: a, b get 1s appended (kernel is in t2)
                    if a.dim() == 2:
                        a = a.view(*a.shape, *([1] * conv_dim))
                    if b.dim() == 2:
                        b = b.view(*b.shape, *([1] * conv_dim))
                else:
                    # Non-tucker conv: b may need 1s appended
                    if b.dim() == 2:
                        b = b.view(*b.shape, *([1] * conv_dim))

        # Reshape input by uq groups
        if is_conv:
            B, _, *rest = x.shape
            h_in_group = x.reshape(B * uq, -1, *rest)
        else:
            h_in_group = x.reshape(*x.shape[:-1], uq, -1)

        # Apply w2 path
        if use_w2:
            hb = op(h_in_group, ba, **kw_dict)
        else:
            if is_conv:
                if tucker:
                    t = t2.to(dtype=x.dtype)
                    if t.dim() == 2:
                        t = t.view(*t.shape, *([1] * conv_dim))
                    ha = op(h_in_group, a)
                    ht = op(ha, t, **kw_dict)
                    hb = op(ht, b)
                else:
                    ha = op(h_in_group, a, **kw_dict)
                    hb = op(ha, b)
            else:
                ha = op(h_in_group, a)
                hb = op(ha, b)

        # Reshape and apply c (w1)
        if is_conv:
            hb = hb.view(B, -1, *hb.shape[1:])
            h_cross_group = hb.transpose(1, -1)
        else:
            h_cross_group = hb.transpose(-1, -2)

        hc = F.linear(h_cross_group, c)

        if is_conv:
            hc = hc.transpose(1, -1)
            out = hc.reshape(B, -1, *hc.shape[3:])
        else:
            hc = hc.transpose(-1, -2)
            out = hc.reshape(*hc.shape[:-2], -1)

        return out * scale
