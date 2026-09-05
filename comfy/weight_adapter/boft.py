import logging
from typing import Optional

import torch
import comfy.model_management
from .base import WeightAdapterBase, weight_decompose


class BOFTAdapter(WeightAdapterBase):
    name = "boft"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor,
        loaded_keys: set[str] = None,
    ) -> Optional["BOFTAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()
        blocks_name = "{}.oft_blocks".format(x)
        rescale_name = "{}.rescale".format(x)

        blocks = None
        if blocks_name in lora.keys():
            blocks = lora[blocks_name]
            if blocks.ndim == 4:
                loaded_keys.add(blocks_name)
            else:
                blocks = None
        if blocks is None:
            return None

        rescale = None
        if rescale_name in lora.keys():
            rescale = lora[rescale_name]
            loaded_keys.add(rescale_name)

        weights = (blocks, rescale, alpha, dora_scale)
        return cls(loaded_keys, weights)

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
        blocks = v[0]
        rescale = v[1]
        alpha = v[2]
        dora_scale = v[3]

        blocks = comfy.model_management.cast_to_device(
            blocks, weight.device, intermediate_dtype
        )
        if rescale is not None:
            rescale = comfy.model_management.cast_to_device(
                rescale, weight.device, intermediate_dtype
            )

        boft_m, block_num, boft_b, *_ = blocks.shape

        try:
            # Get r
            I = torch.eye(boft_b, device=blocks.device, dtype=blocks.dtype)
            # for Q = -Q^T
            q = blocks - blocks.transpose(-1, -2)
            normed_q = q
            # alpha in boft/bboft is the constraint, lycoris stores it unscaled
            constraint = alpha * block_num * boft_b
            if constraint > 0:
                q_norm = torch.norm(q) + 1e-8
                if q_norm > constraint:
                    normed_q = q * constraint / q_norm
            # use float() to prevent unsupported type in .inverse()
            r = (I + normed_q) @ (I - normed_q).float().inverse()
            r = r.to(weight)
            inp = org = weight

            r_b = boft_b // 2
            for i in range(boft_m):
                bi = r[i]
                g = 2
                k = 2**i * r_b
                if strength != 1:
                    bi = bi * strength + (1 - strength) * I
                inp = (
                    inp.unflatten(0, (-1, g, k))
                    .transpose(1, 2)
                    .flatten(0, 2)
                    .unflatten(0, (-1, boft_b))
                )
                inp = torch.einsum("b i j, b j ...-> b i ...", bi, inp)
                inp = (
                    inp.flatten(0, 1)
                    .unflatten(0, (-1, k, g))
                    .transpose(1, 2)
                    .flatten(0, 2)
                )

            if rescale is not None:
                inp = inp * rescale

            lora_diff = inp - org
            lora_diff = comfy.model_management.cast_to_device(
                lora_diff, weight.device, intermediate_dtype
            )
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
                weight += function((strength * lora_diff).type(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight

    def _get_orthogonal_matrices(self, device, dtype):
        """Compute the orthogonal rotation matrices R from BOFT blocks."""
        v = self.weights
        blocks = v[0].to(device=device, dtype=dtype)
        alpha = v[2]
        if alpha is None:
            alpha = 0

        boft_m, block_num, boft_b, _ = blocks.shape
        I = torch.eye(boft_b, device=device, dtype=dtype)

        # Q = blocks - blocks^T (skew-symmetric)
        q = blocks - blocks.transpose(-1, -2)
        normed_q = q

        # Apply constraint, lycoris stores it unscaled
        constraint = alpha * block_num * boft_b
        if constraint > 0:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > constraint:
                normed_q = q * constraint / q_norm

        # Cayley transform: R = (I + Q)(I - Q)^-1
        r = (I + normed_q) @ (I - normed_q).float().inverse()
        return r, boft_m, boft_b

    def g(self, y: torch.Tensor) -> torch.Tensor:
        """
        Output transformation for BOFT: applies butterfly orthogonal transform.

        BOFT uses multiple stages of butterfly-structured orthogonal transforms.

        Reference: LyCORIS ButterflyOFTModule._bypass_forward
        """
        v = self.weights
        rescale = v[1]

        r, boft_m, boft_b = self._get_orthogonal_matrices(y.device, y.dtype)
        r_b = boft_b // 2

        # Apply multiplier
        multiplier = getattr(self, "multiplier", 1.0)
        I = torch.eye(boft_b, device=y.device, dtype=y.dtype)

        # Use module info from bypass injection to determine conv vs linear
        is_conv = getattr(self, "is_conv", y.dim() > 2)

        if is_conv:
            # Conv output: (N, C, H, W, ...) -> transpose to (N, H, W, ..., C)
            y = y.transpose(1, -1)

        # Apply butterfly transform stages
        inp = y
        for i in range(boft_m):
            bi = r[i]  # (block_num, boft_b, boft_b)
            g = 2
            k = 2**i * r_b

            # Interpolate with identity based on multiplier
            if multiplier != 1:
                bi = bi * multiplier + (1 - multiplier) * I

            # Reshape for butterfly: unflatten last dim, transpose, flatten, unflatten
            inp = (
                inp.unflatten(-1, (-1, g, k))
                .transpose(-2, -1)
                .flatten(-3)
                .unflatten(-1, (-1, boft_b))
            )
            # Apply block-diagonal orthogonal transform
            inp = torch.einsum("b i j, ... b j -> ... b i", bi, inp)
            # Reshape back
            inp = (
                inp.flatten(-2).unflatten(-1, (-1, k, g)).transpose(-2, -1).flatten(-3)
            )

        # Apply rescale if present
        if rescale is not None:
            rescale = rescale.to(device=y.device, dtype=y.dtype)
            inp = inp * rescale.transpose(0, -1)

        if is_conv:
            # Transpose back: (N, H, W, ..., C) -> (N, C, H, W, ...)
            inp = inp.transpose(1, -1)

        return inp
