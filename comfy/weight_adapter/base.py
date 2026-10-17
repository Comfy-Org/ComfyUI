from typing import Callable, Optional

import torch
import torch.nn as nn

import comfy.model_management


class WeightAdapterBase:
    """
    Base class for weight adapters (LoRA, LoHa, LoKr, OFT, etc.)

    Bypass Mode:
        All adapters follow the pattern: bypass(f)(x) = g(f(x) + h(x))

        - h(x): Additive component (LoRA path). Returns delta to add to base output.
        - g(y): Output transformation. Applied after base + h(x).

        For LoRA/LoHa/LoKr: g = identity, h = adapter(x)
        For OFT/BOFT: g = transform, h = 0
    """

    name: str
    loaded_keys: set[str]
    weights: list[torch.Tensor]

    # Attributes set by bypass system
    multiplier: float = 1.0
    shape: tuple = None  # (out_features, in_features) or (out_ch, in_ch, *kernel)

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor,
    ) -> Optional["WeightAdapterBase"]:
        raise NotImplementedError

    def to_train(self) -> "WeightAdapterTrainBase":
        raise NotImplementedError

    @classmethod
    def create_train(cls, weight, *args) -> "WeightAdapterTrainBase":
        """
        weight: The original weight tensor to be modified.
        *args: Additional arguments for configuration, such as rank, alpha etc.
        """
        raise NotImplementedError

    def calculate_shape(
        self,
        key
    ):
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
        raise NotImplementedError

    # ===== Bypass Mode Methods =====
    #
    # IMPORTANT: Bypass mode is designed for quantized models where original weights
    # may not be accessible in a usable format. Therefore, h() and bypass_forward()
    # do NOT take org_weight as a parameter. All necessary information (out_channels,
    # in_channels, conv params, etc.) is provided via attributes set by BypassForwardHook.

    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """
        Additive bypass component: h(x, base_out)

        Computes the adapter's contribution to be added to base forward output.
        For adapters that only transform output (OFT/BOFT), returns zeros.

        Note:
            This method does NOT access original model weights. Bypass mode is
            designed for quantized models where weights may not be in a usable format.
            All shape info comes from module attributes set by BypassForwardHook.

        Args:
            x: Input tensor
            base_out: Output from base forward f(x), can be used for shape reference

        Returns:
            Delta tensor to add to base output. Shape matches base output.

        Reference: LyCORIS LoConModule.bypass_forward_diff
        """
        # Default: no additive component (for OFT/BOFT)
        # Simply return zeros matching base_out shape
        return torch.zeros_like(base_out)

    def g(self, y: torch.Tensor) -> torch.Tensor:
        """
        Output transformation: g(y)

        Applied after base forward + h(x). For most adapters this is identity.
        OFT/BOFT override this to apply orthogonal transformation.

        Args:
            y: Combined output (base + h(x))

        Returns:
            Transformed output

        Reference: LyCORIS OFTModule applies orthogonal transform here
        """
        # Default: identity (for LoRA/LoHa/LoKr)
        return y

    def bypass_forward(
        self,
        org_forward: Callable,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full bypass forward: g(f(x) + h(x, f(x)))

        Note:
            This method does NOT take org_weight/org_bias parameters. Bypass mode
            is designed for quantized models where weights may not be accessible.
            The original forward function handles weight access internally.

        Args:
            org_forward: Original module forward function
            x: Input tensor
            *args, **kwargs: Additional arguments for org_forward

        Returns:
            Output with adapter applied in bypass mode

        Reference: LyCORIS LoConModule.bypass_forward
        """
        # Base forward: f(x)
        base_out = org_forward(x, *args, **kwargs)

        # Additive component: h(x, base_out) - base_out provided for shape reference
        h_out = self.h(x, base_out)

        # Output transformation: g(base + h)
        return self.g(base_out + h_out)


class WeightAdapterTrainBase(nn.Module):
    """
    Base class for trainable weight adapters (LoRA, LoHa, LoKr, OFT, etc.)

    Bypass Mode:
        All adapters follow the pattern: bypass(f)(x) = g(f(x) + h(x))

        - h(x): Additive component (LoRA path). Returns delta to add to base output.
        - g(y): Output transformation. Applied after base + h(x).

        For LoRA/LoHa/LoKr: g = identity, h = adapter(x)
        For OFT: g = transform, h = 0

    Note:
        Unlike WeightAdapterBase, TrainBase classes have simplified weight formats
        with fewer branches (e.g., LoKr only has w1/w2, not w1_a/w1_b decomposition).

    We follow the scheme of PR #7032
    """

    # Attributes set by bypass system (BypassForwardHook)
    # These are set before h()/g()/bypass_forward() are called
    multiplier: float = 1.0
    is_conv: bool = False
    conv_dim: int = 0  # 0=linear, 1=conv1d, 2=conv2d, 3=conv3d
    kw_dict: dict = {}  # Conv kwargs: stride, padding, dilation, groups
    kernel_size: tuple = ()
    in_channels: int = None
    out_channels: int = None

    def __init__(self):
        super().__init__()

    def __call__(self, w):
        """
        Weight modification mode: returns modified weight.

        Args:
            w: The original weight tensor to be modified.

        Returns:
            Modified weight tensor.
        """
        raise NotImplementedError

    # ===== Bypass Mode Methods =====

    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """
        Additive bypass component: h(x, base_out)

        Computes the adapter's contribution to be added to base forward output.
        For adapters that only transform output (OFT), returns zeros.

        Args:
            x: Input tensor
            base_out: Output from base forward f(x), can be used for shape reference

        Returns:
            Delta tensor to add to base output. Shape matches base output.

        Subclasses should override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.h() not implemented. "
            "Subclasses must implement h() for bypass mode."
        )

    def g(self, y: torch.Tensor) -> torch.Tensor:
        """
        Output transformation: g(y)

        Applied after base forward + h(x). For most adapters this is identity.
        OFT overrides this to apply orthogonal transformation.

        Args:
            y: Combined output (base + h(x))

        Returns:
            Transformed output
        """
        # Default: identity (for LoRA/LoHa/LoKr)
        return y

    def bypass_forward(
        self,
        org_forward: Callable,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full bypass forward: g(f(x) + h(x, f(x)))

        Args:
            org_forward: Original module forward function
            x: Input tensor
            *args, **kwargs: Additional arguments for org_forward

        Returns:
            Output with adapter applied in bypass mode
        """
        # Base forward: f(x)
        base_out = org_forward(x, *args, **kwargs)

        # Additive component: h(x, base_out) - base_out provided for shape reference
        h_out = self.h(x, base_out)

        # Output transformation: g(base + h)
        return self.g(base_out + h_out)

    def passive_memory_usage(self):
        raise NotImplementedError("passive_memory_usage is not implemented")

    def move_to(self, device):
        self.to(device)
        return self.passive_memory_usage()


def weight_decompose(
    dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function
):
    dora_scale = comfy.model_management.cast_to_device(
        dora_scale, weight.device, intermediate_dtype
    )
    lora_diff *= alpha
    weight_calc = weight + function(lora_diff).type(weight.dtype)

    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]
    if wd_on_output_axis:
        weight_norm = (
            weight_calc.reshape(weight_calc.shape[0], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight_calc.shape[0], *[1] * (weight_calc.dim() - 1))
        )
    else:
        weight_norm = (
            weight_calc.transpose(0, 1)
            .reshape(weight_calc.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
            .transpose(0, 1)
        )
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * (weight_calc)
    else:
        weight[:] = weight_calc
    return weight


def pad_tensor_to_shape(tensor: torch.Tensor, new_shape: list[int]) -> torch.Tensor:
    """
    Pad a tensor to a new shape with zeros.

    Args:
        tensor (torch.Tensor): The original tensor to be padded.
        new_shape (List[int]): The desired shape of the padded tensor.

    Returns:
        torch.Tensor: A new tensor padded with zeros to the specified shape.

    Note:
        If the new shape is smaller than the original tensor in any dimension,
        the original tensor will be truncated in that dimension.
    """
    if any([new_shape[i] < tensor.shape[i] for i in range(len(new_shape))]):
        raise ValueError(
            "The new shape must be larger than the original tensor in all dimensions"
        )

    if len(new_shape) != len(tensor.shape):
        raise ValueError(
            "The new shape must have the same number of dimensions as the original tensor"
        )

    # Create a new tensor filled with zeros
    padded_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    # Create slicing tuples for both tensors
    orig_slices = tuple(slice(0, dim) for dim in tensor.shape)
    new_slices = tuple(slice(0, dim) for dim in tensor.shape)

    # Copy the original tensor into the new tensor
    padded_tensor[new_slices] = tensor[orig_slices]

    return padded_tensor


def tucker_weight_from_conv(up, down, mid):
    up = up.reshape(up.size(0), up.size(1))
    down = down.reshape(down.size(0), down.size(1))
    return torch.einsum("m n ..., i m, n j -> i j ...", mid, up, down)


def tucker_weight(wa, wb, t):
    temp = torch.einsum("i j ..., j r -> i r ...", t, wb)
    return torch.einsum("i j ..., i r -> r j ...", temp, wa)


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 1, 127   127 -> 1, 127    127 -> 1, 127   127 -> 1, 127   127 -> 1, 127
    128 -> 8, 16    128 -> 2, 64     128 -> 4, 32    128 -> 8, 16    128 -> 8, 16
    250 -> 10, 25   250 -> 2, 125    250 -> 2, 125   250 -> 5, 50    250 -> 10, 25
    360 -> 8, 45    360 -> 2, 180    360 -> 4, 90    360 -> 8, 45    360 -> 12, 30
    512 -> 16, 32   512 -> 2, 256    512 -> 4, 128   512 -> 8, 64    512 -> 16, 32
    1024 -> 32, 32  1024 -> 2, 512   1024 -> 4, 256  1024 -> 8, 128  1024 -> 16, 64
    """

    if factor > 0 and (dimension % factor) == 0 and dimension >= factor**2:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n
