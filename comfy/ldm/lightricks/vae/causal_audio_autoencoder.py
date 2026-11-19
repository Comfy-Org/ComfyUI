import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from enum import Enum
from .pixel_norm import PixelNorm
import comfy.ops
import logging

ops = comfy.ops.disable_weight_init


class StringConvertibleEnum(Enum):
    """
    Base enum class that provides string-to-enum conversion functionality.

    This mixin adds a str_to_enum() class method that handles conversion from
    strings, None, or existing enum instances with case-insensitive matching.
    """

    @classmethod
    def str_to_enum(cls, value):
        """
        Convert a string, enum instance, or None to the appropriate enum member.

        Args:
            value: Can be an enum instance of this class, a string, or None

        Returns:
            Enum member of this class

        Raises:
            ValueError: If the value cannot be converted to a valid enum member
        """
        # Already an enum instance of this class
        if isinstance(value, cls):
            return value

        # None maps to NONE member if it exists
        if value is None:
            if hasattr(cls, "NONE"):
                return cls.NONE
            raise ValueError(f"{cls.__name__} does not have a NONE member to map None to")

        # String conversion (case-insensitive)
        if isinstance(value, str):
            value_lower = value.lower()

            # Try to match against enum values
            for member in cls:
                # Handle members with None values
                if member.value is None:
                    if value_lower == "none":
                        return member
                # Handle members with string values
                elif isinstance(member.value, str) and member.value.lower() == value_lower:
                    return member

            # Build helpful error message with valid values
            valid_values = []
            for member in cls:
                if member.value is None:
                    valid_values.append("none")
                elif isinstance(member.value, str):
                    valid_values.append(member.value)

            raise ValueError(f"Invalid {cls.__name__} string: '{value}'. " f"Valid values are: {valid_values}")

        raise ValueError(
            f"Cannot convert type {type(value).__name__} to {cls.__name__} enum. "
            f"Expected string, None, or {cls.__name__} instance."
        )


class AttentionType(StringConvertibleEnum):
    """Enum for specifying the attention mechanism type."""

    VANILLA = "vanilla"
    LINEAR = "linear"
    NONE = "none"


class CausalityAxis(StringConvertibleEnum):
    """Enum for specifying the causality axis in causal convolutions."""

    NONE = None
    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_COMPATIBILITY = "width-compatibility"


def Normalize(in_channels, *, num_groups=32, normtype="group"):
    if normtype == "group":
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif normtype == "pixel":
        return PixelNorm(dim=1, eps=1e-6)
    else:
        raise ValueError(f"Invalid normalization type: {normtype}")


class CausalConv2d(nn.Module):
    """
    A causal 2D convolution.

    This layer ensures that the output at time `t` only depends on inputs
    at time `t` and earlier. It achieves this by applying asymmetric padding
    to the time dimension (width) before the convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ):
        super().__init__()

        self.causality_axis = causality_axis

        # Ensure kernel_size and dilation are tuples
        kernel_size = nn.modules.utils._pair(kernel_size)
        dilation = nn.modules.utils._pair(dilation)

        # Calculate padding dimensions
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        # The padding tuple for F.pad is (pad_left, pad_right, pad_top, pad_bottom)
        match self.causality_axis:
            case CausalityAxis.NONE:
                self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            case CausalityAxis.WIDTH | CausalityAxis.WIDTH_COMPATIBILITY:
                self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
            case CausalityAxis.HEIGHT:
                self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
            case _:
                raise ValueError(f"Invalid causality_axis: {causality_axis}")

        # The internal convolution layer uses no padding, as we handle it manually
        self.conv = ops.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        # Apply causal padding before convolution
        x = F.pad(x, self.padding)
        return self.conv(x)


def make_conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=None,
    dilation=1,
    groups=1,
    bias=True,
    causality_axis: Optional[CausalityAxis] = None,
):
    """
    Create a 2D convolution layer that can be either causal or non-causal.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Convolution stride
        padding: Padding (if None, will be calculated based on causal flag)
        dilation: Dilation rate
        groups: Number of groups for grouped convolution
        bias: Whether to use bias
        causality_axis: Dimension along which to apply causality.

    Returns:
        Either a regular Conv2d or CausalConv2d layer
    """
    if causality_axis is not None:
        # For causal convolution, padding is handled internally by CausalConv2d
        return CausalConv2d(in_channels, out_channels, kernel_size, stride, dilation, groups, bias, causality_axis)
    else:
        # For non-causal convolution, use symmetric padding if not specified
        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = tuple(k // 2 for k in kernel_size)
        return ops.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, causality_axis: CausalityAxis = CausalityAxis.HEIGHT):
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = make_conv2d(in_channels, in_channels, kernel_size=3, stride=1, causality_axis=causality_axis)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            # Drop FIRST element in the causal axis to undo encoder's padding, while keeping the length 1 + 2 * n.
            # For example, if the input is [0, 1, 2], after interpolation, the output is [0, 0, 1, 1, 2, 2].
            # The causal convolution will pad the first element as [-, -, 0, 0, 1, 1, 2, 2],
            # So the output elements rely on the following windows:
            # 0: [-,-,0]
            # 1: [-,0,0]
            # 2: [0,0,1]
            # 3: [0,1,1]
            # 4: [1,1,2]
            # 5: [1,2,2]
            # Notice that the first and second elements in the output rely only on the first element in the input,
            # while all other elements rely on two elements in the input.
            # So we can drop the first element to undo the padding (rather than the last element).
            # This is a no-op for non-causal convolutions.
            match self.causality_axis:
                case CausalityAxis.NONE:
                    pass  # x remains unchanged
                case CausalityAxis.HEIGHT:
                    x = x[:, :, 1:, :]
                case CausalityAxis.WIDTH:
                    x = x[:, :, :, 1:]
                case CausalityAxis.WIDTH_COMPATIBILITY:
                    pass  # x remains unchanged
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

        return x


class Downsample(nn.Module):
    """
    A downsampling layer that can use either a strided convolution
    or average pooling. Supports standard and causal padding for the
    convolutional mode.
    """

    def __init__(self, in_channels, with_conv, causality_axis: CausalityAxis = CausalityAxis.WIDTH):
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and not self.with_conv:
            raise ValueError("causality is only supported when `with_conv=True`.")

        if self.with_conv:
            # Do time downsampling here
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = ops.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            # (pad_left, pad_right, pad_top, pad_bottom)
            match self.causality_axis:
                case CausalityAxis.NONE:
                    pad = (0, 1, 0, 1)
                case CausalityAxis.WIDTH:
                    pad = (2, 0, 0, 1)
                case CausalityAxis.HEIGHT:
                    pad = (0, 1, 2, 0)
                case CausalityAxis.WIDTH_COMPATIBILITY:
                    pad = (1, 0, 0, 1)
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # This branch is only taken if with_conv=False, which implies causality_axis is NONE.
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        norm_type="group",
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ):
        super().__init__()
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and norm_type == "group":
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, normtype=norm_type)
        self.non_linearity = nn.SiLU()
        self.conv1 = make_conv2d(in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis)
        if temb_channels > 0:
            self.temb_proj = ops.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, normtype=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = make_conv2d(out_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
                )
            else:
                self.nin_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, causality_axis=causality_axis
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, normtype=norm_type)
        self.q = ops.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = ops.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = ops.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = ops.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).contiguous()
        q = q.permute(0, 2, 1).contiguous()  # b,hw,c
        k = k.reshape(b, c, h * w).contiguous()  # b,c,hw
        w_ = torch.bmm(q, k).contiguous()  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w).contiguous()
        w_ = w_.permute(0, 2, 1).contiguous()  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_).contiguous()  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w).contiguous()

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla", norm_type="group"):
    # Convert string to enum if needed
    attn_type = AttentionType.str_to_enum(attn_type)

    if attn_type != AttentionType.NONE:
        logging.info(f"making attention of type '{attn_type.value}' with {in_channels} in_channels")
    else:
        logging.info(f"making identity attention with {in_channels} in_channels")

    match attn_type:
        case AttentionType.VANILLA:
            return AttnBlock(in_channels, norm_type=norm_type)
        case AttentionType.NONE:
            return nn.Identity(in_channels)
        case AttentionType.LINEAR:
            raise NotImplementedError(f"Attention type {attn_type.value} is not supported yet.")
        case _:
            raise ValueError(f"Unknown attention type: {attn_type}")


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        attn_type="vanilla",
        mid_block_add_attention=True,
        norm_type="group",
        causality_axis=CausalityAxis.WIDTH.value,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.double_z = double_z
        self.norm_type = norm_type
        # Convert string to enum if needed (for config loading)
        causality_axis = CausalityAxis.str_to_enum(causality_axis)
        self.attn_type = AttentionType.str_to_enum(attn_type)

        # downsampling
        self.conv_in = make_conv2d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )

        self.non_linearity = nn.SiLU()

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=causality_axis,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=self.attn_type, norm_type=self.norm_type))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv, causality_axis=causality_axis)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=causality_axis,
        )
        if mid_block_add_attention:
            self.mid.attn_1 = make_attn(block_in, attn_type=self.attn_type, norm_type=self.norm_type)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=causality_axis,
        )

        # end
        self.norm_out = Normalize(block_in, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape [batch, channels, time, n_mels]

        Returns:
            Encoded latent representation
        """
        feature_maps = [self.conv_in(x)]

        # Process each resolution level (from high to low resolution)
        for resolution_level in range(self.num_resolutions):
            # Apply residual blocks at current resolution level
            for block_idx in range(self.num_res_blocks):
                # Apply ResNet block with optional timestep embedding
                current_features = self.down[resolution_level].block[block_idx](feature_maps[-1], temb=None)

                # Apply attention if configured for this resolution level
                if len(self.down[resolution_level].attn) > 0:
                    current_features = self.down[resolution_level].attn[block_idx](current_features)

                # Store processed features
                feature_maps.append(current_features)

            # Downsample spatial dimensions (except at the final resolution level)
            if resolution_level != self.num_resolutions - 1:
                downsampled_features = self.down[resolution_level].downsample(feature_maps[-1])
                feature_maps.append(downsampled_features)

        # === MIDDLE PROCESSING PHASE ===
        # Take the lowest resolution features for middle processing
        bottleneck_features = feature_maps[-1]

        # Apply first middle ResNet block
        bottleneck_features = self.mid.block_1(bottleneck_features, temb=None)

        # Apply middle attention block
        bottleneck_features = self.mid.attn_1(bottleneck_features)

        # Apply second middle ResNet block
        bottleneck_features = self.mid.block_2(bottleneck_features, temb=None)

        # === OUTPUT PHASE ===
        # Normalize the bottleneck features
        output_features = self.norm_out(bottleneck_features)

        # Apply non-linearity (SiLU activation)
        output_features = self.non_linearity(output_features)

        # Final convolution to produce latent representation
        # [batch, channels, time, n_mels] -> [batch, 2 * z_channels if double_z else z_channels, time, n_mels]
        return self.conv_out(output_features)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        attn_type="vanilla",
        mid_block_add_attention=True,
        norm_type="group",
        causality_axis=CausalityAxis.WIDTH.value,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.norm_type = norm_type
        self.z_channels = z_channels
        # Convert string to enum if needed (for config loading)
        causality_axis = CausalityAxis.str_to_enum(causality_axis)
        self.attn_type = AttentionType.str_to_enum(attn_type)

        # compute block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = make_conv2d(z_channels, block_in, kernel_size=3, stride=1, causality_axis=causality_axis)

        self.non_linearity = nn.SiLU()

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=causality_axis,
        )
        if mid_block_add_attention:
            self.mid.attn_1 = make_attn(block_in, attn_type=self.attn_type, norm_type=self.norm_type)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=causality_axis,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=causality_axis,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=self.attn_type, norm_type=self.norm_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, causality_axis=causality_axis)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, normtype=self.norm_type)
        self.conv_out = make_conv2d(block_in, out_ch, kernel_size=3, stride=1, causality_axis=causality_axis)

    def _adjust_output_shape(self, decoded_output, target_shape):
        """
        Adjust output shape to match target dimensions for variable-length audio.

        This function handles the common case where decoded audio spectrograms need to be
        resized to match a specific target shape.

        Args:
            decoded_output: Tensor of shape (batch, channels, time, frequency)
            target_shape: Target shape tuple (batch, channels, time, frequency)

        Returns:
            Tensor adjusted to match target_shape exactly
        """
        # Current output shape: (batch, channels, time, frequency)
        _, _, current_time, current_freq = decoded_output.shape
        _, target_channels, target_time, target_freq = target_shape

        # Step 1: Crop first to avoid exceeding target dimensions
        decoded_output = decoded_output[
            :, :target_channels, : min(current_time, target_time), : min(current_freq, target_freq)
        ]

        # Step 2: Calculate padding needed for time and frequency dimensions
        time_padding_needed = target_time - decoded_output.shape[2]
        freq_padding_needed = target_freq - decoded_output.shape[3]

        # Step 3: Apply padding if needed
        if time_padding_needed > 0 or freq_padding_needed > 0:
            # PyTorch padding format: (pad_left, pad_right, pad_top, pad_bottom)
            # For audio: pad_left/right = frequency, pad_top/bottom = time
            padding = (
                0,
                max(freq_padding_needed, 0),  # frequency padding (left, right)
                0,
                max(time_padding_needed, 0),  # time padding (top, bottom)
            )
            decoded_output = F.pad(decoded_output, padding)

        # Step 4: Final safety crop to ensure exact target shape
        decoded_output = decoded_output[:, :target_channels, :target_time, :target_freq]

        return decoded_output

    def get_config(self):
        return {
            "ch": self.ch,
            "out_ch": self.out_ch,
            "ch_mult": self.ch_mult,
            "num_res_blocks": self.num_res_blocks,
            "in_channels": self.in_channels,
            "resolution": self.resolution,
            "z_channels": self.z_channels,
        }

    def forward(self, latent_features, target_shape=None):
        """
        Decode latent features back to audio spectrograms.

        Args:
            latent_features: Encoded latent representation of shape (batch, channels, height, width)
            target_shape: Optional target output shape (batch, channels, time, frequency)
                         If provided, output will be cropped/padded to match this shape

        Returns:
            Reconstructed audio spectrogram of shape (batch, channels, time, frequency)
        """
        assert target_shape is not None, "Target shape is required for CausalAudioAutoencoder Decoder"

        # Transform latent features to decoder's internal feature dimension
        hidden_features = self.conv_in(latent_features)

        # Middle processing
        hidden_features = self.mid.block_1(hidden_features, temb=None)
        hidden_features = self.mid.attn_1(hidden_features)
        hidden_features = self.mid.block_2(hidden_features, temb=None)

        # Upsampling
        # Progressively increase spatial resolution from lowest to highest
        for resolution_level in reversed(range(self.num_resolutions)):
            # Apply residual blocks at current resolution level
            for block_index in range(self.num_res_blocks + 1):
                hidden_features = self.up[resolution_level].block[block_index](hidden_features, temb=None)

                if len(self.up[resolution_level].attn) > 0:
                    hidden_features = self.up[resolution_level].attn[block_index](hidden_features)

            if resolution_level != 0:
                hidden_features = self.up[resolution_level].upsample(hidden_features)

        # Output
        if self.give_pre_end:
            # Return intermediate features before final processing (for debugging/analysis)
            decoded_output = hidden_features
        else:
            # Standard output path: normalize, activate, and convert to output channels
            # Final normalization layer
            hidden_features = self.norm_out(hidden_features)

            # Apply SiLU (Swish) activation function
            hidden_features = self.non_linearity(hidden_features)

            # Final convolution to map to output channels (typically 2 for stereo audio)
            decoded_output = self.conv_out(hidden_features)

            # Optional tanh activation to bound output values to [-1, 1] range
            if self.tanh_out:
                decoded_output = torch.tanh(decoded_output)

        # Adjust shape for audio data
        if target_shape is not None:
            decoded_output = self._adjust_output_shape(decoded_output, target_shape)

        return decoded_output


class processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(128))
        self.register_buffer("mean-of-means", torch.empty(128))

    def un_normalize(self, x):
        return (x * self.get_buffer("std-of-means").to(x)) + self.get_buffer("mean-of-means").to(x)

    def normalize(self, x):
        return (x - self.get_buffer("mean-of-means").to(x)) / self.get_buffer("std-of-means").to(x)


class CausalAudioAutoencoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = self.get_default_config()

        model_config = config.get("model", {}).get("params", {})

        self.sampling_rate = model_config.get(
            "sampling_rate", config.get("sampling_rate", 16000)
        )
        encoder_config = model_config.get("encoder", model_config.get("ddconfig", {}))
        decoder_config = model_config.get("decoder", encoder_config)

        # Load mel spectrogram parameters
        self.mel_bins = encoder_config.get("mel_bins", 64)
        self.mel_hop_length = config.get("preprocessing", {}).get("stft", {}).get("hop_length", 160)
        self.n_fft = config.get("preprocessing", {}).get("stft", {}).get("filter_length", 1024)

        # Store causality configuration at VAE level (not just in encoder internals)
        causality_axis_value = encoder_config.get("causality_axis", CausalityAxis.HEIGHT.value)
        self.causality_axis = CausalityAxis.str_to_enum(causality_axis_value)
        self.is_causal = self.causality_axis == CausalityAxis.HEIGHT

        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)

        self.per_channel_statistics = processor()

    def get_default_config(self):
        ddconfig = {
            "double_z": True,
            "mel_bins": 64,
            "z_channels": 8,
            "resolution": 256,
            "downsample_time": False,
            "in_channels": 2,
            "out_ch": 2,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
            "mid_block_add_attention": False,
            "norm_type": "pixel",
            "causality_axis": "height",
        }

        config = {
            "model": {
                "params": {
                    "ddconfig": ddconfig,
                    "sampling_rate": 16000,
                }
            },
            "preprocessing": {
                "stft": {
                    "filter_length": 1024,
                    "hop_length": 160,
                },
            },
        }

        return config

    def get_config(self):
        return {
            "sampling_rate": self.sampling_rate,
            "mel_bins": self.mel_bins,
            "mel_hop_length": self.mel_hop_length,
            "n_fft": self.n_fft,
            "causality_axis": self.causality_axis.value,
            "is_causal": self.is_causal,
        }

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, target_shape=None):
        return self.decoder(x, target_shape=target_shape)
