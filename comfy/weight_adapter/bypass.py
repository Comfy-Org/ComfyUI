"""
Bypass mode implementation for weight adapters (LoRA, LoKr, LoHa, etc.)

Bypass mode applies adapters during forward pass without modifying base weights:
    bypass(f)(x) = g(f(x) + h(x))

Where:
    - f(x): Original layer forward
    - h(x): Additive component from adapter (LoRA path)
    - g(y): Output transformation (identity for most adapters)

This is useful for:
    - Training with gradient checkpointing
    - Avoiding weight modifications when weights are offloaded
    - Supporting multiple adapters with different strengths dynamically
"""

import functools
import logging
from typing import Optional, Union

import torch
import torch.nn as nn

import comfy.model_management
from .base import WeightAdapterBase, WeightAdapterTrainBase
from comfy.patcher_extension import PatcherInjection

# Type alias for adapters that support bypass mode
BypassAdapter = Union[WeightAdapterBase, WeightAdapterTrainBase]


def get_module_type_info(module: nn.Module) -> dict:
    """
    Determine module type and extract conv parameters from module class.

    This is more reliable than checking weight.ndim, especially for quantized layers
    where weight shape might be different.

    Returns:
        dict with keys: is_conv, conv_dim, stride, padding, dilation, groups
    """
    info = {
        "is_conv": False,
        "conv_dim": 0,
        "stride": (1,),
        "padding": (0,),
        "dilation": (1,),
        "groups": 1,
        "kernel_size": (1,),
        "in_channels": None,
        "out_channels": None,
    }

    # Determine conv type
    if isinstance(module, nn.Conv1d):
        info["is_conv"] = True
        info["conv_dim"] = 1
    elif isinstance(module, nn.Conv2d):
        info["is_conv"] = True
        info["conv_dim"] = 2
    elif isinstance(module, nn.Conv3d):
        info["is_conv"] = True
        info["conv_dim"] = 3
    elif isinstance(module, nn.Linear):
        info["is_conv"] = False
        info["conv_dim"] = 0
    else:
        # Try to infer from class name for custom/quantized layers
        class_name = type(module).__name__.lower()
        if "conv3d" in class_name:
            info["is_conv"] = True
            info["conv_dim"] = 3
        elif "conv2d" in class_name:
            info["is_conv"] = True
            info["conv_dim"] = 2
        elif "conv1d" in class_name:
            info["is_conv"] = True
            info["conv_dim"] = 1
        elif "conv" in class_name:
            info["is_conv"] = True
            info["conv_dim"] = 2

    # Extract conv parameters if it's a conv layer
    if info["is_conv"]:
        # Try to get stride, padding, dilation, groups, kernel_size from module
        info["stride"] = getattr(module, "stride", (1,) * info["conv_dim"])
        info["padding"] = getattr(module, "padding", (0,) * info["conv_dim"])
        info["dilation"] = getattr(module, "dilation", (1,) * info["conv_dim"])
        info["groups"] = getattr(module, "groups", 1)
        info["kernel_size"] = getattr(module, "kernel_size", (1,) * info["conv_dim"])
        info["in_channels"] = getattr(module, "in_channels", None)
        info["out_channels"] = getattr(module, "out_channels", None)

        # Ensure they're tuples
        if isinstance(info["stride"], int):
            info["stride"] = (info["stride"],) * info["conv_dim"]
        if isinstance(info["padding"], int):
            info["padding"] = (info["padding"],) * info["conv_dim"]
        if isinstance(info["dilation"], int):
            info["dilation"] = (info["dilation"],) * info["conv_dim"]
        if isinstance(info["kernel_size"], int):
            info["kernel_size"] = (info["kernel_size"],) * info["conv_dim"]

    return info


def has_custom_bypass(adapter: BypassAdapter) -> bool:
    """Whether the adapter overrides bypass_forward, i.e. rewrites the module input (GLoRA)."""
    bypass = type(adapter).bypass_forward
    return (
        bypass is not WeightAdapterBase.bypass_forward
        and bypass is not WeightAdapterTrainBase.bypass_forward
    )


def is_transposed_conv(module: nn.Module) -> bool:
    """Whether the module is a transposed conv, by class or, for custom/quantized conv layers, by a
    class name holding both "conv" and "transpose"; linear layers never are."""
    if isinstance(module, nn.modules.conv._ConvTransposeNd):
        return True
    if isinstance(module, nn.Linear):
        return False
    name = type(module).__name__.lower()
    return "conv" in name and "transpose" in name


def output_features(module: nn.Module) -> Optional[int]:
    """Output features (Linear) or channels (conv) of a module, from its attributes or weight shape."""
    info = get_module_type_info(module)
    if info["out_channels"] is not None:
        return info["out_channels"]
    out = getattr(module, "out_features", None)
    if out is None and getattr(module, "weight", None) is not None:
        out = module.weight.shape[0]
    return out


class BypassForwardHook:
    """
    Hook that wraps a layer's forward to apply its adapters in bypass mode.

    Stores the original forward and replaces it with the bypass version. A
    module holds one hook with an ordered list of entries (adapter,
    multiplier, offset); offset (0, start, size) restricts an adapter to that
    slice of the output features (e.g. q, k or v of a fused qkv linear), None
    applies it to the whole output.

    Supports both:
        - WeightAdapterBase: Inference adapters (uses self.weights tuple)
        - WeightAdapterTrainBase: Training adapters (nn.Module with parameters)
    """

    def __init__(
        self,
        module: nn.Module,
        adapter: BypassAdapter,
        multiplier: float = 1.0,
        offset: Optional[tuple] = None,
    ):
        self.module = module
        self.entries: list[tuple[BypassAdapter, float, Optional[tuple]]] = []
        self.original_forward = None

        # Determine layer type and conv params from module class (works for quantized layers)
        self.module_info = get_module_type_info(module)
        # Output features are dim 1 of a conv output, the last dim of a linear output
        self.channel_dim = 1 if self.module_info["is_conv"] else -1

        self.add(adapter, multiplier, offset)
        # First entry, as before entries existed
        self.adapter = adapter
        self.multiplier = multiplier

    def add(self, adapter: BypassAdapter, multiplier: float = 1.0, offset: Optional[tuple] = None):
        """Append an adapter; entries apply in the order they were added."""
        module_info = self.module_info

        # Set multiplier and layer type info on adapter for use in h()
        adapter.multiplier = multiplier
        adapter.is_conv = module_info["is_conv"]
        adapter.conv_dim = module_info["conv_dim"]
        adapter.kernel_size = module_info["kernel_size"]
        adapter.in_channels = module_info["in_channels"]
        # A sliced adapter sees only its slice of the output
        adapter.out_channels = offset[2] if offset is not None else module_info["out_channels"]
        # Store kw_dict for conv operations (like LyCORIS extra_args)
        if module_info["is_conv"]:
            adapter.kw_dict = {
                "stride": module_info["stride"],
                "padding": module_info["padding"],
                "dilation": module_info["dilation"],
                "groups": module_info["groups"],
            }
        else:
            adapter.kw_dict = {}

        self.entries.append((adapter, multiplier, offset))

    def _bypass_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Bypass forward: f(x) with input-rewriting adapters applied, then for each
        additive/transform adapter in order out = g(out + h(x)) on its slice of out.

        Note:
            Bypass mode does NOT access original model weights (org_weight).
            This is intentional - bypass mode is designed for quantized models
            where weights may not be in a usable format. All necessary shape
            information is provided via adapter attributes set during inject().
        """
        # Adapters with a custom bypass_forward (e.g., GLoRA) wrap the module forward itself
        forward = self.original_forward
        for adapter, _, offset in self.entries:
            if offset is None and has_custom_bypass(adapter):
                forward = functools.partial(adapter.bypass_forward, forward)
        out = forward(x, *args, **kwargs)

        # Default bypass per entry: g(f(x) + h(x, f(x))) on the whole output or its slice
        for adapter, _, offset in self.entries:
            if has_custom_bypass(adapter):
                continue
            if offset is None:
                out = adapter.g(out + adapter.h(x, out))
                continue
            _, start, size = offset
            part = out.narrow(self.channel_dim, start, size)
            part = adapter.g(part + adapter.h(x, part))
            out = out.slice_scatter(
                part, dim=self.channel_dim, start=start, end=start + size
            )
        return out

    def inject(self):
        """Replace module forward with bypass version."""
        if self.original_forward is not None:
            logging.debug(
                f"[BypassHook] Already injected for {type(self.module).__name__}"
            )
            return  # Already injected

        # Move adapter weights to compute device (GPU)
        # Use get_torch_device() instead of module.weight.device because
        # with offloading, module weights may be on CPU while compute happens on GPU
        device = comfy.model_management.get_torch_device()

        # Get dtype from module weight if available
        dtype = None
        if hasattr(self.module, "weight") and self.module.weight is not None:
            dtype = self.module.weight.dtype

        # Only use dtype if it's a standard float type, not quantized
        if dtype is not None and dtype not in (torch.float32, torch.float16, torch.bfloat16):
            dtype = None

        for adapter, _, _ in self.entries:
            self._move_adapter_weights_to_device(adapter, device, dtype)

        self.original_forward = self.module.forward
        self.original_force_forward = getattr(self.module, "comfy_force_forward", False)
        self.module.comfy_force_forward = True
        self.module.forward = self._bypass_forward
        logging.debug(
            f"[BypassHook] Injected bypass forward for {type(self.module).__name__} "
            f"(adapters={[type(a).__name__ for a, _, _ in self.entries]})"
        )

    @staticmethod
    def _move_adapter_weights_to_device(adapter, device, dtype=None):
        """Move adapter weights to specified device to avoid per-forward transfers.

        Handles both:
            - WeightAdapterBase: has self.weights tuple of tensors
            - WeightAdapterTrainBase: nn.Module with parameters, uses .to() method
        """
        # Check if adapter is an nn.Module (WeightAdapterTrainBase)
        if isinstance(adapter, nn.Module):
            # In training mode we don't touch dtype as trainer will handle it
            adapter.to(device=device)
            logging.debug(
                f"[BypassHook] Moved training adapter (nn.Module) to {device}"
            )
            return

        # WeightAdapterBase: handle self.weights tuple
        if not hasattr(adapter, "weights") or adapter.weights is None:
            return

        weights = adapter.weights
        if isinstance(weights, (list, tuple)):
            new_weights = []
            for w in weights:
                if isinstance(w, torch.Tensor):
                    if dtype is not None:
                        new_weights.append(w.to(device=device, dtype=dtype))
                    else:
                        new_weights.append(w.to(device=device))
                else:
                    new_weights.append(w)
            adapter.weights = (
                tuple(new_weights) if isinstance(weights, tuple) else new_weights
            )
        elif isinstance(weights, torch.Tensor):
            if dtype is not None:
                adapter.weights = weights.to(device=device, dtype=dtype)
            else:
                adapter.weights = weights.to(device=device)

        logging.debug(f"[BypassHook] Moved adapter weights to {device}")

    def eject(self):
        """Restore original module forward."""
        if self.original_forward is None:
            logging.debug(f"[BypassHook] Not injected for {type(self.module).__name__}")
            return  # Not injected

        self.module.forward = self.original_forward
        self.module.comfy_force_forward = self.original_force_forward
        self.original_forward = None
        logging.debug(
            f"[BypassHook] Ejected bypass forward for {type(self.module).__name__}"
        )


class BypassInjectionManager:
    """
    Manages bypass mode injection for a collection of adapters.

    Creates PatcherInjection objects that can be used with ModelPatcher.

    Supports both inference adapters (WeightAdapterBase) and training adapters
    (WeightAdapterTrainBase).

    Usage:
        manager = BypassInjectionManager()
        manager.add_adapter("model.layers.0.self_attn.q_proj", lora_adapter, strength=0.8)
        manager.add_adapter("model.layers.0.self_attn.k_proj", lora_adapter, strength=0.8)

        injections = manager.create_injections(model)
        model_patcher.set_injections("bypass_lora", injections)
    """

    def __init__(self):
        # module key -> [(adapter, strength, offset)] in the order added
        self.adapters: dict[str, list[tuple[BypassAdapter, float, Optional[tuple]]]] = {}
        self.hooks: list[BypassForwardHook] = []

    def add_adapter(
        self,
        key: str,
        adapter: BypassAdapter,
        strength: float = 1.0,
        offset: Optional[tuple] = None,
    ):
        """
        Add an adapter for a specific weight key.

        Args:
            key: Weight key (e.g., "model.layers.0.self_attn.q_proj.weight")
            adapter: The weight adapter (LoRAAdapter, LoKrAdapter, etc.)
            strength: Multiplier for adapter effect
            offset: (dim, start, size) slice of the weight the adapter targets
                (e.g. q of a fused qkv), or None for the whole weight; only
                dim 0 (output features) is supported
        """
        if offset is not None:
            if offset[1] < 0 or offset[2] <= 0:
                raise ValueError(
                    f"[BypassManager] {key}: slice {offset} needs a non-negative start and a positive size"
                )
            if offset[0] != 0:
                raise ValueError(
                    f"[BypassManager] {key}: adapter targets a slice of weight dim {offset[0]}; "
                    "bypass mode supports output-dim (dim 0) slices only"
                )
            if has_custom_bypass(adapter):
                raise ValueError(
                    f"[BypassManager] {key}: {type(adapter).__name__} rewrites the module input "
                    f"and cannot apply to the output slice {offset}"
                )

        # Remove .weight suffix if present for module lookup
        module_key = key
        if module_key.endswith(".weight"):
            module_key = module_key[:-7]
            logging.debug(
                f"[BypassManager] Stripped .weight suffix: {key} -> {module_key}"
            )

        self.adapters.setdefault(module_key, []).append((adapter, strength, offset))
        logging.debug(
            f"[BypassManager] Added adapter: {module_key} (type={type(adapter).__name__}, "
            f"strength={strength}, offset={offset})"
        )

    def clear_adapters(self):
        """Remove all adapters."""
        self.adapters.clear()

    def _get_module_by_key(self, model: nn.Module, key: str) -> Optional[nn.Module]:
        """Get a submodule by dot-separated key."""
        parts = key.split(".")
        module = model
        try:
            for i, part in enumerate(parts):
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            logging.debug(
                f"[BypassManager] Found module for key {key}: {type(module).__name__}"
            )
            return module
        except (AttributeError, IndexError, KeyError) as e:
            logging.error(f"[BypassManager] Failed to find module for key {key}: {e}")
            logging.error(
                f"[BypassManager] Failed at part index {i}, part={part}, current module type={type(module).__name__}"
            )
            return None

    def create_injections(self, model: nn.Module) -> list[PatcherInjection]:
        """
        Create PatcherInjection objects for all registered adapters.

        Args:
            model: The model to inject into (e.g., model_patcher.model)

        Returns:
            List of PatcherInjection objects to use with model_patcher.set_injections()
        """
        self.hooks.clear()

        logging.debug(
            f"[BypassManager] create_injections called with {len(self.adapters)} adapters"
        )
        logging.debug(f"[BypassManager] Model type: {type(model).__name__}")

        for key, entries in self.adapters.items():
            logging.debug(f"[BypassManager] Looking for module: {key}")
            module = self._get_module_by_key(model, key)

            if module is None:
                logging.warning(f"[BypassManager] Module not found for key {key}")
                continue

            if not hasattr(module, "weight"):
                logging.warning(
                    f"[BypassManager] Module {key} has no weight attribute (type={type(module).__name__})"
                )
                continue

            features = output_features(module)
            for _, _, offset in entries:
                if offset is not None and is_transposed_conv(module):
                    raise ValueError(
                        f"[BypassManager] {key}: weight dim 0 of a transposed conv is its input channels; "
                        f"the slice {offset} has no output-channel form"
                    )
                if offset is not None and features is not None and offset[1] + offset[2] > features:
                    raise ValueError(
                        f"[BypassManager] {key}: slice {offset} exceeds the module's {features} output features"
                    )

            logging.debug(
                f"[BypassManager] Creating hook for {key} (module type={type(module).__name__}, "
                f"weight shape={module.weight.shape}, adapters={len(entries)})"
            )
            (adapter, strength, offset), *rest = entries
            hook = BypassForwardHook(module, adapter, multiplier=strength, offset=offset)
            for adapter, strength, offset in rest:
                hook.add(adapter, strength, offset)
            self.hooks.append(hook)

        logging.debug(f"[BypassManager] Created {len(self.hooks)} hooks")

        # Create single injection that manages all hooks
        def inject_all(model_patcher):
            logging.debug(
                f"[BypassManager] inject_all called, injecting {len(self.hooks)} hooks"
            )
            for hook in self.hooks:
                hook.inject()
                logging.debug(
                    f"[BypassManager] Injected hook for {type(hook.module).__name__}"
                )

        def eject_all(model_patcher):
            logging.debug(
                f"[BypassManager] eject_all called, ejecting {len(self.hooks)} hooks"
            )
            for hook in self.hooks:
                hook.eject()

        return [PatcherInjection(inject=inject_all, eject=eject_all)]

    def get_hook_count(self) -> int:
        """Return number of hooks that will be/are injected."""
        return len(self.hooks)


def create_bypass_injections_from_patches(
    model: nn.Module,
    patches: dict,
    strength: float = 1.0,
) -> list[PatcherInjection]:
    """
    Convenience function to create bypass injections from a patches dict.

    This is useful when you have patches in the format used by model_patcher.add_patches()
    and want to apply them in bypass mode instead.

    Args:
        model: The model to inject into
        patches: Dict mapping weight keys to adapter data
        strength: Global strength multiplier

    Returns:
        List of PatcherInjection objects
    """
    manager = BypassInjectionManager()

    for key, patch_list in patches.items():
        if not patch_list:
            continue

        # patches format: list of (strength_patch, patch_data, strength_model, offset, function)
        for patch in patch_list:
            patch_strength, patch_data, strength_model, offset, function = patch

            # patch_data should be a WeightAdapterBase/WeightAdapterTrainBase or tuple
            if isinstance(patch_data, (WeightAdapterBase, WeightAdapterTrainBase)):
                adapter = patch_data
            else:
                # Skip non-adapter patches
                continue
            if function is not None:
                logging.warning(f"[BypassManager] {key}: a weight function has no bypass form; adapter skipped")
                continue

            combined_strength = strength * patch_strength
            manager.add_adapter(key, adapter, strength=combined_strength, offset=offset)

    return manager.create_injections(model)
