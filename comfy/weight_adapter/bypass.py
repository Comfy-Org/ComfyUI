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


class BypassForwardHook:
    """
    Hook that wraps a layer's forward to apply adapter in bypass mode.

    Stores the original forward and replaces it with bypass version.

    Supports both:
        - WeightAdapterBase: Inference adapters (uses self.weights tuple)
        - WeightAdapterTrainBase: Training adapters (nn.Module with parameters)
    """

    def __init__(
        self,
        module: nn.Module,
        adapter: BypassAdapter,
        multiplier: float = 1.0,
    ):
        self.module = module
        self.adapter = adapter
        self.multiplier = multiplier
        self.original_forward = None

        # Determine layer type and conv params from module class (works for quantized layers)
        module_info = get_module_type_info(module)

        # Set multiplier and layer type info on adapter for use in h()
        adapter.multiplier = multiplier
        adapter.is_conv = module_info["is_conv"]
        adapter.conv_dim = module_info["conv_dim"]
        adapter.kernel_size = module_info["kernel_size"]
        adapter.in_channels = module_info["in_channels"]
        adapter.out_channels = module_info["out_channels"]
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

    def _bypass_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Bypass forward: uses adapter's bypass_forward or default g(f(x) + h(x))

        Note:
            Bypass mode does NOT access original model weights (org_weight).
            This is intentional - bypass mode is designed for quantized models
            where weights may not be in a usable format. All necessary shape
            information is provided via adapter attributes set during inject().
        """
        # Check if adapter has custom bypass_forward (e.g., GLoRA)
        adapter_bypass = getattr(self.adapter, "bypass_forward", None)
        if adapter_bypass is not None:
            # Check if it's overridden (not the base class default)
            # Need to check both base classes since adapter could be either type
            adapter_type = type(self.adapter)
            is_default_bypass = (
                adapter_type.bypass_forward is WeightAdapterBase.bypass_forward
                or adapter_type.bypass_forward is WeightAdapterTrainBase.bypass_forward
            )
            if not is_default_bypass:
                return adapter_bypass(self.original_forward, x, *args, **kwargs)

        # Default bypass: g(f(x) + h(x, f(x)))
        base_out = self.original_forward(x, *args, **kwargs)
        h_out = self.adapter.h(x, base_out)
        return self.adapter.g(base_out + h_out)

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

        self._move_adapter_weights_to_device(device, dtype)

        self.original_forward = self.module.forward
        self.module.forward = self._bypass_forward
        logging.debug(
            f"[BypassHook] Injected bypass forward for {type(self.module).__name__} (adapter={type(self.adapter).__name__})"
        )

    def _move_adapter_weights_to_device(self, device, dtype=None):
        """Move adapter weights to specified device to avoid per-forward transfers.

        Handles both:
            - WeightAdapterBase: has self.weights tuple of tensors
            - WeightAdapterTrainBase: nn.Module with parameters, uses .to() method
        """
        adapter = self.adapter

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
        self.adapters: dict[str, tuple[BypassAdapter, float]] = {}
        self.hooks: list[BypassForwardHook] = []

    def add_adapter(
        self,
        key: str,
        adapter: BypassAdapter,
        strength: float = 1.0,
    ):
        """
        Add an adapter for a specific weight key.

        Args:
            key: Weight key (e.g., "model.layers.0.self_attn.q_proj.weight")
            adapter: The weight adapter (LoRAAdapter, LoKrAdapter, etc.)
            strength: Multiplier for adapter effect
        """
        # Remove .weight suffix if present for module lookup
        module_key = key
        if module_key.endswith(".weight"):
            module_key = module_key[:-7]
            logging.debug(
                f"[BypassManager] Stripped .weight suffix: {key} -> {module_key}"
            )

        self.adapters[module_key] = (adapter, strength)
        logging.debug(
            f"[BypassManager] Added adapter: {module_key} (type={type(adapter).__name__}, strength={strength})"
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

        for key, (adapter, strength) in self.adapters.items():
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

            logging.debug(
                f"[BypassManager] Creating hook for {key} (module type={type(module).__name__}, weight shape={module.weight.shape})"
            )
            hook = BypassForwardHook(module, adapter, multiplier=strength)
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

    def offload(self, model_patcher, unpatch_all):
        for hook in self.hooks:
            hook._move_adapter_weights_to_device(model_patcher.offload_device)

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

            combined_strength = strength * patch_strength
            manager.add_adapter(key, adapter, strength=combined_strength)

    return manager.create_injections(model)
