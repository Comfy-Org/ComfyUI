import logging
import os

import numpy as np
import safetensors
import torch
import torch.nn as nn
import torch.utils.checkpoint
from tqdm.auto import trange
from PIL import Image, ImageDraw, ImageFont
from typing_extensions import override

import comfy.samplers
import comfy.sampler_helpers
import comfy.sd
import comfy.utils
import comfy.model_management
from comfy.conds import CONDRegular, CONDList
from comfy.cli_args import args, PerformanceFeature
import comfy_extras.nodes_custom_sampler
import folder_paths
import node_helpers
from comfy.weight_adapter import adapters, adapter_maps
from comfy.weight_adapter.bypass import BypassInjectionManager
from comfy_api.latest import ComfyExtension, io, ui
from comfy.utils import ProgressBar


class TrainGuider(comfy_extras.nodes_custom_sampler.Guider_Basic):
    """
    CFGGuider with modifications for training specific logic
    """

    def __init__(self, *args, offloading=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.offloading = offloading

    def outer_sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
        latent_shapes=None,
    ):
        self.inner_model, self.conds, self.loaded_models = (
            comfy.sampler_helpers.prepare_sampling(
                self.model_patcher,
                noise.shape,
                self.conds,
                self.model_options,
                force_full_load=not self.offloading,
                force_offload=self.offloading,
            )
        )
        torch.cuda.empty_cache()
        device = self.model_patcher.load_device

        if denoise_mask is not None:
            denoise_mask = comfy.sampler_helpers.prepare_mask(
                denoise_mask, noise.shape, device
            )

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)
        comfy.samplers.cast_to_load_options(
            self.model_options, device=device, dtype=self.model_patcher.model_dtype()
        )

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(
                noise,
                latent_image,
                device,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        finally:
            self.model_patcher.cleanup()

        comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output


def make_batch_extra_option_dict(d, indicies, full_size=None):
    new_dict = {}
    for k, v in d.items():
        newv = v
        if isinstance(v, dict):
            newv = make_batch_extra_option_dict(v, indicies, full_size=full_size)
        elif isinstance(v, torch.Tensor):
            if full_size is None or v.size(0) == full_size:
                newv = v[indicies]
        elif isinstance(v, (list, tuple)) and len(v) == full_size:
            newv = [v[i] for i in indicies]
        new_dict[k] = newv
    return new_dict


def process_cond_list(d, prefix=""):
    if hasattr(d, "__iter__") and not hasattr(d, "items"):
        for index, item in enumerate(d):
            process_cond_list(item, f"{prefix}.{index}")
        return d
    elif hasattr(d, "items"):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                process_cond_list(v, f"{prefix}.{k}")
            elif isinstance(v, torch.Tensor):
                d[k] = v.clone()
            elif isinstance(v, CONDList):
                v.cond = [t.detach() if isinstance(t, torch.Tensor) else t for t in v.cond]
            elif isinstance(v, CONDRegular):
                if isinstance(v.cond, torch.Tensor):
                    v.cond = v.cond.detach()
            elif isinstance(v, (list, tuple)):
                for index, item in enumerate(v):
                    process_cond_list(item, f"{prefix}.{k}.{index}")
    return d


class TrainSampler(comfy.samplers.Sampler):
    def __init__(
        self,
        loss_fn,
        optimizer,
        loss_callback=None,
        batch_size=1,
        grad_acc=1,
        total_steps=1,
        seed=0,
        training_dtype=torch.bfloat16,
        real_dataset=None,
        bucket_latents=None,
        use_grad_scaler=False,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_callback = loss_callback
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.grad_acc = grad_acc
        self.seed = seed
        self.training_dtype = training_dtype
        self.real_dataset: list[torch.Tensor] | None = real_dataset
        # Bucket mode data
        self.bucket_latents: list[torch.Tensor] | None = (
            bucket_latents  # list of (Bi, C, Hi, Wi)
        )
        # GradScaler for fp16 training
        self.grad_scaler = torch.amp.GradScaler() if use_grad_scaler else None
        # Precompute bucket offsets and weights for sampling
        if bucket_latents is not None:
            self._init_bucket_data(bucket_latents)
        else:
            self.bucket_offsets = None
            self.bucket_weights = None
            self.num_images = None

    def _init_bucket_data(self, bucket_latents):
        """Initialize bucket offsets and weights for sampling."""
        self.bucket_offsets = [0]
        bucket_sizes = []
        for lat in bucket_latents:
            bucket_sizes.append(lat.shape[0])
            self.bucket_offsets.append(self.bucket_offsets[-1] + lat.shape[0])
        self.num_images = self.bucket_offsets[-1]
        # Weights for sampling buckets proportional to their size
        self.bucket_weights = torch.tensor(bucket_sizes, dtype=torch.float32)

    def fwd_bwd(
        self,
        model_wrap,
        batch_sigmas,
        batch_noise,
        batch_latent,
        cond,
        indicies,
        extra_args,
        dataset_size,
        bwd=True,
    ):
        xt = model_wrap.inner_model.model_sampling.noise_scaling(
            batch_sigmas, batch_noise, batch_latent, False
        )
        x0 = model_wrap.inner_model.model_sampling.noise_scaling(
            torch.zeros_like(batch_sigmas),
            torch.zeros_like(batch_noise),
            batch_latent,
            False,
        )

        model_wrap.conds["positive"] = [cond[i] for i in indicies]
        batch_extra_args = make_batch_extra_option_dict(
            extra_args, indicies, full_size=dataset_size
        )

        with torch.autocast(xt.device.type, dtype=self.training_dtype):
            x0_pred = model_wrap(
                xt.requires_grad_(True),
                batch_sigmas.requires_grad_(True),
                **batch_extra_args,
            )
            loss = self.loss_fn(x0_pred.float(), x0.float())
        if bwd:
            bwd_loss = loss / self.grad_acc
            if self.grad_scaler is not None:
                self.grad_scaler.scale(bwd_loss).backward()
            else:
                bwd_loss.backward()
        return loss

    def _generate_batch_sigmas(self, model_wrap, batch_size, device):
        """Generate random sigma values for a batch."""
        batch_sigmas = [
            model_wrap.inner_model.model_sampling.percent_to_sigma(
                torch.rand((1,)).item()
            )
            for _ in range(batch_size)
        ]
        return torch.tensor(batch_sigmas).to(device)

    def _train_step_bucket_mode(self, model_wrap, cond, extra_args, noisegen, latent_image, pbar):
        """Execute one training step in bucket mode."""
        # Sample bucket (weighted by size), then sample batch from bucket
        bucket_idx = torch.multinomial(self.bucket_weights, 1).item()
        bucket_latent = self.bucket_latents[bucket_idx]  # (Bi, C, Hi, Wi)
        bucket_size = bucket_latent.shape[0]
        bucket_offset = self.bucket_offsets[bucket_idx]

        # Sample indices from this bucket (use all if bucket_size < batch_size)
        actual_batch_size = min(self.batch_size, bucket_size)
        relative_indices = torch.randperm(bucket_size)[:actual_batch_size].tolist()
        # Convert to absolute indices for fwd_bwd (cond is flattened, use absolute index)
        absolute_indices = [bucket_offset + idx for idx in relative_indices]

        batch_latent = bucket_latent[relative_indices].to(latent_image)  # (actual_batch_size, C, H, W)
        batch_noise = noisegen.generate_noise({"samples": batch_latent}).to(
            batch_latent.device
        )
        batch_sigmas = self._generate_batch_sigmas(model_wrap, actual_batch_size, batch_latent.device)

        loss = self.fwd_bwd(
            model_wrap,
            batch_sigmas,
            batch_noise,
            batch_latent,
            cond,  # Use flattened cond with absolute indices
            absolute_indices,
            extra_args,
            self.num_images,
            bwd=True,
        )
        if self.loss_callback:
            self.loss_callback(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "bucket": bucket_idx})

    def _train_step_standard_mode(self, model_wrap, cond, extra_args, noisegen, latent_image, dataset_size, pbar):
        """Execute one training step in standard (non-bucket, non-multi-res) mode."""
        indicies = torch.randperm(dataset_size)[: self.batch_size].tolist()
        batch_latent = torch.stack([latent_image[i] for i in indicies])
        batch_noise = noisegen.generate_noise({"samples": batch_latent}).to(
            batch_latent.device
        )
        batch_sigmas = self._generate_batch_sigmas(model_wrap, min(self.batch_size, dataset_size), batch_latent.device)

        loss = self.fwd_bwd(
            model_wrap,
            batch_sigmas,
            batch_noise,
            batch_latent,
            cond,
            indicies,
            extra_args,
            dataset_size,
            bwd=True,
        )
        if self.loss_callback:
            self.loss_callback(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    def _train_step_multires_mode(self, model_wrap, cond, extra_args, noisegen, latent_image, dataset_size, pbar):
        """Execute one training step in multi-resolution mode (real_dataset is set)."""
        indicies = torch.randperm(dataset_size)[: self.batch_size].tolist()
        total_loss = 0
        for index in indicies:
            single_latent = self.real_dataset[index].to(latent_image)
            batch_noise = noisegen.generate_noise(
                {"samples": single_latent}
            ).to(single_latent.device)
            batch_sigmas = (
                model_wrap.inner_model.model_sampling.percent_to_sigma(
                    torch.rand((1,)).item()
                )
            )
            batch_sigmas = torch.tensor([batch_sigmas]).to(single_latent.device)
            loss = self.fwd_bwd(
                model_wrap,
                batch_sigmas,
                batch_noise,
                single_latent,
                cond,
                [index],
                extra_args,
                dataset_size,
                bwd=False,
            )
            total_loss += loss
        total_loss = total_loss / self.grad_acc / len(indicies)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        if self.loss_callback:
            self.loss_callback(total_loss.item())
        pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        model_wrap.conds = process_cond_list(model_wrap.conds)
        cond = model_wrap.conds["positive"]
        dataset_size = sigmas.size(0)
        torch.cuda.empty_cache()
        ui_pbar = ProgressBar(self.total_steps)
        for i in (
            pbar := trange(
                self.total_steps,
                desc="Training LoRA",
                smoothing=0.01,
                disable=not comfy.utils.PROGRESS_BAR_ENABLED,
            )
        ):
            noisegen = comfy_extras.nodes_custom_sampler.Noise_RandomNoise(
                self.seed + i * 1000
            )

            if self.bucket_latents is not None:
                self._train_step_bucket_mode(model_wrap, cond, extra_args, noisegen, latent_image, pbar)
            elif self.real_dataset is None:
                self._train_step_standard_mode(model_wrap, cond, extra_args, noisegen, latent_image, dataset_size, pbar)
            else:
                self._train_step_multires_mode(model_wrap, cond, extra_args, noisegen, latent_image, dataset_size, pbar)

            if (i + 1) % self.grad_acc == 0:
                if self.grad_scaler is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                for param_groups in self.optimizer.param_groups:
                    for param in param_groups["params"]:
                        if param.grad is None:
                            continue
                        param.grad.data = param.grad.data.to(param.data.dtype)
                if self.grad_scaler is not None:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            ui_pbar.update(1)
        torch.cuda.empty_cache()
        return torch.zeros_like(latent_image)


class BiasDiff(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def __call__(self, b):
        org_dtype = b.dtype
        return (b.to(self.bias) + self.bias).to(org_dtype)

    def passive_memory_usage(self):
        return self.bias.nelement() * self.bias.element_size()

    def move_to(self, device):
        self.to(device=device)
        return self.passive_memory_usage()


def draw_loss_graph(loss_map, steps):
    width, height = 500, 300
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    min_loss, max_loss = min(loss_map.values()), max(loss_map.values())
    scaled_loss = [(l - min_loss) / (max_loss - min_loss) for l in loss_map.values()]

    prev_point = (0, height - int(scaled_loss[0] * height))
    for i, l in enumerate(scaled_loss[1:], start=1):
        x = int(i / (steps - 1) * width)
        y = height - int(l * height)
        draw.line([prev_point, (x, y)], fill="blue", width=2)
        prev_point = (x, y)

    return img


def find_all_highest_child_module_with_forward(
    model: torch.nn.Module, result=None, name=None
):
    if result is None:
        result = []
    elif hasattr(model, "forward") and not isinstance(
        model, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)
    ):
        result.append(model)
        logging.debug(f"Found module with forward: {name} ({model.__class__.__name__})")
        return result
    name = name or "root"
    for next_name, child in model.named_children():
        find_all_highest_child_module_with_forward(child, result, f"{name}.{next_name}")
    return result


def find_modules_at_depth(
    model: nn.Module, depth: int = 1, result=None, current_depth=0, name=None
) -> list[nn.Module]:
    """
    Find modules at a specific depth level for gradient checkpointing.

    Args:
        model: The model to search
        depth: Target depth level (1 = top-level blocks, 2 = their children, etc.)
        result: Accumulator for results
        current_depth: Current recursion depth
        name: Current module name for logging

    Returns:
        List of modules at the target depth
    """
    if result is None:
        result = []
    name = name or "root"

    # Skip container modules (they don't have meaningful forward)
    is_container = isinstance(model, (nn.ModuleList, nn.Sequential, nn.ModuleDict))
    has_forward = hasattr(model, "forward") and not is_container

    if has_forward:
        current_depth += 1
        if current_depth == depth:
            result.append(model)
            logging.debug(f"Found module at depth {depth}: {name} ({model.__class__.__name__})")
            return result

    # Recurse into children
    for next_name, child in model.named_children():
        find_modules_at_depth(child, depth, result, current_depth, f"{name}.{next_name}")

    return result


class OffloadCheckpointFunction(torch.autograd.Function):
    """
    Gradient checkpointing that works with weight offloading.

    Forward: no_grad -> compute -> weights can be freed
    Backward: enable_grad -> recompute -> backward -> weights can be freed

    For single input, single output modules (Linear, Conv*).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, forward_fn):
        ctx.save_for_backward(x)
        ctx.forward_fn = forward_fn
        with torch.no_grad():
            return forward_fn(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, = ctx.saved_tensors
        forward_fn = ctx.forward_fn

        # Clear context early
        ctx.forward_fn = None

        with torch.enable_grad():
            x_detached = x.detach().requires_grad_(True)
            y = forward_fn(x_detached)
            y.backward(grad_out)
            grad_x = x_detached.grad

        # Explicit cleanup
        del y, x_detached, forward_fn

        return grad_x, None


def patch(m, offloading=False):
    if not hasattr(m, "forward"):
        return
    org_forward = m.forward

    # Branch 1: Linear/Conv* -> offload-compatible checkpoint (single input/output)
    if offloading and isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        def checkpointing_fwd(x):
            return OffloadCheckpointFunction.apply(x, org_forward)
    # Branch 2: Others -> standard checkpoint
    else:
        def fwd(args, kwargs):
            return org_forward(*args, **kwargs)

        def checkpointing_fwd(*args, **kwargs):
            return torch.utils.checkpoint.checkpoint(fwd, args, kwargs, use_reentrant=False)

    m.org_forward = org_forward
    m.forward = checkpointing_fwd


def unpatch(m):
    if hasattr(m, "org_forward"):
        m.forward = m.org_forward
        del m.org_forward


def _process_latents_bucket_mode(latents):
    """Process latents for bucket mode training.

    Args:
        latents: list[{"samples": tensor}] where each tensor is (Bi, C, Hi, Wi)

    Returns:
        list of latent tensors
    """
    bucket_latents = []
    for latent_dict in latents:
        bucket_latents.append(latent_dict["samples"])  # (Bi, C, Hi, Wi)
    return bucket_latents


def _process_latents_standard_mode(latents):
    """Process latents for standard (non-bucket) mode training.

    Args:
        latents: list of latent dicts or single latent dict

    Returns:
        Processed latents (tensor or list of tensors)
    """
    if len(latents) == 1:
        return latents[0]["samples"]  # Single latent dict

    latent_list = []
    for latent in latents:
        latent = latent["samples"]
        bs = latent.shape[0]
        if bs != 1:
            for sub_latent in latent:
                latent_list.append(sub_latent[None])
        else:
            latent_list.append(latent)
    return latent_list


def _process_conditioning(positive):
    """Process conditioning - either single list or list of lists.

    Args:
        positive: list of conditioning

    Returns:
        Flattened conditioning list
    """
    if len(positive) == 1:
        return positive[0]  # Single conditioning list

    # Multiple conditioning lists - flatten
    flat_positive = []
    for cond in positive:
        if isinstance(cond, list):
            flat_positive.extend(cond)
        else:
            flat_positive.append(cond)
    return flat_positive


def _prepare_latents_and_count(latents, dtype, bucket_mode):
    """Convert latents to dtype and compute image counts.

    Args:
        latents: Latents (tensor, list of tensors, or bucket list)
        dtype: Target dtype
        bucket_mode: Whether bucket mode is enabled

    Returns:
        tuple: (processed_latents, num_images, multi_res)
    """
    if bucket_mode:
        # In bucket mode, latents is list of tensors (Bi, C, Hi, Wi)
        latents = [t.to(dtype) for t in latents]
        num_buckets = len(latents)
        num_images = sum(t.shape[0] for t in latents)
        multi_res = False  # Not using multi_res path in bucket mode

        logging.debug(f"Bucket mode: {num_buckets} buckets, {num_images} total samples")
        for i, lat in enumerate(latents):
            logging.debug(f"  Bucket {i}: shape {lat.shape}")
        return latents, num_images, multi_res

    # Non-bucket mode
    if isinstance(latents, list):
        all_shapes = set()
        latents = [t.to(dtype) for t in latents]
        for latent in latents:
            all_shapes.add(latent.shape)
        logging.debug(f"Latent shapes: {all_shapes}")
        if len(all_shapes) > 1:
            multi_res = True
        else:
            multi_res = False
            latents = torch.cat(latents, dim=0)
        num_images = len(latents)
    elif isinstance(latents, torch.Tensor):
        latents = latents.to(dtype)
        num_images = latents.shape[0]
        multi_res = False
    else:
        logging.error(f"Invalid latents type: {type(latents)}")
        num_images = 0
        multi_res = False

    return latents, num_images, multi_res


def _validate_and_expand_conditioning(positive, num_images, bucket_mode):
    """Validate conditioning count matches image count, expand if needed.

    Args:
        positive: Conditioning list
        num_images: Number of images
        bucket_mode: Whether bucket mode is enabled

    Returns:
        Validated/expanded conditioning list

    Raises:
        ValueError: If conditioning count doesn't match image count
    """
    if bucket_mode:
        return positive  # Skip validation in bucket mode

    logging.debug(f"Total Images: {num_images}, Total Captions: {len(positive)}")
    if len(positive) == 1 and num_images > 1:
        return positive * num_images
    elif len(positive) != num_images:
        raise ValueError(
            f"Number of positive conditions ({len(positive)}) does not match number of images ({num_images})."
        )
    return positive


def _load_existing_lora(existing_lora):
    """Load existing LoRA weights if provided.

    Args:
        existing_lora: LoRA filename or "[None]"

    Returns:
        tuple: (existing_weights dict, existing_steps int)
    """
    if existing_lora == "[None]":
        return {}, 0

    lora_path = folder_paths.get_full_path_or_raise("loras", existing_lora)
    # Extract steps from filename like "trained_lora_10_steps_20250225_203716"
    existing_steps = int(existing_lora.split("_steps_")[0].split("_")[-1])
    existing_weights = {}
    if lora_path:
        existing_weights = comfy.utils.load_torch_file(lora_path)
    return existing_weights, existing_steps


def _create_weight_adapter(
    module, module_name, existing_weights, algorithm, lora_dtype, rank
):
    """Create a weight adapter for a module with weight.

    Args:
        module: The module to create adapter for
        module_name: Name of the module
        existing_weights: Dict of existing LoRA weights
        algorithm: Algorithm name for new adapters
        lora_dtype: dtype for LoRA weights
        rank: Rank for new LoRA adapters

    Returns:
        tuple: (train_adapter, lora_params dict)
    """
    key = f"{module_name}.weight"
    shape = module.weight.shape
    lora_params = {}

    logging.debug(f"Creating weight adapter for {key} with shape {shape}")

    if len(shape) >= 2:
        alpha = float(existing_weights.get(f"{key}.alpha", 1.0))
        dora_scale = existing_weights.get(f"{key}.dora_scale", None)

        # Try to load existing adapter
        existing_adapter = None
        for adapter_cls in adapters:
            existing_adapter = adapter_cls.load(
                module_name, existing_weights, alpha, dora_scale
            )
            if existing_adapter is not None:
                break

        if existing_adapter is None:
            adapter_cls = adapter_maps[algorithm]

        if existing_adapter is not None:
            train_adapter = existing_adapter.to_train().to(lora_dtype)
        else:
            # Use LoRA with alpha=1.0 by default
            train_adapter = adapter_cls.create_train(
                module.weight, rank=rank, alpha=1.0
            ).to(lora_dtype)

        for name, parameter in train_adapter.named_parameters():
            lora_params[f"{module_name}.{name}"] = parameter

        return train_adapter.train().requires_grad_(True), lora_params
    else:
        # 1D weight - use BiasDiff
        diff = torch.nn.Parameter(
            torch.zeros(module.weight.shape, dtype=lora_dtype, requires_grad=True)
        )
        diff_module = BiasDiff(diff).train().requires_grad_(True)
        lora_params[f"{module_name}.diff"] = diff
        return diff_module, lora_params


def _create_bias_adapter(module, module_name, lora_dtype):
    """Create a bias adapter for a module with bias.

    Args:
        module: The module with bias
        module_name: Name of the module
        lora_dtype: dtype for LoRA weights

    Returns:
        tuple: (bias_module, lora_params dict)
    """
    bias = torch.nn.Parameter(
        torch.zeros(module.bias.shape, dtype=lora_dtype, requires_grad=True)
    )
    bias_module = BiasDiff(bias).train().requires_grad_(True)
    lora_params = {f"{module_name}.diff_b": bias}
    return bias_module, lora_params


def _setup_lora_adapters(mp, existing_weights, algorithm, lora_dtype, rank):
    """Setup all LoRA adapters on the model.

    Args:
        mp: Model patcher
        existing_weights: Dict of existing LoRA weights
        algorithm: Algorithm name for new adapters
        lora_dtype: dtype for LoRA weights
        rank: Rank for new LoRA adapters

    Returns:
        tuple: (lora_sd dict, all_weight_adapters list)
    """
    lora_sd = {}
    all_weight_adapters = []

    for n, m in mp.model.named_modules():
        if hasattr(m, "weight_function"):
            if m.weight is not None:
                adapter, params = _create_weight_adapter(
                    m, n, existing_weights, algorithm, lora_dtype, rank
                )
                lora_sd.update(params)
                key = f"{n}.weight"
                mp.add_weight_wrapper(key, adapter)
                all_weight_adapters.append(adapter)

            if hasattr(m, "bias") and m.bias is not None:
                bias_adapter, bias_params = _create_bias_adapter(m, n, lora_dtype)
                lora_sd.update(bias_params)
                key = f"{n}.bias"
                mp.add_weight_wrapper(key, bias_adapter)
                all_weight_adapters.append(bias_adapter)

    return lora_sd, all_weight_adapters


def _setup_lora_adapters_bypass(mp, existing_weights, algorithm, lora_dtype, rank):
    """Setup LoRA adapters in bypass mode.

    In bypass mode:
        - Weight adapters (lora/lokr/oft) use bypass injection (forward hook)
        - Bias/norm adapters (BiasDiff) still use weight wrapper (direct modification)

    This is useful when the base model weights are quantized and cannot be
    directly modified.

    Args:
        mp: Model patcher
        existing_weights: Dict of existing LoRA weights
        algorithm: Algorithm name for new adapters
        lora_dtype: dtype for LoRA weights
        rank: Rank for new LoRA adapters

    Returns:
        tuple: (lora_sd dict, all_weight_adapters list, bypass_manager)
    """
    lora_sd = {}
    all_weight_adapters = []
    bypass_manager = BypassInjectionManager()

    for n, m in mp.model.named_modules():
        if hasattr(m, "weight_function"):
            if m.weight is not None:
                adapter, params = _create_weight_adapter(
                    m, n, existing_weights, algorithm, lora_dtype, rank
                )
                lora_sd.update(params)
                all_weight_adapters.append(adapter)

                key = f"{n}.weight"
                # BiasDiff (for 1D weights like norm) uses weight wrapper, not bypass
                # Only use bypass for adapters that have h() method (lora/lokr/oft)
                if isinstance(adapter, BiasDiff):
                    mp.add_weight_wrapper(key, adapter)
                    logging.debug(f"[BypassMode] Added 1D weight adapter (weight wrapper) for {key}")
                else:
                    bypass_manager.add_adapter(key, adapter, strength=1.0)
                    logging.debug(f"[BypassMode] Added weight adapter (bypass) for {key}")

            if hasattr(m, "bias") and m.bias is not None:
                # Bias adapters still use weight wrapper (bias is usually not quantized)
                bias_adapter, bias_params = _create_bias_adapter(m, n, lora_dtype)
                lora_sd.update(bias_params)
                key = f"{n}.bias"
                mp.add_weight_wrapper(key, bias_adapter)
                all_weight_adapters.append(bias_adapter)
                logging.debug(f"[BypassMode] Added bias adapter (weight wrapper) for {key}")

    return lora_sd, all_weight_adapters, bypass_manager


def _create_optimizer(optimizer_name, parameters, learning_rate):
    """Create optimizer based on name.

    Args:
        optimizer_name: Name of optimizer ("Adam", "AdamW", "SGD", "RMSprop")
        parameters: Parameters to optimize
        learning_rate: Learning rate

    Returns:
        Optimizer instance
    """
    if optimizer_name == "Adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(parameters, lr=learning_rate)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate)


def _create_loss_function(loss_function_name):
    """Create loss function based on name.

    Args:
        loss_function_name: Name of loss function ("MSE", "L1", "Huber", "SmoothL1")

    Returns:
        Loss function instance
    """
    if loss_function_name == "MSE":
        return torch.nn.MSELoss()
    elif loss_function_name == "L1":
        return torch.nn.L1Loss()
    elif loss_function_name == "Huber":
        return torch.nn.HuberLoss()
    elif loss_function_name == "SmoothL1":
        return torch.nn.SmoothL1Loss()


def _run_training_loop(
    guider, train_sampler, latents, num_images, seed, bucket_mode, multi_res
):
    """Execute the training loop.

    Args:
        guider: The guider object
        train_sampler: The training sampler
        latents: Latent tensors
        num_images: Number of images
        seed: Random seed
        bucket_mode: Whether bucket mode is enabled
        multi_res: Whether multi-resolution mode is enabled
    """
    sigmas = torch.tensor(range(num_images))
    noise = comfy_extras.nodes_custom_sampler.Noise_RandomNoise(seed)

    if bucket_mode:
        # Use first bucket's first latent as dummy for guider
        dummy_latent = latents[0][:1].repeat(num_images, 1, 1, 1)
        guider.sample(
            noise.generate_noise({"samples": dummy_latent}),
            dummy_latent,
            train_sampler,
            sigmas,
            seed=noise.seed,
        )
    elif multi_res:
        # use first latent as dummy latent if multi_res
        latents = latents[0].repeat(num_images, 1, 1, 1)
        guider.sample(
            noise.generate_noise({"samples": latents}),
            latents,
            train_sampler,
            sigmas,
            seed=noise.seed,
        )
    else:
        guider.sample(
            noise.generate_noise({"samples": latents}),
            latents,
            train_sampler,
            sigmas,
            seed=noise.seed,
        )


class TrainLoraNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TrainLoraNode",
            display_name="Train LoRA",
            category="model/training",
            is_experimental=True,
            is_input_list=True,  # All inputs become lists
            inputs=[
                io.Model.Input("model", tooltip="The model to train the LoRA on."),
                io.Latent.Input(
                    "latents",
                    tooltip="The Latents to use for training, serve as dataset/input of the model.",
                ),
                io.Conditioning.Input(
                    "positive", tooltip="The positive conditioning to use for training."
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=10000,
                    tooltip="The batch size to use for training.",
                ),
                io.Int.Input(
                    "grad_accumulation_steps",
                    default=1,
                    min=1,
                    max=1024,
                    tooltip="The number of gradient accumulation steps to use for training.",
                ),
                io.Int.Input(
                    "steps",
                    default=16,
                    min=1,
                    max=100000,
                    tooltip="The number of steps to train the LoRA for.",
                ),
                io.Float.Input(
                    "learning_rate",
                    default=0.0005,
                    min=0.0000001,
                    max=1.0,
                    step=0.0000001,
                    tooltip="The learning rate to use for training.",
                ),
                io.Int.Input(
                    "rank",
                    default=8,
                    min=1,
                    max=128,
                    tooltip="The rank of the LoRA layers.",
                ),
                io.Combo.Input(
                    "optimizer",
                    options=["AdamW", "Adam", "SGD", "RMSprop"],
                    default="AdamW",
                    tooltip="The optimizer to use for training.",
                ),
                io.Combo.Input(
                    "loss_function",
                    options=["MSE", "L1", "Huber", "SmoothL1"],
                    default="MSE",
                    tooltip="The loss function to use for training.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="The seed to use for training (used in generator for LoRA weight initialization and noise sampling)",
                ),
                io.Combo.Input(
                    "training_dtype",
                    options=["bf16", "fp32", "none"],
                    default="bf16",
                    tooltip="The dtype to use for training. 'none' preserves the model's native compute dtype instead of overriding it. For fp16 models, GradScaler is automatically enabled.",
                ),
                io.Combo.Input(
                    "lora_dtype",
                    options=["bf16", "fp32"],
                    default="bf16",
                    tooltip="The dtype to use for lora.",
                ),
                io.Boolean.Input(
                    "quantized_backward",
                    default=False,
                    tooltip="When using training_dtype 'none' and training on quantized model, doing backward with quantized matmul when enabled.",
                ),
                io.Combo.Input(
                    "algorithm",
                    options=list(adapter_maps.keys()),
                    default=list(adapter_maps.keys())[0],
                    tooltip="The algorithm to use for training.",
                ),
                io.Boolean.Input(
                    "gradient_checkpointing",
                    default=True,
                    tooltip="Use gradient checkpointing for training.",
                ),
                io.Int.Input(
                    "checkpoint_depth",
                    default=1,
                    min=1,
                    max=5,
                    tooltip="Depth level for gradient checkpointing.",
                ),
                io.Boolean.Input(
                    "offloading",
                    default=False,
                    tooltip="Offload model weights to CPU during training to save GPU memory.",
                ),
                io.Combo.Input(
                    "existing_lora",
                    options=folder_paths.get_filename_list("loras") + ["[None]"],
                    default="[None]",
                    tooltip="The existing LoRA to append to. Set to None for new LoRA.",
                ),
                io.Boolean.Input(
                    "bucket_mode",
                    default=False,
                    tooltip="Enable resolution bucket mode. When enabled, expects pre-bucketed latents from ResolutionBucket node.",
                ),
                io.Boolean.Input(
                    "bypass_mode",
                    default=False,
                    tooltip="Enable bypass mode for training. When enabled, adapters are applied via forward hooks instead of weight modification. Useful for quantized models where weights cannot be directly modified.",
                ),
            ],
            outputs=[
                io.Custom("LORA_MODEL").Output(
                    display_name="lora", tooltip="LoRA weights"
                ),
                io.Custom("LOSS_MAP").Output(
                    display_name="loss_map", tooltip="Loss history"
                ),
                io.Int.Output(display_name="steps", tooltip="Total training steps"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        latents,
        positive,
        batch_size,
        steps,
        grad_accumulation_steps,
        learning_rate,
        rank,
        optimizer,
        loss_function,
        seed,
        training_dtype,
        lora_dtype,
        quantized_backward,
        algorithm,
        gradient_checkpointing,
        checkpoint_depth,
        offloading,
        existing_lora,
        bucket_mode,
        bypass_mode,
    ):
        # Extract scalars from lists (due to is_input_list=True)
        model = model[0]
        batch_size = batch_size[0]
        steps = steps[0]
        grad_accumulation_steps = grad_accumulation_steps[0]
        learning_rate = learning_rate[0]
        rank = rank[0]
        optimizer_name = optimizer[0]
        loss_function_name = loss_function[0]
        seed = seed[0]
        training_dtype = training_dtype[0]
        lora_dtype = lora_dtype[0]
        quantized_backward = quantized_backward[0]
        algorithm = algorithm[0]
        gradient_checkpointing = gradient_checkpointing[0]
        offloading = offloading[0]
        checkpoint_depth = checkpoint_depth[0]
        existing_lora = existing_lora[0]
        bucket_mode = bucket_mode[0]
        bypass_mode = bypass_mode[0]

        comfy.model_management.training_fp8_bwd = quantized_backward

        # Process latents based on mode
        if bucket_mode:
            latents = _process_latents_bucket_mode(latents)
        else:
            latents = _process_latents_standard_mode(latents)

        # Process conditioning
        positive = _process_conditioning(positive)

        with torch.inference_mode(False):
            # Setup model and dtype
            mp = model.clone(force_deepcopy=True)
            use_grad_scaler = False
            lora_dtype = node_helpers.string_to_torch_dtype(lora_dtype)
            if training_dtype != "none":
                dtype = node_helpers.string_to_torch_dtype(training_dtype)
                mp.set_model_compute_dtype(dtype)
            else:
                # Detect model's native dtype for autocast
                model_dtype = mp.model.get_dtype()
                if model_dtype == torch.float16:
                    dtype = torch.float16
                    # GradScaler only supports float16 gradients, not bfloat16.
                    # Only enable it when lora params will also be in float16.
                    if lora_dtype != torch.bfloat16:
                        use_grad_scaler = True
                    # Warn about fp16 accumulation instability during training
                    if PerformanceFeature.Fp16Accumulation in args.fast:
                        logging.warning(
                            "WARNING: FP16 model detected with fp16_accumulation enabled. "
                            "This combination can be numerically unstable during training and may cause NaN values. "
                            "Suggested fixes: 1) Set training_dtype to 'bf16', or 2) Disable fp16_accumulation (remove from --fast flags)."
                        )
                else:
                    # For fp8, bf16, or other dtypes, use bf16 autocast
                    dtype = torch.bfloat16

            # Prepare latents and compute counts
            latents_dtype = dtype if dtype not in (None,) else torch.bfloat16
            latents, num_images, multi_res = _prepare_latents_and_count(
                latents, latents_dtype, bucket_mode
            )

            # Validate and expand conditioning
            positive = _validate_and_expand_conditioning(positive, num_images, bucket_mode)

            # Setup models for training
            mp.model.requires_grad_(False).train()

            # Load existing LoRA weights if provided
            existing_weights, existing_steps = _load_existing_lora(existing_lora)

            # Setup LoRA adapters
            bypass_manager = None
            if bypass_mode:
                logging.debug("Using bypass mode for training")
                lora_sd, all_weight_adapters, bypass_manager = _setup_lora_adapters_bypass(
                    mp, existing_weights, algorithm, lora_dtype, rank
                )
            else:
                lora_sd, all_weight_adapters = _setup_lora_adapters(
                    mp, existing_weights, algorithm, lora_dtype, rank
                )

            # Create optimizer and loss function
            optimizer = _create_optimizer(
                optimizer_name, lora_sd.values(), learning_rate
            )
            criterion = _create_loss_function(loss_function_name)

            # Setup gradient checkpointing
            if gradient_checkpointing:
                modules_to_patch = find_modules_at_depth(
                    mp.model.diffusion_model, depth=checkpoint_depth
                )
                logging.info(f"Gradient checkpointing: patching {len(modules_to_patch)} modules at depth {checkpoint_depth}")
                for m in modules_to_patch:
                    patch(m, offloading=offloading)

            torch.cuda.empty_cache()
            # With force_full_load=False we should be able to have offloading
            # But for offloading in training we need custom AutoGrad hooks for fwd/bwd
            comfy.model_management.load_models_gpu(
                [mp], memory_required=1e20, force_full_load=not offloading
            )
            torch.cuda.empty_cache()

            # Setup loss tracking
            loss_map = {"loss": []}

            def loss_callback(loss):
                loss_map["loss"].append(loss)

            # Create sampler
            if bucket_mode:
                train_sampler = TrainSampler(
                    criterion,
                    optimizer,
                    loss_callback=loss_callback,
                    batch_size=batch_size,
                    grad_acc=grad_accumulation_steps,
                    total_steps=steps * grad_accumulation_steps,
                    seed=seed,
                    training_dtype=dtype,
                    bucket_latents=latents,
                    use_grad_scaler=use_grad_scaler,
                )
            else:
                train_sampler = TrainSampler(
                    criterion,
                    optimizer,
                    loss_callback=loss_callback,
                    batch_size=batch_size,
                    grad_acc=grad_accumulation_steps,
                    total_steps=steps * grad_accumulation_steps,
                    seed=seed,
                    training_dtype=dtype,
                    real_dataset=latents if multi_res else None,
                    use_grad_scaler=use_grad_scaler,
                )

            # Setup guider
            guider = TrainGuider(mp, offloading=offloading)
            guider.set_conds(positive)

            # Inject bypass hooks if bypass mode is enabled
            bypass_injections = None
            if bypass_manager is not None:
                bypass_injections = bypass_manager.create_injections(mp.model)
                for injection in bypass_injections:
                    injection.inject(mp)
                logging.debug(f"[BypassMode] Injected {bypass_manager.get_hook_count()} bypass hooks")

            # Run training loop
            try:
                comfy.model_management.in_training = True
                _run_training_loop(
                    guider,
                    train_sampler,
                    latents,
                    num_images,
                    seed,
                    bucket_mode,
                    multi_res,
                )
            finally:
                comfy.model_management.in_training = False
                # Eject bypass hooks if they were injected
                if bypass_injections is not None:
                    for injection in bypass_injections:
                        injection.eject(mp)
                    logging.debug("[BypassMode] Ejected bypass hooks")
                for m in mp.model.modules():
                    unpatch(m)
            del train_sampler, optimizer

            for param in lora_sd:
                lora_sd[param] = lora_sd[param].to(lora_dtype).detach()

            for adapter in all_weight_adapters:
                adapter.requires_grad_(False)
                del adapter
            del all_weight_adapters

            # mp in train node is highly specialized for training
            # use it in inference will result in bad behavior so we don't return it
            return io.NodeOutput(lora_sd, loss_map, steps + existing_steps)


class LoraModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoraModelLoader",
            display_name="Load LoRA Model",
            category="model/loaders",
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model", tooltip="The diffusion model the LoRA will be applied to."
                ),
                io.Custom("LORA_MODEL").Input(
                    "lora", tooltip="The LoRA model to apply to the diffusion model."
                ),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    tooltip="How strongly to modify the diffusion model. This value can be negative.",
                ),
                io.Boolean.Input(
                    "bypass",
                    default=False,
                    tooltip="When enabled, applies LoRA in bypass mode without modifying base model weights. Useful for training and when model weights are offloaded.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model", tooltip="The modified diffusion model."
                ),
            ],
        )

    @classmethod
    def execute(cls, model, lora, strength_model, bypass=False):
        if strength_model == 0:
            return io.NodeOutput(model)

        if bypass:
            model_lora, _ = comfy.sd.load_bypass_lora_for_models(
                model, None, lora, strength_model, 0
            )
        else:
            model_lora, _ = comfy.sd.load_lora_for_models(
                model, None, lora, strength_model, 0
            )
        return io.NodeOutput(model_lora)


class SaveLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveLoRA",
            search_aliases=["export lora"],
            display_name="Save LoRA Weights",
            category="model/merging",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Custom("LORA_MODEL").Input(
                    "lora",
                    tooltip="The LoRA model to save. Do not use the model with LoRA layers.",
                ),
                io.String.Input(
                    "prefix",
                    default="loras/ComfyUI_trained_lora",
                    tooltip="The prefix to use for the saved LoRA file.",
                ),
                io.Int.Input(
                    "steps",
                    optional=True,
                    tooltip="Optional: The number of steps the LoRA has been trained for, used to name the saved file.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, lora, prefix, steps=None):
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(prefix, output_dir)
        )
        if steps is None:
            output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        else:
            output_checkpoint = f"{filename}_{steps}_steps_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)
        safetensors.torch.save_file(lora, output_checkpoint)
        return io.NodeOutput()


class LossGraphNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LossGraphNode",
            search_aliases=["training chart", "training visualization", "plot loss"],
            display_name="Plot Loss Graph",
            category="model/training",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Custom("LOSS_MAP").Input(
                    "loss", tooltip="Loss map from training node."
                ),
                io.String.Input(
                    "filename_prefix",
                    default="loss_graph",
                    tooltip="Prefix for the saved loss graph image.",
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, loss, filename_prefix, prompt=None, extra_pnginfo=None):
        loss_values = loss["loss"]
        width, height = 800, 480
        margin = 40

        img = Image.new(
            "RGB", (width + margin, height + margin), "white"
        )  # Extend canvas
        draw = ImageDraw.Draw(img)

        min_loss, max_loss = min(loss_values), max(loss_values)
        scaled_loss = [(l - min_loss) / (max_loss - min_loss) for l in loss_values]

        steps = len(loss_values)

        prev_point = (margin, height - int(scaled_loss[0] * height))
        for i, l in enumerate(scaled_loss[1:], start=1):
            x = margin + int(i / steps * width)  # Scale X properly
            y = height - int(l * height)
            draw.line([prev_point, (x, y)], fill="blue", width=2)
            prev_point = (x, y)

        draw.line([(margin, 0), (margin, height)], fill="black", width=2)  # Y-axis
        draw.line(
            [(margin, height), (width + margin, height)], fill="black", width=2
        )  # X-axis

        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        # Add axis labels
        draw.text((5, height // 2), "Loss", font=font, fill="black")
        draw.text((width // 2, height + 10), "Steps", font=font, fill="black")

        # Add min/max loss values
        draw.text((margin - 30, 0), f"{max_loss:.2f}", font=font, fill="black")
        draw.text(
            (margin - 30, height - 10), f"{min_loss:.2f}", font=font, fill="black"
        )

        # Convert PIL image to tensor for PreviewImage
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]  # [1, H, W, 3]

        # Return preview UI
        return io.NodeOutput(ui=ui.PreviewImage(img_tensor, cls=cls))


# ========== Extension Setup ==========


class TrainingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TrainLoraNode,
            LoraModelLoader,
            SaveLoRA,
            LossGraphNode,
        ]


async def comfy_entrypoint() -> TrainingExtension:
    return TrainingExtension()
