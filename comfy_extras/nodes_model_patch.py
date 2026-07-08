import torch
from torch import nn
import folder_paths
import comfy.utils
import comfy.ops
import comfy.model_management
import comfy.ldm.common_dit
import comfy.latent_formats
import comfy.ldm.lumina.controlnet
import comfy.ldm.supir.supir_modules
from comfy.ldm.wan.model_multitalk import WanMultiTalkAttentionBlock, MultiTalkAudioProjModel
from comfy_api.latest import io
from comfy.ldm.supir.supir_patch import SUPIRPatch


class BlockWiseControlBlock(torch.nn.Module):
    # [linear, gelu, linear]
    def __init__(self, dim: int = 3072, device=None, dtype=None, operations=None):
        super().__init__()
        self.x_rms = operations.RMSNorm(dim, eps=1e-6)
        self.y_rms = operations.RMSNorm(dim, eps=1e-6)
        self.input_proj = operations.Linear(dim, dim)
        self.act = torch.nn.GELU()
        self.output_proj = operations.Linear(dim, dim)

    def forward(self, x, y):
        x, y = self.x_rms(x), self.y_rms(y)
        x = self.input_proj(x + y)
        x = self.act(x)
        x = self.output_proj(x)
        return x


class QwenImageBlockWiseControlNet(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
        in_dim: int = 64,
        additional_in_dim: int = 0,
        dim: int = 3072,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.additional_in_dim = additional_in_dim
        self.img_in = operations.Linear(in_dim + additional_in_dim, dim, device=device, dtype=dtype)
        self.controlnet_blocks = torch.nn.ModuleList(
            [
                BlockWiseControlBlock(dim, device=device, dtype=dtype, operations=operations)
                for _ in range(num_layers)
            ]
        )

    def process_input_latent_image(self, latent_image):
        latent_image[:, :16] = comfy.latent_formats.Wan21().process_in(latent_image[:, :16])
        patch_size = 2
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(latent_image, (1, patch_size, patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(orig_shape[0], (orig_shape[-2] // 2) * (orig_shape[-1] // 2), orig_shape[1] * 4)
        return self.img_in(hidden_states)

    def control_block(self, img, controlnet_conditioning, block_id):
        return self.controlnet_blocks[block_id](img, controlnet_conditioning)


class SigLIPMultiFeatProjModel(torch.nn.Module):
    """
    SigLIP Multi-Feature Projection Model for processing style features from different layers
    and projecting them into a unified hidden space.

    Args:
        siglip_token_nums (int): Number of SigLIP tokens, default 257
        style_token_nums (int): Number of style tokens, default 256
        siglip_token_dims (int): Dimension of SigLIP tokens, default 1536
        hidden_size (int): Hidden layer size, default 3072
        context_layer_norm (bool): Whether to use context layer normalization, default False
    """

    def __init__(
        self,
        siglip_token_nums: int = 729,
        style_token_nums: int = 64,
        siglip_token_dims: int = 1152,
        hidden_size: int = 3072,
        context_layer_norm: bool = True,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        # High-level feature processing (layer -2)
        self.high_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.high_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.high_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

        # Mid-level feature processing (layer -11)
        self.mid_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.mid_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.mid_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

        # Low-level feature processing (layer -20)
        self.low_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.low_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.low_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

    def forward(self, siglip_outputs):
        """
        Forward pass function

        Args:
            siglip_outputs: Output from SigLIP model, containing hidden_states

        Returns:
            torch.Tensor: Concatenated multi-layer features with shape [bs, 3*style_token_nums, hidden_size]
        """
        dtype = next(self.high_embedding_linear.parameters()).dtype

        # Process high-level features (layer -2)
        high_embedding = self._process_layer_features(
            siglip_outputs[2],
            self.high_embedding_linear,
            self.high_layer_norm,
            self.high_projection,
            dtype
        )

        # Process mid-level features (layer -11)
        mid_embedding = self._process_layer_features(
            siglip_outputs[1],
            self.mid_embedding_linear,
            self.mid_layer_norm,
            self.mid_projection,
            dtype
        )

        # Process low-level features (layer -20)
        low_embedding = self._process_layer_features(
            siglip_outputs[0],
            self.low_embedding_linear,
            self.low_layer_norm,
            self.low_projection,
            dtype
        )

        # Concatenate features from all layersmodel_patch
        return torch.cat((high_embedding, mid_embedding, low_embedding), dim=1)

    def _process_layer_features(
        self,
        hidden_states: torch.Tensor,
        embedding_linear: nn.Module,
        layer_norm: nn.Module,
        projection: nn.Module,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Helper function to process features from a single layer

        Args:
            hidden_states: Input hidden states [bs, seq_len, dim]
            embedding_linear: Embedding linear layer
            layer_norm: Layer normalization
            projection: Projection layer
            dtype: Target data type

        Returns:
            torch.Tensor: Processed features [bs, style_token_nums, hidden_size]
        """
        # Transform dimensions: [bs, seq_len, dim] -> [bs, dim, seq_len] -> [bs, dim, style_token_nums] -> [bs, style_token_nums, dim]
        embedding = embedding_linear(
            hidden_states.to(dtype).transpose(1, 2)
        ).transpose(1, 2)

        # Apply layer normalization
        embedding = layer_norm(embedding)

        # Project to target hidden space
        embedding = projection(embedding)

        return embedding

def z_image_convert(sd):
    replace_keys = {".attention.to_out.0.bias": ".attention.out.bias",
                    ".attention.norm_k.weight": ".attention.k_norm.weight",
                    ".attention.norm_q.weight": ".attention.q_norm.weight",
                    ".attention.to_out.0.weight": ".attention.out.weight"
                    }

    out_sd = {}
    for k in sorted(sd.keys()):
        w = sd[k]

        k_out = k
        if k_out.endswith(".attention.to_k.weight"):
            cc = [w]
            continue
        if k_out.endswith(".attention.to_q.weight"):
            cc = [w] + cc
            continue
        if k_out.endswith(".attention.to_v.weight"):
            cc = cc + [w]
            w = torch.cat(cc, dim=0)
            k_out = k_out.replace(".attention.to_v.weight", ".attention.qkv.weight")

        for r, rr in replace_keys.items():
            k_out = k_out.replace(r, rr)
        out_sd[k_out] = w

    return out_sd

class ModelPatchLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "name": (folder_paths.get_filename_list("model_patches"), ),
                              }}
    RETURN_TYPES = ("MODEL_PATCH",)
    FUNCTION = "load_model_patch"
    EXPERIMENTAL = True

    CATEGORY = "model/loaders"

    def load_model_patch(self, name):
        model_patch_path = folder_paths.get_full_path_or_raise("model_patches", name)
        sd = comfy.utils.load_torch_file(model_patch_path, safe_load=True)
        dtype = comfy.utils.weight_dtype(sd)

        if 'controlnet_blocks.0.y_rms.weight' in sd:
            additional_in_dim = sd["img_in.weight"].shape[1] - 64
            model = QwenImageBlockWiseControlNet(additional_in_dim=additional_in_dim, device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=comfy.ops.manual_cast)
        elif 'feature_embedder.mid_layer_norm.bias' in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"feature_embedder.": ""}, filter_keys=True)
            model = SigLIPMultiFeatProjModel(device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=comfy.ops.manual_cast)
        elif 'control_all_x_embedder.2-1.weight' in sd: # alipai z image fun controlnet
            sd = z_image_convert(sd)
            config = {}
            if 'control_layers.4.adaLN_modulation.0.weight' not in sd:
                config['n_control_layers'] = 3
                config['additional_in_dim'] = 17
                config['refiner_control'] = True
            if 'control_layers.14.adaLN_modulation.0.weight' in sd:
                config['n_control_layers'] = 15
                config['additional_in_dim'] = 17
                config['refiner_control'] = True
                ref_weight = sd.get("control_noise_refiner.0.after_proj.weight", None)
                if ref_weight is not None:
                    if torch.count_nonzero(ref_weight) == 0:
                        config['broken'] = True
            model = comfy.ldm.lumina.controlnet.ZImage_Control(device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=comfy.ops.manual_cast, **config)
        elif "audio_proj.proj1.weight" in sd:
            model = MultiTalkModelPatch(
                    audio_window=5, context_tokens=32, vae_scale=4,
                    in_dim=sd["blocks.0.audio_cross_attn.proj.weight"].shape[0],
                    intermediate_dim=sd["audio_proj.proj1.weight"].shape[0],
                    out_dim=sd["audio_proj.norm.weight"].shape[0],
                    device=comfy.model_management.unet_offload_device(),
                    operations=comfy.ops.manual_cast)
        elif 'model.control_model.input_hint_block.0.weight' in sd or 'control_model.input_hint_block.0.weight' in sd:
            prefix_replace = {}
            if 'model.control_model.input_hint_block.0.weight' in sd:
                prefix_replace["model.control_model."] = "control_model."
                prefix_replace["model.diffusion_model.project_modules."] = "project_modules."
            else:
                prefix_replace["control_model."] = "control_model."
                prefix_replace["project_modules."] = "project_modules."

            # Extract denoise_encoder weights before filter_keys discards them
            de_prefix = "first_stage_model.denoise_encoder."
            denoise_encoder_sd = {}
            for k in list(sd.keys()):
                if k.startswith(de_prefix):
                    denoise_encoder_sd[k[len(de_prefix):]] = sd.pop(k)

            sd = comfy.utils.state_dict_prefix_replace(sd, prefix_replace, filter_keys=True)
            sd.pop("control_model.mask_LQ", None)
            model = comfy.ldm.supir.supir_modules.SUPIR(device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=comfy.ops.manual_cast)
            if denoise_encoder_sd:
                model.denoise_encoder_sd = denoise_encoder_sd

        model_patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())
        model.load_state_dict(sd, assign=model_patcher.is_dynamic())
        return (model_patcher,)


class DiffSynthCnetPatch:
    def __init__(self, model_patch, vae, image, strength, mask=None):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.mask = mask
        self.encoded_image = model_patch.model.process_input_latent_image(self.encode_latent_cond(image))
        self.encoded_image_size = (image.shape[1], image.shape[2])

    def encode_latent_cond(self, image):
        latent_image = self.vae.encode(image)
        if self.model_patch.model.additional_in_dim > 0:
            if self.mask is None:
                mask_ = torch.ones_like(latent_image)[:, :self.model_patch.model.additional_in_dim // 4]
            else:
                mask_ = comfy.utils.common_upscale(self.mask.mean(dim=1, keepdim=True), latent_image.shape[-1], latent_image.shape[-2], "bilinear", "none")

            return torch.cat([latent_image, mask_], dim=1)
        else:
            return latent_image

    def __call__(self, kwargs):
        x = kwargs.get("x")
        img = kwargs.get("img")
        block_index = kwargs.get("block_index")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center")
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.model_patch.model.process_input_latent_image(self.encode_latent_cond(image_scaled.movedim(1, -1)))
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        img[:, :self.encoded_image.shape[1]] += (self.model_patch.model.control_block(img[:, :self.encoded_image.shape[1]], self.encoded_image.to(img.dtype), block_index) * self.strength)
        kwargs['img'] = img
        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self):
        return [self.model_patch]

class ZImageControlPatch:
    def __init__(self, model_patch, vae, image, strength, inpaint_image=None, mask=None):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.inpaint_image = inpaint_image
        self.mask = mask
        self.strength = strength
        self.is_inpaint = self.model_patch.model.additional_in_dim > 0

        skip_encoding = False
        if self.image is not None and self.inpaint_image is not None:
            if self.image.shape != self.inpaint_image.shape:
                skip_encoding = True

        if skip_encoding:
            self.encoded_image = None
        else:
            self.encoded_image = self.encode_latent_cond(self.image, self.inpaint_image)
            if self.image is None:
                self.encoded_image_size = (self.inpaint_image.shape[1], self.inpaint_image.shape[2])
            else:
                self.encoded_image_size = (self.image.shape[1], self.image.shape[2])
        self.temp_data = None

    def encode_latent_cond(self, control_image=None, inpaint_image=None):
        latent_image = None
        if control_image is not None:
            latent_image = comfy.latent_formats.Flux().process_in(self.vae.encode(control_image))

        if self.is_inpaint:
            if inpaint_image is None:
                inpaint_image = torch.ones_like(control_image) * 0.5

            if self.mask is not None:
                mask_inpaint = comfy.utils.common_upscale(self.mask.view(self.mask.shape[0], -1, self.mask.shape[-2], self.mask.shape[-1]).mean(dim=1, keepdim=True), inpaint_image.shape[-2], inpaint_image.shape[-3], "bilinear", "center")
                inpaint_image = ((inpaint_image - 0.5) * mask_inpaint.movedim(1, -1).round()) + 0.5

            inpaint_image_latent = comfy.latent_formats.Flux().process_in(self.vae.encode(inpaint_image))

            if self.mask is None:
                mask_ = torch.zeros_like(inpaint_image_latent)[:, :1]
            else:
                mask_ = comfy.utils.common_upscale(self.mask.view(self.mask.shape[0], -1, self.mask.shape[-2], self.mask.shape[-1]).mean(dim=1, keepdim=True).to(device=inpaint_image_latent.device), inpaint_image_latent.shape[-1], inpaint_image_latent.shape[-2], "nearest", "center")

            if latent_image is None:
                latent_image = comfy.latent_formats.Flux().process_in(self.vae.encode(torch.ones_like(inpaint_image) * 0.5))

            return torch.cat([latent_image, mask_, inpaint_image_latent], dim=1)
        else:
            return latent_image

    def __call__(self, kwargs):
        x = kwargs.get("x")
        img = kwargs.get("img")
        img_input = kwargs.get("img_input")
        txt = kwargs.get("txt")
        pe = kwargs.get("pe")
        vec = kwargs.get("vec")
        block_index = kwargs.get("block_index")
        block_type = kwargs.get("block_type", "")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = None
            if self.image is not None:
                image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center").movedim(1, -1)
                self.encoded_image_size = (image_scaled.shape[-3], image_scaled.shape[-2])

            inpaint_scaled = None
            if self.inpaint_image is not None:
                inpaint_scaled = comfy.utils.common_upscale(self.inpaint_image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center").movedim(1, -1)
                self.encoded_image_size = (inpaint_scaled.shape[-3], inpaint_scaled.shape[-2])

            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.encode_latent_cond(image_scaled, inpaint_scaled)
            comfy.model_management.load_models_gpu(loaded_models)

        cnet_blocks = self.model_patch.model.n_control_layers
        div = round(30 / cnet_blocks)

        cnet_index = (block_index // div)
        cnet_index_float = (block_index / div)

        kwargs.pop("img")  # we do ops in place
        kwargs.pop("txt")

        if cnet_index_float > (cnet_blocks - 1):
            self.temp_data = None
            return kwargs

        if self.temp_data is None or self.temp_data[0] > cnet_index:
            if block_type == "noise_refiner":
                self.temp_data = (-3, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))
            else:
                self.temp_data = (-1, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))

        if block_type == "noise_refiner":
            next_layer = self.temp_data[0] + 1
            self.temp_data = (next_layer, self.model_patch.model.forward_noise_refiner_block(block_index, self.temp_data[1][1], img_input[:, :self.temp_data[1][1].shape[1]], None, pe, vec))
            if self.temp_data[1][0] is not None:
                img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * self.strength)
        else:
            while self.temp_data[0] < cnet_index and (self.temp_data[0] + 1) < cnet_blocks:
                next_layer = self.temp_data[0] + 1
                self.temp_data = (next_layer, self.model_patch.model.forward_control_block(next_layer, self.temp_data[1][1], img_input[:, :self.temp_data[1][1].shape[1]], None, pe, vec))

            if cnet_index_float == self.temp_data[0]:
                img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * self.strength)
                if cnet_blocks == self.temp_data[0] + 1:
                    self.temp_data = None

        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            if self.encoded_image is not None:
                self.encoded_image = self.encoded_image.to(device_or_dtype)
            self.temp_data = None
        return self

    def models(self):
        return [self.model_patch]

class QwenImageDiffsynthControlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "model_patch": ("MODEL_PATCH",),
                              "vae": ("VAE",),
                              "image": ("IMAGE",),
                              "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              },
                "optional": {"mask": ("MASK",)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "diffsynth_controlnet"
    EXPERIMENTAL = True

    CATEGORY = "model/patch/qwen"

    def diffsynth_controlnet(self, model, model_patch, vae, image=None, strength=1.0, inpaint_image=None, mask=None):
        model_patched = model.clone()
        if image is not None:
            image = image[:, :, :, :3]
        if inpaint_image is not None:
            inpaint_image = inpaint_image[:, :, :, :3]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.ndim == 4:
                mask = mask.unsqueeze(2)
            mask = 1.0 - mask

        if isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control):
            patch = ZImageControlPatch(model_patch, vae, image, strength, inpaint_image=inpaint_image, mask=mask)
            model_patched.set_model_noise_refiner_patch(patch)
            model_patched.set_model_double_block_patch(patch)
        else:
            model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask))
        return (model_patched,)

class ZImageFunControlnet(QwenImageDiffsynthControlnet):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "model_patch": ("MODEL_PATCH",),
                              "vae": ("VAE",),
                              "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              },
                "optional": {"image": ("IMAGE",), "inpaint_image": ("IMAGE",), "mask": ("MASK",)}}

    CATEGORY = "model/patch/z-image"

class UsoStyleProjectorPatch:
    def __init__(self, model_patch, encoded_image):
        self.model_patch = model_patch
        self.encoded_image = encoded_image

    def __call__(self, kwargs):
        txt_ids = kwargs.get("txt_ids")
        txt = kwargs.get("txt")
        siglip_embedding = self.model_patch.model(self.encoded_image.to(txt.dtype)).to(txt.dtype)
        txt = torch.cat([siglip_embedding, txt], dim=1)
        kwargs['txt'] = txt
        kwargs['txt_ids'] = torch.cat([torch.zeros(siglip_embedding.shape[0], siglip_embedding.shape[1], 3, dtype=txt_ids.dtype, device=txt_ids.device), txt_ids], dim=1)
        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self):
        return [self.model_patch]


class USOStyleReference:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "model_patch": ("MODEL_PATCH",),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    EXPERIMENTAL = True

    CATEGORY = "model/patch/flux"

    def apply_patch(self, model, model_patch, clip_vision_output):
        encoded_image = torch.stack((clip_vision_output.all_hidden_states[:, -20], clip_vision_output.all_hidden_states[:, -11], clip_vision_output.penultimate_hidden_states))
        model_patched = model.clone()
        model_patched.set_model_post_input_patch(UsoStyleProjectorPatch(model_patch, encoded_image))
        return (model_patched,)


class MultiTalkModelPatch(torch.nn.Module):
    def __init__(
        self,
        audio_window: int = 5,
        intermediate_dim: int = 512,
        in_dim: int = 5120,
        out_dim: int = 768,
        context_tokens: int = 32,
        vae_scale: int = 4,
        num_layers: int = 40,

        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.audio_proj = MultiTalkAudioProjModel(
                seq_len=audio_window,
                seq_len_vf=audio_window+vae_scale-1,
                intermediate_dim=intermediate_dim,
                out_dim=out_dim,
                context_tokens=context_tokens,
                device=device,
                dtype=dtype,
                operations=operations
        )
        self.blocks = torch.nn.ModuleList(
            [
                WanMultiTalkAttentionBlock(in_dim, out_dim, device=device, dtype=dtype, operations=operations)
                for _ in range(num_layers)
            ]
        )


class SUPIRApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SUPIRApply",
            category="model/patch/supir",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.ModelPatch.Input("model_patch"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Float.Input("strength_start", default=1.0, min=0.0, max=10.0, step=0.01,
                               tooltip="Control strength at the start of sampling (high sigma)."),
                io.Float.Input("strength_end", default=1.0, min=0.0, max=10.0, step=0.01,
                               tooltip="Control strength at the end of sampling (low sigma). Linearly interpolated from start."),
                io.Float.Input("restore_cfg", default=4.0, min=0.0, max=20.0, step=0.1, advanced=True,
                               tooltip="Pulls denoised output toward the input latent. Higher = stronger fidelity to input. 0 to disable."),
                io.Float.Input("restore_cfg_s_tmin", default=0.05, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Sigma threshold below which restore_cfg is disabled."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def _encode_with_denoise_encoder(cls, vae, model_patch, image):
        """Encode using denoise_encoder weights from SUPIR checkpoint if available."""
        denoise_sd = getattr(model_patch.model, 'denoise_encoder_sd', None)
        if not denoise_sd:
            return vae.encode(image)

        # Clone VAE patcher, apply denoise_encoder weights to clone, encode
        orig_patcher = vae.patcher
        vae.patcher = orig_patcher.clone()
        patches = {f"encoder.{k}": (v,) for k, v in denoise_sd.items()}
        vae.patcher.add_patches(patches, strength_patch=1.0, strength_model=0.0)
        try:
            return vae.encode(image)
        finally:
            vae.patcher = orig_patcher

    @classmethod
    def execute(cls, *, model: io.Model.Type, model_patch: io.ModelPatch.Type, vae: io.Vae.Type, image: io.Image.Type,
                strength_start: float, strength_end: float, restore_cfg: float, restore_cfg_s_tmin: float) -> io.NodeOutput:
        model_patched = model.clone()
        hint_latent = model.get_model_object("latent_format").process_in(
            cls._encode_with_denoise_encoder(vae, model_patch, image[:, :, :, :3]))
        patch = SUPIRPatch(model_patch, model_patch.model.project_modules, hint_latent, strength_start, strength_end)
        patch.register(model_patched)

        if restore_cfg > 0.0:
            # Round-trip to match original pipeline: decode hint, re-encode with regular VAE
            latent_format = model.get_model_object("latent_format")
            decoded = vae.decode(latent_format.process_out(hint_latent))
            x_center = latent_format.process_in(vae.encode(decoded[:, :, :, :3]))
            sigma_max = 14.6146

            def restore_cfg_function(args):
                denoised = args["denoised"]
                sigma = args["sigma"]
                if sigma.dim() > 0:
                    s = sigma[0].item()
                else:
                    s = sigma.item()
                if s > restore_cfg_s_tmin:
                    ref = x_center.to(device=denoised.device, dtype=denoised.dtype)
                    b = denoised.shape[0]
                    if ref.shape[0] != b:
                        ref = ref.expand(b, -1, -1, -1) if ref.shape[0] == 1 else ref.repeat((b + ref.shape[0] - 1) // ref.shape[0], 1, 1, 1)[:b]
                    sigma_val = sigma.view(-1, 1, 1, 1) if sigma.dim() > 0 else sigma
                    d_center = denoised - ref
                    denoised = denoised - d_center * ((sigma_val / sigma_max) ** restore_cfg)
                return denoised

            model_patched.set_model_sampler_post_cfg_function(restore_cfg_function)

        return io.NodeOutput(model_patched)


NODE_CLASS_MAPPINGS = {
    "ModelPatchLoader": ModelPatchLoader,
    "QwenImageDiffsynthControlnet": QwenImageDiffsynthControlnet,
    "ZImageFunControlnet": ZImageFunControlnet,
    "USOStyleReference": USOStyleReference,
    "SUPIRApply": SUPIRApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPatchLoader": "Load Model Patch",
    "QwenImageDiffsynthControlnet": "Apply Qwen Image DiffSynth ControlNet",
    "ZImageFunControlnet": "Apply Z-Image Fun ControlNet",
    "USOStyleReference": "Apply USO Style Reference",
    "SUPIRApply": "Apply SUPIR Patch",
}
