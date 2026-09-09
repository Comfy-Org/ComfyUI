import torch
from PIL import Image
from comfy.cli_args import args, LatentPreviewMethod
from comfy.taesd.taesd import TAESD
from comfy.sd import VAE
import comfy.model_management
import comfy.model_prefetch
import folder_paths
import comfy.utils
import logging

default_preview_method = args.preview_method

MAX_PREVIEW_RESOLUTION = args.preview_size
VIDEO_TAES = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2", "taeh3"]

def preview_to_image(latent_image, do_scale=True):
        if do_scale:
            latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                                .mul(0xFF)  # to 0..255
                                )
        else:
            latents_ubyte = (latent_image.clamp(0, 1)
                                .mul(0xFF)  # to 0..255
                                )
        if comfy.model_management.directml_enabled:
                latents_ubyte = latents_ubyte.to(dtype=torch.uint8)
        latents_ubyte = latents_ubyte.to(device="cpu", dtype=torch.uint8, non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device))

        return Image.fromarray(latents_ubyte.numpy())

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)

class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1])[0].movedim(0, 2)
        return preview_to_image(x_sample)

class TAEHVPreviewerImpl(TAESDPreviewerImpl):
    def __init__(self, taesd, compile_preview=False):
        super().__init__(taesd)
        self.compile_preview = compile_preview

    def decode_latent_to_preview(self, x0):
        samples = x0[:1, :, :1]
        if self.compile_preview and comfy.model_prefetch.malloc_graph_enabled(self.taesd.device):
            comfy.model_prefetch.malloc_graph_begin(self.taesd.device)
        x_sample = self.taesd.decode(samples)[0][0]
        comfy.model_prefetch.malloc_graph_end()
        return preview_to_image(x_sample, do_scale=False)

class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None, latent_rgb_factors_reshape=None):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")
        self.latent_rgb_factors_reshape = latent_rgb_factors_reshape

    def decode_latent_to_preview(self, x0):
        if self.latent_rgb_factors_reshape is not None:
            x0 = self.latent_rgb_factors_reshape(x0)
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        if x0.ndim == 5:
            x0 = x0[0, :, 0]
        else:
            x0 = x0[0]

        latent_image = torch.nn.functional.linear(x0.movedim(0, -1), self.latent_rgb_factors, bias=self.latent_rgb_factors_bias)
        # latent_image = x0[0].permute(1, 2, 0) @ self.latent_rgb_factors

        return preview_to_image(latent_image)


def get_previewer(device, latent_format):
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer methods
        taesd_decoder_path = None
        if latent_format.taesd_decoder_name is not None:
            taesd_decoder_path = next(
                (fn for fn in folder_paths.get_filename_list("vae_approx")
                    if fn.startswith(latent_format.taesd_decoder_name)),
                ""
            )
            taesd_decoder_path = folder_paths.get_full_path("vae_approx", taesd_decoder_path)

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB

        if method == LatentPreviewMethod.TAESD:
            if taesd_decoder_path:
                if latent_format.taesd_decoder_name in VIDEO_TAES:
                    taesd = VAE(comfy.utils.load_torch_file(taesd_decoder_path))
                    taesd.first_stage_model.show_progress_bar = False
                    previewer = TAEHVPreviewerImpl(taesd, compile_preview=latent_format.compile_preview)
                else:
                    taesd = TAESD(None, taesd_decoder_path, latent_channels=latent_format.latent_channels).to(device)
                    previewer = TAESDPreviewerImpl(taesd)
            else:
                logging.warning("Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(latent_format.taesd_decoder_name))

        if previewer is None:
            if latent_format.latent_rgb_factors is not None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors, latent_format.latent_rgb_factors_bias, latent_format.latent_rgb_factors_reshape)
    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer(model.load_device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            if x0.is_nested:
                x0 = x0.tensors[0]
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

def set_preview_method(override: str = None):
    if override and override != "default":
        method = LatentPreviewMethod.from_string(override)
        if method is not None:
            args.preview_method = method
            return
    args.preview_method = default_preview_method
