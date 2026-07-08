import torch
from tqdm import tqdm
from typing_extensions import override

import comfy.model_patcher
import comfy.utils
import folder_paths
from comfy import model_management
from comfy_extras.frame_interpolation_models.ifnet import IFNet, detect_rife_config
from comfy_extras.frame_interpolation_models.film_net import FILMNet
from comfy_api.latest import ComfyExtension, io

FrameInterpolationModel = io.Custom("INTERP_MODEL")


class FrameInterpolationModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FrameInterpolationModelLoader",
            display_name="Load Frame Interpolation Model",
            category="model/loaders",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("frame_interpolation"),
                               tooltip="Select a frame interpolation model to load. Models must be placed in the 'frame_interpolation' folder."),
            ],
            outputs=[
                FrameInterpolationModel.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("frame_interpolation", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)

        model = cls._detect_and_load(sd)
        dtype = torch.float16 if model_management.should_use_fp16(model_management.get_torch_device()) else torch.float32
        model.eval().to(dtype)
        patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=model_management.get_torch_device(),
            offload_device=model_management.unet_offload_device(),
        )
        return io.NodeOutput(patcher)

    @classmethod
    def _detect_and_load(cls, sd):
        # Try FILM
        if "extract.extract_sublevels.convs.0.0.conv.weight" in sd:
            model = FILMNet()
            model.load_state_dict(sd)
            return model

        # Try RIFE (needs key remapping for raw checkpoints)
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": "", "flownet.": ""})
        key_map = {}
        for k in sd:
            for i in range(5):
                if k.startswith(f"block{i}."):
                    key_map[k] = f"blocks.{i}.{k[len(f'block{i}.'):]}"
        if key_map:
            sd = {key_map.get(k, k): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not k.startswith(("teacher.", "caltime."))}

        try:
            head_ch, channels = detect_rife_config(sd)
        except (KeyError, ValueError):
            raise ValueError("Unrecognized frame interpolation model format")
        model = IFNet(head_ch=head_ch, channels=channels)
        model.load_state_dict(sd)
        return model


class FrameInterpolate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FrameInterpolate",
            display_name="Run Frame Interpolation Model",
            category="video",
            search_aliases=["rife", "film", "frame interpolation", "slow motion", "interpolate frames", "vfi"],
            inputs=[
                FrameInterpolationModel.Input("interp_model"),
                io.Image.Input("images"),
                io.Int.Input("multiplier", default=2, min=2, max=16),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, interp_model, images, multiplier) -> io.NodeOutput:
        offload_device = model_management.intermediate_device()

        num_frames = images.shape[0]
        if num_frames < 2 or multiplier < 2:
            return io.NodeOutput(images)

        device = interp_model.load_device
        dtype = interp_model.model_dtype()
        inference_model = interp_model.model
        activation_mem = inference_model.memory_used_forward(images.shape, dtype)
        model_management.load_models_gpu([interp_model], memory_required=activation_mem)
        align = getattr(inference_model, "pad_align", 1)
        H, W = images.shape[1], images.shape[2]

        # Prepare a single padded frame on device for determining output dimensions
        def prepare_frame(idx):
            frame = images[idx:idx + 1].movedim(-1, 1).to(dtype=dtype, device=device)
            if align > 1:
                from comfy.ldm.common_dit import pad_to_patch_size
                frame = pad_to_patch_size(frame, (align, align), padding_mode="reflect")
            return frame

        # Count total interpolation passes for progress bar
        total_pairs = num_frames - 1
        num_interp = multiplier - 1
        total_steps = total_pairs * num_interp
        pbar = comfy.utils.ProgressBar(total_steps)
        tqdm_bar = tqdm(total=total_steps, desc="Frame interpolation")

        batch = num_interp  # reduced on OOM and persists across pairs (same resolution = same limit)
        t_values = [t / multiplier for t in range(1, multiplier)]

        out_dtype = model_management.intermediate_dtype()
        total_out_frames = total_pairs * multiplier + 1
        result = torch.empty((total_out_frames, 3, H, W), dtype=out_dtype, device=offload_device)
        result[0] = images[0].movedim(-1, 0).to(out_dtype)
        out_idx = 1

        # Pre-compute timestep tensor on device (padded dimensions needed)
        sample = prepare_frame(0)
        pH, pW = sample.shape[2], sample.shape[3]
        ts_full = torch.tensor(t_values, device=device, dtype=dtype).reshape(num_interp, 1, 1, 1)
        ts_full = ts_full.expand(-1, 1, pH, pW)
        del sample

        multi_fn = getattr(inference_model, "forward_multi_timestep", None)
        feat_cache = {}
        prev_frame = None

        try:
            for i in range(total_pairs):
                img0_single = prev_frame if prev_frame is not None else prepare_frame(i)
                img1_single = prepare_frame(i + 1)
                prev_frame = img1_single

                # Cache features: img1 of pair N becomes img0 of pair N+1
                feat_cache["img0"] = feat_cache.pop("next") if "next" in feat_cache else inference_model.extract_features(img0_single)
                feat_cache["img1"] = inference_model.extract_features(img1_single)
                feat_cache["next"] = feat_cache["img1"]

                used_multi = False
                if multi_fn is not None:
                    # Models with timestep-independent flow can compute it once for all timesteps
                    try:
                        mids = multi_fn(img0_single, img1_single, t_values, cache=feat_cache)
                        result[out_idx:out_idx + num_interp] = mids[:, :, :H, :W].to(out_dtype)
                        out_idx += num_interp
                        pbar.update(num_interp)
                        tqdm_bar.update(num_interp)
                        used_multi = True
                    except model_management.OOM_EXCEPTION:
                        model_management.soft_empty_cache()
                        multi_fn = None  # fall through to single-timestep path

                if not used_multi:
                    j = 0
                    while j < num_interp:
                        b = min(batch, num_interp - j)
                        try:
                            img0 = img0_single.expand(b, -1, -1, -1)
                            img1 = img1_single.expand(b, -1, -1, -1)
                            mids = inference_model(img0, img1, timestep=ts_full[j:j + b], cache=feat_cache)
                            result[out_idx:out_idx + b] = mids[:, :, :H, :W].to(out_dtype)
                            out_idx += b
                            pbar.update(b)
                            tqdm_bar.update(b)
                            j += b
                        except model_management.OOM_EXCEPTION:
                            if batch <= 1:
                                raise
                            batch = max(1, batch // 2)
                            model_management.soft_empty_cache()

                result[out_idx] = images[i + 1].movedim(-1, 0).to(out_dtype)
                out_idx += 1
        finally:
            tqdm_bar.close()

        # BCHW -> BHWC
        result = result.movedim(1, -1).clamp_(0.0, 1.0)
        return io.NodeOutput(result)


class FrameInterpolationExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FrameInterpolationModelLoader,
            FrameInterpolate,
        ]


async def comfy_entrypoint() -> FrameInterpolationExtension:
    return FrameInterpolationExtension()
