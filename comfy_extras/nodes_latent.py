import comfy.utils
import comfy_extras.nodes_post_processing
import torch
import nodes
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import logging
import math

def reshape_latent_to(target_shape, latent, repeat_batch=True):
    if latent.shape[1:] != target_shape[1:]:
        latent = comfy.utils.common_upscale(latent, target_shape[-1], target_shape[-2], "bilinear", "center")
    if repeat_batch:
        return comfy.utils.repeat_to_batch_size(latent, target_shape[0])
    else:
        return latent


class LatentAdd(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentAdd",
            search_aliases=["combine latents", "sum latents"],
            category="model/latent/advanced",
            inputs=[
                io.Latent.Input("samples1"),
                io.Latent.Input("samples2"),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples1, samples2) -> io.NodeOutput:
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)
        samples_out["samples"] = s1 + s2
        return io.NodeOutput(samples_out)

class LatentSubtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentSubtract",
            search_aliases=["difference latent", "remove features"],
            category="model/latent/advanced",
            inputs=[
                io.Latent.Input("samples1"),
                io.Latent.Input("samples2"),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples1, samples2) -> io.NodeOutput:
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)
        samples_out["samples"] = s1 - s2
        return io.NodeOutput(samples_out)

class LatentMultiply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentMultiply",
            search_aliases=["scale latent", "amplify latent", "latent gain"],
            category="model/latent/advanced",
            inputs=[
                io.Latent.Input("samples"),
                io.Float.Input("multiplier", default=1.0, min=-10.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, multiplier) -> io.NodeOutput:
        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = s1 * multiplier
        return io.NodeOutput(samples_out)

class LatentInterpolate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentInterpolate",
            search_aliases=["blend latent", "mix latent", "lerp latent", "transition"],
            category="model/latent/advanced",
            inputs=[
                io.Latent.Input("samples1"),
                io.Latent.Input("samples2"),
                io.Float.Input("ratio", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples1, samples2, ratio) -> io.NodeOutput:
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)

        m1 = torch.linalg.vector_norm(s1, dim=(1))
        m2 = torch.linalg.vector_norm(s2, dim=(1))

        s1 = torch.nan_to_num(s1 / m1)
        s2 = torch.nan_to_num(s2 / m2)

        t = (s1 * ratio + s2 * (1.0 - ratio))
        mt = torch.linalg.vector_norm(t, dim=(1))
        st = torch.nan_to_num(t / mt)

        samples_out["samples"] = st * (m1 * ratio + m2 * (1.0 - ratio))
        return io.NodeOutput(samples_out)

class LatentConcat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentConcat",
            search_aliases=["join latents", "stitch latents"],
            category="model/latent/advanced",
            inputs=[
                io.Latent.Input("samples1"),
                io.Latent.Input("samples2"),
                io.Combo.Input("dim", options=["x", "-x", "y", "-y", "t", "-t"]),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples1, samples2, dim) -> io.NodeOutput:
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]
        s2 = comfy.utils.repeat_to_batch_size(s2, s1.shape[0])

        if "-" in dim:
            c = (s2, s1)
        else:
            c = (s1, s2)

        if "x" in dim:
            dim = -1
        elif "y" in dim:
            dim = -2
        elif "t" in dim:
            dim = -3

        samples_out["samples"] = torch.cat(c, dim=dim)
        return io.NodeOutput(samples_out)

class LatentCut(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentCut",
            search_aliases=["crop latent", "slice latent", "extract region"],
            category="model/latent/advanced",
            inputs=[
                io.Latent.Input("samples"),
                io.Combo.Input("dim", options=["x", "y", "t"]),
                io.Int.Input("index", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("amount", default=1, min=1, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, dim, index, amount) -> io.NodeOutput:
        samples_out = samples.copy()

        s1 = samples["samples"]

        if "x" in dim:
            dim = s1.ndim - 1
        elif "y" in dim:
            dim = s1.ndim - 2
        elif "t" in dim:
            dim = s1.ndim - 3

        if index >= 0:
            index = min(index, s1.shape[dim] - 1)
            amount = min(s1.shape[dim] - index, amount)
        else:
            index = max(index, -s1.shape[dim])
            amount = min(-index, amount)

        samples_out["samples"] = torch.narrow(s1, dim, index, amount)
        return io.NodeOutput(samples_out)

class LatentCutToBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentCutToBatch",
            search_aliases=["slice to batch", "split latent", "tile latent"],
            category="model/latent/advanced",
            inputs=[
                io.Latent.Input("samples"),
                io.Combo.Input("dim", options=["t", "x", "y"]),
                io.Int.Input("slice_size", default=1, min=1, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, dim, slice_size) -> io.NodeOutput:
        samples_out = samples.copy()

        s1 = samples["samples"]

        if "x" in dim:
            dim = s1.ndim - 1
        elif "y" in dim:
            dim = s1.ndim - 2
        elif "t" in dim:
            dim = s1.ndim - 3

        if dim < 2:
            return io.NodeOutput(samples)

        s = s1.movedim(dim, 1)
        if s.shape[1] < slice_size:
            slice_size = s.shape[1]
        elif s.shape[1] % slice_size != 0:
            s = s[:, :math.floor(s.shape[1] / slice_size) * slice_size]
        new_shape = [-1, slice_size] + list(s.shape[2:])
        samples_out["samples"] = s.reshape(new_shape).movedim(1, dim)
        return io.NodeOutput(samples_out)

class LatentBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentBatch",
            search_aliases=["combine latents", "merge latents", "join latents"],
            display_name="Batch Latents (DEPRECATED)",
            category="model/latent/batch",
            is_deprecated=True,
            inputs=[
                io.Latent.Input("samples1"),
                io.Latent.Input("samples2"),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples1, samples2) -> io.NodeOutput:
        samples_out = samples1.copy()
        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2, repeat_batch=False)
        s = torch.cat((s1, s2), dim=0)
        samples_out["samples"] = s
        samples_out["batch_index"] = samples1.get("batch_index", [x for x in range(0, s1.shape[0])]) + samples2.get("batch_index", [x for x in range(0, s2.shape[0])])
        return io.NodeOutput(samples_out)

class LatentBatchSeedBehavior(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentBatchSeedBehavior",
            category="model/latent/advanced",
            inputs=[
                io.Latent.Input("samples"),
                io.Combo.Input("seed_behavior", options=["random", "fixed"], default="fixed"),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, seed_behavior) -> io.NodeOutput:
        samples_out = samples.copy()
        latent = samples["samples"]
        if seed_behavior == "random":
            if 'batch_index' in samples_out:
                samples_out.pop('batch_index')
        elif seed_behavior == "fixed":
            batch_number = samples_out.get("batch_index", [0])[0]
            samples_out["batch_index"] = [batch_number] * latent.shape[0]

        return io.NodeOutput(samples_out)

class LatentApplyOperation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentApplyOperation",
            search_aliases=["transform latent"],
            category="model/latent/advanced/operations",
            is_experimental=True,
            inputs=[
                io.Latent.Input("samples"),
                io.LatentOperation.Input("operation"),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, operation) -> io.NodeOutput:
        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = operation(latent=s1)
        return io.NodeOutput(samples_out)

class LatentApplyOperationCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentApplyOperationCFG",
            category="model/latent/advanced/operations",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.LatentOperation.Input("operation"),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, operation) -> io.NodeOutput:
        m = model.clone()

        def pre_cfg_function(args):
            conds_out = args["conds_out"]
            if len(conds_out) == 2:
                conds_out[0] = operation(latent=(conds_out[0] - conds_out[1])) + conds_out[1]
            else:
                conds_out[0] = operation(latent=conds_out[0])
            return conds_out

        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
        return io.NodeOutput(m)

class LatentOperationTonemapReinhard(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentOperationTonemapReinhard",
            search_aliases=["hdr latent"],
            category="model/latent/advanced/operations",
            is_experimental=True,
            inputs=[
                io.Float.Input("multiplier", default=1.0, min=0.0, max=100.0, step=0.01),
            ],
            outputs=[
                io.LatentOperation.Output(),
            ],
        )

    @classmethod
    def execute(cls, multiplier) -> io.NodeOutput:
        def tonemap_reinhard(latent, **kwargs):
            latent_vector_magnitude = (torch.linalg.vector_norm(latent, dim=(1)) + 0.0000000001)[:,None]
            normalized_latent = latent / latent_vector_magnitude

            dims = list(range(1, latent_vector_magnitude.ndim))
            mean = torch.mean(latent_vector_magnitude, dim=dims, keepdim=True)
            std = torch.std(latent_vector_magnitude, dim=dims, keepdim=True)

            top = (std * 5 + mean) * multiplier

            #reinhard
            latent_vector_magnitude *= (1.0 / top)
            new_magnitude = latent_vector_magnitude / (latent_vector_magnitude + 1.0)
            new_magnitude *= top

            return normalized_latent * new_magnitude
        return io.NodeOutput(tonemap_reinhard)

class LatentOperationSharpen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentOperationSharpen",
            category="model/latent/advanced/operations",
            is_experimental=True,
            inputs=[
                io.Int.Input("sharpen_radius", default=9, min=1, max=31, step=1, advanced=True),
                io.Float.Input("sigma", default=1.0, min=0.1, max=10.0, step=0.1, advanced=True),
                io.Float.Input("alpha", default=0.1, min=0.0, max=5.0, step=0.01, advanced=True),
            ],
            outputs=[
                io.LatentOperation.Output(),
            ],
        )

    @classmethod
    def execute(cls, sharpen_radius, sigma, alpha) -> io.NodeOutput:
        def sharpen(latent, **kwargs):
            luminance = (torch.linalg.vector_norm(latent, dim=(1)) + 1e-6)[:,None]
            normalized_latent = latent / luminance
            channels = latent.shape[1]

            kernel_size = sharpen_radius * 2 + 1
            kernel = comfy_extras.nodes_post_processing.gaussian_kernel(kernel_size, sigma, device=luminance.device)
            center = kernel_size // 2

            kernel *= alpha * -10
            kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0

            padded_image = torch.nn.functional.pad(normalized_latent, (sharpen_radius,sharpen_radius,sharpen_radius,sharpen_radius), 'reflect')
            sharpened = torch.nn.functional.conv2d(padded_image, kernel.repeat(channels, 1, 1).unsqueeze(1), padding=kernel_size // 2, groups=channels)[:,:,sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]

            return luminance * sharpened
        return io.NodeOutput(sharpen)

class ReplaceVideoLatentFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ReplaceVideoLatentFrames",
            display_name="Replace Video Latent Frames",
            category="model/latent/batch",
            inputs=[
                io.Latent.Input("destination", tooltip="The destination latent where frames will be replaced."),
                io.Latent.Input("source", optional=True, tooltip="The source latent providing frames to insert into the destination latent. If not provided, the destination latent is returned unchanged."),
                io.Int.Input("index", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, step=1, tooltip="The starting latent frame index in the destination latent where the source latent frames will be placed. Negative values count from the end."),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, destination, index, source=None) -> io.NodeOutput:
        if source is None:
            return io.NodeOutput(destination)
        dest_frames = destination["samples"].shape[2]
        source_frames = source["samples"].shape[2]
        if index < 0:
            index = dest_frames + index
        if index > dest_frames:
            logging.warning(f"ReplaceVideoLatentFrames: Index {index} is out of bounds for destination latent frames {dest_frames}.")
            return io.NodeOutput(destination)
        if index + source_frames > dest_frames:
            logging.warning(f"ReplaceVideoLatentFrames: Source latent frames {source_frames} do not fit within destination latent frames {dest_frames} at the specified index {index}.")
            return io.NodeOutput(destination)
        s = source.copy()
        s_source = source["samples"]
        s_destination = destination["samples"].clone()
        s_destination[:, :, index:index + s_source.shape[2]] = s_source
        s["samples"] = s_destination
        return io.NodeOutput(s)

class LatentExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LatentAdd,
            LatentSubtract,
            LatentMultiply,
            LatentInterpolate,
            LatentConcat,
            LatentCut,
            LatentCutToBatch,
            LatentBatch,
            LatentBatchSeedBehavior,
            LatentApplyOperation,
            LatentApplyOperationCFG,
            LatentOperationTonemapReinhard,
            LatentOperationSharpen,
            ReplaceVideoLatentFrames
        ]


async def comfy_entrypoint() -> LatentExtension:
    return LatentExtension()
