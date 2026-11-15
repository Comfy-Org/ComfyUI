import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing_extensions import override

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from comfy_extras.void_noise_warp import RaftOpticalFlow

OpticalFlow = io.Custom("OPTICAL_FLOW")

# larger RAFT input only costs time: it gave no better result, and the correlation volume grows with the square of the pixel count
RAFT_MAX_PIXELS = 768 * 768
LAPLACIAN = ((0.0, -1.0, 0.0), (-1.0, 4.0, -1.0), (0.0, -1.0, 0.0))


def warp(x, flow, grid, norm):
    # backward warp: sample x at each pixel's position + flow, norm maps flow pixels to grid units
    return F.grid_sample(x, torch.addcmul(grid, flow.movedim(1, -1), norm), mode="bilinear", padding_mode="border", align_corners=True)


def gaussian_blur(x, kernel):
    r = kernel.shape[-1] // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kernel)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), kernel.transpose(-1, -2))


def restrict(x):
    # 2x2 block means with odd sizes zero padded, a quarter of the transpose of nearest upsampling
    return F.avg_pool2d(F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2)), 2)


def residual(op, x, r):
    stencil, diag, _ = op
    ax = F.conv2d(x, stencil, padding=1, groups=x.shape[1]).addcmul_(diag, x)
    return torch.sub(r, ax, out=ax)


def vcycle(ops, r, k=0):
    # one damped Jacobi sweep before and after the coarse correction, four on the coarsest level
    wminv = ops[k][2]
    x = r * wminv
    if k == len(ops) - 1:
        for _ in range(3):
            x.addcmul_(residual(ops[k], x, r), wminv)
        return x
    coarse = vcycle(ops, restrict(residual(ops[k], x, r)), k + 1)
    x += F.interpolate(coarse, scale_factor=2, mode="nearest")[..., :r.shape[-2], :r.shape[-1]]
    return x.addcmul_(residual(ops[k], x, r), wminv)


def screened_poisson(levels, d, r, iters=100, tol=1e-6):
    # conv(x, stencil) + (fold + d) * x = r on the first level by multigrid-preconditioned CG, overwrites r
    ops = []
    for i, (stencil, fold) in enumerate(levels):
        if i:
            d = restrict(d)
        diag = fold + d
        ops.append((stencil, diag, diag.add(stencil[0, 0, 1, 1]).reciprocal_().mul_(0.8)))
    x = torch.zeros_like(r)
    p = vcycle(ops, r)
    rz = torch.dot(r.view(-1), p.view(-1))
    stop = rz * tol
    for _ in range(iters):
        if rz <= stop:
            break
        ap = F.conv2d(p, levels[0][0], padding=1, groups=r.shape[1]).addcmul_(ops[0][1], p)
        alpha = rz / torch.dot(p.view(-1), ap.view(-1))
        x.addcmul_(p, alpha)
        r.addcmul_(ap, alpha, value=-1)
        z = vcycle(ops, r)
        rz, rz_prev = torch.dot(r.view(-1), z.view(-1)), rz
        p = z.addcmul_(p, rz / rz_prev)
    return x


class VideoDeflicker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoDeflicker",
            display_name="Video Deflicker",
            category="video",
            search_aliases=["temporal consistency", "flicker", "stabilize"],
            description="Temporal consistency after Bonneel et al. 2015, with a fidelity term against drift: each frame keeps its own detail while following its neighbours along the guide video's optical flow.",
            inputs=[
                OpticalFlow.Input("optical_flow"),
                io.Image.Input("images", tooltip="Flickering frames, e.g. a video upscaled or processed frame by frame."),
                io.Image.Input("guide", optional=True,
                               tooltip="Stable source frames to follow the motion of, at any resolution. Defaults to images, which removes less flicker."),
                io.Float.Input("strength", default=1.0, min=0.0, max=100.0, step=0.1,
                               tooltip="How strongly each frame follows its neighbours. Higher removes more flicker and softens more fine detail."),
                io.Combo.Input("direction", options=["bidirectional", "forward"], default="bidirectional",
                               tooltip="bidirectional averages a forward and a backward pass, which avoids trailing behind motion at twice the time."),
                io.Float.Input("threshold", default=0.1, min=0.0, max=1.0, step=0.01,
                               tooltip="Changes larger than this between a frame and its motion-compensated neighbour count as real and are smoothed less. 0 smooths every change alike."),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, optical_flow, images, strength, direction, threshold, guide=None) -> io.NodeOutput:
        if strength == 0:
            return io.NodeOutput(images)
        if guide is None:
            guide = images
        h, w = images.shape[1:3]
        s = min(1.0, math.sqrt(RAFT_MAX_PIXELS / (guide.shape[1] * guide.shape[2])))
        fh, fw = round(guide.shape[1] * s / 8) * 8, round(guide.shape[2] * s / 8) * 8
        # RAFT's correlation volume plus ~36 float32 planes at the output resolution
        comfy.model_management.load_models_gpu([optical_flow], memory_required=3 * 4 * (fh * fw // 64) ** 2 + 36 * 4 * h * w)
        device = optical_flow.load_device
        raft = RaftOpticalFlow(optical_flow.model)
        eye = torch.eye(2, 3, device=device)[None]
        grid, grid_flow = F.affine_grid(eye, (1, 1, h, w), align_corners=True), F.affine_grid(eye, (1, 1, fh, fw), align_corners=True)
        norm = torch.tensor([2 / (w - 1) * w / fw, 2 / (h - 1) * h / fh], device=device)
        norm_flow = torch.tensor([2 / (fw - 1), 2 / (fh - 1)], device=device)
        # multigrid levels halve the grid and quarter the Laplacian until the strength / 4 fidelity term dominates it;
        # border neighbours folded onto the diagonal give the Neumann Laplacian
        mu = strength / 4
        levels = []
        size, scale = (h, w), 1.0
        while True:
            stencil = torch.tensor(LAPLACIAN, device=device).mul_(scale).expand(images.shape[-1], 1, 3, 3)
            levels.append((stencil, F.conv2d(torch.ones((1, 1, *size), device=device), stencil[:1], padding=1).neg_()))
            if 8 * scale <= mu or min(size) < 32:
                break
            size, scale = ((size[0] + 1) // 2, (size[1] + 1) // 2), scale / 4
        blur = torch.exp(-torch.arange(-6, 7, device=device, dtype=torch.float32).square() / 8)
        blur = (blur / blur.sum()).view(1, 1, 1, -1)

        def guide_frame(t, frame):
            if guide is not images:
                frame = guide[t].to(device, torch.float32).movedim(-1, 0)[None]
            return F.interpolate(frame[:, :3], size=(fh, fw), mode="area")

        def deflicker_pass(order):
            # yields each frame of order deflickered on the device
            prev = images[order[0]].to(device, torch.float32).movedim(-1, 0)[None]
            g_prev = guide_frame(order[0], prev)
            yield order[0], prev
            for t in order[1:]:
                frame = images[t].to(device, torch.float32, copy=True).movedim(-1, 0)[None]
                g = guide_frame(t, frame)
                # flow mapping guide frame t onto the previous one in order, in pixels of the flow resolution
                fl = raft(g[0], g_prev[0])[None]
                # trust the temporal term only where the flow explains the guide
                lw = warp(g_prev, fl, grid_flow, norm_flow).sub_(g).square_().sum(1, keepdim=True).mul_(-50).exp_()
                lw = F.interpolate(lw, size=(h, w), mode="bilinear").mul_(strength)
                diff = warp(prev, F.interpolate(fl, size=(h, w), mode="bilinear"), grid, norm).sub_(frame)
                if threshold > 0:
                    lw.mul_(gaussian_blur(torch.linalg.vector_norm(diff, 1, dim=1, keepdim=True), blur).div_(threshold * diff.shape[1]).square_().add_(1).reciprocal_())
                b = diff.mul_(lw)
                prev = frame.add_(screened_poisson(levels, lw.add_(mu), b)).clamp_(0, 1)
                g_prev = g
                yield t, prev

        n = min(images.shape[0], guide.shape[0])
        out = torch.empty(images.shape, dtype=comfy.model_management.intermediate_dtype(), device=comfy.model_management.intermediate_device())
        out[n:].copy_(images[n:])
        # a pinned output takes the frames by DMA without blocking; the copies stay ordered on the compute stream
        pinned = comfy.model_management.pin_memory(out, evict_active=False)
        non_blocking = pinned and comfy.model_management.device_supports_non_blocking(device)
        bidirectional = direction == "bidirectional"
        pbar = comfy.utils.ProgressBar(n * (2 if bidirectional else 1))
        try:
            for t, frame in tqdm(deflicker_pass(range(n)), total=n, desc="Video deflicker forward"):
                out[t].copy_(frame[0].movedim(0, -1), non_blocking=non_blocking)
                pbar.update(1)
            if bidirectional:
                for t, frame in tqdm(deflicker_pass(range(n)[::-1]), total=n, desc="Video deflicker backward"):
                    # frame is the pass's running state, average into a new tensor
                    mean = out[t].to(device, torch.float32, non_blocking=non_blocking).add_(frame[0].movedim(0, -1)).mul_(0.5)
                    out[t].copy_(mean, non_blocking=non_blocking)
                    pbar.update(1)
        finally:
            if pinned:
                comfy.model_management.synchronize()
                comfy.model_management.unpin_memory(out)
        return io.NodeOutput(out)


class VideoDeflickerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            VideoDeflicker,
        ]


async def comfy_entrypoint() -> VideoDeflickerExtension:
    return VideoDeflickerExtension()
