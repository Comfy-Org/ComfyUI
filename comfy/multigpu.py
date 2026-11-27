from __future__ import annotations
import itertools
import queue
import threading
import torch
import logging

from collections import namedtuple
from typing import Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.sd import VAE
import comfy.utils
import comfy.patcher_extension
import comfy.model_management
import comfy.model_prefetch


class MultiGPUThreadPool:
    """Persistent thread pool for multi-GPU work distribution.

    Maintains one worker thread per extra GPU device. Each thread calls
    set_torch_device() once at startup so that compiled kernel caches
    (inductor/triton) stay warm across diffusion steps.
    """

    def __init__(self, devices: list[torch.device]):
        self._workers: list[threading.Thread] = []
        self._work_queues: dict[torch.device, queue.Queue] = {}
        self._result_queues: dict[torch.device, queue.Queue] = {}

        for device in devices:
            wq = queue.Queue()
            rq = queue.Queue()
            self._work_queues[device] = wq
            self._result_queues[device] = rq
            t = threading.Thread(target=self._worker_loop, args=(device, wq, rq), daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_loop(self, device: torch.device, work_q: queue.Queue, result_q: queue.Queue):
        try:
            comfy.model_management.set_torch_device(device)
        except Exception as e:
            logging.error(f"MultiGPUThreadPool: failed to set device {device}: {e}")
            while True:
                item = work_q.get()
                if item is None:
                    return
                result_q.put((None, e))
            return
        try:
            while True:
                item = work_q.get()
                if item is None:
                    break
                fn, args, kwargs = item
                try:
                    result = fn(*args, **kwargs)
                    result_q.put((result, None))
                except comfy.model_management.InterruptProcessingException as e:
                    result_q.put((None, e))
                except Exception as e:
                    result_q.put((None, e))
        finally:
            comfy.model_prefetch.cleanup_malloc_graph()

    def submit(self, device: torch.device, fn, *args, **kwargs):
        self._work_queues[device].put((fn, args, kwargs))

    def get_result(self, device: torch.device):
        return self._result_queues[device].get()

    @property
    def devices(self) -> list[torch.device]:
        return list(self._work_queues.keys())

    def shutdown(self):
        for wq in self._work_queues.values():
            wq.put(None)  # sentinel
        for t in self._workers:
            t.join(timeout=5.0)

    def map_ordered(self, items, device_fns: dict[torch.device, Callable], max_in_flight: int = None):
        """Run ``items`` across pool devices via a dynamic work queue and yield results in input order.

        Acquires a bounded in-flight permit before popping from ``task_q`` so permits always
        belong to the lowest unassigned indices, preventing order-inversion deadlocks.
        """
        items = list(items)
        if not items:
            return

        devices = [d for d in self.devices if d in device_fns]
        if len(devices) < 2 or len(items) == 1:
            dev = devices[0] if devices else next(iter(device_fns))
            fn = device_fns[dev]
            with comfy.model_management.cuda_device_context(dev):
                for item in items:
                    yield fn(item)
            return

        permits = max(len(devices), max_in_flight if max_in_flight is not None else len(devices) * 2)
        slot_sem = threading.Semaphore(permits)
        task_q: queue.Queue = queue.Queue()
        done_q: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        grad_enabled = torch.is_grad_enabled()
        for idx, item in enumerate(items):
            task_q.put((idx, item))

        def _drain(dev, fn):
            with comfy.model_management.cuda_device_context(dev), torch.set_grad_enabled(grad_enabled):
                while not stop_event.is_set():
                    slot_sem.acquire()
                    if stop_event.is_set():
                        slot_sem.release()
                        break
                    try:
                        idx, item = task_q.get_nowait()
                    except queue.Empty:
                        slot_sem.release()
                        break
                    try:
                        res = fn(item)
                    except BaseException as err:
                        stop_event.set()
                        slot_sem.release()
                        done_q.put((idx, None, err))
                        break
                    done_q.put((idx, res, None))
                comfy.model_management.synchronize()

        for dev in devices:
            self.submit(dev, _drain, dev, device_fns[dev])

        had_error = False
        try:
            ready = {}
            next_idx = 0
            completed = 0
            total = len(items)
            while completed < total:
                idx, res, err = done_q.get()
                if err is not None:
                    stop_event.set()
                    raise err
                completed += 1
                ready[idx] = res
                while next_idx in ready:
                    val = ready.pop(next_idx)
                    next_idx += 1
                    slot_sem.release()
                    yield val
        except BaseException:
            had_error = True
            raise
        finally:
            stop_event.set()
            while True:
                try:
                    task_q.get_nowait()
                except queue.Empty:
                    break
            for _ in range(len(devices) + permits):
                slot_sem.release()
            first_worker_err = None
            for dev in devices:
                _, worker_err = self.get_result(dev)
                if worker_err is not None and first_worker_err is None:
                    first_worker_err = worker_err
            if not had_error and first_worker_err is not None:
                raise first_worker_err


class GPUOptions:
    def __init__(self, device_index: int, relative_speed: float):
        self.device_index = device_index
        self.relative_speed = relative_speed

    def clone(self):
        return GPUOptions(self.device_index, self.relative_speed)

    def create_dict(self):
        return {
            "relative_speed": self.relative_speed
        }

class GPUOptionsGroup:
    def __init__(self):
        self.options: dict[int, GPUOptions] = {}

    def add(self, info: GPUOptions):
        self.options[info.device_index] = info

    def clone(self):
        c = GPUOptionsGroup()
        for opt in self.options.values():
            c.add(opt)
        return c

    def register(self, model: ModelPatcher):
        opts_dict = {}
        # get devices that are valid for this model
        devices: list[torch.device] = [model.load_device]
        for extra_model in model.get_additional_models_with_key("multigpu"):
            extra_model: ModelPatcher
            devices.append(extra_model.load_device)
        # create dictionary with actual device mapped to its GPUOptions
        device_opts_list: list[GPUOptions] = []
        for device in devices:
            device_opts = self.options.get(device.index, GPUOptions(device_index=device.index, relative_speed=1.0))
            opts_dict[device] = device_opts.create_dict()
            device_opts_list.append(device_opts)
        # make relative_speed relative to 1.0
        min_speed = min([x.relative_speed for x in device_opts_list])
        for value in opts_dict.values():
            value['relative_speed'] /= min_speed
        model.model_options['multigpu_options'] = opts_dict


def create_multigpu_deepclones(model: ModelPatcher, max_gpus: int, gpu_options: GPUOptionsGroup=None, reuse_loaded=False):
    'Prepare ModelPatcher to contain deepclones of its BaseModel and related properties.'
    model = model.clone()
    # check if multigpu is already prepared - get the load devices from them if possible to exclude
    skip_devices = set()
    multigpu_models = model.get_additional_models_with_key("multigpu")
    if len(multigpu_models) > 0:
        for mm in multigpu_models:
            skip_devices.add(mm.load_device)
    skip_devices = list(skip_devices)

    # Exclude the primary model's actual device, not the global current device:
    # after SelectModelDevice(gpu:N) the primary may not live on the process's
    # current CUDA device, and excluding the wrong device picks bad extras.
    all_devices = comfy.model_management.get_all_torch_devices(exclude_current=False)
    full_extra_devices = [d for d in all_devices if d != model.load_device]
    limit_extra_devices = full_extra_devices[:max_gpus-1]
    extra_devices = limit_extra_devices.copy()
    # exclude skipped devices
    for skip in skip_devices:
        if skip in extra_devices:
            extra_devices.remove(skip)
    # create new deepclones
    if len(extra_devices) > 0:
        for device in extra_devices:
            device_patcher = None
            if reuse_loaded:
                # Only reuse a previously-loaded MultiGPU clone. A SelectModelDevice
                # patcher on the same device shares clone_base_uuid but has
                # is_multigpu_base_clone=False, which would later be filtered out by
                # prepare_model_patcher_multigpu_clones() and silently shrink the
                # work split back to one GPU.
                loaded_models: list[ModelPatcher] = comfy.model_management.loaded_models()
                for lm in loaded_models:
                    if lm.model is None:
                        continue
                    if lm.load_device != device:
                        continue
                    if lm.clone_base_uuid != model.clone_base_uuid:
                        continue
                    if not getattr(lm, "is_multigpu_base_clone", False):
                        continue
                    device_patcher = lm.clone()
                    logging.info(f"Reusing loaded multigpu deepclone of {device_patcher.model.__class__.__name__} for {device}")
                    break
            if device_patcher is None:
                device_patcher = model.deepclone_multigpu(new_load_device=device)
            # Always flag the clone; whether reused or freshly deepcloned, it must
            # advertise itself as a MultiGPU base clone so the cond scheduler picks
            # it up in prepare_model_patcher_multigpu_clones().
            device_patcher.is_multigpu_base_clone = True
            multigpu_models = model.get_additional_models_with_key("multigpu")
            multigpu_models.append(device_patcher)
            model.set_additional_models("multigpu", multigpu_models)
        model.match_multigpu_clones()
        if gpu_options is None:
            gpu_options = GPUOptionsGroup()
        gpu_options.register(model)
    else:
        logging.info("No extra torch devices need initialization, skipping initializing MultiGPU Work Units.")
    # only keep model clones that don't go 'past' the intended max_gpu count;
    # this prunes any inherited multigpu clones whose load_device is no longer allowed
    # when max_gpus is lowered between runs.
    allowed_devices = set(limit_extra_devices)
    allowed_devices.add(model.load_device)
    multigpu_models = model.get_additional_models_with_key("multigpu")
    new_multigpu_models = [m for m in multigpu_models if m.load_device in allowed_devices]
    if len(new_multigpu_models) != len(multigpu_models):
        model.set_additional_models("multigpu", new_multigpu_models)
        model.match_multigpu_clones()
    return model


LoadBalance = namedtuple('LoadBalance', ['work_per_device', 'idle_time'])
def load_balance_devices(model_options: dict[str], total_work: int, return_idle_time=False, work_normalized: int=None):
    'Optimize work assigned to different devices, accounting for their relative speeds and splittable work.'
    opts_dict = model_options['multigpu_options']
    devices = list(model_options['multigpu_clones'].keys())
    speed_per_device = []
    work_per_device = []
    # get sum of each device's relative_speed
    total_speed = 0.0
    for opts in opts_dict.values():
        total_speed += opts['relative_speed']
    # get relative work for each device;
    # obtained by w = (W*r)/R
    for device in devices:
        relative_speed = opts_dict[device]['relative_speed']
        relative_work = (total_work*relative_speed) / total_speed
        speed_per_device.append(relative_speed)
        work_per_device.append(relative_work)
    # relative work must be expressed in whole numbers, but likely is a decimal;
    # perform rounding while maintaining total sum equal to total work (sum of relative works)
    work_per_device = round_preserved(work_per_device)
    dict_work_per_device = {}
    for device, relative_work in zip(devices, work_per_device):
        dict_work_per_device[device] = relative_work
    if not return_idle_time:
        return LoadBalance(dict_work_per_device, None)
    # divide relative work by relative speed to get estimated completion time of said work by each device;
    # time here is relative and does not correspond to real-world units
    completion_time = [w/r for w,r in zip(work_per_device, speed_per_device)]
    # calculate relative time spent by the devices waiting on each other after their work is completed
    idle_time = abs(min(completion_time) - max(completion_time))
    # if need to compare work idle time, need to normalize to a common total work
    if work_normalized:
        idle_time *= (work_normalized/total_work)

    return LoadBalance(dict_work_per_device, idle_time)

def round_preserved(values: list[float]):
    'Round all values in a list, preserving the combined sum of values.'
    # get floor of values; casting to int does it too
    floored = [int(x) for x in values]
    total_floored = sum(floored)
    # get remainder to distribute
    remainder = round(sum(values)) - total_floored
    # pair values with fractional portions
    fractional = [(i, x-floored[i]) for i, x in enumerate(values)]
    # sort by fractional part in descending order
    fractional.sort(key=lambda x: x[1], reverse=True)
    # distribute the remainder
    for i in range(remainder):
        index = fractional[i][0]
        floored[index] += 1
    return floored


def tiled_scale_multidim_multigpu(samples, functions: dict[torch.device, Callable], tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", downscale=False, index_formulas=None, pbar=None, pool: MultiGPUThreadPool = None):
    """Multi-GPU variant of ``comfy.utils.tiled_scale_multidim`` using a dynamic work queue.

    Dispatches tile coordinates across devices in ``functions`` via ``pool.map_ordered`` and
    accumulates into a single shared ``output`` and ``out_div`` buffer on the main thread.
    """
    if len(functions) < 2:
        return comfy.utils.tiled_scale_multidim(
            samples,
            next(iter(functions.values())),
            tile=tile,
            overlap=overlap,
            upscale_amount=upscale_amount,
            out_channels=out_channels,
            output_device=output_device,
            downscale=downscale,
            index_formulas=index_formulas,
            pbar=pbar,
        )

    dims = len(tile)

    if not isinstance(upscale_amount, (tuple, list)):
        upscale_amount = [upscale_amount] * dims
    if not isinstance(overlap, (tuple, list)):
        overlap = [overlap] * dims
    if index_formulas is None:
        index_formulas = upscale_amount
    if not isinstance(index_formulas, (tuple, list)):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        return up(val) if callable(up) else up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        return up(val) if callable(up) else val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        return up(val) if callable(up) else up * val

    def get_downscale_pos(dim, val):
        up = index_formulas[dim]
        return up(val) if callable(up) else val / up

    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos

    def mult_list_upscale(a):
        return [round(get_scale(i, a[i])) for i in range(len(a))]

    output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)
    samples_cpu = samples if samples.device.type == "cpu" else samples.to("cpu")

    owns_pool = pool is None
    if owns_pool:
        pool = MultiGPUThreadPool(list(functions.keys()))

    try:
        for b in range(samples_cpu.shape[0]):
            s = samples_cpu[b:b + 1]

            if all(s.shape[d + 2] <= tile[d] for d in range(dims)):
                primary_dev = next(iter(functions))
                with comfy.model_management.cuda_device_context(primary_dev):
                    output[b:b + 1] = functions[primary_dev](s).to(output_device)
                if pbar is not None:
                    pbar.update(1)
                continue

            out = output[b:b + 1].zero_()
            out_div = torch.zeros([s.shape[0], 1] + mult_list_upscale(s.shape[2:]), device=output_device)

            positions = [range(0, s.shape[d + 2] - overlap[d], tile[d] - overlap[d]) if s.shape[d + 2] > tile[d] else [0] for d in range(dims)]
            all_positions = list(itertools.product(*positions))

            def _make_tile_fn(fn):
                def _eval_tile(it):
                    s_in = s
                    upscaled = []
                    for d in range(dims):
                        pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                        length = min(tile[d], s.shape[d + 2] - pos)
                        s_in = s_in.narrow(d + 2, pos, length)
                        upscaled.append(round(get_pos(d, pos)))

                    ps = fn(s_in).to(output_device)
                    mask = torch.ones([1, 1] + list(ps.shape[2:]), device=output_device)

                    for d in range(2, dims + 2):
                        feather = round(get_scale(d - 2, overlap[d - 2]))
                        if feather >= mask.shape[d]:
                            continue
                        for t in range(feather):
                            a = (t + 1) / feather
                            mask.narrow(d, t, 1).mul_(a)
                            mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

                    return upscaled, ps, mask

                return _eval_tile

            tile_fns = {dev: _make_tile_fn(fn) for dev, fn in functions.items()}
            for upscaled, ps, mask in pool.map_ordered(all_positions, tile_fns):
                o = out
                o_d = out_div
                ps_view = ps
                mask_view = mask
                for d in range(dims):
                    length = min(ps_view.shape[d + 2], o.shape[d + 2] - upscaled[d])
                    o = o.narrow(d + 2, upscaled[d], length)
                    o_d = o_d.narrow(d + 2, upscaled[d], length)
                    if length < ps_view.shape[d + 2]:
                        ps_view = ps_view.narrow(d + 2, 0, length)
                        mask_view = mask_view.narrow(d + 2, 0, length)

                o.add_(ps_view * mask_view)
                o_d.add_(mask_view)

                if pbar is not None:
                    pbar.update(1)

            out.div_(out_div)
    finally:
        if owns_pool:
            pool.shutdown()

    return output


def create_vae_multigpu_deepclones(vae: VAE, max_gpus: int, reuse_loaded: bool = True) -> VAE:
    """Prepare a VAE wrapper with per-device deepclones for multi-GPU encode/decode."""
    vae.throw_exception_if_invalid()
    cloned = vae.clone()
    primary_device = cloned.patcher.load_device
    cloned.device = primary_device
    if max_gpus <= 1 or comfy.model_management.is_device_cpu(primary_device):
        cloned.multigpu_clones = {}
        return cloned

    all_devices = comfy.model_management.get_all_torch_devices(exclude_current=False)
    extra_devices = [d for d in all_devices if d != primary_device][:max_gpus - 1]
    if not extra_devices:
        logging.info("No extra torch devices need initialization, skipping initializing MultiGPU VAE.")
        cloned.multigpu_clones = {}
        return cloned

    allowed = set(extra_devices)
    clones: dict[torch.device, VAE] = {d: c.clone() for d, c in cloned.multigpu_clones.items() if d in allowed}
    has_manual_dtype = "manual_cast_dtype" in cloned.patcher.object_patches
    manual_dtype = cloned.patcher.object_patches.get("manual_cast_dtype", None)

    for device in extra_devices:
        if device not in clones:
            device_patcher = None
            if reuse_loaded:
                loaded_models: list[ModelPatcher] = comfy.model_management.loaded_models()
                for lm in loaded_models:
                    if lm.model is None:
                        continue
                    if lm.load_device != device:
                        continue
                    if lm.clone_base_uuid != cloned.patcher.clone_base_uuid:
                        continue
                    if not getattr(lm, "is_multigpu_base_clone", False):
                        continue
                    device_patcher = lm.clone()
                    logging.info(f"Reusing loaded multigpu deepclone of {device_patcher.model.__class__.__name__} for {device}")
                    break
            if device_patcher is None:
                device_patcher = cloned.patcher.deepclone_multigpu(new_load_device=device)
            device_patcher.offload_device = comfy.model_management.vae_offload_device()
            device_patcher.is_multigpu_base_clone = True

            clone_vae = cloned.clone()
            clone_vae.multigpu_clones = {}
            clone_vae.patcher = device_patcher
            clone_vae.first_stage_model = device_patcher.model
            clone_vae.first_stage_model.current_patcher = device_patcher
            clone_vae.device = device
            if not has_manual_dtype and not comfy.model_management.supports_dtype(device, clone_vae.vae_dtype):
                clone_vae.vae_dtype = comfy.model_management.vae_dtype(device, clone_vae.working_dtypes)
            clones[device] = clone_vae

        if has_manual_dtype:
            clones[device].vae_dtype = cloned.vae_dtype
            clones[device].patcher.set_model_compute_dtype(manual_dtype)

    cloned.multigpu_clones = clones
    return cloned
