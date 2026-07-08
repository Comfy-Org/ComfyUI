import comfy.options
comfy.options.enable_args_parsing()

from comfy.cli_args import args

if args.list_feature_flags:
    import json
    from comfy_api.feature_flags import CLI_FEATURE_FLAG_REGISTRY
    print(json.dumps(CLI_FEATURE_FLAG_REGISTRY, indent=2))  # noqa: T201
    raise SystemExit(0)

import os
import importlib.util
import shutil
import importlib.metadata
import folder_paths
import time
from comfy.cli_args import enables_dynamic_vram
from app.logger import setup_logger
setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

from app.assets.seeder import asset_seeder
from app.assets.services import register_output_files
import itertools
import utils.extra_config
from utils.mime_types import init_mime_types
import faulthandler
import logging
import signal
import sys
from comfy_execution.progress import get_progress_state
from comfy_execution.utils import get_executing_context
from comfy_api import feature_flags
from app.database.db import init_db, dependencies_available

if __name__ == "__main__":
    #NOTE: These do not do anything on core ComfyUI, they are for custom nodes.
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

faulthandler.enable(file=sys.stderr, all_threads=args.debug_hang)
if __name__ == "__main__" and args.debug_hang:
    dumping_traceback = False

    def dump_traceback_on_sigint(signum, frame):
        global dumping_traceback
        if dumping_traceback:
            raise KeyboardInterrupt
        dumping_traceback = True
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, dump_traceback_on_sigint)

import comfy_aimdo.control

if enables_dynamic_vram():
    try:
        comfy_aimdo.control.init(simple_vram_headroom=None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3))
    except TypeError:
        # comfy-aimdo 0.4.9 protocol.
        comfy_aimdo.control.init()

if os.name == "nt":
    os.environ['MIMALLOC_PURGE_DELAY'] = '0'

if __name__ == "__main__":
    os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
    if args.default_device is not None:
        default_dev = args.default_device
        devices = list(range(32))
        devices.remove(default_dev)
        devices.insert(0, default_dev)
        devices = ','.join(map(str, devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(devices)
        os.environ['HIP_VISIBLE_DEVICES'] = str(devices)

    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
        logging.info("Set oneapi device selector to: {}".format(args.oneapi_device_selector))

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc
    if "rocm" in cuda_malloc.get_torch_version_noimport():
        os.environ['OCL_SET_SVM_SIZE'] = '262144'  # set at the request of AMD


def handle_comfyui_manager_unavailable():
    manager_req_path = os.path.join(os.path.dirname(os.path.abspath(folder_paths.__file__)), "manager_requirements.txt")
    uv_available = shutil.which("uv") is not None

    pip_cmd = f"{sys.executable} -m pip install -r {manager_req_path}"
    msg = f"\n\nTo use the `--enable-manager` feature, the `comfyui-manager` package must be installed first.\ncommand:\n\t{pip_cmd}"
    if uv_available:
        msg += f"\nor using uv:\n\tuv pip install -r {manager_req_path}"
    msg += "\n"
    logging.warning(msg)
    args.enable_manager = False


if args.enable_manager:
    if importlib.util.find_spec("comfyui_manager"):
        import comfyui_manager

        if not comfyui_manager.__file__ or not comfyui_manager.__file__.endswith('__init__.py'):
            handle_comfyui_manager_unavailable()
    else:
        handle_comfyui_manager_unavailable()


def apply_custom_paths():
    # extra model paths
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    # --base-directory
    if args.base_directory:
        logging.info(f"Setting base directory to: {folder_paths.base_path}")

    # --output-directory, --input-directory, --user-directory
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)


def execute_prestartup_script():
    if args.disable_all_custom_nodes and len(args.whitelist_custom_nodes) == 0:
        return

    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)

            if args.enable_manager:
                if comfyui_manager.should_be_disabled(module_path):
                    continue

            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                    logging.info(f"Prestartup Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                    continue
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        logging.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")

apply_custom_paths()
init_mime_types()

if args.enable_manager:
    comfyui_manager.prestartup()

execute_prestartup_script()


# Main code
import asyncio
import threading
import gc

if 'torch' in sys.modules:
    logging.warning("WARNING: Potential Error in code: Torch already imported, torch should never be imported before this point.")


import comfy.utils

import execution
import server
from protocol import BinaryEventTypes
import nodes
import comfy.model_management
import comfyui_version
import app.logger
import hook_breaker_ac10a0

import comfy.memory_management
import comfy.model_patcher

if args.enable_dynamic_vram or (enables_dynamic_vram() and comfy.model_management.is_nvidia() and not comfy.model_management.is_wsl()):
    if (not args.enable_dynamic_vram) and (comfy.model_management.torch_version_numeric < (2, 8)):
        logging.warning("Unsupported Pytorch detected. DynamicVRAM support requires Pytorch version 2.8 or later. Falling back to legacy ModelPatcher. VRAM estimates may be unreliable especially on Windows")
    else:
        try:
            aimdo_initialized = comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in comfy.model_management.get_all_torch_devices())
        except TypeError:
            # comfy-aimdo 0.4.9 protocol.
            aimdo_initialized = comfy_aimdo.control.init_devices(d.index for d in comfy.model_management.get_all_torch_devices())

        if aimdo_initialized:
            if args.verbose == 'DEBUG':
                comfy_aimdo.control.set_log_debug()
            elif args.verbose == 'CRITICAL':
                comfy_aimdo.control.set_log_critical()
            elif args.verbose == 'ERROR':
                comfy_aimdo.control.set_log_error()
            elif args.verbose == 'WARNING':
                comfy_aimdo.control.set_log_warning()
            else: #INFO
                comfy_aimdo.control.set_log_info()

            comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
            comfy.memory_management.aimdo_enabled = True
            logging.info("DynamicVRAM support detected and enabled")
        else:
            logging.warning("No working comfy-aimdo install detected. DynamicVRAM support disabled. Falling back to legacy ModelPatcher. VRAM estimates may be unreliable especially on Windows")


def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


def _collect_output_absolute_paths(history_result: dict) -> list[str]:
    """Extract absolute file paths for output items from a history result."""
    paths: list[str] = []
    seen: set[str] = set()
    for node_output in history_result.get("outputs", {}).values():
        for items in node_output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type not in ("output", "temp"):
                    continue
                base_dir = folder_paths.get_directory_by_type(item_type)
                if base_dir is None:
                    continue
                base_dir = os.path.abspath(base_dir)
                filename = item.get("filename")
                if not filename:
                    continue
                abs_path = os.path.abspath(
                    os.path.join(base_dir, item.get("subfolder", ""), filename)
                )
                if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
                    continue
                if abs_path not in seen:
                    seen.add(abs_path)
                    paths.append(abs_path)
    return paths


def prompt_worker(q, server_instance):
    current_time: float = 0.0
    cache_ram = 0
    cache_ram_inactive = 0
    if not args.cache_classic and not args.cache_none and args.cache_lru <= 0:
        cache_ram = min(10.0, max(2.0, comfy.model_management.total_ram * 0.10 / 1024.0))
        cache_ram_inactive = min(96.0, comfy.model_management.total_ram / 1024.0)
        if len(args.cache_ram) > 0:
            cache_ram = args.cache_ram[0]
        if len(args.cache_ram) > 1:
            cache_ram_inactive = args.cache_ram[1]

    cache_type = execution.CacheType.RAM_PRESSURE
    if args.cache_classic:
        cache_type = execution.CacheType.CLASSIC
    elif args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_none:
        cache_type = execution.CacheType.NONE

    e = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_args={ "lru" : args.cache_lru, "ram" : cache_ram, "ram_inactive" : cache_ram_inactive } )
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server_instance.last_prompt_id = prompt_id

            sensitive = item[5]
            extra_data = item[3].copy()
            for k in sensitive:
                extra_data[k] = sensitive[k]

            asset_seeder.pause()
            e.execute(item[2], prompt_id, extra_data, item[4])

            need_gc = True

            remove_sensitive = lambda prompt: prompt[:5] + prompt[6:]
            q.task_done(item_id,
                        e.history_result,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages), process_item=remove_sensitive)
            if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time

            # Log Time in a more readable way after 10 minutes
            if execution_time > 600:
                execution_time = time.strftime("%H:%M:%S", time.gmtime(execution_time))
                logging.info(f"Prompt executed in {execution_time}", extra={'color': 'green'})
            else:
                logging.info("Prompt executed in {:.2f} seconds".format(execution_time), extra={'color': 'green'})

            if not asset_seeder.is_disabled():
                paths = _collect_output_absolute_paths(e.history_result)
                register_output_files(paths, job_id=prompt_id)

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False
                hook_breaker_ac10a0.restore_functions()

                if not asset_seeder.is_disabled():
                    asset_seeder.enqueue_enrich(roots=("output",), compute_hashes=args.enable_asset_hashing)
                asset_seeder.resume()


async def run(server_instance, address='', port=8188, verbose=True, call_on_start=None):
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(
        server_instance.start_multi_address(addresses, call_on_start, verbose), server_instance.publish_loop()
    )

def hijack_progress(server_instance):
    def hook(value, total, preview_image, prompt_id=None, node_id=None):
        executing_context = get_executing_context()
        if prompt_id is None and executing_context is not None:
            prompt_id = executing_context.prompt_id
        if node_id is None and executing_context is not None:
            node_id = executing_context.node_id
        comfy.model_management.throw_exception_if_processing_interrupted()
        if prompt_id is None:
            prompt_id = server_instance.last_prompt_id
        if node_id is None:
            node_id = server_instance.last_node_id
        progress = {"value": value, "max": total, "prompt_id": prompt_id, "node": node_id}
        get_progress_state().update_progress(node_id, value, total, preview_image)

        server_instance.send_sync("progress", progress, server_instance.client_id)
        if preview_image is not None:
            # Only send old method if client doesn't support preview metadata
            if not feature_flags.supports_feature(
                server_instance.sockets_metadata,
                server_instance.client_id,
                "supports_preview_metadata",
            ):
                server_instance.send_sync(
                    BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                    preview_image,
                    server_instance.client_id,
                )

    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def setup_database():
    try:
        if dependencies_available():
            init_db()
            if args.enable_assets:
                if asset_seeder.start(roots=("models", "input", "output"), prune_first=True, compute_hashes=args.enable_asset_hashing):
                    logging.info("Background asset scan initiated for models, input, output")
    except Exception as e:
        if "database is locked" in str(e):
            logging.error(
                "Database is locked. Another ComfyUI process is already using this database.\n"
                "To resolve this, specify a separate database file for this instance:\n"
                "  --database-url sqlite:///path/to/another.db"
            )
            sys.exit(1)
        if args.enable_assets:
            logging.error(
                f"Failed to initialize database: {e}\n"
                "The --enable-assets flag requires a working database connection.\n"
                "To resolve this, try one of the following:\n"
                "  1. Install the latest requirements: pip install -r requirements.txt\n"
                "  2. Specify an alternative database URL: --database-url sqlite:///path/to/your.db\n"
                "  3. Use an in-memory database: --database-url sqlite:///:memory:"
            )
            sys.exit(1)
        logging.error(f"Failed to initialize database. Please ensure you have installed the latest requirements. If the error persists, please report this as in future the database will be required: {e}")


def start_comfyui(asyncio_loop=None):
    """
    Starts the ComfyUI server using the provided asyncio event loop or creates a new one.
    Returns the event loop, server instance, and a function to start the server asynchronously.
    """
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if not asyncio_loop:
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)
    prompt_server = server.PromptServer(asyncio_loop)

    if args.enable_manager and not args.disable_manager_ui:
        comfyui_manager.start()

    hook_breaker_ac10a0.save_functions()
    asyncio_loop.run_until_complete(nodes.init_extra_nodes(
        init_custom_nodes=(not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
        init_api_nodes=not args.disable_api_nodes
    ))

    # Re-apply Comfy's cuDNN benchmark policy after custom-node imports. Benchmark
    # mode can request near-card-sized autotune workspaces, and some custom nodes set it at import time.
    comfy.model_management.set_cudnn_benchmark()

    hook_breaker_ac10a0.restore_functions()

    cuda_malloc_warning()
    setup_database()

    prompt_server.add_routes()
    hijack_progress(prompt_server)

    threading.Thread(target=prompt_worker, daemon=True, args=(prompt_server.prompt_queue, prompt_server,)).start()

    if args.quick_test_for_ci:
        exit(0)

    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server

    async def start_all():
        await prompt_server.setup()
        await run(prompt_server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start)

    # Returning these so that other code can integrate with the ComfyUI loop and server
    return asyncio_loop, prompt_server, start_all


if __name__ == "__main__":
    # Running directly, just start ComfyUI.
    logging.info("Python version: {}".format(sys.version))
    logging.info("ComfyUI version: {}".format(comfyui_version.__version__))
    for package in ("comfy-aimdo", "comfy-kitchen"):
        try:
            logging.info("{} version: {}".format(package, importlib.metadata.version(package)))
        except:
            pass

    if sys.version_info.major == 3 and sys.version_info.minor < 10:
        logging.warning("WARNING: You are using a python version older than 3.10, please upgrade to a newer one. 3.12 and above is recommended.")

    if args.disable_dynamic_vram:
        logging.warning(
            "Dynamic vram disabled with argument. If you have any issues with "
            "dynamic vram enabled please give us a detailed reports as this "
            "argument will be removed soon. If you use gguf we recommend keeping "
            "dynamic vram enabled and using native ComfyUI model formats instead. "
            "ComfyUI native formats like fp8 will be faster even if they are larger than your memory."
        )
    event_loop, _, start_all_func = start_comfyui()
    try:
        x = start_all_func()
        app.logger.print_startup_warnings()
        event_loop.run_until_complete(x)
    except KeyboardInterrupt:
        logging.info("\nStopped server")
    finally:
        asset_seeder.shutdown()
        cleanup_temp()
