import argparse
import enum
import os
import comfy.options


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


parser = argparse.ArgumentParser()

parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0,::", help="Specify the IP address to listen on (default: 127.0.0.1). You can give a list of ip addresses by separating them with a comma like: 127.2.2.2,127.3.3.3 If --listen is provided without an argument, it defaults to 0.0.0.0,:: (listens on all ipv4 and ipv6)")
parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
parser.add_argument("--tls-keyfile", type=str, help="Path to TLS (SSL) key file. Enables TLS, makes app accessible at https://... requires --tls-certfile to function")
parser.add_argument("--tls-certfile", type=str, help="Path to TLS (SSL) certificate file. Enables TLS, makes app accessible at https://... requires --tls-keyfile to function")
parser.add_argument("--enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*", help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
parser.add_argument("--max-upload-size", type=float, default=100, help="Set the maximum upload size in MB.")

parser.add_argument("--base-directory", type=str, default=None, help="Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories.")
parser.add_argument("--extra-model-paths-config", type=str, default=None, metavar="PATH", nargs='+', action='append', help="Load one or more extra_model_paths.yaml files.")
parser.add_argument("--output-directory", type=str, default=None, help="Set the ComfyUI output directory. Overrides --base-directory.")
parser.add_argument("--temp-directory", type=str, default=None, help="Set the ComfyUI temp directory (default is in the ComfyUI directory). Overrides --base-directory.")
parser.add_argument("--input-directory", type=str, default=None, help="Set the ComfyUI input directory. Overrides --base-directory.")
parser.add_argument("--auto-launch", action="store_true", help="Automatically launch ComfyUI in the default browser.")
parser.add_argument("--disable-auto-launch", action="store_true", help="Disable auto launching the browser.")
parser.add_argument("--cuda-device", type=str, default=None, metavar="DEVICE_ID", help="Set the ids of cuda devices this instance will use, as a comma-separated list (e.g. '0' or '0,1'). All other devices will not be visible.")
parser.add_argument("--default-device", type=int, default=None, metavar="DEFAULT_DEVICE_ID", help="Set the id of the default device, all other devices will stay visible.")
cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument("--cuda-malloc", action="store_true", help="Enable cudaMallocAsync (enabled by default for torch 2.0 and up).")
cm_group.add_argument("--disable-cuda-malloc", action="store_true", help="Disable cudaMallocAsync.")


fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--force-fp32", action="store_true", help="Force fp32 (If this makes your GPU work better please report it).")
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--fp32-unet", action="store_true", help="Run the diffusion model in fp32.")
fpunet_group.add_argument("--fp64-unet", action="store_true", help="Run the diffusion model in fp64.")
fpunet_group.add_argument("--bf16-unet", action="store_true", help="Run the diffusion model in bf16.")
fpunet_group.add_argument("--fp16-unet", action="store_true", help="Run the diffusion model in fp16")
fpunet_group.add_argument("--fp8_e4m3fn-unet", action="store_true", help="Store unet weights in fp8_e4m3fn.")
fpunet_group.add_argument("--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2.")
fpunet_group.add_argument("--fp8_e8m0fnu-unet", action="store_true", help="Store unet weights in fp8_e8m0fnu.")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--fp16-vae", action="store_true", help="Run the VAE in fp16, might cause black images.")
fpvae_group.add_argument("--fp32-vae", action="store_true", help="Run the VAE in full precision fp32.")
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16.")

parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU.")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--fp8_e4m3fn-text-enc", action="store_true", help="Store text encoder weights in fp8 (e4m3fn variant).")
fpte_group.add_argument("--fp8_e5m2-text-enc", action="store_true", help="Store text encoder weights in fp8 (e5m2 variant).")
fpte_group.add_argument("--fp16-text-enc", action="store_true", help="Store text encoder weights in fp16.")
fpte_group.add_argument("--fp32-text-enc", action="store_true", help="Store text encoder weights in fp32.")
fpte_group.add_argument("--bf16-text-enc", action="store_true", help="Store text encoder weights in bf16.")

parser.add_argument("--fp16-intermediates", action="store_true", help="Experimental: Use fp16 for intermediate tensors between nodes instead of fp32.")

parser.add_argument("--force-channels-last", action="store_true", help="Force channels last format when inferencing the models.")

parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1, help="Use torch-directml.")

parser.add_argument("--oneapi-device-selector", type=str, default=None, metavar="SELECTOR_STRING", help="Sets the oneAPI device(s) this instance will use.")
parser.add_argument("--supports-fp8-compute", action="store_true", help="ComfyUI will act like if the device supports fp8 compute.")
parser.add_argument("--enable-triton-backend", action="store_true", help="ComfyUI will enable the use of Triton backend in comfy-kitchen. Is disabled at launch by default.")

class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"

    @classmethod
    def from_string(cls, value: str):
        for member in cls:
            if member.value == value:
                return member
        return None

parser.add_argument("--preview-method", type=LatentPreviewMethod, default=LatentPreviewMethod.NoPreviews, help="Default preview method for sampler nodes.", action=EnumAction)

parser.add_argument("--preview-size", type=int, default=512, help="Sets the maximum preview size for sampler nodes.")

cache_group = parser.add_mutually_exclusive_group()
cache_group.add_argument("--cache-ram", nargs='*', type=float, default=[], metavar="GB", help="Use RAM pressure caching with the specified headroom thresholds. This is the default caching mode. The first value sets the active-cache threshold; the optional second value sets the inactive-cache/pin threshold. Defaults when no values are provided: active 10%% of system RAM (min 2GB, max 10GB), inactive 100%% of system RAM (max 96GB).")
cache_group.add_argument("--cache-classic", action="store_true", help="Use the old style (aggressive) caching.")
cache_group.add_argument("--cache-lru", type=int, default=0, help="Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.")
cache_group.add_argument("--cache-none", action="store_true", help="Reduced RAM/VRAM usage at the expense of executing every node for each run.")
cache_group.add_argument("--high-ram", action="store_true", help="Can improve performance slightly on high RAM or on systems where pagefile use is preferred over model loading.")

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--use-split-cross-attention", action="store_true", help="Use the split cross attention optimization. Ignored when xformers is used.")
attn_group.add_argument("--use-quad-cross-attention", action="store_true", help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.")
attn_group.add_argument("--use-pytorch-cross-attention", action="store_true", help="Use the new pytorch 2.0 cross attention function.")
attn_group.add_argument("--use-sage-attention", action="store_true", help="Use sage attention.")
attn_group.add_argument("--use-flash-attention", action="store_true", help="Use FlashAttention.")

parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

upcast = parser.add_mutually_exclusive_group()
upcast.add_argument("--force-upcast-attention", action="store_true", help="Force enable attention upcasting, please report if it fixes black images.")
upcast.add_argument("--dont-upcast-attention", action="store_true", help="Disable all upcasting of attention. Should be unnecessary except for debugging.")


parser.add_argument("--enable-manager", action="store_true", help="Enable the ComfyUI-Manager feature.")
manager_group = parser.add_mutually_exclusive_group()
manager_group.add_argument("--disable-manager-ui", action="store_true", help="Disables only the ComfyUI-Manager UI and endpoints. Scheduled installations and similar background tasks will still operate.")
manager_group.add_argument("--enable-manager-legacy-ui", action="store_true", help="Enables the legacy UI of ComfyUI-Manager. Implies --enable-manager.")


vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--gpu-only", action="store_true", help="Store and run everything (text encoders/CLIP models, etc... on the GPU).")
vram_group.add_argument("--highvram", action="store_true", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
vram_group.add_argument("--lowvram", action="store_true", help="Doesn't do anything if dynamic vram is enabled. If dynamic vram isn't being used this option makes the text encoders run on the CPU.")
vram_group.add_argument("--novram", action="store_true", help="When lowvram isn't enough.")
vram_group.add_argument("--cpu", action="store_true", help="To use the CPU for everything (slow).")

parser.add_argument("--reserve-vram", type=float, default=None, help="Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS.")
parser.add_argument("--vram-headroom", type=float, default=0, help="Set the amount of vram in GB for DynamicVRAM to maintain as extra headroom above default. ComfyUI will try and keep this much VRAM completely free and unused, even counting VRAM from other apps.")

parser.add_argument("--async-offload", nargs='?', const=2, type=int, default=None, metavar="NUM_STREAMS", help="Use async weight offloading. An optional argument controls the amount of offload streams. Default is 2. Enabled by default on Nvidia.")
parser.add_argument("--disable-async-offload", action="store_true", help="Disable async weight offloading.")
parser.add_argument("--disable-dynamic-vram", action="store_true", help="Disable dynamic VRAM and use estimate based model loading.")
parser.add_argument("--enable-dynamic-vram", action="store_true", help="Enable dynamic VRAM on systems where it's not enabled by default.")
parser.add_argument("--fast-disk", action="store_true", help="Prefer disk-backed dynamic loading and offload over unpinned RAM. Can be faster for users with fast NVME disks.")

parser.add_argument("--force-non-blocking", action="store_true", help="Force ComfyUI to use non-blocking operations for all applicable tensors. This may improve performance on some non-Nvidia systems but can cause issues with some workflows.")

parser.add_argument("--default-hashing-function", type=str, choices=['md5', 'sha1', 'sha256', 'sha512'], default='sha256', help="Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.")

parser.add_argument("--disable-smart-memory", action="store_true", help="Force ComfyUI to agressively offload to regular ram instead of keeping models in vram when it can.")
parser.add_argument("--deterministic", action="store_true", help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.")

class PerformanceFeature(enum.Enum):
    Fp16Accumulation = "fp16_accumulation"
    Fp8MatrixMultiplication = "fp8_matrix_mult"
    CublasOps = "cublas_ops"
    AutoTune = "autotune"

parser.add_argument("--fast", nargs="*", type=PerformanceFeature, help="Enable some untested and potentially quality deteriorating optimizations. This is used to test new features so using it might crash your comfyui. --fast with no arguments enables everything. You can pass a list specific optimizations if you only want to enable specific ones. Current valid optimizations: {}".format(" ".join(map(lambda c: c.value, PerformanceFeature))))

parser.add_argument("--debug-hang", action="store_true", help="Enable stack trace dumps on Ctrl-C for debugging hangs.")

parser.add_argument("--disable-pinned-memory", action="store_true", help="Disable pinned memory use.")

parser.add_argument("--mmap-torch-files", action="store_true", help="Use mmap when loading ckpt/pt files.")
parser.add_argument("--disable-mmap", action="store_true", help="Don't use mmap when loading safetensors.")

parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")
parser.add_argument("--windows-standalone-build", action="store_true", help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup).")

parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")
parser.add_argument("--disable-all-custom-nodes", action="store_true", help="Disable loading all custom nodes.")
parser.add_argument("--whitelist-custom-nodes", type=str, nargs='+', default=[], help="Specify custom node folders to load even when --disable-all-custom-nodes is enabled.")
parser.add_argument("--disable-api-nodes", action="store_true", help="Disable loading all api nodes. Also prevents the frontend from communicating with the internet.")

parser.add_argument("--multi-user", action="store_true", help="Enables per-user storage.")

parser.add_argument("--verbose", default='INFO', const='DEBUG', nargs="?", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level')
parser.add_argument("--log-stdout", action="store_true", help="Send normal process output to stdout instead of stderr (default).")


# The default built-in provider hosted under web/
DEFAULT_VERSION_STRING = "comfyanonymous/ComfyUI@latest"

parser.add_argument(
    "--front-end-version",
    type=str,
    default=DEFAULT_VERSION_STRING,
    help="""
    Specifies the version of the frontend to be used. This command needs internet connectivity to query and
    download available frontend implementations from GitHub releases.

    The version string should be in the format of:
    [repoOwner]/[repoName]@[version]
    where version is one of: "latest" or a valid version number (e.g. "1.0.0")
    """,
)

def is_valid_directory(path: str) -> str:
    """Validate if the given path is a directory, and check permissions."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"The path '{path}' does not exist.")
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a directory.")
    if not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(f"You do not have read permissions for '{path}'.")
    return path

parser.add_argument(
    "--front-end-root",
    type=is_valid_directory,
    default=None,
    help="The local filesystem path to the directory where the frontend is located. Overrides --front-end-version.",
)

parser.add_argument("--user-directory", type=is_valid_directory, default=None, help="Set the ComfyUI user directory with an absolute path. Overrides --base-directory.")

parser.add_argument("--enable-compress-response-body", action="store_true", help="Enable compressing response body.")

parser.add_argument(
    "--comfy-api-base",
    type=str,
    default="https://api.comfy.org",
    help="Set the base URL for the ComfyUI API.  (default: https://api.comfy.org)",
)

database_default_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "user", "comfyui.db")
)
parser.add_argument("--database-url", type=str, default=f"sqlite:///{database_default_path}", help="Specify the database URL, e.g. for an in-memory database you can use 'sqlite:///:memory:'.")
parser.add_argument("--enable-assets", action="store_true", help="Enable the assets system (API routes, database synchronization, and background scanning).")
parser.add_argument("--enable-asset-hashing", action="store_true", help="Compute blake3 content hashes when scanning assets. Hashing enables future asset-portability features (deduplication, cross-machine model resolution) but adds startup cost and per-output cost on large models directories. Off by default; enable to opt in.")
parser.add_argument("--feature-flag", type=str, action='append', default=[], metavar="KEY[=VALUE]", help="Set a server feature flag. Use KEY=VALUE to set an explicit value, or bare KEY to set it to true. Can be specified multiple times. Boolean values (true/false) and numbers are auto-converted. Examples: --feature-flag show_signin_button=true  or  --feature-flag show_signin_button")
parser.add_argument("--list-feature-flags", action="store_true", help="Print the registry of known CLI-settable feature flags as JSON and exit.")

if comfy.options.args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

if args.cache_ram is not None and len(args.cache_ram) > 2:
    parser.error("--cache-ram accepts at most two values: active GB and inactive GB")

if args.high_ram:
    args.cache_classic = True

if args.windows_standalone_build:
    args.auto_launch = True

if args.disable_auto_launch:
    args.auto_launch = False

if args.force_fp16:
    args.fp16_unet = True

# '--enable-manager-legacy-ui' is meaningless unless the manager is enabled, so imply '--enable-manager'.
if args.enable_manager_legacy_ui:
    args.enable_manager = True


# '--fast' is not provided, use an empty set
if args.fast is None:
    args.fast = set()
# '--fast' is provided with an empty list, enable all optimizations
elif args.fast == []:
    args.fast = set(PerformanceFeature)
# '--fast' is provided with a list of performance features, use that list
else:
    args.fast = set(args.fast)

def enables_dynamic_vram():
    if args.enable_dynamic_vram:
        return True
    return not args.disable_dynamic_vram and not args.highvram and not args.gpu_only and not args.novram and not args.cpu
