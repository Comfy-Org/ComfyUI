# Installing ComfyUI

This is the long form install reference: every GPU vendor, every accelerator, and the
post-install options that are easy to miss.

Most people do not need this page. The [desktop app](https://www.comfy.org/download) is
the easiest and best way to use ComfyUI, and [comfy-cli](https://www.comfy.org/cli)
(`pip install comfy-cli && comfy install`) covers most of the rest. See the
[README](../README.md#install) for the short version, and
[docs.comfy.org](https://docs.comfy.org/installation/manual_install) for the same
material with navigation and translations.

## Contents

- [Manual install (Windows, Linux)](#manual-install-windows-linux)
  - [NVIDIA](#nvidia)
  - [AMD GPUs (Linux)](#amd-gpus-linux)
  - [AMD GPUs (Windows, ROCm 10.0)](#amd-gpus-windows-rocm-100)
  - [Intel GPUs (Windows and Linux)](#intel-gpus-windows-and-linux)
  - [Apple Mac silicon](#apple-mac-silicon)
  - [Ascend NPUs](#ascend-npus)
  - [Cambricon MLUs](#cambricon-mlus)
  - [Iluvatar Corex](#iluvatar-corex)
  - [Dependencies](#dependencies)
- [Windows Portable](#windows-portable)
- [ComfyUI-Manager](#comfyui-manager)
- [Running](#running)
- [Sharing models with another UI](#sharing-models-with-another-ui)
- [High-quality previews](#high-quality-previews)
- [TLS/SSL](#tlsssl)

## Manual install (Windows, Linux)

Python 3.14 works but some custom nodes may have issues. The free threaded variant works
but some dependencies will enable the GIL so it's not fully supported.

Python 3.13 is very well supported. If you have trouble with some custom node
dependencies on 3.13 you can try 3.12.

torch 2.7 is minimally supported but using a newer version is extremely recommended.
Using a cu130 or above version of pytorch is required on Nvidia 20 series and above. Some
features and optimizations might only work on newer versions. We generally recommend
using the latest major version of pytorch with the latest cuda version unless it is less
than 2 weeks old. If your pytorch is more than 6 months old, please update it.

### Instructions

Git clone this repo.

Put your SD checkpoints (the huge ckpt/safetensors files) in: `models/checkpoints`

Put your VAE in: `models/vae`

Then install PyTorch for your hardware using the relevant section below, install the
[dependencies](#dependencies), and launch with `python main.py`.

### NVIDIA

Nvidia users should install stable pytorch using this command:

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
```

This is the command to install pytorch nightly instead which might have performance
improvements:

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132
```

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

```
pip uninstall torch
```

And install it again with the command above.

### AMD GPUs (Linux)

AMD users can install rocm and pytorch with pip if you don't have it already installed,
this is the command to install the stable version:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

This is the command to install the nightly with ROCm 7.2 which might have some
performance improvements:

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

### AMD GPUs (Windows, ROCm 10.0)

Use AMD's [multi-architecture PyTorch packages](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
The `device-*` extras install your GPU's kernels and the matching ROCm runtime
automatically; a separate HIP SDK installation is not needed.

Use Windows 11, a current [AMD graphics driver](https://www.amd.com/en/support/download/drivers.html),
and 64-bit Python 3.13.

The install command below uses `device-all` to install kernels for all supported GPUs. To
reduce download size and disk usage, optionally replace **both** occurrences of
`device-all` with the target for your GPU:

| GPU | Device extra |
| --- | --- |
| RX 9070 / XT, Radeon AI PRO R9700 | `device-gfx1201` |
| RX 9060 / XT | `device-gfx1200` |
| RX 7900 XT / XTX | `device-gfx1100` |
| RX 7700 XT / 7800 XT | `device-gfx1101` |
| RX 7600 / XT | `device-gfx1102` |
| Ryzen AI Max / Max+ (Strix Halo) | `device-gfx1151` |

**Note:** This table only lists examples. A GPU missing from it may still be supported:
supported architectures include RDNA 2, RDNA 3, RDNA 3.5, and RDNA 4. Keep `device-all`
to install kernels for all supported targets. For other models, see AMD's
[GPU target table](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#gfx-target-lookup-table)
and [ROCm compatibility matrix](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html).

**ROCm 10.0.0 with PyTorch 2.13:**

```bat
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-all]==2.13.0+rocm10.0.0" "torchvision[device-all]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
```

### Intel GPUs (Windows and Linux)

Intel Arc GPU users can install native PyTorch with torch.xpu support using pip. More
information can be found [here](https://pytorch.org/docs/main/notes/get_start_xpu.html).

To install PyTorch xpu, use the following command:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

This is the command to install the Pytorch xpu nightly which might have some performance
improvements:

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
```

See also [Intel GPU issues](https://docs.comfy.org/troubleshooting/overview#intel-gpu-issues)
in the troubleshooting docs.

### Apple Mac silicon

You can install ComfyUI in Apple Mac silicon (M1, M2, M3 or M4) with any recent macOS
version.

1. Install pytorch nightly. For instructions, read the
   [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
   Apple Developer guide (make sure to install the latest pytorch nightly).
2. Follow the [manual installation](#manual-install-windows-linux) instructions above.
3. Install the ComfyUI [dependencies](#dependencies).
4. Launch ComfyUI by running `python main.py`.

> **Note**: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy
> folders, as discussed in [manual installation](#manual-install-windows-linux).

### Ascend NPUs

For models compatible with Ascend Extension for PyTorch (torch_npu). To get started,
ensure your environment meets the prerequisites outlined on the
[installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) page.
Here's a step-by-step guide tailored to your platform and installation method:

1. Begin by installing the recommended or newer kernel version for Linux as specified in
   the Installation page of torch-npu, if necessary.
2. Proceed with the installation of Ascend Basekit, which includes the driver, firmware,
   and CANN, following the instructions provided for your specific platform.
3. Next, install the necessary packages for torch-npu by adhering to the platform-specific
   instructions on the [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) page.
4. Finally, adhere to the [manual installation](#manual-install-windows-linux) guide for
   Linux. Once all components are installed, you can run ComfyUI as described earlier.

### Cambricon MLUs

For models compatible with Cambricon Extension for PyTorch (torch_mlu). Here's a
step-by-step guide tailored to your platform and installation method:

1. Install the Cambricon CNToolkit by adhering to the platform-specific instructions on
   the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html) page.
2. Next, install the PyTorch(torch_mlu) following the instructions on the
   [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html) page.
3. Launch ComfyUI by running `python main.py`.

### Iluvatar Corex

For models compatible with Iluvatar Extension for PyTorch. Here's a step-by-step guide
tailored to your platform and installation method:

1. Install the Iluvatar Corex Toolkit by adhering to the platform-specific instructions on
   the [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536) page.
2. Launch ComfyUI by running `python main.py`.

### Dependencies

Install the dependencies by opening your terminal inside the ComfyUI folder and:

```
pip install -r requirements.txt
```

After this you should have everything installed and can proceed to running ComfyUI.

## Windows Portable

> **Note:** the portable build is not recommended for regular users, who should use the
> [desktop app](https://www.comfy.org/download) instead.

There is a portable standalone build for Windows that should work for running on Nvidia
GPUs or for running on your CPU only.

Simply download, extract with [7-Zip](https://7-zip.org) or with the windows explorer on
recent windows versions and run. For smaller models you normally only need to put the
checkpoints (the huge ckpt/safetensors files) in `ComfyUI\models\checkpoints`, but many
of the larger models have multiple files. Make sure to follow the instructions to know
which subfolder to put them in under `ComfyUI\models\`.

If you have trouble extracting it, right click the file -> properties -> unblock.

The portable currently comes with python 3.13 and pytorch cuda 13.0. Update your Nvidia
drivers if it doesn't start.

All official portable downloads:

| Build | Notes |
| --- | --- |
| [Nvidia GPUs](https://github.com/Comfy-Org/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z) | Supports 20 series and above |
| [Nvidia GPUs, pytorch cuda 12.6 + python 3.12](https://github.com/Comfy-Org/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia_cu126.7z) | Supports Nvidia 10 series and older GPUs. DO NOT USE THIS ON NEWER 20 SERIES AND ABOVE GPUS |
| [AMD GPUs](https://github.com/Comfy-Org/ComfyUI/releases/latest/download/ComfyUI_windows_portable_amd.7z) | |
| [Intel GPUs](https://github.com/Comfy-Org/ComfyUI/releases/latest/download/ComfyUI_windows_portable_intel.7z) | |

## ComfyUI-Manager

[ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager/tree/manager-v4) is an
extension that allows you to easily install, update, and manage custom nodes for ComfyUI.

1. Install the manager dependencies:

   ```bash
   pip install -r manager_requirements.txt
   ```

2. Enable the manager with the `--enable-manager` flag when running ComfyUI:

   ```bash
   python main.py --enable-manager
   ```

### Command line options

| Flag | Description |
|------|-------------|
| `--enable-manager` | Enable ComfyUI-Manager |
| `--enable-manager-legacy-ui` | Use the legacy manager UI instead of the new UI (implies `--enable-manager`) |
| `--disable-manager-ui` | Disable the manager UI and endpoints while keeping background features like security checks and scheduled installation completion (requires `--enable-manager`) |

See the [manager docs](https://docs.comfy.org/manager/overview) for full configuration.

## Running

```
python main.py
```

The full list of launch options is documented at
[docs.comfy.org/development/comfyui-server/startup-flags](https://docs.comfy.org/development/comfyui-server/startup-flags).

### For AMD cards not officially supported by ROCm

Try running it with this command if you have issues:

For 6700, 6600 and maybe other RDNA2 or older:

```
HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py
```

For AMD 7600 and maybe other RDNA3 cards:

```
HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py
```

### AMD ROCm tips

You can try setting this env variable `PYTORCH_TUNABLEOP_ENABLED=1` which might speed
things up at the cost of a very slow initial run.

## Sharing models with another UI

See the [config file](../extra_model_paths.yaml.example) to set the search paths for
models. In the standalone windows build you can find this file in the ComfyUI directory.
Rename this file to `extra_model_paths.yaml` and edit it with your favorite text editor.

## High-quality previews

Use `--preview-method auto` to enable previews.

The default installation includes a fast latent preview method that's low-resolution. To
enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd),
download the [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and
taef1_decoder.pth](https://github.com/madebyollin/taesd/) and place them in the
`models/vae_approx` folder. Once they're installed, restart ComfyUI and launch it with
`--preview-method taesd` to enable high-quality previews.

## TLS/SSL

Generate a self-signed certificate (not appropriate for shared/production use) and key by
running the command:

```
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"
```

Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL, the app will now be
accessible with `https://...` instead of `http://...`.

> Note: Windows users can use [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl)
> or one of the [3rd party binary distributions](https://wiki.openssl.org/index.php/Binaries)
> to run the command example above.
>
> If you use a container, note that the volume mount `-v` can be a relative path so
> `... -v ".\:/openssl-certs" ...` would create the key & cert files in the current
> directory of your command prompt or powershell terminal.
