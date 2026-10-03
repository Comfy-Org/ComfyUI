@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo  ComfyUI Windows launcher
echo  Performing quick preflight checks...
echo.

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║                    Python Not Found                       ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo  ▓ ComfyUI needs Python to run, but we couldn't find it on your computer.
    echo.
    echo  ▓ What to do:
    echo    1. Download Python from: https://www.python.org/downloads/
    echo    2. During installation, make sure to check "Add Python to PATH"
    echo    3. Restart your computer after installing
    echo    4. Try running this script again
    echo.
    pause
    exit /b 1
)

REM Get Python environment information
python -c "import sys, os; venv = os.environ.get('VIRTUAL_ENV', ''); is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix); env_type = 'VENV_DETECTED' if (venv or is_venv) else 'SYSTEM_PYTHON'; print(env_type); print('PYTHON_PATH=' + sys.executable)" > env_info.tmp
for /f "tokens=1,* delims==" %%a in (env_info.tmp) do (
    if "%%a"=="VENV_DETECTED" set ENV_TYPE=VENV_DETECTED
    if "%%a"=="SYSTEM_PYTHON" set ENV_TYPE=SYSTEM_PYTHON
    if "%%a"=="PYTHON_PATH" set PYTHON_PATH=%%b
)
del env_info.tmp

REM ---------------------------------------------------------------
REM Weekly full check logic (informational checks only)
REM Force with: run_comfyui.bat --full-check
REM ---------------------------------------------------------------
set STATE_DIR=%LOCALAPPDATA%\ComfyUI\state
if not exist "%STATE_DIR%" mkdir "%STATE_DIR%" >nul 2>&1
set FULL_STAMP=%STATE_DIR%\last_full_check.stamp

set NEED_FULL=
for %%A in (%*) do (
    if /i "%%~A"=="--full-check" set NEED_FULL=1
)

if not defined NEED_FULL (
    if not exist "%FULL_STAMP%" (
        set NEED_FULL=1
    ) else (
        forfiles /P "%STATE_DIR%" /M "last_full_check.stamp" /D -7 >nul 2>&1
        if errorlevel 1 set NEED_FULL=
        if not errorlevel 1 set NEED_FULL=1
    )
)

REM Dependency presence check (informational only)
if not defined NEED_FULL goto :check_pytorch
python -c "import importlib.util as u; mods=['yaml','torch','torchvision','torchaudio','numpy','einops','transformers','tokenizers','sentencepiece','safetensors','aiohttp','yarl','PIL','scipy','tqdm','psutil','alembic','sqlalchemy','av']; missing=[m for m in mods if not u.find_spec(m)]; print('MISSING:' + (','.join(missing) if missing else 'NONE'))" > deps_check.tmp
for /f "tokens=1,* delims=:" %%a in (deps_check.tmp) do (
    if "%%a"=="MISSING" set MISSING_CRITICAL=%%b
)
del deps_check.tmp

if not "%MISSING_CRITICAL%"=="NONE" (
    echo.
    echo  Missing required Python packages:
    echo    %MISSING_CRITICAL%
    echo.
    if "%ENV_TYPE%"=="SYSTEM_PYTHON" (
        echo  Tip: Creating a virtual environment is recommended:
        echo    python -m venv venv ^&^& venv\Scripts\activate
    )
    echo.
    echo  Install the dependencies, then run this script again:
    echo    python -m pip install -r requirements.txt
    echo.
    exit /b 1
)
type nul > "%FULL_STAMP%"
goto :check_pytorch

:check_pytorch
REM Fast path: read torch version without importing (import is slow)
python -c "import sys; from importlib import util, metadata; s=util.find_spec('torch'); print('HAS_TORCH:' + ('1' if s else '0')); print('PYTORCH_VERSION:' + (metadata.version('torch') if s else 'NONE'))" > torch_meta.tmp 2>nul
set HAS_TORCH=
set PYTORCH_VERSION=NONE
for /f "tokens=1,* delims=:" %%a in (torch_meta.tmp) do (
    if "%%a"=="HAS_TORCH" set HAS_TORCH=%%b
    if "%%a"=="PYTORCH_VERSION" set PYTORCH_VERSION=%%b
)
del torch_meta.tmp 2>nul

REM Default CUDA vars
set CUDA_AVAILABLE=False
set CUDA_VERSION=NONE

REM Only import torch to check CUDA if present and not CPU build
if "%HAS_TORCH%"=="1" (
    echo %PYTORCH_VERSION% | findstr /C:"+cpu" >nul
    if errorlevel 1 (
        python -c "import torch; print('CUDA_AVAILABLE:' + str(torch.cuda.is_available())); print('CUDA_VERSION:' + (torch.version.cuda or 'NONE'))" > pytorch_check.tmp 2>nul
        if not errorlevel 1 (
            for /f "tokens=1,* delims=:" %%a in (pytorch_check.tmp) do (
                if "%%a"=="CUDA_AVAILABLE" set CUDA_AVAILABLE=%%b
                if "%%a"=="CUDA_VERSION" set CUDA_VERSION=%%b
            )
        )
        del pytorch_check.tmp 2>nul
    )
)

REM Check if PyTorch version contains "+cpu" indicating CPU-only build
echo %PYTORCH_VERSION% | findstr /C:"+cpu" >nul
if not errorlevel 1 (
    echo.
    echo  CPU-only PyTorch detected.
    echo  ComfyUI requires a CUDA-enabled PyTorch build for GPU acceleration.
    echo.
    echo  Install CUDA-enabled PyTorch, then run this script again. Example:
    echo    python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
    echo.
    exit /b 1
)

REM Check if CUDA is not available but PyTorch doesn't have "+cpu" (might be CUDA build but no GPU)
if "%CUDA_AVAILABLE%"=="False" (
    echo %PYTORCH_VERSION% | findstr /C:"+cpu" >nul
    if errorlevel 1 (
        echo.
        echo  ╔═══════════════════════════════════════════════════════════╗
        echo  ║                    GPU Not Detected                       ║
        echo  ╚═══════════════════════════════════════════════════════════╝
        echo.
        echo  ▓ PyTorch has GPU support installed, but we couldn't find your graphics card.
        echo.
        echo  ▓ This could mean:
        echo    - You don't have an NVIDIA graphics card
        echo    - Your graphics card drivers need to be updated
        echo    - Your graphics card isn't properly connected
        echo.
        echo  ▓ ComfyUI will run on your CPU instead, which will be slower.
        echo.
        set /p CONTINUE_CHOICE="Continue anyway? (Y/N): "
        if /i not "%CONTINUE_CHOICE%"=="Y" (
            echo.
            echo  ▓ Exiting. Check your graphics card setup and try again.
            pause
            exit /b 0
        )
)
)

REM Proceed to launch
goto :check_port

:check_port
if "%COMFY_PORT%"=="" set COMFY_PORT=8188
netstat -ano | findstr /r /c:":%COMFY_PORT% .*LISTENING" >nul
if errorlevel 1 (
    goto :port_ok
) else (
    for /l %%P in (8189,1,8199) do (
        netstat -ano | findstr /r /c:":%%P .*LISTENING" >nul
        if errorlevel 1 (
            set COMFY_PORT=%%P
            echo.
            echo  ▓ Port 8188 is busy. Rolling to free port %COMFY_PORT% in 5 seconds...
            timeout /t 5 /nobreak >nul
            goto :port_ok
        )
    )
    echo.
    echo  ▓ All fallback ports 8189-8199 appear busy. Please free a port and try again.
    echo.
    pause
    exit /b 1
)

:port_ok
goto :start_comfyui

:start_comfyui
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║                  Starting ComfyUI...                      ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.
set GUI_URL=http://127.0.0.1:%COMFY_PORT%
REM Spawn a background helper that opens the browser when the server is ready
start "" cmd /c "for /l %%i in (1,1,20) do (powershell -NoProfile -Command \"try{(Invoke-WebRequest -Uri '%GUI_URL%' -Method Head -TimeoutSec 1)>$null; exit 0}catch{exit 1}\" ^& if not errorlevel 1 goto open ^& timeout /t 1 ^>nul) ^& :open ^& start \"\" \"%GUI_URL%\""
python main.py --port %COMFY_PORT%
if errorlevel 1 (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║                    ComfyUI Crashed                        ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo  ▓ ComfyUI encountered an error and stopped. Here's what might help:
    echo.
    echo  ▓ Error: "Port already in use"
    echo    Solution: Close other ComfyUI instances or let this script auto-select a free port.
    echo.
    echo  ▓ Error: "Torch not compiled with CUDA enabled"
    echo    Solution: You need to install the GPU version of PyTorch (see instructions above)
    echo.
    echo  ▓ Error: "ModuleNotFoundError" or "No module named"
    echo    Solution: Run this script again to install missing packages
    echo.
    echo  ▓ Error: "CUDA out of memory" or "OOM"
    echo    Solution: Your graphics card doesn't have enough memory. Try using smaller models.
    echo.
    echo  ▓ For other errors, check the error message above for clues.
    echo    You can also visit: https://github.com/comfyanonymous/ComfyUI/issues
    echo.
    echo  ▓ The full error details are shown above.
    echo.
)
pause



