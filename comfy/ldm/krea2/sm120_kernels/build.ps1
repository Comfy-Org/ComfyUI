[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$OutputDirectory,
    [Parameter(Mandatory)] [string]$CublasLt,
    [string]$CudaRoot = $env:CUDA_PATH
)

$ErrorActionPreference = 'Stop'
if (-not $CudaRoot) { throw 'Specify -CudaRoot for CUDA Toolkit 13.0.' }
$cuda = (Resolve-Path -LiteralPath $CudaRoot).Path
$nvcc = Join-Path $cuda 'bin/nvcc.exe'
$include = Join-Path $cuda 'include'
$verified = (Resolve-Path -LiteralPath $CublasLt).Path
$expected = 'CAD63434448E7141629E240EA093AD596A7EF6A0F67B468BA9BC1DF6E1EEEE33'
if ((Get-FileHash -LiteralPath $verified -Algorithm SHA256).Hash -ne $expected) {
    throw 'Expected the NVIDIA cuBLASLt 13.8.0.4 x64 DLL.'
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue) -or -not $env:INCLUDE -or -not $env:LIB) {
    $x86 = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
    $vswhere = Join-Path $x86 'Microsoft Visual Studio/Installer/vswhere.exe'
    $installation = & $vswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $installation) { throw 'Visual Studio 2022 C++ x64 build tools are required.' }
    $vcvars = Join-Path $installation 'VC/Auxiliary/Build/vcvars64.bat'
    $buildEnvironment = & $env:ComSpec /d /c ('call "{0}" >nul && set' -f $vcvars)
    if ($LASTEXITCODE -ne 0) { throw 'Visual Studio environment setup failed.' }
    foreach ($line in $buildEnvironment) {
        if ($line -match '^(PATH|INCLUDE|LIB|LIBPATH)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
Push-Location -LiteralPath $OutputDirectory
try {
    foreach ($name in 'mx', 'half') {
        & cl.exe /nologo /std:c++17 /EHsc /O2 /LD "/I$include" (Join-Path $PSScriptRoot "$name.cpp") /link "/OUT:$name.dll"
        if ($LASTEXITCODE -ne 0) { throw "Building $name failed." }
    }
    & $nvcc --shared -std=c++17 -O3 -arch=sm_120 -o rms.dll (Join-Path $PSScriptRoot 'rms.cu')
    if ($LASTEXITCODE -ne 0) { throw 'Building RMSNorm failed.' }
    & $nvcc --shared -std=c++17 -O3 -arch=sm_120 -o mxlt.dll (Join-Path $PSScriptRoot 'mxlt.cu') -lcublasLt
    if ($LASTEXITCODE -ne 0) { throw 'Building the bundled cuBLASLt adapter failed.' }
    $destination = Join-Path (Get-Location) 'cublasLt64_13_8.dll'
    if ($verified -ne $destination) { Copy-Item -LiteralPath $verified -Destination $destination }
} finally {
    Pop-Location
}
