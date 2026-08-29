@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "VENV=%PROJECT_ROOT%\.venv"
set "CMAKE=%VENV%\Scripts\cmake.exe"

if not exist "%CMAKE%" (
  echo [FAIL] Run scripts\bootstrap-nightly.ps1 first.
  exit /b 1
)

set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;C:\Program Files\Git\cmd;C:\Program Files\Git\usr\bin;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo [FAIL] Could not initialize the Visual Studio x64 build environment.
  exit /b 1
)

if not defined ROCM_ROOT for /f "delims=" %%I in ('"%VENV%\Scripts\python.exe" -m rocm_sdk path --root') do set "ROCM_ROOT=%%I"
if not defined ROCM_ROOT (
  echo [FAIL] Could not locate ROCm. Set ROCM_ROOT or install rocm[devel].
  exit /b 1
)
if not defined WAVMG_GPU_ARCH set "WAVMG_GPU_ARCH=gfx1201"
set "ROCM_ROOT_FWD=%ROCM_ROOT:\=/%"
set "ROCM_PATH=%ROCM_ROOT_FWD%"
set "HIP_PATH=%ROCM_ROOT_FWD%"
set "PATH=%ROCM_ROOT%\bin;%ROCM_ROOT%\lib\llvm\bin;%VENV%\Scripts;%PATH%"
if not defined VULKAN_SDK set "VULKAN_SDK=C:\VulkanSDK\1.4.350.0"

"%CMAKE%" ^
  -S "%PROJECT_ROOT%\native" ^
  -B "%PROJECT_ROOT%\build\native" ^
  -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER="%ROCM_ROOT_FWD%/lib/llvm/bin/clang++.exe" ^
  -DCMAKE_HIP_COMPILER="%ROCM_ROOT_FWD%/lib/llvm/bin/clang++.exe" ^
  -DCMAKE_HIP_PLATFORM=amd ^
  -DCMAKE_HIP_ARCHITECTURES=%WAVMG_GPU_ARCH% ^
  -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT= ^
  -DCMAKE_SHARED_LINKER_FLAGS= ^
  -DCMAKE_SHARED_LINKER_FLAGS_RELEASE= ^
  -DCMAKE_EXE_LINKER_FLAGS= ^
  -DCMAKE_EXE_LINKER_FLAGS_RELEASE= ^
  -DCMAKE_HIP_FLAGS="--rocm-device-lib-path=%ROCM_ROOT_FWD%/lib/llvm/amdgcn/bitcode" ^
  -DROCM_PATH="%ROCM_ROOT_FWD%"
if errorlevel 1 exit /b 1

"%CMAKE%" --build "%PROJECT_ROOT%\build\native" --config Release
exit /b %errorlevel%
