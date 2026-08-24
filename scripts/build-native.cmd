@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "VENV=%PROJECT_ROOT%\.venv"
set "CMAKE=%VENV%\Scripts\cmake.exe"

if not exist "%CMAKE%" (
  echo [FAIL] Run scripts\bootstrap-nightly.ps1 first.
  exit /b 1
)

set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo [FAIL] Could not initialize the Visual Studio x64 build environment.
  exit /b 1
)

for /f "delims=" %%I in ('"%VENV%\Scripts\python.exe" -m rocm_sdk path --root') do set "ROCM_ROOT=%%I"
if not defined ROCM_ROOT (
  echo [FAIL] Could not locate the ROCm SDK wheel.
  exit /b 1
)
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
  -DCMAKE_CXX_COMPILER="%ROCM_ROOT_FWD%/lib/llvm/bin/clang-cl.exe" ^
  -DCMAKE_HIP_COMPILER="%ROCM_ROOT_FWD%/lib/llvm/bin/clang-cl.exe" ^
  -DCMAKE_HIP_PLATFORM=amd ^
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 ^
  -DCMAKE_HIP_FLAGS="--rocm-device-lib-path=%ROCM_ROOT_FWD%/lib/llvm/amdgcn/bitcode" ^
  -DROCM_PATH="%ROCM_ROOT_FWD%"
if errorlevel 1 exit /b 1

"%CMAKE%" --build "%PROJECT_ROOT%\build\native" --config Release
exit /b %errorlevel%
