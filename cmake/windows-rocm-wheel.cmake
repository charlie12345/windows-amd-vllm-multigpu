# Windows toolchain for AMD's ROCm SDK Python wheels.
# ROCM_PATH must be passed by the configure script before project().

set(CMAKE_SYSTEM_NAME Windows)

if(NOT ROCM_PATH AND DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm wheel root")
endif()
if(NOT ROCM_PATH)
  message(FATAL_ERROR "ROCM_PATH must point to the _rocm_sdk_devel wheel root")
endif()

file(TO_CMAKE_PATH "${ROCM_PATH}" _wavmg_rocm_path)
set(
  CMAKE_CXX_COMPILER
  "${_wavmg_rocm_path}/lib/llvm/bin/amdclang++.exe"
  CACHE FILEPATH "AMD clang++ from the ROCm wheel" FORCE
)
set(
  CMAKE_C_COMPILER
  "${_wavmg_rocm_path}/lib/llvm/bin/amdclang.exe"
  CACHE FILEPATH "AMD clang from the ROCm wheel" FORCE
)
set(
  CMAKE_HIP_COMPILER
  "${_wavmg_rocm_path}/lib/llvm/bin/clang++.exe"
  CACHE FILEPATH "HIP compiler from the ROCm wheel" FORCE
)
set(CMAKE_HIP_PLATFORM amd CACHE STRING "HIP platform" FORCE)
set(HIP_PLATFORM amd CACHE STRING "HIP platform" FORCE)
set(
  CMAKE_CXX_FLAGS_INIT
  "--rocm-device-lib-path=${_wavmg_rocm_path}/lib/llvm/amdgcn/bitcode"
)
set(
  CMAKE_HIP_FLAGS_INIT
  "--rocm-device-lib-path=${_wavmg_rocm_path}/lib/llvm/amdgcn/bitcode"
)
list(PREPEND CMAKE_PREFIX_PATH "${_wavmg_rocm_path}")
unset(_wavmg_rocm_path)
