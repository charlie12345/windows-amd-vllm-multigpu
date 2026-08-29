cmake_minimum_required(VERSION 4.0)
project(wavmg_hip VERSION 0.1 LANGUAGES CXX HIP)

if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES "gfx1201" CACHE STRING "HIP architectures")
endif()

add_library(wavmg_hip SHARED mapped_peer_reduce.hip)
target_compile_features(wavmg_hip PRIVATE cxx_std_17)
set_target_properties(
  wavmg_hip
  PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    PREFIX ""
    OUTPUT_NAME "wavmg_hip_v1"
)

if(WIN32)
  find_package(Vulkan REQUIRED)
  add_executable(vulkan_hip_external_memory_probe vulkan_hip_external_memory_probe.cpp)
  set_source_files_properties(vulkan_hip_external_memory_probe.cpp PROPERTIES LANGUAGE HIP)
  target_compile_features(vulkan_hip_external_memory_probe PRIVATE cxx_std_17)
  set_target_properties(
    vulkan_hip_external_memory_probe
    PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
  )
  target_link_libraries(vulkan_hip_external_memory_probe PRIVATE Vulkan::Vulkan)

  add_executable(d3d12_hip_cross_adapter_probe d3d12_hip_cross_adapter_probe.cpp)
  set_source_files_properties(d3d12_hip_cross_adapter_probe.cpp PROPERTIES LANGUAGE HIP)
  target_compile_features(d3d12_hip_cross_adapter_probe PRIVATE cxx_std_17)
  set_target_properties(
    d3d12_hip_cross_adapter_probe
    PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
  )
  target_link_libraries(d3d12_hip_cross_adapter_probe PRIVATE d3d12 dxgi)

  add_executable(hip_large_kernarg_probe hip_large_kernarg_probe.cpp)
  set_source_files_properties(hip_large_kernarg_probe.cpp PROPERTIES LANGUAGE HIP)
  target_compile_features(hip_large_kernarg_probe PRIVATE cxx_std_17)
  set_target_properties(
    hip_large_kernarg_probe
    PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
  )

  add_library(wavmg_d3d12 SHARED wavmg_d3d12.cpp)
  set_source_files_properties(wavmg_d3d12.cpp PROPERTIES LANGUAGE HIP)
  target_compile_features(wavmg_d3d12 PRIVATE cxx_std_17)
  set_target_properties(
    wavmg_d3d12
    PROPERTIES
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
      PREFIX ""
      OUTPUT_NAME "wavmg_d3d12_v1"
  )
  target_link_libraries(wavmg_d3d12 PRIVATE d3d12 dxgi)
endif()
