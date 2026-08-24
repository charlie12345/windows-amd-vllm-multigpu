cmake_minimum_required(VERSION 4.0)
project(wavmg_hip VERSION 0.1 LANGUAGES CXX HIP)

add_library(wavmg_hip SHARED mapped_peer_reduce.hip)
target_compile_features(wavmg_hip PRIVATE cxx_std_17)
set_target_properties(
  wavmg_hip
  PROPERTIES
    HIP_ARCHITECTURES gfx1201
    PREFIX ""
    OUTPUT_NAME "wavmg_hip_v1"
)
