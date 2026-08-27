cmake_minimum_required(VERSION 3.24)

project(wavmg_llama_rccl_shim VERSION 0.2.0 LANGUAGES CXX HIP)

include(CMakePackageConfigHelpers)

set(RCCL_ROOT "" CACHE PATH "Optional staged Windows RCCL package root")
set(WAC_RCCL_DLL "" CACHE FILEPATH "Path to the real Windows rccl.dll")
set(WAC_RCCL_INCLUDE_DIR "" CACHE PATH "Directory containing nccl.h")
set(WAC_HIP_ARCHITECTURES "gfx1201" CACHE STRING "HIP architectures for the D3D12 reduction kernels")

if(NOT WIN32)
    message(FATAL_ERROR "The llama RCCL shim currently supports Windows only")
endif()

if(RCCL_ROOT)
    if(NOT WAC_RCCL_DLL)
        set(WAC_RCCL_DLL "${RCCL_ROOT}/bin/rccl.dll")
    endif()
    if(NOT WAC_RCCL_INCLUDE_DIR)
        set(WAC_RCCL_INCLUDE_DIR "${RCCL_ROOT}/include")
    endif()
endif()

if(NOT EXISTS "${WAC_RCCL_INCLUDE_DIR}/nccl.h")
    message(FATAL_ERROR "WAC_RCCL_INCLUDE_DIR must contain nccl.h")
endif()
if(NOT EXISTS "${WAC_RCCL_DLL}")
    message(FATAL_ERROR "WAC_RCCL_DLL must name the real Windows rccl.dll")
endif()

find_package(hip REQUIRED)

add_library(rccl SHARED
    src/d3d12_transport.cu
    src/d3d12_transport.hpp
    src/rccl_shim.cpp
    src/rccl_shim.def
)
set_source_files_properties(src/d3d12_transport.cu PROPERTIES LANGUAGE HIP)

target_compile_features(rccl PRIVATE cxx_std_17)
target_include_directories(rccl
    PRIVATE
        "${WAC_RCCL_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    INTERFACE
        "$<BUILD_INTERFACE:${WAC_RCCL_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(rccl PUBLIC hip::host)
target_link_libraries(rccl PRIVATE d3d12 dxgi)
target_compile_definitions(rccl PRIVATE _CRT_SECURE_NO_WARNINGS)

set_target_properties(rccl PROPERTIES
    HIP_ARCHITECTURES "${WAC_HIP_ARCHITECTURES}"
    OUTPUT_NAME "rccl"
    PREFIX ""
    VERSION "${PROJECT_VERSION}"
    SOVERSION "0"
)

if(MSVC)
    target_compile_options(rccl PRIVATE /W4)
else()
    target_compile_options(rccl PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_command(TARGET rccl POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${WAC_RCCL_DLL}"
        "$<TARGET_FILE_DIR:rccl>/rccl-real.dll"
    VERBATIM
)

install(TARGETS rccl
    EXPORT rccl-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES "${WAC_RCCL_DLL}" DESTINATION bin RENAME rccl-real.dll)
install(DIRECTORY "${WAC_RCCL_INCLUDE_DIR}/" DESTINATION include)
install(FILES README.md DESTINATION share/doc/wavmg-llama-rccl-shim)
install(PROGRAMS run-with-llama-plugin.ps1 DESTINATION .)

get_filename_component(WAVMG_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
install(FILES
    "${WAVMG_REPOSITORY_ROOT}/docs/install-llama-rocmfpx.md"
    "${WAVMG_REPOSITORY_ROOT}/docs/architecture.md"
    "${WAVMG_REPOSITORY_ROOT}/docs/licensing.md"
    DESTINATION docs
)
install(FILES
    "${WAVMG_REPOSITORY_ROOT}/LICENSE"
    "${WAVMG_REPOSITORY_ROOT}/NOTICE"
    DESTINATION licenses
)
install(DIRECTORY "${WAVMG_REPOSITORY_ROOT}/LICENSES/" DESTINATION licenses/LICENSES)

configure_package_config_file(
    cmake/rccl-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/rccl-config.cmake"
    INSTALL_DESTINATION lib/cmake/rccl
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rccl-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/rccl-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rccl-config-version.cmake"
    DESTINATION lib/cmake/rccl
)
install(EXPORT rccl-targets
    FILE rccl-targets.cmake
    NAMESPACE roc::
    DESTINATION lib/cmake/rccl
)
