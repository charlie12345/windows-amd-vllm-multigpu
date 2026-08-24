#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define WAVMG_D3D12_EXPORT extern "C" __declspec(dllexport)
#else
#define WAVMG_D3D12_EXPORT extern "C"
#endif

struct wavmg_d3d12_context;

WAVMG_D3D12_EXPORT int wavmg_d3d12_create(
    int hip_device,
    const wchar_t* base_name,
    std::size_t size,
    wavmg_d3d12_context** context,
    void** device_pointer);

WAVMG_D3D12_EXPORT int wavmg_d3d12_open(
    int hip_device,
    const wchar_t* base_name,
    std::size_t size,
    wavmg_d3d12_context** context,
    void** device_pointer);

WAVMG_D3D12_EXPORT int wavmg_d3d12_signal(
    wavmg_d3d12_context* context,
    std::uint64_t value,
    void* stream);

WAVMG_D3D12_EXPORT int wavmg_d3d12_wait(
    wavmg_d3d12_context* context,
    std::uint64_t value,
    void* stream);

WAVMG_D3D12_EXPORT int wavmg_d3d12_close(wavmg_d3d12_context* context);

WAVMG_D3D12_EXPORT const char* wavmg_d3d12_last_error();
