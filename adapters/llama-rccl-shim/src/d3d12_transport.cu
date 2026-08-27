// SPDX-FileCopyrightText: 2026 Carlo Pasquale (https://github.com/Charlie12345)
// SPDX-License-Identifier: Apache-2.0

#include "d3d12_transport.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

using Microsoft::WRL::ComPtr;

namespace {

void check_hresult(HRESULT result, const char * operation) {
    if (FAILED(result)) {
        std::ostringstream message;
        message << operation << " failed: HRESULT 0x" << std::hex
                << static_cast<unsigned long>(result);
        throw std::runtime_error(message.str());
    }
}

void check_hip(hipError_t result, const char * operation) {
    if (result != hipSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: HIP " + std::to_string(result) +
                                 " (" + hipGetErrorString(result) + ")");
    }
}

std::uint64_t align_heap_size(std::size_t size) {
    constexpr std::uint64_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    return (static_cast<std::uint64_t>(size) + alignment - 1) & ~(alignment - 1);
}

bool luid_equal(const LUID & lhs, const char rhs[8]) {
    std::array<std::uint8_t, 8> bytes{};
    std::memcpy(bytes.data(), &lhs, bytes.size());
    return std::memcmp(bytes.data(), rhs, bytes.size()) == 0;
}

std::wstring object_name(const std::wstring & base, const wchar_t * suffix) {
    return base + suffix;
}

struct device_binding {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter4> adapter;
    ComPtr<ID3D12Device> device;
};

device_binding bind_device(int hip_device) {
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, hip_device), "hipGetDeviceProperties");

    device_binding binding;
    check_hresult(CreateDXGIFactory2(0, IID_PPV_ARGS(&binding.factory)), "CreateDXGIFactory2");
    for (std::uint32_t index = 0;; ++index) {
        ComPtr<IDXGIAdapter4> candidate;
        const HRESULT result = binding.factory->EnumAdapterByGpuPreference(
                index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
        if (result == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        check_hresult(result, "EnumAdapterByGpuPreference");

        DXGI_ADAPTER_DESC3 description{};
        check_hresult(candidate->GetDesc3(&description), "IDXGIAdapter4::GetDesc3");
        if ((description.Flags & DXGI_ADAPTER_FLAG3_SOFTWARE) == 0 &&
            description.VendorId == 0x1002 &&
            luid_equal(description.AdapterLuid, properties.luid)) {
            binding.adapter = std::move(candidate);
            check_hresult(D3D12CreateDevice(binding.adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                            IID_PPV_ARGS(&binding.device)),
                          "D3D12CreateDevice");
            return binding;
        }
    }

    throw std::runtime_error("No D3D12 adapter LUID matched HIP device " +
                             std::to_string(hip_device));
}

D3D12_RESOURCE_DESC buffer_description(std::size_t size) {
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = static_cast<std::uint64_t>(size);
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = static_cast<D3D12_RESOURCE_FLAGS>(
            D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    return description;
}

ComPtr<ID3D12Resource> create_placed_buffer(ID3D12Device * device, ID3D12Heap * heap,
                                            std::size_t size) {
    const D3D12_RESOURCE_DESC description = buffer_description(size);
    ComPtr<ID3D12Resource> resource;
    check_hresult(device->CreatePlacedResource(heap, 0, &description, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, IID_PPV_ARGS(&resource)),
                  "ID3D12Device::CreatePlacedResource");
    return resource;
}

struct shared_buffer {
    int hip_device = -1;
    std::size_t size = 0;
    std::uint64_t heap_size = 0;
    device_binding binding;
    ComPtr<ID3D12Heap> heap;
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Fence> fence;
    HANDLE heap_handle = nullptr;
    HANDLE fence_handle = nullptr;
    hipExternalMemory_t external_memory = nullptr;
    hipExternalSemaphore_t external_semaphore = nullptr;
    void * device_pointer = nullptr;

    ~shared_buffer() {
        if (hip_device >= 0) {
            (void)hipSetDevice(hip_device);
        }
        if (external_semaphore != nullptr) {
            (void)hipDestroyExternalSemaphore(external_semaphore);
        }
        if (external_memory != nullptr) {
            (void)hipDestroyExternalMemory(external_memory);
        }
        if (fence_handle != nullptr) {
            CloseHandle(fence_handle);
        }
        if (heap_handle != nullptr) {
            CloseHandle(heap_handle);
        }
    }

    void import_hip_objects() {
        check_hip(hipSetDevice(hip_device), "hipSetDevice");

        hipExternalMemoryHandleDesc memory_desc{};
        memory_desc.type = hipExternalMemoryHandleTypeD3D12Heap;
        memory_desc.handle.win32.handle = heap_handle;
        memory_desc.size = heap_size;
        check_hip(hipImportExternalMemory(&external_memory, &memory_desc),
                  "hipImportExternalMemory(D3D12Heap)");

        hipExternalMemoryBufferDesc buffer_desc{};
        buffer_desc.offset = 0;
        buffer_desc.size = size;
        check_hip(hipExternalMemoryGetMappedBuffer(&device_pointer, external_memory, &buffer_desc),
                  "hipExternalMemoryGetMappedBuffer");

        hipExternalSemaphoreHandleDesc semaphore_desc{};
        semaphore_desc.type = hipExternalSemaphoreHandleTypeD3D12Fence;
        semaphore_desc.handle.win32.handle = fence_handle;
        check_hip(hipImportExternalSemaphore(&external_semaphore, &semaphore_desc),
                  "hipImportExternalSemaphore(D3D12Fence)");
    }

    void initialize_creator(int device_id, const std::wstring & base_name, std::size_t size_bytes) {
        hip_device = device_id;
        size = size_bytes;
        heap_size = align_heap_size(size_bytes);
        binding = bind_device(device_id);

        D3D12_HEAP_DESC heap_desc{};
        heap_desc.SizeInBytes = heap_size;
        heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap_desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_desc.Properties.CreationNodeMask = 1;
        heap_desc.Properties.VisibleNodeMask = 1;
        heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        heap_desc.Flags = static_cast<D3D12_HEAP_FLAGS>(D3D12_HEAP_FLAG_SHARED |
                                                        D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER);
        check_hresult(binding.device->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap)),
                      "ID3D12Device::CreateHeap(cross-adapter)");
        resource = create_placed_buffer(binding.device.Get(), heap.Get(), size);

        const std::wstring memory_name = object_name(base_name, L".memory");
        check_hresult(binding.device->CreateSharedHandle(heap.Get(), nullptr, GENERIC_ALL,
                                                         memory_name.c_str(), &heap_handle),
                      "ID3D12Device::CreateSharedHandle(memory)");

        const D3D12_FENCE_FLAGS fence_flags = static_cast<D3D12_FENCE_FLAGS>(
                D3D12_FENCE_FLAG_SHARED | D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER);
        check_hresult(binding.device->CreateFence(0, fence_flags, IID_PPV_ARGS(&fence)),
                      "ID3D12Device::CreateFence(cross-adapter)");
        const std::wstring fence_name = object_name(base_name, L".fence");
        check_hresult(binding.device->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL,
                                                         fence_name.c_str(), &fence_handle),
                      "ID3D12Device::CreateSharedHandle(fence)");
        import_hip_objects();
    }

    void initialize_opener(int device_id, const std::wstring & base_name, std::size_t size_bytes) {
        hip_device = device_id;
        size = size_bytes;
        heap_size = align_heap_size(size_bytes);
        binding = bind_device(device_id);

        HANDLE source_heap_handle = nullptr;
        HANDLE source_fence_handle = nullptr;
        try {
            const std::wstring memory_name = object_name(base_name, L".memory");
            check_hresult(binding.device->OpenSharedHandleByName(memory_name.c_str(), GENERIC_ALL,
                                                                 &source_heap_handle),
                          "ID3D12Device::OpenSharedHandleByName(memory)");
            check_hresult(binding.device->OpenSharedHandle(source_heap_handle, IID_PPV_ARGS(&heap)),
                          "ID3D12Device::OpenSharedHandle(memory)");
            resource = create_placed_buffer(binding.device.Get(), heap.Get(), size);
            check_hresult(binding.device->CreateSharedHandle(heap.Get(), nullptr, GENERIC_ALL,
                                                             nullptr, &heap_handle),
                          "ID3D12Device::CreateSharedHandle(peer memory)");

            const std::wstring fence_name = object_name(base_name, L".fence");
            check_hresult(binding.device->OpenSharedHandleByName(fence_name.c_str(), GENERIC_ALL,
                                                                 &source_fence_handle),
                          "ID3D12Device::OpenSharedHandleByName(fence)");
            check_hresult(
                    binding.device->OpenSharedHandle(source_fence_handle, IID_PPV_ARGS(&fence)),
                    "ID3D12Device::OpenSharedHandle(fence)");
            check_hresult(binding.device->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL,
                                                             nullptr, &fence_handle),
                          "ID3D12Device::CreateSharedHandle(peer fence)");
            import_hip_objects();
        } catch (...) {
            if (source_heap_handle != nullptr) {
                CloseHandle(source_heap_handle);
            }
            if (source_fence_handle != nullptr) {
                CloseHandle(source_fence_handle);
            }
            throw;
        }
        CloseHandle(source_heap_handle);
        CloseHandle(source_fence_handle);
    }

    void signal(std::uint64_t value, hipStream_t stream) {
        check_hip(hipSetDevice(hip_device), "hipSetDevice");
        hipExternalSemaphoreSignalParams parameters{};
        parameters.params.fence.value = value;
        check_hip(hipSignalExternalSemaphoresAsync(&external_semaphore, &parameters, 1, stream),
                  "hipSignalExternalSemaphoresAsync");
    }

    void wait(std::uint64_t value, hipStream_t stream) {
        check_hip(hipSetDevice(hip_device), "hipSetDevice");
        hipExternalSemaphoreWaitParams parameters{};
        parameters.params.fence.value = value;
        check_hip(hipWaitExternalSemaphoresAsync(&external_semaphore, &parameters, 1, stream),
                  "hipWaitExternalSemaphoresAsync");
    }
};

template <typename T> __device__ __forceinline__ T add_values(T lhs, T rhs) { return lhs + rhs; }

template <> __device__ __forceinline__ __half add_values(__half lhs, __half rhs) {
    return __float2half(__half2float(lhs) + __half2float(rhs));
}

template <> __device__ __forceinline__ hip_bfloat16 add_values(hip_bfloat16 lhs, hip_bfloat16 rhs) {
    return hip_bfloat16(static_cast<float>(lhs) + static_cast<float>(rhs));
}

template <typename T>
__global__ void add_kernel(const T * local, const T * peer, T * output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = add_values(local[index], peer[index]);
    }
}

template <typename T>
void enqueue_rank(shared_buffer & own, shared_buffer & peer, const void * send, void * recv,
                  void * peer_copy, std::size_t count, std::uint64_t ready_value,
                  std::uint64_t consumed_value, hipStream_t stream) {
    const std::size_t nbytes = count * sizeof(T);
    check_hip(hipMemcpyAsync(own.device_pointer, send, nbytes, hipMemcpyDefault, stream),
              "hipMemcpyAsync(input to D3D12)");
    own.signal(ready_value, stream);
    peer.wait(ready_value, stream);
    check_hip(hipMemcpyAsync(peer_copy, peer.device_pointer, nbytes, hipMemcpyDefault, stream),
              "hipMemcpyAsync(D3D12 peer to scratch)");

    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    hipLaunchKernelGGL(add_kernel<T>, dim3(blocks), dim3(threads), 0, stream,
                       static_cast<const T *>(send), static_cast<const T *>(peer_copy),
                       static_cast<T *>(recv), count);
    check_hip(hipGetLastError(), "D3D12 AllReduce add kernel");
    peer.signal(consumed_value, stream);
    own.wait(consumed_value, stream);
}

std::atomic<std::uint64_t> g_context_counter{0};

} // namespace

class wac_d3d12_transport {
  public:
    std::array<int, 2> devices{};
    std::size_t max_size_bytes = 0;
    std::uint64_t epoch = 0;
    std::array<std::unique_ptr<shared_buffer>, 2> own;
    std::array<std::unique_ptr<shared_buffer>, 2> peer;
    std::array<void *, 2> scratch{};

    ~wac_d3d12_transport() {
        for (std::size_t rank = 0; rank < devices.size(); ++rank) {
            (void)hipSetDevice(devices[rank]);
            (void)hipDeviceSynchronize();
            if (scratch[rank] != nullptr) {
                (void)hipFree(scratch[rank]);
            }
        }
    }
};

void wac_d3d12_deleter::operator()(wac_d3d12_transport * transport) const { delete transport; }

wac_d3d12_transport_ptr wac_d3d12_create(const int devices[2], std::size_t max_size_bytes,
                                         std::string & error) {
    if (devices == nullptr || max_size_bytes == 0) {
        error = "D3D12 AllReduce requires two devices and a nonzero buffer size";
        return nullptr;
    }

    try {
        wac_d3d12_transport_ptr transport(new wac_d3d12_transport());
        transport->devices = {devices[0], devices[1]};
        transport->max_size_bytes = max_size_bytes;

        const std::uint64_t counter = g_context_counter.fetch_add(1, std::memory_order_relaxed);
        const std::wstring base = L"Local\\wavmg-llama-rccl-shim-" +
                                  std::to_wstring(GetCurrentProcessId()) + L"-" +
                                  std::to_wstring(counter);
        const std::array<std::wstring, 2> names = {base + L".rank0", base + L".rank1"};

        for (std::size_t rank = 0; rank < 2; ++rank) {
            transport->own[rank] = std::make_unique<shared_buffer>();
            transport->own[rank]->initialize_creator(devices[rank], names[rank], max_size_bytes);
        }
        for (std::size_t rank = 0; rank < 2; ++rank) {
            transport->peer[rank] = std::make_unique<shared_buffer>();
            transport->peer[rank]->initialize_opener(devices[rank], names[1 - rank],
                                                     max_size_bytes);
            check_hip(hipSetDevice(devices[rank]), "hipSetDevice");
            check_hip(hipMalloc(&transport->scratch[rank], max_size_bytes),
                      "hipMalloc(D3D12 scratch)");
        }

        error.clear();
        return transport;
    } catch (const std::exception & exception) {
        error = exception.what();
        return nullptr;
    }
}

bool wac_d3d12_allreduce(wac_d3d12_transport & transport, const wac_rank_call calls[2],
                         wac_data_type type, std::string & error) {
    if (calls == nullptr || calls[0].send == nullptr || calls[0].recv == nullptr ||
        calls[1].send == nullptr || calls[1].recv == nullptr || calls[0].count != calls[1].count) {
        error = "D3D12 AllReduce received mismatched calls";
        return false;
    }

    const std::size_t element_size = type == wac_data_type::f32 ? 4 : 2;
    if (calls[0].count > transport.max_size_bytes / element_size) {
        error = "D3D12 AllReduce payload exceeds the configured buffer";
        return false;
    }
    if (calls[0].count == 0) {
        error.clear();
        return true;
    }

    try {
        const std::uint64_t ready_value = (++transport.epoch * 2) - 1;
        const std::uint64_t consumed_value = ready_value + 1;
        for (std::size_t rank = 0; rank < 2; ++rank) {
            if (calls[rank].device != transport.devices[rank]) {
                throw std::runtime_error("D3D12 AllReduce rank/device order changed");
            }
            check_hip(hipSetDevice(calls[rank].device), "hipSetDevice");
            switch (type) {
            case wac_data_type::f32:
                enqueue_rank<float>(*transport.own[rank], *transport.peer[rank], calls[rank].send,
                                    calls[rank].recv, transport.scratch[rank], calls[rank].count,
                                    ready_value, consumed_value, calls[rank].stream);
                break;
            case wac_data_type::f16:
                enqueue_rank<__half>(*transport.own[rank], *transport.peer[rank], calls[rank].send,
                                     calls[rank].recv, transport.scratch[rank], calls[rank].count,
                                     ready_value, consumed_value, calls[rank].stream);
                break;
            case wac_data_type::bf16:
                enqueue_rank<hip_bfloat16>(*transport.own[rank], *transport.peer[rank],
                                           calls[rank].send, calls[rank].recv,
                                           transport.scratch[rank], calls[rank].count, ready_value,
                                           consumed_value, calls[rank].stream);
                break;
            }
        }
        error.clear();
        return true;
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }
}
