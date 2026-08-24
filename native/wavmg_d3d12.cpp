#include "wavmg_d3d12.h"

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <hip/hip_runtime.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

using Microsoft::WRL::ComPtr;

namespace {

thread_local std::string g_last_error;

void check_hr(HRESULT result, const char* operation) {
  if (FAILED(result)) {
    std::ostringstream message;
    message << operation << " failed: HRESULT 0x" << std::hex
            << static_cast<unsigned long>(result);
    throw std::runtime_error(message.str());
  }
}

void check_hip(hipError_t result, const char* operation) {
  if (result != hipSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: HIP " + std::to_string(result) +
        " (" + hipGetErrorString(result) + ")");
  }
}

std::uint64_t align_heap_size(std::size_t size) {
  const auto alignment =
      static_cast<std::uint64_t>(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
  return (static_cast<std::uint64_t>(size) + alignment - 1) & ~(alignment - 1);
}

bool luid_equal(const LUID& lhs, const char rhs[8]) {
  std::array<std::uint8_t, 8> bytes{};
  std::memcpy(bytes.data(), &lhs, bytes.size());
  return std::memcmp(bytes.data(), rhs, bytes.size()) == 0;
}

std::wstring object_name(const wchar_t* base_name, const wchar_t* suffix) {
  if (base_name == nullptr || base_name[0] == L'\0') {
    throw std::invalid_argument("base_name must not be empty");
  }
  return std::wstring(base_name) + suffix;
}

struct DeviceBinding {
  ComPtr<IDXGIFactory6> factory;
  ComPtr<IDXGIAdapter4> adapter;
  ComPtr<ID3D12Device> device;
};

DeviceBinding bind_device(int hip_device) {
  hipDeviceProp_t hip_properties{};
  check_hip(
      hipGetDeviceProperties(&hip_properties, hip_device),
      "hipGetDeviceProperties");

  DeviceBinding binding;
  check_hr(
      CreateDXGIFactory2(0, IID_PPV_ARGS(&binding.factory)),
      "CreateDXGIFactory2");
  for (std::uint32_t index = 0;; ++index) {
    ComPtr<IDXGIAdapter4> candidate;
    const HRESULT result = binding.factory->EnumAdapterByGpuPreference(
        index,
        DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        IID_PPV_ARGS(&candidate));
    if (result == DXGI_ERROR_NOT_FOUND) {
      break;
    }
    check_hr(result, "EnumAdapterByGpuPreference");
    DXGI_ADAPTER_DESC3 description{};
    check_hr(candidate->GetDesc3(&description), "IDXGIAdapter4::GetDesc3");
    if ((description.Flags & DXGI_ADAPTER_FLAG3_SOFTWARE) == 0 &&
        description.VendorId == 0x1002 &&
        luid_equal(description.AdapterLuid, hip_properties.luid)) {
      binding.adapter = std::move(candidate);
      check_hr(
          D3D12CreateDevice(
              binding.adapter.Get(),
              D3D_FEATURE_LEVEL_12_0,
              IID_PPV_ARGS(&binding.device)),
          "D3D12CreateDevice");
      return binding;
    }
  }
  throw std::runtime_error(
      "No D3D12 adapter LUID matched HIP device " +
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
      D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER |
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  return description;
}

ComPtr<ID3D12Resource> create_placed_buffer(
    ID3D12Device* device,
    ID3D12Heap* heap,
    std::size_t size) {
  const auto description = buffer_description(size);
  ComPtr<ID3D12Resource> resource;
  check_hr(
      device->CreatePlacedResource(
          heap,
          0,
          &description,
          D3D12_RESOURCE_STATE_COMMON,
          nullptr,
          IID_PPV_ARGS(&resource)),
      "ID3D12Device::CreatePlacedResource");
  return resource;
}

}  // namespace

struct wavmg_d3d12_context {
  int hip_device = -1;
  std::size_t size = 0;
  std::uint64_t heap_size = 0;
  DeviceBinding binding;
  ComPtr<ID3D12Heap> heap;
  ComPtr<ID3D12Resource> resource;
  ComPtr<ID3D12Fence> fence;
  HANDLE heap_handle = nullptr;
  HANDLE fence_handle = nullptr;
  hipExternalMemory_t external_memory = nullptr;
  hipExternalSemaphore_t external_semaphore = nullptr;
  void* device_pointer = nullptr;

  ~wavmg_d3d12_context() {
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
};

namespace {

void import_hip_objects(wavmg_d3d12_context& context) {
  check_hip(hipSetDevice(context.hip_device), "hipSetDevice");

  hipExternalMemoryHandleDesc memory_desc{};
  memory_desc.type = hipExternalMemoryHandleTypeD3D12Heap;
  memory_desc.handle.win32.handle = context.heap_handle;
  memory_desc.size = context.heap_size;
  check_hip(
      hipImportExternalMemory(&context.external_memory, &memory_desc),
      "hipImportExternalMemory(D3D12Heap)");

  hipExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.offset = 0;
  buffer_desc.size = context.size;
  check_hip(
      hipExternalMemoryGetMappedBuffer(
          &context.device_pointer, context.external_memory, &buffer_desc),
      "hipExternalMemoryGetMappedBuffer");

  hipExternalSemaphoreHandleDesc semaphore_desc{};
  semaphore_desc.type = hipExternalSemaphoreHandleTypeD3D12Fence;
  semaphore_desc.handle.win32.handle = context.fence_handle;
  check_hip(
      hipImportExternalSemaphore(
          &context.external_semaphore, &semaphore_desc),
      "hipImportExternalSemaphore(D3D12Fence)");
}

void initialize_creator(
    wavmg_d3d12_context& context,
    const wchar_t* base_name) {
  D3D12_HEAP_DESC heap_desc{};
  heap_desc.SizeInBytes = context.heap_size;
  heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
  heap_desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap_desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap_desc.Properties.CreationNodeMask = 1;
  heap_desc.Properties.VisibleNodeMask = 1;
  heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
  heap_desc.Flags = static_cast<D3D12_HEAP_FLAGS>(
      D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER);
  check_hr(
      context.binding.device->CreateHeap(
          &heap_desc, IID_PPV_ARGS(&context.heap)),
      "ID3D12Device::CreateHeap(cross-adapter)");
  context.resource = create_placed_buffer(
      context.binding.device.Get(), context.heap.Get(), context.size);

  const auto memory_name = object_name(base_name, L".memory");
  check_hr(
      context.binding.device->CreateSharedHandle(
          context.heap.Get(),
          nullptr,
          GENERIC_ALL,
          memory_name.c_str(),
          &context.heap_handle),
      "ID3D12Device::CreateSharedHandle(memory)");

  const auto fence_flags = static_cast<D3D12_FENCE_FLAGS>(
      D3D12_FENCE_FLAG_SHARED | D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER);
  check_hr(
      context.binding.device->CreateFence(
          0, fence_flags, IID_PPV_ARGS(&context.fence)),
      "ID3D12Device::CreateFence(cross-adapter)");
  const auto fence_name = object_name(base_name, L".fence");
  check_hr(
      context.binding.device->CreateSharedHandle(
          context.fence.Get(),
          nullptr,
          GENERIC_ALL,
          fence_name.c_str(),
          &context.fence_handle),
      "ID3D12Device::CreateSharedHandle(fence)");
  import_hip_objects(context);
}

void initialize_opener(
    wavmg_d3d12_context& context,
    const wchar_t* base_name) {
  HANDLE source_heap_handle = nullptr;
  HANDLE source_fence_handle = nullptr;
  const auto memory_name = object_name(base_name, L".memory");
  const auto fence_name = object_name(base_name, L".fence");
  try {
    check_hr(
        context.binding.device->OpenSharedHandleByName(
            memory_name.c_str(), GENERIC_ALL, &source_heap_handle),
        "ID3D12Device::OpenSharedHandleByName(memory)");
    check_hr(
        context.binding.device->OpenSharedHandle(
            source_heap_handle, IID_PPV_ARGS(&context.heap)),
        "ID3D12Device::OpenSharedHandle(memory)");
    context.resource = create_placed_buffer(
        context.binding.device.Get(), context.heap.Get(), context.size);
    check_hr(
        context.binding.device->CreateSharedHandle(
            context.heap.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &context.heap_handle),
        "ID3D12Device::CreateSharedHandle(peer memory)");

    check_hr(
        context.binding.device->OpenSharedHandleByName(
            fence_name.c_str(), GENERIC_ALL, &source_fence_handle),
        "ID3D12Device::OpenSharedHandleByName(fence)");
    check_hr(
        context.binding.device->OpenSharedHandle(
            source_fence_handle, IID_PPV_ARGS(&context.fence)),
        "ID3D12Device::OpenSharedHandle(fence)");
    check_hr(
        context.binding.device->CreateSharedHandle(
            context.fence.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &context.fence_handle),
        "ID3D12Device::CreateSharedHandle(peer fence)");
    import_hip_objects(context);
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

template <typename Initializer>
int create_context(
    int hip_device,
    const wchar_t* base_name,
    std::size_t size,
    wavmg_d3d12_context** output_context,
    void** output_pointer,
    Initializer initializer) {
  if (size == 0 || output_context == nullptr || output_pointer == nullptr) {
    g_last_error = "size and output pointers must be valid";
    return 1;
  }
  try {
    auto context = std::make_unique<wavmg_d3d12_context>();
    context->hip_device = hip_device;
    context->size = size;
    context->heap_size = align_heap_size(size);
    context->binding = bind_device(hip_device);
    initializer(*context, base_name);
    *output_pointer = context->device_pointer;
    *output_context = context.release();
    g_last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return 1;
  }
}

}  // namespace

int wavmg_d3d12_create(
    int hip_device,
    const wchar_t* base_name,
    std::size_t size,
    wavmg_d3d12_context** context,
    void** device_pointer) {
  return create_context(
      hip_device,
      base_name,
      size,
      context,
      device_pointer,
      initialize_creator);
}

int wavmg_d3d12_open(
    int hip_device,
    const wchar_t* base_name,
    std::size_t size,
    wavmg_d3d12_context** context,
    void** device_pointer) {
  return create_context(
      hip_device,
      base_name,
      size,
      context,
      device_pointer,
      initialize_opener);
}

int wavmg_d3d12_signal(
    wavmg_d3d12_context* context,
    std::uint64_t value,
    void* stream) {
  if (context == nullptr) {
    g_last_error = "context must not be null";
    return 1;
  }
  try {
    check_hip(hipSetDevice(context->hip_device), "hipSetDevice");
    hipExternalSemaphoreSignalParams parameters{};
    parameters.params.fence.value = value;
    check_hip(
        hipSignalExternalSemaphoresAsync(
            &context->external_semaphore,
            &parameters,
            1,
            reinterpret_cast<hipStream_t>(stream)),
        "hipSignalExternalSemaphoresAsync");
    g_last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return 1;
  }
}

int wavmg_d3d12_wait(
    wavmg_d3d12_context* context,
    std::uint64_t value,
    void* stream) {
  if (context == nullptr) {
    g_last_error = "context must not be null";
    return 1;
  }
  try {
    check_hip(hipSetDevice(context->hip_device), "hipSetDevice");
    hipExternalSemaphoreWaitParams parameters{};
    parameters.params.fence.value = value;
    check_hip(
        hipWaitExternalSemaphoresAsync(
            &context->external_semaphore,
            &parameters,
            1,
            reinterpret_cast<hipStream_t>(stream)),
        "hipWaitExternalSemaphoresAsync");
    g_last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return 1;
  }
}

int wavmg_d3d12_close(wavmg_d3d12_context* context) {
  try {
    delete context;
    g_last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return 1;
  }
}

const char* wavmg_d3d12_last_error() {
  return g_last_error.c_str();
}
