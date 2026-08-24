#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <hip/hip_runtime.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr std::uint64_t kBufferSize = 64 * 1024;
constexpr std::uint64_t kHeapSize = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

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

std::string luid_string(const LUID& luid) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(8)
         << static_cast<std::uint32_t>(luid.HighPart) << ':' << std::setw(8)
         << luid.LowPart;
  return stream.str();
}

bool luid_equal(const LUID& lhs, const char rhs[8]) {
  std::array<std::uint8_t, 8> bytes{};
  std::memcpy(bytes.data(), &lhs, bytes.size());
  return std::memcmp(bytes.data(), rhs, bytes.size()) == 0;
}

std::vector<std::uint8_t> make_pattern(std::size_t source) {
  std::vector<std::uint8_t> data(kBufferSize);
  for (std::size_t index = 0; index < data.size(); ++index) {
    data[index] = static_cast<std::uint8_t>((index * 37 + source * 83 + 19) & 0xff);
  }
  return data;
}

struct AdapterDevice {
  ComPtr<IDXGIAdapter4> adapter;
  ComPtr<ID3D12Device> device;
  DXGI_ADAPTER_DESC3 description{};
  int hip_device = -1;
};

struct HipImport {
  hipExternalMemory_t memory = nullptr;
  void* pointer = nullptr;

  HipImport() = default;
  HipImport(const HipImport&) = delete;
  HipImport& operator=(const HipImport&) = delete;
  HipImport(HipImport&& other) noexcept
      : memory(other.memory), pointer(other.pointer) {
    other.memory = nullptr;
    other.pointer = nullptr;
  }
  ~HipImport() {
    if (memory != nullptr) {
      (void)hipDestroyExternalMemory(memory);
    }
  }
};

HipImport import_heap_into_hip(int hip_device, HANDLE handle) {
  check_hip(hipSetDevice(hip_device), "hipSetDevice");
  hipExternalMemoryHandleDesc memory_desc{};
  memory_desc.type = hipExternalMemoryHandleTypeD3D12Heap;
  memory_desc.handle.win32.handle = handle;
  memory_desc.size = kHeapSize;

  HipImport imported;
  check_hip(
      hipImportExternalMemory(&imported.memory, &memory_desc),
      "hipImportExternalMemory(D3D12Heap)");
  hipExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.offset = 0;
  buffer_desc.size = kBufferSize;
  check_hip(
      hipExternalMemoryGetMappedBuffer(
          &imported.pointer, imported.memory, &buffer_desc),
      "hipExternalMemoryGetMappedBuffer");
  return imported;
}

hipExternalSemaphore_t import_fence_into_hip(int hip_device, HANDLE handle) {
  check_hip(hipSetDevice(hip_device), "hipSetDevice");
  hipExternalSemaphoreHandleDesc semaphore_desc{};
  semaphore_desc.type = hipExternalSemaphoreHandleTypeD3D12Fence;
  semaphore_desc.handle.win32.handle = handle;
  hipExternalSemaphore_t semaphore = nullptr;
  check_hip(
      hipImportExternalSemaphore(&semaphore, &semaphore_desc),
      "hipImportExternalSemaphore(D3D12Fence)");
  return semaphore;
}

}  // namespace

int main() {
  try {
    int hip_count = 0;
    check_hip(hipGetDeviceCount(&hip_count), "hipGetDeviceCount");
    std::vector<hipDeviceProp_t> hip_properties(static_cast<std::size_t>(hip_count));
    for (int device = 0; device < hip_count; ++device) {
      check_hip(
          hipGetDeviceProperties(&hip_properties[device], device),
          "hipGetDeviceProperties");
      char bus_id[32]{};
      check_hip(
          hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), device),
          "hipDeviceGetPCIBusId");
      std::cout << "HIP " << device << " PCI " << bus_id << "\n";
    }

    ComPtr<IDXGIFactory6> factory;
    check_hr(
        CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
        "CreateDXGIFactory2");
    std::vector<AdapterDevice> adapters;
    for (std::uint32_t index = 0;; ++index) {
      ComPtr<IDXGIAdapter4> adapter;
      const HRESULT result = factory->EnumAdapterByGpuPreference(
          index,
          DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
          IID_PPV_ARGS(&adapter));
      if (result == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      check_hr(result, "EnumAdapterByGpuPreference");
      DXGI_ADAPTER_DESC3 description{};
      check_hr(adapter->GetDesc3(&description), "IDXGIAdapter4::GetDesc3");
      if ((description.Flags & DXGI_ADAPTER_FLAG3_SOFTWARE) != 0 ||
          description.VendorId != 0x1002) {
        continue;
      }
      AdapterDevice entry;
      entry.adapter = adapter;
      entry.description = description;
      check_hr(
          D3D12CreateDevice(
              adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&entry.device)),
          "D3D12CreateDevice");
      for (int hip_device = 0; hip_device < hip_count; ++hip_device) {
        if (luid_equal(description.AdapterLuid, hip_properties[hip_device].luid)) {
          entry.hip_device = hip_device;
          break;
        }
      }
      std::wcout << L"DXGI " << adapters.size() << L" " << description.Description;
      std::cout << " LUID " << luid_string(description.AdapterLuid)
                << " HIP " << entry.hip_device << "\n";
      adapters.push_back(std::move(entry));
    }

    if (adapters.size() < 2) {
      throw std::runtime_error("Fewer than two AMD D3D12 adapters were found");
    }

    bool d3d12_open_worked = false;
    bool same_device_import_worked = false;
    bool cross_device_import_worked = false;
    bool cross_device_semaphore_worked = false;
    for (std::size_t source = 0; source < adapters.size(); ++source) {
      auto& source_adapter = adapters[source];
      std::cout << "Source DXGI " << source << " HIP "
                << source_adapter.hip_device << "\n";
      if (source_adapter.hip_device < 0) {
        std::cout << "  no HIP LUID match; skipping\n";
        continue;
      }

      D3D12_HEAP_DESC heap_desc{};
      heap_desc.SizeInBytes = kHeapSize;
      heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
      heap_desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
      heap_desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
      heap_desc.Properties.CreationNodeMask = 1;
      heap_desc.Properties.VisibleNodeMask = 1;
      heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
      heap_desc.Flags = static_cast<D3D12_HEAP_FLAGS>(
          D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER);
      ComPtr<ID3D12Heap> heap;
      check_hr(
          source_adapter.device->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap)),
          "ID3D12Device::CreateHeap(cross-adapter)");

      D3D12_RESOURCE_DESC resource_desc{};
      resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      resource_desc.Alignment = 0;
      resource_desc.Width = kBufferSize;
      resource_desc.Height = 1;
      resource_desc.DepthOrArraySize = 1;
      resource_desc.MipLevels = 1;
      resource_desc.Format = DXGI_FORMAT_UNKNOWN;
      resource_desc.SampleDesc.Count = 1;
      resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      resource_desc.Flags = static_cast<D3D12_RESOURCE_FLAGS>(
          D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER |
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
      ComPtr<ID3D12Resource> resource;
      check_hr(
          source_adapter.device->CreatePlacedResource(
              heap.Get(),
              0,
              &resource_desc,
              D3D12_RESOURCE_STATE_COMMON,
              nullptr,
              IID_PPV_ARGS(&resource)),
          "ID3D12Device::CreatePlacedResource(cross-adapter)");

      HANDLE shared_handle = nullptr;
      check_hr(
          source_adapter.device->CreateSharedHandle(
              heap.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle),
          "ID3D12Device::CreateSharedHandle");
      std::vector<HANDLE> handles_by_hip(
          static_cast<std::size_t>(hip_count), shared_handle);
      std::vector<HANDLE> peer_owned_handles;
      std::vector<ComPtr<ID3D12Heap>> peer_heaps;
      std::vector<ComPtr<ID3D12Resource>> peer_resources;
      try {
        for (std::size_t target = 0; target < adapters.size(); ++target) {
          if (target == source) {
            continue;
          }
          ComPtr<ID3D12Heap> peer_heap;
          const HRESULT open_result = adapters[target].device->OpenSharedHandle(
              shared_handle, IID_PPV_ARGS(&peer_heap));
          std::cout << "  D3D12 target " << target << ": OpenSharedHandle="
                    << (SUCCEEDED(open_result) ? "yes" : "no") << "\n";
          d3d12_open_worked = d3d12_open_worked || SUCCEEDED(open_result);
          if (SUCCEEDED(open_result)) {
            ComPtr<ID3D12Resource> peer_resource;
            const HRESULT resource_result =
                adapters[target].device->CreatePlacedResource(
                    peer_heap.Get(),
                    0,
                    &resource_desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(&peer_resource));
            std::cout << "    CreatePlacedResource="
                      << (SUCCEEDED(resource_result) ? "yes" : "no") << "\n";
            HANDLE peer_handle = nullptr;
            const HRESULT handle_result =
                adapters[target].device->CreateSharedHandle(
                    peer_heap.Get(),
                    nullptr,
                    GENERIC_ALL,
                    nullptr,
                    &peer_handle);
            std::cout << "    CreateSharedHandle(peer-owned)="
                      << (SUCCEEDED(handle_result) ? "yes" : "no") << "\n";
            if (SUCCEEDED(resource_result) && SUCCEEDED(handle_result) &&
                adapters[target].hip_device >= 0) {
              handles_by_hip[adapters[target].hip_device] = peer_handle;
              peer_owned_handles.push_back(peer_handle);
              peer_resources.push_back(std::move(peer_resource));
              peer_heaps.push_back(std::move(peer_heap));
            } else if (peer_handle != nullptr) {
              CloseHandle(peer_handle);
            }
          }
        }

        const auto expected = make_pattern(source);
        {
          auto imported = import_heap_into_hip(
              source_adapter.hip_device, shared_handle);
          check_hip(
              hipMemcpy(
                  imported.pointer,
                  expected.data(),
                  expected.size(),
                  hipMemcpyHostToDevice),
              "hipMemcpy(source write)");
          std::vector<std::uint8_t> observed(kBufferSize);
          check_hip(
              hipMemcpy(
                  observed.data(),
                  imported.pointer,
                  observed.size(),
                  hipMemcpyDeviceToHost),
              "hipMemcpy(source read)");
          const bool correct = observed == expected;
          same_device_import_worked = same_device_import_worked || correct;
          std::cout << "  HIP source " << source_adapter.hip_device
                    << ": import=yes data="
                    << (correct ? "correct" : "different") << "\n";
        }

        for (int target_hip = 0; target_hip < hip_count; ++target_hip) {
          if (target_hip == source_adapter.hip_device) {
            continue;
          }
          try {
            auto imported = import_heap_into_hip(
                target_hip, handles_by_hip[target_hip]);
            std::vector<std::uint8_t> observed(kBufferSize);
            check_hip(
                hipMemcpy(
                    observed.data(),
                    imported.pointer,
                    observed.size(),
                    hipMemcpyDeviceToHost),
                "hipMemcpy(peer read)");
            const bool correct = observed == expected;
            cross_device_import_worked = cross_device_import_worked || correct;
            std::cout << "  HIP peer " << target_hip << ": import=yes data="
                      << (correct ? "correct" : "different") << "\n";
          } catch (const std::exception& error) {
            std::cout << "  HIP peer " << target_hip << ": " << error.what()
                      << "\n";
          }
        }

        for (std::size_t target = 0; target < adapters.size(); ++target) {
          if (target == source || adapters[target].hip_device < 0) {
            continue;
          }
          hipExternalSemaphore_t source_semaphore = nullptr;
          hipExternalSemaphore_t peer_semaphore = nullptr;
          hipStream_t source_stream = nullptr;
          hipStream_t peer_stream = nullptr;
          HANDLE source_fence_handle = nullptr;
          HANDLE peer_fence_handle = nullptr;
          try {
            const auto fence_flags = static_cast<D3D12_FENCE_FLAGS>(
                D3D12_FENCE_FLAG_SHARED |
                D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER);
            ComPtr<ID3D12Fence> source_fence;
            check_hr(
                source_adapter.device->CreateFence(
                    0, fence_flags, IID_PPV_ARGS(&source_fence)),
                "ID3D12Device::CreateFence(cross-adapter)");
            check_hr(
                source_adapter.device->CreateSharedHandle(
                    source_fence.Get(),
                    nullptr,
                    GENERIC_ALL,
                    nullptr,
                    &source_fence_handle),
                "ID3D12Device::CreateSharedHandle(fence)");
            ComPtr<ID3D12Fence> peer_fence;
            check_hr(
                adapters[target].device->OpenSharedHandle(
                    source_fence_handle, IID_PPV_ARGS(&peer_fence)),
                "ID3D12Device::OpenSharedHandle(fence)");
            check_hr(
                adapters[target].device->CreateSharedHandle(
                    peer_fence.Get(),
                    nullptr,
                    GENERIC_ALL,
                    nullptr,
                    &peer_fence_handle),
                "ID3D12Device::CreateSharedHandle(peer fence)");

            const int source_hip = source_adapter.hip_device;
            const int peer_hip = adapters[target].hip_device;
            auto source_import =
                import_heap_into_hip(source_hip, handles_by_hip[source_hip]);
            auto peer_import =
                import_heap_into_hip(peer_hip, handles_by_hip[peer_hip]);
            source_semaphore =
                import_fence_into_hip(source_hip, source_fence_handle);
            peer_semaphore =
                import_fence_into_hip(peer_hip, peer_fence_handle);

            check_hip(hipSetDevice(source_hip), "hipSetDevice(source)");
            check_hip(
                hipStreamCreateWithFlags(&source_stream, hipStreamNonBlocking),
                "hipStreamCreateWithFlags(source)");
            check_hip(hipSetDevice(peer_hip), "hipSetDevice(peer)");
            check_hip(
                hipStreamCreateWithFlags(&peer_stream, hipStreamNonBlocking),
                "hipStreamCreateWithFlags(peer)");

            const auto synchronized_pattern = make_pattern(source + 17);
            check_hip(hipSetDevice(source_hip), "hipSetDevice(source)");
            check_hip(
                hipMemcpyAsync(
                    source_import.pointer,
                    synchronized_pattern.data(),
                    synchronized_pattern.size(),
                    hipMemcpyHostToDevice,
                    source_stream),
                "hipMemcpyAsync(source write)");
            hipExternalSemaphoreSignalParams signal_params{};
            signal_params.params.fence.value = 1;
            check_hip(
                hipSignalExternalSemaphoresAsync(
                    &source_semaphore, &signal_params, 1, source_stream),
                "hipSignalExternalSemaphoresAsync");

            std::vector<std::uint8_t> synchronized_observed(kBufferSize);
            check_hip(hipSetDevice(peer_hip), "hipSetDevice(peer)");
            hipExternalSemaphoreWaitParams wait_params{};
            wait_params.params.fence.value = 1;
            check_hip(
                hipWaitExternalSemaphoresAsync(
                    &peer_semaphore, &wait_params, 1, peer_stream),
                "hipWaitExternalSemaphoresAsync");
            check_hip(
                hipMemcpyAsync(
                    synchronized_observed.data(),
                    peer_import.pointer,
                    synchronized_observed.size(),
                    hipMemcpyDeviceToHost,
                    peer_stream),
                "hipMemcpyAsync(peer read)");
            check_hip(hipStreamSynchronize(peer_stream), "hipStreamSynchronize(peer)");
            const bool synchronized_correct =
                synchronized_observed == synchronized_pattern;
            cross_device_semaphore_worked =
                cross_device_semaphore_worked || synchronized_correct;
            std::cout << "  HIP " << source_hip << " -> HIP " << peer_hip
                      << " D3D12 fence synchronization="
                      << (synchronized_correct ? "correct" : "different")
                      << "\n";
          } catch (const std::exception& error) {
            std::cout << "  D3D12 fence/HIP semaphore test: " << error.what()
                      << "\n";
          }
          if (source_stream != nullptr) {
            (void)hipSetDevice(source_adapter.hip_device);
            (void)hipStreamDestroy(source_stream);
          }
          if (peer_stream != nullptr) {
            (void)hipSetDevice(adapters[target].hip_device);
            (void)hipStreamDestroy(peer_stream);
          }
          if (source_semaphore != nullptr) {
            (void)hipSetDevice(source_adapter.hip_device);
            (void)hipDestroyExternalSemaphore(source_semaphore);
          }
          if (peer_semaphore != nullptr) {
            (void)hipSetDevice(adapters[target].hip_device);
            (void)hipDestroyExternalSemaphore(peer_semaphore);
          }
          if (source_fence_handle != nullptr) {
            CloseHandle(source_fence_handle);
          }
          if (peer_fence_handle != nullptr) {
            CloseHandle(peer_fence_handle);
          }
        }
      } catch (...) {
        for (HANDLE peer_handle : peer_owned_handles) {
          CloseHandle(peer_handle);
        }
        CloseHandle(shared_handle);
        throw;
      }
      for (HANDLE peer_handle : peer_owned_handles) {
        CloseHandle(peer_handle);
      }
      CloseHandle(shared_handle);
    }

    std::cout << "d3d12_cross_adapter_open_worked=" << d3d12_open_worked << "\n";
    std::cout << "same_device_import_worked=" << same_device_import_worked << "\n";
    std::cout << "cross_device_import_worked=" << cross_device_import_worked << "\n";
    std::cout << "cross_device_semaphore_worked="
              << cross_device_semaphore_worked << "\n";
    return cross_device_import_worked && cross_device_semaphore_worked ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << "\n";
    return 1;
  }
}
