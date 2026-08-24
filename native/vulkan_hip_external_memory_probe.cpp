#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>

#include <hip/hip_runtime.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr VkDeviceSize kBufferSize = 64 * 1024;

void check_vk(VkResult result, const char* operation) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + " failed: Vulkan " + std::to_string(result));
  }
}

void check_hip(hipError_t result, const char* operation) {
  if (result != hipSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: HIP " + std::to_string(result) +
        " (" + hipGetErrorString(result) + ")");
  }
}

std::string uuid_string(const std::uint8_t* uuid) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < VK_UUID_SIZE; ++index) {
    if (index == 4 || index == 6 || index == 8 || index == 10) {
      stream << '-';
    }
    stream << std::setw(2) << static_cast<unsigned int>(uuid[index]);
  }
  return stream.str();
}

bool has_extension(VkPhysicalDevice physical_device, const char* name) {
  std::uint32_t count = 0;
  check_vk(
      vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr),
      "vkEnumerateDeviceExtensionProperties(count)");
  std::vector<VkExtensionProperties> extensions(count);
  check_vk(
      vkEnumerateDeviceExtensionProperties(
          physical_device, nullptr, &count, extensions.data()),
      "vkEnumerateDeviceExtensionProperties(list)");
  for (const auto& extension : extensions) {
    if (std::strcmp(extension.extensionName, name) == 0) {
      return true;
    }
  }
  return false;
}

std::uint32_t find_memory_type(
    VkPhysicalDevice physical_device,
    std::uint32_t allowed_types) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
  for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
    if ((allowed_types & (1u << index)) != 0 &&
        (properties.memoryTypes[index].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
      return index;
    }
  }
  for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
    if ((allowed_types & (1u << index)) != 0) {
      return index;
    }
  }
  throw std::runtime_error("No compatible Vulkan memory type");
}

struct VulkanAllocation {
  VkDevice device = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  HANDLE handle = nullptr;
  VkDeviceSize allocation_size = 0;

  VulkanAllocation() = default;
  VulkanAllocation(const VulkanAllocation&) = delete;
  VulkanAllocation& operator=(const VulkanAllocation&) = delete;
  VulkanAllocation(VulkanAllocation&& other) noexcept
      : device(other.device),
        buffer(other.buffer),
        memory(other.memory),
        handle(other.handle),
        allocation_size(other.allocation_size) {
    other.device = VK_NULL_HANDLE;
    other.buffer = VK_NULL_HANDLE;
    other.memory = VK_NULL_HANDLE;
    other.handle = nullptr;
  }

  ~VulkanAllocation() {
    if (device != VK_NULL_HANDLE) {
      if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
      }
      if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
      }
      vkDestroyDevice(device, nullptr);
    }
  }
};

VulkanAllocation create_exported_allocation(VkPhysicalDevice physical_device) {
  const char* extensions[] = {VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME};
  VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.enabledExtensionCount = 1;
  device_info.ppEnabledExtensionNames = extensions;

  VulkanAllocation allocation;
  check_vk(
      vkCreateDevice(physical_device, &device_info, nullptr, &allocation.device),
      "vkCreateDevice");

  VkExternalMemoryBufferCreateInfo external_buffer{
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  external_buffer.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
  VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.pNext = &external_buffer;
  buffer_info.size = kBufferSize;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check_vk(
      vkCreateBuffer(allocation.device, &buffer_info, nullptr, &allocation.buffer),
      "vkCreateBuffer");

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(allocation.device, allocation.buffer, &requirements);
  allocation.allocation_size = requirements.size;

  VkExportMemoryAllocateInfo export_info{
      VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
  export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
  VkMemoryAllocateInfo memory_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  memory_info.pNext = &export_info;
  memory_info.allocationSize = requirements.size;
  memory_info.memoryTypeIndex =
      find_memory_type(physical_device, requirements.memoryTypeBits);
  check_vk(
      vkAllocateMemory(allocation.device, &memory_info, nullptr, &allocation.memory),
      "vkAllocateMemory");
  check_vk(
      vkBindBufferMemory(allocation.device, allocation.buffer, allocation.memory, 0),
      "vkBindBufferMemory");

  auto get_handle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
      vkGetDeviceProcAddr(allocation.device, "vkGetMemoryWin32HandleKHR"));
  if (get_handle == nullptr) {
    throw std::runtime_error("vkGetMemoryWin32HandleKHR is unavailable");
  }
  VkMemoryGetWin32HandleInfoKHR handle_info{
      VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  handle_info.memory = allocation.memory;
  handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
  check_vk(
      get_handle(allocation.device, &handle_info, &allocation.handle),
      "vkGetMemoryWin32HandleKHR");
  return allocation;
}

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

HipImport import_into_hip(
    int device,
    HANDLE handle,
    VkDeviceSize allocation_size) {
  check_hip(hipSetDevice(device), "hipSetDevice");
  hipExternalMemoryHandleDesc memory_desc{};
  memory_desc.type = hipExternalMemoryHandleTypeOpaqueWin32Kmt;
  memory_desc.handle.win32.handle = handle;
  memory_desc.size = allocation_size;

  HipImport imported;
  check_hip(
      hipImportExternalMemory(&imported.memory, &memory_desc),
      "hipImportExternalMemory");
  hipExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.offset = 0;
  buffer_desc.size = kBufferSize;
  check_hip(
      hipExternalMemoryGetMappedBuffer(
          &imported.pointer, imported.memory, &buffer_desc),
      "hipExternalMemoryGetMappedBuffer");
  return imported;
}

std::vector<std::uint8_t> make_pattern(std::size_t source_index) {
  std::vector<std::uint8_t> data(kBufferSize);
  for (std::size_t index = 0; index < data.size(); ++index) {
    data[index] = static_cast<std::uint8_t>((index * 29 + source_index * 71 + 13) & 0xff);
  }
  return data;
}

}  // namespace

int main() {
  try {
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "windows-amd-vulkan-hip-probe";
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    VkInstance instance = VK_NULL_HANDLE;
    check_vk(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

    std::uint32_t physical_count = 0;
    check_vk(
        vkEnumeratePhysicalDevices(instance, &physical_count, nullptr),
        "vkEnumeratePhysicalDevices(count)");
    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    check_vk(
        vkEnumeratePhysicalDevices(
            instance, &physical_count, physical_devices.data()),
        "vkEnumeratePhysicalDevices(list)");

    int hip_count = 0;
    check_hip(hipGetDeviceCount(&hip_count), "hipGetDeviceCount");
    struct HipPciIdentity {
      unsigned int domain = 0;
      unsigned int bus = 0;
      unsigned int device = 0;
      unsigned int function = 0;
    };
    std::vector<HipPciIdentity> hip_pci(static_cast<std::size_t>(hip_count));
    std::cout << "Vulkan devices: " << physical_count << "\n";
    std::cout << "HIP devices: " << hip_count << "\n";
    for (int hip_device = 0; hip_device < hip_count; ++hip_device) {
      char bus_id[32]{};
      check_hip(
          hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), hip_device),
          "hipDeviceGetPCIBusId");
      if (::sscanf_s(
              bus_id,
              "%x:%x:%x.%x",
              &hip_pci[hip_device].domain,
              &hip_pci[hip_device].bus,
              &hip_pci[hip_device].device,
              &hip_pci[hip_device].function) != 4) {
        throw std::runtime_error(std::string("Could not parse HIP PCI ID ") + bus_id);
      }
      std::cout << "HIP " << hip_device << " PCI " << bus_id << "\n";
    }

    bool same_device_import_worked = false;
    bool cross_device_import_worked = false;
    for (std::size_t source = 0; source < physical_devices.size(); ++source) {
      VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
      VkPhysicalDevicePCIBusInfoPropertiesEXT pci{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
      const bool has_pci_bus_info =
          has_extension(physical_devices[source], VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
      if (has_pci_bus_info) {
        id.pNext = &pci;
      }
      VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      properties.pNext = &id;
      vkGetPhysicalDeviceProperties2(physical_devices[source], &properties);
      std::cout << "Vulkan " << source << " " << properties.properties.deviceName
                << " UUID " << uuid_string(id.deviceUUID);
      if (has_pci_bus_info) {
        std::cout << " PCI " << std::hex << std::setfill('0') << std::setw(4)
                  << pci.pciDomain << ':' << std::setw(2) << pci.pciBus << ':'
                  << std::setw(2) << pci.pciDevice << '.' << pci.pciFunction
                  << std::dec;
      }
      std::cout << "\n";

      int source_hip_device = -1;
      if (has_pci_bus_info) {
        for (int hip_device = 0; hip_device < hip_count; ++hip_device) {
          const auto& candidate = hip_pci[hip_device];
          if (candidate.domain == pci.pciDomain && candidate.bus == pci.pciBus &&
              candidate.device == pci.pciDevice &&
              candidate.function == pci.pciFunction) {
            source_hip_device = hip_device;
            break;
          }
        }
      } else if (properties.properties.vendorID == 0x1002) {
        // AMD's Windows driver encodes the PCI bus/device in UUID bytes 4/5.
        // This is a fallback for the current driver, which does not expose
        // VK_EXT_pci_bus_info; the values are printed above for auditability.
        for (int hip_device = 0; hip_device < hip_count; ++hip_device) {
          const auto& candidate = hip_pci[hip_device];
          if (candidate.bus == id.deviceUUID[4] &&
              candidate.device == id.deviceUUID[5]) {
            source_hip_device = hip_device;
            break;
          }
        }
      }
      if (source_hip_device < 0) {
        std::cout << "  no matching HIP PCI device; skipping alias test\n";
        continue;
      }

      if (!has_extension(
              physical_devices[source],
              VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME)) {
        std::cout << "  external Win32 memory: unavailable\n";
        continue;
      }

      VkPhysicalDeviceExternalBufferInfo external_info{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
      external_info.flags = 0;
      external_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      external_info.handleType =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
      VkExternalBufferProperties external_properties{
          VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
      vkGetPhysicalDeviceExternalBufferProperties(
          physical_devices[source], &external_info, &external_properties);
      const auto features =
          external_properties.externalMemoryProperties.externalMemoryFeatures;
      std::cout << "  KMT exportable="
                << ((features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0)
                << " importable="
                << ((features & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0)
                << "\n";
      if ((features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) == 0) {
        continue;
      }

      auto allocation = create_exported_allocation(physical_devices[source]);
      const auto expected = make_pattern(source);
      std::vector<int> target_order;
      target_order.push_back(source_hip_device);
      for (int target = 0; target < hip_count; ++target) {
        if (target != source_hip_device) {
          target_order.push_back(target);
        }
      }
      for (const int target : target_order) {
        try {
          auto imported =
              import_into_hip(target, allocation.handle, allocation.allocation_size);
          if (target == source_hip_device) {
            check_hip(
                hipMemcpy(
                    imported.pointer,
                    expected.data(),
                    expected.size(),
                    hipMemcpyHostToDevice),
                "hipMemcpy(write)");
          }
          std::vector<std::uint8_t> observed(kBufferSize);
          check_hip(
              hipMemcpy(
                  observed.data(),
                  imported.pointer,
                  observed.size(),
                  hipMemcpyDeviceToHost),
              "hipMemcpy(read)");
          const bool correct = observed == expected;
          std::cout << "  HIP target " << target << ": import=yes data="
                    << (correct ? "correct" : "different") << "\n";
          if (target == source_hip_device) {
            same_device_import_worked = same_device_import_worked || correct;
          } else {
            cross_device_import_worked = cross_device_import_worked || correct;
          }
        } catch (const std::exception& error) {
          std::cout << "  HIP target " << target << ": " << error.what() << "\n";
        }
      }
    }

    vkDestroyInstance(instance, nullptr);
    std::cout << "same_device_import_worked=" << same_device_import_worked << "\n";
    std::cout << "cross_device_import_worked=" << cross_device_import_worked << "\n";
    return cross_device_import_worked ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << "\n";
    return 1;
  }
}
