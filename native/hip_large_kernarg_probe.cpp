#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr std::size_t kArgumentBytes = 5U << 10;
constexpr std::uint64_t kMagic = 0x5741564d47524944ULL;

struct alignas(16) LargeArguments {
  std::uint64_t magic;
  std::uint64_t* output;
  std::array<std::uint8_t, kArgumentBytes - 16> padding;
};

static_assert(sizeof(LargeArguments) == kArgumentBytes);

__global__ void large_arguments_kernel(LargeArguments arguments) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    arguments.output[0] = arguments.magic;
    arguments.output[1] = reinterpret_cast<std::uintptr_t>(arguments.output);
    arguments.output[2] = sizeof(long);
    arguments.output[3] = sizeof(void*);
    arguments.output[4] = sizeof(LargeArguments);
    arguments.output[5] = arguments.padding.front();
    arguments.output[6] = arguments.padding.back();
    arguments.output[7] = sizeof(ulong2);
    arguments.output[8] = sizeof(arguments.output[0] == 0 ? ulong2{}.x : ulong2{}.x);
    arguments.output[9] = alignof(ulong2);
  }
}

void check(hipError_t result, const char* operation) {
  if (result == hipSuccess) return;
  throw std::runtime_error(std::string(operation) + " failed: HIP " +
                           std::to_string(static_cast<int>(result)) + " (" +
                           hipGetErrorString(result) + ")");
}

bool run_launch(hipFunction_t function, LargeArguments& arguments,
                std::uint64_t* device_output, bool use_extra) {
  check(hipMemset(device_output, 0, 10 * sizeof(std::uint64_t)), "hipMemset");

  hipError_t result;
  if (use_extra) {
    std::size_t argument_bytes = sizeof(arguments);
    void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &arguments,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &argument_bytes,
                     HIP_LAUNCH_PARAM_END};
    result = hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                   nullptr, extra);
  } else {
    void* parameters[] = {&arguments};
    result = hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                   parameters, nullptr);
  }
  check(result, use_extra ? "hipModuleLaunchKernel(extra)"
                          : "hipModuleLaunchKernel(kernelParams)");
  check(hipDeviceSynchronize(), "hipDeviceSynchronize");

  std::array<std::uint64_t, 10> host_output{};
  check(hipMemcpy(host_output.data(), device_output,
                  host_output.size() * sizeof(std::uint64_t),
                  hipMemcpyDeviceToHost),
        "hipMemcpy(device-to-host)");
  const bool passed =
      host_output[0] == kMagic &&
      host_output[1] == reinterpret_cast<std::uintptr_t>(device_output) &&
      host_output[2] == sizeof(long) && host_output[3] == sizeof(void*) &&
      host_output[4] == sizeof(LargeArguments) && host_output[5] == 0x5a &&
      host_output[6] == 0xa5 && host_output[7] == sizeof(ulong2) &&
      host_output[8] == sizeof(ulong2{}.x) &&
      host_output[9] == alignof(ulong2);
  std::cout << (use_extra ? "extra" : "kernelParams")
            << ": passed=" << (passed ? "true" : "false")
            << " device_long=" << host_output[2]
            << " device_pointer=" << host_output[3]
            << " argument_bytes=" << host_output[4]
            << " device_ulong2=" << host_output[7]
            << " device_ulong_lane=" << host_output[8]
            << " device_ulong2_align=" << host_output[9] << '\n';
  return passed;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int device = argc > 1 ? std::stoi(argv[1]) : 0;
    check(hipSetDevice(device), "hipSetDevice");

    std::uint64_t* device_output = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&device_output),
                    10 * sizeof(std::uint64_t)),
          "hipMalloc");

    LargeArguments arguments{};
    arguments.magic = kMagic;
    arguments.output = device_output;
    arguments.padding.front() = 0x5a;
    arguments.padding.back() = 0xa5;

    hipFunction_t function = nullptr;
    check(hipGetFuncBySymbol(
              &function, reinterpret_cast<const void*>(large_arguments_kernel)),
          "hipGetFuncBySymbol");

    const bool extra_passed =
        run_launch(function, arguments, device_output, true);
    const bool parameters_passed =
        run_launch(function, arguments, device_output, false);
    check(hipFree(device_output), "hipFree");
    return extra_passed && parameters_passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
