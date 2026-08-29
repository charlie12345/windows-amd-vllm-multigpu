#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cstddef>

#if defined(_WIN32)
#define WAVMG_EXPORT extern "C" __declspec(dllexport)
#else
#define WAVMG_EXPORT extern "C" __attribute__((visibility("default")))
#endif

__global__ void wavmg_add_f32_kernel(
    const float* local,
    const float* peer_host,
    float* output,
    std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = local[index] + peer_host[index];
  }
}

__global__ void wavmg_add_f16_kernel(
    const __half* local,
    const __half* peer_host,
    __half* output,
    std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __float2half(
        __half2float(local[index]) + __half2float(peer_host[index]));
  }
}

__global__ void wavmg_add_bf16_kernel(
    const hip_bfloat16* local,
    const hip_bfloat16* peer_host,
    hip_bfloat16* output,
    std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = hip_bfloat16(
        static_cast<float>(local[index]) + static_cast<float>(peer_host[index]));
  }
}

__global__ void wavmg_copy_bytes_kernel(
    const unsigned char* source,
    unsigned char* destination,
    std::size_t size_bytes) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size_bytes) {
    destination[index] = source[index];
  }
}

template <typename Kernel, typename Value>
int launch_add(
    Kernel kernel,
    const Value* local,
    const Value* peer_host,
    Value* output,
    std::size_t count,
    void* stream_pointer) {
  if (count == 0) {
    return static_cast<int>(hipSuccess);
  }
  constexpr unsigned int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((count + threads - 1) / threads);
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  hipLaunchKernelGGL(
      kernel,
      dim3(blocks),
      dim3(threads),
      0,
      stream,
      local,
      peer_host,
      output,
      count);
  return static_cast<int>(hipGetLastError());
}

template <typename Kernel, typename Value>
int enqueue_all_reduce(
    Kernel kernel,
    const Value* local,
    Value* local_host,
    const Value* peer_host,
    Value* output,
    std::size_t count,
    uint32_t* own_ready,
    uint32_t* peer_ready,
    uint32_t* own_consumed,
    uint32_t* peer_consumed,
    uint32_t epoch,
    void* stream_pointer) {
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  hipError_t error = hipSuccess;
  if (epoch > 1) {
    error = hipStreamWaitValue32(
        stream,
        own_consumed,
        epoch - 1,
        hipStreamWaitValueGte,
        0xFFFFFFFFu);
    if (error != hipSuccess) {
      return static_cast<int>(error);
    }
  }
  error = hipMemcpyAsync(
      local_host,
      local,
      count * sizeof(Value),
      hipMemcpyDeviceToHost,
      stream);
  if (error != hipSuccess) {
    return static_cast<int>(error);
  }
  error = hipStreamWriteValue32(stream, own_ready, epoch, 0);
  if (error != hipSuccess) {
    return static_cast<int>(error);
  }
  error = hipStreamWaitValue32(
      stream,
      peer_ready,
      epoch,
      hipStreamWaitValueGte,
      0xFFFFFFFFu);
  if (error != hipSuccess) {
    return static_cast<int>(error);
  }
  const int launch_error = launch_add(
      kernel, local, peer_host, output, count, stream_pointer);
  if (launch_error != static_cast<int>(hipSuccess)) {
    return launch_error;
  }
  return static_cast<int>(
      hipStreamWriteValue32(stream, peer_consumed, epoch, 0));
}

#define WAVMG_ALL_REDUCE_EXPORT(NAME, TYPE, KERNEL)                         \
  WAVMG_EXPORT int NAME(                                                   \
      const TYPE* local,                                                   \
      TYPE* local_host,                                                    \
      const TYPE* peer_host,                                               \
      TYPE* output,                                                        \
      std::size_t count,                                                   \
      uint32_t* own_ready,                                                 \
      uint32_t* peer_ready,                                                \
      uint32_t* own_consumed,                                              \
      uint32_t* peer_consumed,                                             \
      uint32_t epoch,                                                      \
      void* stream) {                                                      \
    return enqueue_all_reduce(                                             \
        KERNEL,                                                            \
        local,                                                             \
        local_host,                                                        \
        peer_host,                                                         \
        output,                                                            \
        count,                                                             \
        own_ready,                                                         \
        peer_ready,                                                        \
        own_consumed,                                                      \
        peer_consumed,                                                     \
        epoch,                                                             \
        stream);                                                           \
  }

WAVMG_ALL_REDUCE_EXPORT(
    wavmg_all_reduce_f32_async, float, wavmg_add_f32_kernel)
WAVMG_ALL_REDUCE_EXPORT(
    wavmg_all_reduce_f16_async, __half, wavmg_add_f16_kernel)
WAVMG_ALL_REDUCE_EXPORT(
    wavmg_all_reduce_bf16_async, hip_bfloat16, wavmg_add_bf16_kernel)

WAVMG_EXPORT int wavmg_add_f32_async(
    const float* local,
    const float* peer_host,
    float* output,
    std::size_t count,
    void* stream) {
  return launch_add(
      wavmg_add_f32_kernel, local, peer_host, output, count, stream);
}

WAVMG_EXPORT int wavmg_add_f16_async(
    const __half* local,
    const __half* peer_host,
    __half* output,
    std::size_t count,
    void* stream) {
  return launch_add(
      wavmg_add_f16_kernel, local, peer_host, output, count, stream);
}

WAVMG_EXPORT int wavmg_add_bf16_async(
    const hip_bfloat16* local,
    const hip_bfloat16* peer_host,
    hip_bfloat16* output,
    std::size_t count,
    void* stream) {
  return launch_add(
      wavmg_add_bf16_kernel, local, peer_host, output, count, stream);
}

WAVMG_EXPORT int wavmg_copy_async(
    const void* source,
    void* destination,
    std::size_t size_bytes,
    void* stream_pointer) {
  if (size_bytes == 0) {
    return static_cast<int>(hipSuccess);
  }
  constexpr unsigned int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((size_bytes + threads - 1) / threads);
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  hipLaunchKernelGGL(
      wavmg_copy_bytes_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      stream,
      static_cast<const unsigned char*>(source),
      static_cast<unsigned char*>(destination),
      size_bytes);
  return static_cast<int>(hipGetLastError());
}

WAVMG_EXPORT int wavmg_add_f32(
    const float* local,
    const float* peer_host,
    float* output,
    std::size_t count) {
  const int error = wavmg_add_f32_async(
      local, peer_host, output, count, nullptr);
  if (error != static_cast<int>(hipSuccess)) {
    return error;
  }
  return static_cast<int>(hipDeviceSynchronize());
}
