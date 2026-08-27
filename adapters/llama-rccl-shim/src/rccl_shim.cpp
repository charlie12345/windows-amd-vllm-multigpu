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

#include <nccl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint64_t comm_magic = 0x574143434f4d4d31ULL;

enum class transport_mode {
    hybrid,
    d3d12,
    rccl,
};

enum class rccl_dispatch_mode {
    threaded,
    native_group,
};

struct real_rccl_api {
    HMODULE module = nullptr;
    ncclResult_t (*comm_init_all)(ncclComm_t *, int, const int *) = nullptr;
    ncclResult_t (*comm_destroy)(ncclComm_t) = nullptr;
    const char * (*get_error_string)(ncclResult_t) = nullptr;
    ncclResult_t (*group_start)() = nullptr;
    ncclResult_t (*group_end)() = nullptr;
    ncclResult_t (*all_reduce)(const void *, void *, std::size_t, ncclDataType_t, ncclRedOp_t,
                               ncclComm_t, hipStream_t) = nullptr;
    std::string error;
};

real_rccl_api g_real;
std::once_flag g_real_once;

std::wstring module_directory() {
    HMODULE module = nullptr;
    const auto address = reinterpret_cast<LPCWSTR>(&module_directory);
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            address, &module)) {
        return L".";
    }

    std::array<wchar_t, 32768> path{};
    const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length == path.size()) {
        return L".";
    }
    std::wstring result(path.data(), length);
    const std::size_t separator = result.find_last_of(L"\\/");
    return separator == std::wstring::npos ? L"." : result.substr(0, separator);
}

template <typename T>
bool load_symbol(HMODULE module, const char * name, T & destination, std::string & error) {
    destination = reinterpret_cast<T>(GetProcAddress(module, name));
    if (destination != nullptr) {
        return true;
    }
    error = std::string("rccl-real.dll does not export ") + name;
    return false;
}

void load_real_rccl_once() {
    const std::wstring path = module_directory() + L"\\rccl-real.dll";
    g_real.module = LoadLibraryW(path.c_str());
    if (g_real.module == nullptr) {
        g_real.error = "Could not load rccl-real.dll beside the plugin";
        return;
    }

    if (!load_symbol(g_real.module, "ncclCommInitAll", g_real.comm_init_all, g_real.error) ||
        !load_symbol(g_real.module, "ncclCommDestroy", g_real.comm_destroy, g_real.error) ||
        !load_symbol(g_real.module, "ncclGetErrorString", g_real.get_error_string, g_real.error) ||
        !load_symbol(g_real.module, "ncclGroupStart", g_real.group_start, g_real.error) ||
        !load_symbol(g_real.module, "ncclGroupEnd", g_real.group_end, g_real.error) ||
        !load_symbol(g_real.module, "ncclAllReduce", g_real.all_reduce, g_real.error)) {
        FreeLibrary(g_real.module);
        g_real.module = nullptr;
    }
}

real_rccl_api & real_rccl() {
    std::call_once(g_real_once, load_real_rccl_once);
    return g_real;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

transport_mode configured_transport_mode() {
    const char * value = std::getenv("WAC_MODE");
    if (value == nullptr) {
        return transport_mode::hybrid;
    }
    const std::string mode = lowercase(value);
    if (mode == "d3d12") {
        return transport_mode::d3d12;
    }
    if (mode == "rccl") {
        return transport_mode::rccl;
    }
    return transport_mode::hybrid;
}

rccl_dispatch_mode configured_dispatch_mode() {
    const char * value = std::getenv("WAC_RCCL_DISPATCH");
    if (value != nullptr && lowercase(value) == "native-group") {
        return rccl_dispatch_mode::native_group;
    }
    return rccl_dispatch_mode::threaded;
}

std::size_t configured_size(const char * name, std::size_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed > std::numeric_limits<std::size_t>::max()) {
        return fallback;
    }
    return static_cast<std::size_t>(parsed);
}

struct clique_context {
    std::vector<int> devices;
    wac_d3d12_transport_ptr d3d12;
    transport_mode mode = transport_mode::hybrid;
    rccl_dispatch_mode rccl_dispatch = rccl_dispatch_mode::threaded;
    std::size_t d3d12_buffer_bytes = 64ULL * 1024 * 1024;
    std::size_t d3d12_route_min_bytes = 0;
    std::size_t d3d12_route_max_bytes = 32ULL * 1024 - 1;
    std::atomic<bool> logged_d3d12{false};
    std::atomic<bool> logged_rccl{false};
};

struct plugin_comm {
    std::uint64_t magic = comm_magic;
    std::shared_ptr<clique_context> clique;
    ncclComm_t real_comm = nullptr;
    int rank = -1;
};

plugin_comm * as_plugin_comm(ncclComm_t comm) {
    auto * wrapper = reinterpret_cast<plugin_comm *>(comm);
    return wrapper != nullptr && wrapper->magic == comm_magic ? wrapper : nullptr;
}

struct pending_allreduce {
    const void * send = nullptr;
    void * recv = nullptr;
    std::size_t count = 0;
    ncclDataType_t datatype = ncclFloat32;
    ncclRedOp_t op = ncclSum;
    plugin_comm * comm = nullptr;
    hipStream_t stream = nullptr;
};

struct group_state {
    int depth = 0;
    ncclResult_t error = ncclSuccess;
    std::vector<pending_allreduce> calls;
};

thread_local group_state g_group;
thread_local std::string g_last_error;

const char * local_error_string(ncclResult_t result) {
    switch (result) {
    case ncclSuccess: return "no error";
    case ncclUnhandledCudaError: return "unhandled HIP error";
    case ncclSystemError: return "system error";
    case ncclInternalError: return "internal error";
    case ncclInvalidArgument: return "invalid argument";
    case ncclInvalidUsage: return "invalid usage";
    case ncclRemoteError: return "remote error";
    case ncclInProgress: return "in progress";
    case ncclTimeout: return "timeout";
    default: return "unknown RCCL error";
    }
}

bool datatype_info(ncclDataType_t datatype, std::size_t & element_size, wac_data_type & type) {
    switch (datatype) {
    case ncclFloat16:
        element_size = 2;
        type = wac_data_type::f16;
        return true;
    case ncclFloat32:
        element_size = 4;
        type = wac_data_type::f32;
        return true;
    case ncclBfloat16:
        element_size = 2;
        type = wac_data_type::bf16;
        return true;
    default: return false;
    }
}

ncclResult_t dispatch_real_native(const std::vector<pending_allreduce> & calls) {
    real_rccl_api & real = real_rccl();
    if (real.module == nullptr) {
        g_last_error = real.error;
        return ncclSystemError;
    }

    ncclResult_t result = real.group_start();
    if (result != ncclSuccess) {
        return result;
    }
    ncclResult_t operation_result = ncclSuccess;
    for (const pending_allreduce & call : calls) {
        if (call.comm->real_comm == nullptr) {
            operation_result = ncclInvalidUsage;
            break;
        }
        result = real.all_reduce(call.send, call.recv, call.count, call.datatype, call.op,
                                 call.comm->real_comm, call.stream);
        if (result != ncclSuccess) {
            operation_result = result;
            break;
        }
    }
    const ncclResult_t end_result = real.group_end();
    return operation_result != ncclSuccess ? operation_result : end_result;
}

ncclResult_t dispatch_real_threaded(const std::array<pending_allreduce, 2> & calls) {
    real_rccl_api & real = real_rccl();
    if (real.module == nullptr) {
        g_last_error = real.error;
        return ncclSystemError;
    }
    if (calls[0].comm->real_comm == nullptr || calls[1].comm->real_comm == nullptr) {
        return ncclInvalidUsage;
    }

    std::array<ncclResult_t, 2> results = {ncclSuccess, ncclSuccess};
    std::mutex mutex;
    std::condition_variable condition;
    int ready = 0;
    bool go = false;
    std::array<std::thread, 2> workers;

    for (std::size_t rank = 0; rank < workers.size(); ++rank) {
        workers[rank] = std::thread([&, rank]() {
            {
                std::unique_lock<std::mutex> lock(mutex);
                ++ready;
                if (ready == 2) {
                    go = true;
                    condition.notify_all();
                } else {
                    condition.wait(lock, [&]() { return go; });
                }
            }
            (void)hipSetDevice(calls[rank].comm->clique->devices[calls[rank].comm->rank]);
            results[rank] = real.all_reduce(calls[rank].send, calls[rank].recv, calls[rank].count,
                                            calls[rank].datatype, calls[rank].op,
                                            calls[rank].comm->real_comm, calls[rank].stream);
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }
    return results[0] != ncclSuccess ? results[0] : results[1];
}

ncclResult_t dispatch_group() {
    if (g_group.error != ncclSuccess) {
        return g_group.error;
    }
    if (g_group.calls.empty()) {
        return ncclSuccess;
    }

    if (g_group.calls.size() != 2) {
        return dispatch_real_native(g_group.calls);
    }

    std::array<pending_allreduce, 2> calls = {g_group.calls[0], g_group.calls[1]};
    if (calls[0].comm->clique != calls[1].comm->clique ||
        calls[0].comm->rank == calls[1].comm->rank) {
        return dispatch_real_native(g_group.calls);
    }
    if (calls[0].comm->rank > calls[1].comm->rank) {
        std::swap(calls[0], calls[1]);
    }

    const std::shared_ptr<clique_context> & clique = calls[0].comm->clique;
    std::size_t element_size = 0;
    wac_data_type type = wac_data_type::f32;
    const bool supported_type = datatype_info(calls[0].datatype, element_size, type);
    const bool matching = calls[0].count == calls[1].count &&
                          calls[0].datatype == calls[1].datatype && calls[0].op == ncclSum &&
                          calls[1].op == ncclSum;
    const bool size_valid =
            element_size != 0 &&
            calls[0].count <= std::numeric_limits<std::size_t>::max() / element_size;
    const std::size_t bytes = size_valid ? calls[0].count * element_size : 0;
    const bool in_hybrid_range =
            bytes >= clique->d3d12_route_min_bytes && bytes <= clique->d3d12_route_max_bytes;
    const bool use_d3d12 = clique->d3d12 != nullptr && supported_type && matching && size_valid &&
                           bytes <= clique->d3d12_buffer_bytes &&
                           (clique->mode == transport_mode::d3d12 || in_hybrid_range);

    if (use_d3d12) {
        if (!clique->logged_d3d12.exchange(true)) {
            std::fprintf(stderr, "wavmg-llama-rccl-shim: route=D3D12 payload=%zu bytes\n", bytes);
        }
        const std::array<wac_rank_call, 2> d3d12_calls = {{
                {calls[0].send, calls[0].recv, calls[0].count, clique->devices[0], calls[0].stream},
                {calls[1].send, calls[1].recv, calls[1].count, clique->devices[1], calls[1].stream},
        }};
        std::string error;
        if (!wac_d3d12_allreduce(*clique->d3d12, d3d12_calls.data(), type, error)) {
            g_last_error = error;
            return ncclUnhandledCudaError;
        }
        return ncclSuccess;
    }

    if (!clique->logged_rccl.exchange(true)) {
        std::fprintf(stderr,
                     "wavmg-llama-rccl-shim: route=Direct-RCCL payload=%zu bytes dispatch=%s\n",
                     bytes,
                     clique->rccl_dispatch == rccl_dispatch_mode::threaded ? "threaded"
                                                                           : "native-group");
    }
    if (clique->rccl_dispatch == rccl_dispatch_mode::native_group) {
        return dispatch_real_native(g_group.calls);
    }
    return dispatch_real_threaded(calls);
}

} // namespace

#define WAC_EXPORT extern "C"

WAC_EXPORT ncclResult_t ncclCommInitAll(ncclComm_t * comms, int ndev, const int * devlist) {
    if (comms == nullptr || ndev <= 0) {
        return ncclInvalidArgument;
    }

    std::vector<int> devices(static_cast<std::size_t>(ndev));
    for (int rank = 0; rank < ndev; ++rank) {
        devices[rank] = devlist == nullptr ? rank : devlist[rank];
        comms[rank] = nullptr;
    }

    auto clique = std::make_shared<clique_context>();
    clique->devices = devices;
    clique->mode = configured_transport_mode();
    clique->rccl_dispatch = configured_dispatch_mode();
    clique->d3d12_buffer_bytes = configured_size("WAC_D3D12_BUFFER_BYTES", 64ULL * 1024 * 1024);
    clique->d3d12_route_min_bytes = configured_size("WAC_D3D12_MIN_BYTES", 0);
    clique->d3d12_route_max_bytes = configured_size("WAC_D3D12_MAX_BYTES", 32ULL * 1024 - 1);
    if (clique->d3d12_route_max_bytes > clique->d3d12_buffer_bytes) {
        clique->d3d12_route_max_bytes = clique->d3d12_buffer_bytes;
    }

    std::vector<ncclComm_t> real_comms(static_cast<std::size_t>(ndev), nullptr);
    ncclResult_t real_result = ncclSystemError;
    if (clique->mode != transport_mode::d3d12) {
        real_rccl_api & real = real_rccl();
        if (real.module != nullptr) {
            real_result = real.comm_init_all(real_comms.data(), ndev, devices.data());
        } else {
            g_last_error = real.error;
        }
    }

    std::string d3d12_error;
    if (clique->mode != transport_mode::rccl && ndev == 2) {
        const int pair[2] = {devices[0], devices[1]};
        clique->d3d12 = wac_d3d12_create(pair, clique->d3d12_buffer_bytes, d3d12_error);
    }

    const bool real_ready = real_result == ncclSuccess;
    const bool d3d12_ready = clique->d3d12 != nullptr;
    if (clique->mode == transport_mode::rccl && !real_ready) {
        return real_result;
    }
    if (clique->mode == transport_mode::d3d12 && !d3d12_ready) {
        g_last_error = d3d12_error;
        return ncclSystemError;
    }
    if (!real_ready && !d3d12_ready) {
        if (!d3d12_error.empty()) {
            g_last_error = d3d12_error;
        }
        return real_result;
    }

    std::vector<std::unique_ptr<plugin_comm>> wrappers;
    try {
        wrappers.reserve(static_cast<std::size_t>(ndev));
        for (int rank = 0; rank < ndev; ++rank) {
            auto wrapper = std::make_unique<plugin_comm>();
            wrapper->clique = clique;
            wrapper->rank = rank;
            wrapper->real_comm = real_ready ? real_comms[rank] : nullptr;
            wrappers.push_back(std::move(wrapper));
        }
        for (int rank = 0; rank < ndev; ++rank) {
            comms[rank] = reinterpret_cast<ncclComm_t>(wrappers[rank].release());
        }
    } catch (...) {
        real_rccl_api & real = real_rccl();
        if (real.module != nullptr) {
            for (ncclComm_t real_comm : real_comms) {
                if (real_comm != nullptr) {
                    (void)real.comm_destroy(real_comm);
                }
            }
        }
        return ncclSystemError;
    }

    std::fprintf(stderr, "wavmg-llama-rccl-shim: initialized ranks=%d D3D12=%s Direct-RCCL=%s\n",
                 ndev, d3d12_ready ? "ready" : "off", real_ready ? "ready" : "off");
    return ncclSuccess;
}

WAC_EXPORT ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    plugin_comm * wrapper = as_plugin_comm(comm);
    if (wrapper == nullptr) {
        return ncclInvalidArgument;
    }

    ncclResult_t result = ncclSuccess;
    if (wrapper->real_comm != nullptr) {
        real_rccl_api & real = real_rccl();
        if (real.module == nullptr) {
            result = ncclSystemError;
        } else {
            result = real.comm_destroy(wrapper->real_comm);
        }
    }
    wrapper->magic = 0;
    delete wrapper;
    return result;
}

WAC_EXPORT const char * ncclGetErrorString(ncclResult_t result) {
    if (!g_last_error.empty()) {
        return g_last_error.c_str();
    }
    real_rccl_api & real = real_rccl();
    return real.module != nullptr ? real.get_error_string(result) : local_error_string(result);
}

WAC_EXPORT ncclResult_t ncclGroupStart() {
    if (g_group.depth == 0) {
        g_group.calls.clear();
        g_group.error = ncclSuccess;
        g_last_error.clear();
    }
    ++g_group.depth;
    return ncclSuccess;
}

WAC_EXPORT ncclResult_t ncclAllReduce(const void * sendbuff, void * recvbuff, std::size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                      hipStream_t stream) {
    plugin_comm * wrapper = as_plugin_comm(comm);
    if (wrapper == nullptr || sendbuff == nullptr || recvbuff == nullptr) {
        return ncclInvalidArgument;
    }

    if (g_group.depth > 0) {
        g_group.calls.push_back({sendbuff, recvbuff, count, datatype, op, wrapper, stream});
        return ncclSuccess;
    }

    if (wrapper->real_comm == nullptr) {
        g_last_error = "D3D12 interception requires paired ncclGroupStart/ncclGroupEnd calls";
        return ncclInvalidUsage;
    }
    real_rccl_api & real = real_rccl();
    if (real.module == nullptr) {
        g_last_error = real.error;
        return ncclSystemError;
    }
    return real.all_reduce(sendbuff, recvbuff, count, datatype, op, wrapper->real_comm, stream);
}

WAC_EXPORT ncclResult_t ncclGroupEnd() {
    if (g_group.depth <= 0) {
        return ncclInvalidUsage;
    }
    --g_group.depth;
    if (g_group.depth > 0) {
        return ncclSuccess;
    }

    const ncclResult_t result = dispatch_group();
    g_group.calls.clear();
    g_group.error = ncclSuccess;
    return result;
}
