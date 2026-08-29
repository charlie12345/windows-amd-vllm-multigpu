// SPDX-FileCopyrightText: 2026 Carlo Pasquale (https://github.com/Charlie12345)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>
#include <string>

enum class wac_data_type {
    f16,
    f32,
    bf16,
};

struct wac_rank_call {
    const void * send = nullptr;
    void * recv = nullptr;
    std::size_t count = 0;
    int device = -1;
    hipStream_t stream = nullptr;
};

class wac_d3d12_transport;

struct wac_d3d12_deleter {
    void operator()(wac_d3d12_transport * transport) const;
};

using wac_d3d12_transport_ptr = std::unique_ptr<wac_d3d12_transport, wac_d3d12_deleter>;

wac_d3d12_transport_ptr wac_d3d12_create(const int devices[2], std::size_t max_size_bytes,
                                         std::string & error);

bool wac_d3d12_allreduce(wac_d3d12_transport & transport, const wac_rank_call calls[2],
                         wac_data_type type, std::string & error);
