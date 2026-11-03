/*
 Copyright 2016-2026 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

/* proxy_staging.hpp
 *
 * Staging memory management and related utilities for the proxy service.
 */

#pragma once

#include "oneapi/ccl/types.hpp"
#include "comm/comm.hpp"

#include "proxy/proxy_types.hpp"
#include "proxy/host_initiated/cmd_ring_types.hpp"
#include "proxy/host_initiated/cmd_ring_host.hpp"
#include "proxy/host_initiated/cmd_ring_context.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <thread>
#include <atomic>
#include <vector>

/* ======== Staging utils for non RDMA path ======== */

struct ccl_proxy_staging {
    // configuration flags
    bool m_is_rdma_supported = false;
    bool m_is_link_engine_supported = false;

    // copy engines available on the device
    ccl::ze::ze_staging_copy_engines ze_engine;
    // context for staging buffer management and copy submission
    ccl::ze::ze_staging_copy_context ze_copy_context;

    void init(sycl::queue& q);
    void destroy(sycl::queue& q);

    void copy_d2h_and_wait(const void* dev_src, void* host_dst, size_t bytes);
    void copy_h2d_and_wait(const void* host_src, void* dev_dst, size_t bytes);

    static void do_sendrecv(ccl_proxy_staging* staging,
                            const void* send_dev_ptr,
                            size_t send_bytes,
                            int send_to,
                            void* recv_dev_ptr,
                            size_t recv_bytes,
                            int recv_from,
                            size_t pipeline_chunk_size,
                            ccl_comm* comm,
                            int tag_base);
};