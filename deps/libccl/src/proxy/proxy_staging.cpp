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

/* proxy_staging.cpp
 *
 * Staging memory management and related utilities for the proxy service.
 */

#include "common/env/env.hpp"
#include "common/global/global.hpp"

// Level Zero primitives
#include "sched/entry/ze/ze_primitives.hpp"

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"

#include "proxy/proxy.hpp"
#include "proxy/proxy_utils.hpp"
#include "proxy/proxy_func.hpp"

#include "proxy/host_initiated/cmd_ring_types.hpp"
#include "proxy/host_initiated/cmd_persistent_kernel.hpp"

#include <atomic>
#include <thread>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <chrono>

#include <cstdlib>
#include <new>

#include <immintrin.h>

/* ======== ccl_proxy_staging implementation ======== */

void ccl_proxy_staging::init(sycl::queue& q) {
    // Run configuration checks
    bool is_rdma_supported = transport_rdma_enabled();
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    if (ccl::ze::should_disable_rdma(ze_dev)) {
        is_rdma_supported = false;
    }
    m_is_rdma_supported = is_rdma_supported;
    // Note: currently supported only on PVC GPUs
    m_is_link_engine_supported = !ccl::ze::is_arc_card(ccl::ze::get_device_family(ze_dev));

    if (!(m_is_rdma_supported || m_is_link_engine_supported)) {
        LOG_WARN("Neither RDMA nor link copy engine is available, cannot proceed with proxy operations");
    }

    if (!m_is_rdma_supported && m_is_link_engine_supported) {
        // Once per proxy thread, after sycl::queue is created:
        ze_engine = ccl::ze::ze_find_copy_engines(q);
        ze_copy_context = ccl::ze::ze_copy_context_create(ze_engine);
    }
}

void ccl_proxy_staging::destroy(sycl::queue& q) {
    if (!m_is_rdma_supported && m_is_link_engine_supported) {
        ccl::ze::ze_copy_context_destroy(ze_copy_context);
    }
}

void ccl_proxy_staging::copy_d2h_and_wait(const void* dev_src, void* host_dst, size_t bytes) {
    CCL_THROW_IF_NOT(m_is_link_engine_supported,
                     "Link copy engine is not supported on this device");
    // Copy to host using link copy engine.
    ccl::ze::ze_submit_copy(
        ze_copy_context.d2h_cmdlist, host_dst, dev_src, bytes, ze_copy_context.d2h_ev[0]);
    // Wait for D2H copy to complete.
    ccl::ze::ze_wait_and_reset(ze_copy_context.d2h_ev[0]);
}

void ccl_proxy_staging::copy_h2d_and_wait(const void* host_src, void* dev_dst, size_t bytes) {
    CCL_THROW_IF_NOT(m_is_link_engine_supported,
                     "Link copy engine is not supported on this device");
    // Copy back to device using link copy engine.
    ccl::ze::ze_submit_copy(
        ze_copy_context.h2d_cmdlist, dev_dst, host_src, bytes, ze_copy_context.h2d_ev[0]);
    // Wait for H2D copy to complete.
    ccl::ze::ze_wait_and_reset(ze_copy_context.h2d_ev[0]);
}

// ==== Pipelined host-side sendrecv using Level Zero link copy engine (for non-RDMA PVC path) ====

void ccl_proxy_staging::do_sendrecv(ccl_proxy_staging* staging,
                                    const void* send_dev_ptr,
                                    size_t send_bytes,
                                    int send_to,
                                    void* recv_dev_ptr,
                                    size_t recv_bytes,
                                    int recv_from,
                                    size_t pipeline_chunk_size,
                                    ccl_comm* comm,
                                    int tag_base) {
    LOG_DEBUG("do_sendrecv: send_bytes=", send_bytes, ", recv_bytes=", recv_bytes, ", pipeline_chunk_size=", pipeline_chunk_size);
    CCL_THROW_IF_NOT(staging->m_is_link_engine_supported,
                     "Link copy engine is not supported on this device");

    // obtain transport atl comm
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    const int ep_idx = 0;

    ccl::ze::ze_staging_copy_context& zc = staging->ze_copy_context;

    const auto* send_dev = static_cast<const uint8_t*>(send_dev_ptr);
    auto* recv_dev = static_cast<uint8_t*>(recv_dev_ptr);
    const size_t total = std::max(send_bytes, recv_bytes);

    // Number of host staging buffers kept in flight (double-buffering).
    // must not exceed the communicator's allocated chunk buffers.
    constexpr int num_buffers_needed = ccl::ze::ze_staging_copy_context::pipeline_depth;
    const int num_chunk_buffs = ccl_scaleout_pipeline_bufs::num_chunk_buffs;
    CCL_THROW_IF_NOT(num_buffers_needed <= num_chunk_buffs,
                     "staging pipeline depth exceeds allocated chunk buffers");
    void** send_host_chunks = comm->get_scaleout_send_pipeline_bufs(num_chunk_buffs);
    void** recv_host_chunks = comm->get_scaleout_recv_pipeline_bufs(num_chunk_buffs);

    void* send_host[num_buffers_needed];
    void* recv_host[num_buffers_needed];
    for (int b = 0; b < num_buffers_needed; ++b) {
        send_host[b] = send_host_chunks[b];
        recv_host[b] = recv_host_chunks[b];
    }

    // "pending" means an async copy was submitted and may still be running
    // need to check event state before reusing the buffer for the next copy
    bool d2h_pending[num_buffers_needed] = {};
    bool h2d_pending[num_buffers_needed] = {};

    // Define helper lambdas for launching and waiting on copies. This keeps the copy/wait logic organized.
    // Copy API can be anything, as long as it supports async copy and separate events for synchronization
    auto launch_d2h = [&](int b, size_t off, size_t bytes) {
        if (bytes == 0) {
            d2h_pending[b] = false;
            return;
        }
        ccl::ze::ze_submit_copy(zc.d2h_cmdlist, send_host[b], send_dev + off, bytes, zc.d2h_ev[b]);
        d2h_pending[b] = true;
    };
    auto launch_h2d = [&](int b, size_t off, size_t bytes) {
        if (bytes == 0) {
            h2d_pending[b] = false;
            return;
        }
        ccl::ze::ze_submit_copy(zc.h2d_cmdlist, recv_dev + off, recv_host[b], bytes, zc.h2d_ev[b]);
        h2d_pending[b] = true;
    };
    auto wait_d2h = [&](int b) {
        if (d2h_pending[b]) {
            ccl::ze::ze_wait_and_reset(zc.d2h_ev[b]);
            d2h_pending[b] = false;
        }
    };
    auto wait_h2d = [&](int b) {
        if (h2d_pending[b]) {
            ccl::ze::ze_wait_and_reset(zc.h2d_ev[b]);
            h2d_pending[b] = false;
        }
    };

    // Prime D2H for slice 0 into buffer 0.
    if (total > 0) {
        const size_t max0 = std::min(pipeline_chunk_size, total);
        const size_t send0 = (0 < send_bytes) ? std::min(max0, send_bytes) : 0;
        launch_d2h(0, 0, send0);
    }

    size_t offset = 0;
    int slice_idx = 0;

    // same as pipe_sendrecv_plain, but without sycl tasks
    // launch_d2h/wait_d2h can be used with other copy engines (not only link copy engines)
    // recv0   send0   |  send1     recv1   |  send2     recv2 | send3   recv3
    // s_copy0 s_copy1 |  s_copy2   r_copy0 |  s_copy3 r_copy1 | r_copy2 r_copy3
    while (offset < total) {
        const int b = slice_idx % num_buffers_needed;

        const size_t max_cur = std::min(pipeline_chunk_size, total - offset);

        const size_t send_cur = (offset < send_bytes) ? std::min(max_cur, send_bytes - offset) : 0;
        const size_t recv_cur = (offset < recv_bytes) ? std::min(max_cur, recv_bytes - offset) : 0;

        const int tag = tag_base + slice_idx;

        atl_req_t send_req, recv_req;
        int nreq = 0;

        // Only wait on H2D for buffer b if we are about to reuse recv_host[b] for a new recv.
        // This enables overlap of H2D(previous) with ATL(current) on the other buffer.
        if (recv_cur > 0) {
            wait_h2d(b);
            ATL_CALL_THROW_IF_ERROR(atl_comm->recv(
                ep_idx, recv_host[b], static_cast<int>(recv_cur), recv_from, tag, recv_req));
            nreq++;
        }

        // Wait for D2H(slice) before send.
        if (send_cur > 0) {
            wait_d2h(b);
            ATL_CALL_THROW_IF_ERROR(atl_comm->send(
                ep_idx, send_host[b], static_cast<int>(send_cur), send_to, tag, send_req));
            nreq++;
        }

        // Launch D2H for the next slice into the other buffer.
        const size_t next_off = offset + max_cur;
        if (next_off < total) {
            const int nb = (slice_idx + 1) % num_buffers_needed;
            const size_t next_max = std::min(pipeline_chunk_size, total - next_off);
            const size_t next_send =
                (next_off < send_bytes) ? std::min(next_max, send_bytes - next_off) : 0;

            wait_d2h(nb);
            launch_d2h(nb, next_off, next_send);
        }

        if (nreq > 0) {
            if (recv_cur > 0)
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            if (send_cur > 0)
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
        }

        // Launch H2D of received slice (async) and DO NOT wait here.
        launch_h2d(b, offset, recv_cur);

        offset += max_cur;
        ++slice_idx;
    }

    // Ensure last H2D copies are complete before returning.
    for (int b = 0; b < num_buffers_needed; ++b) {
        wait_h2d(b);
    }
}