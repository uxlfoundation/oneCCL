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

/* proxy.hpp
 *
 * Host-side proxy thread and service.
 */

#pragma once

#include "oneapi/ccl/types.hpp"
#include "comm/comm.hpp"

#include "proxy/proxy_types.hpp"
#include "proxy/proxy_staging.hpp"
#include "proxy/host_initiated/cmd_ring_types.hpp"
#include "proxy/host_initiated/cmd_ring_host.hpp"
#include "proxy/host_initiated/cmd_ring_context.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <thread>
#include <atomic>
#include <vector>
#include <queue>
#include <mutex>

class ccl_proxy_thread {
public:
    ccl_proxy_thread();
    ~ccl_proxy_thread();

    ccl_proxy_thread(const ccl_proxy_thread&) = delete;
    ccl_proxy_thread& operator=(const ccl_proxy_thread&) = delete;

    void start(ccl_cmd_ring_producer* producer,
               ccl_kernel_lifecycle* lifecycle,
               ccl_proxy_staging* staging,
               sycl::queue* queue);

    // Returns true if the thread was joined and everything it points at (ring,
    // staging, queue, and this object) is safe to free. Returns false if it did
    // not exit within the timeout: the thread is then detached and still
    // dereferencing all of the above, so the caller must leak them.
    bool stop();

    void submit(const ccl_proxy_op& op);

private:
    void work_loop();
    bool poll_host_op(ccl_proxy_op& out_op);

    ccl_cmd_ring_producer* m_producer{ nullptr };
    ccl_kernel_lifecycle* m_lifecycle{ nullptr };
    ccl_proxy_staging* m_staging{ nullptr };
    sycl::queue* m_queue{ nullptr };

    std::queue<ccl_proxy_op> m_host_op_queue;
    std::mutex m_host_queue_mutex;

    std::thread m_thread;
    std::atomic<bool> m_stop_flag{ false };
    // Set by work_loop as its last action. Lets stop() bound its wait without
    // pthread_timedjoin_np: observing it true means join() cannot block.
    std::atomic<bool> m_exited{ false };

    int m_cpu_affinity;
    int m_mem_affinity;
};

class ccl_proxy_service {
public:
    ccl_proxy_service();
    ~ccl_proxy_service();

    ccl_proxy_service(const ccl_proxy_service&) = delete;
    ccl_proxy_service& operator=(const ccl_proxy_service&) = delete;

    void init(sycl::queue& q);

    void start();
    void stop();
    void reset();

    void submit(const ccl_proxy_op& op);

    bool is_started() const noexcept {
        return m_started;
    }

    ccl_kernel_lifecycle* lifecycle() const {
        return m_cmd_ctx.lifecycle();
    }
    ccl_cmd_ring* ring() const {
        return m_cmd_ctx.ring();
    }
    ccl_cmd_ring_completions* completions() const {
        return m_cmd_ctx.completions();
    }
    ccl_proxy_staging& staging() {
        return m_staging;
    }

    // Assign the next monotonic generation for an op/kernel launch pair.
    // Single source of truth for the lifecycle handshake.
    uint32_t next_generation() {
        return m_cmd_ctx.next_generation();
    }

private:
    ccl_cmd_ring_context m_cmd_ctx;
    ccl_proxy_staging m_staging;
    sycl::queue* m_proxy_sycl_queue{ nullptr };

    std::unique_ptr<ccl_proxy_thread> m_proxy_thread;
    std::atomic<bool> m_started{ false };
};
