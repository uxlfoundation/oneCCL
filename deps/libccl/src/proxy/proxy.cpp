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

/* proxy.cpp
 *
 * Host-side proxy thread implementation.
 */

#include "common/env/env.hpp"
#include "common/global/global.hpp"

#include "proxy/proxy.hpp"
#include "proxy/proxy_utils.hpp"
#include "proxy/proxy_func.hpp"

#include "proxy/host_initiated/cmd_ring_types.hpp"
#include "proxy/host_initiated/cmd_persistent_kernel.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <algorithm>

#include <cstdlib>
#include <new>

/* ======== Proxy thread implementation ======== */

ccl_proxy_thread::ccl_proxy_thread() = default;
ccl_proxy_thread::~ccl_proxy_thread() {
    // The owner (ccl_proxy_service::stop) stops the thread and needs stop()'s
    // return to decide what may be freed, so here the thread is already joined
    // and stop() is a no-op. Kept because ~std::thread on a joinable thread
    // terminates. A false return means we detached a live thread that will keep
    // reading this freed object — unpreventable here, so just report it.
    if (!stop()) {
        LOG_ERROR("[proxy] destroyed while its thread was still running and the thread "
                  "did not exit in time; it is now detached and will keep reading this "
                  "freed object. Owner must stop() first and leak on a false return.");
    }
}

void ccl_proxy_thread::start(ccl_cmd_ring_producer* producer,
                             ccl_kernel_lifecycle* lifecycle,
                             ccl_proxy_staging* staging,
                             sycl::queue* queue) {
    if (m_thread.joinable())
        return;

    auto& env = ccl::global_data::env();
    auto& global_data = ccl::global_data::get();

    size_t cpu_affinity = env.worker_affinity[global_data.get_local_proc_idx()];
    size_t mem_affinity = env.worker_mem_affinity[global_data.get_local_proc_idx()];

    m_cpu_affinity = static_cast<int>(cpu_affinity);
    m_mem_affinity = static_cast<int>(mem_affinity);

    m_producer = producer;
    m_lifecycle = lifecycle;
    m_staging = staging;
    m_queue = queue;
    m_stop_flag = false;
    m_exited = false;

    m_thread = std::thread(&ccl_proxy_thread::work_loop, this);
    set_thread_affinity(m_thread.native_handle(), m_cpu_affinity);
}

bool ccl_proxy_thread::stop() {
    if (!m_thread.joinable())
        return true;

    // Notify the thread to exit its work loop
    m_stop_flag.store(true, std::memory_order_release);

    // How long stop() waits for the proxy thread to leave an in-flight op before
    // giving up and detaching it. Not a tunable: at finalize no legitimate op can
    // still be in flight, so any wait longer than a generous slack means the thread
    // is stuck in one of the deadline-free waits (kernel-start handshake, ring
    // completion, ATL wait) and will never come back.
    constexpr auto proxy_stop_timeout = std::chrono::seconds(10);
    constexpr auto proxy_stop_poll_interval = std::chrono::milliseconds(1);
    // The flag is only tested at the top of work_loop, so a thread inside
    // process_host_initiated_op reaches no check until its op completes — and
    // none of the waits in there take a deadline. A plain join() would turn a
    // stuck proxy (kernel that never launched, dead peer) into a silent
    // ccl::finalize hang, so bound the wait and report instead.
    const auto deadline = std::chrono::steady_clock::now() + proxy_stop_timeout;
    while (!m_exited.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            LOG_ERROR("[proxy] thread did not exit within ",
                      std::chrono::duration_cast<std::chrono::seconds>(proxy_stop_timeout).count(),
                      " s; it is stuck in an in-flight op (kernel-start handshake, ring "
                      "completion, or ATL wait). Detaching and skipping teardown: leaking on "
                      "shutdown is preferable to freeing memory it is still using.");
            m_thread.detach();
            return false;
        }
        std::this_thread::sleep_for(proxy_stop_poll_interval);
    }

    // m_exited is set as work_loop's last action, so this cannot block.
    m_thread.join();
    return true;
}

void ccl_proxy_thread::submit(const ccl_proxy_op& op) {
    std::lock_guard<std::mutex> lock(m_host_queue_mutex);
    m_host_op_queue.push(op);
}

bool ccl_proxy_thread::poll_host_op(ccl_proxy_op& out_op) {
    std::lock_guard<std::mutex> lock(m_host_queue_mutex);
    if (m_host_op_queue.empty())
        return false;

    out_op = std::move(m_host_op_queue.front());
    m_host_op_queue.pop();
    return true;
}

void ccl_proxy_thread::work_loop() {
    // Report the resolved CPU/NUMA placement
    LOG_DEBUG("[proxy] thread: cpu: ",
              m_cpu_affinity,
              ", numa: ",
              ccl::global_data::get().hwloc_wrapper->get_numa_node(m_mem_affinity).to_string());
    // set mem affinity
    ccl::global_data::get().hwloc_wrapper->membind_thread(m_mem_affinity);
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::set_thread_name("ccl_proxy_thread");
#endif // CCL_ENABLE_ITT

    ccl_proxy_backoff backoff;
    while (!m_stop_flag.load(std::memory_order_relaxed)) {
        // poll host-initiated operations
        ccl_proxy_op host_op;
        if (poll_host_op(host_op)) {
            // The handler is what makes the stack unwind at all: without one,
            // terminate runs with the frames intact and ~ccl_kernel_lifecycle_guard
            // never submits the exit command that releases the resident work-groups.
            try {
                process_host_initiated_op(host_op, m_producer, m_lifecycle, m_staging, m_queue);
            }
            catch (const std::exception& e) {
                CCL_FATAL("[proxy] host-initiated op failed: ", e.what());
            }
            backoff.reset();
            continue;
        }

        // TODO: Poll device-initiated operations (lock-free ring buffer poll)
        // if (poll_device_op(device_op)) {
        //     process_device_initiated_op(device_op, ...);
        //     backoff.reset();
        //     continue;
        // }

        // No work available
        backoff.idle();
    }

    // Last action: tells stop() that join() cannot block
    m_exited.store(true, std::memory_order_release);
}

/* ======== Proxy service ======== */

ccl_proxy_service::ccl_proxy_service() = default;
ccl_proxy_service::~ccl_proxy_service() {
    // Runs from global_data::reset() at finalize. stop() frees device resources
    // through ZE_CALL, which throws on failure (e.g. device lost); escaping this
    // implicitly noexcept destructor would call std::terminate().
    // A failed teardown is only worth reporting: whatever it skipped is leaked.
    try {
        stop();
    }
    catch (const std::exception& e) {
        LOG_ERROR("[proxy] teardown failed: ", e.what());
    }
}

void ccl_proxy_service::init(sycl::queue& q) {
    // Initialize the proxy algorithm dispatch table
    // should be once per process
    init_proxy_algo_table();

    // Create an auxiliary in-order queue on the same device
    // for proxy operations and staging buffer management
    m_proxy_sycl_queue =
        new sycl::queue(q.get_context(), q.get_device(), sycl::property::queue::in_order{});

    m_cmd_ctx.init(*m_proxy_sycl_queue);
    m_staging.init(*m_proxy_sycl_queue);
}

void ccl_proxy_service::start() {
    if (m_started)
        return;

    m_proxy_thread = std::make_unique<ccl_proxy_thread>();
    m_proxy_thread->start(
        &m_cmd_ctx.producer(), m_cmd_ctx.lifecycle(), &m_staging, m_proxy_sycl_queue);
    m_started = true;
}

void ccl_proxy_service::stop() {
    if (!m_started)
        return;

    if (m_proxy_thread) {
        if (!m_proxy_thread->stop()) {
            // The thread was detached and still dereferences its producer,
            // staging, queue and its own members — so nothing here may be
            // freed: not the ring/completions/lifecycle, not the staging
            // context, not the queue, not the ccl_proxy_thread itself
            // (release(), not reset()). This is a deliberate shutdown leak; the
            // alternative is a host- and device-side use-after-free, and a
            // device-side one needs a GPU reset to clear.
            m_proxy_thread.release();
            m_started = false;
            return;
        }
        m_proxy_thread.reset();
    }
    // A joined proxy thread means no *host* side still touches the ring.
    // It is not proof that the device side is done.
    m_cmd_ctx.destroy(*m_proxy_sycl_queue);
    m_staging.destroy(*m_proxy_sycl_queue);

    delete m_proxy_sycl_queue;
    m_proxy_sycl_queue = nullptr;

    m_started = false;
}

void ccl_proxy_service::reset() {
    m_cmd_ctx.reset();
}

void ccl_proxy_service::submit(const ccl_proxy_op& op) {
    if (m_proxy_thread)
        m_proxy_thread->submit(op);
}
