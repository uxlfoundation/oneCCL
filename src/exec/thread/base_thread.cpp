/*
 Copyright 2016-2020 Intel Corporation
 
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
#include "common/global/global.hpp"
#include "common/utils/yield.hpp"
#include "exec/thread/base_thread.hpp"

ccl::status ccl_base_thread::start(int cpu_affinity, int mem_affinity) {
    LOG_DEBUG(name(), " ", idx);

    start_cpu_affinity = cpu_affinity;
    start_mem_affinity = mem_affinity;

    /* start thread with initial CPU affinity */
    pthread_attr_t attr;
    int err = pthread_attr_init(&attr);
    CCL_THROW_IF_NOT(err == 0, "pthread_attr_init failed with error: ", err);

    cpu_set_t cpuset;
    __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);
    __CPU_SET_S(cpu_affinity, sizeof(cpu_set_t), &cpuset);
    err = pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
    CCL_THROW_IF_NOT(err == 0, "pthread_attr_setaffinity_np failed with error: ", err);

    err = pthread_create(&thread, &attr, thread_function, get_this());
    if (err) {
        LOG_ERROR(
            "error while creating ", name(), " thread #", idx, " pthread_create returns ", err);
        return ccl::status::runtime_error;
    }

    while (!started.load(std::memory_order_relaxed)) {
        ccl_yield(ccl::global_data::env().yield_type);
    }

    return ccl::status::success;
}

ccl::status ccl_base_thread::stop() {
    LOG_DEBUG(name(), " # ", idx);

    void* exit_code;
    int err;

    should_stop = true;

    if (ccl::global_data::env().worker_wait) {
        std::unique_lock<std::mutex> lock(wait.mtx);
        wait.var.notify_one();
    }

    while (started.load(std::memory_order_relaxed)) {
        ccl_yield(ccl::global_data::env().yield_type);
    }

    err = pthread_join(thread, &exit_code);
    if (err) {
        LOG_INFO("error while joining thread # ", idx, " , pthread_join returns ", err);
    }
    else {
        LOG_DEBUG("thread # ",
                  idx,
                  ", exited with code (",
                  (uintptr_t)exit_code,
                  (exit_code == PTHREAD_CANCELED) ? "PTHREAD_CANCELED" : "?",
                  ")");
    }

    return ccl::status::success;
}

ccl::status ccl_base_thread::set_cpu_affinity(int cpu_affinity) {
    /* unused, cpu affinity is set on thread start */

    LOG_DEBUG(name(), " # ", idx, ", CPU affinity ", cpu_affinity);

    int pthread_err;
    cpu_set_t cpuset;

    __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);
    __CPU_SET_S(cpu_affinity, sizeof(cpu_set_t), &cpuset);

    if ((pthread_err = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) != 0) {
        LOG_ERROR("pthread_setaffinity_np failed, err ", pthread_err);
        return ccl::status::runtime_error;
    }

    if (get_real_cpu_affinity() != cpu_affinity) {
        LOG_ERROR(name(), " ", idx, " is not pinned to CPU ", cpu_affinity);
        return ccl::status::runtime_error;
    }

    return ccl::status::success;
}

int ccl_base_thread::get_real_cpu_affinity() {
    int pthread_err;
    int result = CCL_UNDEFINED_CPU_ID;
    cpu_set_t cpuset;

    __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);

    if ((pthread_err = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) != 0) {
        LOG_ERROR("pthread_getaffinity_np failed, err ", pthread_err);
    }

    for (int cpu_idx = 0; cpu_idx < CPU_SETSIZE; cpu_idx++) {
        if (__CPU_ISSET_S(cpu_idx, sizeof(cpu_set_t), &cpuset)) {
            if (result == CCL_UNDEFINED_CPU_ID) {
                result = cpu_idx;
            }
            else {
                CCL_THROW("multiple affinity cores, previous ", result, ", new ", cpu_idx);
            }
        }
    }

    CCL_THROW_IF_NOT(result != CCL_UNDEFINED_CPU_ID, "can't retrieve CPU affinity");

    return result;
}
