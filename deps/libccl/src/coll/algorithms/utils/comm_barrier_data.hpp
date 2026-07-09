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

#pragma once

#include <array>
#include <sycl/sycl.hpp>

#include "common/global/global.hpp"
#include "common/stream/stream.hpp"
#include "common/utils/utils.hpp"
#include "coll/algorithms/utils/consts.hpp"

inline bool use_recording_path(const sycl::queue &q) {
    return ccl::global_data::env().sycl_force_recording_path ||
           q.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording;
}

inline bool use_recording_path(const ccl_stream *stream) {
    if (stream) {
        return use_recording_path(stream->get_native_stream());
    }
    if (ccl::global_data::env().sycl_force_recording_path) {
        // LOG_WARN("trying to force recording on null stream; falling back to non-recording");
        CCL_THROW("recording null stream"); // TODO WARN or THROW ?
    }
    return false;
}

class ccl_kernel_barrier_data {
public:
    static constexpr int slots = 3;

    ccl_kernel_barrier_data() = default;
    ccl_kernel_barrier_data(const ccl_kernel_barrier_data &) = default;
    ccl_kernel_barrier_data &operator=(const ccl_kernel_barrier_data &) = default;

    void set_sync_ptrs(size_t *ptrs) {
        m_sync_ptrs = ptrs;
    }

    // advance to the next slot
    ccl_kernel_barrier_data inc_slot(int count = 1) {
        m_count += count;
        return *this;
    }

    // get current slot pointer
    size_t *get_sync_ptr() const {
        size_t curr_slot = m_count % slots;
        return m_sync_ptrs + curr_slot;
    }

    // reset data in the farthest slot
    void reset_sync_data() {
        size_t reset_slot = (m_count + (slots - 1)) % slots;
        m_sync_ptrs[reset_slot] = 0;
    }

private:
    size_t *m_sync_ptrs;
    size_t m_count = 0;
};

class alignas(CACHELINE_SIZE) ccl_comm_barrier_data {
private:
    int m_rank;
    int m_size;
    size_t m_count = slots - 1;
    size_t *d_count = nullptr;
    bool m_is_set = false;
    bool m_use_remote_atomics;
    std::array<size_t *, MAX_NODE_RANKS> m_remote_ptrs{};
    std::array<size_t *, MAX_NODE_RANKS> d_remote_ptrs{};

public:
    static constexpr int slots = 3;

    // setting use_remote_atomics is postponed in coll_init
    // until topo manager is created.
    ccl_comm_barrier_data(int rank, int size)
            : m_rank(rank),
              m_size(size),
              m_use_remote_atomics(false) {}

    int rank() const {
        return m_rank;
    }

    int size() const {
        return m_size;
    }

    bool is_set() const {
        return m_is_set;
    }

    bool use_remote_atomics() const {
        return m_use_remote_atomics;
    }

    void set_remote_atomics(bool use_remote_atomics) {
        m_use_remote_atomics = use_remote_atomics;
    }

    size_t inc(size_t n) {
        m_count += n;
        return m_count;
    }

    size_t count() const {
        return m_count / slots;
    }

    int slot() const {
        return m_count % slots;
    }

    std::array<size_t *, MAX_NODE_RANKS> remote_ptrs() const {
        return m_remote_ptrs;
    }

    size_t inc_gpu(size_t n) {
        *d_count = *d_count + n;
        return *d_count;
    }

    size_t count_gpu() const {
        return *d_count / slots;
    }

    int slot_gpu() const {
        return *d_count % slots;
    }

    std::array<size_t *, MAX_NODE_RANKS> remote_ptrs_gpu() const {
        return d_remote_ptrs;
    }

    void set_count_gpu(size_t *count) {
        d_count = count;
    }

    void set_remote_ptrs(std::array<size_t *, MAX_NODE_RANKS> ptrs,
                         std::array<size_t *, MAX_NODE_RANKS> gpu_barrier_ptrs) {
        assert(!is_set());
        m_is_set = true;
        m_remote_ptrs = ptrs;
        d_remote_ptrs = gpu_barrier_ptrs;
    }
};

class alignas(CACHELINE_SIZE) ccl_comm_flag_data {
private:
    int m_rank;
    int m_size;
    size_t *d_count = nullptr;
    std::array<size_t *, MAX_NODE_RANKS> d_remote_ptrs{};

public:
    static constexpr int slots = 4;

    ccl_comm_flag_data(int rank, int size) : m_rank(rank), m_size(size) {}

    int rank() const {
        return m_rank;
    }

    int size() const {
        return m_size;
    }

    size_t inc(size_t n) {
        *d_count = *d_count + n;
        return *d_count;
    }

    size_t count() const {
        return *d_count / slots;
    }

    int slot() const {
        return *d_count % slots;
    }

    std::array<size_t *, MAX_NODE_RANKS> remote_ptrs() const {
        return d_remote_ptrs;
    }

    void set_count(size_t *count) {
        d_count = count;
    }

    void set_remote_ptrs(std::array<size_t *, MAX_NODE_RANKS> ptrs) {
        d_remote_ptrs = ptrs;
    }
};
