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

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <array>

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

#include "atl/atl_base_comm.hpp"
#include "topology/topo_manager.hpp"

namespace ccl {

static std::atomic<int> next_op_id{ 0 };

// === Handshake data for non-group operations ===
struct handshake_data {
    std::mutex m;
    std::condition_variable cv;

    // Indicate that the receiver has published its pointer
    bool recv_published = false;

    // Indicate that the sender has finished its memcpy
    bool copy_done = false;
};

struct op_id_entry {
    std::mutex m;
    std::condition_variable cv;
    int id = -1;
    bool ready = false;
};

// === Simplified group buffer management ===
// this entry is unique for each pair of ranks
struct group_buffer_entry {
    void *buffer_ptr = nullptr;
    bool group_discovery_phase{ true };
    std::atomic<int> copy_submitted{ 0 }, reg_done{ 0 }; //signaling counters
    int copy_counter = 1, reg_counter = 1; // checking counters
#ifdef CCL_ENABLE_SYCL
    sycl::event recv_ready, copy_event;
#endif // CCL_ENABLE_SYCL
};

class shared_resources {
private:
    std::unordered_map<int, std::mutex> resource_mutexes;
    std::unordered_map<int, int> barrier_initialized_flags;

public:
    std::unordered_map<int, rank_info_vec_t> rank_info_vec_globs;
    std::unordered_map<int, std::vector<char>> all_hostnames_raw_globs;
    std::unordered_map<int, std::unordered_map<int, std::vector<void *>>> hash_table;
    std::unordered_map<int, std::unordered_map<int, std::vector<void *>>> pt2pt_hash_table;
#ifdef CCL_ENABLE_SYCL
    sycl::event copy_event;
#endif // CCL_ENABLE_SYCL
    std::unordered_map<int, handshake_data> handshakes;
    std::unordered_map<uint64_t, op_id_entry> op_id_map;
#ifdef CCL_ENABLE_SYCL
    std::unordered_map<int, sycl::event> receiver_ready_events_map;
#endif // CCL_ENABLE_SYCL
    // === Simplified group buffer management ===
    std::vector<std::vector<std::vector<group_buffer_entry>>> group_buffers;
    std::mutex allocation_mutex;
    std::mutex barrier_waits_mutex;
    std::mutex rank_info_mutex;
    std::mutex hostnames_mutex;
#ifdef CCL_ENABLE_SYCL
    std::mutex events_mutex;
#endif

    bool is_multi_thread_instance = false;
    // int current_global_id = ccl_comm::invalid_id;
    std::unordered_map<int, std::unique_ptr<pthread_barrier_t>> barrier_waits;

    shared_resources() = default;
    shared_resources(const shared_resources &) = delete;
    shared_resources &operator=(const shared_resources &) = delete;

    ~shared_resources() {
        hash_table.clear();
        pt2pt_hash_table.clear();
        barrier_waits.clear();
        rank_info_vec_globs.clear();
        barrier_initialized_flags.clear();
        resource_mutexes.clear();
        all_hostnames_raw_globs.clear();
    }
#ifdef CCL_ENABLE_SYCL
    void set_receiver_ready_event(int op_id, const sycl::event &evt) {
        std::lock_guard<std::mutex> lk(events_mutex);
        receiver_ready_events_map[op_id] = evt;
    }

    sycl::event get_receiver_ready_event(int op_id) {
        std::lock_guard<std::mutex> lk(events_mutex);
        return receiver_ready_events_map.at(op_id);
    }
#endif // CCL_ENABLE_SYCL
    void resize_rank_info_vec_glob(size_t new_size, int global_id) {
        std::lock_guard<std::mutex> lock(rank_info_mutex);
        rank_info_vec_globs[global_id].resize(new_size);
    }

    void resize_all_hostnames_raw(size_t new_size, int global_id) {
        std::lock_guard<std::mutex> lock(hostnames_mutex);
        all_hostnames_raw_globs[global_id].resize(new_size);
    }

    std::mutex &get_resource_mutex(int global_id) {
        static std::mutex mu;
        std::lock_guard<std::mutex> guard(mu);
        return resource_mutexes[global_id];
    }

    std::vector<char> &get_all_hostnames_raw_glob(int global_id) {
        std::lock_guard<std::mutex> lock(hostnames_mutex);
        return all_hostnames_raw_globs[global_id];
    }

    rank_info_vec_t &get_rank_info_vec_glob(int global_id) {
        std::lock_guard<std::mutex> lock(rank_info_mutex);
        return rank_info_vec_globs[global_id];
    }

    void init_barrier_wait(int threads_num, int global_id) {
        pthread_barrier_t *barrier_ptr = nullptr;
        {
            std::lock_guard<std::mutex> guard(barrier_waits_mutex);
            if (barrier_initialized_flags.find(global_id) == barrier_initialized_flags.end() ||
                !barrier_initialized_flags[global_id]) {
                barrier_waits[global_id] = std::make_unique<pthread_barrier_t>();
                int rc = pthread_barrier_init(barrier_waits[global_id].get(), nullptr, threads_num);
                CCL_THROW_IF_NOT(rc == 0, "pthread_barrier_init failed, rc: ", rc);
                barrier_initialized_flags[global_id] = true;
            }
            barrier_ptr = barrier_waits[global_id].get();
        }

        pthread_barrier_wait(barrier_ptr);
    }

    pthread_barrier_t* get_barrier(int global_id) {
        std::lock_guard<std::mutex> guard(barrier_waits_mutex);
        auto it = barrier_waits.find(global_id);
        CCL_THROW_IF_NOT(it != barrier_waits.end() && it->second,
                         "barrier_waits is not initialized for global_id ",
                         global_id);
        return it->second.get();
    }

    int get_node_rank(int ranks[2], int pair_comm_size) {
        // Possibly this calculation has to be universalized
        return ranks[0] * pair_comm_size + ranks[1];
    }

    int get_shared_op_id(int global_id, bool is_sender) {
        auto &entry = op_id_map[global_id];

        if (is_sender) {
            // sender path: create & broadcast
            {
                std::unique_lock<std::mutex> lk(entry.m);
                entry.id = next_op_id.fetch_add(1, std::memory_order_relaxed);
                entry.ready = true;
            } // lock is automatically released here
            entry.cv.notify_one(); // wake exactly one waiting receiver
            return entry.id;
        }
        else {
            // receiver path: wait until sender sets it
            std::unique_lock<std::mutex> lk(entry.m);
            entry.cv.wait(lk, [&] {
                return entry.ready;
            });
            int id = entry.id;
            entry.ready = false; // reset so the same entry can be reused
            return id;
        }
    }
};

} // namespace ccl
