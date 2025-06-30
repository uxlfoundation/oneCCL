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
#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>
#include <condition_variable>

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

#include "atl/atl_base_comm.hpp"
#include "comm/comm.hpp"
#include "topology/topo_manager.hpp"

namespace ccl {

static std::atomic<int> next_op_id{ 0 };

struct op_id_entry {
    std::mutex m;
    std::condition_variable cv;
    int id = -1;
    bool ready = false;
};

struct handshake_data {
    std::mutex m;
    std::condition_variable cv;

    // Indicate that the receiver has published its pointer
    bool recv_published = false;

    // Indicate that the sender has finished its memcpy
    bool copy_done = false;
};

class shared_resources {
private:
    std::unordered_map<int, std::mutex> resource_mutexes;
    std::unordered_map<int, int> barrier_initialized_flags;

public:
    std::unordered_map<int, rank_info_vec_t> rank_info_vec_globs;
    std::unordered_map<int, std::vector<char>> all_hostnames_raw_globs;
    std::unordered_map<int, std::unordered_map<int, std::vector<void *>>> hash_table;
#ifdef CCL_ENABLE_SYCL
    sycl::event copy_event;
#endif // CCL_ENABLE_SYCL
    std::unordered_map<int, handshake_data> handshakes;
    std::unordered_map<int, op_id_entry> op_id_map;
#ifdef CCL_ENABLE_SYCL
    std::unordered_map<int, sycl::event> receiver_ready_events_map;
#endif // CCL_ENABLE_SYCL
    int current_global_id = ccl_comm::invalid_id;
    std::unordered_map<int, pthread_barrier_t> barrier_waits;

    shared_resources() = default;
    shared_resources(const shared_resources &other) = default;
    shared_resources &operator=(const shared_resources &other) = default;

    ~shared_resources() {
        hash_table.clear();
        barrier_waits.clear();
        rank_info_vec_globs.clear();
        barrier_initialized_flags.clear();
        resource_mutexes.clear();
        all_hostnames_raw_globs.clear();
    }
#ifdef CCL_ENABLE_SYCL
    void set_receiver_ready_event(int op_id, const sycl::event &evt) {
        std::lock_guard<std::mutex> lk(get_resource_mutex(op_id));
        receiver_ready_events_map[op_id] = evt;
    }

    sycl::event get_receiver_ready_event(int op_id) {
        std::lock_guard<std::mutex> lk(get_resource_mutex(op_id));
        return receiver_ready_events_map.at(op_id);
    }
#endif // CCL_ENABLE_SYCL
    void resize_rank_info_vec_glob(size_t new_size, int global_id) {
        std::lock_guard<std::mutex> lock(get_resource_mutex(global_id));
        rank_info_vec_globs[global_id].resize(new_size);
    }

    void resize_all_hostnames_raw(size_t new_size, int global_id) {
        std::lock_guard<std::mutex> lock(get_resource_mutex(global_id));
        all_hostnames_raw_globs[global_id].resize(new_size);
    }

    std::mutex &get_resource_mutex(int global_id) {
        static std::mutex mu;
        std::lock_guard<std::mutex> guard(mu);
        return resource_mutexes[global_id];
    }

    std::vector<char> &get_all_hostnames_raw_glob(int global_id) {
        std::lock_guard<std::mutex> lock(get_resource_mutex(global_id));
        return all_hostnames_raw_globs[global_id];
    }

    rank_info_vec_t &get_rank_info_vec_glob(int global_id) {
        std::lock_guard<std::mutex> lock(get_resource_mutex(global_id));
        return rank_info_vec_globs[global_id];
    }

    void init_barrier_wait(int threads_num, int global_id) {
        std::mutex &mutex = get_resource_mutex(global_id);

        mutex.lock();

        // Check if the barrier has already been initialized
        if (barrier_initialized_flags.find(global_id) == barrier_initialized_flags.end() ||
            !barrier_initialized_flags[global_id]) {
            // Initialize the barrier for the current global_id
            pthread_barrier_init(&barrier_waits[global_id], NULL, threads_num);
            // Set the flag to indicate barrier is initialized
            barrier_initialized_flags[global_id] = true;
        }

        mutex.unlock();

        pthread_barrier_wait(&barrier_waits[global_id]);
    }

    void do_ipc_exchangeExt(
        ccl_comm *comm,
        std::unordered_map<int, std::unordered_map<int, std::vector<void *>>> &hash_table,
        ccl_stream *stream,
        std::vector<void *> ptrs,
        int exchange_id = 0,
        bool is_pt2pt = false) {
        {
            static std::mutex hash_table_mutex;
            std::lock_guard<std::mutex> guard(hash_table_mutex);
            hash_table[exchange_id][comm->rank()] = ptrs;
        }
        if (!is_pt2pt) {
            pthread_barrier_wait(&barrier_waits[comm->global_current_id]);
        }
    }

    int get_node_rank(int ranks[2], int pair_comm_size) {
        // Possibly this calculation has to be universalized
        return ranks[0] * pair_comm_size + ranks[1];
    }

    template <typename T, int N>
    std::array<T *, N> get_ipc_ptrsExt(
        std::shared_ptr<ccl_comm> comm,
        std::unordered_map<int, std::unordered_map<int, std::vector<void *>>> &hash_table,
        const int comm_index,
        const int handle_index,
        void *local_ptr,
        int exchange_id = 0,
        bool dummy_copy = 0,
        std::shared_ptr<ccl_comm> even_comm = nullptr,
        std::shared_ptr<ccl_comm> pair_comm = nullptr) {
        std::array<T *, N> remote_ptrs = {};
        const int rank = comm->rank();
        const int size = comm->size();

        remote_ptrs[rank] = (T *)local_ptr;
        for (int i = 1; i < size; i++) {
            int peer_rank = (rank + i) % size;
            int peer_rank_node = peer_rank;
            if (comm_index > 0) {
                int ranks[2] = { even_comm->rank(), pair_comm->rank() };
                ranks[comm_index - 1] = peer_rank;
                peer_rank_node = get_node_rank(ranks, pair_comm->size());
            }
            if (hash_table.find(exchange_id) != hash_table.end() &&
                hash_table[exchange_id].find(peer_rank_node) != hash_table[exchange_id].end()) {
                const auto &ptr = hash_table[exchange_id][peer_rank_node][handle_index];
                remote_ptrs[peer_rank] = (T *)ptr;
            }
            CCL_ASSERT(remote_ptrs[peer_rank] != NULL);
        }
        return remote_ptrs;
    }

    int get_shared_op_id(int global_id, bool is_sender) {
        auto &entry = op_id_map[global_id];

        if (is_sender) {
            // sender path: create & broadcast
            std::unique_lock<std::mutex> lk(entry.m);
            entry.id = next_op_id.fetch_add(1, std::memory_order_relaxed);
            entry.ready = true;
            lk.unlock();
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
