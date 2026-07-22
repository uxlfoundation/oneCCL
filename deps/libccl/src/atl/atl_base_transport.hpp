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

#include <string>
#include <vector>

#include "atl/atl_def.h"

class atl_comm_id_storage {
public:
    static constexpr int max_comm_id = 1024;
    static constexpr int invalid_comm_id = -1;

    atl_comm_id_storage() {
        free_comm_id_map.resize(max_comm_id, 1);
    }

    ~atl_comm_id_storage() = default;

    std::vector<int> get_map() const {
        return free_comm_id_map;
    }

    void acquire(int value) {
        if (value == invalid_comm_id)
            return;
        CCL_THROW_IF_NOT((value >= 0) && (value < max_comm_id), "unexpected comm_id ", value);
        free_comm_id_map[value] = 0;
    }

    void release(int value) {
        if (value == invalid_comm_id)
            return;
        CCL_THROW_IF_NOT((value >= 0) && (value < max_comm_id), "unexpected comm_id ", value);
        free_comm_id_map[value] = 1;
    }

private:
    std::vector<int> free_comm_id_map{};
};

class atl_base_transport {
public:
    virtual ~atl_base_transport() = default;

    virtual atl_status_t init(int* argc,
                              char*** argv,
                              atl_attr_t* attr,
                              const char* main_addr,
                              std::shared_ptr<ipmi> pmi) = 0;

    virtual atl_status_t update(std::shared_ptr<ipmi> pmi) = 0;

    virtual atl_status_t mr_reg(const void* buf, size_t len, atl_mr_t** mr) = 0;

    virtual atl_status_t mr_dereg(atl_mr_t* mr) = 0;

    virtual atl_status_t send(atl_ep_t& ep,
                              const void* buf,
                              size_t len,
                              int dst_proc_idx,
                              uint64_t tag,
                              atl_req_t& req) = 0;

    virtual atl_status_t recv(atl_ep_t& ep,
                              void* buf,
                              size_t len,
                              int src_proc_idx,
                              uint64_t tag,
                              atl_req_t& req) = 0;

    virtual atl_status_t probe(atl_ep_t& ep,
                               int src_proc_idx,
                               uint64_t tag,
                               int* found,
                               size_t* recv_len) = 0;

    virtual atl_status_t allgather(atl_ep_t& ep,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t len,
                                   atl_req_t& req) = 0;

    virtual atl_status_t allgatherv(atl_ep_t& ep,
                                    const void* send_buf,
                                    size_t send_len,
                                    void* recv_buf,
                                    const size_t* recv_lens,
                                    const size_t* offsets,
                                    atl_req_t& req) = 0;

    virtual atl_status_t allreduce(atl_ep_t& ep,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t len,
                                   atl_datatype_t dtype,
                                   atl_reduction_t op,
                                   atl_req_t& req) = 0;

    virtual atl_status_t alltoall(atl_ep_t& ep,
                                  const void* send_buf,
                                  void* recv_buf,
                                  int len,
                                  atl_req_t& req) = 0;

    virtual atl_status_t alltoallv(atl_ep_t& ep,
                                   const void* send_buf,
                                   const size_t* send_lens,
                                   const size_t* send_offsets,
                                   void* recv_buf,
                                   const size_t* recv_lens,
                                   const size_t* recv_offsets,
                                   atl_req_t& req) = 0;

    virtual atl_status_t barrier(atl_ep_t& ep, atl_req_t& req) = 0;

    virtual atl_status_t bcast(atl_ep_t& ep, void* buf, size_t len, int root, atl_req_t& req) = 0;
    virtual atl_status_t broadcast(atl_ep_t& ep,
                                   void* send_buf,
                                   void* recv_buf,
                                   size_t len,
                                   int root,
                                   atl_req_t& req) = 0;

    virtual atl_status_t reduce(atl_ep_t& ep,
                                const void* send_buf,
                                void* recv_buf,
                                size_t len,
                                int root,
                                atl_datatype_t dtype,
                                atl_reduction_t op,
                                atl_req_t& req) = 0;

    virtual atl_status_t reduce_scatter(atl_ep_t& ep,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t recv_len,
                                        atl_datatype_t dtype,
                                        atl_reduction_t op,
                                        atl_req_t& req) = 0;

    virtual atl_status_t read(atl_ep_t& ep,
                              void* buf,
                              size_t len,
                              atl_mr_t* mr,
                              uint64_t addr,
                              uintptr_t remote_key,
                              int dst_proc_idx,
                              atl_req_t& req) = 0;

    virtual atl_status_t write(atl_ep_t& ep,
                               const void* buf,
                               size_t len,
                               atl_mr_t* mr,
                               uint64_t addr,
                               uintptr_t remote_key,
                               int dst_proc_idx,
                               atl_req_t& req) = 0;

    virtual atl_status_t wait(atl_ep_t& ep, atl_req_t& req) = 0;

    virtual atl_status_t wait_all(atl_ep_t& ep, std::vector<atl_req_t>& reqs, size_t count) = 0;

    virtual atl_status_t cancel(atl_ep_t& ep, atl_req_t& req) = 0;

    virtual atl_status_t poll(atl_ep_t& ep) = 0;

    virtual atl_status_t check(atl_ep_t& ep, atl_req_t& req) = 0;

    virtual atl_proc_coord_t create_proc_coord(atl_ep_t& ep) = 0;

    virtual void comms_free(std::vector<atl_ep_t>& eps) = 0;

    virtual atl_status_t comm_split(const std::vector<atl_ep_t>& base_eps,
                                    std::vector<atl_ep_t>& eps,
                                    size_t color,
                                    int key,
                                    int local_idx,
                                    int local_count) = 0;

    virtual atl_status_t get_rank2proc_map(std::shared_ptr<ipmi> pmi,
                                           std::vector<int>& rank2proc_map,
                                           atl_proc_coord_t coord) = 0;

    virtual std::string to_string() = 0;

    virtual atl_status_t finalize(int global_idx = 0) = 0;

    bool is_inited() {
        return inited;
    }

    atl_proc_coord_t get_proc_coord() {
        return coord;
    }

    std::vector<atl_ep_t> get_eps() {
        return eps;
    }

    atl_comm_id_storage& get_comm_id_storage() {
        return comm_id_storage;
    }

protected:
    atl_comm_id_storage comm_id_storage;

    atl_proc_coord_t coord;
    std::vector<atl_ep_t> eps;

    bool is_finalized{ false };
    bool inited{ false };
};
