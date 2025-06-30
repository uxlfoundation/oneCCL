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

#include <memory>
#include <mutex>
#include <list>
#include <vector>

#include "atl/atl_base_transport.hpp"
#include "atl/atl_def.h"
#include "comm/atl_tag.hpp"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/ikvs_wrapper.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.h"

class ccl_executor;

class atl_base_comm {
protected:
    atl_base_comm() = default;

public:
    virtual ~atl_base_comm();

    virtual atl_status_t main_addr_reserve(char* main_addr) {
        return ATL_STATUS_UNSUPPORTED;
    }

    virtual atl_status_t finalize() = 0;

    virtual atl_status_t update() {
        return ATL_STATUS_UNSUPPORTED;
    }

    virtual atl_status_t wait_notification() {
        return ATL_STATUS_UNSUPPORTED;
    }

    virtual atl_status_t set_resize_function(atl_resize_fn_t fn) {
        return ATL_STATUS_UNSUPPORTED;
    }

    virtual atl_status_t mr_reg(const void* buf, size_t len, atl_mr_t** mr) {
        return transport->mr_reg(buf, len, mr);
    }

    virtual atl_status_t mr_dereg(atl_mr_t* mr) {
        return transport->mr_dereg(mr);
    }

    virtual atl_status_t send(size_t ep_idx,
                              const void* buf,
                              size_t len,
                              int dst_proc_idx,
                              uint64_t tag,
                              atl_req_t& req) {
        return transport->send(eps[ep_idx], buf, len, dst_proc_idx, tag, req);
    }

    virtual atl_status_t recv(size_t ep_idx,
                              void* buf,
                              size_t len,
                              int src_proc_idx,
                              uint64_t tag,
                              atl_req_t& req) {
        return transport->recv(eps[ep_idx], buf, len, src_proc_idx, tag, req);
    }

    virtual atl_status_t probe(size_t ep_idx,
                               int src_proc_idx,
                               uint64_t tag,
                               int* found,
                               size_t* recv_len) {
        return transport->probe(eps[ep_idx], src_proc_idx, tag, found, recv_len);
    }

    virtual atl_status_t allgather(size_t ep_idx,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t len,
                                   atl_req_t& req) {
        return transport->allgather(eps[ep_idx], send_buf, recv_buf, len, req);
    }

    virtual atl_status_t allgatherv(size_t ep_idx,
                                    const void* send_buf,
                                    size_t send_len,
                                    void* recv_buf,
                                    const size_t* recv_lens,
                                    const size_t* offsets,
                                    atl_req_t& req) {
        return transport->allgatherv(
            eps[ep_idx], send_buf, send_len, recv_buf, recv_lens, offsets, req);
    }

    virtual atl_status_t allreduce(size_t ep_idx,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t len,
                                   atl_datatype_t dtype,
                                   atl_reduction_t op,
                                   atl_req_t& req) {
        return transport->allreduce(eps[ep_idx], send_buf, recv_buf, len, dtype, op, req);
    }

    virtual atl_status_t alltoall(size_t ep_idx,
                                  const void* send_buf,
                                  void* recv_buf,
                                  int len,
                                  atl_req_t& req) {
        return transport->alltoall(eps[ep_idx], send_buf, recv_buf, len, req);
    }

    virtual atl_status_t alltoallv(size_t ep_idx,
                                   const void* send_buf,
                                   const size_t* send_lens,
                                   const size_t* send_offsets,
                                   void* recv_buf,
                                   const size_t* recv_lens,
                                   const size_t* recv_offsets,
                                   atl_req_t& req) {
        return transport->alltoallv(
            eps[ep_idx], send_buf, send_lens, send_offsets, recv_buf, recv_lens, recv_offsets, req);
    }

    virtual atl_status_t barrier(size_t ep_idx, atl_req_t& req) {
        return transport->barrier(eps[ep_idx], req);
    }

    virtual atl_status_t bcast(size_t ep_idx, void* buf, size_t len, int root, atl_req_t& req) {
        return transport->bcast(eps[ep_idx], buf, len, root, req);
    }

    virtual atl_status_t broadcast(size_t ep_idx,
                                   void* send_buf,
                                   void* recv_buf,
                                   size_t len,
                                   int root,
                                   atl_req_t& req) {
        return transport->broadcast(eps[ep_idx], send_buf, recv_buf, len, root, req);
    }

    virtual atl_status_t reduce(size_t ep_idx,
                                const void* send_buf,
                                void* recv_buf,
                                size_t len,
                                int root,
                                atl_datatype_t dtype,
                                atl_reduction_t op,
                                atl_req_t& req) {
        return transport->reduce(eps[ep_idx], send_buf, recv_buf, len, root, dtype, op, req);
    }

    virtual atl_status_t reduce_scatter(size_t ep_idx,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t recv_len,
                                        atl_datatype_t dtype,
                                        atl_reduction_t op,
                                        atl_req_t& req) {
        return transport->reduce_scatter(eps[ep_idx], send_buf, recv_buf, recv_len, dtype, op, req);
    }

    virtual atl_status_t read(size_t ep_idx,
                              void* buf,
                              size_t len,
                              atl_mr_t* mr,
                              uint64_t addr,
                              uintptr_t remote_key,
                              int dst_proc_idx,
                              atl_req_t& req) {
        return transport->read(eps[ep_idx], buf, len, mr, addr, remote_key, dst_proc_idx, req);
    }

    virtual atl_status_t write(size_t ep_idx,
                               const void* buf,
                               size_t len,
                               atl_mr_t* mr,
                               uint64_t addr,
                               uintptr_t remote_key,
                               int dst_proc_idx,
                               atl_req_t& req) {
        return transport->write(eps[ep_idx], buf, len, mr, addr, remote_key, dst_proc_idx, req);
    }

    virtual atl_status_t wait(size_t ep_idx, atl_req_t& req) {
        return transport->wait(eps[ep_idx], req);
    }

    virtual atl_status_t wait_all(size_t ep_idx, std::vector<atl_req_t>& reqs, size_t count) {
        return transport->wait_all(eps[ep_idx], reqs, count);
    }

    virtual atl_status_t cancel(size_t ep_idx, atl_req_t& req) {
        return transport->cancel(eps[ep_idx], req);
    }

    virtual atl_status_t poll(size_t ep_idx) {
        return transport->poll(eps[ep_idx]);
    }

    virtual atl_status_t check(size_t ep_idx, atl_req_t& req) {
        return transport->check(eps[ep_idx], req);
    }

    virtual std::shared_ptr<atl_base_comm> comm_split(int color, int key) = 0;

    int get_rank() const {
        return rank;
    }

    int get_size() const {
        return size;
    }

    int get_r2r_color() const {
        return coord.local_idx;
    }

    int get_host_color() const {
        return coord.hostname_hash;
    }

    std::vector<int> get_rank2rank_map() const {
        return rank2rank_map;
    }

    int create_comm_id();

    int get_comm_id() const {
        return comm_id;
    }

    void reset_comm_id() {
        comm_id = atl_comm_id_storage::invalid_comm_id;
    }

    std::shared_ptr<ccl_atl_tag> tag_creator;
    static atl_attr_t attr;

    virtual int get_mpi_comm() = 0;

protected:
    void init_tag();
    void update_executor();

    friend class atl_comm_manager;

    int rank;
    int size;

    int parent_rank;
    int parent_size;

    std::vector<int> rank2rank_map{};
    std::vector<int> rank2proc_map{};
    atl_proc_coord_t coord;

    int comm_id = atl_comm_id_storage::invalid_comm_id;

    std::shared_ptr<ipmi> pmi;

    std::vector<atl_ep_t> eps;

    static ccl_executor* executor;
    static atl_base_transport* transport;
    static std::atomic<size_t> comm_count;
    static ccl_spinlock comm_id_storage_guard;
};

class atl_comm_manager {
public:
    static std::shared_ptr<atl_base_comm> create();
    static std::shared_ptr<atl_base_comm> create(ccl::ccl_comm_attr_impl& attr);
    static std::shared_ptr<atl_base_comm> create(std::shared_ptr<ikvs_wrapper> k,
                                                 ccl::ccl_comm_attr_impl& attr);
    static std::shared_ptr<atl_base_comm> create(int comm_size,
                                                 const std::vector<int>& ranks,
                                                 std::shared_ptr<ikvs_wrapper> k,
                                                 ccl::ccl_comm_attr_impl& attr);
    static std::shared_ptr<atl_base_comm> create_with_id(const std::shared_ptr<atl_base_comm> other,
                                                         int comm_id);

    static void set_internal_env(const atl_attr_t& attr);
    static void set_executor(ccl_executor* exec);
};
