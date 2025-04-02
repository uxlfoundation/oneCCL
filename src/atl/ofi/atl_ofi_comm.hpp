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

#include "atl/atl_base_comm.hpp"
#include "atl/ofi/atl_ofi.hpp"

class atl_ofi_comm : public atl_base_comm {
public:
    ~atl_ofi_comm() = default;

    atl_ofi_comm();
    atl_ofi_comm(std::shared_ptr<ikvs_wrapper> k);
    atl_ofi_comm(int comm_size, const std::vector<int>& ranks, std::shared_ptr<ikvs_wrapper> k);

    atl_status_t main_addr_reserve(char* main_addr) override {
        return pmi->pmrt_main_addr_reserve(main_addr);
    }

    atl_status_t finalize() override {
        ATL_CHECK_STATUS(pmi->pmrt_finalize(), "failed to finalize pmi");
        return transport->finalize();
    }

    atl_status_t update() override {
        return transport->update(pmi);
    }

    atl_status_t wait_notification() override {
        return pmi->pmrt_wait_notification();
    }

    atl_status_t set_resize_function(atl_resize_fn_t fn) override {
        return pmi->pmrt_set_resize_function(fn);
    }

    atl_status_t send(size_t ep_idx,
                      const void* buf,
                      size_t len,
                      int dst_proc_idx,
                      uint64_t tag,
                      atl_req_t& req) override {
        return transport->send(eps[ep_idx], buf, len, rank2proc_map[dst_proc_idx], tag, req);
    }

    atl_status_t recv(size_t ep_idx,
                      void* buf,
                      size_t len,
                      int src_proc_idx,
                      uint64_t tag,
                      atl_req_t& req) override {
        return transport->recv(eps[ep_idx], buf, len, rank2proc_map[src_proc_idx], tag, req);
    }

    atl_status_t probe(size_t ep_idx,
                       int src_proc_idx,
                       uint64_t tag,
                       int* found,
                       size_t* recv_len) override {
        return transport->probe(eps[ep_idx], rank2proc_map[src_proc_idx], tag, found, recv_len);
    }

    atl_status_t allgather(size_t ep_idx,
                           const void* send_buf,
                           void* recv_buf,
                           size_t len,
                           atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t allgatherv(size_t ep_idx,
                            const void* send_buf,
                            size_t send_len,
                            void* recv_buf,
                            const size_t* recv_lens,
                            const size_t* offsets,
                            atl_req_t& req) override;

    atl_status_t allreduce(size_t ep_idx,
                           const void* send_buf,
                           void* recv_buf,
                           size_t len,
                           atl_datatype_t dtype,
                           atl_reduction_t op,
                           atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t alltoall(size_t ep_idx,
                          const void* send_buf,
                          void* recv_buf,
                          int len,
                          atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t alltoallv(size_t ep_idx,
                           const void* send_buf,
                           const size_t* send_lens,
                           const size_t* send_offsets,
                           void* recv_buf,
                           const size_t* recv_lens,
                           const size_t* recv_offsets,
                           atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t barrier(size_t ep_idx, atl_req_t& req) override;

    atl_status_t bcast(size_t ep_idx, void* buf, size_t len, int root, atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t reduce(size_t ep_idx,
                        const void* send_buf,
                        void* recv_buf,
                        size_t len,
                        int root,
                        atl_datatype_t dtype,
                        atl_reduction_t op,
                        atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t reduce_scatter(size_t ep_idx,
                                const void* send_buf,
                                void* recv_buf,
                                size_t recv_len,
                                atl_datatype_t dtype,
                                atl_reduction_t op,
                                atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t read(size_t ep_idx,
                      void* buf,
                      size_t len,
                      atl_mr_t* mr,
                      uint64_t addr,
                      uintptr_t remote_key,
                      int dst_proc_idx,
                      atl_req_t& req) override {
        return transport->read(
            eps[ep_idx], buf, len, mr, addr, remote_key, rank2proc_map[dst_proc_idx], req);
    }

    atl_status_t write(size_t ep_idx,
                       const void* buf,
                       size_t len,
                       atl_mr_t* mr,
                       uint64_t addr,
                       uintptr_t remote_key,
                       int dst_proc_idx,
                       atl_req_t& req) override {
        return transport->write(
            eps[ep_idx], buf, len, mr, addr, remote_key, rank2proc_map[dst_proc_idx], req);
    }

    std::shared_ptr<atl_base_comm> comm_split(int color, int key) override;

    int get_mpi_comm() override {
        return 0;
    }

private:
    friend atl_comm_manager;

    // color, parent_rank, hostname_hash
    using rank_info_t = std::tuple<int, int, size_t>;

    atl_ofi_comm(atl_ofi_comm* parent, int color);
    atl_status_t init_transport(bool is_new);

    uint64_t tag_counter = 0;
};
