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

#include "scatter_strategy.hpp"

#ifdef CCL_ENABLE_SYCL
#include "sycl_coll.hpp"

template <class Dtype>
struct sycl_scatter_coll : sycl_base_coll<Dtype, scatter_strategy_impl> {
    using coll_base = sycl_base_coll<Dtype, scatter_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::host_send_buf;
    using coll_base::host_recv_buf;

    sycl_scatter_coll(bench_init_attr init_attr) : coll_base(init_attr) {}

    virtual void prepare_internal(size_t elem_count,
                                  ccl::communicator& comm,
                                  ccl::stream& stream,
                                  size_t rank_idx) override {
        int comm_rank = comm.rank();
        size_t send_count = elem_count * comm.size();
        size_t recv_count = elem_count;
        size_t send_bytes = send_count * base_coll::get_dtype_size();
        size_t recv_bytes = recv_count * base_coll::get_dtype_size();

        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                // In-order queue: enqueue the fill(s) without intermediate waits
                // and block once at the end, before start() reads the buffers off
                // a different queue.
                auto& q = stream.get_native();
                // Only the root holds a send buffer; non-root send_bufs are nullptr.
                if (comm_rank == COLL_ROOT) {
                    host_send_buf = base_coll::template get_initial_values<Dtype>(
                        send_count, static_cast<int>(b_idx));
                    q.memcpy(send_bufs[b_idx][rank_idx], host_send_buf.data(), send_bytes);
                }
                // In-place root has no standalone recv buffer; its receive region
                // lives inside send_bufs at root*elem_count and is already filled.
                if (recv_bufs[b_idx][rank_idx] != nullptr) {
                    q.memset(recv_bufs[b_idx][rank_idx], 0, recv_bytes);
                }
                q.wait();
            }
            else {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx][rank_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx][rank_idx]));

                if (comm_rank == COLL_ROOT) {
                    auto send_buf_acc = send_buf->get_host_access(sycl::write_only);
                    for (size_t e_idx = 0; e_idx < send_count; e_idx++) {
                        send_buf_acc[e_idx] = get_val<Dtype>(static_cast<float>(b_idx));
                    }
                }
                auto recv_buf_acc = recv_buf->get_host_access(sycl::write_only);
                for (size_t e_idx = 0; e_idx < recv_count; e_idx++) {
                    recv_buf_acc[e_idx] = 0;
                }
            }
        }
    }

    virtual void finalize_internal(size_t elem_count,
                                   ccl::communicator& comm,
                                   ccl::stream& stream,
                                   size_t rank_idx) override {
        size_t recv_bytes = elem_count * base_coll::get_dtype_size();
        bool is_root = comm.rank() == COLL_ROOT;

        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                // In-place root reads its result from inside send_bufs at root*elem_count.
                void* recv = recv_bufs[b_idx][rank_idx];
                if (recv == nullptr && is_root) {
                    recv = static_cast<Dtype*>(send_bufs[b_idx][rank_idx]) + elem_count * COLL_ROOT;
                }
                stream.get_native().memcpy(host_recv_buf.data(), recv, recv_bytes).wait();
            }
            else {
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx][rank_idx]));
                auto recv_buf_acc = recv_buf->get_host_access(sycl::read_only);
                stream.get_native()
                    .memcpy(host_recv_buf.data(), recv_buf_acc.get_pointer(), recv_bytes)
                    .wait();
            }

            // Root fills send buffer uniformly with b_idx; every rank receives that value.
            Dtype expected = get_val<Dtype>(static_cast<float>(b_idx));
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++) {
                Dtype value = host_recv_buf[e_idx];
                if (base_coll::check_error<Dtype>(value, expected, comm)) {
                    std::cout << this->name() << " recv_bufs: buf_idx " << b_idx << ", rank_idx "
                              << rank_idx << ", elem_idx " << e_idx << ", expected " << expected
                              << ", got " << value << std::endl;
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};
#endif // CCL_ENABLE_SYCL
