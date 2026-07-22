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

#include "reduce_scatter_strategy.hpp"

#ifdef CCL_ENABLE_SYCL
#include "sycl_coll.hpp"

template <class Dtype>
struct sycl_reduce_scatter_coll : sycl_base_coll<Dtype, reduce_scatter_strategy_impl> {
    using coll_base = sycl_base_coll<Dtype, reduce_scatter_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::host_send_buf;
    using coll_base::host_recv_buf;

    sycl_reduce_scatter_coll(bench_init_attr init_attr) : coll_base(init_attr) {}

    virtual void finalize_internal(size_t elem_count,
                                   ccl::communicator& comm,
                                   ccl::stream& stream,
                                   size_t rank_idx) override {
        Dtype sbuf_expected = get_val<Dtype>(static_cast<float>(comm.rank()));
        Dtype rbuf_expected = get_val<Dtype>((comm.size() - 1) * ((float)comm.size() / 2));

        size_t recv_elem_count = elem_count / comm.size();

        size_t send_bytes = elem_count * base_coll::get_dtype_size();
        size_t recv_bytes = elem_count * base_coll::get_dtype_size();

        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                stream.get_native()
                    .memcpy(host_send_buf.data(), send_bufs[b_idx][rank_idx], send_bytes)
                    .wait();

                if (!base_coll::get_inplace()) {
                    stream.get_native()
                        .memcpy(host_recv_buf.data(), recv_bufs[b_idx][rank_idx], recv_bytes)
                        .wait();
                }
                else {
                    auto recv_buf_adjusted = static_cast<Dtype*>(send_bufs[b_idx][rank_idx]) +
                                             recv_elem_count * comm.rank();
                    auto recv_elem_bytes = recv_elem_count * base_coll::get_dtype_size();
                    stream.get_native()
                        .memcpy(host_recv_buf.data(), recv_buf_adjusted, recv_elem_bytes)
                        .wait();
                }
            }
            else {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx][rank_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx][rank_idx]));
                auto send_buf_acc = send_buf->get_host_access(sycl::read_only);
                auto recv_buf_acc = recv_buf->get_host_access(sycl::read_only);

                stream.get_native()
                    .memcpy(host_send_buf.data(), send_buf_acc.get_pointer(), send_bytes)
                    .wait();

                if (!base_coll::get_inplace()) {
                    stream.get_native()
                        .memcpy(host_recv_buf.data(), recv_buf_acc.get_pointer(), recv_bytes)
                        .wait();
                }
                else {
                    auto recv_buf_adjusted =
                        send_buf_acc.get_pointer() + recv_elem_count * comm.rank();
                    auto recv_elem_bytes = recv_elem_count * base_coll::get_dtype_size();
                    stream.get_native()
                        .memcpy(host_recv_buf.data(), recv_buf_adjusted, recv_elem_bytes)
                        .wait();
                }
            }

            for (size_t e_idx = 0; e_idx < elem_count; e_idx++) {
                Dtype value = host_send_buf[e_idx];
                if (!base_coll::get_inplace() && value != sbuf_expected) {
                    std::cout << this->name() << " send_bufs: buf_idx " << b_idx << ", rank_idx "
                              << rank_idx << ", elem_idx " << e_idx << ", expected "
                              << sbuf_expected << ", got " << value << std::endl;
                    ASSERT(0, "unexpected value");
                }
            }

            for (size_t e_idx = 0; e_idx < recv_elem_count; e_idx++) {
                Dtype value = host_recv_buf[e_idx];
                if (base_coll::check_error<Dtype>(value, rbuf_expected, comm)) {
                    std::cout << this->name() << " recv_bufs: buf_idx " << b_idx << ", rank_idx "
                              << rank_idx << ", elem_idx " << e_idx << ", expected "
                              << rbuf_expected << ", got " << value << std::endl;
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};
#endif // CCL_ENABLE_SYCL
