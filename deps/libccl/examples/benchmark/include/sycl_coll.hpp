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

#include <iostream>
#include <map>
#include <set>
#include <string>

#include "coll.hpp"
#include "sycl_base.hpp" /* from examples/include */

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>

/* sycl-specific base implementation */
template <class Dtype, class strategy>
struct sycl_base_coll : base_coll, private strategy {
    using coll_strategy = strategy;

    template <class... Args>
    sycl_base_coll(bench_init_attr init_attr, Args&&... args)
            : base_coll(init_attr),
              coll_strategy() {
        auto& transport = transport_data::instance();
        auto streams = transport.get_bench_streams();

        size_t send_multiplier = coll_strategy::get_send_multiplier();
        size_t recv_multiplier = coll_strategy::get_recv_multiplier();

        size_t send_count = base_coll::get_allocation_count() * send_multiplier;
        size_t recv_count = base_coll::get_allocation_count() * recv_multiplier;

        if (base_coll::get_max_elem_count() == 0) {
            return;
        }
        for (size_t rank_idx = 0; rank_idx < base_coll::get_ranks_per_proc(); rank_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                allocators.push_back(buf_allocator<Dtype>(streams[rank_idx].get_native()));

                auto& allocator = allocators[rank_idx];

                sycl::usm::alloc usm_alloc_type;
                auto bench_alloc_type = base_coll::get_sycl_usm_type();
                if (bench_alloc_type == SYCL_USM_SHARED)
                    usm_alloc_type = sycl::usm::alloc::shared;
                else if (bench_alloc_type == SYCL_USM_DEVICE)
                    usm_alloc_type = sycl::usm::alloc::device;
                else
                    ASSERT(0, "unexpected bench_alloc_type %d", bench_alloc_type);

                for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
                    void* send_buf = nullptr;
                    void* recv_buf = nullptr;
                    if (!base_coll::get_inplace()) {
                        send_buf = allocator.allocate(send_count, usm_alloc_type);
                        recv_buf = allocator.allocate(recv_count, usm_alloc_type);
                    }
                    else {
                        bool is_allgatherv = strcmp(coll_strategy::class_name(), "allgatherv") == 0;
                        bool is_allgather = strcmp(coll_strategy::class_name(), "allgather") == 0;
                        bool is_reduce_scatter =
                            strcmp(coll_strategy::class_name(), "reduce_scatter") == 0;
                        if (is_allgatherv || is_allgather) {
                            // send_buf will be set when the count is known, since the offset is unknown at this point.
                            send_buf = nullptr;
                            recv_buf = allocator.allocate(recv_count, usm_alloc_type);
                        }
                        else if (is_reduce_scatter) {
                            send_buf = allocator.allocate(send_count, usm_alloc_type);
                            // recv_buf will be set when the count is known, since the offset is unknown at this point.
                            recv_buf = nullptr;
                        }
                        else {
                            send_buf = allocator.allocate(std::max(send_count, recv_count),
                                                          usm_alloc_type);
                            recv_buf = send_buf;
                        }
                    }

                    // offset send_buf, original deallocated automatically by ~buf_allocator()
                    if (send_buf) {
                        send_buf = static_cast<Dtype*>(send_buf) + base_coll::get_elem_offset();
                    }

                    // offset recv_buf, original deallocated automatically by ~buf_allocator()
                    if (recv_buf) {
                        recv_buf = static_cast<Dtype*>(recv_buf) + base_coll::get_elem_offset();
                    }

                    send_bufs[idx][rank_idx] = send_buf;
                    recv_bufs[idx][rank_idx] = recv_buf;
                }
            }
            else {
                for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
                    send_bufs[idx][rank_idx] = new sycl::buffer<Dtype, 1>(send_count);
                    recv_bufs[idx][rank_idx] = new sycl::buffer<Dtype, 1>(recv_count);
                }
            }
        }
        host_send_buf.resize(send_count);
        host_recv_buf.resize(recv_count);
    }

    sycl_base_coll(bench_init_attr init_attr) : sycl_base_coll(init_attr, 1, 1) {}

    virtual ~sycl_base_coll() {
        for (size_t rank_idx = 0; rank_idx < base_coll::get_ranks_per_proc(); rank_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_BUF) {
                for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
                    delete static_cast<sycl_buffer_t<Dtype>*>(send_bufs[idx][rank_idx]);
                    if (!base_coll::get_inplace()) {
                        delete static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[idx][rank_idx]);
                    }
                }
            }
        }
    }

    const char* name() const noexcept override {
        return coll_strategy::class_name();
    }

    virtual void start(size_t count,
                       size_t buf_idx,
                       const bench_exec_attr& attr,
                       req_list_t& reqs) override {
        auto& transport = transport_data::instance();
        auto& comms = transport.get_comms();
        auto streams = transport.get_streams();
        size_t ranks_per_proc = base_coll::get_ranks_per_proc();

        for (size_t rank_idx = 0; rank_idx < ranks_per_proc; rank_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                coll_strategy::start_internal(comms[rank_idx],
                                              count,
                                              static_cast<Dtype*>(send_bufs[buf_idx][rank_idx]),
                                              static_cast<Dtype*>(recv_bufs[buf_idx][rank_idx]),
                                              attr,
                                              reqs,
                                              streams[rank_idx],
                                              coll_strategy::get_op_attr(attr));
            }
            else {
                throw std::runtime_error(std::string(__FUNCTION__) +
                                         " - only USM buffers are supported\n");
            }
        }
    }

    virtual void prepare_internal(size_t elem_count,
                                  ccl::communicator& comm,
                                  ccl::stream& stream,
                                  size_t rank_idx) override {
        int comm_rank = comm.rank();

        size_t send_count = coll_strategy::get_send_multiplier() * elem_count;
        size_t recv_count = coll_strategy::get_recv_multiplier() * elem_count;

        size_t send_bytes = send_count * base_coll::get_dtype_size();
        size_t recv_bytes = recv_count * base_coll::get_dtype_size();

        host_send_buf = get_initial_values<Dtype>(send_count, comm_rank);

        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                if ((strcmp(coll_strategy::class_name(), "allgatherv") == 0 ||
                     strcmp(coll_strategy::class_name(), "allgather") == 0) &&
                    base_coll::get_inplace()) {
                    // for inplace allgather(v), the input data needs to be at an index comm_rank*send_count
                    // of the send_buffer rather than at index 0 for the non-inplace case
                    //  TODO: create prepare_internal for allgatherv
                    stream.get_native()
                        .memcpy((char*)(recv_bufs[b_idx][rank_idx]) + send_bytes * comm_rank,
                                host_send_buf.data(),
                                send_bytes)
                        .wait();
                }
                else {
                    stream.get_native()
                        .memcpy(send_bufs[b_idx][rank_idx], host_send_buf.data(), send_bytes)
                        .wait();
                }
                if (!base_coll::get_inplace()) {
                    stream.get_native().memset(recv_bufs[b_idx][rank_idx], -1, recv_bytes).wait();
                }
            }
            else {
                stream.get_native()
                    .submit([&](sycl::handler& h) {
                        auto send_buf =
                            (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx][rank_idx]));
                        auto send_buf_acc =
                            send_buf->template get_access<sycl::access::mode::write>(h, send_count);
                        h.fill(send_buf_acc, static_cast<Dtype>(comm_rank));
                    })
                    .wait();

                stream.get_native()
                    .submit([&](sycl::handler& h) {
                        auto recv_buf =
                            (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx][rank_idx]));
                        auto recv_buf_acc =
                            recv_buf->template get_access<sycl::access::mode::write>(h, recv_count);
                        h.fill(recv_buf_acc, static_cast<Dtype>(-1));
                    })
                    .wait();
            }
        }
    }

    ccl::datatype get_dtype() const override final {
        return get_ccl_dtype<Dtype>();
    }

    /* used on fill/check phases */
    aligned_vector<Dtype> host_send_buf;
    aligned_vector<Dtype> host_recv_buf;

private:
    std::vector<buf_allocator<Dtype>> allocators;
};

#endif // CCL_ENABLE_SYCL
