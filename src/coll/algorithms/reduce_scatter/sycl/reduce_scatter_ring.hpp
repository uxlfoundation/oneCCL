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

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_sycl.hpp"
#include "coll/coll_util.hpp"

namespace ccl {
namespace v1 {

template <typename T>
inline void reduce_kernel(const void *in1_, const void *in2_, void *out_, size_t idx) {
    T *i1 = (T *)in1_;
    T *i2 = (T *)in2_;
    T *out = (T *)out_;
    out[idx] = i1[idx] + i2[idx];
}

template <typename T, int vec_size>
inline void reduce_pair(const void *in1,
                        const void *in2,
                        void *out,
                        const size_t count,
                        const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;
    const size_t packed_count = count / vec_size;
    if (idx < packed_count) {
        reduce_kernel<AT>(in1, in2, out, idx);
    }
    else {
        const size_t new_idx = vec_size * packed_count + idx - packed_count;
        if (new_idx < count) {
            reduce_kernel<T>(in1, in2, out, new_idx);
        }
    }
}

//#define PRINT_TIMING

// blocking algorithm calls MPI GPU pipelining directly
template <typename T>
inline sycl::event reduce_scatter_ring_blocking_impl(sycl::queue &q,
                                                     const void *send_buf,
                                                     void *recv_buf,
                                                     size_t count,
                                                     datatype dtype,
                                                     reduction reduction,
                                                     ccl_comm *comm,
                                                     const vector_class<event> &deps,
                                                     bool &done) {
    sycl::event sycl_e;
    int world = comm->size();
    int rank = comm->rank();

    done = true;

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    // prepare ring block counts/sizes
    size_t main_block_count = count / world;
    size_t last_block_count = main_block_count + count % world;
    size_t main_block_size = main_block_count * ccl_dtype.size();
    size_t last_block_size = last_block_count * ccl_dtype.size();

    bool in_place =
        ccl::is_reduce_scatter_inplace(send_buf, recv_buf, main_block_count, ccl_dtype.size(), rank, world);

    // to make sure that buffer con hold the larges possible block
    size_t buf_size = last_block_size;
    // align an allocation size up to 4 bytes, to have a better vector reduction
    size_t aligned_buf_size = (buf_size + 3) / 4 * 4;
    void *buffers[2] = { NULL };
    int to_free = 0;
    if (world > 2 || in_place) {
        if (aligned_buf_size <= comm->get_scaleout_device_buf_size() / 2) {
            buffers[0] = comm->get_scaleout_device_buf(q);
        }
        else {
            buffers[0] = sycl::malloc_device(aligned_buf_size * 2, q);
            to_free = 1;
        }
        buffers[1] = (char *)buffers[0] + aligned_buf_size;
    }

    // blocking
    q.wait();

    auto reduce_invoke =
        [=, &q]<int VS, int SGS>(
            void *in1, void *in2, void *out, size_t reduce_count, std::vector<sycl::event> l_dep_events) {
            constexpr int vec_size = VS, wg_size = SGS, sg_size = SGS;
            const size_t kernel_threads = reduce_count / vec_size + reduce_count % vec_size;
            const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
            sycl::event local_event = q.submit([=](sycl::handler &h) {
                h.depends_on(l_dep_events);
                h.parallel_for(sycl::nd_range<1>(kernel_size, wg_size), [=](sycl::nd_item<1> it) {
                    reduce_pair<T, vec_size>(in1, in2, out, reduce_count, it);
                });
            });
            return local_event;
        };

    // left and right neighbour
    const int right = (rank + 1) % world;
    const int left = (rank - 1 + world) % world;

    // starting indexes
    int s = (rank - 1 + world) % world;
    int r = (s - 1 + world) % world;

    // send/recv pointers
    void *send_ptr, *recv_ptr, *out_ptr;
    size_t send_block_count, recv_block_count;
    size_t send_block_size, recv_block_size;

    // tag creation
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t pt2pt_sched_id = atl_comm->tag_creator->get_pt2pt_sched_id();
    int64_t tag = atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), pt2pt_sched_id, 0);

    int ep_idx = 0;
    int iter = 0;
    int index = 0;
    while (iter < world - 1) {
        send_block_count = (s == (world - 1)) ? last_block_count : main_block_count;
        recv_block_count = (r == (world - 1)) ? last_block_count : main_block_count;
        send_block_size = send_block_count * ccl_dtype.size();
        recv_block_size = recv_block_count * ccl_dtype.size();

        // recv -> reduce -> send
        if (iter == 0)
            send_ptr = (char *)send_buf + s * main_block_size;
        // for the last iteration, reduce directly to the recv_buf
        if (iter == world - 2) {
            recv_ptr = !in_place ? recv_buf : buffers[index];
            out_ptr = recv_buf;
        }
        else {
            recv_ptr = buffers[index];
            out_ptr = recv_ptr;
        }

#ifdef PRINT_TIMING
        cpu_timer<1> ctimer;
        ctimer.start(0);
#endif // PRINT_TIMING

        atl_req_t send_req, recv_req;
        ATL_CALL_THROW_IF_ERROR(atl_comm->recv(ep_idx, recv_ptr, recv_block_size, left, tag, recv_req));
        ATL_CALL_THROW_IF_ERROR(atl_comm->send(ep_idx, send_ptr, send_block_size, right, tag, send_req));
        ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));

#ifdef PRINT_TIMING
        ctimer.stop(0);
        printf("[%d] sendrecv takes: iter: %d size: %ld  takes: %f us\n",
               rank,
               iter,
               recv_block_size,
               ctimer.get_us(0));
        // do reduce on recv_ptr and send_buf + r
        // sycl kernel or MPI_Reduce_local
        ctimer.start(0);
#endif // PRINT_TIMING

        char *send_offset_ptr = (char *)send_buf + r * main_block_size;
        bool use_full_vector =
            can_use_full_vector(send_offset_ptr, recv_ptr, recv_block_count, ccl_dtype.size()) &&
            (uintptr_t)out_ptr % 4 == 0;
        if (use_full_vector) {
            constexpr int vec_size = get_num_elements<T, 8, true>();
            sycl_e = reduce_invoke.template operator()<vec_size, 16>(
                send_offset_ptr, recv_ptr, out_ptr, recv_block_count, {});
        }
        else {
            constexpr int vec_size = get_num_elements<T, 8, false>();
            sycl_e = reduce_invoke.template operator()<vec_size, 64>(
                send_offset_ptr, recv_ptr, out_ptr, recv_block_count, {});
        }
        sycl_e.wait();

#ifdef PRINT_TIMING
        ctimer.stop(0);
        printf("[%d] Reduce_local iter: %d recv_block_count: %ld takes: %f us\n",
               rank,
               iter,
               recv_block_count,
               ctimer.get_us(0));
#endif // PRINT_TIMING

        // wait for send to finish is not time critical
        ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));

        // shift buffers
        send_ptr = recv_ptr;
        index = index ^ 1;

        // next iteration
        s = r;
        r = (r - 1 + world) % world;
        iter++;
    }

    if (world > 2 || in_place) {
        if (to_free) {
            sycl::free(buffers[0], q);
        }
        else {
            comm->put_scaleout_device_buf(buffers[0]);
        }
    }

    return sycl_e;
}

// blocking algorithm calls MPI GPU pipelining directly
template <typename T>
inline sycl::event reduce_scatter_ring_blocking(sycl::queue &q,
                                                const void *send_buf,
                                                void *recv_buf,
                                                size_t recv_count,
                                                datatype dtype,
                                                reduction reduction,
                                                ccl_comm *comm,
                                                const vector_class<event> &deps,
                                                bool &done) {
    size_t total_count = recv_count * comm->size();
    return reduce_scatter_ring_blocking_impl<T>(
        q, send_buf, recv_buf, total_count, dtype, reduction, comm, deps, done);
}

template <typename T>
inline sycl::event reduce_scatter_ring_nonblocking_impl(sycl::queue &q,
                                                        const void *send_buf,
                                                        void *recv_buf,
                                                        size_t count,
                                                        datatype dtype,
                                                        reduction reduction,
                                                        ccl_comm *comm,
                                                        const vector_class<event> &deps,
                                                        bool original_deps,
                                                        sycl_reduce_scatter_tune_attr tune_attr,
                                                        bool &done) {
    sycl::event sycl_e;
    int world = comm->size();
    int rank = comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    // tuning parameters
    size_t pipeline_chunk_size = tune_attr.pipeline_chunk_size;

    // prepare ring block counts/sizes, the last block handles the remainder
    size_t main_block_count = count / world;
    size_t last_block_count = main_block_count + count % world;
    size_t main_block_size = main_block_count * ccl_dtype.size();
    size_t last_block_size = last_block_count * ccl_dtype.size();

    bool in_place =
        ccl::is_reduce_scatter_inplace(send_buf, recv_buf, main_block_count, ccl_dtype.size(), rank, world);

    // to make sure that buffer con hold the larges possible block
    size_t buf_size = last_block_size;
    // align an allocation size up to 4 bytes, to have a better vector reduction
    int typesize = std::max(4, (int)ccl_dtype.size());
    size_t aligned_buf_size = (buf_size + typesize - 1) / typesize * typesize;
    void *buffers[2] = { NULL };
    int to_free = 0;
    if (world > 2 || in_place) {
        if (aligned_buf_size <= comm->get_scaleout_device_buf_size() / 2) {
            buffers[0] = comm->get_scaleout_device_buf(q);
        }
        else {
            buffers[0] = sycl::malloc_device(aligned_buf_size * 2, q);
            to_free = 1;
            LOG_WARN("reduce_scatter_ring_nonblocking device buffer not big enough");
        }
        buffers[1] = (char *)buffers[0] + aligned_buf_size;
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    // in case deps is from user, set up wait event for the out of order queue
    if (original_deps) {
        sycl_e = submit_wait_on_events(q, dep_events);
        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));
    }

    // use an out-of-order queue
    sycl::queue q_worker(q.get_device());

    auto reduce_invoke = [=]<int VS, int SGS>(sycl::queue &q,
                                              void *in1,
                                              void *in2,
                                              void *out,
                                              size_t reduce_count,
                                              std::vector<sycl::event> l_dep_events) {
        constexpr int vec_size = VS, wg_size = SGS, sg_size = SGS;
        const size_t kernel_threads = reduce_count / vec_size + reduce_count % vec_size;
        const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
        return q.submit([=](sycl::handler &h) {
            h.depends_on(l_dep_events);
            h.parallel_for(sycl::nd_range<1>(kernel_size, wg_size),
                           //[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                           [=](sycl::nd_item<1> it) {
                               reduce_pair<T, vec_size>(in1, in2, out, reduce_count, it);
                           });
        });
    };

    // left and right neighbour
    int right = (rank + 1) % world;
    int left = (rank - 1 + world) % world;

    // start indexes
    int s = (rank - 1 + world) % world;
    int r = (s - 1 + world) % world;

    // send/recv pointers
    void *send_ptr, *recv_ptr, *out_ptr;
    size_t send_block_count, recv_block_count;
    size_t send_block_size, recv_block_size;

    // tag creation
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t pt2pt_sched_id = atl_comm->tag_creator->get_pt2pt_sched_id();
    int64_t tag = atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), pt2pt_sched_id, 0);

    // calculate the number of chunks required for pipeline
    size_t nchunks;
    pipe_prep(main_block_count, last_block_count, ccl_dtype.size(), pipeline_chunk_size, nchunks);
    LOG_DEBUG("Reduce_scatter ring block_count: ",
              main_block_count,
              " nchunks: ",
              nchunks,
              " chunk_size: ",
              pipeline_chunk_size);

    int index = 0;
    int iter = 0;
    sycl::event sendrecv_e;
    while (iter < world - 1) {
        // find the right send/recv block size
        send_block_count = (s == (world - 1)) ? last_block_count : main_block_count;
        recv_block_count = (r == (world - 1)) ? last_block_count : main_block_count;
        send_block_size = send_block_count * ccl_dtype.size();
        recv_block_size = recv_block_count * ccl_dtype.size();

        // recv -> reduce -> send
        if (iter == 0) {
            send_ptr = (char *)send_buf + s * main_block_size;
        }
        // for the last iteration, reduce directly to the recv_buf
        if (iter == world - 2) {
            recv_ptr = !in_place ? recv_buf : buffers[index];
            out_ptr = recv_buf;
        }
        else {
            recv_ptr = buffers[index];
            out_ptr = recv_ptr;
        }

        sendrecv_e = pipe_sendrecv(q_worker,
                                   send_ptr,
                                   send_block_count,
                                   right,
                                   tag,
                                   recv_ptr,
                                   recv_block_count,
                                   left,
                                   tag,
                                   dtype,
                                   nchunks,
                                   comm,
                                   dep_events,
                                   ccl::global_data::env().sycl_enable_pipeline_gpu_rdma); // GPU RDMA

        // sycl kernel or MPI_Reduce_local
        char *send_offset_ptr = (char *)send_buf + r * main_block_size;
        bool use_full_vector =
            can_use_full_vector(send_offset_ptr, recv_ptr, recv_block_count, ccl_dtype.size()) &&
            (uintptr_t)out_ptr % 4 == 0;
        if (use_full_vector) {
            constexpr int vec_size = get_num_elements<T, 8, true>();
            sycl_e = reduce_invoke.template operator()<vec_size, 16>(
                q_worker, send_offset_ptr, recv_ptr, out_ptr, recv_block_count, { sendrecv_e });
        }
        else {
            constexpr int vec_size = get_num_elements<T, 8, false>();
            sycl_e = reduce_invoke.template operator()<vec_size, 64>(
                q_worker, send_offset_ptr, recv_ptr, out_ptr, recv_block_count, { sendrecv_e });
        }

        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));

        // shift buffers
        send_ptr = recv_ptr;
        index = index ^ 1;

        // next iteration
        s = r;
        r = (r - 1 + world) % world;
        iter++;
    }

    // submit to in-order queue
    sycl_e = submit_wait_on_events(q, dep_events);

    if (world > 2 || in_place) {
        if (to_free) {
            sycl_e = q.submit([=](sycl::handler &h) {
                h.depends_on(sycl_e);
                h.host_task([=]() {
                    sycl::free(buffers[0], q);
                });
            });
        }
        else {
            comm->put_scaleout_device_buf(buffers[0]);
        }
    }

    done = true;
    return sycl_e;
}

template <typename T>
inline sycl::event reduce_scatter_ring_nonblocking(sycl::queue &q,
                                                   const void *send_buf,
                                                   void *recv_buf,
                                                   size_t recv_count,
                                                   datatype dtype,
                                                   reduction reduction,
                                                   ccl_comm *comm,
                                                   const vector_class<event> &deps,
                                                   bool original_deps,
                                                   sycl_reduce_scatter_tune_attr &tune_attr,
                                                   bool &done) {
    size_t total_count = recv_count * comm->size();

    return reduce_scatter_ring_nonblocking_impl<T>(
        q, send_buf, recv_buf, total_count, dtype, reduction, comm, deps, original_deps, tune_attr, done);
}

inline sycl::event reduce_scatter_ring(sycl::queue &q,
                                       const void *send_buf,
                                       void *recv_buf,
                                       size_t recv_count,
                                       datatype dtype,
                                       reduction reduction,
                                       ccl_comm *comm,
                                       const vector_class<event> &deps,
                                       bool original_deps,
                                       sycl_reduce_scatter_tune_attr &tune_attr,
                                       bool &done) {
    auto lambda = [&]<typename T>() {
        if (!ccl::global_data::env().enable_op_sync) {
            return reduce_scatter_ring_nonblocking<T>(
                q, send_buf, recv_buf, recv_count, dtype, reduction, comm, deps, original_deps, tune_attr, done);
        }
        else {
            return reduce_scatter_ring_blocking<T>(
                q, send_buf, recv_buf, recv_count, dtype, reduction, comm, deps, done);
        }
    };

    return invoke_scaleout(lambda, dtype);
}

} // namespace v1
} // namespace ccl
