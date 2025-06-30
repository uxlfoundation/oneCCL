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

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_ring.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

namespace ccl {
namespace v1 {

template <typename T>
sycl::event allreduce_sycl_rabenseifner_blocking(sycl::queue &q,
                                                 const void *send_buf,
                                                 void *recv_buf,
                                                 size_t count,
                                                 datatype dtype,
                                                 reduction reduction,
                                                 ccl_comm *comm,
                                                 const ccl::vector_class<ccl::event> &deps,
                                                 bool &done) {
    sycl::event sycl_e;
    int newrank;

    int world = comm->size();
    int rank = comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    int pow2 = comm->pof2();
    if (count < pow2) {
        LOG_WARN("allreduce_sycl_rabenseifner_blocking: count is too small");
        done = false;
        return sycl_e;
    }

    // blocking
    q.wait();

    // to make sure that buffer con hold the largest possible block
    // align an allocation size up to 4 bytes, to have a better vector reduction
    size_t total_size = count * ccl_dtype.size();
    void *tmp_buf = NULL;
    int to_free = 0;
    if (total_size <= comm->get_scaleout_device_buf_size()) {
        tmp_buf = comm->get_scaleout_device_buf(q);
    }
    else {
        LOG_WARN("allreduce_recursive_halving_blocking device buffer is not big enough",
                 total_size,
                 " ",
                 comm->get_scaleout_device_buf_size());
        tmp_buf = sycl::malloc_device(total_size, q);
        to_free = 1;
    }
    //fprintf(stderr, "calling allreduce_sycl_reduce_scatter_allgather_blocking count: %ld\n", count);

    if (send_buf != recv_buf) {
        sycl_e = q.submit([=](sycl::handler &h) {
            h.memcpy(recv_buf, send_buf, total_size);
        });
        sycl_e.wait();
    }

    int rem = world - pow2;

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

    int ep_idx = 0;
    // tag creation
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t pt2pt_sched_id = atl_comm->tag_creator->get_pt2pt_sched_id();
    int64_t tag =
        atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), pt2pt_sched_id, 0);

    // ranks are divided into two groups, first 2 * rem, and the rest.
    // In the first group, even rank send its data to odd rank, and they do
    // not participate the rabenseifner.
    // After this phase, the odd ranks of the first group and the second
    // group go to the main algorithm.
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            atl_req_t send_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, recv_buf, total_size, rank + 1, tag, send_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            newrank = -1;
        }
        else {
            atl_req_t recv_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->recv(ep_idx, tmp_buf, total_size, rank - 1, tag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));

            // tmp buffers are aligned
            bool use_full_vector = can_use_full_vector(tmp_buf, recv_buf, 1, 4);
            if (use_full_vector) {
                constexpr int vec_size = get_num_elements<T, 8, true>();
                sycl_e = reduce_invoke.template operator()<vec_size, 16>(
                    q, tmp_buf, recv_buf, recv_buf, count, { sycl_e });
            }
            else {
                constexpr int vec_size = get_num_elements<T, 8, false>();
                sycl_e = reduce_invoke.template operator()<vec_size, 64>(
                    q, tmp_buf, recv_buf, recv_buf, count, { sycl_e });
            }
            sycl_e.wait();
            newrank = rank / 2;
        }
    }
    else {
        newrank = rank - rem;
    }

    if (newrank != -1) {
        std::vector<size_t> newcnts(pow2, count / pow2);
        std::vector<size_t> newdisps(pow2);
        int re = count % pow2;
        if (re > 0) {
            for (int i = 0; i < re; i++)
                newcnts[i] += 1;
        }

        newdisps[0] = 0;
        for (int i = 1; i < pow2; i++) {
            newdisps[i] = newdisps[i - 1] + newcnts[i - 1];
        }

        int mask = 0x1;
        int send_idx = 0, recv_idx = 0;
        int last_idx = pow2;
        while (mask < pow2) {
            int newdst = newrank ^ mask;
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                // recv first half, send the second half
                send_idx = recv_idx + pow2 / (mask * 2);
                for (int i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
                for (int i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
            }
            else {
                // send first half, recv the second half
                recv_idx = send_idx + pow2 / (mask * 2);
                for (int i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (int i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }

            //fprintf(stderr, "RS Rank %d, mask: %x newrank: %d dst: %d send_idx %d, recv_idx %d, last_idx %d send_cnt %ld, recv_cnt %ld\n", rank, mask, newrank, dst, send_idx, recv_idx, last_idx, send_cnt, recv_cnt);
            atl_req_t send_req, recv_req;
            char *sbuf = (char *)recv_buf + newdisps[send_idx] * ccl_dtype.size();
            char *tbuf = (char *)tmp_buf + newdisps[recv_idx] * ccl_dtype.size();
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, sbuf, send_cnt * ccl_dtype.size(), dst, tag, send_req));
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->recv(ep_idx, tbuf, recv_cnt * ccl_dtype.size(), dst, tag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));

            char *rbuf = (char *)recv_buf + newdisps[recv_idx] * ccl_dtype.size();
            bool use_full_vector = can_use_full_vector(tbuf, rbuf, 4, 4);
            if (use_full_vector) {
                constexpr int vec_size = get_num_elements<T, 8, true>();
                sycl_e = reduce_invoke.template operator()<vec_size, 16>(
                    q, tbuf, rbuf, rbuf, recv_cnt, { sycl_e });
            }
            else {
                constexpr int vec_size = get_num_elements<T, 8, false>();
                sycl_e = reduce_invoke.template operator()<vec_size, 64>(
                    q, tbuf, rbuf, rbuf, recv_cnt, { sycl_e });
            }
            sycl_e.wait();

            send_idx = recv_idx;
            mask <<= 1;

            if (mask < pow2)
                last_idx = recv_idx + pow2 / mask;
        }

        // allgather
        mask >>= 1;
        int half = 1;
        while (mask > 0) {
            int newdst = newrank ^ mask;
            /* find real rank of dest */
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pow2 / 2)
                    last_idx = last_idx + half;

                recv_idx = send_idx + half;
                for (int i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (int i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }
            else {
                recv_idx = send_idx - half;
                for (int i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
                for (int i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
            }

            //fprintf(stderr, "Allgather Rank %d, mask: %x newrank: %d dst: %d send_idx %d, recv_idx %d, last_idx %d, send_cnt %ld, recv_cnt %ld\n", rank, mask, newrank, dst, send_idx, recv_idx, last_idx, send_cnt, recv_cnt);
            atl_req_t send_req, recv_req;
            char *sbuf = (char *)recv_buf + newdisps[send_idx] * ccl_dtype.size();
            char *rbuf = (char *)recv_buf + newdisps[recv_idx] * ccl_dtype.size();
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, sbuf, send_cnt * ccl_dtype.size(), dst, tag, send_req));
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->recv(ep_idx, rbuf, recv_cnt * ccl_dtype.size(), dst, tag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
            half <<= 1;
        }
    }

    if (rank < 2 * rem) {
        if (rank % 2) {
            atl_req_t send_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, recv_buf, total_size, rank - 1, tag, send_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
        }
        else {
            atl_req_t recv_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->recv(ep_idx, recv_buf, total_size, rank + 1, tag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
        }
    }

    if (to_free) {
        sycl::free(tmp_buf, q);
    }
    else {
        comm->put_scaleout_device_buf(tmp_buf);
    }

    done = true;
    return sycl_e;
}

template <typename T>
sycl::event allreduce_sycl_rabenseifner_nonblocking(sycl::queue &q,
                                                    const void *send_buf,
                                                    void *recv_buf,
                                                    size_t count,
                                                    datatype dtype,
                                                    reduction reduction,
                                                    ccl_comm *comm,
                                                    const ccl::vector_class<ccl::event> &deps,
                                                    bool original_deps,
                                                    sycl_allreduce_tune_attr tune_attr,
                                                    bool &done) {
    sycl::event sycl_e;
    int newrank;

    int world = comm->size();
    int rank = comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    // tuning parameters
    size_t pipeline_chunk_size = tune_attr.pipeline_chunk_size;

    int pow2 = comm->pof2();
    if (count < pow2) {
        LOG_WARN("allreduce_sycl_rabenseifner_nonblocking: count is too small");
        done = false;
        return sycl_e;
    }

    size_t total_size = count * ccl_dtype.size();
    void *tmp_buf = NULL;
    int to_free = 0;
    if (total_size <= comm->get_scaleout_device_buf_size()) {
        tmp_buf = comm->get_scaleout_device_buf(q);
    }
    else {
        LOG_WARN("allreduce_recursive_halving_blocking device buffer is not big enough:",
                 total_size,
                 " ",
                 comm->get_scaleout_device_buf_size());
        tmp_buf = sycl::malloc_device(total_size, q);
        to_free = 1;
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    if (original_deps) {
        sycl_e = submit_wait_on_events(q, dep_events);
        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));
    }
    else {
        CCL_THROW_IF_NOT(dep_events.size() > 0,
                         "allreduce_sycl_rabenseifner_nonblocking: empty dependence");
    }
    if (send_buf != recv_buf) {
        sycl_e = q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.memcpy(recv_buf, send_buf, total_size);
        });
        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));
    }

    int rem = world - pow2;

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

    int ep_idx = 0;
    // tag creation
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t pt2pt_sched_id = atl_comm->tag_creator->get_pt2pt_sched_id();
    int64_t tag =
        atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), pt2pt_sched_id, 0);

    // ranks are divided into two groups, first 2 * rem, and the rest.
    // In the first group, even rank send its data to odd rank, and they do
    // not participate the rabenseifner.
    // After this phase, the odd ranks of the first group and the second
    // group go to the main algorithm.
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            sycl_e = gpu_send_plain(q, recv_buf, count, rank + 1, tag, dtype, comm, dep_events);
            newrank = -1;
        }
        else {
            sycl_e = gpu_recv_plain(q, tmp_buf, count, rank - 1, tag, dtype, comm, dep_events);
            // tmp buffers are aligned
            bool use_full_vector = can_use_full_vector(tmp_buf, recv_buf, 1, 4);
            if (use_full_vector) {
                constexpr int vec_size = get_num_elements<T, 8, true>();
                sycl_e = reduce_invoke.template operator()<vec_size, 16>(
                    q, tmp_buf, recv_buf, recv_buf, count, { sycl_e });
            }
            else {
                constexpr int vec_size = get_num_elements<T, 8, false>();
                sycl_e = reduce_invoke.template operator()<vec_size, 64>(
                    q, tmp_buf, recv_buf, recv_buf, count, { sycl_e });
            }
            newrank = rank / 2;
        }
        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));
    }
    else {
        newrank = rank - rem;
    }

    if (newrank != -1) {
        // out-of-order queue
        sycl::queue q_worker(q.get_device());

        std::vector<size_t> newcnts(pow2, count / pow2);
        std::vector<size_t> newdisps(pow2);
        int re = count % pow2;
        if (re > 0) {
            for (int i = 0; i < re; i++)
                newcnts[i] += 1;
        }

        newdisps[0] = 0;
        for (int i = 1; i < pow2; i++) {
            newdisps[i] = newdisps[i - 1] + newcnts[i - 1];
        }

        // calculate the number of chunks required for pipeline
        size_t nchunks;
        //pipe_prep(count / pow2, count / 2 + count % pow2, ccl_dtype.size(), nchunks);

        int mask = 0x1;
        int send_idx = 0, recv_idx = 0;
        int last_idx = pow2;
        while (mask < pow2) {
            int newdst = newrank ^ mask;
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                // recv first half, send the second half
                send_idx = recv_idx + pow2 / (mask * 2);
                for (int i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
                for (int i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
            }
            else {
                // send first half, recv the second half
                recv_idx = send_idx + pow2 / (mask * 2);
                for (int i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (int i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }

            pipe_prep(send_cnt > recv_cnt ? recv_cnt : send_cnt,
                      send_cnt > recv_cnt ? send_cnt : recv_cnt,
                      ccl_dtype.size(),
                      pipeline_chunk_size,
                      nchunks);
            LOG_DEBUG("Allreduce rabenseifner RS send count: ",
                      send_cnt,
                      " recv count: ",
                      recv_cnt,
                      " nchunks: ",
                      nchunks,
                      " pipeline_chunk_size: ",
                      pipeline_chunk_size);

            //fprintf(stderr, "Rank %d, newrank: %d dst: %d send_idx %d, recv_idx %d, send_size %ld, recv_size %ld, last_idx %d\n", rank, newrank, dst, send_idx, recv_idx, send_size, recv_size, last_idx);

            void *sbuf = (char *)recv_buf + newdisps[send_idx] * ccl_dtype.size();
            void *tbuf = (char *)tmp_buf + newdisps[recv_idx] * ccl_dtype.size();
            sycl_e = pipe_sendrecv(q_worker,
                                   sbuf,
                                   send_cnt,
                                   dst,
                                   tag,
                                   tbuf,
                                   recv_cnt,
                                   dst,
                                   tag,
                                   dtype,
                                   nchunks,
                                   comm,
                                   dep_events,
                                   ccl::global_data::env().sycl_enable_pipeline_gpu_rdma);

            char *rbuf = (char *)recv_buf + newdisps[recv_idx] * ccl_dtype.size();
            bool use_full_vector = can_use_full_vector(tbuf, rbuf, 4, 4);
            if (use_full_vector) {
                constexpr int vec_size = get_num_elements<T, 8, true>();
                sycl_e = reduce_invoke.template operator()<vec_size, 16>(
                    q_worker, tbuf, rbuf, rbuf, recv_cnt, { sycl_e });
            }
            else {
                constexpr int vec_size = get_num_elements<T, 8, false>();
                sycl_e = reduce_invoke.template operator()<vec_size, 64>(
                    q_worker, tbuf, rbuf, rbuf, recv_cnt, { sycl_e });
            }
            dep_events.clear();
            dep_events.push_back(std::move(sycl_e));

            send_idx = recv_idx;
            mask <<= 1;

            if (mask < pow2)
                last_idx = recv_idx + pow2 / mask;
        }

        // allgather
        mask >>= 1;
        int half = 1;
        while (mask > 0) {
            int newdst = newrank ^ mask;
            /* real rank of dest */
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pow2 / 2)
                    last_idx = last_idx + half;

                recv_idx = send_idx + half;
                for (int i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (int i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }
            else {
                recv_idx = send_idx - half;
                for (int i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
                for (int i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
            }

            pipe_prep(send_cnt > recv_cnt ? recv_cnt : send_cnt,
                      send_cnt > recv_cnt ? send_cnt : recv_cnt,
                      ccl_dtype.size(),
                      pipeline_chunk_size,
                      nchunks);
            LOG_DEBUG("Allreduce rabenseifner AG send count: ",
                      send_cnt,
                      " recv count: ",
                      recv_cnt,
                      " nchunks: ",
                      nchunks,
                      " pipeline_chunk_size: ",
                      pipeline_chunk_size);

            void *sbuf = (char *)recv_buf + newdisps[send_idx] * ccl_dtype.size();
            void *rbuf = (char *)recv_buf + newdisps[recv_idx] * ccl_dtype.size();
            sycl_e = pipe_sendrecv(q_worker,
                                   sbuf,
                                   send_cnt,
                                   dst,
                                   tag,
                                   rbuf,
                                   recv_cnt,
                                   dst,
                                   tag,
                                   dtype,
                                   nchunks,
                                   comm,
                                   dep_events,
                                   ccl::global_data::env().sycl_enable_pipeline_gpu_rdma);
            dep_events.clear();
            dep_events.push_back(std::move(sycl_e));

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
            half <<= 1;
        }
    }

    if (rank < 2 * rem) {
        if (rank % 2) {
            sycl_e = gpu_send_plain(q, recv_buf, count, rank - 1, tag, dtype, comm, dep_events);
        }
        else {
            sycl_e = gpu_recv_plain(q, recv_buf, count, rank + 1, tag, dtype, comm, dep_events);
        }
        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));
    }

    // submit to in-order queue
    sycl_e = submit_wait_on_events(q, dep_events);

    if (to_free) {
        sycl_e = q.submit([=](sycl::handler &h) {
            h.depends_on(sycl_e);
            h.host_task([=]() {
                sycl::free(tmp_buf, q);
            });
        });
    }
    else {
        comm->put_scaleout_device_buf(tmp_buf);
    }

    done = true;
    return sycl_e;
}

inline sycl::event allreduce_scaleout_sycl_rabenseifner(sycl::queue &q,
                                                        const void *send_buf,
                                                        void *recv_buf,
                                                        size_t count,
                                                        datatype dtype,
                                                        reduction reduction,
                                                        ccl_comm *comm,
                                                        const ccl::vector_class<ccl::event> &deps,
                                                        bool original_deps,
                                                        sycl_allreduce_tune_attr tune_attr,
                                                        bool &done) {
    auto lambda = [&]<typename T>() {
        if (ccl::global_data::env().enable_op_sync) {
            return allreduce_sycl_rabenseifner_blocking<T>(
                q, send_buf, recv_buf, count, dtype, reduction, comm, deps, done);
        }
        else {
            return allreduce_sycl_rabenseifner_nonblocking<T>(q,
                                                              send_buf,
                                                              recv_buf,
                                                              count,
                                                              dtype,
                                                              reduction,
                                                              comm,
                                                              deps,
                                                              original_deps,
                                                              tune_attr,
                                                              done);
        }
    };

    return invoke_scaleout(lambda, dtype);
}

} // namespace v1
} // namespace ccl
