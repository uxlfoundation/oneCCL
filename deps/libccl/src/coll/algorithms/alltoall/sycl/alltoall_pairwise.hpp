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

#include "proxy/proxy_types.hpp"
#include "proxy/host_initiated/cmd_persistent_kernel.hpp"

// pairwise exchange for all ranks inside a numa comm
static sycl::event exchange_numa_self(sycl::queue q,
                                      ccl_comm* global_comm,
                                      int rank,
                                      int my_numa_id,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t size) {
    ccl_comm* numa_comm = global_comm->get_numa_comm().get();
    int numa_size = numa_comm->size();
    int ep_idx = 0;
    std::shared_ptr<atl_base_comm> atl_comm = global_comm->get_atl_comm();
    ccl_sched_id_t sched_id = global_comm->get_sched_id(true, false);

    sycl::event out_e = q.submit([=](sycl::handler& h) {
        h.host_task([=]() {
            int64_t tag;
            int src, dst;
            std::vector<atl_req_t> reqs(numa_size * 2);
            int nreqs = 0;
            for (int i = 0; i < numa_size; i++) {
                src = rank ^ i;
                dst = src;

                tag = atl_comm->tag_creator->create(
                    rank, global_comm->get_comm_id(), sched_id, 1 + i);
                ATL_CALL_THROW_IF_ERROR(atl_comm->send(
                    ep_idx, (char*)send_buf + dst * size, size, dst, tag, reqs[nreqs++]));
                tag =
                    atl_comm->tag_creator->create(src, global_comm->get_comm_id(), sched_id, 1 + i);
                ATL_CALL_THROW_IF_ERROR(atl_comm->recv(
                    ep_idx, (char*)recv_buf + src * size, size, src, tag, reqs[nreqs++]));
            }
            for (int i = 0; i < nreqs; i++) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, reqs[i]));
            }
        });
    });
    return out_e;
}

// pairwise exchange between two numas
// suitable to be called from a host_task
static void exchange_numa_fn(sycl::queue q,
                             ccl_comm* global_comm,
                             int rank,
                             int my_numa_id,
                             int dst_numa_id,
                             const void* send_buf,
                             void* recv_buf,
                             size_t size,
                             std::vector<atl_req_t>& reqs,
                             int& nreqs) {
    ccl_comm* numa_comm = global_comm->get_numa_comm().get();
    int numa_size = numa_comm->size();
    int ep_idx = 0;
    std::shared_ptr<atl_base_comm> atl_comm = global_comm->get_atl_comm();
    ccl_sched_id_t sched_id = global_comm->get_sched_id(true, false);

    int64_t tag;
    int src, dst;
    int peer_first_rank = dst_numa_id * numa_size;
    int my_first_rank = my_numa_id * numa_size;
    int my_local_rank = rank - my_first_rank;
    int lrank = my_numa_id <= dst_numa_id ? my_local_rank : numa_size + my_local_rank;
    int peer_first_lrank = my_numa_id <= dst_numa_id ? numa_size : 0;
    for (int i = numa_size; i < numa_size * 2; i++) {
        int dst_lrank = lrank ^ i;
        src = peer_first_rank + dst_lrank - peer_first_lrank;
        dst = src;

        tag = atl_comm->tag_creator->create(src, global_comm->get_comm_id(), sched_id, 1 + i);
        ATL_CALL_THROW_IF_ERROR(
            atl_comm->recv(ep_idx, (char*)recv_buf + src * size, size, src, tag, reqs[nreqs++]));

        tag = atl_comm->tag_creator->create(rank, global_comm->get_comm_id(), sched_id, 1 + i);
        ATL_CALL_THROW_IF_ERROR(
            atl_comm->send(ep_idx, (char*)send_buf + dst * size, size, dst, tag, reqs[nreqs++]));
    }
}

// pairwise exchange between two numas
// block until done
static sycl::event exchange_numa(sycl::queue& q,
                                 ccl_comm* global_comm,
                                 int rank,
                                 int my_numa_id,
                                 int dst_numa_id,
                                 const void* send_buf,
                                 void* recv_buf,
                                 size_t size) {
    ccl_comm* numa_comm = global_comm->get_numa_comm().get();
    int numa_size = numa_comm->size();
    int ep_idx = 0;
    std::shared_ptr<atl_base_comm> atl_comm = global_comm->get_atl_comm();
    ccl_sched_id_t sched_id = global_comm->get_sched_id(true, false);

    sycl::event out_e = q.submit([=](sycl::handler& h) {
        h.host_task([=]() {
            int64_t tag;
            int src, dst;
            int peer_first_rank = dst_numa_id * numa_size;
            int my_first_rank = my_numa_id * numa_size;
            int my_local_rank = rank - my_first_rank;
            int lrank = my_numa_id <= dst_numa_id ? my_local_rank : numa_size + my_local_rank;
            int peer_first_lrank = my_numa_id <= dst_numa_id ? numa_size : 0;
            std::vector<atl_req_t> reqs(numa_size * 2);
            int nreqs = 0;
            for (int i = numa_size; i < numa_size * 2; i++) {
                int dst_lrank = lrank ^ i;
                src = peer_first_rank + dst_lrank - peer_first_lrank;
                dst = src;
                tag =
                    atl_comm->tag_creator->create(src, global_comm->get_comm_id(), sched_id, 1 + i);
                ATL_CALL_THROW_IF_ERROR(atl_comm->recv(
                    ep_idx, (char*)recv_buf + src * size, size, src, tag, reqs[nreqs++]));

                tag = atl_comm->tag_creator->create(
                    rank, global_comm->get_comm_id(), sched_id, 1 + i);
                ATL_CALL_THROW_IF_ERROR(atl_comm->send(
                    ep_idx, (char*)send_buf + dst * size, size, dst, tag, reqs[nreqs++]));
            }
            for (int i = 0; i < nreqs; i++) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, reqs[i]));
            }
        });
    });
    return out_e;
}

// numa-level pairwise exchange
ccl::event alltoall_sycl_numa_pairwise_rdma(sycl::queue& q,
                                            const void* send_buf,
                                            void* recv_buf,
                                            size_t count,
                                            ccl::datatype dtype,
                                            ccl_comm* comm,
                                            ccl_stream* global_stream,
                                            const ccl::vector_class<ccl::event>& deps,
                                            bool batch_mode,
                                            bool& done) {
    ccl_comm* node_comm = comm->get_node_comm().get();
    ccl_comm* numa_comm = comm->get_numa_comm().get();
    uint32_t world = comm->size();
    uint32_t rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();
    int ep_idx = 0;
    int pof2 = is_pof2(world);
    sycl::event sycl_e;

    done = true;

    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    int numa_size = numa_comm->size();
    int num_numas = world / numa_size;
    int my_numa_id = rank / numa_size;
    for (int i = 0; i < num_numas; i++) {
        int dst_numa_id = my_numa_id ^ i;
        if (dst_numa_id == my_numa_id)
            sycl_e = exchange_numa_self(q, comm, rank, my_numa_id, send_buf, recv_buf, size);
        else
            sycl_e =
                exchange_numa(q, comm, rank, my_numa_id, dst_numa_id, send_buf, recv_buf, size);
        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));
        sycl_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.host_task([=]() {
                atl_req_t req;
                atl_comm->barrier(ep_idx, req);
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
            });
        });
    }

    return ccl::event::create_from_native(sycl_e);
}

// host_task scale-out exchange of alltoall_sycl_numa_pairwise_rdma_oneshot:
// pairwise to all ranks outside of the numa comm, barrier after each chunk
// TODO: may be merged with alltoall_scatter_exchange_host
sycl::event alltoall_numa_pairwise_exchange_host(sycl::queue& q,
                                                 const void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 ccl_comm* comm,
                                                 int peer_start,
                                                 const std::vector<sycl::event>& dep_events) {
    uint32_t world = comm->size();
    uint32_t rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();
    int ep_idx = 0;
    int pof2 = is_pof2(world);
    sycl::event sycl_e;

    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t sched_id = comm->get_sched_id(true, false);

    /* Do the pairwise exchanges */
    size_t block = size;
    if (size > 512 * 1024 * 1024) {
        block /= 2;
    }
    size_t progress = 0;
    int c = 0;
    while (progress < size) {
        size_t tocopy = progress + block < size ? block : size - progress;
        sycl_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.host_task([=]() {
                std::vector<atl_req_t> reqs(world * 2);
                int nreqs = 0;
                // posed all recvs
                for (int i = peer_start; i < world; i++) {
                    int src;
                    if (pof2) {
                        src = rank ^ i;
                    }
                    else {
                        src = (rank - i + world) % world;
                    }

                    int64_t recv_tag =
                        atl_comm->tag_creator->create(src, comm->get_comm_id(), sched_id, c + 1);
                    ATL_CALL_THROW_IF_ERROR(atl_comm->recv(ep_idx,
                                                           (char*)recv_buf + src * size + progress,
                                                           tocopy,
                                                           src,
                                                           recv_tag,
                                                           reqs[nreqs++]));
                }

                // posed all sends
                for (int i = peer_start; i < world; i++) {
                    int dst;
                    if (pof2) {
                        dst = rank ^ i;
                    }
                    else {
                        dst = (rank + i) % world;
                    }

                    int64_t send_tag =
                        atl_comm->tag_creator->create(rank, comm->get_comm_id(), sched_id, c + 1);
                    ATL_CALL_THROW_IF_ERROR(atl_comm->send(ep_idx,
                                                           (char*)send_buf + dst * size + progress,
                                                           tocopy,
                                                           dst,
                                                           send_tag,
                                                           reqs[nreqs++]));
                }

                // waitall
                for (int i = 0; i < nreqs; i++) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, reqs[i]));
                }
                // barrier
                if (progress + tocopy < size || 1) {
                    atl_req_t req;
                    ATL_CALL_THROW_IF_ERROR(atl_comm->barrier(ep_idx, req));
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
                }
            });
        });
        progress += tocopy;
        c++;
    }

    return sycl_e;
}

// pairwise exchange starts in p2p_comm comm
// and then pairwise to all other ranks outside of the p2p_comm comm
//
// with use_proxy, the scale-out exchange runs on the proxy thread instead of in a
// host_task on the caller's queue
ccl::event alltoall_sycl_numa_pairwise_rdma_oneshot(sycl::queue& q,
                                                    const void* send_buf,
                                                    void* recv_buf,
                                                    size_t count,
                                                    ccl::datatype dtype,
                                                    ccl_comm* comm,
                                                    std::shared_ptr<ccl_comm> p2p_comm,
                                                    std::shared_ptr<ccl_comm> r2r_comm,
                                                    ccl_stream* global_stream,
                                                    const ccl::vector_class<ccl::event>& deps,
                                                    bool batch_mode,
                                                    bool use_proxy,
                                                    bool& done) {
    uint32_t world = comm->size();
    uint32_t rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();

    done = true;

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    int p2p_size = p2p_comm->size();

    // The exchange skips peers i < peer_start (== p2p_size) of its `rank ^ i` enumeration,
    // assuming those are exactly the ranks the scale-up phase handled. That only holds for
    // a power-of-two world (`rank ^ i` is a permutation of [0, world)) and a power-of-two
    // p2p_size (XOR with i < p2p_size then stays inside the p2p domain). Otherwise peers
    // are silently exchanged twice or not at all.
    CCL_THROW_IF_NOT(is_pof2(world) && is_pof2(p2p_size),
                     "alltoall numa pairwise requires power-of-two comm sizes, got world ",
                     world,
                     " p2p_comm ",
                     p2p_size);

    LOG_DEBUG("alltoall_sycl_numa_pairwise_rdma_oneshot: rank ",
              rank,
              "/",
              world,
              ", count ",
              count,
              ", p2p_size ",
              p2p_size,
              ", use_proxy ",
              use_proxy);

    /* scale up */
    if (p2p_size > 1) {
        int first_numa_rank = p2p_comm->get_global_rank(0);
        sycl::queue scaleup_q = q;
        ccl::event up_e = alltoall_scaleup_sycl(scaleup_q,
                                                (char*)send_buf + first_numa_rank * size,
                                                (char*)recv_buf + first_numa_rank * size,
                                                count,
                                                dtype,
                                                comm,
                                                p2p_comm,
                                                true, /* is_numa */
                                                0,
                                                global_stream,
                                                deps,
                                                done);
    }
    else {
        // copy self
        sycl::event copy_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy((char*)recv_buf + rank * size, (char*)send_buf + rank * size, size);
        });
    }

    /* scale out */
    sycl::event sycl_e;
    if (use_proxy) {
        ccl_proxy_op op{ ccl_proxy_op_type::alltoall_pairwise,
                         (void*)send_buf,
                         recv_buf,
                         count,
                         dtype,
                         ccl::reduction::sum, // unused by alltoall
                         comm,
                         /*aux=*/nullptr };
        op.peer_start = p2p_size;
        // No on-device compute remains, so a single sync-only work-group is fine.
        op.sync_only = true;

        sycl_e = submit_and_launch_persistent_kernel(q, op, dep_events);
    }
    else {
        sycl_e = alltoall_numa_pairwise_exchange_host(
            q, send_buf, recv_buf, count, dtype, comm, p2p_size, dep_events);
    }

    return ccl::event::create_from_native(sycl_e);
}

ccl::event alltoall_sycl_numa_pairwise_rdma_oneshot_split(sycl::queue& q,
                                                          const void* send_buf,
                                                          void* recv_buf,
                                                          size_t count,
                                                          ccl::datatype dtype,
                                                          ccl_comm* comm,
                                                          ccl_stream* global_stream,
                                                          const ccl::vector_class<ccl::event>& deps,
                                                          bool batch_mode,
                                                          bool& done) {
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    ccl_comm* numa_comm = comm->get_numa_comm().get();
    ccl_comm* numa_r2r_comm = comm->get_numa_r2r_comm().get();
    uint32_t world = comm->size();
    uint32_t rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();
    int ep_idx = 0;
    int pof2 = is_pof2(world);
    sycl::event sycl_e;

    done = true;

    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t sched_id = comm->get_sched_id(true, false);
    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    int node_size = node_comm->size();
    int numa_size = numa_comm->size();
    int numa_rank = numa_comm->rank();
    int num_numas = node_size / numa_size;
    int my_numa_id = rank / numa_size;
    int numa_split_count = ccl::global_data::env().sycl_numa_nodes_split;
    bool split_numa = ccl::global_data::env().sycl_split_numa;

    /* scale up */
    if (numa_size > 1) {
        int first_numa_rank = numa_comm->get_global_rank(0);
        sycl::queue scaleup_q = q;
        int first_rank = node_comm->get_global_rank(0);
        ccl::event up_e = alltoall_scaleup_sycl(scaleup_q,
                                                (char*)send_buf + first_rank * size,
                                                (char*)recv_buf + first_rank * size,
                                                count,
                                                dtype,
                                                comm,
                                                node_comm,
                                                true,
                                                numa_split_count,
                                                global_stream,
                                                deps,
                                                done);
    }
    else {
        // copy self
        sycl::event copy_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy((char*)recv_buf + rank * size, (char*)send_buf + rank * size, size);
        });
    }

    /* Do the pairwise exchanges */
    size_t block = size;
    if (size > 512 * 1024 * 1024) {
        block /= 2;
    }
    //block = 32 * 1024 * 1024;
    size_t progress = 0;
    int c = 0;

    // if split_numa, only the first numa_split_count ranks will do
    // exchange in memory semantics with extra numa_split_count ranks with
    // the other numa.
    // if not split_numa, each rank will do numa_size + numa_split_count
    // steps of exchange, and the rest steps are network semantics.
    // The load is balanced and split between memory and network
    // semantics
    int start = numa_size;
    if (!split_numa && numa_split_count)
        start += numa_split_count;
    if (split_numa && numa_split_count && numa_rank < numa_split_count)
        start += numa_split_count;
    while (progress < size) {
        size_t tocopy = progress + block < size ? block : size - progress;
        sycl_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.host_task([=]() {
                std::vector<atl_req_t> reqs(world * 2);
                int nreqs = 0;
                // post all recvs
                for (int i = start; i < world; i++) {
                    int src;
                    if (pof2) {
                        src = rank ^ i;
                    }
                    else {
                        src = (rank - i + world) % world;
                    }

                    int64_t recv_tag = atl_comm->tag_creator->create(
                        src, comm->get_comm_id(), sched_id, c + i + 1);
                    ATL_CALL_THROW_IF_ERROR(atl_comm->recv(ep_idx,
                                                           (char*)recv_buf + src * size + progress,
                                                           tocopy,
                                                           src,
                                                           recv_tag,
                                                           reqs[nreqs++]));
                }

                // post all sends
                for (int i = start; i < world; i++) {
                    int dst;
                    if (pof2) {
                        dst = rank ^ i;
                    }
                    else {
                        dst = (rank + i) % world;
                    }

                    int64_t send_tag = atl_comm->tag_creator->create(
                        rank, comm->get_comm_id(), sched_id, c + i + 1);
                    ATL_CALL_THROW_IF_ERROR(atl_comm->send(ep_idx,
                                                           (char*)send_buf + dst * size + progress,
                                                           tocopy,
                                                           dst,
                                                           send_tag,
                                                           reqs[nreqs++]));
                }

                // waitall
                for (int i = 0; i < nreqs; i++) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, reqs[i]));
                }
                // barrier
                if (progress + tocopy < size && 0) {
                    atl_req_t req;
                    ATL_CALL_THROW_IF_ERROR(atl_comm->barrier(ep_idx, req));
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
                }
            });
        });
        progress += tocopy;
        c++;
    }

    return ccl::event::create_from_native(sycl_e);
}

// assume GPU RDMA is enabled
// does not work for in place
// no scaleup, all scaleout
// for each step, send and recv and wait for each pair to finish
ccl::event alltoall_sycl_global_pairwise_rdma(sycl::queue& q,
                                              const void* send_buf,
                                              void* recv_buf,
                                              size_t count,
                                              ccl::datatype dtype,
                                              ccl_comm* comm,
                                              ccl_stream* global_stream,
                                              const ccl::vector_class<ccl::event>& deps,
                                              bool& done) {
    ccl_comm* node_comm = comm->get_node_comm().get();
    uint32_t world = comm->size();
    uint32_t rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();
    int ep_idx = 0;
    int pof2 = is_pof2(world);
    sycl::event sycl_e;

    done = true;

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    // copy to self
    sycl::event copy_e = q.submit([=](sycl::handler& h) {
        h.depends_on(dep_events);
        h.memcpy((char*)recv_buf + rank * size, (char*)send_buf + rank * size, size);
    });
    dep_events.clear();
    dep_events.push_back(std::move(copy_e));

    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t sched_id = comm->get_sched_id(true, false);

    /* Do the pairwise exchanges */
    size_t block = size;
    if (size > 512 * 1024 * 1024) {
        block /= 2;
    }
    size_t progress = 0;
    int c = 0;
    while (progress < size) {
        size_t tocopy = progress + block < size ? block : size - progress;
        sycl_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.host_task([=]() {
                int src, dst;

                std::vector<atl_req_t> reqs(world * 2);
                int nreqs = 0;
                for (int i = 1; i < world; i++) {
                    if (pof2) {
                        src = dst = rank ^ i;
                    }
                    else {
                        src = (rank - i + world) % world;
                        dst = (rank + i) % world;
                    }

                    int64_t recv_tag =
                        atl_comm->tag_creator->create(src, comm->get_comm_id(), sched_id, c + 1);
                    ATL_CALL_THROW_IF_ERROR(atl_comm->recv(ep_idx,
                                                           (char*)recv_buf + src * size + progress,
                                                           tocopy,
                                                           src,
                                                           recv_tag,
                                                           reqs[nreqs++]));

                    int64_t send_tag =
                        atl_comm->tag_creator->create(rank, comm->get_comm_id(), sched_id, c + 1);
                    ATL_CALL_THROW_IF_ERROR(atl_comm->send(ep_idx,
                                                           (char*)send_buf + dst * size + progress,
                                                           tocopy,
                                                           dst,
                                                           send_tag,
                                                           reqs[nreqs++]));
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, reqs[nreqs - 2]));
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, reqs[nreqs - 1]));
                }
                // barrier
                if (progress + tocopy < size) {
                    atl_req_t req;
                    atl_comm->barrier(ep_idx, req);
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
                }
            });
        });
        progress += tocopy;
        c++;
    }

    return ccl::event::create_from_native(sycl_e);
}

// host_task scale-out exchange of alltoall_sycl_scatter_rdma:
// post all recv, and post all sends and wait all in the end
// no throttling
sycl::event alltoall_scatter_exchange_host(sycl::queue& q,
                                           const void* send_buf,
                                           void* recv_buf,
                                           size_t count,
                                           ccl::datatype dtype,
                                           ccl_comm* comm,
                                           int peer_start,
                                           const std::vector<sycl::event>& dep_events) {
    uint32_t world = comm->size();
    uint32_t rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();
    int ep_idx = 0;
    int pof2 = is_pof2(world);
    sycl::event sycl_e;

    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t sched_id = comm->get_sched_id(true, false);

    /* Do the pairwise exchanges */
    size_t block = size;
    if (size > 512 * 1024 * 1024) {
        block /= 2;
    }
    size_t progress = 0;
    int c = 0;
    int start = peer_start;
    while (progress < size) {
        size_t tocopy = progress + block < size ? block : size - progress;
        sycl_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.host_task([=]() {
                std::vector<atl_req_t> reqs(world * 2);
                int nreqs = 0;
                // posed all recvs
                for (int i = start; i < world; i++) {
                    int src;
                    if (pof2) {
                        src = rank ^ i;
                    }
                    else {
                        src = (rank - i + world) % world;
                    }

                    int64_t recv_tag =
                        atl_comm->tag_creator->create(src, comm->get_comm_id(), sched_id, c + 1);
                    ATL_CALL_THROW_IF_ERROR(atl_comm->recv(ep_idx,
                                                           (char*)recv_buf + src * size + progress,
                                                           tocopy,
                                                           src,
                                                           recv_tag,
                                                           reqs[nreqs++]));
                }

                // posed all sends
                for (int i = start; i < world; i++) {
                    int dst;
                    if (pof2) {
                        dst = rank ^ i;
                    }
                    else {
                        dst = (rank + i) % world;
                    }

                    int64_t send_tag =
                        atl_comm->tag_creator->create(rank, comm->get_comm_id(), sched_id, c + 1);
                    ATL_CALL_THROW_IF_ERROR(atl_comm->send(ep_idx,
                                                           (char*)send_buf + dst * size + progress,
                                                           tocopy,
                                                           dst,
                                                           send_tag,
                                                           reqs[nreqs++]));
                }

                // waitall
                for (int i = 0; i < nreqs; i++) {
                    atl_comm->poll(ep_idx);
                    while (!reqs[i].is_completed) {
                        ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, reqs[i]));
                    }
                }
                // barrier
                if (0) {
                    atl_req_t req;
                    atl_comm->barrier(ep_idx, req);
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
                }
            });
        });
        progress += tocopy;
        c++;
    }

    return sycl_e;
}

// scatter: no scaleup, all scaleout
//
// with use_proxy, the exchange runs on the proxy thread instead of in a host_task on
// the caller's queue
ccl::event alltoall_sycl_scatter_rdma(sycl::queue& q,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      ccl::datatype dtype,
                                      ccl_comm* comm,
                                      ccl_stream* global_stream,
                                      const ccl::vector_class<ccl::event>& deps,
                                      bool use_proxy,
                                      bool& done) {
    uint32_t rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();

    done = true;

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    // for small message sizes, using transport instead of
    // launching a copy kernel which has higher overhead
    bool fuse = (size <= 262144) ? true : false;

    LOG_DEBUG("alltoall_sycl_scatter_rdma: rank ",
              rank,
              "/",
              comm->size(),
              ", count ",
              count,
              ", fuse ",
              fuse,
              ", use_proxy ",
              use_proxy);

    // copy to self
    if (!fuse) {
        sycl::event copy_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy((char*)recv_buf + rank * size, (char*)send_buf + rank * size, size);
        });
        dep_events.clear();
        dep_events.push_back(std::move(copy_e));
    }

    // ranks [0, peer_start) are handled here rather than by the exchange
    int peer_start = fuse ? 0 : 1;

    sycl::event sycl_e;
    if (use_proxy) {
        ccl_proxy_op op{ ccl_proxy_op_type::alltoall_pairwise,
                         (void*)send_buf,
                         recv_buf,
                         count,
                         dtype,
                         ccl::reduction::sum, // unused by alltoall
                         comm,
                         /*aux=*/nullptr };
        op.peer_start = peer_start;
        // No on-device compute remains, so a single sync-only work-group is fine.
        op.sync_only = true;

        sycl_e = submit_and_launch_persistent_kernel(q, op, dep_events);
    }
    else {
        sycl_e = alltoall_scatter_exchange_host(
            q, send_buf, recv_buf, count, dtype, comm, peer_start, dep_events);
    }

    return ccl::event::create_from_native(sycl_e);
}

// scaleup + scaleout with pairwise algorithm
// assume GPU RDMA is enabled
// does not work for in place
ccl::event alltoall_sycl_pairwise_rdma(sycl::queue& q,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t count,
                                       ccl::datatype dtype,
                                       ccl_comm* comm,
                                       std::shared_ptr<ccl_comm> p2p_comm,
                                       std::shared_ptr<ccl_comm> r2r_comm,
                                       ccl_stream* global_stream,
                                       const ccl::vector_class<ccl::event>& deps,
                                       bool& done) {
    uint32_t world = comm->size();
    uint32_t rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();
    int ep_idx = 0;
    int pof2 = is_pof2(world);

    done = true;

    LOG_DEBUG("pairwise on rank: ", rank, " comm size: ", world);
#if 1
    sycl::queue scaleout_q =
        sycl::queue(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
#else
    sycl::queue scaleout_q = q;
#endif

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    const ccl::topo_manager& topo_manager = comm->get_topo_manager();
    std::vector<int> local_ar(world);
    int first_node_rank = world;
    auto rank_info = topo_manager.get_filtered_rank_info_vec(topo_manager.get_host_idx());
    for (int rank_idx = 0; rank_idx < world; rank_idx++) {
        local_ar[rank_idx] = 0;
        for (auto& local_info : rank_info) {
            if (rank_idx == local_info.rank) {
                local_ar[rank_idx] = 1;
                if (rank_idx < first_node_rank)
                    first_node_rank = rank_idx;
            }
        }
    }

    /* scale up */
    if (p2p_comm->size() > 1) {
        sycl::queue scaleup_q = q;
        int first_node_rank = p2p_comm->get_global_rank(0);
        ccl::event up_e = alltoall_scaleup_sycl(scaleup_q,
                                                (char*)send_buf + first_node_rank * size,
                                                (char*)recv_buf + first_node_rank * size,
                                                count,
                                                dtype,
                                                comm,
                                                p2p_comm,
                                                false,
                                                0,
                                                global_stream,
                                                deps,
                                                done);
    }
    else {
        // copy self
        sycl::event copy_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy((char*)recv_buf + rank * size, (char*)send_buf + rank * size, size);
        });
    }

    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t sched_id = comm->get_sched_id(true, false);
    int64_t tag = atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), sched_id, 0);

    /* Do the pairwise exchanges */
    sycl::event out_e = scaleout_q.submit([=](sycl::handler& h) {
        h.depends_on(dep_events);
        h.host_task([=]() {
            int src, dst;
            for (int i = 1; i < world; i++) {
                if (pof2) {
                    src = dst = rank ^ i;
                }
                else {
                    src = (rank - i + world) % world;
                    dst = (rank + i) % world;
                }

                atl_req_t send_req, recv_req;
                if (local_ar[dst] == 0) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->send(
                        ep_idx, (char*)send_buf + dst * size, size, dst, tag, send_req));
                }
                if (local_ar[src] == 0) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->recv(
                        ep_idx, (char*)recv_buf + src * size, size, src, tag, recv_req));
                }
                if (local_ar[dst] == 0) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
                }
                if (local_ar[src] == 0) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
                }
            }
        });
    });

    dep_events.clear();
    dep_events.push_back(std::move(out_e));
    sycl::event sycl_e = submit_wait_on_events(q, dep_events);

    return ccl::event::create_from_native(sycl_e);
}

/* ======== Proxy-service exchange ======== */

// Pairwise exchange body used by the use_proxy paths of
// alltoall_sycl_numa_pairwise_rdma_oneshot and alltoall_sycl_scatter_rdma. It is the
// host_task body of their non-proxy exchanges, moved onto the proxy thread: the
// exchange is pure host ATL traffic, so no command-ring work is issued here.
//
// Ranks [0, peer_start) are handled by the caller (scale-up phase, or the self-copy
// on the caller's queue), so the loop starts at peer_start.
void alltoall_proxy_pairwise_exchange(const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      ccl::datatype dtype,
                                      ccl_comm* comm,
                                      int peer_start) {
    uint32_t world = comm->size();
    uint32_t rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();
    int ep_idx = 0;
    int pof2 = is_pof2(world);

    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    // Obtained here rather than passed in from the caller: get_sched_id() only
    // bumps a per-communicator counter, and the proxy thread is the sole consumer
    // of the host op queue, so this is the only thread advancing it for this comm.
    ccl_sched_id_t sched_id = comm->get_sched_id(true, false);

    LOG_DEBUG("alltoall_proxy_pairwise_exchange: rank ",
              rank,
              "/",
              world,
              ", count ",
              count,
              ", peer_start ",
              peer_start);

    /* Do the pairwise exchanges */
    size_t block = size;
    if (size > 512 * 1024 * 1024) {
        block /= 2;
    }
    size_t progress = 0;
    int c = 0;
    while (progress < size) {
        size_t tocopy = progress + block < size ? block : size - progress;

        std::vector<atl_req_t> reqs(world * 2);
        int nreqs = 0;
        // post all recvs
        for (int i = peer_start; i < world; i++) {
            int src;
            if (pof2) {
                src = rank ^ i;
            }
            else {
                src = (rank - i + world) % world;
            }

            int64_t recv_tag =
                atl_comm->tag_creator->create(src, comm->get_comm_id(), sched_id, c + 1);
            ATL_CALL_THROW_IF_ERROR(atl_comm->recv(ep_idx,
                                                   (char*)recv_buf + src * size + progress,
                                                   tocopy,
                                                   src,
                                                   recv_tag,
                                                   reqs[nreqs++]));
        }

        // post all sends
        for (int i = peer_start; i < world; i++) {
            int dst;
            if (pof2) {
                dst = rank ^ i;
            }
            else {
                dst = (rank + i) % world;
            }

            int64_t send_tag =
                atl_comm->tag_creator->create(rank, comm->get_comm_id(), sched_id, c + 1);
            ATL_CALL_THROW_IF_ERROR(atl_comm->send(ep_idx,
                                                   (char*)send_buf + dst * size + progress,
                                                   tocopy,
                                                   dst,
                                                   send_tag,
                                                   reqs[nreqs++]));
        }

        // waitall
        for (int i = 0; i < nreqs; i++) {
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, reqs[i]));
        }

        progress += tocopy;
        c++;
    }
}

// Proxy service wrapper shared by both alltoall algorithms - hence the single
// alltoall_pairwise op type. They differ only in the host-side setup that runs before
// the launch (scale-up phase vs. fused self-copy), which is already reflected in
// op.peer_start by the time we get here.
void proxy_alltoall_pairwise(const ccl_proxy_op& op, ccl_proxy_context& /*ctx*/) {
    alltoall_proxy_pairwise_exchange(
        op.src, op.dst, op.count, op.dtype, op.comm, op.peer_start);
}
