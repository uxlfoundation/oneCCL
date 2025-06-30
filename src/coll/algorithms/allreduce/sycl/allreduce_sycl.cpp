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
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/allreduce/sycl/allreduce_sycl.hpp"
#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_sycl.hpp"
#include "coll/algorithms/broadcast/sycl/broadcast_sycl.hpp"
#include "allreduce_ring_ll256.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

namespace ccl {
namespace v1 {

ccl::event allreduce_sycl_single_node(sycl::queue& q,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      ccl::datatype dtype,
                                      ccl::reduction reduction,
                                      ccl_comm* global_comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done) {
    ccl::event e;
    done = true;

    uint32_t world = global_comm->size();
    int rank = global_comm->rank();

    world = global_comm->get_node_comm()->size();
    rank = global_comm->get_node_comm()->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    if (world == 1) {
        sycl::event sycl_e;
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        if (send_buf != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: allreduce");
            sycl_e = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: allreduce");
            sycl_e = submit_wait_on_events(q, dep_events);
        }
        return ccl::event::create_from_native(sycl_e);
    }

    const bool is_single_tile = global_comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected =
        global_comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| is_single_tile: ",
              is_single_tile,
              ", has_all_vertices_connected: ",
              has_all_vertices_connected);

    // for ARC GPUs to do ring LL256
    if (is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()))) {
        if (!is_aligned(send_buf, recv_buf, count, 0, 4)) {
            done = false;
            return e;
        }
        if (!ccl::global_data::env().sycl_enable_arc_allreduce) {
            LOG_DEBUG("invoking allreduce LL256 kernel allreduce_ll_ring, count:",
                      count,
                      " datatype: ",
                      dtype);
            e = allreduce_ll_ring(
                send_buf, recv_buf, count, dtype, reduction, global_comm, global_stream, done);
            if (done) {
                LOG_DEBUG("invoking allreduce LL256 kernel, count:",
                          count,
                          " datatype: ",
                          dtype,
                          " done");
                return e;
            }
        }
        done = true;
        // ARC 770 does not support fp64
        if (ccl::ze::get_device_family(global_stream->get_ze_device()) ==
                ccl::device_family::family6 &&
            dtype == ccl::datatype::float64) {
            LOG_DEBUG("arc_allreduce does not support fp64");
            done = false;
            return e;
        }
        LOG_DEBUG(
            "invoking allreduce LL256 kernel arc_allreduce, count:", count, " datatype: ", dtype);
        e = arc_allreduce(send_buf, recv_buf, count, dtype, reduction, global_comm, global_stream);
        LOG_DEBUG("invoking allreduce LL256 kernel, count:", count, " datatype: ", dtype, " done");
        return e;
    }

    if (!ccl::global_data::env().sycl_esimd) {
        if (count * ccl_dtype.size() <= ccl::global_data::env().sycl_allreduce_small_threshold) {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("allreduce_small", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("invoking small allreduce kernel, count:", count, " datatype: ", dtype);
            e = allreduce_small(
                send_buf, recv_buf, count, dtype, reduction, global_comm, global_stream, deps);
            LOG_DEBUG(
                "invoking small allreduce kernel, count:", count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        }
        else {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("allreduce_large", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("invoking large allreduce kernel, count:", count, " datatype: ", dtype);
            e = allreduce_large(
                send_buf, recv_buf, count, dtype, reduction, global_comm, global_stream, deps);
            LOG_DEBUG(
                "invoking large allreduce kernel, count:", count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        }

        return e;
    }

    // ESIMD
    if (count * ccl_dtype.size() <= ccl::global_data::env().sycl_allreduce_small_threshold &&
        has_all_vertices_connected) {
        init_allreduce_small(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allreduce_small", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allreduce selects small kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_small(dtype, q, send_buf, recv_buf, count, reduction, deps, done);
        LOG_DEBUG("|CCL_SYCL| allreduce selects small kernel, count:",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        if (done)
            return e;
        // continue to medium kernel if not done
    }
    if ((count * ccl_dtype.size() <= ccl::global_data::env().sycl_allreduce_medium_threshold ||
         (global_comm->size() == 2 && !ccl::global_data::env().sycl_allreduce_tmp_buf)) &&
        !is_single_tile) { // medium message sizes
        init_allreduce_medium(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allreduce_medium", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG(
            "|CCL_SYCL| allreduce selects medium kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_medium(dtype, q, send_buf, recv_buf, count, reduction, deps, done);
        LOG_DEBUG("|CCL_SYCL| allreduce selects medium kernel, count:",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else if (!is_single_tile) { // large message sizes
        init_allreduce_large(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allreduce_large", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allreduce selects large kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_large(dtype, q, send_buf, recv_buf, count, reduction, deps, done);
        LOG_DEBUG("|CCL_SYCL| allreduce selects large kernel, count:",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
        done = false;
    }

    return e;
}

static bool do_fallback_to_scheduler(size_t size) {
    bool is_above_threshold = size > ccl::global_data::env().sycl_allreduce_scaleout_threshold;
    bool exception_cases = ccl::global_data::env().sycl_esimd ||
                           (ccl::global_data::env().atl_transport == ccl_atl_ofi &&
                            (ccl::global_data::env().sycl_allreduce_scaleout_algo == "auto" ||
                             ccl::global_data::env().sycl_allreduce_scaleout_algo == "direct"));
    return is_above_threshold || exception_cases;
}

bool is_rs_remainder_supported(size_t recv_count,
                               size_t rem_count,
                               size_t comm_size,
                               ccl_datatype ccl_dtype) {
    return rem_count != 0 && (recv_count * comm_size + rem_count) * ccl_dtype.size() <=
                                 ccl::global_data::env().sycl_reduce_scatter_small_threshold;
}

bool is_ag_remainder_supported(size_t send_count, size_t rem_count, ccl_datatype ccl_dtype) {
    return rem_count != 0 && (send_count + rem_count) * ccl_dtype.size() <=
                                 ccl::global_data::env().sycl_allgatherv_small_threshold;
}

ccl::event allreduce_sycl_multi_node_rs_phase(sycl::queue& q,
                                              const void* send_buf,
                                              void* recv_buf,
                                              size_t recv_count,
                                              const void* remainder_send_buf,
                                              void* remainder_recv_buf,
                                              size_t remainder_count,
                                              ccl::datatype dtype,
                                              ccl::reduction reduction,
                                              ccl_comm* node_comm,
                                              ccl_stream* global_stream,
                                              const vector_class<ccl::event>& deps,
                                              bool& done) {
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    // in scale-out case, average kernel is invoked
    // after scale-up phase calculated the sum first
    if (reduction == ccl::reduction::avg) {
        reduction = ccl::reduction::sum;
    }

    if (is_rs_remainder_supported(recv_count, remainder_count, node_comm->size(), ccl_dtype)) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "reduce_scatter_small", "send_size", recv_count * node_comm->size() * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        ev = reduce_scatter_small(send_buf,
                                  recv_buf,
                                  recv_count,
                                  remainder_count,
                                  dtype,
                                  reduction,
                                  node_comm,
                                  global_stream,
                                  deps);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
        sycl_coll_scaleup_attr coll_attr;
        coll_attr.force_use_tmp = true;
        ev = reduce_scatter_sycl_single_node(q,
                                             send_buf,
                                             recv_buf,
                                             recv_count,
                                             dtype,
                                             reduction,
                                             node_comm,
                                             global_stream,
                                             deps,
                                             done,
                                             coll_attr);
        if (!done) {
            LOG_INFO("allreduce_sycl reduce_scatter was not done -- falling back");
            // fallback
            return ev;
        }

        // use allreduce to handle reduction for the remainder
        if (remainder_count) {
            std::vector<event> evs;
            evs.push_back(std::move(ev));

            ev = allreduce_sycl_single_node(q,
                                            remainder_send_buf,
                                            remainder_recv_buf,
                                            remainder_count,
                                            dtype,
                                            reduction,
                                            node_comm,
                                            global_stream,
                                            evs,
                                            done);

            if (!done) {
                LOG_INFO("allreduce_sycl reduce_scatter was not done -- falling back");
                // fallback
                return ev;
            }
        }
    }
    return ev;
}

ccl::event allreduce_sycl_multi_node_ag_phase(sycl::queue& q,
                                              const void* send_buf,
                                              void* recv_buf,
                                              size_t send_count,
                                              const void* remainder_send_buf,
                                              void* remainder_recv_buf,
                                              size_t remainder_count,
                                              ccl::datatype dtype,
                                              ccl_comm* node_comm,
                                              ccl_stream* global_stream,
                                              const vector_class<event>& deps,
                                              bool& done) {
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int last_node_comm_rank = node_comm->size() - 1;

    if (is_ag_remainder_supported(send_count, remainder_count, ccl_dtype)) {
        const size_t last_block_count = send_count + remainder_count;

        size_t ag_send_count =
            node_comm->rank() == last_node_comm_rank ? last_block_count : send_count;
        std::vector<size_t> recv_counts(node_comm->size() - 1, send_count);
        recv_counts.push_back(last_block_count);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "allgatherv_small", "send_size", ag_send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        ev = allgatherv_small(send_buf,
                              ag_send_count,
                              recv_buf,
                              recv_counts,
                              {},
                              dtype,
                              node_comm,
                              global_stream,
                              deps);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
        std::vector<size_t> recv_counts(node_comm->size(), send_count);

        sycl_coll_scaleup_attr coll_attr;
        coll_attr.wait_on_deps = true;
        coll_attr.force_use_tmp = true;
        ev = allgather_sycl_single_node(q,
                                        send_buf,
                                        send_count,
                                        recv_buf,
                                        recv_counts,
                                        {},
                                        dtype,
                                        node_comm,
                                        global_stream,
                                        deps,
                                        done,
                                        coll_attr);
        if (!done) {
            // fallback
            LOG_INFO("allreduce_sycl allgatherv was not done -- falling back");
            return ev;
        }

        // last rank copies/broadcasts the remainder to other ranks
        if (remainder_count) {
            std::vector<event> rem_evs;
            rem_evs.push_back(std::move(ev));

            ev = broadcast_sycl_single_node(q,
                                            remainder_recv_buf,
                                            remainder_recv_buf,
                                            remainder_count,
                                            dtype,
                                            last_node_comm_rank,
                                            node_comm,
                                            global_stream,
                                            rem_evs,
                                            done);
        }
    }
    return ev;
}

ccl::event allreduce_sycl_multi_node(sycl::queue& q,
                                     const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     ccl_comm* global_comm,
                                     ccl_stream* global_stream,
                                     const vector_class<ccl::event>& deps,
                                     bool& done) {
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl_comm* node_comm = global_comm->get_node_comm().get();
    ccl_comm* r2r_comm = global_comm->get_r2r_comm().get();

    const int last_node_comm_rank = node_comm->size() - 1;

    done = true;

    // double check single node case
    if (r2r_comm->size() == 1) {
        LOG_DEBUG("allreduce calls single node");
        return allreduce_sycl_single_node(
            q, send_buf, recv_buf, count, dtype, reduction, global_comm, global_stream, deps, done);
    }

    if (do_fallback_to_scheduler(count * ccl_dtype.size())) {
        LOG_DEBUG("allreduce count size = ",
                  count * ccl_dtype.size(),
                  " is above scaleout SYCL threshold = ",
                  ccl::global_data::env().sycl_allreduce_scaleout_threshold,
                  " or sycl::esimd mode is enabled, or other conditions not met -- falling back");
        done = false;
        return ev;
    }

    // TODO: Sycl allgatherv does not support counts that are non-divisible by the node_comm size.
    //       Once this support is enabled, the algorithm will be simplified.

    size_t line_size = ccl::global_data::env().sycl_kernels_line_size;
    CCL_THROW_IF_NOT(
        !(line_size % ccl_dtype.size()), "datatype size not divisible by line_size=", line_size);

    size_t counts_per_line = line_size / ccl_dtype.size();

    const int buf_size = global_comm->get_scaleout_device_buf_size();
    size_t max_iter_count;
    if (count * ccl_dtype.size() <= buf_size) {
        max_iter_count = count;
    }
    else {
        max_iter_count = buf_size / ccl_dtype.size();
        max_iter_count = max_iter_count / counts_per_line / node_comm->size() * counts_per_line *
                         node_comm->size();
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    std::vector<event> evs;
    for (int i = 0; i < dep_events.size(); i++) {
        ev = ccl::event::create_from_native(dep_events[i]);
        evs.push_back(std::move(ev));
    }

    size_t displ = 0, displ_count = 0;
    int nchunks = (count + max_iter_count - 1) / max_iter_count;
    for (int i = 0; i < nchunks; i++) {
        size_t iter_count = i < nchunks - 1 ? max_iter_count : count - displ_count;

        size_t counts_per_rank = iter_count / node_comm->size();
        size_t remainder_count = iter_count % node_comm->size();

        {
            size_t total_lines = iter_count / counts_per_line;
            remainder_count = iter_count % counts_per_line;

            size_t lines_per_rank = total_lines / node_comm->size();
            size_t remainder_lines = total_lines % node_comm->size();

            counts_per_rank = lines_per_rank * counts_per_line;
            remainder_count += remainder_lines * counts_per_line;

            CCL_THROW_IF_NOT(iter_count == remainder_count + (counts_per_rank * node_comm->size()),
                             "Incorrect calculations for lines_per_rank.");
        }

        LOG_DEBUG("allreduce_sycl count=",
                  iter_count,
                  " counts_per_rank=",
                  counts_per_rank,
                  " remainder_count=",
                  remainder_count);

        // single remainder case (which is the last chunk) is handled by the
        // direct MPI call due to performance reasons
        if (counts_per_rank == 0 && remainder_count) {
            CCL_ASSERT(i == nchunks - 1);
            LOG_DEBUG("using CPU-side algorithm for the remainder count=", remainder_count);

            if (ccl::global_data::env().atl_transport == ccl_atl_ofi) {
                // fallback
                LOG_DEBUG("allreduce count size = ",
                          count * ccl_dtype.size(),
                          " has only a single remainder to compute = ",
                          remainder_count * ccl_dtype.size(),
                          ", OFI transport cannot handle the case ",
                          "-- falling back");
                done = false;
                return ev;
            }

            sycl_allreduce_tune_attr scaleout_tune_attr = { allreduce_scaleout_algo::direct };
            ev = allreduce_scaleout_sycl(q,
                                         (char*)send_buf + displ,
                                         (char*)recv_buf + displ,
                                         iter_count,
                                         dtype,
                                         ccl::reduction::sum,
                                         global_comm,
                                         evs,
                                         true, /* original_deps */
                                         scaleout_tune_attr,
                                         done,
                                         false /*is_cpu_buffers*/);
            if (!done) {
                LOG_INFO("allreduce_sycl allreduce_scaleout_sycl for remainder count"
                         " was not done -- falling back");
                // fallback
                return ev;
            }

            if (reduction == ccl::reduction::avg) {
                // set dependencies
                std::vector<sycl::event> avg_deps_evs;
                avg_deps_evs.push_back(ev.get_native());
                // average divisor
                int total_ranks = global_comm->size();
                LOG_DEBUG("allreduce_sycl calculate average on counts: ",
                          count,
                          ", ranks: ",
                          total_ranks);

                sycl::event reduce_event = sycl_average(
                    q, (char*)recv_buf + displ, iter_count, total_ranks, dtype, avg_deps_evs);
                ev = ccl::event::create_from_native(reduce_event);
            }
            return ev;
        }

        // prepare current node rank offset inside recv buffer
        const size_t recv_count_offset_bytes =
            node_comm->rank() * counts_per_rank * ccl_dtype.size();
        auto recv_rank_ptr = ptr_offset((char*)recv_buf + displ, recv_count_offset_bytes);

        // prepare buffers offset to handle the remainder
        const size_t remainder_offset_count = iter_count - remainder_count;
        const size_t remainder_offset_bytes = remainder_offset_count * ccl_dtype.size();
        auto remainder_send_buf = ptr_offset((char*)send_buf + displ, remainder_offset_bytes);
        auto remainder_recv_buf = ptr_offset((char*)recv_buf + displ, remainder_offset_bytes);

        // prepare scale-out data
        // last rank is a bit bigger to handle the remainder
        size_t scaleout_counts = node_comm->rank() == last_node_comm_rank
                                     ? counts_per_rank + remainder_count
                                     : counts_per_rank;

        // scale-up reduce-scatter phase
        if (node_comm->size() > 1) {
            ev = allreduce_sycl_multi_node_rs_phase(q,
                                                    (char*)send_buf + displ,
                                                    recv_rank_ptr,
                                                    counts_per_rank,
                                                    remainder_send_buf,
                                                    remainder_recv_buf,
                                                    remainder_count,
                                                    dtype,
                                                    reduction,
                                                    node_comm,
                                                    global_stream,
                                                    evs,
                                                    done);
            if (!done) {
                LOG_INFO("allreduce_sycl reduce_scatter phase was not done -- falling back");
                // fallback
                return ev;
            }

            evs.clear();
            evs.push_back(std::move(ev));
        }

        // scaleout allreduce phase
        {
            sycl_allreduce_tune_attr scaleout_tune_attr = allreduce_select_tune_attr(
                counts_per_rank * ccl_dtype.size(), r2r_comm->size(), ccl_dtype);
            LOG_DEBUG("allreduce_sycl scaleout count: ",
                      count,
                      " and scaleout_count: ",
                      scaleout_counts,
                      " and pipeline_chunk_size: ",
                      scaleout_tune_attr.pipeline_chunk_size,
                      " and #chunk: ",
                      i,
                      " of nchunks: ",
                      nchunks,
                      " - allreduce_scaleout_sycl");
            void* scaleout_send_ptr =
                node_comm->size() > 1 ? recv_rank_ptr : (char*)send_buf + displ;
            bool original_deps = node_comm->size() == 1 ? (evs.size() == 0) : false;
            ev = allreduce_scaleout_sycl(q,
                                         scaleout_send_ptr,
                                         recv_rank_ptr,
                                         scaleout_counts,
                                         dtype,
                                         ccl::reduction::sum,
                                         r2r_comm,
                                         evs,
                                         original_deps,
                                         scaleout_tune_attr,
                                         done,
                                         false /*is_cpu_buffers*/);
            if (!done) {
                LOG_INFO("allreduce_sycl scaleout was not done -- falling back");
                return ev;
            }

            if (reduction == ccl::reduction::avg) {
                // set dependencies
                std::vector<sycl::event> avg_deps_evs;
                avg_deps_evs.push_back(ev.get_native());
                // average divisor
                int total_ranks = global_comm->size();
                LOG_DEBUG("allreduce_sycl calculate average on counts: ",
                          scaleout_counts,
                          ", ranks: ",
                          total_ranks);

                sycl::event reduce_event = sycl_average(
                    q, recv_rank_ptr, scaleout_counts, total_ranks, dtype, avg_deps_evs);
                ev = ccl::event::create_from_native(reduce_event);
            }
        }

        if (node_comm->size() > 1) {
            evs.clear();
            evs.push_back(std::move(ev));

            // scale-up allgatherv phase
            ev = allreduce_sycl_multi_node_ag_phase(q,
                                                    recv_rank_ptr,
                                                    (char*)recv_buf + displ,
                                                    counts_per_rank,
                                                    remainder_send_buf,
                                                    remainder_recv_buf,
                                                    remainder_count,
                                                    dtype,
                                                    node_comm,
                                                    global_stream,
                                                    evs,
                                                    done);
            if (!done) {
                // fallback
                LOG_INFO("allreduce_sycl allgatherv phase was not done -- falling back");
                return ev;
            }
        }

        if (i < nchunks - 1) {
            evs.clear();
            evs.push_back(std::move(ev));
            displ_count += iter_count;
            displ += iter_count * ccl_dtype.size();
        }
    } // for chunking

    return ev;
}

event allreduce_sycl(sycl::queue& q,
                     const void* send_buf,
                     void* recv_buf,
                     size_t count,
                     datatype dtype,
                     reduction reduction,
                     ccl_comm* global_comm,
                     ccl_stream* global_stream,
                     const allreduce_attr& attr,
                     const vector_class<event>& deps,
                     bool& done) {
    done = true;
    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = global_comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (count == 0) {
        auto sycl_deps = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_deps);
        return ccl::event::create_from_native(e);
    }

    if (is_single_node && ccl::global_data::env().sycl_single_node_algorithm) {
        LOG_DEBUG("is_single_node");
        return allreduce_sycl_single_node(
            q, send_buf, recv_buf, count, dtype, reduction, global_comm, global_stream, deps, done);
    }

    return allreduce_sycl_multi_node(
        q, send_buf, recv_buf, count, dtype, reduction, global_comm, global_stream, deps, done);
}

} // namespace v1
} // namespace ccl
