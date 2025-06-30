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
#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"

namespace ccl {
namespace v1 {

ccl::event allgather_sycl_single_node(sycl::queue& q,
                                      const void* send_buf,
                                      size_t send_count,
                                      void* recv_buf,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      const ccl::vector_class<size_t>& offsets,
                                      ccl::datatype dtype,
                                      ccl_comm* comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done,
                                      sycl_coll_scaleup_attr coll_attr) {
    ccl::event e;
    done = true;

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const bool is_single_tile = comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected = comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected", has_all_vertices_connected);

    uint32_t world = comm->get_node_comm()->size();
    int rank = comm->get_node_comm()->rank();

    for (uint32_t i = 0; i < recv_counts.size(); i++) {
        if (send_count != recv_counts[i]) {
            LOG_ERROR("Allgatherv only supports the case when all recv_counts are the same");
            done = false;
            return e;
        }
        assert(send_count == recv_counts[i]);
    }

    if (world == 1) {
        sycl::event sycl_e;
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        if (send_buf != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: allgatherv");
            void* dst_buf = offsets.empty() ? recv_buf : (char*)recv_buf + offsets[0];
            sycl_e = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(dst_buf, send_buf, send_count * ccl_dtype.size());
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: allgatherv");
            sycl_e = submit_wait_on_events(q, dep_events);
        }
        return ccl::event::create_from_native(sycl_e);
    }

    // PCIe ring LL256
    if (is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()))) {
        if (!is_aligned(send_buf, recv_buf, send_count, ccl_dtype.size(), 4) ||
            ccl::global_data::env().sycl_enable_arc_allreduce) {
            done = false;
            return e;
        }
        LOG_DEBUG("invoking allgatherv LL256 kernel, send_count:", send_count, " datatype: ", dtype);
        e = allgatherv_ll_ring(
            send_buf, send_count, recv_buf, recv_counts, offsets, dtype, comm, global_stream, deps, done);
        LOG_DEBUG("invoking allgatherv LL256 kernel, count:", send_count, " datatype: ", dtype, " done");
        return e;
    }

    if (!ccl::global_data::env().sycl_esimd) {
        if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_small_threshold) {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("allgatherv_small", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype);
            e = allgatherv_small(
                send_buf, send_count, recv_buf, recv_counts, offsets, dtype, comm, global_stream, deps);
            LOG_DEBUG(
                "|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        }
        else {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("allgatherv_large", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| invoking large allgatherv: count: ", send_count, " datatype: ", dtype);
            e = allgatherv_large(
                send_buf, send_count, recv_buf, recv_counts, offsets, dtype, comm, global_stream, deps, coll_attr);
            LOG_DEBUG(
                "|CCL_SYCL| allgatherv selects large kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        }

        return e;
    }

    // ESIMD
    if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_small_threshold &&
        has_all_vertices_connected) {
        init_allgatherv_small(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_small", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype);
        e = run_allgatherv_small(dtype, q, send_buf, send_count, recv_buf, recv_counts, done);
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        if (done)
            return e;
    }
    if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_medium_threshold &&
        !is_single_tile) {
        init_allgatherv_medium(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_medium", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allgatherv selects medium kernel: count: ", send_count, " datatype: ", dtype);
        e = run_allgatherv_medium(dtype, q, send_buf, send_count, recv_buf, recv_counts, done);
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects medium kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_large", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| invoking large allgatherv: count: ", send_count, " datatype: ", dtype);

#if defined(CCL_SYCL_VEC_SUPPORT_FP16) && defined(CCL_SYCL_VEC_SUPPORT_BF16)
        e = allgatherv_large(
            send_buf, send_count, recv_buf, recv_counts, offsets, dtype, comm, global_stream, deps);
#else
        // allgatherv_large is sycl::vec based algorithm
        // when 16-bit datatypes are not supported, gather by int16 instead
        ccl::datatype new_dtype = ccl::datatype::int16;
        size_t new_send_count = send_count * ccl_dtype.size() / 2;
        ccl::vector_class<size_t> new_recv_counts;
        for (size_t i = 0; i < recv_counts.size(); i++) {
            new_recv_counts.push_back(recv_counts[i] * ccl_dtype.size() / 2);
        }
        e = allgatherv_large(
            send_buf, new_send_count, recv_buf, new_recv_counts, offsets, new_dtype, comm, global_stream, deps);
#endif // defined(CCL_SYCL_VEC_SUPPORT_FP16) && defined(CCL_SYCL_VEC_SUPPORT_BF16)
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects large kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }

    return e;
}

ccl::event allgatherv_sycl_multi_node(sycl::queue& q,
                                      const void* send_buf,
                                      size_t send_count,
                                      void* recv_buf,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      ccl::datatype dtype,
                                      ccl_comm* global_comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done) {
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl_comm* node_comm = global_comm->get_node_comm().get();
    ccl_comm* r2r_comm = global_comm->get_r2r_comm().get();

    LOG_DEBUG("allgatherv_sycl send_count=", send_count);

    size_t r2r_size = r2r_comm->size();
    std::vector<size_t> recv_scaleout_counts(r2r_size);
    std::vector<size_t> scaleout_offsets(r2r_size);
    size_t total_scaleout_count = 0;
    for (int i = 0; i < r2r_size; i++) {
        const int global_rank = r2r_comm->get_global_rank(i);
        total_scaleout_count += recv_counts[global_rank];
    }

    // ----- Scaleout Allgatherv Phase -----
    const int scaleout_buf_size = global_comm->get_scaleout_device_buf_size();
    void* scaleout_buf = global_comm->get_scaleout_device_buf(q);
    size_t max_pack_count;
    if (total_scaleout_count * ccl_dtype.size() <= scaleout_buf_size) {
        max_pack_count = send_count;
    }
    else {
        max_pack_count = scaleout_buf_size / r2r_size;
        int typesize = std::max(4, (int)ccl_dtype.size());
        max_pack_count = max_pack_count / typesize * typesize;
        max_pack_count = max_pack_count / ccl_dtype.size();
        CCL_ASSERT(max_pack_count > 0);
    }

    size_t node_size = node_comm->size();

    std::vector<size_t> global_offsets(recv_counts.size());
    global_offsets[0] = 0;
    for (int i = 1; i < recv_counts.size(); i++)
        global_offsets[i] = global_offsets[i - 1] + recv_counts[i - 1] * ccl_dtype.size();

    std::vector<event> evs;
    void* scaleout_send;
    int send_offset = 0;
    int nchunks = (send_count + max_pack_count - 1) / max_pack_count;
    for (int iter = 0; iter < nchunks; iter++) {
        int pack_count = (iter < nchunks - 1) ? max_pack_count : send_count - send_offset;

        // ----- Scaleout Allgatherv Phase -----
        scaleout_send = (char*)send_buf + send_offset * ccl_dtype.size();
        for (int i = 0; i < r2r_size; i++) {
            recv_scaleout_counts[i] = pack_count;
            scaleout_offsets[i] = (i * pack_count) * ccl_dtype.size();
        }

        sycl_allgatherv_tune_attr scaleout_tune_attr =
            allgatherv_select_tune_attr(pack_count * ccl_dtype.size(), r2r_comm->size(), ccl_dtype);
        ev = allgatherv_scaleout_sycl(q,
                                      scaleout_send,
                                      pack_count,
                                      scaleout_buf,
                                      recv_scaleout_counts,
                                      dtype,
                                      r2r_comm,
                                      iter == 0 ? deps : evs,
                                      iter == 0 ? true : false,
                                      done,
                                      scaleout_tune_attr,
                                      false);
        if (!done) {
            LOG_INFO("allgatherv_sycl scaleout was not done -- falling back");
            return ev;
        }

        evs.clear();
        evs.push_back(std::move(ev));

        // ----- Scaleup Allgatherv Inplace Phase -----
        {
            std::vector<size_t> scaleup_counts(node_size, pack_count);
            sycl_coll_scaleup_attr coll_attr;
            coll_attr.force_use_tmp = true;
            for (int i = 0; i < r2r_size; i++) {
                std::vector<size_t> scaleup_offsets(global_offsets.begin() + i * node_size,
                                                    global_offsets.begin() + (i + 1) * node_size);
                ev = allgather_sycl_single_node(q,
                                                (char*)(scaleout_buf) + scaleout_offsets[i],
                                                recv_scaleout_counts[i],
                                                (char*)recv_buf,
                                                scaleup_counts,
                                                scaleup_offsets,
                                                dtype,
                                                node_comm,
                                                global_stream,
                                                evs,
                                                done,
                                                coll_attr);

                if (!done) {
                    // fallback
                    LOG_ERROR("allgatherv_sycl allgatherv single node was not done -- falling back");
                    return ev;
                }

                evs.clear();
                evs.push_back(std::move(ev));
            }
        }

        if (iter < nchunks - 1) {
            send_offset += pack_count;

            for (int i = 0; i < global_offsets.size(); i++) {
                global_offsets[i] += pack_count * ccl_dtype.size();
            }
        }
    }

    //auto sycl_ev = ev.get_native();
    auto sycl_evs = get_sycl_events(evs);
    auto e = q.submit([=](sycl::handler& h) {
        h.depends_on(sycl_evs);
        h.host_task([=]() {
            global_comm->put_scaleout_device_buf(scaleout_buf);
        });
    });
    ev = ccl::event::create_from_native(e);

    return ev;
}

ccl::event allgather_sycl(sycl::queue& q,
                          const void* send_buf,
                          size_t send_count,
                          void* recv_buf,
                          const ccl::vector_class<size_t>& recv_counts,
                          ccl::datatype dtype,
                          ccl_comm* comm,
                          ccl_stream* op_stream,
                          const allgatherv_attr& attr,
                          const vector_class<event>& deps,
                          bool& done) {
    if (send_count == 0) {
        done = true;
        auto sycl_events = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_events);
        return ccl::event::create_from_native(e);
    }

    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (is_single_node) {
        LOG_DEBUG("is_single_node");
        return allgather_sycl_single_node(
            q, send_buf, send_count, recv_buf, recv_counts, {}, dtype, comm, op_stream, deps, done);
    }

    return allgatherv_sycl_multi_node(
        q, send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, deps, done);
}

} // namespace v1
} // namespace ccl
