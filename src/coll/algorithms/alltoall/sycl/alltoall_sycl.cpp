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
#include "coll/algorithms/alltoall/sycl/alltoall_sycl.hpp"
#include "coll/algorithms/alltoall/sycl/alltoall_ll256.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/transmit/transmit.hpp"

namespace ccl {
namespace v1 {

ccl::event alltoall_sycl_single_node(sycl::queue& q,
                                     const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     ccl_comm* comm,
                                     bool is_numa_comm,
                                     int numa_split,
                                     ccl_stream* global_stream,
                                     const vector_class<event>& deps,
                                     bool& done) {
    ccl::event e;
    done = true;

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t dt_sz = ccl_dtype.size();
    //
    //     const bool is_single_tile = comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected = comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected", has_all_vertices_connected);
    //
    uint32_t world = comm->get_node_comm()->size();
    int rank = comm->get_node_comm()->rank();

    if (world == 1) {
        sycl::event sycl_e;
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        if (send_buf != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: alltoall");
            sycl_e = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: alltoall");
            sycl_e = submit_wait_on_events(q, dep_events);
        }
        return ccl::event::create_from_native(sycl_e);
    }

    bool is_arc = is_arc_card(ccl::global_data::get().ze_data->devices[0].family);
    // PCIe ring LL256
    if (is_arc && ccl::global_data::env().sycl_alltoall_tmp_buf) {
        ccl::event e;
        if (!ccl::global_data::env().sycl_enable_arc_alltoall_ll) {
            int node_size = comm->size();
            const int chunk_size = ccl::global_data::env().sycl_alltoall_chunking_threshold;
            size_t max_pack_count;

            if (send_buf != recv_buf) {
                q.memcpy((char*)recv_buf + rank * count * ccl_dtype.size(),
                         (char*)send_buf + rank * count * ccl_dtype.size(),
                         count * ccl_dtype.size());
            }

            if (chunk_size == 0 || count * ccl_dtype.size() <= chunk_size) {
                max_pack_count = count;
            }
            else {
                max_pack_count = chunk_size;
                int typesize = std::max(4, (int)ccl_dtype.size());
                max_pack_count = max_pack_count / typesize * typesize;
                max_pack_count = max_pack_count / ccl_dtype.size();
                CCL_ASSERT(max_pack_count > 0);
            }
            size_t send_offset = 0;
            int nchunks = divUp(count, max_pack_count);
            for (int iter = 0; iter < nchunks; iter++) {
                int pack_count = (iter < nchunks - 1) ? max_pack_count : count - send_offset;
                std::vector<size_t> offsets(node_size);
                for (int i = 0; i < node_size; i++) {
                    offsets[i] = i * count * ccl_dtype.size();
                }
#ifdef CCL_ENABLE_ITT
                ccl::profile::itt::task_begin(
                    "alltoall_ll", "send_size", pack_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
                LOG_DEBUG("invoking alltoall LL256 kernel alltoall_ll, count:",
                          pack_count,
                          " datatype: ",
                          dtype);
                e = alltoall_ll((char*)send_buf + send_offset * ccl_dtype.size(),
                                (char*)recv_buf + send_offset * ccl_dtype.size(),
                                pack_count,
                                offsets,
                                dtype,
                                comm,
                                global_stream,
                                deps,
                                done);
                if (!done)
                    break;
                send_offset += pack_count;
            } // for nchunks
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            if (done) {
                LOG_DEBUG(
                    "invoking alltoall LL256 kernel, count:", count, " datatype: ", dtype, " done");
                return e;
            }
        }

        size_t dt_sz = ccl_dtype.size();
        if ((world & (world - 1)) == 0) {
            done = true;
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("arc_alltoall", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG(
                "|CCL_SYCL| alltoall selects arc_alltoall, count: ", count, " datatype: ", dtype);
            e = arc_alltoall(
                send_buf, recv_buf, count, dtype, comm, is_numa_comm, numa_split, global_stream);
            LOG_DEBUG("|CCL_SYCL| alltoall selects arc_alltoall, count: ",
                      count,
                      " datatype: ",
                      dtype,
                      " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            return e;
        }
    }

    if (ccl::global_data::env().sycl_esimd) {
        LOG_WARN(
            "|CCL_SYCL| sycl ESIMD requested for alltoall collective; ESIMD not supported, falling back to alltoall sycl implementation");
    }

    if (send_buf != recv_buf) {
        // ARC does chunking as needed
        const int chunk_size =
            is_arc ? ccl::global_data::env().sycl_alltoall_chunking_threshold : 0;
        size_t max_pack_count;
        size_t nchunks = calculate_chunking_pack_count(chunk_size, count, dt_sz, max_pack_count);
        size_t offset = 0;
        for (size_t iter = 0; iter < nchunks; iter++) {
            size_t pack_count = (iter < nchunks - 1) ? max_pack_count : count - offset;
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin(
                "alltoall_large", "send_size", pack_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| alltoall selects large kernel, count: ",
                      pack_count,
                      " datatype: ",
                      dtype);
            std::vector<size_t> scaleup_offsets(world);
            for (int r = 0; r < world; r++) {
                scaleup_offsets[r] = r * count + offset;
            }
            e = alltoall_large(
                send_buf, recv_buf, pack_count, scaleup_offsets, dtype, comm, global_stream, deps);
            LOG_DEBUG("|CCL_SYCL| alltoall selects large kernel, count: ",
                      pack_count,
                      " datatype: ",
                      dtype,
                      " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            offset += pack_count;
        } // end for
    }
    else {
        LOG_WARN(
            "|CCL_SYCL| sycl inplace requested for alltoall collective; inplace not supported, falling back");
        done = false;
    }

    return e;
}

ccl::event alltoall_sycl_multi_node(sycl::queue& q,
                                    const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    ccl_comm* comm,
                                    ccl_stream* global_stream,
                                    const vector_class<event>& deps,
                                    bool& done) {
    if (send_buf == recv_buf) {
        CCL_THROW("oneCCL does not support in-place Alltoall");
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    sycl_alltoall_tune_attr scaleout_tune_attr =
        alltoall_select_tune_attr(count * ccl_dtype.size(), comm->size(), ccl_dtype);

    return alltoall_scaleout_sycl(q,
                                  send_buf,
                                  recv_buf,
                                  count,
                                  dtype,
                                  comm,
                                  global_stream,
                                  deps,
                                  true,
                                  scaleout_tune_attr,
                                  done);
}

ccl::event alltoall_sycl(sycl::queue q,
                         const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         ccl_comm* comm,
                         ccl_stream* op_stream,
                         const alltoall_attr& attr,
                         const vector_class<event>& deps,
                         bool& done) {
    if (count == 0) {
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

    if (is_single_node && ccl::global_data::env().sycl_single_node_algorithm &&
        ccl::global_data::env().sycl_alltoall_single_node_algorithm) {
        if (send_buf != recv_buf) {
            LOG_DEBUG("is_single_node");
            return alltoall_sycl_single_node(
                q, send_buf, recv_buf, count, dtype, comm, false, 0, op_stream, deps, done);
        }
        else {
            LOG_WARN(
                "|CCL_SYCL| sycl inplace requested for alltoall collective; inplace not supported, falling back");
            done = false;
            return ccl::event();
        }
    }

    return alltoall_sycl_multi_node(
        q, send_buf, recv_buf, count, dtype, comm, op_stream, deps, done);
}

} // namespace v1
} // namespace ccl
