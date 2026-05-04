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
#include "common/global/global.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/alltoall/sycl/alltoall_pcie.hpp"
#endif // CCL_ENABLE_SYCL

ccl::event alltoall_ll(const void *send_buf,
                       void *recv_buf,
                       size_t send_count,
                       const ccl::vector_class<size_t> &offsets,
                       ccl::datatype dtype,
                       ccl_comm *comm,
                       ccl_stream *global_stream,
                       const ccl::vector_class<ccl::event> &deps,
                       bool &done) {
    sycl::event sycl_e;
    sycl::queue q = global_stream->get_native_stream();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    bool recording = use_recording_path(q);
    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t dt_sz = ccl_dtype.size();
    size_t send_size = send_count * ccl_dtype.size();

    bool p2p = node_comm->get_topo_manager().has_p2p_access();

    auto lambda = [&]<typename T, template <typename, int> class Proto>(int NRanks) {
        const size_t *offs = offsets.empty() ? NULL : offsets.data();

        T *peerbuf0[NRanks];
        T *peerbuf1[NRanks];
        T *ipcbuf0;
        T *ipcbuf1;
        if (ccl::global_data::env().sycl_ll_buffer_global) {
            // large buffer
            for (int i = 0; i < NRanks; i++) {
                peerbuf0[i] = (T *)get_remote_node_tmp_buf(0, comm)[i];
                peerbuf1[i] = (T *)get_remote_node_tmp_buf(1, comm)[i];
            }
            ipcbuf0 = (T *)get_tmp_buf(0, comm);
            ipcbuf1 = (T *)get_tmp_buf(1, comm);
        }
        else {
            // small buffer
            auto [local_tmp_buf, remote_ptrs] = recording ? node_comm->get_all_tmp_bufs_gpu(true)
                                                          : node_comm->get_all_tmp_bufs(true);
            for (int i = 0; i < NRanks; i++) {
                peerbuf0[i] = (T *)remote_ptrs[i];
                peerbuf1[i] = (T *)((char *)remote_ptrs[i] + node_comm->get_tmp_buf_size() / 2);
            }
            ipcbuf0 = (T *)local_tmp_buf;
            ipcbuf1 = (T *)((char *)local_tmp_buf + node_comm->get_tmp_buf_size() / 2);
        }

        sycl::event e;

        if (recording) {
            e = AllToAll<T, Proto, RingTransmit, true>::launch(
                comm->get_node_comm()->get_pattern_data_gpu(),
                NRanks,
                (T *)send_buf,
                (T *)recv_buf,
                offs,
                ipcbuf0,
                ipcbuf1,
                RingTransmit<int, Rt64_128_PCIE, true>::ringSize / sizeof(T),
                peerbuf1,
                send_count,
                comm_rank,
                -1,
                comm->global_current_id,
                q,
                pattern_type::collective,
                node_comm,
                p2p,
                done);
        }
        else {
            e = AllToAll<T, Proto, RingTransmit, false>::launch(
                comm->get_node_comm()->get_pattern_data(),
                NRanks,
                (T *)send_buf,
                (T *)recv_buf,
                offs,
                ipcbuf0,
                ipcbuf1,
                RingTransmit<int, Rt64_128_PCIE, false>::ringSize / sizeof(T),
                peerbuf1,
                send_count,
                comm_rank,
                -1,
                comm->global_current_id,
                q,
                pattern_type::collective,
                node_comm,
                p2p,
                done);
        }

        return e;
    };

    if (send_size <= ccl::global_data::env().sycl_alltoall_ll_threshold) {
        // small ring with LL
        sycl_e = invoke_pcie_type<Rt64_PCIE>(lambda, comm_size, dtype);
    }
    else {
        // simple ring with LL256
        sycl_e = invoke_pcie_type<Rt64_128_PCIE>(lambda, comm_size, dtype);
    }

    return ccl::event::create_from_native(sycl_e);
}
