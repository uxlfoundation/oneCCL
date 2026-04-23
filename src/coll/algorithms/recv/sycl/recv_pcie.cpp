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
#include "coll/algorithms/recv/sycl/recv_pcie.hpp"
#endif // CCL_ENABLE_SYCL

ccl::event recv_ll(const void *recv_buf,
                   size_t recv_count,
                   ccl::datatype dtype,
                   int peer_rank,
                   ccl_comm *comm,
                   ccl_stream *global_stream,
                   const ccl::vector_class<ccl::event> &deps,
                   bool &done) {
    sycl::event sycl_e;
    sycl::queue q = global_stream->get_native_stream();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t dt_sz = ccl_dtype.size();
    size_t recv_size = recv_count * ccl_dtype.size();

    bool p2p = node_comm->get_topo_manager().has_p2p_access();
    uint32_t pattern = node_comm->get_rt_pattern(pattern_type::recv, peer_rank);

    LOG_DEBUG("recv_ll recv_count: ", recv_count, " peer rank: ", peer_rank);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    auto lambda = [&]<typename T, template <typename, int> class Proto>(int NRanks) {
        T *peerbuf0[NRanks];
        T *peerbuf1[NRanks];
        T *ipcbuf0;
        T *ipcbuf1;
        // use small tmp buffer. Can not use large tmp buffer because
        // there is only single copy of the buffer.
        // Otherwise a barrier is needed her
        auto [local_tmp_buf, remote_ptrs] = node_comm->get_all_tmp_bufs(true);
        for (int i = 0; i < NRanks; i++) {
            peerbuf0[i] = (T *)remote_ptrs[i];
            peerbuf1[i] = (T *)((char *)remote_ptrs[i] + ccl_tmp_bufs::buf_size / 2);
        }
        ipcbuf0 = (T *)local_tmp_buf;
        ipcbuf1 = (T *)((char *)local_tmp_buf + ccl_tmp_bufs::buf_size / 2);

        sycl::event e = Recv<T, Proto, RingTransmit>::launch(NRanks,
                                                             (T *)recv_buf,
                                                             ipcbuf0,
                                                             ipcbuf1,
                                                             peerbuf0,
                                                             peerbuf1,
                                                             recv_count,
                                                             comm_rank,
                                                             peer_rank,
                                                             pattern,
                                                             q,
                                                             node_comm,
                                                             dep_events,
                                                             p2p,
                                                             done);
        return e;
    };

    if (recv_size <= ccl::global_data::env().sycl_allgatherv_ll_threshold) {
        // small ring with LL
        sycl_e = invoke_pcie_type<Rt64_PCIE>(lambda, comm_size, dtype);
    }
    else {
        // simple ring with LL256
        sycl_e = invoke_pcie_type<Rt64_128_PCIE>(lambda, comm_size, dtype);
    }

    return ccl::event::create_from_native(sycl_e);
}
