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
#include "coll/algorithms/allgatherv/sycl/allgatherv_pcie.hpp"
#endif // CCL_ENABLE_SYCL

ccl::event allgatherv_ll_ring(const void *send_buf,
                              size_t send_count,
                              void *recv_buf,
                              const ccl::vector_class<size_t> &recv_counts,
                              const ccl::vector_class<size_t> &offsets,
                              ccl::datatype dtype,
                              ccl_comm *comm,
                              ccl_stream *global_stream,
                              const ccl::vector_class<ccl::event> &deps,
                              bool &done) {
    sycl::event sycl_e;
    sycl::queue q = global_stream->get_native_stream();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    coll_init(comm, global_stream);

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t dt_sz = ccl_dtype.size();
    size_t send_size = send_count * ccl_dtype.size();

    bool p2p = node_comm->get_topo_manager().has_p2p_access();
    uint32_t pattern = pattern_counter;

    auto lambda = [&]<typename T, int NRanks, template <typename, int> class Proto>() {
        T *peerbuf0[NRanks];
        T *peerbuf1[NRanks];
        for (int i = 0; i < NRanks; i++) {
            peerbuf0[i] = (T *)get_remote_node_tmp_buf(0, comm)[i];
            peerbuf1[i] = (T *)get_remote_node_tmp_buf(1, comm)[i];
        }
        T *ipcbuf0 = (T *)get_tmp_buf(0, comm);
        T *ipcbuf1 = (T *)get_tmp_buf(1, comm);
        sycl::event e = AllGather<T, NRanks, Proto, RingTransmit>::launch((T *)send_buf,
                                                                          (T *)recv_buf,
                                                                          ipcbuf0,
                                                                          ipcbuf1,
                                                                          peerbuf0,
                                                                          peerbuf1,
                                                                          send_count,
                                                                          comm_rank,
                                                                          pattern,
                                                                          q,
                                                                          p2p,
                                                                          done);
        // update pattern
        pattern_counter = pattern;
        return e;
    };

    if (send_size <= ccl::global_data::env().sycl_allgatherv_ll_threshold) {
        // small ring with LL
        sycl_e = invoke_pcie<Rt64_PCIE>(lambda, comm, dtype);
    }
    else {
        // simple ring with LL256
        sycl_e = invoke_pcie<Rt64_128_PCIE>(lambda, comm, dtype);
    }

    return ccl::event::create_from_native(sycl_e);
}
