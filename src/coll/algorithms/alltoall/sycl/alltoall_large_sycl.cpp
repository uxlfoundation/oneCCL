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
#include "coll/algorithms/alltoall/sycl/alltoall_large_sycl_impl.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

ccl::event alltoall_large(const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          ccl::datatype dtype,
                          ccl_comm* comm,
                          ccl_stream* global_stream,
                          const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("invoking alltoall_large");

    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();

    const size_t dsize = ccl::global_data::get().dtypes->get(dtype).size();

    CCL_THROW_IF_NOT(!ccl::global_data::env().sycl_copy_engine,
                     "alltoall using copy engines not supported");

    // no constraints on rank reordering - performance should not be affected
    // end-to-end transfers are performed
    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("|MT|: invoking alltoall_large");
        coll_initExt(comm, ccl::global_data::get().shared_data->hash_table, global_stream);
    }
    else {
        LOG_DEBUG("invoking alltoall_large");
        coll_init(comm, global_stream);
    }

    auto lambda = [&]<typename T, int NE, int NP>() {
        return alltoall_large_impl<T, NE * NP>(
            send_buf, recv_buf, count, dtype, comm, global_stream, deps);
    };
    return invoke_collective(lambda, comm, dtype);
}
