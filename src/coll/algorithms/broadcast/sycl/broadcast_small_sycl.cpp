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
#include "coll/algorithms/broadcast/sycl/broadcast_small_sycl_impl.hpp"

ccl::event broadcast_small(const void* send_buf,
                           void* recv_buf,
                           size_t count,
                           ccl::datatype dtype,
                           int root,
                           ccl_comm* comm,
                           ccl_stream* global_stream,
                           const ccl::vector_class<ccl::event>& deps) {
    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("|MT|: invoking broadcast_small");
        coll_initExt(comm, ccl::global_data::get().shared_data->hash_table, global_stream);
    }
    else {
        LOG_DEBUG("invoking broadcast_small");
        coll_init(comm, global_stream);
    }

    auto lambda = [&]<typename T, int NE, int NP>() {
        return broadcast_small_impl<T, NE, NP>(
            send_buf, recv_buf, count, dtype, root, comm, global_stream, deps);
    };

    return invoke_collective(lambda, comm, dtype);
}
