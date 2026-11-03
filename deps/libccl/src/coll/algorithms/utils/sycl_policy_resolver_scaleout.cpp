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

#include "coll/algorithms/utils/sycl_policy_resolver_scaleout.hpp"

#include "coll/algorithms/utils/sycl_policy.hpp"

namespace ccl::sycl_policy_resolver {

sycl_scaleout_tune_attr scaleout_resolver::allgatherv() const {
    return sycl_selector{query.msg_bytes, query.r2r_comm_size,
                         query.dtype,     query.is_recording}.allgatherv();
}

sycl_scaleout_tune_attr scaleout_resolver::allreduce() const {
    return sycl_selector{query.msg_bytes, query.r2r_comm_size,
                         query.dtype,     query.is_recording}.allreduce();
}

sycl_scaleout_tune_attr scaleout_resolver::reduce_scatter() const {
    return sycl_selector{query.msg_bytes, query.r2r_comm_size,
                         query.dtype,     query.is_recording}.reduce_scatter();
}

sycl_scaleout_tune_attr scaleout_resolver::alltoall() const {
    return sycl_selector{query.msg_bytes, query.r2r_comm_size,
                         query.dtype,     query.is_recording}.alltoall();
}

}  // namespace ccl::sycl_policy_resolver
