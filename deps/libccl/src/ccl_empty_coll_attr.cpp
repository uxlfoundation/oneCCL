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

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/aliases.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/type_traits.hpp"

#include "oneapi/ccl/coll_attr_ids.hpp"
#include "oneapi/ccl/coll_attr_ids_traits.hpp"
#include "oneapi/ccl/coll_attr.hpp"

namespace ccl {

namespace v1 {

template <class attr>
CCL_API attr ccl_empty_attr::create_empty() {
    return attr{ ccl_empty_attr::version };
}

CCL_API allgather_attr default_allgather_attr = ccl_empty_attr::create_empty<allgather_attr>();
CCL_API allgatherv_attr default_allgatherv_attr = ccl_empty_attr::create_empty<allgatherv_attr>();
CCL_API allreduce_attr default_allreduce_attr = ccl_empty_attr::create_empty<allreduce_attr>();
CCL_API alltoall_attr default_alltoall_attr = ccl_empty_attr::create_empty<alltoall_attr>();
CCL_API alltoallv_attr default_alltoallv_attr = ccl_empty_attr::create_empty<alltoallv_attr>();
CCL_API barrier_attr default_barrier_attr = ccl_empty_attr::create_empty<barrier_attr>();
CCL_API broadcast_attr default_broadcast_attr = ccl_empty_attr::create_empty<broadcast_attr>();
CCL_API pt2pt_attr default_pt2pt_attr = ccl_empty_attr::create_empty<pt2pt_attr>();
CCL_API reduce_attr default_reduce_attr = ccl_empty_attr::create_empty<reduce_attr>();
CCL_API reduce_scatter_attr default_reduce_scatter_attr =
    ccl_empty_attr::create_empty<reduce_scatter_attr>();
CCL_API scatter_attr default_scatter_attr = ccl_empty_attr::create_empty<scatter_attr>();

} // namespace v1

} // namespace ccl
