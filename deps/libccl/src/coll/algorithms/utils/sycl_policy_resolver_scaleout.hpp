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

#pragma once

#include "coll/algorithms/algorithm_utils.hpp"      // ccl_coll_type
#include "coll/algorithms/utils/sycl_policy.hpp"    // policy_query
#include "coll/algorithms/utils/sycl_selection.hpp" // tune_attr structs

#include <utility>

class ccl_comm;

namespace ccl::sycl_policy_resolver {

class scaleout_resolver {
public:
    explicit scaleout_resolver(sycl_policy::policy_query q) noexcept
        : query(std::move(q)) {}

    sycl_scaleout_tune_attr allgatherv()     const;
    sycl_scaleout_tune_attr allreduce()      const;
    sycl_scaleout_tune_attr reduce_scatter() const;
    sycl_scaleout_tune_attr alltoall()       const;

    sycl_policy::policy_query query;
};

}  // namespace ccl::sycl_policy_resolver
