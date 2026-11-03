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

#include "coll/algorithms/algorithm_utils.hpp"       // ccl_coll_type
#include "coll/algorithms/utils/sycl_policy.hpp"     // policy_query, sycl_scaleup_tune_attr

#include <utility>

namespace ccl::sycl_policy_resolver {

class scaleup_resolver {
public:
    explicit scaleup_resolver(sycl_policy::policy_query q) noexcept
        : query(std::move(q)) {}

    sycl_policy::sycl_scaleup_tune_attr allgatherv()     const;
    sycl_policy::sycl_scaleup_tune_attr allreduce()      const;
    sycl_policy::sycl_scaleup_tune_attr reduce_scatter() const;
    sycl_policy::sycl_scaleup_tune_attr alltoall()       const;
    sycl_policy::sycl_scaleup_tune_attr alltoallv()      const;
    sycl_policy::sycl_scaleup_tune_attr broadcast()      const;
    sycl_policy::sycl_scaleup_tune_attr scatter()        const;
    sycl_policy::sycl_scaleup_tune_attr reduce()         const;

    sycl_policy::policy_query query;
};

}  // namespace ccl::sycl_policy_resolver
