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
#pragma once

#include <mutex>
#include <unordered_map>

#include "oneapi/ccl/types.hpp"
#include "common/log/log.hpp"
#include "common/utils/spinlock.hpp"

enum class ccl_reduction_type_internal : int {
    ccl_sum = 0,
    ccl_prod,
    ccl_min,
    ccl_max,
    ccl_avg,
    ccl_pre_mul_sum,
    ccl_num_redops
};

struct ccl_reduction_data {
    ccl_reduction_type_internal op_type{};
    bool scalar_arg_is_ptr{ true };
    uint64_t scalar_arg{ 0 };
};

class ccl_reduction_type_storage {
    using ccl_redtype_lock_t = ccl_spinlock;
    using ccl_redtype_table_t = std::unordered_map<int, ccl_reduction_data>;

public:
    ccl_reduction_type_storage() = default;
    ~ccl_reduction_type_storage();

    ccl_reduction_type_storage(const ccl_reduction_type_storage& other) = delete;
    ccl_reduction_type_storage& operator=(const ccl_reduction_type_storage& other) = delete;

    // create and free custom reduction types
    int create(ccl_reduction_data redop);
    void free(int idx);

    // convert ccl::reduction enum to storage index and vice versa
    static int type_to_idx(ccl::reduction rtype) {
        return static_cast<int>(rtype) - static_cast<int>(ccl::reduction::predefined_ops);
    }
    static ccl::reduction idx_to_type(int idx) {
        ccl::reduction custom_reduction =
            static_cast<ccl::reduction>(static_cast<int>(ccl::reduction::predefined_ops) + idx);
        CCL_THROW_IF_NOT(custom_reduction < ccl::reduction::max_ops,
                         "reduction type is not a valid enum");
        return custom_reduction;
    }

    // get stored value/scalar from custom reduction type
    const ccl_reduction_data& get(ccl::reduction idx);
    // check if the reduction type is user-defined
    static bool is_custom(ccl::reduction idx) {
        CCL_THROW_IF_NOT(idx < ccl::reduction::max_ops, "reduction type is not a valid enum");
        if (idx < ccl::reduction::predefined_ops) {
            return false;
        }
        return true;
    }
    // convert public reduction enum to internal enum
    static ccl_reduction_type_internal convert_to_internal(ccl::reduction idx) {
        CCL_THROW_IF_NOT(idx < ccl::reduction::predefined_ops,
                         "reduction type is not a predefined type");
        if (idx == ccl::reduction::sum)
            return ccl_reduction_type_internal::ccl_sum;
        if (idx == ccl::reduction::prod)
            return ccl_reduction_type_internal::ccl_prod;
        if (idx == ccl::reduction::min)
            return ccl_reduction_type_internal::ccl_min;
        if (idx == ccl::reduction::max)
            return ccl_reduction_type_internal::ccl_max;
        if (idx == ccl::reduction::avg)
            return ccl_reduction_type_internal::ccl_avg;
        return static_cast<ccl_reduction_type_internal>(idx);
    }

    ccl_redtype_lock_t guard{};
    ccl_redtype_table_t custom_table{};
};

/* implementation of public API (user-defined reductions) */

void reduction_create_pre_mul_sum_impl(ccl::reduction* rtype,
                                       void* scalar,
                                       ccl::datatype dtype,
                                       ccl::scalar_residence_type residence);
void reduction_destroy_impl(ccl::reduction rtype);
