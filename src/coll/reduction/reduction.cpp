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
#include "coll/reduction/reduction.hpp"
#include "common/global/global.hpp"
#include "comm/comm.hpp"

ccl_reduction_type_storage::~ccl_reduction_type_storage() {
    // free all user-defined reductions
    custom_table.clear();
}

int ccl_reduction_type_storage::create(ccl_reduction_data redop) {
    std::lock_guard<ccl_redtype_lock_t> lock{ guard };

    const int start_index = 1;
    const int max_index = static_cast<int>(ccl::reduction::max_ops) -
                          static_cast<int>(ccl::reduction::predefined_ops);

    // if table is not full, find the lowest available index
    if (static_cast<int>(custom_table.size()) < max_index) {
        for (int idx = start_index; idx <= max_index; ++idx) {
            if (custom_table.find(idx) == custom_table.end()) {
                custom_table[idx] = redop;
                return idx;
            }
        }
    }
    CCL_THROW("no available slot for new custom reduction type");
}

void ccl_reduction_type_storage::free(int idx) {
    std::lock_guard<ccl_redtype_lock_t> lock{ guard };
    // remove the custom reduction type from the table
    auto it = custom_table.find(idx);
    if (it != custom_table.end()) {
        custom_table.erase(it);
    }
}

const ccl_reduction_data& ccl_reduction_type_storage::get(ccl::reduction rtype) {
    std::lock_guard<ccl_redtype_lock_t> lock{ guard };
    // convert ccl::reduction enum to storage index
    int index_to_find = ccl_reduction_type_storage::type_to_idx(rtype);
    // look up the custom reduction operation
    auto it = custom_table.find(index_to_find);
    if (it != custom_table.end()) {
        return it->second;
    }
    CCL_THROW(
        "user-defined reduction type with index ", index_to_find, " not found in the storage");
}

void reduction_create_pre_mul_sum_impl(ccl::reduction* rtype,
                                       void* scalar,
                                       ccl::datatype dtype,
                                       ccl::scalar_residence_type residence) {
    // register new reduction type
    ccl_reduction_data user_reduction;
    user_reduction.op_type = ccl_reduction_type_internal::ccl_pre_mul_sum;
    if (residence == ccl::scalar_residence_type::scalar_device) {
        user_reduction.scalar_arg_is_ptr = true; // scalar is a pointer to the value
        user_reduction.scalar_arg = reinterpret_cast<uint64_t>(scalar);
    }
    else {
        int dtype_size = ccl::global_data::get().dtypes->get(dtype).size();
        user_reduction.scalar_arg_is_ptr = false; // scalar is a value
        std::memcpy(&user_reduction.scalar_arg, scalar, dtype_size);
    }

    int custom_index = ccl::global_data::get().redtype_storage->create(user_reduction);
    // convert registered index to ccl::reduction enum
    *rtype = ccl_reduction_type_storage::idx_to_type(custom_index);
}

void reduction_destroy_impl(ccl::reduction rtype) {
    CCL_THROW_IF_NOT(ccl_reduction_type_storage::is_custom(rtype),
                     "cannot destroy built-in reduction type");
    // erase the custom reduction type from the storage
    int index_to_free = ccl_reduction_type_storage::type_to_idx(rtype);
    ccl::global_data::get().redtype_storage->free(index_to_free);
}
