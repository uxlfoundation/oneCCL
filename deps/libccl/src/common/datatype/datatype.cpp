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

#include <limits>

#include "common/datatype/datatype.hpp"
#include "common/global/global.hpp"
#include "common/utils/enums.hpp"
#include "exec/exec.hpp"

const ccl::datatype last_predefined_dt = ccl::datatype::bfloat16;

namespace ccl {
using datatype_str_enum =
    ccl::utils::enum_to_str<ccl::utils::enum_to_underlying(last_predefined_dt) + 1>;
string_class to_string(const datatype& dt) {
    return datatype_str_enum({ "INT8",
                               "UINT8",
                               "INT16",
                               "UINT16",
                               "INT32",
                               "UINT32",
                               "INT64",
                               "UINT64",
                               "FLOAT16",
                               "FLOAT32",
                               "FLOAT64",
                               "BFLOAT16" })
        .choose(dt, "CUSTOM_TYPE");
}
} // namespace ccl

ccl_datatype ccl_datatype_int8;

ccl::datatype& operator++(ccl::datatype& d) {
    using IntType = typename std::underlying_type<ccl::datatype>::type;
    d = static_cast<ccl::datatype>(static_cast<IntType>(d) + 1);
    return d;
}

ccl::datatype operator++(ccl::datatype& d, int) {
    ccl::datatype tmp(d);
    ++d;
    return tmp;
}

std::ostream& operator<<(std::ostream& os, const ccl::datatype& dt) {
    os << ccl::to_string(dt);
    return os;
}

ccl_datatype::ccl_datatype(ccl::datatype idx, size_t size) : m_idx(idx), m_size(size) {
    CCL_THROW_IF_NOT(m_size > 0, "unexpected datatype size ", m_size);
}

ccl_datatype_storage::ccl_datatype_storage() {
    LOG_DEBUG("create datatype_storage");

    using IntType = typename std::underlying_type<ccl::datatype>::type;
    custom_idx = static_cast<ccl::datatype>(static_cast<IntType>(last_predefined_dt) + 1);

    size_t size = 0;
    std::string name_str;

    for (ccl::datatype idx = ccl::datatype::int8; idx <= last_predefined_dt; idx++) {
        /* fill table with predefined datatypes */
        size = (idx == ccl::datatype::int8)       ? sizeof(int8_t)
               : (idx == ccl::datatype::uint8)    ? sizeof(uint8_t)
               : (idx == ccl::datatype::int16)    ? sizeof(int16_t)
               : (idx == ccl::datatype::uint16)   ? sizeof(uint16_t)
               : (idx == ccl::datatype::int32)    ? sizeof(int32_t)
               : (idx == ccl::datatype::uint32)   ? sizeof(uint32_t)
               : (idx == ccl::datatype::int64)    ? sizeof(int64_t)
               : (idx == ccl::datatype::uint64)   ? sizeof(uint64_t)
               : (idx == ccl::datatype::float16)  ? sizeof(uint16_t)
               : (idx == ccl::datatype::float32)  ? sizeof(float)
               : (idx == ccl::datatype::float64)  ? sizeof(double)
               : (idx == ccl::datatype::bfloat16) ? sizeof(uint16_t)
                                                  : 0;

        CCL_ASSERT(size > 0, "Unexpected data type size: ", size, ", for idx: ", idx);
        name_str = (idx == ccl::datatype::int8)       ? "int8"
                   : (idx == ccl::datatype::uint8)    ? "uint8"
                   : (idx == ccl::datatype::int16)    ? "int16"
                   : (idx == ccl::datatype::uint16)   ? "uint16"
                   : (idx == ccl::datatype::int32)    ? "int32"
                   : (idx == ccl::datatype::uint32)   ? "uint32"
                   : (idx == ccl::datatype::int64)    ? "int64"
                   : (idx == ccl::datatype::uint64)   ? "uint64"
                   : (idx == ccl::datatype::float16)  ? "float16"
                   : (idx == ccl::datatype::float32)  ? "float32"
                   : (idx == ccl::datatype::float64)  ? "float64"
                   : (idx == ccl::datatype::bfloat16) ? "bfloat16"
                                                      : "";

        create_internal(predefined_table, idx, size, name_str);

        const ccl_datatype& dtype = get(idx);
        const std::string& dtype_name = name(dtype);

        CCL_THROW_IF_NOT(dtype == idx, "unexpected datatype idx ", dtype.idx(), ", expected ", idx);
        CCL_THROW_IF_NOT(
            dtype.size() == size, "unexpected datatype size ", dtype.size(), ", expected ", size);
        CCL_THROW_IF_NOT(!dtype_name.compare(name_str),
                         "unexpected datatype name ",
                         dtype_name,
                         ", expected ",
                         name_str);
    }

    ccl_datatype_int8 = get(ccl::datatype::int8);
}

ccl_datatype_storage::~ccl_datatype_storage() {
    std::lock_guard<ccl_datatype_lock_t> lock{ guard };
    predefined_table.clear();
    custom_table.clear();
}

void ccl_datatype_storage::create_internal(ccl_datatype_table_t& table,
                                           ccl::datatype idx,
                                           size_t size,
                                           const std::string& name) {
    CCL_THROW_IF_NOT(table.find(idx) == table.end(), "datatype index is busy, idx ", idx);
    table[idx] = std::make_pair(ccl_datatype(idx, size), name);
    // LOG_DEBUG("created datatype idx: ", idx, ", size: ", size, ", name: ", name);
}

ccl::datatype ccl_datatype_storage::create_by_datatype_size(size_t datatype_size) {
    std::lock_guard<ccl_datatype_lock_t> lock{ guard };

    while (custom_table.find(custom_idx) != custom_table.end() ||
           is_predefined_datatype(custom_idx)) {
        custom_idx++;
        if (custom_idx < ccl::datatype::int8)
            custom_idx = ccl::datatype::int8;
    }

    CCL_ASSERT(datatype_size > 0);
    create_internal(
        custom_table,
        custom_idx,
        datatype_size,
        std::string("DTYPE_") + std::to_string(ccl::utils::enum_to_underlying(custom_idx)));

    return custom_idx;
}

ccl::datatype ccl_datatype_storage::create(const ccl::datatype_attr& attr) {
    size_t size = attr.get<ccl::datatype_attr_id::size>();
    return create_by_datatype_size(size);
}

void ccl_datatype_storage::free(ccl::datatype idx) {
    std::lock_guard<ccl_datatype_lock_t> lock{ guard };

    if (is_predefined_datatype(idx)) {
        CCL_THROW("attempt to free predefined datatype idx ", idx);
        return;
    }

    if (custom_table.find(idx) == custom_table.end()) {
        CCL_THROW("attempt to free non-existing datatype idx ", idx);
        return;
    }

    LOG_DEBUG("free datatype idx ", idx);
    custom_table.erase(idx);
}

const ccl_datatype& ccl_datatype_storage::get(ccl::datatype idx) const {
    if (is_predefined_datatype(idx)) {
        auto it = predefined_table.find(idx);
        if (it != predefined_table.end()) {
            return it->second.first;
        }
        LOG_WARN("unexpected idx for ccl_datatype_storage::get(), returning default datatype");
        return default_datatype;
    }
    std::lock_guard<ccl_datatype_lock_t> lock{ guard };
    auto it = custom_table.find(idx);
    if (it != custom_table.end()) {
        return it->second.first;
    }
    LOG_WARN("unexpected idx for ccl_datatype_storage::get(), returning default datatype");
    return default_datatype;
}

const std::string& ccl_datatype_storage::name(const ccl_datatype& dtype) const {
    ccl::datatype idx = dtype.idx();
    if (is_predefined_datatype(idx)) {
        auto it = predefined_table.find(idx);
        if (it != predefined_table.end()) {
            return it->second.second;
        }
        LOG_WARN(
            "unexpected datatype for ccl_datatype_storage::name(), returning \"undefined\" type name");
        return default_type_str;
    }
    std::lock_guard<ccl_datatype_lock_t> lock{ guard };
    auto it = custom_table.find(idx);
    if (it != custom_table.end()) {
        return it->second.second;
    }
    LOG_WARN(
        "unexpected datatype for ccl_datatype_storage::name(), returning \"undefined\" type name");
    return default_type_str;
}

const std::string& ccl_datatype_storage::name(ccl::datatype idx) const {
    return name(get(idx));
}

bool ccl_datatype_storage::is_predefined_datatype(ccl::datatype idx) {
    return (idx >= ccl::datatype::int8 && idx <= last_predefined_dt) ? true : false;
}
