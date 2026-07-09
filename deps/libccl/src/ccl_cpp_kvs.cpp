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

#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/types_policy.hpp"

#include "oneapi/ccl/kvs_attr_ids.hpp"
#include "oneapi/ccl/kvs_attr_ids_traits.hpp"
#include "oneapi/ccl/kvs_attr.hpp"

#include "common/global/global.hpp"

#include "kvs_impl.hpp"
#ifdef CCL_ENABLE_STUB_BACKEND
#include "stub_kvs_impl.hpp"
#endif // CCL_ENABLE_STUB_BACKEND

namespace ccl {
base_kvs_impl::base_kvs_impl(const kvs::address_type& addr) : addr(addr) {}

native_kvs_impl::native_kvs_impl(const kvs_attr& attr) : base_kvs_impl() {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "incorrect non-native backend is used");

    inter_kvs.reset(new internal_kvs());
    if (attr.is_valid<kvs_attr_id::ip_port>()) {
        std::string server_address = attr.get<kvs_attr_id::ip_port>();
        inter_kvs->set_server_address(server_address);
    }

    inter_kvs->kvs_main_server_address_reserve(get_addr().data());
    inter_kvs->kvs_init(get_addr().data());
}

native_kvs_impl::native_kvs_impl(const kvs::address_type& addr, const kvs_attr& attr)
        : base_kvs_impl(addr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "incorrect non-native backend is used");

    inter_kvs.reset(new internal_kvs());
    if (attr.is_valid<kvs_attr_id::ip_port>()) {
        std::string server_address = attr.get<kvs_attr_id::ip_port>();
        inter_kvs->set_server_address(server_address);
    }

    inter_kvs->kvs_init(get_addr().data());
}

vector_class<char> native_kvs_impl::get(const string_class& key) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "incorrect non-native backend is used");

    std::string ret;
    CCL_THROW_IF_NOT(inter_kvs->kvs_get_value_by_name_key(prefix.c_str(), key.c_str(), ret) ==
                         KVS_STATUS_SUCCESS,
                     "kvs get failed");
    return { ret.begin(), ret.end() };
}

void native_kvs_impl::set(const string_class& key, const vector_class<char>& data) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "incorrect non-native backend is used");

    CCL_THROW_IF_NOT(!data.empty(), "data should have at least one element");
    CCL_THROW_IF_NOT(data.back() == '\0', "data should have terminating symbol");
    CCL_THROW_IF_NOT(data.data(), "data pointer should be non-null");
    inter_kvs->kvs_set_value(prefix.c_str(), key.c_str(), data.data());
}

std::shared_ptr<internal_kvs> native_kvs_impl::get() const {
    return inter_kvs;
}

namespace v1 {

kvs::address_type CCL_API kvs::get_address() const {
    return pimpl->get_addr();
}

vector_class<char> CCL_API kvs::get(const string_class& key) {
    return pimpl->get(key);
}

void CCL_API kvs::set(const string_class& key, const vector_class<char>& data) {
    pimpl->set(key, data);
}

int CCL_API kvs::get_id() {
    return id;
}

static base_kvs_impl* get_kvs_impl(const kvs::address_type& addr, const kvs_attr& attr) {
#ifdef CCL_ENABLE_STUB_BACKEND
    if (ccl::global_data::env().backend == backend_mode::stub) {
        // in case of stub kvs we use a limited functionality and ignore attr parameter
        return new stub_kvs_impl(addr);
    }
#endif // CCL_ENABLE_STUB_BACKEND

    return new native_kvs_impl(addr, attr);
}

static base_kvs_impl* get_kvs_impl(const kvs_attr& attr) {
#ifdef CCL_ENABLE_STUB_BACKEND
    if (ccl::global_data::env().backend == backend_mode::stub) {
        // in case of stub kvs we use a limited functionality and ignore attr parameter
        return new stub_kvs_impl();
    }
#endif // CCL_ENABLE_STUB_BACKEND

    return new native_kvs_impl(attr);
}

CCL_API kvs::kvs(const kvs::address_type& addr, const kvs_attr& attr)
        : pimpl(get_kvs_impl(addr, attr)) {}

CCL_API const base_kvs_impl& kvs::get_impl() {
    return *pimpl;
}

std::atomic<int> kvs::id_counter{ 0 };
CCL_API kvs::kvs(const kvs_attr& attr) : pimpl(get_kvs_impl(attr)), id(id_counter++) {}

CCL_API kvs::~kvs() {}

} // namespace v1

} // namespace ccl
