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

#include <cstring>

#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.h"
#include "common/log/log.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/kvs.hpp"

namespace ccl {

class base_kvs_impl {
public:
    base_kvs_impl() = default;
    base_kvs_impl(const kvs::address_type& addr);

    virtual vector_class<char> get(const string_class& key) {
        CCL_THROW("get() is not implemented");
    }

    virtual void set(const string_class& key, const vector_class<char>& data) {
        CCL_THROW("set() is not implemented");
    }

    kvs::address_type& get_addr() {
        return addr;
    }
    const kvs::address_type& get_addr() const {
        return addr;
    }

    virtual ~base_kvs_impl() = default;

protected:
    kvs::address_type addr;
};

class native_kvs_impl : public base_kvs_impl {
public:
    native_kvs_impl(const kvs_attr& attr = default_kvs_attr);

    native_kvs_impl(const kvs::address_type& addr, const kvs_attr& attr = default_kvs_attr);

    vector_class<char> get(const string_class& key) override;

    void set(const string_class& key, const vector_class<char>& data) override;

    std::shared_ptr<internal_kvs> get() const;

private:
    const std::string prefix = "USER_DATA";
    std::shared_ptr<internal_kvs> inter_kvs;
};

template <class T>
const T* get_kvs_impl_typed(std::shared_ptr<ccl::kvs> kvs) {
    auto kvs_impl = dynamic_cast<const T*>(&kvs->get_impl());
    CCL_THROW_IF_NOT(kvs_impl != nullptr, "kvs impl doesn't correspond to the type");

    return kvs_impl;
}

} // namespace ccl
