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

#ifdef CCL_ENABLE_STUB_BACKEND

#include "stub_kvs_impl.hpp"

namespace ccl {

static int get_unique_id() {
    const char* env_val = getenv("CCL_STUB_BACKEND_COMM_ID");
    int id = 0;
    if (env_val != nullptr)
        id = atoi(env_val);

    return id;
}

static kvs::address_type convert_id_to_addr(int id) {
    kvs::address_type addr{ 0 };

    memcpy(addr.data(), &id, sizeof(id));

    return addr;
}

static int convert_addr_to_id(const kvs::address_type& addr) {
    int id = 0;
    memcpy(&id, addr.data(), sizeof(id));

    return id;
}

stub_kvs_impl::stub_kvs_impl() : base_kvs_impl(convert_id_to_addr(get_unique_id())) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::stub, "unexpected backend");
}

stub_kvs_impl::stub_kvs_impl(const kvs::address_type& addr) : base_kvs_impl(addr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::stub, "unexpected backend");
}

int stub_kvs_impl::get_id() const {
    return convert_addr_to_id(base_kvs_impl::get_addr());
}

} // namespace ccl

#endif // CCL_ENABLE_STUB_BACKEND
