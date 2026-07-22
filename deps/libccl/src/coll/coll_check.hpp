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

#include "coll/coll_param.hpp"

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_SYCL
void ccl_check_usm_pointers(const std::vector<void*>& ptrs,
                            const sycl::device& dev,
                            const sycl::context& ctx);
#endif // CCL_ENABLE_SYCL

void ccl_coll_validate_user_input(const ccl_coll_param& param, const ccl_coll_attr& attr);
