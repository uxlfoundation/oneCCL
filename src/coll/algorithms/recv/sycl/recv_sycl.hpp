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

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"

namespace ccl {
namespace v1 {

ccl::event recv_sycl(sycl::queue& q,
                     void* recv_buf,
                     size_t count,
                     ccl::datatype dtype,
                     int peer_rank,
                     ccl_comm* global_comm,
                     ccl_stream* global_stream,
                     const pt2pt_attr& attr,
                     const vector_class<event>& deps,
                     bool& done);
} // namespace v1
} // namespace ccl
