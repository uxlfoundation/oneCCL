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

namespace ccl {
namespace v1 {

event scatter_sycl_single_node(sycl::queue& q,
                               const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               datatype dtype,
                               int root,
                               ccl_comm* comm,
                               ccl_stream* global_stream,
                               const vector_class<event>& deps,
                               bool& done);

event scatter_sycl(sycl::queue q,
                   const void* send_buf,
                   void* recv_buf,
                   size_t count,
                   datatype dtype,
                   int root,
                   ccl_comm* global_comm,
                   ccl_stream* global_stream,
                   const scatter_attr& attr,
                   const vector_class<event>& deps,
                   bool& done);

} // namespace v1
} // namespace ccl
