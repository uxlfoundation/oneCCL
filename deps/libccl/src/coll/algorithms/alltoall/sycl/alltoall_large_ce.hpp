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
/*
* AlltoAll for large payload using copy engine
*/

#include "coll/algorithms/algorithm_utils.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

ccl::event alltoall_large_copy_engine(sycl::queue &q,
                                      ccl_comm* comm,
                                      const std::array<void *, MAX_NODE_RANKS> &send_bufs,
                                      std::array<void *, MAX_NODE_RANKS> &recv_bufs,
                                      const size_t count,
                                      ccl::datatype dtype,
                                      std::vector<size_t> &offsets,
                                      const size_t rank,
                                      const size_t comm_size,
                                      bool is_multi_thread = false);
