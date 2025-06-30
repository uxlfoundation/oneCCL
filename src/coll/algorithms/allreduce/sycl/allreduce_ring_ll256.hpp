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

#define ARC_NUM (8)

#define LL256_BUF_SIZE    (32 * 1024 * 1024)
#define GATHER_BUF_OFFSET (LL256_BUF_SIZE / 2)

ccl::event arc_allreduce(const void *src,
                         void *dst,
                         size_t count,
                         ccl::datatype dtype,
                         ccl::reduction reduction,
                         ccl_comm *comm,
                         ccl_stream *global_stream);
