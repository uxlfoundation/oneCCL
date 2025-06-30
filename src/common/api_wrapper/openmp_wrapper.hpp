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

#if defined(CCL_ENABLE_MPI)

#include "coll/coll_param.hpp"
#include "oneapi/ccl.hpp"
#include "oneapi/ccl/types.hpp"

namespace ccl {

typedef struct openmp_lib_ops {
    void (*init)(const std::string& omp_allreduce_num_threads);
    void (*allreduce)(const void* send_buf,
                      void* recv_buf,
                      size_t count,
                      ccl::datatype dtype,
                      ccl::reduction reduction,
                      const ccl_coll_attr& attr,
                      ccl_comm* comm,
                      const ccl_stream* stream,
                      const std::vector<ccl::event>& deps);
    void (*allgatherv)(const void* send_buf,
                       size_t send_len,
                       void* recv_buf,
                       const size_t* recv_lens,
                       const size_t* offsets,
                       const ccl_coll_attr& attr,
                       ccl_comm* comm,
                       const ccl_stream* stream,
                       const std::vector<ccl::event>& deps);
    int (*thread_num)();
} openmp_lib_ops_t;

static std::vector<std::string> openmp_fn_names = {
    // The name must contain `ccl` in order for the symbol
    // to be visible externally (i.e from libccl.so).
    // Please see ccl.map for more details.
    "ccl_openmp_init",
    "ccl_openmp_allreduce",
    "ccl_openmp_allgatherv",
    "ccl_openmp_thread_num"
};

extern openmp_lib_ops_t openmp_lib_ops;

bool openmp_api_init();
void openmp_api_fini();

} // namespace ccl

#endif //CCL_ENABLE_MPI
