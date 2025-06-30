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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/environment.hpp"
#include "comm/comm.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "comm/comm_interface.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#include "common/global/global.hpp"
#include "coll/selection/selection.hpp"

// dispatching between SYCL path and Scheduler path
bool can_use_sycl_kernels(const ccl_selector_param& param);

// scale-up attributes used for various purposes
struct sycl_coll_scaleup_attr {
    bool wait_on_deps{ false };
    bool force_use_tmp{ false };
};

// scale-out collective tuning parameters
enum class allreduce_scaleout_algo { direct, rabenseifner, ring };
enum class reduce_scatter_scaleout_algo { direct, ring };
enum class allgatherv_scaleout_algo { direct, ring };

struct sycl_allreduce_tune_attr {
    allreduce_scaleout_algo algo{ allreduce_scaleout_algo::direct };
    size_t pipeline_chunk_size{ 2 * 1024 * 1024 };
};

struct sycl_reduce_scatter_tune_attr {
    reduce_scatter_scaleout_algo algo{ reduce_scatter_scaleout_algo::direct };
    size_t pipeline_chunk_size{ 2 * 1024 * 1024 };
};

struct sycl_allgatherv_tune_attr {
    allgatherv_scaleout_algo algo{ allgatherv_scaleout_algo::direct };
    size_t pipeline_chunk_size{ 2 * 1024 * 1024 };
};

// alleduce
size_t allreduce_select_chunk_size(allreduce_scaleout_algo algo, size_t size, size_t comm_size);
sycl_allreduce_tune_attr allreduce_select_tune_attr(size_t size,
                                                    size_t comm_size,
                                                    ccl_datatype ccl_dtype);

// reduce-scatter
sycl_reduce_scatter_tune_attr reduce_scatter_select_tune_attr(size_t size,
                                                              size_t comm_size,
                                                              ccl_datatype ccl_dtype);
size_t reduce_scatter_select_chunk_size(reduce_scatter_scaleout_algo algo,
                                        size_t size,
                                        size_t comm_size);

// allgatherv
sycl_allgatherv_tune_attr allgatherv_select_tune_attr(size_t size,
                                                      size_t comm_size,
                                                      ccl_datatype ccl_dtype);

size_t allgatherv_select_chunk_size(allgatherv_scaleout_algo algo, size_t size, size_t comm_size);
