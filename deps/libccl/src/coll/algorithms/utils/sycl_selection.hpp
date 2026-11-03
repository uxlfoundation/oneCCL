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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/environment.hpp"
#include "comm/comm.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "comm/comm_interface.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#include "common/global/global.hpp"
#include "coll/algorithms/utils/sycl_policy.hpp"  // ccl::sycl_policy::algorithm
#include "coll/selection/selection.hpp"

// dispatching between SYCL path and Scheduler path
bool can_use_sycl_kernels(const ccl_selector_param& param);

using sycl_scaleout_algo = ccl::sycl_policy::algorithm;
using scaleout_sub_kernel = ccl::sycl_policy::scaleout_sub_kernel;

// One tune_attr for every collective. Collectives that don't use a given axis
// simply leave that field at its default and ignore it on read.
//   allreduce:      algo (direct|rabenseifner|ring|proxy_direct|proxy_ring), chunk
//   reduce_scatter: algo (direct|ring|proxy_ring),                          chunk
//   allgatherv:     algo (direct|ring|proxy_ring),                          chunk, sub_kernel
//   alltoall(v):    algo (direct|gdr_only_pairwise|scatter|numa_gdr_split|fallback)
//   scatter:        algo (direct),                                                 sub_kernel
class sycl_scaleout_tune_attr {
public:
    sycl_scaleout_tune_attr() = default;
    explicit sycl_scaleout_tune_attr(sycl_scaleout_algo a) noexcept
        : algo_(a) {}
    sycl_scaleout_tune_attr(sycl_scaleout_algo a, size_t chunk_bytes) noexcept
        : algo_(a), pipeline_chunk_size_(chunk_bytes) {}

    sycl_scaleout_algo  algo()                const noexcept { return algo_; }
    size_t              pipeline_chunk_size() const noexcept { return pipeline_chunk_size_; }
    scaleout_sub_kernel sub_kernel()          const noexcept { return sub_kernel_; }

    sycl_scaleout_tune_attr& set_algo(sycl_scaleout_algo a) noexcept { algo_ = a; return *this; }
    sycl_scaleout_tune_attr& set_pipeline_chunk_size(size_t s) noexcept    { pipeline_chunk_size_ = s; return *this; }
    sycl_scaleout_tune_attr& set_sub_kernel(scaleout_sub_kernel k) noexcept { sub_kernel_ = k; return *this; }

private:
    sycl_scaleout_algo  algo_{ sycl_scaleout_algo::direct };
    size_t              pipeline_chunk_size_{ 2 * 1024 * 1024 };
    scaleout_sub_kernel sub_kernel_{ scaleout_sub_kernel::direct };
};


class sycl_selector {
public:
    sycl_selector(size_t sz,
                  size_t cs,
                  ccl_datatype dt,
                  bool rec) noexcept
        : size(sz), comm_size(cs), dtype(dt), is_recording(rec) {}

    sycl_scaleout_tune_attr allreduce()      const;
    sycl_scaleout_tune_attr reduce_scatter() const;
    sycl_scaleout_tune_attr allgatherv()     const;
    sycl_scaleout_tune_attr alltoall()       const;

    static size_t allreduce_chunk_size(sycl_scaleout_algo algo, size_t size, size_t comm_size);
    static size_t reduce_scatter_chunk_size(sycl_scaleout_algo algo, size_t size, size_t comm_size);
    static size_t allgatherv_chunk_size(sycl_scaleout_algo algo, size_t size, size_t comm_size);

    size_t size;
    size_t comm_size;
    ccl_datatype dtype;
    bool is_recording;
};
