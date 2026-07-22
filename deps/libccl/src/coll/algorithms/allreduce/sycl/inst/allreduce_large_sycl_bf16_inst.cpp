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


// Per-dtype explicit instantiations; see matching extern template
// block at the bottom of the included impl header.

#include "coll/algorithms/allreduce/sycl/allreduce_large_sycl_impl.hpp"
#include "coll/algorithms/allreduce/sycl/allreduce_large_sycl_ring.hpp"

#ifdef CCL_SYCL_VEC_SUPPORT_BF16
TEMPLATE_ALLREDUCE_LARGE(sycl::ext::oneapi::bfloat16, true);
TEMPLATE_ALLREDUCE_LARGE(sycl::ext::oneapi::bfloat16, false);
TEMPLATE_ALLREDUCE_LARGE_SU_RING(sycl::ext::oneapi::bfloat16);
TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(sycl::ext::oneapi::bfloat16);
#endif
