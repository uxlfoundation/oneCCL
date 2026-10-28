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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ccl {
namespace v1 {

class communicator;
struct impl_dispatch;

/**
 * Device-accessible communicator data — rank topology and barrier resources.
 * Captured by value into the kernel lambda, so reading a rank costs no
 * indirection. The pointers inside address device memory.
 */
struct dev_comm_data {
    int rank;       /**< rank within the communicator (world team) */
    int nranks;     /**< size of the communicator (world team) */
    int lsa_rank;   /**< rank within the LSA team */
    int lsa_size;   /**< size of the LSA team */

    /**
     * Per-rank inbox of [n_barriers * nranks] uint32_t slots, indexed
     * barrier_inboxes[owner][barrier_index * nranks + sender]. One dedicated
     * slot per sender, so arrive() plain-stores its epoch into each peer's slot
     * and wait() polls its own — no contention, no RMW.
     */
    uint32_t** barrier_inboxes;
    int n_barriers; /**< number of barrier slots per rank */
    uint32_t* barrier_epochs; /**< per-slot epoch counter, persists across kernel launches */
};

static_assert(std::is_trivially_copyable<dev_comm_data>::value,
              "dev_comm_data is captured by value into kernels");

/** A device communicator is a value, not a handle — dev_comm_create fills one. */
typedef dev_comm_data dev_comm_t;

/** Requirements for creating a device communicator. */
struct dev_comm_requirements {
    int lsa_barrier_count; /**< number of barrier slots to allocate (e.g. nCTAs) */
};

} // namespace v1

using v1::dev_comm_data;
using v1::dev_comm_t;
using v1::dev_comm_requirements;

} // namespace ccl
