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

#include "coll/algorithms/utils/sycl_policy_resolver_scaleup.hpp"

#include "common/env/env.hpp"
#include "common/global/global.hpp"
#include "coll/algorithms/broadcast/sycl/broadcast_ring_auto_policy.hpp"
#include "comm/comm.hpp"

namespace ccl::sycl_policy_resolver {

using sycl_policy::scaleup_algo;
using sycl_policy::scaleup_protocol;
using sycl_policy::sycl_scaleup_tune_attr;

namespace {

// Shared shape used by allgatherv / allreduce / reduce_scatter — the "arc LL
// gate + ESIMD env + default" cascade. Each collective has its own
// `simple_threshold` env var; take a pointer to it as the parameter.
// `threshold_active` — when false, skip the msg_bytes-vs-threshold check.
// alltoallv gates the threshold on `sycl_alltoall_tmp_buf`.
sycl_scaleup_tune_attr resolve_ll_or_esimd_or_default(const sycl_policy::policy_query& q,
                                                     std::size_t simple_threshold,
                                                     bool threshold_active = true) {
    if (q.is_arc() &&
        (q.is_multi_thread() ||
         (threshold_active && q.msg_bytes <= simple_threshold) ||
         !q.has_p2p_access())) {
        return sycl_scaleup_tune_attr{ scaleup_protocol::ll, scaleup_algo::unspecified };
    }
#ifdef CCL_ENABLE_ESIMD
    if (ccl::global_data::env().sycl_esimd) {
        return sycl_scaleup_tune_attr{ scaleup_protocol::esimd, scaleup_algo::unspecified };
    }
#endif
    return sycl_scaleup_tune_attr{ scaleup_protocol::default_kernel, scaleup_algo::unspecified };
}

}  // namespace

// -----------------------------------------------------------------------------
// allgatherv scaleup
//
// Arc + (multi-thread || below simple threshold || no p2p) → LL256 chunked
// ring. Per-chunk oneshot/ring picked by dispatcher from CCL_SYCL_ALLGATHERV_LL,
// so `algo=simple` here. ESIMD only when arc conditions don't apply.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::allgatherv() const {
    return resolve_ll_or_esimd_or_default(
        query, ccl::global_data::env().sycl_allgatherv_simple_threshold);
}

// -----------------------------------------------------------------------------
// reduce_scatter scaleup — same shape as allgatherv.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::reduce_scatter() const {
    return resolve_ll_or_esimd_or_default(
        query, ccl::global_data::env().sycl_reduce_scatter_simple_threshold);
}

// -----------------------------------------------------------------------------
// reduce_scatter scaleup — same shape as allreduce.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::allreduce() const {
    return resolve_ll_or_esimd_or_default(
        query, ccl::global_data::env().sycl_allreduce_simple_threshold);
}

// -----------------------------------------------------------------------------
// alltoall scaleup — no ESIMD path. LL family internally cascades LL256 →
// arc_alltoall based on shape (dispatcher's decision).
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::alltoall() const {
    return resolve_ll_or_esimd_or_default(
        query, ccl::global_data::env().sycl_alltoall_simple_threshold);
}

// -----------------------------------------------------------------------------
// alltoallv scaleup — separate from alltoall because the gate is gated on
// `sycl_alltoall_tmp_buf` AND the threshold is compared against ELEMENTS
// (options.max_count), not bytes. Caller passes max_count as msg_bytes here.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::alltoallv() const {
    return resolve_ll_or_esimd_or_default(
        query,
        ccl::global_data::env().sycl_alltoall_simple_threshold,
        /*threshold_active=*/ccl::global_data::env().sycl_alltoall_tmp_buf);
}

// -----------------------------------------------------------------------------
// broadcast scaleup — Arc + (recording || !ring_env || !is_large) selects the
// RT256 ring (LL family). Otherwise the default small/large kernel.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::broadcast() const {
    if (!query.is_arc()) {
        return sycl_scaleup_tune_attr{ scaleup_protocol::default_kernel, scaleup_algo::unspecified };
    }
    const bool is_large = query.msg_bytes > ccl::global_data::env().sycl_broadcast_small_threshold;
    const bool use_ring = ccl::global_data::env().sycl_broadcast_ring;
    const bool use_ring_auto = ccl::global_data::env().sycl_broadcast_ring_auto;
    bool effective_use_ring = use_ring;
    // `use_ring_auto` consults a (world, size) lookup table to decide whether
    // the pipelined ring is a good fit; callers set `p2p_comm` so we can
    // read its size.
    if (use_ring && use_ring_auto && query.p2p_comm) {
        const broadcast_ring_auto_decision decision{ query.msg_bytes,
                                                      static_cast<int>(query.p2p_comm->size()) };
        effective_use_ring = decision.use_ring;
    }
    if (query.is_recording || !effective_use_ring || !is_large) {
        return sycl_scaleup_tune_attr{ scaleup_protocol::ll, scaleup_algo::ring };
    }
    return sycl_scaleup_tune_attr{ scaleup_protocol::default_kernel, scaleup_algo::unspecified };
}

// -----------------------------------------------------------------------------
// scatter scaleup — no arc/LL split. Just the default kernel; dispatcher
// picks small/large by threshold.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::scatter() const {
    return sycl_scaleup_tune_attr{ scaleup_protocol::default_kernel, scaleup_algo::unspecified };
}

// -----------------------------------------------------------------------------
// reduce scaleup — Arc-only; non-Arc returns `default_kernel` which the
// dispatcher treats as "unsupported → done=false, caller falls back".
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::reduce() const {
    if (query.is_arc()) {
        return sycl_scaleup_tune_attr{ scaleup_protocol::ll, scaleup_algo::ring };
    }
    return sycl_scaleup_tune_attr{ scaleup_protocol::default_kernel, scaleup_algo::unspecified };
}

}  // namespace ccl::sycl_policy_resolver
