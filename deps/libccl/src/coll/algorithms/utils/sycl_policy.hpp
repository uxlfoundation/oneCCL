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

#include <cstddef>
#include <cstdint>

#include "coll/algorithms/algorithm_utils.hpp"    // ccl_coll_type
#include "common/datatype/datatype.hpp"           // ccl_datatype
#include "comm/comm.hpp"                          // ccl_comm topology accessors
#include "common/stream/stream.hpp"               // ccl_stream::get_ze_device
#include "common/global/ze/ze_data.hpp"           // ccl::ze::get_device_family
#include "sched/entry/ze/ze_primitives.hpp"       // is_arc_card

class ccl_comm;
class ccl_stream;

namespace ccl::sycl_policy {

// -----------------------------------------------------------------------------
// Scaleout vocabulary (used by the scaleout auto-tuner and resolver).
// -----------------------------------------------------------------------------

enum class scaleout_sub_kernel : std::uint8_t { direct, chunked };

enum class algorithm : std::uint8_t {
    none,
    ring,
    direct,
    rabenseifner,
    chunked,               // sub-kernel of direct
    gdr_only_pairwise,
    scatter,
    numa_gdr_split,
    fallback,
    proxy_direct,
    proxy_ring,
};

// -----------------------------------------------------------------------------
// Scaleup vocabulary — two orthogonal axes returned by scaleup_resolver.
// -----------------------------------------------------------------------------

// Family of scaleup kernels the caller should dispatch into.
enum class scaleup_protocol : std::uint8_t {
    ll,              // Arc LL256 family
    default_kernel,  // non-Arc SYCL small/large kernels
    esimd,           // ESIMD small/medium/large kernels
};


enum class scaleup_algo : std::uint8_t {
    unspecified,     // resolver doesn't select an algo — dispatcher decides
    // ll protocol sub-choices:
    ring,
    oneshot,
    twoshots,
    direct,          // single-shot, no chunk loop (arc_alltoall / arc_allreduce)
    // default / esimd protocol sub-choices:
    small,
    medium,
    large,
};


class sycl_scaleup_tune_attr {
public:
    sycl_scaleup_tune_attr(scaleup_protocol p, scaleup_algo a) noexcept
        : protocol(p), algo(a) {}

    scaleup_protocol protocol = scaleup_protocol::default_kernel;
    scaleup_algo     algo     = scaleup_algo::unspecified;
};


class policy_query {
public:
    policy_query(ccl_coll_type ct,
                 std::size_t   bytes,
                 ccl_datatype  dt,
                 std::size_t   r2r_size,
                 std::size_t   global_size,
                 bool          rec = false) noexcept
        : ctype(ct),
          msg_bytes(bytes),
          dtype(dt),
          r2r_comm_size(r2r_size == 0 ? global_size : r2r_size),
          global_comm_size(global_size),
          is_recording(rec) {}

    // for the query (e.g. scaleup consultations that need topology info).
    policy_query& with_comm(ccl_comm* c) noexcept       { comm = c;          return *this; }
    policy_query& with_p2p_comm(ccl_comm* p) noexcept   { p2p_comm = p;      return *this; }
    policy_query& with_stream(ccl_stream* s) noexcept   { global_stream = s; return *this; }

    bool is_arc() const {
        if (!global_stream) return false;
        return ccl::ze::is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()));
    }

    bool is_multi_thread() const {
        return comm && comm->is_multi_thread_instance();
    }

    // - allgatherv / alltoall / alltoallv → outer `comm`
    // - reduce_scatter / allreduce        → `p2p_comm`
    bool has_p2p_access() const {
        ccl_comm* c = p2p_comm ? p2p_comm : comm;
        return c && c->get_topo_manager().has_p2p_access();
    }

    ccl_coll_type ctype;
    std::size_t   msg_bytes;
    ccl_datatype  dtype;
    std::size_t   r2r_comm_size;
    std::size_t   global_comm_size;
    bool          is_recording;
    ccl_comm*     comm          = nullptr;
    ccl_comm*     p2p_comm      = nullptr;
    ccl_stream*   global_stream = nullptr;
};

}  // namespace ccl::sycl_policy
