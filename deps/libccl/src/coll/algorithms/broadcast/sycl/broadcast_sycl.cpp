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

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/broadcast/sycl/broadcast_sycl.hpp"
#include "coll/algorithms/broadcast/sycl/broadcast_scaleout_scatter_allgather.hpp"
#include "coll/algorithms/utils/sycl_policy_resolver_scaleup.hpp"
#include "coll/algorithms/utils/barrier_data.hpp"  // use_recording_path
#include "common/log/log.hpp"

namespace {

// world == 1 fast path.
ccl::event broadcast_scaleup_single_rank(const void* send_buf,
                                         void* recv_buf,
                                         size_t count,
                                         ccl::datatype dtype,
                                         ccl_stream* global_stream,
                                         const ccl::vector_class<ccl::event>& deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    sycl::event sycl_e;
    auto sycl_q = global_stream->get_native_stream();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    if (send_buf != recv_buf) {
        LOG_DEBUG("single rank: out-of-place case, coll: bcast");
        sycl_e = sycl_q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
        });
    }
    else {
        LOG_DEBUG("single rank: inplace case, coll: bcast");
        sycl_e = submit_wait_on_events(sycl_q, dep_events);
    }
    return ccl::event::create_from_native(sycl_e);
}

// Arc RT256 ring. Caller must have already checked `should_attempt_rt_ring`.
// On unaligned buffers sets done=false and returns empty — a hard bail; the
// caller must NOT fall through to the vec path (the whole scaleup returns as
// not-done). On done=true, the kernel ran. On done=false with aligned buffers,
// the kernel soft-bailed and the caller may fall through to vec.
ccl::event broadcast_scaleup_rt_ring(const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     int root,
                                     ccl_comm* comm,
                                     std::shared_ptr<ccl_comm> p2p_comm,
                                     ccl_stream* global_stream,
                                     bool& done,
                                     bool& hard_bail) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    if (!is_aligned(send_buf, recv_buf, count, ccl_dtype.size(), 4)) {
        done = false;
        hard_bail = true;
        return {};
    }
    LOG_DEBUG("invoking broadcast RT256 kernel broadcast_rt_ring, count:", count, " datatype: ", dtype);
    ccl::event e = broadcast_rt_ring(
        send_buf, recv_buf, count, dtype, root, comm, p2p_comm, global_stream, done);
    LOG_DEBUG("invoking broadcast RT256 kernel, recv_count:", count, " datatype: ", dtype, " done");
    return e;
}

// Non-RT path: small kernel for below-threshold sizes, large kernel otherwise.
ccl::event broadcast_scaleup_default(sycl::queue& /*q*/,
                                 const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 ccl::datatype dtype,
                                 int root,
                                 ccl_comm* comm,
                                 std::shared_ptr<ccl_comm> p2p_comm,
                                 ccl_stream* global_stream,
                                 const ccl::vector_class<ccl::event>& deps,
                                 bool is_large) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t total_bytes = count * ccl_dtype.size();
    ccl::event e;
    if (!is_large) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "broadcast_small", "send_size", total_bytes, comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| broadcast selects small kernel, count: ", count, " datatype: ", dtype);
        e = broadcast_small(
            send_buf, recv_buf, count, dtype, root, p2p_comm.get(), global_stream, deps);
        LOG_DEBUG("|CCL_SYCL| broadcast selects small kernel, count: ",
                  count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "broadcast_large", "send_size", total_bytes, comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| broadcast selects large kernel, count: ", count, " datatype: ", dtype);
        e = broadcast_large(
            send_buf, recv_buf, count, dtype, root, comm, p2p_comm, global_stream, deps);
        LOG_DEBUG("|CCL_SYCL| broadcast selects large kernel, count: ",
                  count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    return e;
}

}  // namespace

namespace ccl {
namespace v1 {

ccl::event broadcast_scaleup_sycl(sycl::queue& q,
                                  const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  int root,
                                  ccl_comm* comm,
                                  std::shared_ptr<ccl_comm> p2p_comm,
                                  ccl_stream* global_stream,
                                  const vector_class<event>& deps,
                                  bool& done) {
    done = true;

    if (p2p_comm->size() == 1) {
        return broadcast_scaleup_single_rank(send_buf, recv_buf, count, dtype, global_stream, deps);
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t total_bytes = count * ccl_dtype.size();
    const bool is_large = total_bytes > ccl::global_data::env().sycl_broadcast_small_threshold;

    const auto tune = ccl::sycl_policy_resolver::scaleup_resolver{
        ccl::sycl_policy::policy_query(
            ccl_coll_broadcast, total_bytes, ccl_dtype,
            /*r2r_size=*/0, comm->size(), use_recording_path(q))
            .with_comm(comm)
            .with_p2p_comm(p2p_comm.get())
            .with_stream(global_stream)
    }.broadcast();

    // Arc RT256 first: may hard-bail (unaligned), succeed (done=true), or
    // soft-bail (done=false → fall through to default).
    if (tune.protocol == ccl::sycl_policy::scaleup_protocol::ll) {
        bool hard_bail = false;
        auto ev = broadcast_scaleup_rt_ring(send_buf, recv_buf, count, dtype, root,
                                            comm, p2p_comm, global_stream, done, hard_bail);
        if (hard_bail) return ev;      // caller falls back — do NOT try default
        if (done) return ev;
        // rt_ring soft-bailed; continue to default below.
    }

    return broadcast_scaleup_default(q, send_buf, recv_buf, count, dtype, root,
                                 comm, p2p_comm, global_stream, deps, is_large);
}

static bool do_fallback_to_scheduler(size_t size) {
    bool is_above_threshold = size > ccl::global_data::env().sycl_broadcast_scaleout_threshold;
    bool exception_cases = false;

    return is_above_threshold || exception_cases;
}

ccl::event broadcast_sycl_multi_domain(sycl::queue& q,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t count,
                                       ccl::datatype dtype,
                                       int root,
                                       ccl_comm* global_comm,
                                       std::shared_ptr<ccl_comm> p2p_comm,
                                       std::shared_ptr<ccl_comm> r2r_comm,
                                       ccl_stream* global_stream,
                                       const vector_class<event>& deps,
                                       bool& done) {
    done = true;
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    if (do_fallback_to_scheduler(count * ccl_dtype.size())) {
        LOG_DEBUG("broadcast count size = ",
                  count * ccl_dtype.size(),
                  " is above scaleout SYCL threshold = ",
                  ccl::global_data::env().sycl_broadcast_scaleout_threshold,
                  " or sycl::esimd mode is enabled, or other conditions not met -- falling back");
        done = false;
        return ev;
    }

    // Master switch: only ENABLE=1 considers Scatter-AllGather (SAG).
    // FORCE=1 is a truth-mode benchmark knob: fail loudly instead of hiding fallback.
    if (ccl::global_data::env().sycl_broadcast_scatter_allgather_enable == 1) {
        const bool force_sag = ccl::global_data::env().sycl_broadcast_scatter_allgather_force;
        const bool sag_preconditions =
            is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device())) &&
            p2p_comm->size() >= 2 && r2r_comm->size() >= 2;
        if (!sag_preconditions) {
            if (force_sag) {
                CCL_THROW(
                    "forced broadcast scatter-allgather cannot run: requires ARC GPU and both p2p/r2r communicator sizes >= 2");
            }
        }
        else {
            bool sr_done = false;
            ccl::event sr_ev = broadcast_scaleout_scatter_allgather(q,
                                                                    send_buf,
                                                                    recv_buf,
                                                                    count,
                                                                    dtype,
                                                                    root,
                                                                    global_comm,
                                                                    p2p_comm,
                                                                    r2r_comm,
                                                                    global_stream,
                                                                    deps,
                                                                    sr_done);
            if (sr_done) {
                done = true;
                return sr_ev;
            }
            if (force_sag) {
                CCL_THROW("forced broadcast scatter-allgather returned not-done");
            }
            // else: fall through to the default chunked broadcast below
        }
    }

    LOG_DEBUG("broadcast_sycl multi-node count: ", count);

    // initial dependencies for the loop
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    std::vector<ccl::event> evs;
    for (int i = 0; i < dep_events.size(); i++) {
        ev = ccl::event::create_from_native(dep_events[i]);
        evs.push_back(std::move(ev));
    }

    // copy to recv_buf or not
    const bool inplace = send_buf == recv_buf;
    // Project the global root into the r2r and p2p communicator rank spaces.
    const int root_rank_in_r2r_comm = root / p2p_comm->size();
    const int root_rank_in_p2p_comm = root % p2p_comm->size();

    // chunking buffer size computation
    // host buffer is used in direct algorithm
    const int buf_size = global_comm->get_scaleout_host_buf_size();
    size_t max_iter_count;
    if (count * ccl_dtype.size() <= buf_size) {
        max_iter_count = count;
    }
    else {
        max_iter_count = buf_size / ccl_dtype.size();
    }

    size_t displ = 0, displ_count = 0;
    int nchunks = (count + max_iter_count - 1) / max_iter_count;
    for (int i = 0; i < nchunks; i++) {
        size_t iter_count = i < nchunks - 1 ? max_iter_count : count - displ_count;

        auto send_buf_ptr = ptr_offset(send_buf, displ);
        auto recv_buf_ptr = ptr_offset(recv_buf, displ);

        LOG_DEBUG("broadcast_sycl count: ",
                  count,
                  " root: ",
                  root,
                  " and #chunk: ",
                  i,
                  " of nchunks: ",
                  nchunks,
                  " - broadcast_scaleout_sycl");

        // Scale-out runs only on the root rank from each p2p domain.
        if (p2p_comm->rank() == root_rank_in_p2p_comm) {
            bool original_deps = (i == 0 && !evs.empty());
            ev = broadcast_scaleout_sycl(q,
                                         send_buf_ptr,
                                         recv_buf_ptr,
                                         iter_count,
                                         dtype,
                                         root_rank_in_r2r_comm,
                                         r2r_comm.get(),
                                         evs,
                                         original_deps,
                                         done);

            if (!done) {
                LOG_INFO("broadcast_sycl scale-out phase was not done -- falling back");
                // fallback
                return ev;
            }
        }

        // scale-up phase, broadcast from new root
        if (p2p_comm->size() > 1) {
            // create a dependency with scale-out phase only for the ranks
            // that are actually participated in scale-out phase
            if (p2p_comm->rank() == root_rank_in_p2p_comm) {
                evs.clear();
                evs.push_back(std::move(ev));
            }
            // scale-up phase will be inplace
            send_buf_ptr = recv_buf_ptr;

            ev = broadcast_scaleup_sycl(q,
                                        send_buf_ptr,
                                        recv_buf_ptr,
                                        iter_count,
                                        dtype,
                                        root_rank_in_p2p_comm,
                                        global_comm,
                                        p2p_comm,
                                        global_stream,
                                        evs,
                                        done);

            if (!done) {
                LOG_INFO("broadcast_sycl scale-up phase was not done -- falling back");
                // fallback
                return ev;
            }
        }

        if (i < nchunks - 1) {
            evs.clear();
            evs.push_back(std::move(ev));
            displ_count += iter_count;
            displ += iter_count * ccl_dtype.size();
        }
    }
    return ev;
}

ccl::event broadcast_sycl(sycl::queue q,
                          const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          ccl::datatype dtype,
                          int root,
                          ccl_comm* global_comm,
                          ccl_stream* global_stream,
                          const broadcast_attr& attr,
                          const vector_class<ccl::event>& deps,
                          bool& done) {
    if (count == 0) {
        done = true;
        auto sycl_deps = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_deps);
        return ccl::event::create_from_native(e);
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    auto p2p_level = ccl_p2p_domain::sycl_get_broadcast_p2p_level(
        global_comm, count * ccl_dtype.size(), ccl::global_data::env().sycl_p2p_level_broadcast_threshold);

    auto [p2p_comm, r2r_comm] = ccl_p2p_domain::get_p2p_comms(global_comm, p2p_level);

    // single p2p domain
    if (r2r_comm->is_uniform && r2r_comm->size() == 1) {
        LOG_DEBUG("is_single_domain");
        return broadcast_scaleup_sycl(q,
                                      send_buf,
                                      recv_buf,
                                      count,
                                      dtype,
                                      root,
                                      global_comm,
                                      p2p_comm,
                                      global_stream,
                                      deps,
                                      done);
    }
    return broadcast_sycl_multi_domain(q,
                                       send_buf,
                                       recv_buf,
                                       count,
                                       dtype,
                                       root,
                                       global_comm,
                                       p2p_comm,
                                       r2r_comm,
                                       global_stream,
                                       deps,
                                       done);
}

} // namespace v1
} // namespace ccl
