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
#include "coll/algorithms/scatter/sycl/scatter_sycl.hpp"
#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"
#include "common/log/log.hpp"

namespace ccl {
namespace v1 {

namespace {

// Scatter-AllGather (SAG) is the broadcast variant implemented here:
// local scatter, cross-domain rail broadcast, then local allgather.
// The helpers below may use the shorter SAG name after this point.

// Truth-mode switch for benchmarking the requested algorithm without silent
// fallback. When enabled, unsupported shapes throw instead of returning
// done=false to legacy.
bool sag_force_enabled() {
    return ccl::global_data::env().sycl_broadcast_scatter_allgather_force;
}

// Optional safety guard: cap this algorithm to local node sizes where it is
// known useful. A value of 0 disables the cap; force mode also ignores it to
// expose raw SAG behavior.
int sag_max_ppn() {
    return ccl::global_data::env().sycl_broadcast_scatter_allgather_max_ppn;
}

// Convert normal fallback reasons into hard failures when FORCE mode is enabled.
void sag_throw_if_forced(bool force_sag, const char* reason) {
    if (force_sag) {
        CCL_THROW("forced broadcast scatter-allgather cannot run: ", reason);
    }
}

} // namespace

// Run broadcast as Scatter-AllGather (SAG): local scatter on the root p2p
// domain, rail broadcast across matching local ranks, then local allgather
// inside each p2p domain.
ccl::event broadcast_scaleout_scatter_allgather(sycl::queue& q,
                                                const void* send_buf,
                                                void* recv_buf,
                                                size_t count,
                                                ccl::datatype dtype,
                                                int root,
                                                ccl_comm* global_comm,
                                                const std::shared_ptr<ccl_comm>& p2p_comm,
                                                const std::shared_ptr<ccl_comm>& r2r_comm,
                                                ccl_stream* global_stream,
                                                const ccl::vector_class<ccl::event>& deps,
                                                bool& done) {
    ccl::event ev;
    // Default to not-done; flip to true only on success. So every early
    // return means "SAG did not handle it."
    done = false;

    // Force mode is for measurement, not production selection. It prevents
    // quiet fallback so a scatter_ag benchmark row cannot secretly be legacy.
    const bool force_sag = sag_force_enabled();

    // The algorithm needs a local P2P-domain dimension and a same-rank-across-domains
    // dimension. The P2P domain may be a whole node or a smaller locality domain
    // such as a PCIe switch inside one physical host.
    const int p2p_domain_size = static_cast<int>(p2p_comm->size());
    const int num_domains = static_cast<int>(r2r_comm->size());
    if (p2p_domain_size < 2 || num_domains < 2) {
        sag_throw_if_forced(force_sag, "requires both p2p and r2r communicator sizes >= 2");
        return ev; // caller falls back
    }

    // In normal mode, protect larger ppn values from known bad/flat performance.
    // In force mode, ignore this guard so we can measure the raw algorithm honestly.
    const int max_ppn = sag_max_ppn();
    if (!force_sag && max_ppn > 0 && p2p_domain_size > max_ppn) {
        return ev; // caller falls back
    }

    // allgather_scaleup_sycl currently requires equal recv_counts, so avoid
    // doing scatter/rail work for uneven chunks that would fall back later.
    if ((count % static_cast<size_t>(p2p_domain_size)) != 0) {
        sag_throw_if_forced(force_sag, "message element count must divide evenly by local ppn");
        return ev;
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t elem_size = ccl_dtype.size();
    const size_t total_bytes = count * elem_size;

    // In normal mode, the configured threshold selects between SAG and the
    // lower-overhead legacy path. Force mode bypasses this selection gate.
    if (!force_sag &&
        total_bytes < ccl::global_data::env().sycl_broadcast_scatter_allgather_threshold) {
        return ev; // caller falls back
    }

    // The current rank-to-domain math is intentionally conservative: it supports
    // block-ordered domains such as node-level ppn blocks or equal-sized sub-node
    // locality blocks. If placement is irregular, fall back rather than sending
    // a rail broadcast from the wrong source rank.
    // This verifies the rank layout matches the math.
    if (!p2p_comm->is_block_ordered_with_r2r()) {
        sag_throw_if_forced(force_sag, "requires block-ordered rectangular p2p/r2r domains");
        return ev;
    }

    // Map the global root into the two-dimensional p2p-domain/rank layout.
    // This tells phase 1 who scatters locally and phase 2 which rail owns the source.
    const int root_domain_idx = root / p2p_domain_size;
    const int root_rank_in_domain = root - root_domain_idx * p2p_domain_size;
    const int my_p2p_rank = p2p_comm->rank();
    const int my_r2r_rank = r2r_comm->rank();
    // Tells whether this rank is in the p2p domain that contains the root
    const bool i_am_on_root_domain = (my_r2r_rank == root_domain_idx);

    // Split the message into one local chunk per rank position.
    // The earlier divisibility guard keeps these counts equal for allgather correctness.
    const size_t chunk_max = (count + p2p_domain_size - 1) / p2p_domain_size;
    const size_t my_chunk_count = (my_p2p_rank < p2p_domain_size - 1)
                                      ? chunk_max
                                      : (count - chunk_max * (p2p_domain_size - 1));
    const size_t my_chunk_offset_bytes = static_cast<size_t>(my_p2p_rank) * chunk_max * elem_size;
    void* my_chunk_ptr = ptr_offset(recv_buf, my_chunk_offset_bytes);

    // Convert incoming dependencies into a local event chain we can move between phases.
    // This preserves caller ordering while letting each phase depend on the previous one.
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    ccl::vector_class<ccl::event> chain;
    for (auto& se : dep_events) {
        chain.push_back(ccl::event::create_from_native(se));
    }

    // -----------------------------------------------------------------
    // PHASE 1: scatter on the root's P2P domain. Other domains skip this.
    //   Each rank on the root domain ends up with its 1/p2p_domain_size-th
    //   of send_buf at its pre-assigned offset inside recv_buf.
    // -----------------------------------------------------------------
    // Only ranks in the root's p2p domain run scatter.
    if (i_am_on_root_domain) {
        bool scatter_done = false;
        // SAG explicitly uses pre-exchanged tmp buffers for its local scatter.
        sycl_scaleup_options scatter_attr{/*force_tmp=*/true};
        ccl::event scatter_ev =
            scatter_sycl_single_domain(q,
                                       send_buf, // full source buffer (only meaningful on root)
                                       my_chunk_ptr, // this rank's slot inside recv_buf
                                       chunk_max, // per-rank count (uniform for symmetry)
                                       dtype,
                                       root_rank_in_domain,
                                       global_comm,
                                       p2p_comm,
                                       global_stream,
                                       chain,
                                       scatter_done,
                                       scatter_attr);

        if (!scatter_done) {
            sag_throw_if_forced(force_sag, "local scatter phase did not complete");
            return ev; // scatter didn't handle this case → fall back
        }
        chain.clear();
        // Phase 2 depends on scatter.
        chain.push_back(std::move(scatter_ev));
    }

    // -----------------------------------------------------------------
    // PHASE 2: rail broadcast. Every rank position i does a broadcast
    //   inside its r2r_comm[i] column. All p2p_domain_size rails happen
    //   in parallel (each rank participates in exactly one r2r_comm).
    //   Source of each rail is the rank on the root domain (r2r rank =
    //   root_domain_idx). Destination(s) are the ranks on the other domains.
    // -----------------------------------------------------------------
    {
        bool rail_done = false;
        ccl::event rail_ev = broadcast_scaleout_sycl(
            q,
            my_chunk_ptr, // in-place: send from and recv into same slot
            my_chunk_ptr,
            my_chunk_count, // rank-specific count (last rank may be smaller)
            dtype,
            root_domain_idx, // source rank inside r2r_comm
            r2r_comm.get(),
            chain,
            /*original_deps=*/false,
            rail_done);

        if (!rail_done) {
            sag_throw_if_forced(force_sag, "rail broadcast phase did not complete");
            return ev; // rail didn't complete → fall back
        }
        chain.clear();
        chain.push_back(std::move(rail_ev));
    }

    // -----------------------------------------------------------------
    // PHASE 3: local allgather. Every p2p domain's ranks pool their
    //   chunks into their local recv_buf.
    //   Contract of allgather_scaleup_sycl: each rank contributes
    //   send_count elements from send_buf; the concatenation lands in
    //   recv_buf at the given offsets, using per-rank recv_counts.
    // -----------------------------------------------------------------
    ccl::vector_class<size_t> recv_counts(p2p_domain_size, chunk_max);
    if (count > chunk_max * (p2p_domain_size - 1)) {
        recv_counts[p2p_domain_size - 1] = count - chunk_max * (p2p_domain_size - 1);
    }

    // Offsets describe the final full receive layout: chunk 0, chunk 1, ...
    // All local ranks use the same layout so the broadcast result is identical everywhere.
    ccl::vector_class<size_t> offsets(p2p_domain_size, 0);
    for (int i = 1; i < p2p_domain_size; ++i) {
        offsets[i] = offsets[i - 1] + recv_counts[i - 1] * elem_size;
    }

    // Local allgather rebuilds the complete broadcast buffer per p2p domain.
    // Its completion event represents the full SAG algorithm.
    bool ag_done = false;
    // Use pre-exchanged tmp buffers so the final local allgather does not
    // perform a blocking host-side IPC exchange for the receive buffer.
    sycl_scaleup_options ag_attr{/*force_tmp=*/true};
    ccl::event ag_ev = allgather_scaleup_sycl(q,
                                              my_chunk_ptr, // each rank sends its chunk
                                              my_chunk_count,
                                              recv_buf, // gather into recv_buf at layout above
                                              recv_counts,
                                              offsets,
                                              dtype,
                                              global_comm,
                                              p2p_comm,
                                              global_stream,
                                              chain,
                                              ag_done,
                                              ag_attr);

    if (!ag_done) {
        sag_throw_if_forced(force_sag, "local allgather phase did not complete");
        return ev; // allgather bailed → caller falls back
    }

    done = true;
    return ag_ev;
}

} // namespace v1
} // namespace ccl
