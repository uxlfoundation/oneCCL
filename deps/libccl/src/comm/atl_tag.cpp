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

#include "comm/atl_tag.hpp"
#include "exec/exec.hpp"

std::string ccl_atl_tag::to_string() const {
    std::stringstream ss;
    ss << "{ "
       << "bits: " << tag_bits << ", max: " << max_tag << ", mask: " << max_tag_mask
       << ", pof2: " << ccl::utils::pof2(max_tag) << " }";
    return ss.str();
}

template <typename Layout>
ccl_sched_id_t ccl_atl_tag_impl<Layout>::get_max_sched_count() {
    return Layout::max_sched_count;
}

template <typename Layout>
ccl_sched_id_t ccl_atl_tag_impl<Layout>::get_pt2pt_sched_id() {
    return Layout::pt2pt_sched_id;
}

template <typename Layout>
ccl_sched_id_t ccl_atl_tag_impl<Layout>::get_pt2pt_ack_tag() {
    return Layout::pt2pt_ack_tag;
}

template <typename Layout>
std::tuple<ccl_sched_id_t, ccl_sched_id_t> ccl_atl_tag_impl<Layout>::get_pt2pt_sync_tags() {
    auto ack_first = Layout::pt2pt_ack_first;
    auto ack_second = Layout::pt2pt_ack_second;
    return std::make_tuple(ack_first, ack_second);
}

template <typename Layout>
uint64_t ccl_atl_tag_impl<Layout>::create(int rank,
                                          ccl_comm_id_t comm_id,
                                          ccl_sched_id_t sched_id,
                                          ccl_op_id_t op_id) {
    uint64_t tag = 0;

    tag |= (static_cast<uint64_t>(op_id) << Layout::op_id_shift) & Layout::op_id_mask;
    tag |= (static_cast<uint64_t>(sched_id) << Layout::sched_id_shift) & Layout::sched_id_mask;
    tag |= (static_cast<uint64_t>(comm_id) << Layout::comm_id_shift) & Layout::comm_id_mask;
    tag |= (static_cast<uint64_t>(rank) << Layout::rank_shift) & Layout::rank_mask;

    if (tag > max_tag)
        tag &= max_tag_mask;

    LOG_DEBUG("tag ",
              tag,
              " (rank ",
              rank,
              ", comm_id: ",
              comm_id,
              ", sched_id: ",
              sched_id,
              ", op_id: ",
              (int)op_id,
              ")");

    CCL_THROW_IF_NOT(tag <= max_tag,
                     "unexpected tag value ",
                     tag,
                     ", max_tag ",
                     max_tag,
                     " (rank ",
                     rank,
                     ", comm_id: ",
                     comm_id,
                     ", sched_id: ",
                     sched_id,
                     ", op_id: ",
                     (int)op_id,
                     ")");

    return tag;
}

// Explicit template instantiation to avoid linker errors
template class ccl_atl_tag_impl<common_tag_layout>;
template class ccl_atl_tag_impl<ofi_cxi_tag_layout>;
template class ccl_atl_tag_impl<mpi_tag_layout>;
