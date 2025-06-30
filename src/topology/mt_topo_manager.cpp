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
#include "common/global/global.hpp"
#include "exec/exec.hpp"
#include "topology/topo_manager.hpp"

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/utils/sycl_utils.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

namespace ccl {

void topo_manager::init(int size,
                        int rank,
                        int global_current_id,
                        std::shared_ptr<ccl::device> device,
                        std::shared_ptr<ccl::context> context) {
    CCL_THROW_IF_NOT(ccl::global_data::env().topo_color == ccl::topo_color_mode::fixed,
                     "fixed color is only supoorted yet");
    base_init(size, rank, global_current_id, device, context);
    if (device) {
        // TODO: move intra/inter card logic under ZE define
        post_init(size, rank, global_current_id);
    }
}

rank_info_vec_t topo_manager::get_filtered_rank_info_vec(int filter_host_idx,
                                                         rank_info_vec_t in_rank_info_vec) const {
    CCL_THROW_IF_NOT(!in_rank_info_vec.empty());
    rank_info_vec_t info_vec;
    std::copy_if(in_rank_info_vec.begin(),
                 in_rank_info_vec.end(),
                 std::back_inserter(info_vec),
                 [filter_host_idx](const topo_rank_info& info) {
                     return (info.host_idx == filter_host_idx);
                 });
    return info_vec;
}

void topo_manager::allgather(int size,
                             int rank,
                             const void* send_data,
                             void* recv_data,
                             int data_len,
                             int global_current_id) {
    static std::mutex local_data_mutex;
    {
        std::lock_guard<std::mutex> lock(local_data_mutex);
        std::memcpy(static_cast<char*>(recv_data) + rank * data_len, send_data, data_len);
    }

    // Wait for all threads to store their data
    pthread_barrier_wait(&ccl::global_data::get().shared_data->barrier_waits[global_current_id]);
}

void topo_manager::build_host_info(int size, int rank, int global_current_id) {
    CCL_THROW_IF_NOT(host_info_vec.empty());

    int comm_rank = rank;
    int comm_size = size;

    ccl::global_data::get().shared_data->resize_all_hostnames_raw(max_hostname_len * comm_size,
                                                                  global_current_id);
    char my_hostname[max_hostname_len] = { 0 };

    gethostname(my_hostname, max_hostname_len - 1);
    LOG_DEBUG("rank: ",
              comm_rank,
              ", size: ",
              comm_size,
              ", host: ",
              my_hostname,
              ", current_global_id: ",
              global_current_id);

    allgather(
        size,
        rank,
        my_hostname,
        ccl::global_data::get().shared_data->get_all_hostnames_raw_glob(global_current_id).data(),
        max_hostname_len,
        global_current_id);

    std::vector<std::string> all_hostnames(comm_size);
    std::set<std::string> unique_hostnames;
    for (int idx = 0; idx < comm_size; idx++) {
        auto str = std::string(ccl::global_data::get()
                                       .shared_data->get_all_hostnames_raw_glob(global_current_id)
                                       .data() +
                                   (idx * max_hostname_len),
                               max_hostname_len);
        str.erase(std::find(str.begin(), str.end(), '\0'), str.end());
        unique_hostnames.insert(str);
        all_hostnames[idx] = std::move(str);
    }
    CCL_THROW_IF_NOT(!unique_hostnames.empty(), "empty unique_hostnames");

    CCL_THROW_IF_NOT(unique_hostnames.find(my_hostname) != unique_hostnames.end(),
                     "unique_hostnames does not include my_hostname ",
                     my_hostname);
    host_idx = std::distance(unique_hostnames.begin(), unique_hostnames.find(my_hostname));
    CCL_THROW_IF_NOT((host_idx != topo_manager::invalid_host_idx) && (host_idx >= 0),
                     "invalid host index, host: ",
                     my_hostname);

    for (const auto& hostname : unique_hostnames) {
        size_t unique_host_idx =
            std::distance(unique_hostnames.begin(), unique_hostnames.find(hostname));
        host_info_vec.push_back({ (int)unique_host_idx, hostname });
        CCL_THROW_IF_NOT(unique_host_idx == (host_info_vec.size() - 1));
    }

    for (int rank = 0; rank < comm_size; rank++) {
        const auto& rank_hostname = all_hostnames[rank];
        int rank_host_idx =
            std::distance(unique_hostnames.begin(), unique_hostnames.find(rank_hostname));
        host_info_vec[rank_host_idx].ranks.insert(rank);
    }

    for (int h_idx = 0; h_idx < (int)host_info_vec.size(); h_idx++) {
        const auto& info = host_info_vec[h_idx];
        CCL_THROW_IF_NOT(!info.ranks.empty() && (int)info.ranks.size() <= comm_size,
                         "host_idx: ",
                         info.idx,
                         ", unexpected number of ranks: ",
                         info.ranks.size());
        CCL_THROW_IF_NOT(
            info.idx == h_idx, "unexpected host_idx: ", info.idx, ", expected: ", h_idx);
    }

    is_single_node = (host_info_vec.size() == 1) ? true : false;
    is_single_card = is_single_node && (comm_size <= max_ranks_per_card);
    is_same_ppn =
        std::all_of(host_info_vec.begin(), host_info_vec.end(), [this](const topo_host_info& info) {
            return (info.ranks.size() == host_info_vec.front().ranks.size());
        });
}

void topo_manager::base_init(int size,
                             int rank,
                             int global_current_id,
                             const std::shared_ptr<ccl::device>& device,
                             const std::shared_ptr<ccl::context>& context) {
    int comm_rank = rank;
    int comm_size = size;

    intra_card_colors.resize(comm_size, topo_manager::invalid_color);
    inter_card_colors.resize(comm_size, topo_manager::invalid_color);
    uuids.resize(comm_size);

    ccl::global_data::get().shared_data->resize_rank_info_vec_glob(comm_size, global_current_id);

    build_host_info(size, rank, global_current_id);

    // exchange common rank info
    topo_rank_info rank_info{};
    rank_info.rank = comm_rank;
    rank_info.host_idx = host_idx;
    rank_info.local_proc_idx = rank; // TODO: ccl::global_data::get().get_local_proc_idx();
    std::string rank_uuid = topo_manager::generate_uuid();
    std::copy(rank_uuid.begin(), rank_uuid.end(), rank_info.uuid);

    allgather(size,
              rank,
              &rank_info,
              ccl::global_data::get().shared_data->get_rank_info_vec_glob(global_current_id).data(),
              sizeof(rank_info),
              global_current_id);
    pthread_barrier_wait(&ccl::global_data::get().shared_data->barrier_waits[global_current_id]);
    for (size_t idx = 0;
         idx <
         ccl::global_data::get().shared_data->get_rank_info_vec_glob(global_current_id).size();
         idx++) {
        uuids[idx] = std::string(ccl::global_data::get()
                                     .shared_data->get_rank_info_vec_glob(global_current_id)[idx]
                                     .uuid);
    }

    if (!(device && context)) {
        return;
    }

    if (ccl::global_data::env().topo_color == topo_color_mode::fixed) {
        for (int h_idx = 0; h_idx < (int)host_info_vec.size(); h_idx++) {
            fill_fixed_colors(get_filtered_rank_info_vec(
                h_idx,
                ccl::global_data::get().shared_data->get_rank_info_vec_glob(global_current_id)));
            pthread_barrier_wait(
                &ccl::global_data::get().shared_data->barrier_waits[global_current_id]);
        }
    }
    else {
        CCL_THROW("unknown topo color mode: ", (int)ccl::global_data::env().topo_color);
    }

    // update is_single_card
    bool no_invalid_colors = (std::find(intra_card_colors.begin(),
                                        intra_card_colors.end(),
                                        topo_manager::invalid_color) == intra_card_colors.end())
                                 ? true
                                 : false;
    bool all_same_colors =
        std::all_of(intra_card_colors.begin(), intra_card_colors.end(), [this](int c) {
            return (c == intra_card_colors.front());
        });

    is_single_card = (is_single_node && (comm_size <= topo_manager::max_ranks_per_card) &&
                      no_invalid_colors && all_same_colors);
    // TODO:
    // if (comm_rank == 0) {
    //     LOG_INFO("rank_info_vec: ", ccl::to_string(ccl::global_data::get().shared_data->get_rank_info_vec_glob(), host_info_vec));
    // }
}

void topo_manager::post_init(int size, int rank, int global_current_id) {
    for (int rank = 0; rank < size; rank++) {
        intra_card_colors[rank] += ccl::global_data::get()
                                       .shared_data->get_rank_info_vec_glob(global_current_id)[rank]
                                       .host_idx *
                                   max_ranks_per_host;
        inter_card_colors[rank] += ccl::global_data::get()
                                       .shared_data->get_rank_info_vec_glob(global_current_id)[rank]
                                       .host_idx *
                                   max_ranks_per_host;
    }
    is_same_domains = check_colors();
}

} // namespace ccl
