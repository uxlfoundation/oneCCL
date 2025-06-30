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

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "oneapi/ccl/types.hpp"
#include "coll/coll.hpp"
#include "common/framework/framework.hpp"
#include "common/log/log.hpp"
#include "common/utils/utils.hpp"
#include "common/utils/yield.hpp"
#include "comp/bf16/bf16_utils.hpp"
#include "comp/fp16/fp16_utils.hpp"
#include "sched/cache/cache.hpp"
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/global/ze/ze_fd_manager.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
#include "topology/topo_manager.hpp"

#include "common/env/vars.hpp"
#include "common/env/vars_experimental.hpp"

enum ccl_priority_mode { ccl_priority_none,
                         ccl_priority_direct,
                         ccl_priority_lifo };

enum ccl_atl_transport { ccl_atl_ofi,
                         ccl_atl_mpi };

enum ccl_atl_send_proxy {
    ccl_atl_send_proxy_none,
    ccl_atl_send_proxy_regular,
    ccl_atl_send_proxy_usm
};

enum ccl_staging_buffer { ccl_staging_regular,
                          ccl_staging_usm };

enum class backend_mode {
    native,
#ifdef CCL_ENABLE_STUB_BACKEND
    stub
#endif // CCL_ENABLE_STUB_BACKEND
};

enum class process_launcher_mode {
    hydra,
    torchrun,
#ifdef CCL_ENABLE_PMIX
    pmix,
#endif // CCL_ENABLE_PMIX
    none
};

enum ccl_selection_border_type {
    ccl_selection_border_left,
    ccl_selection_border_right,
    ccl_selection_border_both
};

template <typename algo_group_type>
using ccl_selection_table_t =
    std::map<size_t, std::pair<algo_group_type, ccl_selection_border_type>>;

enum class ccl_sycl_alltoall_protocol {
    read,
    write
};

namespace ccl {

class env_data {
public:
    env_data();
    ~env_data() = default;

    env_data(const env_data&) = delete;
    env_data(env_data&&) = delete;

    env_data& operator=(const env_data&) = delete;
    env_data& operator=(env_data&&) = delete;

    void parse();
    void print(int rank, bool is_profile_mode, bool is_mt_enabled = false);
    void set_internal_env();

    bool was_printed;
#if defined(CCL_ENABLE_PROFILING)
    bool was_profiled;
#endif
    ccl_spinlock print_guard{};

    ccl_log_level log_level;
    bool abort_on_throw;
    bool queue_dump;
    bool sched_dump;
    bool sched_profile;
    ssize_t entry_max_update_time_sec;

    ccl_framework_type fw_type;

    size_t worker_count;
    bool worker_offload;
    bool worker_wait;
    bool worker_affinity_set;
    std::vector<ssize_t> worker_affinity;
    std::vector<ssize_t> worker_mem_affinity;

    ccl_atl_transport atl_transport;
    kvs_mode kvs_init_mode;
    int kvs_connection_timeout;
    bool kvs_mpi_allgather;
    bool kvs_use_mpi_ranks;
    bool enable_shm;
    bool enable_rma;
    bool enable_hmem;
    ccl_atl_send_proxy atl_send_proxy;
    bool enable_atl_cache;
    bool enable_sync_coll;
    bool enable_extra_ep;
    bool enable_auto_cache;
#if defined(CCL_ENABLE_MPI) && defined(CCL_ENABLE_OMP)
    bool enable_omp_coll;
    bool enable_omp_allreduce;
    bool enable_omp_allgatherv;
    std::string omp_allreduce_num_threads;
#endif // CCL_ENABLE_MPI && CCL_ENABLE_OMP

    atl_mnic_t mnic_type;
    std::string mnic_name_raw;
    ssize_t mnic_count;
    atl_mnic_offset_t mnic_offset;

    /*
       parsing logic can be quite complex
       so hide it inside algorithm_selector module
       and store only raw strings in env_data
    */
    std::shared_ptr<ccl_selection_table_t<ccl_coll_recv_algo>> fallback_recv, store_fallback_recv;
    std::shared_ptr<ccl_selection_table_t<ccl_coll_send_algo>> fallback_send, store_fallback_send;
    bool enable_algo_fallback;
    // main algorithm selection
    std::string allgather_algo_raw;
    std::string allgatherv_algo_raw;
    std::string allreduce_algo_raw;
    std::string alltoall_algo_raw;
    std::string alltoallv_algo_raw;
    std::string barrier_algo_raw;
    std::string bcast_algo_raw;
    std::string broadcast_algo_raw;
    std::string recv_algo_raw;
    std::string reduce_algo_raw;
    std::string reduce_scatter_algo_raw;
    std::string send_algo_raw;
    // scale-out selection part
    std::string allgather_scaleout_algo_raw;
    std::string allgatherv_scaleout_algo_raw;
    std::string allreduce_scaleout_algo_raw;
    std::string alltoall_scaleout_algo_raw;
    std::string alltoallv_scaleout_algo_raw;
    std::string barrier_scaleout_algo_raw;
    std::string bcast_scaleout_algo_raw;
    std::string broadcast_scaleout_algo_raw;
    std::string recv_scaleout_algo_raw;
    std::string reduce_scaleout_algo_raw;
    std::string reduce_scatter_scaleout_algo_raw;
    std::string send_scaleout_algo_raw;
    bool enable_unordered_coll;

    bool enable_fusion;
    int fusion_bytes_threshold;
    int fusion_count_threshold;
    bool fusion_check_urgent;
    float fusion_cycle_ms;

    ccl_priority_mode priority_mode;
    size_t spin_count;
    ccl_yield_type yield_type;
    size_t max_short_size;
    ssize_t bcast_part_count;
    ccl_cache_key_type cache_key_type;
    bool enable_cache_flush;
    bool enable_buffer_cache;
    bool enable_strict_order;
    ccl_staging_buffer staging_buffer;
    bool enable_op_sync;
    bool enable_hostname_sharing;
    bool enable_init_hostname_sharing;

    size_t chunk_count;
    size_t min_chunk_size;
    size_t ze_tmp_buf_size;
    size_t rs_chunk_count;
    size_t rs_min_chunk_size;

#ifdef CCL_ENABLE_SYCL
    bool allgatherv_topo_large_scale;
    bool allgatherv_topo_read;
    bool alltoallv_topo_read;
    bool reduce_scatter_topo_read;
    bool reduce_scatter_monolithic_kernel;
    bool reduce_scatter_monolithic_pipeline_kernel;
    bool reduce_scatter_fallback_algo;
    bool allgatherv_monolithic_kernel;
    bool allgatherv_monolithic_pipeline_kernel;
    bool alltoallv_monolithic_kernel;
    bool alltoallv_monolithic_read_kernel;

    size_t allgatherv_pipe_chunk_count;
    size_t allreduce_pipe_chunk_count;
    size_t reduce_scatter_pipe_chunk_count;
    size_t reduce_pipe_chunk_count;

    bool sycl_allreduce_tmp_buf;
    size_t sycl_allreduce_small_threshold;
    size_t sycl_allreduce_medium_threshold;
    size_t sycl_allreduce_scaleout_threshold;
    std::string sycl_allreduce_scaleout_algo;
    bool sycl_enable_arc_allreduce;
    size_t sycl_allreduce_ll_threshold;

    bool sycl_reduce_scatter_tmp_buf;
    size_t sycl_reduce_scatter_small_threshold;
    size_t sycl_reduce_scatter_medium_threshold;
    size_t sycl_reduce_scatter_scaleout_threshold;
    std::string sycl_reduce_scatter_scaleout_algo;

    bool sycl_allgatherv_tmp_buf;
    size_t sycl_allgatherv_small_threshold;
    size_t sycl_allgatherv_medium_threshold;
    size_t sycl_allgatherv_scaleout_threshold;
    std::string sycl_allgatherv_scaleout_algo;
    size_t sycl_allgatherv_ll_threshold;

    bool sycl_broadcast_tmp_buf;
    size_t sycl_broadcast_small_threshold;
    size_t sycl_broadcast_scaleout_threshold;

    bool enable_sycl_kernels;

    bool sycl_ccl_barrier;
    bool sycl_kernel_sync;
    bool sycl_single_node_algorithm;
    bool sycl_auto_use_tmp_buf;
    bool sycl_force_use_tmp_buf_scaleout;
    bool sycl_copy_engine;
    bool sycl_kernel_copy;
    bool sycl_esimd;
    bool sycl_full_vector;
    bool sycl_force_recording_path;
    size_t sycl_tmp_buf_size;
    size_t sycl_scaleout_host_buf_size;
    size_t sycl_scaleout_device_buf_size;
    size_t sycl_kernels_line_size;
    size_t sycl_max_pipeline_chunk_size;
    ssize_t sycl_pipeline_chunk_size;
    bool sycl_enable_pipeline_gpu_rdma;
    bool sycl_enable_direct_gpu_rdma;
    bool sycl_sub_communicator;
    bool sycl_force_pcie;
    ccl::utils::alloc_mode sycl_scaleout_buf_alloc_mode;
    bool sycl_pt2pt_read;
    bool sycl_pt2pt_enable;
#endif // CCL_ENABLE_SYCL

    bool allreduce_nreduce_buffering;
    ssize_t allreduce_nreduce_segment_size;

    size_t allreduce_2d_chunk_count;
    size_t allreduce_2d_min_chunk_size;
    bool allreduce_2d_switch_dims;

    ssize_t dtree_partition_count;

    bool check_inplace_aliasing;

    ssize_t alltoall_scatter_max_ops;

    backend_mode backend;

    int local_rank;
    int local_size;
    process_launcher_mode process_launcher;

    bool enable_topo_algo;
    topo_color_mode topo_color;
    int enable_p2p_access;
    bool enable_fabric_vertex_connection_check;
    bool enable_wa_fabric_vertex_connection_check;
    bool use_mpi_bcast_wa;
    bool use_root_print_wa;

#ifdef CCL_ENABLE_MPI
    std::string mpi_lib_path;
    bool mpi_bf16_native;
    bool mpi_fp16_native;
#endif // CCL_ENABLE_MPI
    std::string ofi_lib_path;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_UMF)
    int umf_enable;
    std::string umf_lib_path;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE && CCL_ENABLE_UMF

#ifdef CCL_ENABLE_SYCL
    std::string kernel_path;
    bool kernel_debug;
    bool kernel_module_cache;
    ssize_t kernel_group_size;
    ssize_t kernel_group_count;
    ssize_t kernel_mem_align;
    bool enable_kernel_sync;
    bool kernel_1s_lead;
    bool enable_kernel_1s_copy_ops;
    bool enable_kernel_1s_ipc_wa;
    bool enable_close_fd_wa;

    bool enable_sycl_output_event;
    bool use_hmem;

    bool sync_barrier;
    bool sync_deps;

    bool enable_ze_barrier;
    bool enable_ze_bidir_algo;
    bool enable_ze_cache;
    bool ze_device_cache_evict_smallest;
    long ze_device_cache_upper_limit;
    int ze_device_cache_num_blocks_in_chunk;
    ccl::ze::device_cache_policy_mode ze_device_cache_policy;
    size_t ze_pointer_registration_threshold;
    bool enable_ze_cache_cmdlists;
    bool enable_ze_cache_cmdqueues;
    bool enable_ze_cache_event_pools;
    bool enable_ze_cache_open_ipc_handles;
    int ze_cache_open_ipc_handles_threshold;
    int ze_cache_get_ipc_handles_threshold;
    bool enable_ze_cache_get_ipc_handles;
    bool enable_ze_single_list;
    bool disable_ze_family_check;
    bool disable_ze_port_check;
    bool ze_enable_oversubscription_fallback;
    bool ze_enable_oversubscription_throw;
    bool ze_serialize_mode;
    ccl::ze::copy_engine_mode ze_copy_engine;
    ccl::ze::h2d_copy_engine_mode ze_h2d_copy_engine;
    ccl::ze::d2d_copy_engine_mode ze_d2d_copy_engine;
    ssize_t ze_max_compute_queues;
    ssize_t ze_max_copy_queues;
    bool ze_enable_ccs_fallback_for_copy;
    bool enable_ze_list_dump;
    int ze_queue_index_offset;
    bool ze_close_ipc_wa;
    std::string ze_lib_path;
    bool ze_enable;
    bool ze_fini_wa;
    bool ze_multi_workers;
    ccl_sycl_alltoall_protocol sycl_alltoall_protocol;
    bool enable_ze_auto_tune_ports;
    ccl::ze::ipc_exchange_mode ze_ipc_exchange;
    bool ze_drm_bdf_support;
    bool ze_pt2pt_read;
    type2_tune_mode type2_mode;
    std::string drmfd_dev_render_dir_path;
    std::string drmfd_dev_render_suffix;
#endif // CCL_ENABLE_SYCL
    bool ipc_allgatherv_wa;

#ifdef CCL_ENABLE_PMIX
    std::string pmix_lib_path;
#endif // CCL_ENABLE_PMIX

#ifdef CCL_ENABLE_ITT
    int itt_level;
#endif // CCL_ENABLE_ITT
    int debug_timestamps_level;

    bool enable_profiling;

    ccl_bf16_impl_type bf16_impl_type;
    ccl_fp16_impl_type fp16_impl_type;

    template <class T>
    static std::string str_by_enum(const std::map<T, std::string>& values, const T& val) {
        typename std::map<T, std::string>::const_iterator it;

        it = values.find(val);
        if (it != values.end()) {
            return it->second;
        }
        else {
            CCL_THROW("unexpected val ", static_cast<int>(val));
            return NULL;
        }
    }

    static std::map<ccl_priority_mode, std::string> priority_mode_names;
    static std::map<ccl_atl_transport, std::string> atl_transport_names;
    static std::map<ccl_atl_send_proxy, std::string> atl_send_proxy_names;
    static std::map<ccl_staging_buffer, std::string> staging_buffer_names;
    static std::map<backend_mode, std::string> backend_names;
    static std::map<process_launcher_mode, std::string> process_launcher_names;

    int env_2_worker_affinity(int local_proc_idx, int local_proc_count);
    int env_2_worker_mem_affinity(int local_proc_count);

private:
    int env_2_worker_affinity_auto(int local_proc_idx, size_t workers_per_process);
    static int parse_affinity(const std::string& input,
                              std::vector<ssize_t>& output,
                              size_t expected_output_size);
    static int parse_number(const std::string& number_str, size_t& result);
};

} // namespace ccl
