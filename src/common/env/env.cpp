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
#include <libgen.h>
#include <link.h>
#include <memory>
#include <sstream>
#include <unistd.h>

#include "coll/selection/selection.hpp"
#include "common/env/env.hpp"
#include "common/env/env_parser.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "exec/exec.hpp"
#include "oneapi/ccl/environment.hpp"
#include "common/utils/version.hpp"

namespace ccl {

static std::map<ccl_sycl_alltoall_protocol, std::string> sycl_alltoall_protocols = {
    std::make_pair(ccl_sycl_alltoall_protocol::read, "read"),
    std::make_pair(ccl_sycl_alltoall_protocol::write, "write"),
};

std::map<ccl_priority_mode, std::string> env_data::priority_mode_names = {
    std::make_pair(ccl_priority_none, "none"),
    std::make_pair(ccl_priority_direct, "direct"),
    std::make_pair(ccl_priority_lifo, "lifo")
};

std::map<ccl_atl_transport, std::string> env_data::atl_transport_names = {
    std::make_pair(ccl_atl_ofi, "ofi")
#ifdef CCL_ENABLE_MPI
        ,
    std::make_pair(ccl_atl_mpi, "mpi")
#endif // CCL_ENABLE_MPI
};

std::map<ccl_atl_send_proxy, std::string> env_data::atl_send_proxy_names = {
    std::make_pair(ccl_atl_send_proxy_none, "none"),
    std::make_pair(ccl_atl_send_proxy_regular, "regular"),
    std::make_pair(ccl_atl_send_proxy_usm, "usm")
};

std::map<ccl_staging_buffer, std::string> env_data::staging_buffer_names = {
    std::make_pair(ccl_staging_regular, "regular"),
    std::make_pair(ccl_staging_usm, "usm")
};

std::map<backend_mode, std::string> env_data::backend_names = {
    std::make_pair(backend_mode::native, "native"),
#ifdef CCL_ENABLE_STUB_BACKEND
    std::make_pair(backend_mode::stub, "stub")
#endif // CCL_ENABLE_STUB_BACKEND
};

std::map<process_launcher_mode, std::string> env_data::process_launcher_names = {
    std::make_pair(process_launcher_mode::hydra, "hydra"),
    std::make_pair(process_launcher_mode::torchrun, "torchrun"),
#ifdef CCL_ENABLE_PMIX
    std::make_pair(process_launcher_mode::pmix, "pmix"),
#endif // CCL_ENABLE_PMIX
    std::make_pair(process_launcher_mode::none, "none")
};

env_data::env_data()
        : was_printed(false),
#if defined(CCL_ENABLE_PROFILING)
          was_profiled(false),
#endif
          log_level(ccl_log_level::warn),
          abort_on_throw(false),
          queue_dump(false),
          sched_dump(false),
          sched_profile(false),
          entry_max_update_time_sec(CCL_ENV_SIZET_NOT_SPECIFIED),

          fw_type(ccl_framework_none),

          worker_count(1),
          worker_offload(true),
          worker_wait(true),
          worker_affinity_set(0),
#ifdef CCL_ENABLE_MPI
          atl_transport(ccl_atl_mpi),
#else // CCL_ENABLE_MPI
          atl_transport(ccl_atl_ofi),
#endif // CCL_ENABLE_MPI
          kvs_init_mode(kvs_mode::pmi),
          kvs_connection_timeout(120),
          kvs_mpi_allgather(true),
          kvs_use_mpi_ranks(false),
          enable_shm(0),
          enable_rma(0),
          enable_hmem(0),
          atl_send_proxy(ccl_atl_send_proxy_none),
          enable_atl_cache(1),
          enable_sync_coll(0),
          enable_extra_ep(0),
          enable_auto_cache(0),
#if defined(CCL_ENABLE_MPI) && defined(CCL_ENABLE_OMP)
          enable_omp_coll(1),
          enable_omp_allreduce(1),
          enable_omp_allgatherv(1),
          omp_allreduce_num_threads("1:1-511;2:512-1023;4:1024-4095;8:4096-8191;16:8192-32767;32:32768-max"),
#endif // CCL_ENABLE_MPI && CCL_ENABLE_OMP
          mnic_type(ATL_MNIC_NONE),
          mnic_count(CCL_ENV_SIZET_NOT_SPECIFIED),
          mnic_offset(ATL_MNIC_OFFSET_NONE),

          enable_algo_fallback(1),
          enable_unordered_coll(0),

          enable_fusion(0),
          fusion_bytes_threshold(16384),
          fusion_count_threshold(256),
          fusion_check_urgent(1),
          fusion_cycle_ms(0.2),

          priority_mode(ccl_priority_none),
          spin_count(100),
          yield_type(ccl_yield_pause),
          max_short_size(0),
          bcast_part_count(CCL_ENV_SIZET_NOT_SPECIFIED),
          cache_key_type(ccl_cache_key_match_id),
#ifdef CCL_ENABLE_SYCL
          enable_cache_flush(1),
#else // CCL_ENABLE_SYCL
          enable_cache_flush(0),
#endif // CCL_ENABLE_SYCL
          enable_buffer_cache(1),
          enable_strict_order(0),
          staging_buffer(ccl_staging_regular),
          enable_op_sync(0),
          enable_hostname_sharing(0),
          enable_init_hostname_sharing(0),

          chunk_count(1),
          min_chunk_size(65536),
          ze_tmp_buf_size(512 * 1024 * 1024),
          rs_chunk_count(1),
          rs_min_chunk_size(65536),

#ifdef CCL_ENABLE_SYCL
          allgatherv_topo_large_scale(0),
          allgatherv_topo_read(1),
          alltoallv_topo_read(1),
          reduce_scatter_topo_read(1),
          // monolithic kernel for xelink transfer
          reduce_scatter_monolithic_kernel(0),
          // reduce_scatter pipelined monolithic kernel for xelink + mdfi transfer
          reduce_scatter_monolithic_pipeline_kernel(1),
          // reduce_scatter fallback implementation using allreduce
          reduce_scatter_fallback_algo(0),
          allgatherv_monolithic_kernel(0), // monolithic kernel for xelink transfer
          allgatherv_monolithic_pipeline_kernel(
              1), // pipelined monolithic kernel for xelink + mdfi transfer
          alltoallv_monolithic_kernel(1),
          alltoallv_monolithic_read_kernel(1),

          // TODO: not used values yet, add log_info and setting them
          // once they are needed
          allgatherv_pipe_chunk_count(0),
          allreduce_pipe_chunk_count(0),
          reduce_scatter_pipe_chunk_count(0),
          reduce_pipe_chunk_count(0),

          sycl_allreduce_tmp_buf(0),
          sycl_allreduce_small_threshold(524288),
          sycl_allreduce_medium_threshold(16777216),
          sycl_allreduce_scaleout_threshold(4294967296),
          sycl_allreduce_scaleout_algo("auto"),
          sycl_enable_arc_allreduce(0),
          sycl_allreduce_ll_threshold(4096),

          sycl_reduce_scatter_tmp_buf(0),
          sycl_reduce_scatter_small_threshold(2097152),
          sycl_reduce_scatter_medium_threshold(67108864),
          sycl_reduce_scatter_scaleout_threshold(4294967296),
          sycl_reduce_scatter_scaleout_algo("auto"),

          sycl_allgatherv_tmp_buf(0),
          sycl_allgatherv_small_threshold(131072),
          sycl_allgatherv_medium_threshold(2097152),
          sycl_allgatherv_scaleout_threshold(1073741824),
          sycl_allgatherv_scaleout_algo("auto"),
          sycl_allgatherv_ll_threshold(2048),

          sycl_broadcast_tmp_buf(0),
          sycl_broadcast_small_threshold(524288),
          sycl_broadcast_scaleout_threshold(4294967296),

          enable_sycl_kernels(1),

          sycl_ccl_barrier(0),
          sycl_kernel_sync(1),
          sycl_single_node_algorithm(1),
          sycl_auto_use_tmp_buf(1),
          sycl_force_use_tmp_buf_scaleout(1),
          sycl_copy_engine(0),
          sycl_kernel_copy(1),
          sycl_esimd(0),
          sycl_full_vector(1),
          sycl_force_recording_path(0),
          sycl_tmp_buf_size(3 * 128 * 1024 * 1024),
          sycl_scaleout_host_buf_size(1024 * 1024 * 1024),
          sycl_scaleout_device_buf_size(1024 * 1024 * 1024),
          sycl_kernels_line_size(128),
          sycl_scaleout_buf_alloc_mode(ccl::utils::alloc_mode::hwloc),
          sycl_pt2pt_read(0),
          sycl_pt2pt_enable(1),
          sycl_max_pipeline_chunk_size(32 * 1024 * 1024),
          sycl_pipeline_chunk_size(CCL_ENV_SIZET_NOT_SPECIFIED),
          sycl_enable_pipeline_gpu_rdma(0),
          sycl_enable_direct_gpu_rdma(0),
          sycl_sub_communicator(1),
          sycl_force_pcie(0),
#endif // CCL_ENABLE_SYCL

          allreduce_nreduce_buffering(0),
          allreduce_nreduce_segment_size(CCL_ENV_SIZET_NOT_SPECIFIED),

          allreduce_2d_chunk_count(1),
          allreduce_2d_min_chunk_size(65536),
          allreduce_2d_switch_dims(0),

          dtree_partition_count(CCL_ENV_SIZET_NOT_SPECIFIED),

          check_inplace_aliasing(1),

          alltoall_scatter_max_ops(CCL_ENV_SIZET_NOT_SPECIFIED),

          backend(backend_mode::native),

          local_rank(CCL_ENV_INT_NOT_SPECIFIED),
          local_size(CCL_ENV_INT_NOT_SPECIFIED),

          process_launcher(process_launcher_mode::hydra),

          enable_topo_algo(1),
#ifdef CCL_ENABLE_SYCL
          topo_color(topo_color_mode::ze),
#else // CCL_ENABLE_SYCL
          topo_color(topo_color_mode::fixed),
#endif // CCL_ENABLE_SYCL
          enable_p2p_access(CCL_ENV_INT_NOT_SPECIFIED),
          enable_fabric_vertex_connection_check(1),
          enable_wa_fabric_vertex_connection_check(0),
          use_mpi_bcast_wa(0),
          use_root_print_wa(0),

#ifdef CCL_ENABLE_MPI
          mpi_lib_path(),
          mpi_bf16_native(1),
          mpi_fp16_native(1),
#endif // CCL_ENABLE_MPI
          ofi_lib_path(),

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_UMF)
          umf_enable(0),
          umf_lib_path(),
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE && CCL_ENABLE_UMF

#ifdef CCL_ENABLE_SYCL
          kernel_path(),
          kernel_debug(0),
          kernel_module_cache(0),

          // 32 is more generic constant value
          // for gpus to avoid imbalance issue
          kernel_group_size(32),
          kernel_group_count(CCL_ENV_SIZET_NOT_SPECIFIED),
          kernel_mem_align(128),

          enable_kernel_sync(1),
          kernel_1s_lead(0),
          enable_kernel_1s_copy_ops(0),
          enable_kernel_1s_ipc_wa(0),
          enable_close_fd_wa(1),

          enable_sycl_output_event(1),
          use_hmem(1),

          sync_barrier(0),
          sync_deps(0),

          enable_ze_barrier(0),
          enable_ze_bidir_algo(1),
          enable_ze_cache(1),
          ze_device_cache_evict_smallest(1),
          ze_device_cache_upper_limit(800 * 1024L * 1024L),
          ze_device_cache_num_blocks_in_chunk(1),
          ze_device_cache_policy(ccl::ze::device_cache_policy_mode::chunk),
          ze_pointer_registration_threshold(4 * 1024L * 1024L * 1024L),

          // Note: env. vars are required when
          // functionality is completed to support bypass/cache
          enable_ze_cache_cmdlists(1),
          enable_ze_cache_cmdqueues(1),
          enable_ze_cache_event_pools(1),

          enable_ze_cache_open_ipc_handles(1),
          ze_cache_open_ipc_handles_threshold(1000),
          ze_cache_get_ipc_handles_threshold(1000),
          enable_ze_cache_get_ipc_handles(1),
          enable_ze_single_list(1),
          disable_ze_family_check(0),
          disable_ze_port_check(0),
          ze_enable_oversubscription_fallback(1),
          ze_enable_oversubscription_throw(1),
          ze_serialize_mode(0),
          ze_copy_engine(ccl::ze::copy_engine_mode::link),
          ze_h2d_copy_engine(ccl::ze::h2d_copy_engine_mode::none),
          ze_d2d_copy_engine(ccl::ze::d2d_copy_engine_mode::none),
          ze_max_compute_queues(1),
          ze_max_copy_queues(CCL_ENV_SIZET_NOT_SPECIFIED),
          ze_enable_ccs_fallback_for_copy(1),
          enable_ze_list_dump(0),
          ze_queue_index_offset(0),
          ze_close_ipc_wa(0),
          ze_lib_path(),
          ze_enable(1),
          ze_fini_wa(0),
          ze_multi_workers(0),
          sycl_alltoall_protocol(ccl_sycl_alltoall_protocol::write),
          enable_ze_auto_tune_ports(1),
          ze_ipc_exchange(ccl::ze::ipc_exchange_mode::pidfd),
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
          ze_drm_bdf_support(1),
#else // ZE_PCI_PROPERTIES_EXT_NAME
          ze_drm_bdf_support(0),
#endif // ZE_PCI_PROPERTIES_EXT_NAME
          ze_pt2pt_read(1),
          type2_mode(type2_tune_mode::undetected),
#ifdef CCL_ENABLE_DRM
          drmfd_dev_render_dir_path("/dev/dri/by-path/"),
          drmfd_dev_render_suffix("-render"),
#endif // CCL_ENABLE_DRM
#endif // CCL_ENABLE_SYCL
          ipc_allgatherv_wa(1),

#ifdef CCL_ENABLE_PMIX
          pmix_lib_path(),
#endif // CCL_ENABLE_PMIX

#ifdef CCL_ENABLE_ITT
          itt_level(0),
#endif // CCL_ENABLE_ITT
          debug_timestamps_level(0),

          enable_profiling(0),

          bf16_impl_type(ccl_bf16_scalar),
          fp16_impl_type(ccl_fp16_no_compiler_support) {
}

void env_data::parse() {
    env_parser p;
    p.env_2_enum(CCL_LOG_LEVEL, ccl_logger::level_names, log_level);
    p.env_2_type(CCL_ABORT_ON_THROW, abort_on_throw);
    ccl_logger::set_log_level(log_level);
    ccl_logger::set_abort_on_throw(abort_on_throw);
    p.env_2_type(CCL_QUEUE_DUMP, queue_dump);
    p.env_2_type(CCL_SCHED_DUMP, sched_dump);
    p.env_2_type(CCL_SCHED_PROFILE, sched_profile);
    p.env_2_type(CCL_ENTRY_MAX_UPDATE_TIME_SEC, entry_max_update_time_sec);
    CCL_THROW_IF_NOT(
        entry_max_update_time_sec == CCL_ENV_SIZET_NOT_SPECIFIED || entry_max_update_time_sec > 0,
        "incorrect ",
        CCL_ENTRY_MAX_UPDATE_TIME_SEC,
        " ",
        entry_max_update_time_sec);

    if (fw_type == ccl_framework_none) {
        /* try to automatically detect framework */
        void* handle = dlopen(NULL, RTLD_GLOBAL | RTLD_NOW);
        if (handle) {
            horovod_init_function =
                (ccl_horovod_init_function)dlsym(handle, horovod_init_function_name);
            dlclose(handle);
        }

        if (horovod_init_function) {
            LOG_INFO("found horovod_init function");
            fw_type = ccl_framework_horovod;
        }
    }
    p.env_2_enum(CCL_FRAMEWORK, ccl_framework_type_names, fw_type);

    if (fw_type == ccl_framework_horovod) {
        // enable_sync_coll = 1;
        // enable_extra_ep = 1;
        worker_wait = true;
        yield_type = ccl_yield_sched_yield;
    }

    p.env_2_type(CCL_WORKER_COUNT, worker_count);
    CCL_THROW_IF_NOT(worker_count >= 1, "incorrect ", CCL_WORKER_COUNT, " ", worker_count);
    p.env_2_type(CCL_WORKER_OFFLOAD, worker_offload);
    p.env_2_type(CCL_WORKER_WAIT, worker_wait);

    p.env_2_atl_transport(atl_transport_names, atl_transport);
    p.env_2_enum(CCL_KVS_MODE, kvs_mode_names, kvs_init_mode);
    p.env_2_type(CCL_KVS_CONNECTION_TIMEOUT, kvs_connection_timeout);
    p.env_2_type(CCL_KVS_MPI_ALLGATHER, kvs_mpi_allgather);
    p.env_2_type(CCL_KVS_USE_MPI_RANKS, kvs_use_mpi_ranks);
    p.env_2_type(CCL_ATL_SHM, enable_shm);
    p.env_2_type(CCL_ATL_RMA, enable_rma);
    p.env_2_type(CCL_ATL_HMEM, enable_hmem);
    if (atl_transport == ccl_atl_mpi && enable_hmem) {
        LOG_INFO("atl hmem requested, switch to single worker");
        worker_count = 1;
    }
    p.env_2_enum(CCL_ATL_SEND_PROXY, atl_send_proxy_names, atl_send_proxy);
    p.env_2_type(CCL_ATL_CACHE, enable_atl_cache);
    p.env_2_type(CCL_ATL_SYNC_COLL, enable_sync_coll);
    p.env_2_type(CCL_ATL_EXTRA_EP, enable_extra_ep);
    p.env_2_type(CCL_ENABLE_AUTO_CACHE, enable_auto_cache);
#if defined(CCL_ENABLE_MPI) && defined(CCL_ENABLE_OMP)
    p.env_2_type(CCL_ENABLE_OMP_COLL, enable_omp_coll);
    p.env_2_type(CCL_ENABLE_OMP_ALLREDUCE, enable_omp_allreduce);
    p.env_2_type(CCL_ENABLE_OMP_ALLGATHERV, enable_omp_allgatherv);
    p.env_2_type(CCL_OMP_ALLREDUCE_NUM_THREADS, omp_allreduce_num_threads);
#endif // CCL_ENABLE_MPI && CCL_ENABLE_OMP
    p.env_2_enum(CCL_MNIC, mnic_type_names, mnic_type);
    p.env_2_type(CCL_MNIC_NAME, mnic_name_raw);
    p.env_2_type(CCL_MNIC_COUNT, mnic_count);
    if (mnic_count == CCL_ENV_SIZET_NOT_SPECIFIED) {
        mnic_count = worker_count;
    }
    p.env_2_enum(CCL_MNIC_OFFSET, mnic_offset_names, mnic_offset);

    p.env_2_type(CCL_ALGO_FALLBACK, enable_algo_fallback);
    // main algorithm selection
    p.env_2_type(CCL_ALLGATHER, allgather_algo_raw);
    p.env_2_type(CCL_ALLGATHERV, allgatherv_algo_raw);
    p.env_2_type(CCL_ALLREDUCE, allreduce_algo_raw);
    p.env_2_type(CCL_ALLTOALL, alltoall_algo_raw);
    p.env_2_type(CCL_ALLTOALLV, alltoallv_algo_raw);
    p.env_2_type(CCL_BARRIER, barrier_algo_raw);
    p.env_2_type(CCL_BCAST, bcast_algo_raw);
    p.env_2_type(CCL_BROADCAST, broadcast_algo_raw);
    p.env_2_type(CCL_RECV, recv_algo_raw);
    p.env_2_type(CCL_REDUCE, reduce_algo_raw);
    p.env_2_type(CCL_REDUCE_SCATTER, reduce_scatter_algo_raw);
    p.env_2_type(CCL_SEND, send_algo_raw);
    // scale-out selection part
    p.env_2_type(CCL_ALLGATHER_SCALEOUT, allgather_scaleout_algo_raw);
    p.env_2_type(CCL_ALLGATHERV_SCALEOUT, allgatherv_scaleout_algo_raw);
    p.env_2_type(CCL_ALLREDUCE_SCALEOUT, allreduce_scaleout_algo_raw);
    p.env_2_type(CCL_ALLTOALL_SCALEOUT, alltoall_scaleout_algo_raw);
    p.env_2_type(CCL_ALLTOALLV_SCALEOUT, alltoallv_scaleout_algo_raw);
    p.env_2_type(CCL_REDUCE_SCALEOUT, reduce_scaleout_algo_raw);
    p.env_2_type(CCL_REDUCE_SCATTER_SCALEOUT, reduce_scatter_scaleout_algo_raw);

    p.env_2_type(CCL_UNORDERED_COLL, enable_unordered_coll);
    if (enable_unordered_coll && atl_transport != ccl_atl_ofi) {
        CCL_THROW("unordered collectives are supported for OFI transport only");
    }

    p.env_2_type(CCL_FUSION, enable_fusion);
    p.env_2_type(CCL_FUSION_BYTES_THRESHOLD, fusion_bytes_threshold);
    p.env_2_type(CCL_FUSION_COUNT_THRESHOLD, fusion_count_threshold);
    p.env_2_type(CCL_FUSION_CHECK_URGENT, fusion_check_urgent);
    p.env_2_type(CCL_FUSION_CYCLE_MS, fusion_cycle_ms);
    if (enable_fusion) {
        CCL_THROW_IF_NOT(fusion_bytes_threshold >= 1,
                         "incorrect ",
                         CCL_FUSION_BYTES_THRESHOLD,
                         " ",
                         fusion_bytes_threshold);
        CCL_THROW_IF_NOT(fusion_count_threshold >= 1,
                         "incorrect ",
                         CCL_FUSION_COUNT_THRESHOLD,
                         " ",
                         fusion_count_threshold);
    }

    if (!worker_offload || enable_fusion)
        worker_wait = false;

    if (worker_wait)
        spin_count = 1000;

    p.env_2_enum(CCL_PRIORITY, priority_mode_names, priority_mode);
    p.env_2_type(CCL_SPIN_COUNT, spin_count);
    p.env_2_enum(CCL_YIELD, ccl_yield_type_names, yield_type);
    p.env_2_type(CCL_MAX_SHORT_SIZE, max_short_size);
    p.env_2_type(CCL_BCAST_PART_COUNT, (size_t&)bcast_part_count);
    p.env_2_enum(CCL_CACHE_KEY, ccl_sched_key::key_type_names, cache_key_type);
    p.env_2_type(CCL_CACHE_FLUSH, enable_cache_flush);
    p.env_2_type(CCL_BUFFER_CACHE, enable_buffer_cache);
    p.env_2_type(CCL_STRICT_ORDER, enable_strict_order);
    if (enable_unordered_coll && enable_strict_order) {
        LOG_INFO("unordered collectives are requested, disable strict order");
        enable_strict_order = 0;
    }
    p.env_2_enum(CCL_STAGING_BUFFER, staging_buffer_names, staging_buffer);
    p.env_2_type(CCL_OP_SYNC, enable_op_sync);
    p.env_2_type(CCL_OFI_ENABLE_HOSTNAME_SHARING, enable_hostname_sharing);
    p.env_2_type(CCL_OFI_INIT_ENABLE_HOSTNAME_SHARING, enable_init_hostname_sharing);

    p.env_2_type(CCL_CHUNK_COUNT, chunk_count);
    CCL_THROW_IF_NOT(chunk_count >= 1, "incorrect ", CCL_CHUNK_COUNT, " ", chunk_count);
    p.env_2_type(CCL_MIN_CHUNK_SIZE, min_chunk_size);
    CCL_THROW_IF_NOT(min_chunk_size >= 1, "incorrect ", CCL_MIN_CHUNK_SIZE, " ", min_chunk_size);
    p.env_2_type(CCL_ZE_TMP_BUF_SIZE, ze_tmp_buf_size);
    CCL_THROW_IF_NOT(ze_tmp_buf_size >= 1 && ze_tmp_buf_size % 4 == 0, "incorrect ", CCL_ZE_TMP_BUF_SIZE, " ", ze_tmp_buf_size);

    p.env_2_type(CCL_RS_CHUNK_COUNT, rs_chunk_count);
    CCL_THROW_IF_NOT(rs_chunk_count >= 1, "incorrect ", CCL_RS_CHUNK_COUNT, " ", rs_chunk_count);
    p.env_2_type(CCL_RS_MIN_CHUNK_SIZE, rs_min_chunk_size);
    CCL_THROW_IF_NOT(
        rs_min_chunk_size >= 1, "incorrect ", CCL_RS_MIN_CHUNK_SIZE, " ", rs_min_chunk_size);

#ifdef CCL_ENABLE_SYCL
    p.env_2_type(CCL_ALLGATHERV_TOPO_LARGE_SCALE, allgatherv_topo_large_scale);
    p.env_2_type(CCL_ALLGATHERV_TOPO_READ, allgatherv_topo_read);
    p.env_2_type(CCL_ALLTOALLV_TOPO_READ, alltoallv_topo_read);
    p.env_2_type(CCL_REDUCE_SCATTER_TOPO_READ, reduce_scatter_topo_read);
    p.env_2_type(CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL, reduce_scatter_monolithic_kernel);
    p.env_2_type(CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL,
                 reduce_scatter_monolithic_pipeline_kernel);
    p.env_2_type(CCL_REDUCE_SCATTER_FALLBACK_ALGO, reduce_scatter_fallback_algo);
    p.env_2_type(CCL_ALLGATHERV_MONOLITHIC_KERNEL, allgatherv_monolithic_kernel);
    p.env_2_type(CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL, allgatherv_monolithic_pipeline_kernel);
    p.env_2_type(CCL_ALLTOALLV_MONOLITHIC_KERNEL, alltoallv_monolithic_kernel);
    p.env_2_type(CCL_ALLTOALLV_MONOLITHIC_READ_KERNEL, alltoallv_monolithic_read_kernel);

    p.env_2_type(CCL_ALLGATHERV_PIPE_CHUNK_COUNT, allgatherv_pipe_chunk_count);
    p.env_2_type(CCL_ALLREDUCE_PIPE_CHUNK_COUNT, allreduce_pipe_chunk_count);
    p.env_2_type(CCL_REDUCE_SCATTER_PIPE_CHUNK_COUNT, reduce_scatter_pipe_chunk_count);
    p.env_2_type(CCL_REDUCE_PIPE_CHUNK_COUNT, reduce_pipe_chunk_count);

    p.env_2_type(CCL_SYCL_ALLREDUCE_TMP_BUF, sycl_allreduce_tmp_buf);
    p.env_2_type(CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD, sycl_allreduce_small_threshold);
    p.env_2_type(CCL_SYCL_ALLREDUCE_MEDIUM_THRESHOLD, sycl_allreduce_medium_threshold);
    p.env_2_type(CCL_SYCL_ALLREDUCE_SCALEOUT_THRESHOLD, sycl_allreduce_scaleout_threshold);
    p.env_2_type(CCL_SYCL_ALLREDUCE_SCALEOUT, sycl_allreduce_scaleout_algo);
    p.env_2_type(CCL_SYCL_ALLREDUCE_ARC, sycl_enable_arc_allreduce);
    p.env_2_type(CCL_SYCL_ALLREDUCE_LL_THRESHOLD, sycl_allreduce_ll_threshold);

    p.env_2_type(CCL_SYCL_REDUCE_SCATTER_TMP_BUF, sycl_reduce_scatter_tmp_buf);
    p.env_2_type(CCL_SYCL_REDUCE_SCATTER_SMALL_THRESHOLD, sycl_reduce_scatter_small_threshold);
    p.env_2_type(CCL_SYCL_REDUCE_SCATTER_MEDIUM_THRESHOLD, sycl_reduce_scatter_medium_threshold);
    p.env_2_type(CCL_SYCL_REDUCE_SCATTER_SCALEOUT_THRESHOLD, sycl_reduce_scatter_scaleout_threshold);
    p.env_2_type(CCL_SYCL_REDUCE_SCATTER_SCALEOUT, sycl_reduce_scatter_scaleout_algo);

    p.env_2_type(CCL_SYCL_ALLGATHERV_TMP_BUF, sycl_allgatherv_tmp_buf);
    p.env_2_type(CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD, sycl_allgatherv_small_threshold);
    p.env_2_type(CCL_SYCL_ALLGATHERV_MEDIUM_THRESHOLD, sycl_allgatherv_medium_threshold);
    p.env_2_type(CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD, sycl_allgatherv_scaleout_threshold);
    p.env_2_type(CCL_SYCL_ALLGATHERV_SCALEOUT, sycl_allgatherv_scaleout_algo);
    p.env_2_type(CCL_SYCL_ALLGATHERV_LL_THRESHOLD, sycl_allgatherv_ll_threshold);

    p.env_2_type(CCL_SYCL_BROADCAST_TMP_BUF, sycl_broadcast_tmp_buf);
    p.env_2_type(CCL_SYCL_BROADCAST_SMALL_THRESHOLD, sycl_broadcast_small_threshold);
    p.env_2_type(CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD, sycl_broadcast_scaleout_threshold);

    p.env_2_type(CCL_ENABLE_SYCL_KERNELS, enable_sycl_kernels);

    p.env_2_type(CCL_SYCL_CCL_BARRIER, sycl_ccl_barrier);
    p.env_2_type(CCL_SYCL_KERNEL_SYNC, sycl_kernel_sync);
    p.env_2_type(CCL_SYCL_SINGLE_NODE_ALGORITHM, sycl_single_node_algorithm);
    p.env_2_type(CCL_SYCL_AUTO_USE_TMP_BUF, sycl_auto_use_tmp_buf);
    p.env_2_type(CCL_SYCL_FORCE_USE_TMP_BUF_SCALEOUT, sycl_force_use_tmp_buf_scaleout);
    p.env_2_type(CCL_SYCL_COPY_ENGINE, sycl_copy_engine);
    p.env_2_type(CCL_SYCL_KERNEL_COPY, sycl_kernel_copy);
    p.env_2_type(CCL_SYCL_ESIMD, sycl_esimd);
    p.env_2_type(CCL_SYCL_FULL_VECTOR, sycl_full_vector);
    p.env_2_type(CCL_SYCL_FORCE_RECORDING_PATH, sycl_force_recording_path);
    p.env_2_type(CCL_SYCL_TMP_BUF_SIZE, sycl_tmp_buf_size);
    p.env_2_type(CCL_SYCL_SCALEOUT_HOST_BUF_SIZE, sycl_scaleout_host_buf_size);
    p.env_2_type(CCL_SYCL_SCALEOUT_DEVICE_BUF_SIZE, sycl_scaleout_device_buf_size);
    p.env_2_type(CCL_SYCL_KERNELS_LINE_SIZE, sycl_kernels_line_size);
    p.env_2_enum(CCL_SYCL_SCALEOUT_BUF_ALLOC_MODE, ccl::utils::alloc_mode_names, sycl_scaleout_buf_alloc_mode);
    p.env_2_type(CCL_SYCL_PT2PT_READ, sycl_pt2pt_read);
    p.env_2_type(CCL_SYCL_PT2PT_ENABLE, sycl_pt2pt_enable);
    p.env_2_type(CCL_SYCL_MAX_PIPELINE_CHUNK_SIZE, sycl_max_pipeline_chunk_size);
    p.env_2_type(CCL_SYCL_PIPELINE_CHUNK_SIZE, (size_t&)sycl_pipeline_chunk_size);
    p.env_2_type(CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA, sycl_enable_pipeline_gpu_rdma);
    p.env_2_type(CCL_SYCL_ENABLE_DIRECT_GPU_RDMA, sycl_enable_direct_gpu_rdma);
    p.env_2_type(CCL_SYCL_SUB_COMMUICATOR, sycl_sub_communicator);
    p.env_2_type(CCL_SYCL_FORCE_PCIE, sycl_force_pcie);
#endif // CCL_ENABLE_SYCL

    p.env_2_type(CCL_ALLREDUCE_NREDUCE_BUFFERING, allreduce_nreduce_buffering);
    p.env_2_type(CCL_ALLREDUCE_NREDUCE_SEGMENT_SIZE, (size_t&)allreduce_nreduce_segment_size);

    p.env_2_type(CCL_DTREE_PARTITION_COUNT, (size_t&)dtree_partition_count);

    p.env_2_type(CCL_ALLREDUCE_2D_CHUNK_COUNT, allreduce_2d_chunk_count);
    CCL_THROW_IF_NOT(allreduce_2d_chunk_count >= 1,
                     "incorrect ",
                     CCL_ALLREDUCE_2D_CHUNK_COUNT,
                     " ",
                     allreduce_2d_chunk_count);
    p.env_2_type(CCL_ALLREDUCE_2D_MIN_CHUNK_SIZE, allreduce_2d_min_chunk_size);
    CCL_THROW_IF_NOT(allreduce_2d_min_chunk_size >= 1,
                     "incorrect ",
                     CCL_ALLREDUCE_2D_MIN_CHUNK_SIZE,
                     " ",
                     allreduce_2d_min_chunk_size);
    p.env_2_type(CCL_ALLREDUCE_2D_SWITCH_DIMS, allreduce_2d_switch_dims);

    p.env_2_type(CCL_CHECK_INPLACE_ALIASING, check_inplace_aliasing);

    p.env_2_type(CCL_ALLTOALL_SCATTER_MAX_OPS, (size_t&)alltoall_scatter_max_ops);

    p.env_2_enum(CCL_BACKEND, backend_names, backend);

    p.env_2_type(CCL_LOCAL_RANK, local_rank);
    p.env_2_type(CCL_LOCAL_SIZE, local_size);

    p.env_2_enum(CCL_PROCESS_LAUNCHER, process_launcher_names, process_launcher);

    p.env_2_type(CCL_TOPO_ALGO, enable_topo_algo);
    p.env_2_topo(CCL_TOPO_COLOR, topo_color_names, topo_color);
    p.env_2_type(CCL_TOPO_P2P_ACCESS, enable_p2p_access);
    p.env_2_type(CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK, enable_fabric_vertex_connection_check);
    p.env_2_type(CCL_TOPO_WA_FABRIC_VERTEX_CONNECTION_CHECK, enable_wa_fabric_vertex_connection_check);
    p.env_2_type(CCL_USE_MPI_BCAST_WA, use_mpi_bcast_wa);
    p.env_2_type(CCL_USE_ROOT_PRINT_WA, use_root_print_wa);

#ifdef CCL_ENABLE_MPI
    p.env_2_type(CCL_MPI_LIBRARY_PATH, mpi_lib_path);
    p.env_2_type(CCL_ATL_MPI_BF16, mpi_bf16_native);
    p.env_2_type(CCL_ATL_MPI_FP16, mpi_fp16_native);
#endif // CCL_ENABLE_MPI
    p.env_2_type(CCL_OFI_LIBRARY_PATH, ofi_lib_path);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_UMF)
    p.env_2_type(CCL_UMF_ENABLE, umf_enable);
    p.env_2_type(CCL_UMF_LIBRARY_PATH, umf_lib_path);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE && CCL_ENABLE_UMF

#ifdef CCL_ENABLE_SYCL
    p.env_2_type(CCL_KERNEL_PATH, kernel_path);
    if (kernel_path.empty()) {
        std::string ccl_root;
        char* ccl_root_env_value = getenv("CCL_ROOT");
        char* oneapi_root_env_value = getenv("ONEAPI_ROOT");
        if (ccl_root_env_value) {
            ccl_root = ccl_root_env_value;
        }
        else if (oneapi_root_env_value) {
            ccl_root = oneapi_root_env_value;
        }

        if (ccl_root.empty()) {
            // CCL_ROOT and ONEAPI_ROOT are missing
            Dl_info info;
            if (dladdr((void*)ccl::get_library_version, &info)) {
                char libccl_path[PATH_MAX];
                std::strncpy(libccl_path, info.dli_fname, PATH_MAX - 1);
                libccl_path[PATH_MAX - 1] = '\0';

                char* libccl_dir = dirname(libccl_path);
                ccl_root = dirname(libccl_dir);
            }
        }

        CCL_THROW_IF_NOT(!ccl_root.empty(),
                         "incorrect comm kernels path, neither CCL_ROOT nor ONEAPI_ROOT found!");
        kernel_path = ccl_root + "/lib/ccl/kernels/";
    }

    p.env_2_type(CCL_KERNEL_DEBUG, kernel_debug);
    p.env_2_type(CCL_KERNEL_MODULE_CACHE, kernel_module_cache);
    p.env_2_type(CCL_KERNEL_GROUP_SIZE, kernel_group_size);
    p.env_2_type(CCL_KERNEL_GROUP_COUNT, kernel_group_count);
    p.env_2_type(CCL_KERNEL_MEM_ALIGN, kernel_mem_align);
    p.env_2_type(CCL_KERNEL_SYNC, enable_kernel_sync);
    p.env_2_type(CCL_KERNEL_1S_LEAD, kernel_1s_lead);
    p.env_2_type(CCL_KERNEL_1S_USE_COPY_OPS, enable_kernel_1s_copy_ops);
    p.env_2_type(CCL_KERNEL_1S_IPC_WA, enable_kernel_1s_ipc_wa);
    p.env_2_type(CCL_KERNEL_CLOSE_FD_WA, enable_close_fd_wa);

    p.env_2_type(CCL_SYCL_OUTPUT_EVENT, enable_sycl_output_event);
    p.env_2_type(CCL_USE_HMEM, use_hmem);

    p.env_2_type(CCL_BARRIER_SYNC, sync_barrier);
    p.env_2_type(CCL_ZE_DEPS_SYNC, sync_deps);

    p.env_2_type(CCL_ZE_BARRIER, enable_ze_barrier);
    p.env_2_type(CCL_ZE_BIDIR_ALGO, enable_ze_bidir_algo);
    p.env_2_type(CCL_ZE_CACHE, enable_ze_cache);
    p.env_2_type(CCL_ZE_DEVICE_CACHE_EVICT_SMALLEST, ze_device_cache_evict_smallest);
    p.env_2_type(CCL_ZE_DEVICE_CACHE_UPPER_LIMIT, ze_device_cache_upper_limit);
    p.env_2_type(CCL_ZE_DEVICE_CACHE_NUM_BLOCKS_IN_CHUNK, ze_device_cache_num_blocks_in_chunk);
    p.env_2_enum(CCL_ZE_DEVICE_CACHE_POLICY, ccl::ze::device_cache_policy_names, ze_device_cache_policy);
    p.env_2_type(CCL_ZE_PTR_REGISTER_THRESHOLD, ze_pointer_registration_threshold);
    p.env_2_type(CCL_ZE_CACHE_OPEN_IPC_HANDLES, enable_ze_cache_open_ipc_handles);
    p.env_2_type(CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD, ze_cache_open_ipc_handles_threshold);
    if (enable_ze_cache == 0) {
        enable_ze_cache_open_ipc_handles = 0;
    }
    else if (enable_ze_cache && enable_ze_cache_open_ipc_handles) {
        CCL_THROW_IF_NOT(ze_cache_open_ipc_handles_threshold > 0,
                         "incorrect ",
                         CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD,
                         " ",
                         ze_cache_open_ipc_handles_threshold);
    }
    p.env_2_type(CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD, ze_cache_get_ipc_handles_threshold);
    p.env_2_type(CCL_ZE_CACHE_GET_IPC_HANDLES, enable_ze_cache_get_ipc_handles);
    p.env_2_type(CCL_ZE_SINGLE_LIST, enable_ze_single_list);
    p.env_2_type(CCL_ZE_DISABLE_FAMILY_CHECK, disable_ze_family_check);
    p.env_2_type(CCL_ZE_DISABLE_PORT_CHECK, disable_ze_port_check);
    p.env_2_type(CCL_ZE_ENABLE_OVERSUBSCRIPTION_FALLBACK, ze_enable_oversubscription_fallback);
    p.env_2_type(CCL_ZE_ENABLE_OVERSUBSCRIPTION_THROW, ze_enable_oversubscription_throw);
    p.env_2_type(CCL_ZE_SERIALIZE, ze_serialize_mode);
    p.env_2_enum(CCL_ZE_COPY_ENGINE, ccl::ze::copy_engine_names, ze_copy_engine);
    p.env_2_enum(CCL_ZE_H2D_COPY_ENGINE, ccl::ze::h2d_copy_engine_names, ze_h2d_copy_engine);
    p.env_2_enum(CCL_ZE_D2D_COPY_ENGINE, ccl::ze::d2d_copy_engine_names, ze_d2d_copy_engine);
    p.env_2_type(CCL_ZE_MAX_COMPUTE_QUEUES, ze_max_compute_queues);
    CCL_THROW_IF_NOT(
        ze_max_compute_queues == CCL_ENV_SIZET_NOT_SPECIFIED || ze_max_compute_queues > 0,
        "incorrect ",
        CCL_ZE_MAX_COMPUTE_QUEUES,
        " ",
        ze_max_compute_queues);
    p.env_2_type(CCL_ZE_MAX_COPY_QUEUES, ze_max_copy_queues);
    CCL_THROW_IF_NOT(ze_copy_engine == ccl::ze::copy_engine_mode::none ||
                         ze_max_copy_queues == CCL_ENV_SIZET_NOT_SPECIFIED ||
                         ze_max_copy_queues > 0,
                     "incorrect ",
                     CCL_ZE_MAX_COPY_QUEUES,
                     " ",
                     ze_max_copy_queues);
    p.env_2_type(CCL_ZE_ENABLE_CCS_FALLBACK_FOR_COPY, ze_enable_ccs_fallback_for_copy);
    p.env_2_type(CCL_ZE_LIST_DUMP, enable_ze_list_dump);
    p.env_2_type(CCL_ZE_QUEUE_INDEX_OFFSET, ze_queue_index_offset);
    CCL_THROW_IF_NOT(ze_queue_index_offset >= 0,
                     "incorrect ",
                     CCL_ZE_QUEUE_INDEX_OFFSET,
                     " ",
                     ze_queue_index_offset);

    p.env_2_type(CCL_ZE_CLOSE_IPC_WA, ze_close_ipc_wa);
    p.env_2_type(CCL_ZE_LIBRARY_PATH, ze_lib_path);
    p.env_2_type(CCL_ZE_ENABLE, ze_enable);
    p.env_2_type(CCL_ZE_FINI_WA, ze_fini_wa);
    p.env_2_type(CCL_ZE_MULTI_WORKERS, ze_multi_workers);
    p.env_2_enum(CCL_SYCL_ALLTOALL_PROTOCOL, sycl_alltoall_protocols, sycl_alltoall_protocol);
    p.env_2_type(CCL_ZE_AUTO_TUNE_PORTS, enable_ze_auto_tune_ports);
    p.env_2_enum(CCL_ZE_IPC_EXCHANGE, ze::ipc_exchange_names, ze_ipc_exchange);
    p.env_2_type(CCL_ZE_DRM_BDF_SUPPORT, ze_drm_bdf_support);
    p.env_2_type(CCL_ZE_PT2PT_READ, ze_pt2pt_read);
    p.env_2_enum(CCL_ZE_TYPE2_TUNE_PORTS, type2_tune_mode_names, type2_mode);
    p.env_2_type(CCL_DRMFD_DEV_RENDER_DIR_PATH, drmfd_dev_render_dir_path);
    p.env_2_type(CCL_DRMFD_DEV_RENDER_SUFFIX, drmfd_dev_render_suffix);
#endif // CCL_ENABLE_SYCL
    p.env_2_type(CCL_IPC_ALLGATHERV_WA, ipc_allgatherv_wa);

#ifdef CCL_ENABLE_PMIX
    p.env_2_type(CCL_PMIX_LIBRARY_PATH, pmix_lib_path);
#endif // CCL_ENABLE_PMIX

#ifdef CCL_ENABLE_ITT
    p.env_2_type(CCL_ITT_LEVEL, itt_level);
#endif // CCL_ENABLE_ITT
    p.env_2_type(CCL_DEBUG_TIMESTAMPS_LEVEL, debug_timestamps_level);

    p.env_2_type(CCL_PROFILING_ENABLE, enable_profiling);

    auto bf16_impl_types = ccl_bf16_get_impl_types();
    bf16_impl_type = *bf16_impl_types.rbegin();
    p.env_2_enum(CCL_BF16, bf16_impl_names, bf16_impl_type);

    auto fp16_impl_types = ccl_fp16_get_impl_types();
    fp16_impl_type = *fp16_impl_types.rbegin();
    p.env_2_enum(CCL_FP16, fp16_env_impl_names, fp16_impl_type);
    CCL_THROW_IF_NOT(fp16_impl_types.find(fp16_impl_type) != fp16_impl_types.end(),
                     "unsupported FP16 impl type: ",
                     fp16_env_impl_names[fp16_impl_type]);

    p.warn_about_unused_var();
}

void env_data::print(int rank, bool is_profile_mode, bool is_mt_enabled) {
    std::lock_guard<ccl_spinlock> lock{ print_guard };

#if defined(CCL_ENABLE_PROFILING)
    if (is_profile_mode) {
        if (was_profiled)
            return;
        else {
            was_profiled = true;
        }
    }
    else {
        if (was_printed)
            return;
        else
            was_printed = true;
    }
#else
    if (was_printed)
        return;
    else
        was_printed = true;
#endif

    auto& global_data = ccl::global_data::get();

    if (rank == 0) {
        auto version = utils::get_library_version();
        LOG_INFO_PROFILED("library version: ", version.full);
        LOG_INFO_PROFILED("specification version: ", ONECCL_SPEC_VERSION);
#ifdef CCL_ENABLE_SYCL
        LOG_INFO_PROFILED("compute backend: ", version.cl_backend_name);
#endif // CCL_ENABLE_SYCL

#ifdef ENABLE_DEBUG
        const char* build_mode = "debug";
#else // ENABLE_DEBUG
        const char* build_mode = "release";
#endif // ENABLE_DEBUG
        LOG_INFO_PROFILED("build mode: ", build_mode);
        LOG_INFO_PROFILED("C compiler: ", CCL_C_COMPILER);
        LOG_INFO_PROFILED("C++ compiler: ", CCL_CXX_COMPILER);
        LOG_INFO_PROFILED(global_data.hwloc_wrapper->to_string());
    }

    // TODO: here is segfault seems in worker_affinity / worker_mem_affinity
    //  when you use threads, fix it.
    if (!is_mt_enabled) {
        auto local_proc_idx = global_data.get_local_proc_idx();
        auto local_proc_count = global_data.get_local_proc_count();

        if (rank < local_proc_count) {
            for (size_t w_idx = 0; w_idx < worker_count; w_idx++) {
                LOG_INFO_PROFILED("local process [",
                                  local_proc_idx,
                                  ":",
                                  local_proc_count,
                                  "]: worker: ",
                                  w_idx,
                                  ", cpu: ",
                                  worker_affinity[local_proc_idx * worker_count + w_idx],
                                  ", numa: ",
                                  worker_mem_affinity[local_proc_idx * worker_count + w_idx]);
            }
        }
    }

    if (rank != 0)
        return;

    LOG_INFO_PROFILED(CCL_WORKER_COUNT, ": ", worker_count);
    LOG_INFO_PROFILED(CCL_WORKER_OFFLOAD, ": ", worker_offload);
    LOG_INFO_PROFILED(CCL_WORKER_WAIT, ": ", worker_wait);

    LOG_INFO_PROFILED(CCL_LOG_LEVEL, ": ", str_by_enum(ccl_logger::level_names, log_level));
    LOG_INFO_PROFILED(CCL_ABORT_ON_THROW, ": ", abort_on_throw);
    LOG_INFO_PROFILED(CCL_QUEUE_DUMP, ": ", queue_dump);
    LOG_INFO_PROFILED(CCL_SCHED_DUMP, ": ", sched_dump);
    LOG_INFO_PROFILED(CCL_SCHED_PROFILE, ": ", sched_profile);
    LOG_INFO_PROFILED(CCL_ENTRY_MAX_UPDATE_TIME_SEC,
                      ": ",
                      (entry_max_update_time_sec != CCL_ENV_SIZET_NOT_SPECIFIED)
                          ? std::to_string(entry_max_update_time_sec)
                          : CCL_ENV_STR_NOT_SPECIFIED);

    LOG_INFO_PROFILED(CCL_FRAMEWORK, ": ", str_by_enum(ccl_framework_type_names, fw_type));

    LOG_INFO_PROFILED(CCL_ATL_TRANSPORT, ": ", str_by_enum(atl_transport_names, atl_transport));
    LOG_INFO_PROFILED(CCL_KVS_MODE, ": ", str_by_enum(kvs_mode_names, kvs_init_mode));
    LOG_INFO_PROFILED(CCL_KVS_CONNECTION_TIMEOUT, ": ", kvs_connection_timeout);
    LOG_INFO_PROFILED(CCL_KVS_MPI_ALLGATHER, ": ", kvs_mpi_allgather);
    LOG_INFO_PROFILED(CCL_KVS_USE_MPI_RANKS, ": ", kvs_use_mpi_ranks);
    LOG_INFO_PROFILED(CCL_ATL_SHM, ": ", enable_shm);
    LOG_INFO_PROFILED(CCL_ATL_RMA, ": ", enable_rma);
    LOG_INFO_PROFILED(CCL_ATL_HMEM, ": ", enable_hmem);
    LOG_INFO_PROFILED(CCL_ATL_SEND_PROXY, ": ", str_by_enum(atl_send_proxy_names, atl_send_proxy));
    LOG_INFO_PROFILED(CCL_ATL_CACHE, ": ", enable_atl_cache);
    LOG_DEBUG(CCL_ATL_SYNC_COLL, ": ", enable_sync_coll);
    LOG_DEBUG(CCL_ATL_EXTRA_EP, ": ", enable_extra_ep);
    LOG_DEBUG(CCL_ENABLE_AUTO_CACHE, ": ", enable_auto_cache);
#if defined(CCL_ENABLE_MPI) && defined(CCL_ENABLE_OMP)
    LOG_DEBUG(CCL_ENABLE_OMP_COLL, ": ", enable_omp_coll);
    LOG_DEBUG(CCL_ENABLE_OMP_ALLREDUCE, ": ", enable_omp_allreduce);
    LOG_DEBUG(CCL_ENABLE_OMP_ALLGATHERV, ": ", enable_omp_allgatherv);
    LOG_DEBUG(CCL_OMP_ALLREDUCE_NUM_THREADS, ": ", omp_allreduce_num_threads);
#endif // CCL_ENABLE_MPI && CCL_ENABLE_OMP
    LOG_INFO_PROFILED(CCL_MNIC, ": ", str_by_enum(mnic_type_names, mnic_type));
    LOG_INFO_PROFILED(
        CCL_MNIC_NAME, ": ", (mnic_name_raw.length()) ? mnic_name_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_MNIC_COUNT, ": ", mnic_count);
    LOG_INFO_PROFILED(CCL_MNIC_OFFSET, ": ", str_by_enum(mnic_offset_names, mnic_offset));

    LOG_INFO_PROFILED(CCL_ALGO_FALLBACK, ": ", enable_algo_fallback);
    LOG_INFO_PROFILED(CCL_ALLGATHER,
                      ": ",
                      (allgather_algo_raw.length()) ? allgather_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ALLGATHERV,
                      ": ",
                      (allgatherv_algo_raw.length()) ? allgatherv_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ALLREDUCE,
                      ": ",
                      (allreduce_algo_raw.length()) ? allreduce_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ALLTOALL,
                      ": ",
                      (alltoall_algo_raw.length()) ? alltoall_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ALLTOALLV,
                      ": ",
                      (alltoallv_algo_raw.length()) ? alltoallv_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_BARRIER,
                      ": ",
                      (barrier_algo_raw.length()) ? barrier_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(
        CCL_BCAST, ": ", (bcast_algo_raw.length()) ? bcast_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(
        CCL_BROADCAST, ": ", (broadcast_algo_raw.length()) ? broadcast_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_RECV, ": ", (recv_algo_raw.length()) ? recv_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(
        CCL_REDUCE, ": ", (reduce_algo_raw.length()) ? reduce_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(
        CCL_REDUCE_SCATTER,
        ": ",
        (reduce_scatter_algo_raw.length()) ? reduce_scatter_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_SEND, ": ", (send_algo_raw.length()) ? send_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ALLGATHER_SCALEOUT,
                      ": ",
                      (allgather_scaleout_algo_raw.length()) ? allgather_scaleout_algo_raw
                                                             : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ALLGATHERV_SCALEOUT,
                      ": ",
                      (allgatherv_scaleout_algo_raw.length()) ? allgatherv_scaleout_algo_raw
                                                              : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ALLREDUCE_SCALEOUT,
                      ": ",
                      (allreduce_scaleout_algo_raw.length()) ? allreduce_scaleout_algo_raw
                                                             : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ALLTOALL_SCALEOUT,
                      ": ",
                      (alltoall_scaleout_algo_raw.length()) ? alltoall_scaleout_algo_raw
                                                            : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ALLTOALLV_SCALEOUT,
                      ": ",
                      (alltoallv_scaleout_algo_raw.length()) ? alltoallv_scaleout_algo_raw
                                                             : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(
        CCL_REDUCE_SCALEOUT,
        ": ",
        (reduce_scaleout_algo_raw.length()) ? reduce_scaleout_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(
        CCL_REDUCE_SCATTER_SCALEOUT,
        ": ",
        (reduce_scatter_scaleout_algo_raw.length()) ? reduce_scatter_scaleout_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_UNORDERED_COLL, ": ", enable_unordered_coll);

    LOG_INFO_PROFILED(CCL_FUSION, ": ", enable_fusion);
    LOG_INFO_PROFILED(CCL_FUSION_BYTES_THRESHOLD, ": ", fusion_bytes_threshold);
    LOG_INFO_PROFILED(CCL_FUSION_COUNT_THRESHOLD, ": ", fusion_count_threshold);
    LOG_INFO_PROFILED(CCL_FUSION_CHECK_URGENT, ": ", fusion_check_urgent);
    LOG_INFO_PROFILED(CCL_FUSION_CYCLE_MS, ": ", fusion_cycle_ms);

    LOG_INFO_PROFILED(CCL_PRIORITY, ": ", str_by_enum(priority_mode_names, priority_mode));
    LOG_INFO_PROFILED(CCL_SPIN_COUNT, ": ", spin_count);
    LOG_INFO_PROFILED(CCL_YIELD, ": ", str_by_enum(ccl_yield_type_names, yield_type));
    LOG_INFO_PROFILED(CCL_MAX_SHORT_SIZE, ": ", max_short_size);
    LOG_INFO_PROFILED(CCL_BCAST_PART_COUNT,
                      ": ",
                      (bcast_part_count != CCL_ENV_SIZET_NOT_SPECIFIED) ? std::to_string(bcast_part_count)
                                                                        : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_CACHE_KEY, ": ", str_by_enum(ccl_sched_key::key_type_names, cache_key_type));
    LOG_INFO_PROFILED(CCL_CACHE_FLUSH, ": ", enable_cache_flush);
    LOG_INFO_PROFILED(CCL_BUFFER_CACHE, ": ", enable_buffer_cache);
    LOG_INFO_PROFILED(CCL_STRICT_ORDER, ": ", enable_strict_order);
    LOG_INFO_PROFILED(CCL_STAGING_BUFFER, ": ", str_by_enum(staging_buffer_names, staging_buffer));
    LOG_INFO_PROFILED(CCL_OP_SYNC, ": ", enable_op_sync);
    LOG_INFO_PROFILED(CCL_OFI_ENABLE_HOSTNAME_SHARING, ": ", enable_hostname_sharing);
    LOG_INFO_PROFILED(CCL_OFI_INIT_ENABLE_HOSTNAME_SHARING, ": ", enable_init_hostname_sharing);

    LOG_INFO_PROFILED(CCL_CHUNK_COUNT, ": ", chunk_count);
    LOG_INFO_PROFILED(CCL_MIN_CHUNK_SIZE, ": ", min_chunk_size);
    LOG_INFO_PROFILED(CCL_RS_CHUNK_COUNT, ": ", rs_chunk_count);
    LOG_INFO_PROFILED(CCL_RS_MIN_CHUNK_SIZE, ": ", rs_min_chunk_size);

#ifdef CCL_ENABLE_SYCL
    LOG_INFO_PROFILED(CCL_ALLGATHERV_TOPO_LARGE_SCALE, ": ", allgatherv_topo_large_scale);
    LOG_INFO_PROFILED(CCL_ALLGATHERV_TOPO_READ, ": ", allgatherv_topo_read);
    LOG_INFO_PROFILED(CCL_ALLTOALLV_TOPO_READ, ": ", alltoallv_topo_read);
    LOG_INFO_PROFILED(CCL_REDUCE_SCATTER_TOPO_READ, ": ", reduce_scatter_topo_read);
    LOG_INFO_PROFILED(CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL, ": ", reduce_scatter_monolithic_kernel);
    LOG_INFO_PROFILED(CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL,
                      ": ",
                      reduce_scatter_monolithic_pipeline_kernel);
    LOG_INFO_PROFILED(CCL_REDUCE_SCATTER_FALLBACK_ALGO, ": ", reduce_scatter_fallback_algo);
    LOG_INFO_PROFILED(CCL_ALLGATHERV_MONOLITHIC_KERNEL, ": ", allgatherv_monolithic_kernel);
    LOG_INFO_PROFILED(
        CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL, ": ", allgatherv_monolithic_pipeline_kernel);
    LOG_INFO_PROFILED(CCL_ALLTOALLV_MONOLITHIC_KERNEL, ": ", alltoallv_monolithic_kernel);
    LOG_INFO_PROFILED(CCL_ALLTOALLV_MONOLITHIC_READ_KERNEL, ": ", alltoallv_monolithic_read_kernel);

    LOG_INFO_PROFILED(CCL_ALLGATHERV_PIPE_CHUNK_COUNT, ": ", allgatherv_pipe_chunk_count);
    LOG_INFO_PROFILED(CCL_ALLREDUCE_PIPE_CHUNK_COUNT, ": ", allreduce_pipe_chunk_count);
    LOG_INFO_PROFILED(CCL_REDUCE_SCATTER_PIPE_CHUNK_COUNT, ": ", reduce_scatter_pipe_chunk_count);
    LOG_INFO_PROFILED(CCL_REDUCE_PIPE_CHUNK_COUNT, ": ", reduce_pipe_chunk_count);

    LOG_INFO_PROFILED(CCL_SYCL_ALLREDUCE_TMP_BUF, ": ", sycl_allreduce_tmp_buf);
    LOG_INFO_PROFILED(CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD, ": ", sycl_allreduce_small_threshold);
    LOG_INFO_PROFILED(CCL_SYCL_ALLREDUCE_MEDIUM_THRESHOLD, ": ", sycl_allreduce_medium_threshold);
    LOG_INFO_PROFILED(CCL_SYCL_ALLREDUCE_SCALEOUT_THRESHOLD, ": ", sycl_allreduce_scaleout_threshold);
    LOG_INFO_PROFILED(CCL_SYCL_ALLREDUCE_SCALEOUT, ": ", (!sycl_allreduce_scaleout_algo.empty()) ? sycl_allreduce_scaleout_algo : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_SYCL_ALLREDUCE_ARC, ": ", sycl_enable_arc_allreduce);
    LOG_INFO_PROFILED(CCL_SYCL_ALLREDUCE_LL_THRESHOLD, ": ", sycl_allreduce_ll_threshold);

    LOG_INFO_PROFILED(CCL_SYCL_REDUCE_SCATTER_TMP_BUF, ": ", sycl_reduce_scatter_tmp_buf);
    LOG_INFO_PROFILED(CCL_SYCL_REDUCE_SCATTER_SMALL_THRESHOLD, ": ", sycl_reduce_scatter_small_threshold);
    LOG_INFO_PROFILED(CCL_SYCL_REDUCE_SCATTER_MEDIUM_THRESHOLD, ": ", sycl_reduce_scatter_medium_threshold);
    LOG_INFO_PROFILED(CCL_SYCL_REDUCE_SCATTER_SCALEOUT_THRESHOLD, ": ", sycl_reduce_scatter_scaleout_threshold);
    LOG_INFO_PROFILED(CCL_SYCL_REDUCE_SCATTER_SCALEOUT, ": ", (!sycl_reduce_scatter_scaleout_algo.empty()) ? sycl_reduce_scatter_scaleout_algo : CCL_ENV_STR_NOT_SPECIFIED);

    LOG_INFO_PROFILED(CCL_SYCL_ALLGATHERV_TMP_BUF, ": ", sycl_allgatherv_tmp_buf);
    LOG_INFO_PROFILED(CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD, ": ", sycl_allgatherv_small_threshold);
    LOG_INFO_PROFILED(CCL_SYCL_ALLGATHERV_MEDIUM_THRESHOLD, ": ", sycl_allgatherv_medium_threshold);
    LOG_INFO_PROFILED(CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD, ": ", sycl_allgatherv_scaleout_threshold);
    LOG_INFO_PROFILED(CCL_SYCL_ALLGATHERV_SCALEOUT, ": ", (!sycl_allgatherv_scaleout_algo.empty()) ? sycl_allgatherv_scaleout_algo : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_SYCL_ALLGATHERV_LL_THRESHOLD, ": ", sycl_allgatherv_ll_threshold);

    LOG_INFO_PROFILED(CCL_SYCL_BROADCAST_TMP_BUF, ": ", sycl_broadcast_tmp_buf);
    LOG_INFO_PROFILED(CCL_SYCL_BROADCAST_SMALL_THRESHOLD, ": ", sycl_broadcast_small_threshold);
    LOG_INFO_PROFILED(CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD, ": ", sycl_broadcast_scaleout_threshold);

    LOG_INFO_PROFILED(CCL_ENABLE_SYCL_KERNELS, ": ", enable_sycl_kernels);

    LOG_INFO_PROFILED(CCL_SYCL_CCL_BARRIER, ": ", sycl_ccl_barrier);
    LOG_INFO_PROFILED(CCL_SYCL_KERNEL_SYNC, ": ", sycl_kernel_sync);
    LOG_INFO_PROFILED(CCL_SYCL_SINGLE_NODE_ALGORITHM, ": ", sycl_single_node_algorithm);
    LOG_INFO_PROFILED(CCL_SYCL_AUTO_USE_TMP_BUF, ": ", sycl_auto_use_tmp_buf);
    LOG_INFO_PROFILED(CCL_SYCL_FORCE_USE_TMP_BUF_SCALEOUT, ": ", sycl_force_use_tmp_buf_scaleout);
    LOG_INFO_PROFILED(CCL_SYCL_COPY_ENGINE, ": ", sycl_copy_engine);
    LOG_INFO_PROFILED(CCL_SYCL_KERNEL_COPY, ": ", sycl_kernel_copy);
    LOG_INFO_PROFILED(CCL_SYCL_ESIMD, ": ", sycl_esimd);
    LOG_INFO_PROFILED(CCL_SYCL_FULL_VECTOR, ": ", sycl_full_vector);
    LOG_INFO_PROFILED(CCL_SYCL_TMP_BUF_SIZE, ": ", sycl_tmp_buf_size);
    LOG_INFO_PROFILED(CCL_SYCL_SCALEOUT_HOST_BUF_SIZE, ": ", sycl_scaleout_host_buf_size);
    LOG_INFO_PROFILED(CCL_SYCL_SCALEOUT_DEVICE_BUF_SIZE, ": ", sycl_scaleout_device_buf_size);
    LOG_INFO_PROFILED(CCL_SYCL_KERNELS_LINE_SIZE, ": ", sycl_kernels_line_size);
    LOG_INFO_PROFILED(CCL_SYCL_SCALEOUT_BUF_ALLOC_MODE, ": ", str_by_enum(ccl::utils::alloc_mode_names, sycl_scaleout_buf_alloc_mode));
    LOG_INFO_PROFILED(CCL_SYCL_PT2PT_READ, ": ", sycl_pt2pt_read);
    LOG_INFO_PROFILED(CCL_SYCL_PT2PT_ENABLE, ": ", sycl_pt2pt_enable);
    LOG_INFO_PROFILED(CCL_SYCL_MAX_PIPELINE_CHUNK_SIZE, ": ", sycl_max_pipeline_chunk_size);
    LOG_INFO_PROFILED(CCL_SYCL_PIPELINE_CHUNK_SIZE, ": ", (sycl_pipeline_chunk_size != CCL_ENV_SIZET_NOT_SPECIFIED) ? std::to_string(sycl_pipeline_chunk_size) : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA, ": ", sycl_enable_pipeline_gpu_rdma);
    LOG_INFO_PROFILED(CCL_SYCL_ENABLE_DIRECT_GPU_RDMA, ": ", sycl_enable_direct_gpu_rdma);
    LOG_INFO_PROFILED(CCL_SYCL_SUB_COMMUICATOR, ": ", sycl_sub_communicator);
    LOG_INFO_PROFILED(CCL_SYCL_FORCE_PCIE, ": ", sycl_force_pcie);
#endif // CCL_ENABLE_SYCL

    LOG_INFO_PROFILED(CCL_ALLREDUCE_NREDUCE_BUFFERING, ": ", allreduce_nreduce_buffering);
    LOG_INFO_PROFILED(CCL_ALLREDUCE_NREDUCE_SEGMENT_SIZE,
                      ": ",
                      (allreduce_nreduce_segment_size != CCL_ENV_SIZET_NOT_SPECIFIED)
                          ? std::to_string(allreduce_nreduce_segment_size)
                          : CCL_ENV_STR_NOT_SPECIFIED);

    LOG_INFO_PROFILED(CCL_DTREE_PARTITION_COUNT,
                      ": ",
                      (dtree_partition_count != CCL_ENV_SIZET_NOT_SPECIFIED)
                          ? std::to_string(dtree_partition_count)
                          : CCL_ENV_STR_NOT_SPECIFIED);

    LOG_INFO_PROFILED(CCL_ALLREDUCE_2D_CHUNK_COUNT, ": ", allreduce_2d_chunk_count);
    LOG_INFO_PROFILED(CCL_ALLREDUCE_2D_MIN_CHUNK_SIZE, ": ", allreduce_2d_min_chunk_size);
    LOG_INFO_PROFILED(CCL_ALLREDUCE_2D_SWITCH_DIMS, ": ", allreduce_2d_switch_dims);

    LOG_INFO_PROFILED(CCL_CHECK_INPLACE_ALIASING, ": ", check_inplace_aliasing);

    LOG_INFO_PROFILED(CCL_ALLTOALL_SCATTER_MAX_OPS,
                      ": ",
                      (alltoall_scatter_max_ops != CCL_ENV_SIZET_NOT_SPECIFIED)
                          ? std::to_string(alltoall_scatter_max_ops)
                          : CCL_ENV_STR_NOT_SPECIFIED);

    LOG_INFO_PROFILED(CCL_BACKEND, ": ", str_by_enum(backend_names, backend));

    LOG_INFO_PROFILED(CCL_LOCAL_RANK,
                      ": ",
                      (local_rank != CCL_ENV_INT_NOT_SPECIFIED) ? std::to_string(local_rank)
                                                                : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_LOCAL_SIZE,
                      ": ",
                      (local_size != CCL_ENV_INT_NOT_SPECIFIED) ? std::to_string(local_size)
                                                                : CCL_ENV_STR_NOT_SPECIFIED);

    LOG_INFO_PROFILED(CCL_PROCESS_LAUNCHER, ": ", str_by_enum(process_launcher_names, process_launcher));

#ifdef CCL_ENABLE_MPI
    LOG_INFO_PROFILED(CCL_MPI_LIBRARY_PATH,
                      ": ",
                      (!mpi_lib_path.empty()) ? mpi_lib_path : CCL_ENV_STR_NOT_SPECIFIED);
#endif // CCL_ENABLE_MPI
    LOG_INFO_PROFILED(CCL_OFI_LIBRARY_PATH,
                      ": ",
                      (!ofi_lib_path.empty()) ? ofi_lib_path : CCL_ENV_STR_NOT_SPECIFIED);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_UMF)
    LOG_INFO_PROFILED(CCL_UMF_ENABLE, ": ", umf_enable);
    LOG_INFO_PROFILED(CCL_UMF_LIBRARY_PATH,
                      ": ",
                      (!umf_lib_path.empty()) ? umf_lib_path : CCL_ENV_STR_NOT_SPECIFIED);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE && CCL_ENABLE_UMF

#ifdef CCL_ENABLE_SYCL
    LOG_INFO_PROFILED(CCL_TOPO_ALGO, ": ", enable_topo_algo);
    LOG_INFO_PROFILED(CCL_TOPO_COLOR, ": ", str_by_enum(topo_color_names, topo_color));
    LOG_INFO_PROFILED(CCL_TOPO_P2P_ACCESS,
                      ": ",
                      (enable_p2p_access != CCL_ENV_INT_NOT_SPECIFIED) ? std::to_string(enable_p2p_access)
                                                                       : CCL_ENV_STR_NOT_SPECIFIED);

    LOG_INFO_PROFILED(CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK, ": ", enable_fabric_vertex_connection_check);
    LOG_INFO_PROFILED(CCL_TOPO_WA_FABRIC_VERTEX_CONNECTION_CHECK, ": ", enable_wa_fabric_vertex_connection_check);
    LOG_INFO_PROFILED(CCL_USE_MPI_BCAST_WA, ": ", use_mpi_bcast_wa);
    LOG_INFO_PROFILED(CCL_USE_ROOT_PRINT_WA, ": ", use_root_print_wa);
    LOG_INFO_PROFILED(
        CCL_KERNEL_PATH, ": ", (!kernel_path.empty()) ? kernel_path : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_KERNEL_DEBUG, ": ", kernel_debug);
    LOG_INFO_PROFILED(CCL_KERNEL_GROUP_SIZE, ": ", kernel_group_size);
    LOG_INFO_PROFILED(CCL_KERNEL_GROUP_COUNT,
                      ": ",
                      (kernel_group_count != CCL_ENV_SIZET_NOT_SPECIFIED)
                          ? std::to_string(kernel_group_count)
                          : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_KERNEL_MEM_ALIGN, ": ", kernel_mem_align);
    LOG_INFO_PROFILED(CCL_KERNEL_SYNC, ": ", enable_kernel_sync);
    LOG_INFO_PROFILED(CCL_KERNEL_1S_LEAD, ": ", kernel_1s_lead);
    LOG_INFO_PROFILED(CCL_KERNEL_1S_USE_COPY_OPS, ": ", enable_kernel_1s_copy_ops);
    LOG_INFO_PROFILED(CCL_KERNEL_1S_IPC_WA, ": ", enable_kernel_1s_ipc_wa);
    LOG_INFO_PROFILED(CCL_KERNEL_CLOSE_FD_WA, ": ", enable_close_fd_wa);

    LOG_INFO_PROFILED(CCL_SYCL_OUTPUT_EVENT, ": ", enable_sycl_output_event);
    LOG_INFO_PROFILED(CCL_BARRIER_SYNC, ": ", sync_barrier);
    LOG_INFO_PROFILED(CCL_ZE_DEPS_SYNC, ": ", sync_deps);
    LOG_INFO_PROFILED(CCL_USE_HMEM, ": ", use_hmem);

    LOG_INFO_PROFILED(CCL_ZE_BARRIER, ": ", enable_ze_barrier);
    LOG_INFO_PROFILED(CCL_ZE_BIDIR_ALGO, ": ", enable_ze_bidir_algo);
    LOG_INFO_PROFILED(CCL_ZE_CACHE, ": ", enable_ze_cache);
    LOG_INFO_PROFILED(CCL_ZE_DEVICE_CACHE_EVICT_SMALLEST, ": ", ze_device_cache_evict_smallest);
    LOG_INFO_PROFILED(CCL_ZE_DEVICE_CACHE_UPPER_LIMIT, ": ", ze_device_cache_upper_limit);
    LOG_INFO_PROFILED(CCL_ZE_DEVICE_CACHE_NUM_BLOCKS_IN_CHUNK, ": ", ze_device_cache_num_blocks_in_chunk);
    LOG_INFO_PROFILED(CCL_ZE_DEVICE_CACHE_POLICY, ": ", str_by_enum(ccl::ze::device_cache_policy_names, ze_device_cache_policy));
    LOG_INFO_PROFILED(CCL_ZE_PTR_REGISTER_THRESHOLD, ": ", ze_pointer_registration_threshold);
    LOG_INFO_PROFILED(CCL_ZE_CACHE_OPEN_IPC_HANDLES, ": ", enable_ze_cache_open_ipc_handles);
    LOG_INFO_PROFILED(CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD, ": ", ze_cache_open_ipc_handles_threshold);
    LOG_INFO_PROFILED(CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD, ": ", ze_cache_get_ipc_handles_threshold);
    LOG_INFO_PROFILED(CCL_ZE_CACHE_GET_IPC_HANDLES, ": ", enable_ze_cache_get_ipc_handles);
    LOG_INFO_PROFILED(CCL_ZE_SINGLE_LIST, ": ", enable_ze_single_list);
    LOG_INFO_PROFILED(CCL_ZE_DISABLE_FAMILY_CHECK, ": ", disable_ze_family_check);
    LOG_INFO_PROFILED(CCL_ZE_DISABLE_PORT_CHECK, ": ", disable_ze_port_check);
    LOG_INFO_PROFILED(CCL_ZE_ENABLE_OVERSUBSCRIPTION_FALLBACK, ": ", ze_enable_oversubscription_fallback);
    LOG_INFO_PROFILED(CCL_ZE_ENABLE_OVERSUBSCRIPTION_THROW, ": ", ze_enable_oversubscription_throw);
    LOG_INFO_PROFILED(CCL_ZE_SERIALIZE, ": ", ze_serialize_mode);
    LOG_INFO_PROFILED(CCL_ZE_COPY_ENGINE, ": ", str_by_enum(ccl::ze::copy_engine_names, ze_copy_engine));
    LOG_INFO_PROFILED(CCL_ZE_H2D_COPY_ENGINE,
                      ": ",
                      str_by_enum(ccl::ze::h2d_copy_engine_names, ze_h2d_copy_engine));
    LOG_INFO_PROFILED(CCL_ZE_D2D_COPY_ENGINE,
                      ": ",
                      str_by_enum(ccl::ze::d2d_copy_engine_names, ze_d2d_copy_engine));
    LOG_INFO_PROFILED(CCL_ZE_MAX_COMPUTE_QUEUES,
                      ": ",
                      (ze_max_compute_queues != CCL_ENV_SIZET_NOT_SPECIFIED)
                          ? std::to_string(ze_max_compute_queues)
                          : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ZE_MAX_COPY_QUEUES,
                      ": ",
                      (ze_max_copy_queues != CCL_ENV_SIZET_NOT_SPECIFIED)
                          ? std::to_string(ze_max_copy_queues)
                          : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ZE_ENABLE_CCS_FALLBACK_FOR_COPY, ": ", ze_enable_ccs_fallback_for_copy);
    LOG_INFO_PROFILED(CCL_ZE_LIST_DUMP, ": ", enable_ze_list_dump);
    LOG_INFO_PROFILED(CCL_ZE_QUEUE_INDEX_OFFSET, ": ", ze_queue_index_offset);
    LOG_INFO_PROFILED(CCL_ZE_CLOSE_IPC_WA, ": ", ze_close_ipc_wa);
    LOG_INFO_PROFILED(CCL_ZE_LIBRARY_PATH,
                      ": ",
                      (!ze_lib_path.empty()) ? ze_lib_path : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO_PROFILED(CCL_ZE_ENABLE, ": ", ze_enable);
    LOG_INFO_PROFILED(CCL_ZE_FINI_WA, ": ", ze_fini_wa);
    LOG_INFO_PROFILED(CCL_ZE_MULTI_WORKERS, ": ", ze_multi_workers);
    LOG_INFO_PROFILED(CCL_ZE_AUTO_TUNE_PORTS, ": ", enable_ze_auto_tune_ports);
    LOG_INFO_PROFILED(CCL_ZE_IPC_EXCHANGE, ": ", str_by_enum(ze::ipc_exchange_names, ze_ipc_exchange));
    LOG_INFO_PROFILED(CCL_ZE_DRM_BDF_SUPPORT, ": ", ze_drm_bdf_support);
    LOG_INFO_PROFILED(CCL_ZE_PT2PT_READ, ": ", ze_pt2pt_read);
    LOG_INFO_PROFILED(CCL_ZE_TYPE2_TUNE_PORTS, ": ", str_by_enum(type2_tune_mode_names, type2_mode));
    LOG_INFO_PROFILED(CCL_DRMFD_DEV_RENDER_DIR_PATH, ": ", drmfd_dev_render_dir_path);
    LOG_INFO_PROFILED(CCL_DRMFD_DEV_RENDER_SUFFIX, ": ", drmfd_dev_render_suffix);
#endif // CCL_ENABLE_SYCL
    LOG_INFO_PROFILED(CCL_IPC_ALLGATHERV_WA, ": ", ipc_allgatherv_wa);

#ifdef CCL_ENABLE_PMIX
    LOG_INFO_PROFILED(CCL_PMIX_LIBRARY_PATH,
                      ": ",
                      (!pmix_lib_path.empty()) ? pmix_lib_path : CCL_ENV_STR_NOT_SPECIFIED);
#endif // CCL_ENABLE_PMIX

#ifdef CCL_ENABLE_ITT
    LOG_INFO_PROFILED(CCL_ITT_LEVEL, ": ", itt_level);
#endif // CCL_ENABLE_ITT
    LOG_INFO_PROFILED(CCL_DEBUG_TIMESTAMPS_LEVEL, ": ", debug_timestamps_level);

    LOG_DEBUG(CCL_PROFILING_ENABLE, ": ", enable_profiling);

    LOG_INFO_PROFILED(CCL_BF16, ": ", str_by_enum(bf16_impl_names, bf16_impl_type));
    LOG_INFO_PROFILED(CCL_FP16, ": ", str_by_enum(fp16_impl_names, fp16_impl_type));

    char* ccl_root = getenv("CCL_ROOT");
    LOG_INFO_PROFILED("CCL_ROOT: ", (ccl_root) ? ccl_root : CCL_ENV_STR_NOT_SPECIFIED);

    char* impi_root = getenv("I_MPI_ROOT");
    LOG_INFO_PROFILED("I_MPI_ROOT: ", (impi_root) ? impi_root : CCL_ENV_STR_NOT_SPECIFIED);

    char* fi_provider_path = getenv("FI_PROVIDER_PATH");
    LOG_INFO_PROFILED("FI_PROVIDER_PATH: ",
                      (fi_provider_path) ? fi_provider_path : CCL_ENV_STR_NOT_SPECIFIED);

    char* fi_provider = getenv("FI_PROVIDER");
    LOG_INFO_PROFILED("FI_PROVIDER: ", (fi_provider) ? fi_provider : CCL_ENV_STR_NOT_SPECIFIED);

    global_data.algorithm_selector->print();
}

void env_data::set_internal_env() {
    auto attr = ccl_executor::generate_atl_attr(*this);
    atl_comm_manager::set_internal_env(attr);
    if (log_level >= ccl_log_level::info) {
        setenv("I_MPI_DEBUG", "4", 0);
    }
}

int env_data::env_2_worker_affinity_auto(int local_proc_idx, size_t workers_per_process) {
    char* available_cores = std::getenv(I_MPI_AVAILABLE_CORES_ENV);
    CCL_THROW_IF_NOT(available_cores && strlen(available_cores) != 0,
                     "auto pinning requires ",
                     I_MPI_AVAILABLE_CORES_ENV,
                     " env variable to be set");

    LOG_DEBUG("available_cores ", available_cores);

    std::vector<size_t> cores;
    ccl::utils::str_to_array(available_cores, std::string(I_MPI_AVAILABLE_CORES_DELIMS), cores);

    CCL_THROW_IF_NOT(workers_per_process <= cores.size(),
                     "failed to implicitly set workers affinity, "
                     "the number of workers (",
                     workers_per_process,
                     ") exceeds the number of available cores per process (",
                     cores.size(),
                     "), consider increasing the number of cores per process ",
                     "or explicitly setting of workers affinity using ",
                     CCL_WORKER_AFFINITY);

    if ((workers_per_process == cores.size()) && worker_offload) {
        LOG_WARN("the number of workers (",
                 workers_per_process,
                 ") matches the number of available cores per process,"
                 " this may lead to contention between workers and"
                 " application threads");
        if (!std::getenv(CCL_WORKER_OFFLOAD)) {
            worker_offload = 0;
            LOG_WARN("workers are disabled,"
                     " to forcibly enable them set ",
                     CCL_WORKER_OFFLOAD,
                     "=1");
        }
        else {
            LOG_WARN("consider increasing the number of cores per process",
                     " or disabling workers using ",
                     CCL_WORKER_OFFLOAD,
                     "=0");
        }
    }

    size_t worker_cores_start = cores.size() - workers_per_process;
    for (size_t idx = 0; idx < workers_per_process; ++idx) {
        worker_affinity[local_proc_idx * workers_per_process + idx] =
            cores[worker_cores_start + idx];
    }
    return 1;
}

int env_data::parse_number(const std::string& number_str, size_t& result) {
    char* end_ptr;
    const char* number_str_ptr = number_str.c_str();

    errno = 0;
    auto core_id = std::strtol(number_str_ptr, &end_ptr, 10);

    if ((errno == ERANGE && (core_id == LONG_MAX || core_id == LONG_MIN)) ||
        (errno != 0 && core_id == 0)) {
        LOG_ERROR("core id value is invalid in string: ", number_str);
        return 0;
    }
    if (end_ptr == number_str_ptr) {
        LOG_ERROR("no digits were found in string: ", number_str);
        return 0;
    }
    if (core_id < 0) {
        LOG_ERROR("core id cannot be less than zero but got ", core_id, " in string: ", number_str);
        return 0;
    }
    result = core_id;
    return 1;
}

int env_data::parse_affinity(const std::string& input,
                             std::vector<ssize_t>& output,
                             size_t expected_output_size) {
    size_t idx;
    char* range_str;

    /* create copy of input string because it will be modified in strsep */
    std::string input_copy(input.c_str());
    char* input_str = (char*)input_copy.c_str();

    output.clear();

    bool list_too_long = false;

    while (input_str) {
        range_str = strsep(&input_str, ",");
        if (!range_str) {
            break;
        }

        auto range = ccl::utils::tokenize<std::vector<std::string>>(std::string(range_str), '-');

        if ((range.size() != 2) && (range.size() != 1)) {
            LOG_ERROR(
                "unexpected format in input: ",
                input,
                ", specify range values using <first_value>-<last_value> or single value using <value>");
            return 0;
        }

        if (range.size() == 1) {
            /* to unify logic below */
            range.push_back(*range.begin());
        }

        CCL_ASSERT(range.size() == 2, "unexpected number of values in range");

        size_t first_value, last_value;
        if (!parse_number(range[0], first_value) || !parse_number(range[1], last_value)) {
            return 0;
        }

        if (first_value > last_value) {
            LOG_ERROR("unexpected first and last values in range: ",
                      range_str,
                      ", first value should be less or equal to last value");
            return 0;
        }

        for (idx = first_value; idx <= last_value; idx++) {
            if (output.size() < expected_output_size) {
                output.push_back(idx);
            }
            else {
                list_too_long = true;
                input_str = nullptr;
                break;
            }
        }
    }

    if (output.size() < expected_output_size) {
        LOG_ERROR("unexpected number of values in input: ",
                  input,
                  ", expected at least ",
                  expected_output_size,
                  " values");
        return 0;
    }

    if (list_too_long) {
        LOG_WARN("unexpected number of values in input: ",
                 input,
                 ", expected ",
                 expected_output_size,
                 " values; input truncated to match the expected size`");
    }

    return 1;
}

int env_data::env_2_worker_affinity(int local_proc_idx, int local_proc_count) {
    CCL_THROW_IF_NOT(local_proc_count > 0);

    size_t idx;
    char* env_to_parse = getenv(CCL_WORKER_AFFINITY);
    long int temp_system_core_count;
    size_t system_core_count;
    size_t affinity_size = local_proc_count * worker_count;

    if (!env_to_parse || (strlen(env_to_parse) == 0) || (strcmp(env_to_parse, "auto") == 0)) {
        worker_affinity.assign(affinity_size, CCL_UNDEFINED_CPU_ID);
        if (std::getenv(I_MPI_AVAILABLE_CORES_ENV)) {
            /* generate auto affinity based on IMPI process pinning */
            return env_2_worker_affinity_auto(local_proc_idx, worker_count);
        }
        else {
            /* generate auto affinity as last N cores */
            temp_system_core_count = sysconf(_SC_NPROCESSORS_ONLN);
            // throw an error on sysconf failure (-1) or if core_count is invalid
            CCL_THROW_IF_NOT(temp_system_core_count > 0,
                             "system_core_count is incorrect: ",
                             temp_system_core_count);
            system_core_count = static_cast<size_t>(temp_system_core_count);
            for (idx = 0; idx < affinity_size; idx++) {
                if (idx < system_core_count) {
                    worker_affinity[idx] = system_core_count - idx - 1;
                }
                else {
                    worker_affinity[idx] = worker_affinity[idx % system_core_count];
                }
            }
            return 1;
        }
    }

    CCL_THROW_IF_NOT(parse_affinity(env_to_parse, worker_affinity, affinity_size),
                     "failed to parse worker affinity");
    worker_affinity_set = true;

    return 1;
}

int env_data::env_2_worker_mem_affinity(int local_proc_count) {
    CCL_THROW_IF_NOT(worker_affinity.size() > 0);
    CCL_THROW_IF_NOT(local_proc_count > 0);

    size_t idx;
    char* env_to_parse = getenv(CCL_WORKER_MEM_AFFINITY);
    size_t affinity_size = local_proc_count * worker_count;
    CCL_THROW_IF_NOT(affinity_size <= worker_affinity.size());

    if (!env_to_parse || (strlen(env_to_parse) == 0) || (strcmp(env_to_parse, "auto") == 0)) {
        worker_mem_affinity.assign(affinity_size, CCL_UNDEFINED_NUMA_NODE);
        /* generate list of default numa nodes, local wrt worker cores */
        for (idx = 0; idx < affinity_size; idx++) {
            worker_mem_affinity[idx] =
                ccl::global_data::get().hwloc_wrapper->get_numa_node_by_cpu(worker_affinity[idx]);
        }
        return 1;
    }

    CCL_THROW_IF_NOT(parse_affinity(env_to_parse, worker_mem_affinity, affinity_size),
                     "failed to parse worker memory affinity");

    return 1;
}

} // namespace ccl
