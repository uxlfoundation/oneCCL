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
#include "coll/algorithms/utils/sycl_selection.hpp"

bool can_use_sycl_kernels(const ccl_selector_param& param) {
// TODO: mitigate overhead added by can_use_sycl_kernels
#ifdef CCL_ENABLE_SYCL
    RETURN_FALSE_IF(!param.comm->get_env()->get_enable_topo_algo(), "topo (sycl) algo is disabled");
#else // CCL_ENABLE_SYCL
    return false;
#endif // CCL_ENABLE_SYCL
    auto supported_colls = { ccl_coll_allgather, ccl_coll_allgatherv, ccl_coll_alltoall,
                             ccl_coll_allreduce, ccl_coll_broadcast,  ccl_coll_reduce_scatter,
                             ccl_coll_send,      ccl_coll_recv };
    RETURN_FALSE_IF(!checkers::is_coll_supported(supported_colls, param.ctype),
                    "coll is not supported");

    // these fields are not expected to be set for sycl kernels
    CCL_THROW_IF_NOT(!param.is_vector_buf, "unexpected is_vector_buf value");
    CCL_THROW_IF_NOT(!param.is_sycl_buf, "unexpected is_sycl_buf value");
    CCL_THROW_IF_NOT((param.ctype == ccl_coll_send || param.ctype == ccl_coll_recv) ||
                         (param.peer_rank == CCL_INVALID_PEER_RANK_IDX),
                     "unexpected peer_rank value");
    CCL_THROW_IF_NOT(!param.is_scaleout, "unexpected is_scaleout value");

    // TODO: it is incorrect and should be revisited
    if (param.comm->is_multi_thread_instance() == true) {
        return true;
    }

    size_t local_proc_count = param.comm->size();
    if (param.comm->is_multi_thread_instance() != true) {
        local_proc_count = ccl::global_data::get().get_local_proc_count();
    }

    LOG_DEBUG("coll ",
              ccl_coll_type_to_str(param.ctype),
              ", local_proc_count ",
              local_proc_count,
              ", comm ",
              param.comm->to_string());

    if (param.comm->get_node_comm()->size() % 2 == 1) {
        // if a tile has a proc, the other tile on 2-tile systems should have it as well
        // case when subsequent cards use a single tile but other cards use two tiles
        // e. g. 2 2 2 2 1 1
        // is not possible due to how procs are allocated.
        // odd count of ranks on a node should be on a single plane
        RETURN_FALSE_IF(param.comm->get_even_comm()->size() != param.comm->get_node_comm()->size(),
                        "ranks on a node with odd count not on a single plane");
    }

    RETURN_FALSE_IF(!checkers::is_gpu_stream(param), "non-gpu stream is not supported");
    RETURN_FALSE_IF(!checkers::is_device_buf(param), "non-device buffers is not supported");
    RETURN_FALSE_IF(!checkers::is_l0_backend(param), "non-l0 backend is not supported");

    // Removed duplicate check for backend_mode::native
    RETURN_FALSE_IF(ccl::global_data::env().backend != backend_mode::native,
                    "stub backend is not supported");

    RETURN_FALSE_IF(ccl::global_data::env().worker_count != 1, "unsupported count of workers");
#ifdef CCL_ENABLE_SYCL
    RETURN_FALSE_IF(param.comm->get_pair_comm()->size() > 2,
                    "unsupported pair_comm size: ",
                    param.comm->get_pair_comm()->size());
    // ARC GPUs are exception
    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_all_vertices_connected() &&
                        !is_arc_card(ccl::ze::get_device_family(param.stream->get_ze_device())),
                    "no connection between vertices");
    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_same_ppn(),
                    "ppn is not the same among the nodes");
    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_same_domains(),
                    "processes are not properly distributed among domains");

    const ccl::topo_manager& topo_manager = param.comm->get_topo_manager();
    bool is_single_node = topo_manager.is_single_node;
    bool is_oversubscription = topo_manager.has_oversubscription();

    // For recv and send, allow uint8 in addition to the other types.
    bool is_dtype_supported =
        (param.dtype.idx() == ccl::datatype::float16 ||
         param.dtype.idx() == ccl::datatype::bfloat16 ||
         param.dtype.idx() == ccl::datatype::float32 ||
         param.dtype.idx() == ccl::datatype::float64 || param.dtype.idx() == ccl::datatype::int32 ||
         param.dtype.idx() == ccl::datatype::uint32 || param.dtype.idx() == ccl::datatype::int64 ||
         param.dtype.idx() == ccl::datatype::uint64 ||
         (param.dtype.idx() == ccl::datatype::uint8 &&
          (param.ctype == ccl_coll_recv || param.ctype == ccl_coll_send)));

    // Common conditions for all collective operations
    RETURN_FALSE_IF(!ccl::global_data::env().enable_sycl_kernels, "SYCL kernels are not enabled");
    RETURN_FALSE_IF(!param.stream->get_native_stream().is_in_order(), "Stream is not in order");
    RETURN_FALSE_IF(!is_dtype_supported, "Data type is not supported");
    RETURN_FALSE_IF(is_oversubscription, "Oversubscription is not allowed");

    if (param.ctype != ccl_coll_allreduce && param.ctype != ccl_coll_allgatherv) {
        RETURN_FALSE_IF(!param.comm->get_topo_manager().has_p2p_access(),
                        "no p2p access between devices");
    }

    // Conditions specific to allreduce
    if (param.ctype == ccl_coll_allreduce) {
        RETURN_FALSE_IF(!ccl::global_data::env().allreduce_algo_raw.empty() &&
                            ccl::global_data::env().allreduce_algo_raw != "topo",
                        "algo of coll: ",
                        ccl_coll_type_to_str(param.ctype),
                        " is specified explicitly as: ",
                        ccl::global_data::env().allreduce_algo_raw,
                        " not supported");
        RETURN_FALSE_IF(
            param.reduction != ccl::reduction::sum && param.reduction != ccl::reduction::avg,
            "Allreduce only supports sum/avg reductions");
    }

    // Conditions specific to allgather
    if (param.ctype == ccl_coll_allgather) {
        RETURN_FALSE_IF(!ccl::global_data::env().allgather_algo_raw.empty() &&
                            ccl::global_data::env().allgather_algo_raw != "topo",
                        "algo of coll: ",
                        ccl_coll_type_to_str(param.ctype),
                        " is specified explicitly as: ",
                        ccl::global_data::env().allgather_algo_raw,
                        " not supported");
    }

    // Conditions specific to allgatherv
    if (param.ctype == ccl_coll_allgatherv) {
        RETURN_FALSE_IF(!ccl::global_data::env().allgatherv_algo_raw.empty() &&
                            ccl::global_data::env().allgatherv_algo_raw != "topo",
                        "algo of coll: ",
                        ccl_coll_type_to_str(param.ctype),
                        " is specified explicitly as: ",
                        ccl::global_data::env().allgatherv_algo_raw,
                        " not supported");

        for (int i = 0; i < param.comm->size(); i++) {
            RETURN_FALSE_IF(
                param.recv_counts[i] != param.count,
                "Allgatherv Sycl kernel is called with non-equal receive counts, fallback to schedule-based implementation");
        }
    }

    // Conditions specific to both allgather/allgatherv
    if (param.ctype == ccl_coll_allgatherv || param.ctype == ccl_coll_allgather) {
        if (!is_single_node) {
            RETURN_FALSE_IF(
                ccl::global_data::env().atl_transport == ccl_atl_ofi,
                "SYCL based Allgather/Allgatherv in multiple node mode supports only MPI transport");

            ccl_comm* r2r_comm = param.comm->get_r2r_comm().get();

            RETURN_FALSE_IF(
                r2r_comm->size() > 8,
                "SYCL based Allgather/Allgatherv is not supported at the moment for the larger scale");
            // Since SYCL based Allgatherv supports only equal receive counts,
            // (send_count == recv_counts[i]) we can simplify the operation
            size_t scaleout_count = r2r_comm->size() * param.count;
            RETURN_FALSE_IF(scaleout_count * param.dtype.size() >
                                ccl::global_data::env().sycl_allgatherv_scaleout_threshold,
                            "The total amount of data for SYCL based Allgather/Allgatherv",
                            " exceeds the threshold for multiple node mode");
        }
    }

    // Conditions specific to alltoall
    if (param.ctype == ccl_coll_alltoall) {
        RETURN_FALSE_IF(!ccl::global_data::env().alltoall_algo_raw.empty() &&
                            ccl::global_data::env().alltoall_algo_raw != "topo",
                        "algo of coll: ",
                        ccl_coll_type_to_str(param.ctype),
                        " is specified explicitly as: ",
                        ccl::global_data::env().alltoall_algo_raw,
                        " not supported");
    }

    // Conditions specific to broadcast
    if (param.ctype == ccl_coll_broadcast) {
        RETURN_FALSE_IF(!ccl::global_data::env().broadcast_algo_raw.empty() &&
                            ccl::global_data::env().broadcast_algo_raw != "topo",
                        "algo of coll: ",
                        ccl_coll_type_to_str(param.ctype),
                        " is specified explicitly as: ",
                        ccl::global_data::env().broadcast_algo_raw,
                        " not supported");

        if (ccl::global_data::env().sycl_esimd) {
            LOG_DEBUG(
                "ESIMD kernels are not implemented for broadcast, so SYCL kernels path is selected");
        }
    }

    // Conditions specific to reduce_scatter
    if (param.ctype == ccl_coll_reduce_scatter) {
        RETURN_FALSE_IF(!ccl::global_data::env().reduce_scatter_algo_raw.empty() &&
                            ccl::global_data::env().reduce_scatter_algo_raw != "topo",
                        "algo of coll: ",
                        ccl_coll_type_to_str(param.ctype),
                        " is specified explicitly as: ",
                        ccl::global_data::env().reduce_scatter_algo_raw,
                        " not supported");
        RETURN_FALSE_IF(
            param.reduction != ccl::reduction::sum && param.reduction != ccl::reduction::avg,
            "Reduce_scatter only supports sum/avg reductions");
    }

    if (!ccl::global_data::env().disable_ze_port_check) {
        RETURN_FALSE_IF(!checkers::is_single_card(param) && topo_manager.has_failed_ports(),
                        "failed fabric ports");
    }

    if (!ccl::global_data::env().disable_ze_family_check) {
        RETURN_FALSE_IF(checkers::is_family1_card(param) && !checkers::is_single_card(param),
                        "multi-card ",
                        ccl_coll_type_to_str(param.ctype),
                        " is not supported for family1");
    }

    if (param.comm->get_node_comm()->size() != local_proc_count) {
        CCL_ASSERT(param.comm->get_node_comm()->size() < local_proc_count);
        RETURN_FALSE_IF(!ccl::global_data::env().sycl_sub_communicator,
                        "SYCL kernels are not enabled for sub-communicators");

        RETURN_FALSE_IF(ccl::global_data::env().sycl_esimd,
                        "SYCL ESIMD kernels are not enabled for sub-communicators");
    }

    if (checkers::is_unknown_device_family(param)) {
        LOG_WARN("Applying sycl-kernels, but device family is not recognized");
    }

#endif // CCL_ENABLE_SYCL
    if (param.ctype == ccl_coll_recv || param.ctype == ccl_coll_send) {
        RETURN_FALSE_IF(ccl::global_data::env().sycl_pt2pt_enable == 0,
                        "SYCL pt2pt kernels are not enabled");
        auto node_comm = param.comm->get_node_comm().get();
        bool peer_rank_in_node_comm = node_comm->try_get_rank_from_global(param.peer_rank);
        bool rank_in_node_comm = node_comm->try_get_rank_from_global(param.comm->rank());

        RETURN_FALSE_IF(!(rank_in_node_comm && peer_rank_in_node_comm),
                        "peer_rank must be on the same node as own rank is: comm_rank: ",
                        param.comm->rank(),
                        ", peer_rank: ",
                        param.peer_rank,
                        ", rank_in_node_comm: ",
                        rank_in_node_comm,
                        ", peer_rank_in_node_comm: ",
                        peer_rank_in_node_comm,
                        ", node_comm_size: ",
                        node_comm->size());

        if (ccl::global_data::env().recv_algo_raw.length() != 0 &&
            ccl::global_data::env().send_algo_raw.length() != 0) {
            auto recv_algo = ccl_algorithm_selector_helper<ccl_coll_recv_algo>::algo_from_str(
                ccl::global_data::env().recv_algo_raw);
            auto send_algo = ccl_algorithm_selector_helper<ccl_coll_send_algo>::algo_from_str(
                ccl::global_data::env().send_algo_raw);
            RETURN_FALSE_IF((recv_algo == ccl_coll_recv_topo) || (send_algo == ccl_coll_send_topo),
                            " pt2pt operations set by user: CCL_SEND=",
                            ccl::global_data::env().send_algo_raw,
                            ", CCL_RECV=",
                            ccl::global_data::env().recv_algo_raw);
            RETURN_FALSE_IF(
                (recv_algo == ccl_coll_recv_direct) || (send_algo == ccl_coll_send_direct),
                " pt2pt operations set by user: CCL_SEND=",
                ccl::global_data::env().send_algo_raw,
                ", CCL_RECV=",
                ccl::global_data::env().recv_algo_raw);
            RETURN_FALSE_IF(
                (recv_algo == ccl_coll_recv_offload) || (send_algo == ccl_coll_send_offload),
                " pt2pt operations set by user: CCL_SEND=",
                ccl::global_data::env().send_algo_raw,
                ", CCL_RECV=",
                ccl::global_data::env().recv_algo_raw);
        }
    }
    LOG_DEBUG("selected algo: coll ", ccl_coll_type_to_str(param.ctype), ", algo ", "topo sycl");

    return true;
}

size_t allreduce_select_chunk_size(allreduce_scaleout_algo algo, size_t size, size_t comm_size) {
    // read defaults and user input
    size_t max_pipeline_chunk_size = ccl::global_data::env().sycl_max_pipeline_chunk_size;
    ssize_t env_pipeline_chunk_size = ccl::global_data::env().sycl_pipeline_chunk_size;
    size_t auto_pipeline_chunk_size = 2 * 1024 * 1024;
    // respect user input
    if (env_pipeline_chunk_size != CCL_ENV_SIZET_NOT_SPECIFIED) {
        return std::min((size_t)env_pipeline_chunk_size, max_pipeline_chunk_size);
    }
    // auto tuning, based on the latest perf measurements
    switch (algo) {
        case allreduce_scaleout_algo::rabenseifner:
            if (size >= 1024L * 1024 * 1024)
                auto_pipeline_chunk_size = 32 * 1024 * 1024;
            else if (size >= 256L * 1024 * 1024)
                auto_pipeline_chunk_size = 16 * 1024 * 1024;
            else
                auto_pipeline_chunk_size = 8 * 1024 * 1024;
            break;
        case allreduce_scaleout_algo::ring:
            if (size >= 1024L * 1024 * 1024)
                auto_pipeline_chunk_size = 32 * 1024 * 1024;
            else if (size >= 256L * 1024 * 1024)
                auto_pipeline_chunk_size = 16 * 1024 * 1024;
            else
                auto_pipeline_chunk_size = comm_size <= 8 ? 4 * 1024 * 1024 : 8 * 1024 * 1024;
            break;
        case allreduce_scaleout_algo::direct:
            LOG_WARN("direct alogrithm is not supporting pipeline chunk size tuning");
            break;
    }
    // error protection
    return std::min(auto_pipeline_chunk_size, max_pipeline_chunk_size);
}

static sycl_allreduce_tune_attr allreduce_select_large_algorithm(size_t size, size_t comm_size) {
    if (is_pof2(comm_size)) {
        size_t raben_chunk_size =
            allreduce_select_chunk_size(allreduce_scaleout_algo::rabenseifner, size, comm_size);
        return { allreduce_scaleout_algo::rabenseifner, raben_chunk_size };
    }
    // TODO: do the performance analysis of larger scale ring problems
    if (comm_size <= 16) {
        size_t ring_chunk_size =
            allreduce_select_chunk_size(allreduce_scaleout_algo::ring, size, comm_size);
        return { allreduce_scaleout_algo::ring, ring_chunk_size };
    }
    // no other choices left
    // TODO: for OFI case, we shouldn't select direct, but fallback
    // to the scheduler. Currently, in case of OFI transport,
    // we fallback to the scheduler at the very beginning of scale-out func.
    return { allreduce_scaleout_algo::direct };
}

// TODO: collect larger scale to enable >64 nodes selection
static sycl_allreduce_tune_attr allreduce_auto_select_tune_attr(size_t size,
                                                                size_t comm_size,
                                                                ccl_datatype ccl_dtype) {
    // small message size
    // direct is the best option based on the latest perf results
    // and message/comm size ranges
    if (ccl::global_data::env().atl_transport != ccl_atl_ofi) {
        if (ccl_dtype == ccl::datatype::float16) {
            // half precision data types: fp16
            if (comm_size <= 8 && size <= 512 * 1024) {
                return { allreduce_scaleout_algo::direct };
            }
            if (comm_size > 8 && size <= 1024 * 1024) {
                return { allreduce_scaleout_algo::direct };
            }
        }
        else if (ccl_dtype == ccl::datatype::bfloat16) {
            // half precision data types: bf16
            if (comm_size <= 8 && size <= 1024 * 1024) {
                return { allreduce_scaleout_algo::direct };
            }
            if (comm_size > 8 && size <= 4 * 1024 * 1024) {
                return { allreduce_scaleout_algo::direct };
            }
        }
        else {
            // full precision data types and other
            if (size <= 4 * 1024 * 1024) {
                return { allreduce_scaleout_algo::direct };
            }
        }
    }
    // medium/large message size
    if (comm_size <= 64) {
        return allreduce_select_large_algorithm(size, comm_size);
    }
    // default/other cases, larger scale
    if (ccl::global_data::env().atl_transport != ccl_atl_ofi) {
        return { allreduce_scaleout_algo::direct };
    }
    else {
        return allreduce_select_large_algorithm(size, comm_size);
    }
}

// TODO: work on selection framework MLSL-3253
sycl_allreduce_tune_attr allreduce_select_tune_attr(size_t size,
                                                    size_t comm_size,
                                                    ccl_datatype ccl_dtype) {
    if (ccl::global_data::env().sycl_allreduce_scaleout_algo == "auto") {
        return allreduce_auto_select_tune_attr(size, comm_size, ccl_dtype);
    }
    if (ccl::global_data::env().sycl_allreduce_scaleout_algo == "direct") {
        return { allreduce_scaleout_algo::direct };
    }
    if (ccl::global_data::env().sycl_allreduce_scaleout_algo == "rabenseifner") {
        size_t chunk_size =
            allreduce_select_chunk_size(allreduce_scaleout_algo::rabenseifner, size, comm_size);
        return { allreduce_scaleout_algo::rabenseifner, chunk_size };
    }
    if (ccl::global_data::env().sycl_allreduce_scaleout_algo == "ring") {
        size_t chunk_size =
            allreduce_select_chunk_size(allreduce_scaleout_algo::ring, size, comm_size);
        return { allreduce_scaleout_algo::ring, chunk_size };
    }
    CCL_THROW("unsupported selection");
}

// reduce-scatter
size_t reduce_scatter_select_chunk_size(reduce_scatter_scaleout_algo algo,
                                        size_t size,
                                        size_t comm_size) {
    // read defaults and user input
    size_t max_pipeline_chunk_size = ccl::global_data::env().sycl_max_pipeline_chunk_size;
    ssize_t env_pipeline_chunk_size = ccl::global_data::env().sycl_pipeline_chunk_size;
    size_t auto_pipeline_chunk_size = 2 * 1024 * 1024;
    // respect user input
    if (env_pipeline_chunk_size != CCL_ENV_SIZET_NOT_SPECIFIED) {
        return std::min((size_t)env_pipeline_chunk_size, max_pipeline_chunk_size);
    }
    // auto tuning, based on the latest perf measurements
    switch (algo) {
        case reduce_scatter_scaleout_algo::ring:
            if (comm_size >= 16)
                auto_pipeline_chunk_size = 8 * 1024 * 1024;
            else if (comm_size >= 4)
                auto_pipeline_chunk_size = 4 * 1024 * 1024;
            else
                auto_pipeline_chunk_size = 2 * 1024 * 1024;
            break;
        case reduce_scatter_scaleout_algo::direct:
            LOG_WARN("direct alogrithm is not supporting pipeline chunk size tuning");
            break;
    }
    // error protection
    return std::min(auto_pipeline_chunk_size, max_pipeline_chunk_size);
}

static sycl_reduce_scatter_tune_attr reduce_scatter_select_large_algorithm(size_t size,
                                                                           size_t comm_size) {
    // TODO: do the performance analysis of larger scale ring problems
    if (comm_size <= 16) {
        size_t ring_chunk_size =
            reduce_scatter_select_chunk_size(reduce_scatter_scaleout_algo::ring, size, comm_size);
        return { reduce_scatter_scaleout_algo::ring, ring_chunk_size };
    }
    // no other choices left
    // TODO: for OFI case, we shouldn't select direct, but fallback
    // to the scheduler. Currently, in case of OFI transport,
    // we fallback to the scheduler at the very beginning of scale-out func.
    return { reduce_scatter_scaleout_algo::direct };
}

static sycl_reduce_scatter_tune_attr reduce_scatter_auto_select_tune_attr(size_t size,
                                                                          size_t comm_size,
                                                                          ccl_datatype ccl_dtype) {
    // small message size
    // direct is the best option based on the latest perf results
    // and message/comm size ranges
    if (ccl::global_data::env().atl_transport != ccl_atl_ofi) {
        if (ccl_dtype == ccl::datatype::float16) {
            // half precision data types: fp16
            if (comm_size <= 8 && size <= 1024 * 1024) {
                return { reduce_scatter_scaleout_algo::direct };
            }
            if (comm_size > 8 && size <= 1024 * 1024) {
                return { reduce_scatter_scaleout_algo::direct };
            }
        }
        else if (ccl_dtype == ccl::datatype::bfloat16) {
            // half precision data types: bf16
            if (comm_size <= 8 && size <= 1024 * 1024) {
                return { reduce_scatter_scaleout_algo::direct };
            }
            if (comm_size > 8 && size <= 1 * 1024 * 1024) {
                return { reduce_scatter_scaleout_algo::direct };
            }
        }
        else {
            // full precision data types and other
            if (size <= 1 * 1024 * 1024) {
                return { reduce_scatter_scaleout_algo::direct };
            }
        }
    }
    // medium/large message size
    if (comm_size <= 64) {
        return reduce_scatter_select_large_algorithm(size, comm_size);
    }
    // default/other cases, larger scale
    if (ccl::global_data::env().atl_transport != ccl_atl_ofi) {
        return { reduce_scatter_scaleout_algo::direct };
    }
    else {
        return reduce_scatter_select_large_algorithm(size, comm_size);
    }
}

sycl_reduce_scatter_tune_attr reduce_scatter_select_tune_attr(size_t size,
                                                              size_t comm_size,
                                                              ccl_datatype ccl_dtype) {
    if (ccl::global_data::env().sycl_reduce_scatter_scaleout_algo == "auto") {
        return reduce_scatter_auto_select_tune_attr(size, comm_size, ccl_dtype);
    }
    if (ccl::global_data::env().sycl_reduce_scatter_scaleout_algo == "direct") {
        return { reduce_scatter_scaleout_algo::direct };
    }
    if (ccl::global_data::env().sycl_reduce_scatter_scaleout_algo == "ring") {
        size_t chunk_size =
            reduce_scatter_select_chunk_size(reduce_scatter_scaleout_algo::ring, size, comm_size);
        return { reduce_scatter_scaleout_algo::ring, chunk_size };
    }
    CCL_THROW("unsupported reduce-scatter algo selection");
}

// allgatherv
static sycl_allgatherv_tune_attr allgatherv_auto_select_tune_attr(size_t size,
                                                                  size_t comm_size,
                                                                  ccl_datatype ccl_dtype) {
    if (ccl::global_data::env().atl_transport != ccl_atl_mpi) {
        return { allgatherv_scaleout_algo::ring };
    }

    // experimental values, should be reviewed later
    if (comm_size <= 4 && size < 262144 || comm_size <= 16 && size < 131072) {
        return { allgatherv_scaleout_algo::direct };
    }
    else {
        size_t chunk_size =
            allgatherv_select_chunk_size(allgatherv_scaleout_algo::ring, size, comm_size);
        return { allgatherv_scaleout_algo::ring, chunk_size };
    }
}

sycl_allgatherv_tune_attr allgatherv_select_tune_attr(size_t size,
                                                      size_t comm_size,
                                                      ccl_datatype ccl_dtype) {
    if (ccl::global_data::env().sycl_allgatherv_scaleout_algo == "auto") {
        return allgatherv_auto_select_tune_attr(size, comm_size, ccl_dtype);
    }
    if (ccl::global_data::env().sycl_allgatherv_scaleout_algo == "direct") {
        return { allgatherv_scaleout_algo::direct };
    }
    if (ccl::global_data::env().sycl_allgatherv_scaleout_algo == "ring") {
        size_t chunk_size =
            allgatherv_select_chunk_size(allgatherv_scaleout_algo::ring, size, comm_size);
        return { allgatherv_scaleout_algo::ring, chunk_size };
    }
    CCL_THROW("unsupported allgatherv algo selection");
}

size_t allgatherv_select_chunk_size(allgatherv_scaleout_algo algo, size_t size, size_t comm_size) {
    // read defaults and user input
    size_t max_pipeline_chunk_size = ccl::global_data::env().sycl_max_pipeline_chunk_size;
    ssize_t env_pipeline_chunk_size = ccl::global_data::env().sycl_pipeline_chunk_size;
    // respect user input
    if (env_pipeline_chunk_size != CCL_ENV_SIZET_NOT_SPECIFIED) {
        return std::min((size_t)env_pipeline_chunk_size, max_pipeline_chunk_size);
    }
    size_t auto_pipeline_chunk_size = 2 * 1024 * 1024;
    switch (algo) {
        case allgatherv_scaleout_algo::ring:
            if (size >= 128 * 1024 * 1024)
                auto_pipeline_chunk_size = 64 * 1024 * 1024;
            else if (size >= 16 * 1024 * 1024)
                auto_pipeline_chunk_size = 8 * 1024 * 1024;
            else
                auto_pipeline_chunk_size = comm_size <= 8 ? 4 * 1024 * 1024 : 8 * 1024 * 1024;
            break;
        case allgatherv_scaleout_algo::direct:
            LOG_WARN("allgatherv direct alogrithm is not supporting pipeline chunk size tuning");
            break;
    }
    // error protection
    return std::min(auto_pipeline_chunk_size, max_pipeline_chunk_size);
}
