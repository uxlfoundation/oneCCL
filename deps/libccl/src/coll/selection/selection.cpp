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

#ifdef CCL_ENABLE_MPI
#include "atl/mpi/atl_mpi_ctx.hpp"
#endif
#include "coll/selection/selection.hpp"
#include "coll/group/group.hpp"
#include "comm/comm.hpp"
#include "common/global/global.hpp"

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/utils/sycl_utils.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

std::string to_string(const ccl_selector_param& param) {
    std::stringstream ss;

    ss << "{ "
       << "coll: " << ccl_coll_type_to_str(param.ctype) << ", count: " << param.count
       << ", dt: " << ccl::global_data::get().dtypes->name(param.dtype);

    if (param.comm) {
        ss << ", comm: { rank: " << param.comm->rank() << ", size: " << param.comm->size();
        if (param.ctype == ccl_coll_recv || param.ctype == ccl_coll_send) {
            ss << ", peer_rank: " << param.peer_rank;
        }
        ss << " }";
    }

    if (param.stream) {
        ss << ", stream: " << param.stream->to_string();
    }

    if (param.buf) {
        ss << ", buf: " << param.buf;
    }

    if (param.is_vector_buf) {
        ss << ", vector_buf";
    }

#ifdef CCL_ENABLE_SYCL
    if (param.is_sycl_buf) {
        ss << ", sycl_buf";
    }
#endif // CCL_ENABLE_SYCL

    if (param.hint_algo.has_value()) {
        ss << ", hint_algo: " << param.hint_algo.value;
    }

    ss << " }";

    return ss.str();
}

bool ccl_is_direct_algo(const ccl_selector_param& param) {
    bool res = false;

    auto& selector = ccl::global_data::get().algorithm_selector;

    if (param.ctype == ccl_coll_allgather) {
        res = (selector->get<ccl_coll_allgather>(param) == ccl_coll_allgather_direct);
    }
    if (param.ctype == ccl_coll_allgatherv) {
        res = (selector->get<ccl_coll_allgatherv>(param) == ccl_coll_allgatherv_direct);
    }
    else if (param.ctype == ccl_coll_allreduce) {
        res = (selector->get<ccl_coll_allreduce>(param) == ccl_coll_allreduce_direct);
    }
    else if (param.ctype == ccl_coll_alltoall) {
        res = (selector->get<ccl_coll_alltoall>(param) == ccl_coll_alltoall_direct);
    }
    else if (param.ctype == ccl_coll_alltoallv) {
        res = (selector->get<ccl_coll_alltoallv>(param) == ccl_coll_alltoallv_direct);
    }
    else if (param.ctype == ccl_coll_barrier) {
        res = (selector->get<ccl_coll_barrier>(param) == ccl_coll_barrier_direct);
    }
    else if (param.ctype == ccl_coll_bcast) {
        res = (selector->get<ccl_coll_bcast>(param) == ccl_coll_bcast_direct);
    }
    else if (param.ctype == ccl_coll_broadcast) {
        res = (selector->get<ccl_coll_broadcast>(param) == ccl_coll_broadcast_direct);
    }
    else if (param.ctype == ccl_coll_recv) {
        res = (selector->get<ccl_coll_recv>(param) == ccl_coll_recv_direct);
    }
    else if (param.ctype == ccl_coll_reduce) {
        res = (selector->get<ccl_coll_reduce>(param) == ccl_coll_reduce_direct);
    }
    else if (param.ctype == ccl_coll_reduce_scatter) {
        res = (selector->get<ccl_coll_reduce_scatter>(param) == ccl_coll_reduce_scatter_direct);
    }
    else if (param.ctype == ccl_coll_send) {
        res = (selector->get<ccl_coll_send>(param) == ccl_coll_send_direct);
    }

    return res;
}

bool ccl_is_offload_pt2pt_algo(const ccl_selector_param& param) {
    bool res = false;

    auto& selector = ccl::global_data::get().algorithm_selector;

    if (param.ctype == ccl_coll_recv) {
        res = (selector->get<ccl_coll_recv>(param) == ccl_coll_recv_offload);
    }
    else if (param.ctype == ccl_coll_send) {
        res = (selector->get<ccl_coll_send>(param) == ccl_coll_send_offload);
    }

    return res;
}

namespace checkers {

bool is_unknown_device_family(const ccl_selector_param& param) {
    if (param.stream) {
        return param.stream->get_device_family() == ccl::device_family::unknown;
    }
    return false;
}

bool is_family1_card(const ccl_selector_param& param) {
    if (param.stream) {
        return param.stream->get_device_family() == ccl::device_family::family1;
    }
    return false;
}

bool is_coll_supported(std::initializer_list<ccl_coll_type> colls, ccl_coll_type value) {
    return std::find(colls.begin(), colls.end(), value) != colls.end();
}

bool is_sycl_buf(const ccl_selector_param& param) {
#ifdef CCL_ENABLE_SYCL
    return param.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    return false;
}

bool is_device_buf(const ccl_selector_param& param) {
#ifdef CCL_ENABLE_SYCL
    if (param.buf && param.stream) {
        auto ctx = param.stream->get_native_stream().get_context();
        return sycl::get_pointer_type(param.buf, ctx) == sycl::usm::alloc::device;
    }
#endif // CCL_ENABLE_SYCL
    return true;
}

bool is_l0_backend(const ccl_selector_param& param) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (param.stream) {
        return param.stream->get_backend() == ccl::utils::get_level_zero_backend();
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    return false;
}

bool is_gpu_stream(const ccl_selector_param& param) {
    if (param.stream) {
        return param.stream->is_gpu();
    }
    return false;
}

bool is_single_node(const ccl_selector_param& param) {
    const ccl_comm* node_comm = param.comm->get_node_comm().get();
    return param.comm->size() == node_comm->size();
}

bool is_single_card(const ccl_selector_param& param) {
    return param.comm->get_topo_manager().is_single_card;
}

} // namespace checkers

static bool ccl_is_device_side_algo(ccl_coll_algo algo, const ccl_selector_param& param) {
    CCL_THROW_IF_NOT(algo.has_value(), "empty algo value");

    if (param.ctype == ccl_coll_allgather) {
        return algo.allgather == ccl_coll_allgather_topo;
    }
    if (param.ctype == ccl_coll_allgatherv) {
        return algo.allgatherv == ccl_coll_allgatherv_topo;
    }
    else if (param.ctype == ccl_coll_allreduce) {
        return algo.allreduce == ccl_coll_allreduce_topo;
    }
    else if (param.ctype == ccl_coll_alltoall) {
        return algo.alltoall == ccl_coll_alltoall_topo;
    }
    else if (param.ctype == ccl_coll_alltoallv) {
        return algo.alltoallv == ccl_coll_alltoallv_topo;
    }
    else if (param.ctype == ccl_coll_bcast) {
        return algo.bcast == ccl_coll_bcast_topo;
    }
    else if (param.ctype == ccl_coll_broadcast) {
        return algo.broadcast == ccl_coll_broadcast_topo;
    }
    else if (param.ctype == ccl_coll_recv) {
        return algo.recv == ccl_coll_recv_topo;
    }
    else if (param.ctype == ccl_coll_reduce) {
        return algo.reduce == ccl_coll_reduce_topo;
    }
    else if (param.ctype == ccl_coll_reduce_scatter) {
        // limitation: topo algorithm is disabled for reduce_scatter with unidirectional algo
#ifdef CCL_ENABLE_SYCL
        return algo.reduce_scatter == ccl_coll_reduce_scatter_topo &&
               ccl::global_data::env().enable_ze_bidir_algo;
#else // CCL_ENABLE_SYCL
        return algo.reduce_scatter == ccl_coll_reduce_scatter_topo;
#endif // CCL_ENABLE_SYCL
    }
    else if (param.ctype == ccl_coll_send) {
        return algo.send == ccl_coll_send_topo;
    }

    return false;
}

bool ccl_is_device_side_algo(const ccl_selector_param& param) {
#ifndef CCL_ENABLE_SYCL
    return false;
#endif // CCL_ENABLE_SYCL

    auto supported_colls = { ccl_coll_allgather,      ccl_coll_allgatherv, ccl_coll_allreduce,
                             ccl_coll_alltoall,       ccl_coll_alltoallv,  ccl_coll_bcast,
                             ccl_coll_broadcast,      ccl_coll_recv,       ccl_coll_reduce,
                             ccl_coll_reduce_scatter, ccl_coll_send };
    RETURN_FALSE_IF(!checkers::is_coll_supported(supported_colls, param.ctype),
                    "coll ",
                    ccl_coll_type_to_str(param.ctype),
                    " is not supported");

    ccl_coll_algo algo{};
    auto& selector = ccl::global_data::get().algorithm_selector;

    if (param.ctype == ccl_coll_allgather) {
        algo.allgather = selector->get<ccl_coll_allgather>(param);
    }
    if (param.ctype == ccl_coll_allgatherv) {
        algo.allgatherv = selector->get<ccl_coll_allgatherv>(param);
    }
    else if (param.ctype == ccl_coll_allreduce) {
        algo.allreduce = selector->get<ccl_coll_allreduce>(param);
    }
    else if (param.ctype == ccl_coll_alltoall) {
        algo.alltoall = selector->get<ccl_coll_alltoall>(param);
    }
    else if (param.ctype == ccl_coll_alltoallv) {
        algo.alltoallv = selector->get<ccl_coll_alltoallv>(param);
    }
    else if (param.ctype == ccl_coll_bcast) {
        algo.bcast = selector->get<ccl_coll_bcast>(param);
    }
    else if (param.ctype == ccl_coll_broadcast) {
        algo.broadcast = selector->get<ccl_coll_broadcast>(param);
    }
    else if (param.ctype == ccl_coll_recv) {
        algo.recv = selector->get<ccl_coll_recv>(param);
    }
    else if (param.ctype == ccl_coll_reduce) {
        algo.reduce = selector->get<ccl_coll_reduce>(param);
    }
    else if (param.ctype == ccl_coll_reduce_scatter) {
        algo.reduce_scatter = selector->get<ccl_coll_reduce_scatter>(param);
    }
    else if (param.ctype == ccl_coll_send) {
        algo.send = selector->get<ccl_coll_send>(param);
    }

    return ccl_is_device_side_algo(algo, param);
}

bool ccl_can_use_topo_algo(const ccl_selector_param& param) {
#ifdef CCL_ENABLE_SYCL
    RETURN_FALSE_IF(!param.comm->get_env()->get_enable_topo_algo(), "topo algo is disabled");
#else // CCL_ENABLE_SYCL
    return false;
#endif // CCL_ENABLE_SYCL

    auto supported_colls = { ccl_coll_allgather,      ccl_coll_allgatherv, ccl_coll_allreduce,
                             ccl_coll_alltoall,       ccl_coll_alltoallv,  ccl_coll_bcast,
                             ccl_coll_broadcast,      ccl_coll_recv,       ccl_coll_reduce,
                             ccl_coll_reduce_scatter, ccl_coll_send };
    RETURN_FALSE_IF(!checkers::is_coll_supported(supported_colls, param.ctype),
                    "coll is not supported");

    size_t local_proc_count = ccl::global_data::get().get_local_proc_count();
    int comm_size = param.comm->size();

    LOG_DEBUG("coll ",
              ccl_coll_type_to_str(param.ctype),
              ", local_proc_count ",
              local_proc_count,
              ", comm ",
              param.comm->to_string());

    RETURN_FALSE_IF(!checkers::is_gpu_stream(param), "non-gpu stream is not supported");
    RETURN_FALSE_IF(checkers::is_sycl_buf(param), "sycl buffer is not supported");
    RETURN_FALSE_IF(!checkers::is_device_buf(param), "non-device buffers is not supported");
    RETURN_FALSE_IF(!checkers::is_l0_backend(param), "non-l0 backend is not supported");

    RETURN_FALSE_IF(ccl::global_data::env().enable_fusion, "fusion is not supported");
    RETURN_FALSE_IF(ccl::global_data::env().enable_unordered_coll,
                    "unordered coll is not supported");
    RETURN_FALSE_IF(ccl::global_data::env().priority_mode != ccl_priority_none, "wrong priority");
    RETURN_FALSE_IF(ccl::global_data::env().worker_count != 1
#ifdef CCL_ENABLE_SYCL
                        && !ccl::global_data::env().ze_multi_workers
#endif
                    ,
                    "unsupported count of workers");

    // this check is for multi-level scale-out for device buffers case:
    // we can't use topo algorithm without sub-communicators
    RETURN_FALSE_IF(!param.comm->get_even_comm().get(), "sub-communicators are not available");

#ifdef CCL_ENABLE_SYCL
    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_p2p_access(),
                    "no p2p access between devices");
    // WA
    if (!ccl::global_data::env().enable_wa_fabric_vertex_connection_check) {
        RETURN_FALSE_IF(!param.comm->get_topo_manager().has_all_vertices_connected(),
                        "no connection between vertices");
    }

    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_same_ppn(),
                    "ppn is not the same among the nodes");

    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_same_domains(),
                    "processes are not properly distributed among domains");

    if (comm_size > 2 && !(param.ctype == ccl_coll_recv || param.ctype == ccl_coll_send)) {
        if (ccl::global_data::env().ze_enable_oversubscription_throw &&
            param.ctype != ccl_coll_allreduce && param.ctype != ccl_coll_bcast) {
            CCL_THROW_IF_NOT(
                !param.comm->get_topo_manager().has_oversubscription(),
                "oversubscription case is detected: \n OneCCL expects max one rank per device, "
                " but count of unique devices: ",
                param.comm->get_topo_manager().get_unique_device_uuids_count(),
                ", comm_size: ",
                comm_size,
                "\n specify comm_size to: ",
                param.comm->get_topo_manager().get_unique_device_uuids_count());
        }
        else {
            if (ccl::global_data::env().ze_enable_oversubscription_fallback) {
                RETURN_FALSE_IF(param.comm->get_topo_manager().has_oversubscription(),
                                "oversubscription case: one rank per device is only supported");
            }
        }
    }

    RETURN_FALSE_IF(!ccl::global_data::env().enable_ze_bidir_algo &&
                        (param.ctype == ccl_coll_alltoall || param.ctype == ccl_coll_alltoallv),
                    "alltoall(v) is supported with bidir only");

    if (!ccl::global_data::env().disable_ze_port_check) {
        RETURN_FALSE_IF(
            !checkers::is_single_card(param) && param.comm->get_topo_manager().has_failed_ports(),
            "failed fabric ports");
    }

    if (!ccl::global_data::env().disable_ze_family_check) {
        RETURN_FALSE_IF(checkers::is_family1_card(param) && !checkers::is_single_card(param),
                        "multi-card ",
                        ccl_coll_type_to_str(param.ctype),
                        " is not supported for family1");

        RETURN_FALSE_IF(
            checkers::is_family1_card(param) && ccl::global_data::env().enable_ze_bidir_algo,
            "bidir ",
            ccl_coll_type_to_str(param.ctype),
            " is not supported for family1");
    }

    if (checkers::is_unknown_device_family(param)) {
        LOG_WARN("Applying topo algorithm, but device family is not recognized");
#ifndef CCL_BF16_GPU_TRUNCATE
        if (param.dtype.idx() == ccl::datatype::bfloat16 &&
            (param.ctype == ccl_coll_allreduce || param.ctype == ccl_coll_reduce ||
             param.ctype == ccl_coll_reduce_scatter)) {
            LOG_WARN("Applying topo algorithm, but bfloat16 reduction may not be"
                     "supported for unknown device family");
        }
#endif // !CCL_BF16_GPU_TRUNCATE
    }
#endif // CCL_ENABLE_SYCL

    RETURN_FALSE_IF((param.ctype == ccl_coll_bcast || param.ctype == ccl_coll_broadcast) &&
                        !checkers::is_single_node(param),
                    "multi-node for ",
                    ccl_coll_type_to_str(param.ctype),
                    " is not supported");

    RETURN_FALSE_IF((comm_size % 2 != 0 && comm_size != 1), "odd comm_size is not supported");

    const int node_comm_size = param.comm->get_node_comm().get()->size();

    RETURN_FALSE_IF((node_comm_size % 2 != 0 && comm_size != 1),
                    "odd node_comm_size is not supported");

    RETURN_FALSE_IF(!checkers::is_single_card(param) && !checkers::is_single_node(param) &&
                        (local_proc_count % 2 != 0),
                    "odd proc count per node is not supported");

    RETURN_FALSE_IF((param.ctype == ccl_coll_reduce) &&
                        (param.count < size_t(param.comm->size())) && (comm_size != 1),
                    "reduce with count < comm_size not supported");

    if (param.ctype == ccl_coll_recv || param.ctype == ccl_coll_send) {
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
            RETURN_FALSE_IF(
                (recv_algo == ccl_coll_recv_direct) || (send_algo == ccl_coll_send_direct),
                " pt2pt operations algo must be the same: CCL_SEND=",
                ccl::global_data::env().send_algo_raw,
                ", CCL_RECV=",
                ccl::global_data::env().recv_algo_raw);
        }

#ifdef CCL_ENABLE_SYCL
        auto rank_color = param.comm->get_topo_manager().get_intra_card_color(param.comm->rank());
        auto peer_rank_color = param.comm->get_topo_manager().get_intra_card_color(param.peer_rank);

        if (rank_color == peer_rank_color && !ccl::global_data::env().ze_pt2pt_read) {
            // TODO: store this and then load back
            ccl::global_data::env().ze_pt2pt_read = 1;
            LOG_DEBUG("pt2pt: force read algo for within card execution case:"
                      " { color: ",
                      rank_color,
                      ", rank: ",
                      node_comm->rank(),
                      " },"
                      " { peer_color: ",
                      peer_rank_color,
                      ", peer_rank: ",
                      param.peer_rank,
                      " }");
        }
#endif // CCL_ENABLE_SYCL

        RETURN_FALSE_IF(
            group_impl::is_group_active && ccl::global_data::env().atl_transport == ccl_atl_ofi,
            "ofi transport is not supported for group API");
    }

    return true;
}

bool ccl_can_use_datatype(ccl_coll_algo algo, const ccl_selector_param& param) {
    if (param.dtype.idx() != ccl::datatype::float16) {
        return true;
    }

    bool can_use = true;

    // algorithms running on device side support fp16
    // so we don't need to require their support on the host
    bool device_side_algo = ccl_is_device_side_algo(algo, param);
    if (device_side_algo) {
        return true;
    }

    if (param.dtype == ccl::datatype::float16) {
        bool fp16_hw_support =
            ccl::global_data::env().fp16_impl_type != ccl_fp16_no_hardware_support;
        bool fp16_compiler_support =
            ccl::global_data::env().fp16_impl_type != ccl_fp16_no_compiler_support;

        can_use = fp16_hw_support && fp16_compiler_support;

        if (!can_use) {
            LOG_DEBUG("FP16 datatype is requested for ",
                      ccl_coll_type_to_str(param.ctype),
                      " running on CPU but not fully supported: hw: ",
                      fp16_hw_support,
                      " compiler: ",
                      fp16_compiler_support);
        }
    }

    return can_use;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
void set_offload_pt2pt_mpi_env() {
#if CCL_ENABLE_MPI
    auto lib_attr = atl_mpi_ctx::get_lib_attr();
    if (lib_attr.type == atl_mpi_ctx::ATL_MPI_LIB_IMPI && lib_attr.hmem == 1) {
        if (ccl::global_data::get().ze_data != nullptr) {
            setenv("I_MPI_OFFLOAD", "2", 0);
            LOG_DEBUG("IMPI case: I_MPI_OFFLOAD is set");
        }
        else {
            // Even though the library was compiled using SYCL, we could not initialze
            // level-zero API. This means that we should not enable GPU offload
            // inside IMPI.
            setenv("I_MPI_OFFLOAD", "0", 0);
            LOG_DEBUG("IMPI case: I_MPI_OFFLOAD is disenabled, could not initialize level-zero");
        }
    }
    else if (lib_attr.type == atl_mpi_ctx::ATL_MPI_LIB_MPICH && lib_attr.hmem == 1) {
        setenv("MPIR_CVAR_CH4_OFI_ENABLE_GPU_PIPELINE", "1", 0);
        setenv("MPIR_CVAR_CH4_OFI_GPU_PIPELINE_THRESHOLD", "2048", 0);
        setenv("MPIR_CVAR_CH4_OFI_GPU_PIPELINE_MAX_NUM_BUFFERS", "4", 0);
        setenv("MPIR_CVAR_CH4_OFI_GPU_PIPELINE_NUM_BUFFERS_PER_CHUNK", "4", 0);
        setenv("MPIR_CVAR_CH4_OFI_GPU_PIPELINE_BUFFER_SZ", "4194304", 0);
        setenv("MPIR_CVAR_CH4_OFI_GPU_PIPELINE_H2D_ENGINE_TYPE", "1", 0);
        setenv("MPIR_CVAR_CH4_OFI_GPU_PIPELINE_D2H_ENGINE_TYPE", "1", 0);
        LOG_DEBUG("MPIR case: MPIR_CVAR_ENABLE_GPU is set in MPICH internally");
    }
#else
    LOG_WARN("set_offload_pt2pt_mpi_env called on build without MPI!");
#endif
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

bool use_pt2pt_offload_algo() {
    bool res = true;

    // TODO: deprecate offload for all pt2pt cases ?
    if (group_impl::is_group_active) {
        return false;
    }

    const char* env_value = std::getenv("PSM3_GPUDIRECT");
    if ((env_value == nullptr || std::strcmp(env_value, "0") == 0) &&
        ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        res = false;
    }
    return res;
}
