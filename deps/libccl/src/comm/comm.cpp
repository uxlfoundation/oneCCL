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

#include "atl/atl_base_comm.hpp"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.h"
#include "exec/exec.hpp"
#include "coll/coll.hpp"
#include "coll/attr/ccl_common_op_attrs.hpp"
#include "coll/group/group.hpp"
#include "comm/comm.hpp"
#include "comm/comm_id_allocator.hpp"
#include "comm/comm_impl.hpp"
#include "common/global/global.hpp"
#include "common/event/impls/host_event.hpp"
#include "common/request/request.hpp"
#include "sched/sched.hpp"
#include "common/utils/exchange_utils.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/kvs.hpp"
#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/ikvs_wrapper.h"
#include "kvs_impl.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

// comm_barrier() used in invoke_barrier() below; comm.hpp no longer
// pulls in sycl_barrier.hpp transitively, so include it here
// explicitly. The SYCL barrier primitives are only compiled when
// SYCL is enabled.
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "coll/algorithms/utils/sycl_barrier.hpp"
#endif

namespace ccl {
namespace v1 {

struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t& operator()(const Object& obj) {
        return obj.get_impl();
    }
};

}; // namespace v1
}; // namespace ccl

// Kernel name template for comm_barrier
class oneccl_invoke_barrier {};

void comm_barrier(const std::shared_ptr<ccl_comm> comm) {
    // based on atl invocation from allreduce_scaleout_sycl
    // call ccl::wrapper for MPI/OFI.
    int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
    atl_req_t req;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ATL_CALL_THROW_IF_ERROR(atl_comm->barrier(ep_idx, req));

    ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
    if (!req.is_completed) {
        // We do not want to call check() in a loop (because we would call MPI_Test repeatedly). Call MPI_Wait() instead.
        ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
    }
    else {
        // The operation was probably blocking, since it finished really quickly
    }
}

#ifdef CCL_ENABLE_SYCL
// invoke the global communication barrier kernel
sycl::event invoke_barrier(const std::shared_ptr<ccl_comm> comm,
                           sycl::queue q,
                           const std::vector<sycl::event>& dep_events,
                           bool use_cpu) {
    bool gpu_increment = use_recording_path(q);
    sycl::event e;
    if (use_cpu) {
        e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.host_task([comm]() {
                comm_barrier(comm);
            });
        });
    }
    else {
        ccl_comm_barrier_data barrier_data =
            gpu_increment ? comm->barrier_data() : comm->barrier_inc();
        e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.parallel_for<oneccl_invoke_barrier>(
                sycl::nd_range<1>(MAX_NODE_RANKS, MAX_NODE_RANKS),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    comm_barrier(barrier_data, it, true, gpu_increment);
                });
        });
    }
    return e;
}
#endif
// ccl_comm_env

ccl_comm_env::ccl_comm_env(std::shared_ptr<ccl::device> device) : device(device) {
#ifdef CCL_ENABLE_SYCL
    enable_topo_algo = ccl::global_data::env().enable_topo_algo;
    ze_copy_engine = ccl::global_data::env().ze_copy_engine;
    ze_h2d_copy_engine = ccl::global_data::env().ze_h2d_copy_engine;

    if (device &&
        (device.get()->get_native().get_backend() == ccl::utils::get_level_zero_backend())) {
        auto ze_device =
            sycl::get_native<ccl::utils::get_level_zero_backend()>(device.get()->get_native());
        CCL_THROW_IF_NOT(ze_device, "null ze device");

        if ((ccl::ze::get_device_family(ze_device) == ccl::device_family::unknown) ||
            (ccl::ze::get_device_family(ze_device) == ccl::device_family::family1)) {
            ze_copy_engine = ccl::ze::copy_engine_mode::none;
            ze_h2d_copy_engine = ccl::ze::h2d_copy_engine_mode::none;
        }
    }
    else {
        enable_topo_algo = 0;
        ze_copy_engine = ccl::ze::copy_engine_mode::none;
        ze_h2d_copy_engine = ccl::ze::h2d_copy_engine_mode::none;
    }
#endif // CCL_ENABLE_SYCL
}

std::string ccl_comm_env::to_string() const {
    std::stringstream ss;
    ss << "{";

#ifdef CCL_ENABLE_SYCL
    if (device) {
        ss << " enable_topo_algo: " << enable_topo_algo;
        ss << ", ze_copy_engine: " << ccl::ze::copy_engine_names[ze_copy_engine];
        ss << ", ze_h2d_copy_engine: " << ccl::ze::h2d_copy_engine_names[ze_h2d_copy_engine];
        ss << " ";
    }
#endif // CCL_ENABLE_SYCL

    ss << "}";

    return ss.str();
}

// ccl_internal_comm

ccl_internal_comm::ccl_internal_comm(int comm_id,
                                     int rank,
                                     int size,
                                     std::shared_ptr<atl_base_comm> comm)
        : m_dtree(size, rank)
#ifdef CCL_ENABLE_SYCL
          ,
          m_barrier_data(rank, size),
          m_flag_data(rank, size)
#endif // CCL_ENABLE_SYCL
{
    atl_comm = atl_comm_manager::create_with_id(comm, comm_id);
    reset(rank, size);

    if (comm_id == comm->get_comm_id()) {
        LOG_DEBUG("comm.id == explicit_id, reset comm.id ", comm_id);
        comm->reset_comm_id();
    }
}

void ccl_internal_comm::reset(int rank, int size) {
    m_rank = rank;
    m_size = size;
    m_pof2 = ccl::utils::pof2(m_size);
}

// ccl_comm

void ccl_comm::init(int comm_id,
                    const std::shared_ptr<atl_base_comm>& atl_comm,
                    bool share_resources,
                    bool is_sub_communicator) {
    if (group_impl::is_group_active) {
        LOG_WARN("Creating communicator inside group operation");
    }

    comm_rank = atl_comm->get_rank();
    comm_size = atl_comm->get_size();
    unique_comm_id = comm_id;

    next_sched_id_internal = atl_comm->tag_creator->get_max_sched_count() / 2;
    next_sched_id_external = 0;

    if (comm_rank >= comm_size || comm_size <= 0) {
        throw ccl::exception("incorrect rank or size when creating \
                             communicator: rank: " +
                             std::to_string(comm_rank) + ", size: " + std::to_string(comm_size));
    }

    comm_impl = std::unique_ptr<ccl_internal_comm>(
        new ccl_internal_comm(comm_id, comm_rank, comm_size, atl_comm));

    if (!share_resources) {
        allocate_resources();
    }

    if (!is_sub_communicator) {
        topo_manager.init(atl_comm, device_ptr, context_ptr);
        if (!comm_rank && device_ptr) {
            LOG_INFO("topo_manager:", topo_manager.to_string());
        }
        create_topo_subcomms(atl_comm);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        // init of fd manager is based on node comm,
        // it initializes for every creation of comm in multi comms case
        CCL_ASSERT(node_comm, "no node_comm");
        init_ipc_exchange_mode(node_comm);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    }
    else {
        local2global_map = atl_comm->get_rank2rank_map();
    }

    env = std::make_shared<ccl_comm_env>(device_ptr);

    if (comm_rank == 0) {
        LOG_DEBUG(to_string_ext());
    }

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (ccl::global_data::env().enable_sycl_kernels && device_ptr != NULL) {
        sycl::queue q(device_ptr->get_native()); // TODO check the context of this queue
        if (q.get_context().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ccl::stream op_stream = ccl::create_stream(q);
            ccl::impl_dispatch disp;
            ccl_stream* cclstream = get_stream_ptr(disp(op_stream));
            LOG_DEBUG("invoking multi-process path");
            coll_init(this, cclstream);
        }
    }
#endif
}

ccl_comm::ccl_comm(int comm_id,
                   std::shared_ptr<atl_base_comm> atl_comm,
                   bool share_resources,
                   bool is_sub_communicator,
                   bool is_Ext,
                   int size,
                   int rank,
                   int group_id) {
    if (is_Ext == true) {
        // prefix Ext means extension for multi threading
        initExt(size, rank, comm_id, std::move(atl_comm), share_resources, is_sub_communicator);
    }
    else {
        init(comm_id, std::move(atl_comm), share_resources, is_sub_communicator);
    }
}

ccl_comm::ccl_comm(std::shared_ptr<atl_base_comm> atl_comm,
                   bool share_resources,
                   bool is_sub_communicator)
        : ccl_comm(atl_comm->create_comm_id(), atl_comm, share_resources, is_sub_communicator) {}

ccl_comm::ccl_comm(device_t device,
                   context_t context,
                   std::shared_ptr<atl_base_comm> atl_comm,
                   bool is_Ext,
                   int size,
                   int rank,
                   int group_id)
        : device_ptr(std::make_shared<ccl::device>(device)),
          context_ptr(std::make_shared<ccl::context>(context)) {
    int id = 0;
    if (is_Ext == true) {
        initExt(size, rank, id, std::move(atl_comm), false, false, group_id);
    }
    else {
        id = atl_comm->create_comm_id();
        init(id, std::move(atl_comm));
    }
}

ccl_comm::ccl_comm(int size,
                   int rank,
                   ccl::shared_ptr_class<ikvs_wrapper> kvs,
                   ccl::ccl_comm_attr_impl& attr)
        : ccl_comm(atl_comm_manager::create(size, { rank }, std::move(kvs), attr)) {}

ccl_comm::ccl_comm(int size, ccl::shared_ptr_class<ikvs_wrapper> kvs, ccl::ccl_comm_attr_impl& attr)
        : ccl_comm(atl_comm_manager::create(size, { 0 }, std::move(kvs), attr)) {}

ccl_comm::ccl_comm() : ccl_comm(atl_comm_manager::create()) {}

ccl_comm::ccl_comm(ccl::ccl_comm_attr_impl& attr) : ccl_comm(atl_comm_manager::create(attr)) {}

ccl_comm::ccl_comm(const ccl_comm& src, int comm_id)
        : ccl_comm(comm_id, src.get_atl_comm(), true, true) {
    r2r_comm = src.r2r_comm;
    node_comm = src.node_comm;
    numa_comm = src.numa_comm;
    numa_r2r_comm = src.numa_r2r_comm;
    even_comm = src.even_comm;
    pair_comm = src.pair_comm;
}

std::shared_ptr<ikvs_wrapper> ccl_comm::get_kvs_wrapper(std::shared_ptr<ccl::kvs_interface> kvs) {
    ccl::shared_ptr_class<ikvs_wrapper> kvs_tmp;
    if (std::dynamic_pointer_cast<ccl::v1::kvs>(kvs) != nullptr) {
        kvs_tmp = ccl::get_kvs_impl_typed<ccl::native_kvs_impl>(
                      std::dynamic_pointer_cast<ccl::v1::kvs>(std::move(kvs)))
                      ->get();
    }
    else {
        kvs_tmp = std::shared_ptr<ikvs_wrapper>(new users_kvs(std::move(kvs)));
    }

    return kvs_tmp;
}

ccl_comm* ccl_comm::create(device_t device,
                           context_t context,
                           int size,
                           int rank,
                           ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                           ccl::ccl_comm_attr_impl& attr) {
    return new ccl_comm(
        device, context, atl_comm_manager::create(size, { rank }, get_kvs_wrapper(kvs), attr));
}

ccl_comm* ccl_comm::create(int size,
                           int rank,
                           ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                           ccl::ccl_comm_attr_impl& attr) {
    return new ccl_comm(size, rank, get_kvs_wrapper(kvs), attr);
}

ccl_comm* ccl_comm::create(int size,
                           ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                           ccl::ccl_comm_attr_impl& attr) {
    return new ccl_comm(size, get_kvs_wrapper(kvs), attr);
}
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
// Helper function to get NUMA node for local GPU
// Returns NUMA node ID, or -1 if cannot be determined
int ccl_comm::get_numa_node_for_gpu(std::shared_ptr<ccl_comm> node_comm,
                                    std::shared_ptr<ccl::device> device_ptr,
                                    std::shared_ptr<ccl::context> context_ptr) {
    if (!(device_ptr && context_ptr)) {
        return -1;
    }

    int my_numa_node = -1;

    auto& sycl_device = device_ptr->get_native();
    if (sycl_device.get_backend() == ccl::utils::get_level_zero_backend()) {
        ze_device_handle_t ze_device =
            sycl::get_native<ccl::utils::get_level_zero_backend()>(sycl_device);

#ifdef ZE_PCI_PROPERTIES_EXT_NAME
        auto& devices = ccl::global_data::get().ze_data->devices;
        for (const auto& dev_info : devices) {
            if (dev_info.device == ze_device) {
                if (ccl::global_data::get().hwloc_wrapper->is_initialized()) {
                    my_numa_node = ccl::global_data::get().hwloc_wrapper->get_numa_node_by_pci(
                        dev_info.pci.domain,
                        dev_info.pci.bus,
                        dev_info.pci.device,
                        dev_info.pci.function);

                    LOG_DEBUG("GPU NUMA node: ",
                              my_numa_node,
                              " (PCI ",
                              (int)dev_info.pci.domain,
                              ":",
                              (int)dev_info.pci.bus,
                              ":",
                              (int)dev_info.pci.device,
                              ".",
                              (int)dev_info.pci.function,
                              ")");
                }
                break;
            }
        }
#endif // ZE_PCI_PROPERTIES_EXT_NAME
    }

    return my_numa_node;
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

void ccl_comm::create_topo_subcomms(std::shared_ptr<atl_base_comm> atl_comm) {
    r2r_comm = std::shared_ptr<ccl_comm>(create_subcomm(atl_comm->get_r2r_color()));

    ccl_comm* node_comm_ptr = create_subcomm(topo_manager.get_host_idx());
    CCL_THROW_IF_NOT(node_comm_ptr, "Failed to create node communicator");
    node_comm = std::shared_ptr<ccl_comm>(node_comm_ptr);

    // Detect NUMA node for local GPU
    int my_numa_node = -1;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    my_numa_node = get_numa_node_for_gpu(node_comm, device_ptr, context_ptr);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    // Create numa_comm using host_idx * 1000 + numa_node as color
    // This mapping is used, because numa_color has to be independent
    // of number of ranks on a single machine. The numa_comm might be
    // a result of comm split with any number of GPUs assigned to any
    // numa, so we have to support configs where there's for example
    // one GPU on numa 0, and three GPUs on numa 1 etc.
    if (my_numa_node >= 0) {
        int numa_color = topo_manager.get_host_idx() * 1000 + my_numa_node;
        LOG_INFO("numa comm color: ",
                 numa_color,
                 " (host_idx=",
                 topo_manager.get_host_idx(),
                 ", numa_node=",
                 my_numa_node,
                 ")");
        numa_comm = std::shared_ptr<ccl_comm>(create_subcomm(numa_color));
        LOG_INFO("numa comm size: ", numa_comm->size());
    }
    else {
        // NUMA detection not available: numa_comm == node_comm
        numa_comm = std::shared_ptr<ccl_comm>(create_subcomm(topo_manager.get_host_idx()));
        LOG_INFO("NUMA detection unavailable, numa_comm same as node_comm, size: ",
                 numa_comm->size());
    }

    // Create numa_r2r_comm: connects ranks at the same position within their
    // NUMA node across all NUMA nodes and all hosts. E.g. position-0 ranks
    // from every (host, numa) group land in one communicator:
    //   {rank0/NUMA0/nodeA, rank0/NUMA1/nodeA, rank0/NUMA0/nodeB, ...}
    //
    // We allgather (host_idx, numa_node) from every rank, compute each
    // rank's position inside its (host, numa) group, and use position as
    // the r2r color.
    //
    // This mapping was added to support more complex rank placement, in
    // which the numa_comms are NOT symmetrical and number of GPUs on
    // each numa might be different on one machine and might be different
    // when from other different machines. For example, previously the code
    // was not working when host A would only have GPUs on numa 0 and host B
    // would have gpus on numa0 AND numa 1.
    if (my_numa_node >= 0) {
        struct numa_rank_info {
            int host_idx;
            int numa_node;
        };

        int world_size = atl_comm->get_size();
        int my_global_rank = atl_comm->get_rank();

        numa_rank_info my_info{ topo_manager.get_host_idx(), my_numa_node };
        std::vector<numa_rank_info> all_info(world_size);

        bool ok =
            ccl::utils::allgather(atl_comm, &my_info, all_info.data(), sizeof(numa_rank_info));
        CCL_THROW_IF_NOT(ok, "Failed to allgather NUMA rank info for numa_r2r_comm");

        // For each (host, numa) group, collect global ranks in order
        std::map<std::pair<int, int>, std::vector<int>> group_ranks;
        for (int r = 0; r < world_size; r++) {
            auto key = std::make_pair(all_info[r].host_idx, all_info[r].numa_node);
            group_ranks[key].push_back(r);
        }

        // Find my position within my (host, numa) group
        auto my_key = std::make_pair(topo_manager.get_host_idx(), my_numa_node);
        const auto& my_group = group_ranks[my_key];
        int my_position = -1;
        for (int i = 0; i < static_cast<int>(my_group.size()); i++) {
            if (my_group[i] == my_global_rank) {
                my_position = i;
                break;
            }
        }
        CCL_THROW_IF_NOT(my_position >= 0, "rank not found in its own NUMA group");

        int numa_r2r_color = my_position;
        LOG_INFO("numa r2r comm color: ",
                 numa_r2r_color,
                 " (numa_node=",
                 my_numa_node,
                 ", position=",
                 my_position,
                 ")");
        numa_r2r_comm = std::shared_ptr<ccl_comm>(create_subcomm(numa_r2r_color));
        LOG_INFO("numa r2r comm size: ", numa_r2r_comm->size());
    }
    else {
        numa_r2r_comm = std::shared_ptr<ccl_comm>(create_subcomm(atl_comm->get_r2r_color()));
        LOG_INFO("NUMA detection unavailable, numa_r2r_comm uses r2r_color, size: ",
                 numa_r2r_comm->size());
    }

    even_comm = std::shared_ptr<ccl_comm>(
        create_subcomm(topo_manager.get_inter_card_color(atl_comm->get_rank())));
    pair_comm = std::shared_ptr<ccl_comm>(create_subcomm(
        topo_manager.get_intra_card_color(atl_comm->get_rank()),
        topo_manager.get_inter_card_color(atl_comm->get_rank()) % topo_manager.max_ranks_per_card));
}

ccl_comm* ccl_comm::create_subcomm(int color, int key) const {
    std::shared_ptr<atl_base_comm> new_atl_comm = get_atl_comm()->comm_split(color, key);
    ccl_comm* comm = new ccl_comm(
        new_atl_comm->get_comm_id(), new_atl_comm, true /*share_resources*/, true /*subcomm*/);
    comm->set_parent_comm(const_cast<ccl_comm*>(this));
    LOG_DEBUG("new subcomm: color ", color, ", ", comm->to_string());
    return comm;
}

ccl_comm* ccl_comm::create_subcomm_split_independent(int color, int key) {
    std::shared_ptr<atl_base_comm> new_atl_comm = get_atl_comm()->comm_split(color, key);

    // Create communicator with is_sub_communicator=true initially so that init()
    // doesn't try to initialize topo_manager with nullptr device_ptr
    ccl_comm* comm = new ccl_comm(new_atl_comm->get_comm_id(),
                                  new_atl_comm,
                                  true /*share_resources*/,
                                  true /*is_sub_communicator*/);

    // Compute unique comm_id using split signature
    // The signature is based on parent comm_id, color, and the membership (global ranks)
    // Note: 'key' is NOT included because it only affects rank ordering, not membership
    // This ensures all ranks in the same sub-communicator get the same comm_id,
    // and different sub-communicators get different comm_ids.
    // Sort the ranks to ensure deterministic signature computation across all processes
    int parent_comm_id = get_atl_comm()->get_comm_id();
    int my_comm_id = comm->get_comm_id();
    std::vector<int> global_ranks = new_atl_comm->get_rank2rank_map();
    std::vector<int> sorted_ranks = global_ranks;
    std::sort(sorted_ranks.begin(), sorted_ranks.end());

    // Compute the split signature (without key - it only affects ordering, not membership)
    uint64_t split_unique_comm_id = ccl::comm_id_allocator::compute_split_signature(
        parent_comm_id, my_comm_id, color, sorted_ranks);
    // update unique comm id for ITT
    comm->update_unique_id(split_unique_comm_id);

    LOG_DEBUG("parent_comm_id=",
              parent_comm_id,
              ", comm_id=",
              my_comm_id,
              ", color=",
              color,
              ", key=",
              key,
              ", unique_comm_id=",
              split_unique_comm_id);

    // Copy device and context pointers from parent
    comm->device_ptr = this->device_ptr;
    comm->context_ptr = this->context_ptr;

    // Now initialize topo_manager and create topo sub-communicators for the split comm
    // These will be properly sized for the split group (not inherited from parent)
    comm->topo_manager.init(new_atl_comm, comm->device_ptr, comm->context_ptr);
    if (!comm->rank() && comm->device_ptr) {
        LOG_INFO("split comm topo_manager:", comm->topo_manager.to_string());
    }
    comm->create_topo_subcomms(new_atl_comm);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    // Initialize IPC exchange mode based on new node_comm
    if (comm->node_comm) {
        comm->init_ipc_exchange_mode(comm->node_comm);
    }

    // Initialize SYCL kernel buffers and do IPC exchange for the split communicator
    if (ccl::global_data::env().enable_sycl_kernels && comm->device_ptr != NULL) {
        sycl::queue q(comm->device_ptr->get_native());
        if (q.get_context().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ccl::stream op_stream = ccl::create_stream(q);
            ccl::impl_dispatch disp;
            ccl_stream* cclstream = get_stream_ptr(disp(op_stream));
            LOG_DEBUG("invoking coll_init for split comm");
            coll_init(comm, cclstream);
        }
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    // Inherit environment settings
    comm->env = this->env;

    LOG_DEBUG("Base rank: ",
              get_atl_comm()->get_rank(),
              ", Color: ",
              color,
              ", Old size: ",
              get_atl_comm()->get_size(),
              " -> New rank: ",
              new_atl_comm->get_rank(),
              ", Color: ",
              color,
              ", New size: ",
              new_atl_comm->get_size());
    LOG_DEBUG("new subcomm: color ", color, ", ", comm->to_string());
    return comm;
}

std::shared_ptr<ccl_comm> ccl_comm::clone_with_new_id(int comm_id) {
    return std::shared_ptr<ccl_comm>(new ccl_comm(*this, comm_id));
}

// NOTE: allocate_resources must be done on ccl_comm level
// if it's called on ccl_internal_comm level
// the ccl_comm object that we need won't be fully constructed
void ccl_comm::allocate_resources() {
    if (ccl::global_data::env().enable_unordered_coll) {
        comm_impl->unordered_coll_manager.reset(new ccl_unordered_coll_manager(*this));
    }
    ccl::global_data::env().print(rank(), true, enable_multi_thread_instance);
    ccl::global_data::env().print(rank(), false, enable_multi_thread_instance);
}

ccl::comm_interface_ptr ccl_comm::split(int color, int key, bool split_external_use) {
    // Check if color and key are within valid range to avoid overflow
    if (color < 0) {
        CCL_THROW(std::string(__FUNCTION__) +
                  " - Invalid 'color' value for communicator split: must be within 0 and " +
                  std::to_string(std::numeric_limits<int>::max()));
    }

    if (key < 0) {
        CCL_THROW(std::string(__FUNCTION__) +
                  " - Invalid 'key' value for communicator split: must be within 0 and " +
                  std::to_string(std::numeric_limits<int>::max()));
    }

    ccl_comm* new_comm = nullptr;
    if (split_external_use) {
        new_comm = create_subcomm_split_independent(color, key);
    }
    else {
        new_comm = create_subcomm(color, key);
    }

    return std::shared_ptr<ccl_comm>(new_comm);
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
void ccl_comm::init_ipc_exchange_mode(std::shared_ptr<ccl_comm> comm) {
    if (device_ptr && context_ptr) {
        LOG_DEBUG("initialize ipc_exchange_mode");
        // if we are using fabric handle, ipc_exchange_mode has to be set to none
        if (ccl::global_data::env().ze_ipc_mem_handle_type ==
            ccl::ze::ipc_mem_handle_type::fabric) {
            if (zeMemGetIpcHandleWithProperties == nullptr) {
                CCL_THROW(
                    "level-zero runtime does not support ipc handle with properties, aborting");
            }
            if (ccl::global_data::env().ze_ipc_exchange != ccl::ze::ipc_exchange_mode::none) {
                CCL_THROW("ipc exchange mode must set to 'none' if fabric handle type is used");
            }
        }
        // if pidfd is supported and env var is set, then use pidfd
        // else if drmfd is set explicitly or if pidfd is not supported - use drmfd
        // else otherwise use sockets mode
        if (ccl::ze::fd_manager::is_pidfd_supported() &&
            ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::pidfd) {
            LOG_DEBUG("pidfd exchange mode is verified successfully");
        }
#ifdef CCL_ENABLE_DRM
        else if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::drmfd ||
                 !ccl::ze::fd_manager::is_pidfd_supported()) {
            if (!ccl::ze::fd_manager::is_pidfd_supported()) {
                LOG_WARN("pidfd is not supported, fallbacks to drmfd exchange mode");
                ccl::global_data::env().ze_ipc_exchange = ccl::ze::ipc_exchange_mode::drmfd;
            }

            fd_manager = std::make_shared<ccl::ze::fd_manager>(comm->get_atl_comm());
            // update physical_idx for each logical device, by default it is invalid
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
            auto& devices = ccl::global_data::get().ze_data->devices;
            for (size_t idx = 0; idx < devices.size(); idx++) {
                devices[idx].physical_idx = ccl::ze::fd_manager::get_physical_device_idx(
                    fd_manager->get_physical_devices(), devices[idx].pci);
            }
#endif // ZE_PCI_PROPERTIES_EXT_NAME
            LOG_DEBUG("drmfd exchange mode is verified successfully");
        }
#endif // CCL_ENABLE_DRM
        else if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::none) {
            ccl::global_data::env().ze_ipc_exchange = ccl::ze::ipc_exchange_mode::none;
            LOG_DEBUG("auto ipc mode is selected");
        }
        else {
            ccl::global_data::env().ze_ipc_exchange = ccl::ze::ipc_exchange_mode::sockets;
            LOG_DEBUG("sockets mode is selected");
        }
    }
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

std::string ccl_comm::to_string() const {
    std::stringstream ss;
    ss << "{ rank: " << rank() << ", size: " << size() << ", id: " << id() << " }";
    return ss.str();
}

std::string ccl_comm::to_string_ext() const {
    std::stringstream ss;
    ss << "{\n";
    ss << "   " << to_string() << "\n";
    ss << "   r2r_comm: " << (r2r_comm ? r2r_comm->to_string() : "{}") << "\n";
    ss << "   node_comm: " << (node_comm ? node_comm->to_string() : "{}") << "\n";
    ss << "   numa_comm: " << (numa_comm ? numa_comm->to_string() : "{}") << "\n";
    ss << "   numa_r2r_comm: " << (numa_r2r_comm ? numa_r2r_comm->to_string() : "{}") << "\n";
    ss << "   even_comm: " << (even_comm ? even_comm->to_string() : "{}") << "\n";
    ss << "   pair_comm: " << (pair_comm ? pair_comm->to_string() : "{}") << "\n";
    ss << "   env: " << (env ? env->to_string() : "{}") << "\n";
    ss << "}";

    return ss.str();
}

int ccl_comm::get_global_rank(int rank) const {
    if (local2global_map.empty()) {
        // global comm and its copies do not have entries in the map
        return rank;
    }

    CCL_THROW_IF_NOT((int)local2global_map.size() > rank,
                     "no rank ",
                     rank,
                     " was found in comm ",
                     this,
                     ", id ",
                     id());
    int global_rank = local2global_map[rank];
    LOG_DEBUG("comm ", this, ", id ", id(), ", map rank ", rank, " to global ", global_rank);
    return global_rank;
}

int ccl_comm::get_rank_from_global(int global_rank) const {
    if (local2global_map.empty()) {
        // global comm and its copies do not have entries in the map
        return global_rank;
    }

    int rank = ccl_comm::invalid_rank;

    // TODO: Add reverse map to speed this up
    for (size_t i = 0; i < local2global_map.size(); ++i) {
        if (local2global_map[i] == global_rank) {
            rank = static_cast<int>(i);
            break;
        }
    }

    CCL_THROW_IF_NOT(rank != ccl_comm::invalid_rank, "can not find rank");

    return rank;
}

bool ccl_comm::try_get_rank_from_global(int global_rank) const {
    bool ret = false;
    if (local2global_map.empty()) {
        // global comm and its copies do not have entries in the map
        return ret;
    }

    for (size_t i = 0; i < local2global_map.size(); ++i) {
        if (local2global_map[i] == global_rank) {
            return true;
        }
    }

    return ret;
}

int ccl_comm::get_node_rank(int rank) const {
    if (this == get_node_comm().get()) {
        CCL_THROW("untested get_node_rank() on node_comm");
        // This is the node_comm, mapping is direct
        return rank;
    }

    // First, get global_rank from rank
    int global_rank = get_global_rank(rank);

    // Then, map global_rank to node_comm's rank
    return get_node_comm()->get_rank_from_global(global_rank);
}

ccl_sched_id_t ccl_comm::get_sched_id(bool use_internal_space, bool is_pt2pt) {
    std::shared_ptr<atl_base_comm> atl_comm = get_atl_comm();
    ccl_sched_id_t& next_sched_id =
        (use_internal_space) ? next_sched_id_internal : next_sched_id_external;

    ccl_sched_id_t max_sched_count = atl_comm->tag_creator->get_max_sched_count();

    ccl_sched_id_t first_sched_id =
        (use_internal_space) ? static_cast<ccl_sched_id_t>(0) : max_sched_count / 2;

    ccl_sched_id_t max_sched_id = (use_internal_space) ? max_sched_count / 2 : max_sched_count;

    ccl_sched_id_t id = next_sched_id;

    // is_pt2pt flag is required in the case
    // to avoid when send-recv communication between ranks
    // less comm_size, the ++next_sched_id op is skipped if
    // is_pt2pt = true
    if (!is_pt2pt) {
        ++next_sched_id;
    }

    if (next_sched_id == max_sched_id) {
        /* wrap the sched numbers around to the start */
        next_sched_id = first_sched_id;
    }

    LOG_DEBUG("sched_id ", id, ", comm_id ", this->id(), ", next sched_id ", next_sched_id);

    return id;
}

#ifdef CCL_ENABLE_SYCL
void* ccl_scaleout_host_bufs::get_scaleout_host_buf() {
    if (!host_bufs[index]) {
        CCL_THROW_IF_NOT(get_scaleout_host_buf_size() > 0,
                         "CCL_SCALEOUT_HOST_BUF_SIZE must be greater than zero");

        auto& env = ccl::global_data::env();
        auto& global_data = ccl::global_data::get();
        switch (env.sycl_scaleout_buf_alloc_mode) {
            case ccl::utils::alloc_mode::hwloc: {
                // fallback to memalign if worker_affinity is not set by user
                if (env.worker_affinity_set) {
                    auto worker_count = env.worker_count;
                    int numa_node_os_idx =
                        env.worker_mem_affinity[global_data.get_local_proc_idx() * worker_count];
                    host_bufs[index] = global_data.hwloc_wrapper->alloc_memory(
                        CCL_REG_MSG_ALIGNMENT, buf_size, numa_node_os_idx);
                    break;
                }
            }
            case ccl::utils::alloc_mode::memalign:
                // internally, CCL_MALLOC calls posix_memalign
                host_bufs[index] = CCL_MALLOC(buf_size, "scaleout_host_buf");
                break;
            case ccl::utils::alloc_mode::malloc: host_bufs[index] = malloc(buf_size); break;
            default: CCL_THROW("unexpected alloc_mode");
        }
        CCL_THROW_IF_NOT(host_bufs[index] != nullptr, "Cannot allocate host buffer");

        if (global_data.ze_data->external_pointer_registration_enabled) {
            global_data.ze_data->import_external_pointer(host_bufs[index], buf_size);
        }
    }

    auto old_index = index;
    index = (index + 1) % buf_count;
    return host_bufs[old_index];
}

void ccl_scaleout_host_bufs::put_scaleout_host_buf(const void* buf) {
    int old_index = (index + buf_count - 1) % buf_count;
    CCL_THROW_IF_NOT(host_bufs[old_index] == buf, "put_scaleout_host_buf in wrong order");
    index = old_index;
}

size_t ccl_scaleout_host_bufs::get_scaleout_host_buf_size() {
    if (buf_size == 0) {
        buf_size = ccl::global_data::env().sycl_scaleout_host_buf_size;
    }
    return buf_size;
}

ccl_scaleout_host_bufs::~ccl_scaleout_host_bufs() {
    try {
        for (int i = 0; i < buf_count; ++i) {
            if (host_bufs[i] != nullptr) {
                if (ccl::global_data::get().ze_data->external_pointer_registration_enabled) {
                    ccl::global_data::get().ze_data->release_imported_pointer(host_bufs[i]);
                }

                switch (ccl::global_data::env().sycl_scaleout_buf_alloc_mode) {
                    case ccl::utils::alloc_mode::hwloc:
                        if (ccl::global_data::env().worker_affinity_set) {
                            ccl::global_data::get().hwloc_wrapper->dealloc_memory(host_bufs[i]);
                            break;
                        }
                    case ccl::utils::alloc_mode::memalign: CCL_FREE(host_bufs[i]); break;
                    case ccl::utils::alloc_mode::malloc: free(host_bufs[i]); break;
                    default:
                        // destructors cannot throw exceptions
                        LOG_ERROR("unexpected alloc_mode");
                }
            }
        }
    }
    catch (const std::exception& e) {
        // printing warning since we are towards the end of execution
        LOG_WARN("exception caught in the destructor of scaleout host buffers: ", e.what());
    }
}

sycl::device& ccl_scaleout_device_bufs::get_device() const {
    static sycl::device device;
    return device;
}

void* ccl_scaleout_device_bufs::get_scaleout_device_buf(sycl::queue& q) {
    auto& device = get_device();
    if (!device_bufs[index]) {
        CCL_THROW_IF_NOT(get_scaleout_device_buf_size() > 0,
                         "CCL_SCALEOUT_HOST_BUF_SIZE must be greater than zero");

        device_bufs[index] = sycl::malloc_device(buf_size, q);
        CCL_THROW_IF_NOT(device_bufs[index], "malloc scaleout_device_buf failed");
        device = q.get_device();
    }
    // current assumption is that we expect same sycl::queue every time
    CCL_ASSERT(device == q.get_device());
    auto old_index = index;
    index = (index + 1) % buf_count;
    return device_bufs[old_index];
}

void ccl_scaleout_device_bufs::put_scaleout_device_buf(void* buf) {
    int old_index = (index - 1 + buf_count) % buf_count;
    CCL_THROW_IF_NOT(device_bufs[old_index] == buf, "put_scaleout_device_buf in wrong order");
    index = old_index;
}

size_t ccl_scaleout_device_bufs::get_scaleout_device_buf_size() {
    if (buf_size == 0) {
        buf_size = ccl::global_data::env().sycl_scaleout_device_buf_size;
    }
    return buf_size;
}

ccl_scaleout_device_bufs::~ccl_scaleout_device_bufs() {
    std::vector<void*> sycl_bufs{};

    for (int i = 0; i < buf_count; ++i) {
        if (device_bufs[i] != nullptr) {
            sycl_bufs.push_back(device_bufs[i]);
        }
    }

    if (sycl_bufs.empty()) {
        return;
    }

    auto& device = get_device();
    sycl::queue q(device);
    for (void* sycl_buf : sycl_bufs) {
        sycl::free(sycl_buf, q);
    }
}

ccl_scaleout_device_bufs::ccl_scaleout_device_bufs(const ccl_scaleout_device_bufs& other)
        : buf_size(other.buf_size),
          index(other.index) {
    auto& device = get_device();
    device = other.get_device();

    sycl::queue q(other.get_device());

    for (int i = 0; i < buf_count; ++i) {
        if (other.device_bufs[i] != nullptr) {
            device_bufs[i] = sycl::malloc_device(buf_size, q);
            q.memcpy(device_bufs[i], other.device_bufs[i], buf_size).wait();
        }
        else {
            device_bufs[i] = nullptr;
        }
    }
}

ccl_scaleout_device_bufs& ccl_scaleout_device_bufs::operator=(
    const ccl_scaleout_device_bufs& other) {
    if (this == &other) {
        return *this;
    }

    std::vector<void*> sycl_bufs{};
    for (int i = 0; i < buf_count; ++i) {
        if (device_bufs[i] != nullptr) {
            sycl_bufs.push_back(device_bufs[i]);
        }
    }

    auto& device = get_device();
    sycl::queue previous_device_queue(device);
    for (void* sycl_buf : sycl_bufs) {
        sycl::free(sycl_buf, previous_device_queue);
    }

    buf_size = other.buf_size;
    index = other.index;
    device = other.get_device();

    sycl::queue other_device_queue(device);

    for (int i = 0; i < buf_count; ++i) {
        if (other.device_bufs[i] != nullptr) {
            device_bufs[i] = sycl::malloc_device(buf_size, other_device_queue);
            other_device_queue.memcpy(device_bufs[i], other.device_bufs[i], buf_size).wait();
        }
        else {
            device_bufs[i] = nullptr;
        }
    }

    return *this;
}

void ccl_scaleout_pipeline_bufs::allocate_pipe_chunks(int num_bufs) {
    CCL_ASSERT(num_chunk_buffs == num_bufs);
    if (send_pipe_buffer || recv_pipe_buffer)
        return;

    auto& env = ccl::global_data::env();
    auto& global_data = ccl::global_data::get();
    size_t max_chunk_size = env.sycl_max_pipeline_chunk_size;
    switch (env.sycl_scaleout_buf_alloc_mode) {
        case ccl::utils::alloc_mode::hwloc: {
            // fallback to memalign if worker_affinity is not set by user
            if (env.worker_affinity_set) {
                auto worker_count = env.worker_count;
                int numa_node_os_idx =
                    env.worker_mem_affinity[global_data.get_local_proc_idx() * worker_count];
                send_pipe_buffer = global_data.hwloc_wrapper->alloc_memory(
                    CCL_REG_MSG_ALIGNMENT, num_chunk_buffs * max_chunk_size, numa_node_os_idx);
                recv_pipe_buffer = global_data.hwloc_wrapper->alloc_memory(
                    CCL_REG_MSG_ALIGNMENT, num_chunk_buffs * max_chunk_size, numa_node_os_idx);
                break;
            }
            [[fallthrough]];
        }
        case ccl::utils::alloc_mode::memalign: {
            // internally, CCL_MALLOC calls posix_memalign
            send_pipe_buffer =
                CCL_MALLOC(num_chunk_buffs * max_chunk_size, "ccl_scaleout_pipeline_bufs");
            recv_pipe_buffer =
                CCL_MALLOC(num_chunk_buffs * max_chunk_size, "ccl_scaleout_pipeline_bufs");
        } break;
        case ccl::utils::alloc_mode::malloc: {
            send_pipe_buffer = malloc(num_chunk_buffs * max_chunk_size);
            recv_pipe_buffer = malloc(num_chunk_buffs * max_chunk_size);
        } break;
        default: CCL_THROW("unexpected alloc_mode");
    }
    CCL_THROW_IF_NOT(send_pipe_buffer, "malloc send_pipe_buffer failed");
    CCL_THROW_IF_NOT(recv_pipe_buffer, "malloc recv_pipe_buffer failed");
    if (global_data.ze_data->external_pointer_registration_enabled) {
        global_data.ze_data->import_external_pointer(send_pipe_buffer,
                                                     num_chunk_buffs * max_chunk_size);
        global_data.ze_data->import_external_pointer(recv_pipe_buffer,
                                                     num_chunk_buffs * max_chunk_size);
    }
    for (int i = 0; i < num_chunk_buffs; i++) {
        send_pipe_chunks[i] = (char*)send_pipe_buffer + i * max_chunk_size;
        recv_pipe_chunks[i] = (char*)recv_pipe_buffer + i * max_chunk_size;
    }
}

ccl_scaleout_pipeline_bufs::~ccl_scaleout_pipeline_bufs() {
    if (!send_pipe_buffer || !recv_pipe_buffer)
        return;
    try {
        auto& global_data = ccl::global_data::get();
        if (global_data.ze_data->external_pointer_registration_enabled) {
            global_data.ze_data->release_imported_pointer(send_pipe_buffer);
            global_data.ze_data->release_imported_pointer(recv_pipe_buffer);
        }
        switch (ccl::global_data::env().sycl_scaleout_buf_alloc_mode) {
            case ccl::utils::alloc_mode::hwloc: {
                // fallback to memalign if worker_affinity is not set by user
                if (ccl::global_data::env().worker_affinity_set) {
                    global_data.hwloc_wrapper->dealloc_memory(send_pipe_buffer);
                    global_data.hwloc_wrapper->dealloc_memory(recv_pipe_buffer);
                    break;
                }
                [[fallthrough]];
            };
            case ccl::utils::alloc_mode::memalign: {
                CCL_FREE(send_pipe_buffer);
                CCL_FREE(recv_pipe_buffer);
            } break;
            case ccl::utils::alloc_mode::malloc: {
                free(send_pipe_buffer);
                free(recv_pipe_buffer);
            } break;
            default:
                // destructors cannot throw exceptions
                LOG_ERROR("unexpected alloc_mode");
        }
    }
    catch (const std::exception& e) {
        // printing warning since we are towards the end of execution
        LOG_WARN("exception caught in the destructor of scaleout pipeline buffers: ", e.what());
    }
}

ccl_scaleout_pipeline_bufs::ccl_scaleout_pipeline_bufs(const ccl_scaleout_pipeline_bufs& other) {
    size_t max_chunk_size = ccl::global_data::env().sycl_max_pipeline_chunk_size;
    if (other.send_pipe_buffer) {
        send_pipe_buffer =
            CCL_MALLOC(num_chunk_buffs * max_chunk_size, "ccl_scaleout_pipeline_bufs");
        CCL_THROW_IF_NOT(send_pipe_buffer, "malloc scaleout_device_buf failed");
        std::memcpy(send_pipe_buffer, other.send_pipe_buffer, num_chunk_buffs * max_chunk_size);
        ccl::global_data::get().ze_data->import_external_pointer(send_pipe_buffer,
                                                                 num_chunk_buffs * max_chunk_size);

        for (int i = 0; i < num_chunk_buffs; i++) {
            send_pipe_chunks[i] = (char*)send_pipe_buffer + i * max_chunk_size;
        }
    }

    if (other.recv_pipe_buffer) {
        recv_pipe_buffer =
            CCL_MALLOC(num_chunk_buffs * max_chunk_size, "ccl_scaleout_pipeline_bufs");
        if (!recv_pipe_buffer) {
            CCL_THROW_IF_NOT(recv_pipe_buffer, "malloc scaleout_device_buf failed");
        }
        std::memcpy(recv_pipe_buffer, other.recv_pipe_buffer, num_chunk_buffs * max_chunk_size);
        ccl::global_data::get().ze_data->import_external_pointer(recv_pipe_buffer,
                                                                 num_chunk_buffs * max_chunk_size);

        for (int i = 0; i < num_chunk_buffs; i++) {
            recv_pipe_chunks[i] = (char*)recv_pipe_buffer + i * max_chunk_size;
        }
    }
}

ccl_scaleout_pipeline_bufs& ccl_scaleout_pipeline_bufs::operator=(
    const ccl_scaleout_pipeline_bufs& other) {
    if (this == &other) {
        return *this;
    }

    if (send_pipe_buffer) {
        CCL_FREE(send_pipe_buffer);
        send_pipe_buffer = nullptr;
    }
    if (recv_pipe_buffer) {
        CCL_FREE(recv_pipe_buffer);
        recv_pipe_buffer = nullptr;
    }

    size_t max_chunk_size = ccl::global_data::env().sycl_max_pipeline_chunk_size;

    if (other.send_pipe_buffer) {
        send_pipe_buffer =
            CCL_MALLOC(num_chunk_buffs * max_chunk_size, "ccl_scaleout_pipeline_bufs");
        CCL_THROW_IF_NOT(send_pipe_buffer, "malloc scaleout_device_buf failed");
        std::memcpy(send_pipe_buffer, other.send_pipe_buffer, num_chunk_buffs * max_chunk_size);
        ccl::global_data::get().ze_data->import_external_pointer(send_pipe_buffer,
                                                                 num_chunk_buffs * max_chunk_size);

        for (int i = 0; i < num_chunk_buffs; i++) {
            send_pipe_chunks[i] = (char*)send_pipe_buffer + i * max_chunk_size;
        }
    }

    if (other.recv_pipe_buffer) {
        recv_pipe_buffer =
            CCL_MALLOC(num_chunk_buffs * max_chunk_size, "ccl_scaleout_pipeline_bufs");
        CCL_THROW_IF_NOT(recv_pipe_buffer, "malloc scaleout_device_buf failed");
        std::memcpy(recv_pipe_buffer, other.recv_pipe_buffer, num_chunk_buffs * max_chunk_size);
        ccl::global_data::get().ze_data->import_external_pointer(recv_pipe_buffer,
                                                                 num_chunk_buffs * max_chunk_size);

        for (int i = 0; i < num_chunk_buffs; i++) {
            recv_pipe_chunks[i] = (char*)recv_pipe_buffer + i * max_chunk_size;
        }
    }

    return *this;
}

// future optimization: store a serialize stream per data it protects (separate for each tmp buffer)
// this should avoid contention
SerializedStream global_serialize_stream;

// after the function finishes:
//
// 1. any future ops on user queue will happen after main queue completes its
// current tasks
// 2. user queue is "in front of" main queue
// 3. main queue is behind user queue
//
// the points 2 and 3 above are equivalent statements
void sync_user_to_main(
    sycl::queue q,
    SerializedStream &ss) {

  bool is_recording = q.ext_oneapi_get_state() ==
                      sycl::ext::oneapi::experimental::queue_state::recording;
    LOG_DEBUG("syncs user to main");
    if (is_recording) {
        // start recording ops on main stream - sycl graph requires the sync as well
        q.ext_oneapi_get_graph().begin_recording(ss.getMainQueue());
    }
    // synchronize to main stream - put user queue "in front of" the main queue
    sycl::event last_event = ss.getMainQueue().ext_oneapi_submit_barrier();
    q.ext_oneapi_set_external_event(last_event);
    if (is_recording) {
        // start recording ops on main stream - sycl graph requires the sync as well
        q.ext_oneapi_get_graph().end_recording(ss.getMainQueue());
    }
}

// after the function finishes:
//
// 1. any future ops on main queue will happen after user queue completes its
// current tasks
// 2. main queue is "in front of" user queue
// 3. user queue is behind main queue
//
// the points 2 and 3 above are equivalent statements
void sync_main_to_user(
    sycl::queue q,
    SerializedStream &ss) {

    bool is_recording = q.ext_oneapi_get_state() ==
                      sycl::ext::oneapi::experimental::queue_state::recording;
    LOG_DEBUG("syncs main to user");
    if (is_recording) {
        // start recording ops on main stream - sycl graph requires the sync as well
        q.ext_oneapi_get_graph().begin_recording(ss.getMainQueue());
    }
    // sync to main stream - put main queue "in front of" the user queue
    std::optional<sycl::event> opt_last_event = q.ext_oneapi_get_last_event();
    if (!opt_last_event.has_value()) {
        // no kernel was submitted, the coll is no-op
        // sync is still required, submit a barrier
        opt_last_event = std::optional(q.ext_oneapi_submit_barrier());
    }
    ss.getMainQueue().ext_oneapi_set_external_event(opt_last_event.value());
    ss.getMainQueue().ext_oneapi_submit_barrier(); // this barrier seems to be required

    if (is_recording) {
        // no more ops on main queue should be recorded
        ss.getMainQueue().ext_oneapi_get_graph().end_recording(ss.getMainQueue());
    }
}

#endif // CCL_ENABLE_SYCL



// different collectives running on same communicator even on different
// streams are serialized
void ccl_comm::pre_coll_serialize(const ccl::stream::impl_value_t& stream) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    ccl_stream* op_stream = get_stream_ptr(stream);
    if (op_stream != nullptr &&
        op_stream->get_backend() == ccl::utils::get_level_zero_backend()) {
        LOG_DEBUG("pre_coll called");

        sync_user_to_main(op_stream->get_native_stream(), global_serialize_stream);
    }
#endif
}

void ccl_comm::post_coll_serialize(const ccl::stream::impl_value_t& stream) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    // collectives in group API can execute out of order
    ccl_stream* op_stream = get_stream_ptr(stream);
    if (!group_impl::is_group_active && op_stream != nullptr &&
        op_stream->get_backend() == ccl::utils::get_level_zero_backend()) {
        LOG_DEBUG("post_coll called");

        sync_main_to_user(op_stream->get_native_stream(), global_serialize_stream);
    }
#endif
}

 // different collectives running on same communicator even on different
 // streams are serialized
const ccl::vector_class<ccl::event>& ccl_comm::pre_coll_events(const ccl::stream::impl_value_t& stream,
                                                               const ccl::vector_class<ccl::event>& deps,
                                                               ccl::vector_class<ccl::event>& newdeps) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    ccl_stream* op_stream = get_stream_ptr(stream);
    if (op_stream != nullptr && last_stream != nullptr && last_stream != op_stream &&
        op_stream->get_backend() == ccl::utils::get_level_zero_backend()) {
        LOG_DEBUG("pre_coll called");
        for (auto& dep : deps) {
            sycl::event sycl_e;
            try {
                sycl_e = dep.get_native();
            }
            catch (...) {
                continue;
            }
            ccl::event ce = ccl::event::create_from_native(sycl_e);
            newdeps.push_back(std::move(ce));
        }
        newdeps.push_back(ccl::event::create_from_native(last_event));
        // clear the stored stream and event, needed in group API
        last_stream = nullptr;
        return newdeps;
    }
#endif
    return deps;
 }

void ccl_comm::post_coll_events(const ccl::stream::impl_value_t& stream, ccl::event& e) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    // collectives in group API can execute out of order
    ccl_stream* op_stream = get_stream_ptr(stream);
    if (!group_impl::is_group_active && op_stream != nullptr &&
        op_stream->get_backend() == ccl::utils::get_level_zero_backend()) {
        LOG_DEBUG("post_coll called");
        // host event may not have native event, catch exceptions and ignore
        try {
            last_event = e.get_native();
            last_stream = op_stream;
        }
        catch (...) {
            last_stream = nullptr;
        }
    }
#endif
}

void ccl_comm::post_coll_for_finalize(const ccl::stream::impl_value_t& stream, ccl::event& e) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    // Skip tracking events during group operations - ccl::group_end() handles synchronization
    if (group_impl::is_group_active) {
        LOG_DEBUG("post_coll_for_finalize skipped - group operation active");
        return;
    }

    // If already detected graph recording usage, skip all further checks and tracking
    if (used_in_graph_recording) {
        return;
    }

    ccl_stream* op_stream = get_stream_ptr(stream);
    if (op_stream != nullptr && op_stream->get_backend() == ccl::utils::get_level_zero_backend()) {
        // Check if queue is in graph recording mode
        bool is_recording = false;
        try {
            auto sycl_queue = op_stream->get_native_stream();
            // Try to detect graph recording state
            // Intel SYCL extension: ext_oneapi_get_state()
#ifdef SYCL_EXT_ONEAPI_GRAPH
            bool is_recording = sycl_queue.ext_oneapi_get_state() ==
                                sycl::ext::oneapi::experimental::queue_state::recording;
#endif
        }
        catch (...) {
            // If we can't determine recording state, means
            // that most likely the compiler does not support graphs
            is_recording = false;
        }

        if (is_recording) {
            used_in_graph_recording = true;
            return;
        }

        LOG_DEBUG("post_coll_for_finalize called for stream: ", op_stream);
        // Store or update the event for this stream
        std::optional<sycl::event> native_event;
        try {
            native_event = e.get_native();
        }
        catch (...) {
            // Host event - scheduler code for CPU path, no more processing
            return;
        }

        // native event is valid
        finalize_events[op_stream] = native_event.value();

        // Garbage collection: check if we have too many tracked streams
        size_t threshold = ccl::global_data::env().finalize_event_gc_threshold;
        if (finalize_events.size() > threshold) {
            LOG_DEBUG("Finalize event GC triggered: ",
                      finalize_events.size(),
                      " streams tracked, threshold: ",
                      threshold);
            size_t removed_count = 0;

            // Remove completed events
            for (auto it = finalize_events.begin(); it != finalize_events.end();) {
                try {
                    auto event_status =
                        it->second.get_info<sycl::info::event::command_execution_status>();
                    if (event_status == sycl::info::event_command_status::complete) {
                        it = finalize_events.erase(it);
                        removed_count++;
                    }
                    else {
                        ++it;
                    }
                }
                catch (...) {
                    // We could not query event status - very unlikely
                    // but better to keep it in case SYCLs implementation
                    // changes and uses some more complex synchronization.
                    ++it;
                }
            }

            LOG_DEBUG("Finalize event GC completed: removed ",
                      removed_count,
                      " completed events, ",
                      finalize_events.size(),
                      " remaining");
        }
    }
#endif
}

// different collectives running on same communicator even on different
// streams are serialized
const ccl::vector_class<ccl::event>& ccl_comm::pre_coll(const ccl::stream::impl_value_t& stream,
                                                        const ccl::vector_class<ccl::event>& deps,
                                                        ccl::vector_class<ccl::event>& newdeps) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    ccl_stream* cclstream = get_stream_ptr(stream);
    if (cclstream != nullptr && cclstream->get_backend() == ccl::utils::get_level_zero_backend() && global_serialize_stream.update_check_ever_recorded(cclstream->get_native_stream())) {
        pre_coll_serialize(stream);
        return deps;
    } else {
        return pre_coll_events(stream, deps, newdeps);
    }
#else
    return pre_coll_events(stream, deps, newdeps);
#endif
}

void ccl_comm::post_coll(const ccl::stream::impl_value_t& stream, ccl::event& e) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    ccl_stream* cclstream = get_stream_ptr(stream);
    if (cclstream != nullptr && cclstream->get_backend() == ccl::utils::get_level_zero_backend() && global_serialize_stream.update_check_ever_recorded(cclstream->get_native_stream())) {
        post_coll_serialize(stream);
    } else {
        post_coll_events(stream, e);
    }
    // Track event for finalize regardless of serialization path
    post_coll_for_finalize(stream, e);
#else
    post_coll_events(stream, e);
#endif
}

/* barrier */
ccl::event ccl_comm::barrier(const ccl::stream::impl_value_t& stream,
                             const ccl::barrier_attr& attr,
                             const ccl::vector_class<ccl::event>& deps) {
    return barrier_impl(stream, attr, deps);
}

ccl::event ccl_comm::barrier_impl(const ccl::stream::impl_value_t& stream,
                                  const ccl::barrier_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("barrier_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e = ccl_barrier(this, stream.get(), use_deps);
    post_coll(stream, e);
    return e;
}

/* finalize */
void ccl_comm::finalize() {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    // Cannot finalize while group operation is active
    if (group_impl::is_group_active) {
        CCL_THROW("comm.finalize() called while group operation is active - call ccl::group_end() first");
    }

    // Check if communicator was used during graph recording
    if (used_in_graph_recording) {
        LOG_WARN("comm.finalize() called on communicator that was used during graph recording. "
                 "Event tracking is incomplete - user must manually synchronize all graph submissions. ");
        return;
    }

    LOG_DEBUG("finalize called on comm: ", std::hex, this, ", waiting on ", finalize_events.size(), " stream(s)");

    // Wait on all outstanding events from all streams
    for (auto& [stream, event] : finalize_events) {
        LOG_DEBUG("waiting on event from stream: ", std::hex, stream);
        try {
            event.wait();
        }
        catch (const std::exception& ex) {
            LOG_ERROR("exception while waiting on event from stream ", std::hex, stream, ": ", ex.what());
        }
        catch (...) {
            LOG_ERROR("unknown exception while waiting on event from stream ", std::hex, stream);
        }
    }

    // Clear the finalize events map after waiting
    finalize_events.clear();
    LOG_DEBUG("finalize completed on comm: ", std::hex, this);
#else
    LOG_DEBUG("finalize called on comm: ", std::hex, this, " (no-op for non-SYCL builds)");
#endif
}

/* allgather */
ccl::event ccl_comm::allgather_impl(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allgather_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("allgather_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e = ccl_allgather(
        send_buf, recv_buf, count, dtype, attr, this, get_stream_ptr(stream), use_deps);
    post_coll(stream, e);
    return e;
}

ccl::event ccl_comm::allgather_impl(const void* send_buf,
                                    const ccl::vector_class<void*>& recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allgather_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    return ccl_allgather(reinterpret_cast<const void*>(send_buf),
                         (void*)(recv_buf.data()),
                         count,
                         dtype,
                         internal_attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

/* allgatherv */
ccl::event ccl_comm::allgatherv_impl(const void* send_buf,
                                     size_t send_count,
                                     void* recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("allgatherv_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e = ccl_allgatherv(send_buf,
                                  send_count,
                                  recv_buf,
                                  recv_counts,
                                  dtype,
                                  attr,
                                  this,
                                  get_stream_ptr(stream),
                                  use_deps);
    post_coll(stream, e);
    return e;
}

ccl::event ccl_comm::allgatherv_impl(const void* send_buf,
                                     size_t send_count,
                                     const ccl::vector_class<void*>& recv_bufs,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    return ccl_allgatherv(reinterpret_cast<const void*>(send_buf),
                          send_count,
                          (void*)(recv_bufs.data()),
                          recv_counts,
                          dtype,
                          internal_attr,
                          this,
                          get_stream_ptr(stream),
                          deps);
}

/* allreduce */
ccl::event ccl_comm::allreduce_impl(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    ccl::reduction reduction,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allreduce_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("allreduce_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e = ccl_allreduce(
        send_buf, recv_buf, count, dtype, reduction, attr, this, get_stream_ptr(stream), use_deps);
    post_coll(stream, e);
    return e;
}

/* alltoall */
ccl::event ccl_comm::alltoall_impl(const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::alltoall_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("alltoall_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e = ccl_alltoall(
        send_buf, recv_buf, count, dtype, attr, this, get_stream_ptr(stream), use_deps);
    post_coll(stream, e);
    return e;
}

ccl::event ccl_comm::alltoall_impl(const ccl::vector_class<void*>& send_buf,
                                   const ccl::vector_class<void*>& recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::alltoall_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    return ccl_alltoall((void*)(send_buf.data()),
                        (void*)(recv_buf.data()),
                        count,
                        dtype,
                        internal_attr,
                        this,
                        get_stream_ptr(stream),
                        deps);
}

/* alltoallv */
ccl::event ccl_comm::alltoallv_impl(const void* send_buf,
                                    const ccl::vector_class<size_t>& send_counts,
                                    void* recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoallv_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("alltoallv_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e = ccl_alltoallv(send_buf,
                                 send_counts.data(),
                                 recv_buf,
                                 recv_counts.data(),
                                 dtype,
                                 attr,
                                 this,
                                 get_stream_ptr(stream),
                                 use_deps);
    post_coll(stream, e);
    return e;
}

ccl::event ccl_comm::alltoallv_impl(const ccl::vector_class<void*>& send_buf,
                                    const ccl::vector_class<size_t>& send_counts,
                                    ccl::vector_class<void*> recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoallv_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("alltoallv_impl_2 called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);

    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    ccl::event e = ccl_alltoallv((void*)send_buf.data(),
                                 send_counts.data(),
                                 (void*)recv_buf.data(),
                                 recv_counts.data(),
                                 dtype,
                                 internal_attr,
                                 this,
                                 get_stream_ptr(stream),
                                 use_deps);
    post_coll(stream, e);
    return e;
}

/* bcast */
ccl::event ccl_comm::broadcast_impl(void* buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    int root,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::broadcast_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("broadcast_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e =
        ccl_broadcast(buf, count, dtype, root, attr, this, get_stream_ptr(stream), use_deps);
    post_coll(stream, e);
    return e;
}

/* broadcast */
ccl::event ccl_comm::broadcast_impl(void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    int root,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::broadcast_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("broadcast_impl_2 called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e = ccl_broadcast(
        send_buf, recv_buf, count, dtype, root, attr, this, get_stream_ptr(stream), use_deps);
    post_coll(stream, e);
    return e;
}

/* reduce */
ccl::event ccl_comm::reduce_impl(const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 ccl::datatype dtype,
                                 ccl::reduction reduction,
                                 int root,
                                 const ccl::stream::impl_value_t& stream,
                                 const ccl::reduce_attr& attr,
                                 const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("reduce_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e = ccl_reduce(send_buf,
                              recv_buf,
                              count,
                              dtype,
                              reduction,
                              root,
                              attr,
                              this,
                              get_stream_ptr(stream),
                              use_deps);
    post_coll(stream, e);
    return e;
}

/* reduce_scatter */
ccl::event ccl_comm::reduce_scatter_impl(const void* send_buf,
                                         void* recv_buf,
                                         size_t recv_count,
                                         ccl::datatype dtype,
                                         ccl::reduction reduction,
                                         const ccl::stream::impl_value_t& stream,
                                         const ccl::reduce_scatter_attr& attr,
                                         const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("reduce_scatter_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e = ccl_reduce_scatter(send_buf,
                                      recv_buf,
                                      recv_count,
                                      dtype,
                                      reduction,
                                      attr,
                                      this,
                                      get_stream_ptr(stream),
                                      use_deps);
    post_coll(stream, e);
    return e;
}

/* scatter */
ccl::event ccl_comm::scatter_impl(const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  int root,
                                  const ccl::stream::impl_value_t& stream,
                                  const ccl::scatter_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("scatter_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e = ccl_scatter(
        send_buf, recv_buf, count, dtype, root, attr, this, get_stream_ptr(stream), use_deps);
    post_coll(stream, e);
    return e;
}

/* recv */
ccl::event ccl_comm::recv_impl(void* recv_buf,
                               size_t recv_count,
                               ccl::datatype dtype,
                               int peer,
                               const ccl::stream::impl_value_t& stream,
                               const ccl::pt2pt_attr& attr,
                               const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("recv_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e =
        ccl_recv(recv_buf, recv_count, dtype, peer, attr, this, get_stream_ptr(stream), use_deps);
    post_coll(stream, e);
    return e;
}

/* send */
ccl::event ccl_comm::send_impl(void* send_buf,
                               size_t send_count,
                               ccl::datatype dtype,
                               int peer,
                               const ccl::stream::impl_value_t& stream,
                               const ccl::pt2pt_attr& attr,
                               const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("send_impl called on comm: ",
              std::hex,
              this,
              " stream: ",
              std::hex,
              get_stream_ptr(stream));
    ccl::vector_class<ccl::event> newdeps;
    const ccl::vector_class<ccl::event>& use_deps = pre_coll(stream, deps, newdeps);
    ccl::event e =
        ccl_send(send_buf, send_count, dtype, peer, attr, this, get_stream_ptr(stream), use_deps);
    post_coll(stream, e);
    return e;
}

COMM_INTERFACE_COLL_INSTANTIATION(ccl_comm);
#ifdef CCL_ENABLE_SYCL
SYCL_COMM_INTERFACE_COLL_INSTANTIATION(ccl_comm);
#endif // CCL_ENABLE_SYCL
